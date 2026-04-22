"""
build a suitable Books dataset from Books_5.json.gz for CF models.
"""

import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Books_5.json.gz"
OUT_PATH = ROOT / "data" / "processed" / "books_big.parquet"

# config
MIN_USER_REVIEWS = 10
MIN_ITEM_REVIEWS = 20
TARGET_ITEMS     = 2000
MAX_REVIEWS_PER_ITEM = 500
MIN_WORDS        = 3
MAX_WORDS        = 1000
SEED             = 42


def extract_format(style):
    if not isinstance(style, dict):
        return "Unknown"
    fmt = style.get("Format:", style.get("Format", ""))
    return fmt.strip() if fmt else "Unknown"


def k_core_filter(df, min_user, min_item, max_iter=20):
    for i in range(max_iter):
        n_before = len(df)

        # filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        # filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]

        n_after = len(df)
        print(f"  k-core iter {i+1}: {n_before:,} → {n_after:,} rows  "
              f"({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")

        if n_after == n_before:
            print("  Converged.")
            break

    return df


def main():
    t0 = time.time()
    rows = []
    lines_read = 0

    with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            reviewer = obj.get("reviewerID")
            asin = obj.get("asin")
            overall = obj.get("overall")
            if not reviewer or not asin or overall is None:
                continue

            review_text = (obj.get("reviewText") or "").strip()
            summary_text = (obj.get("summary") or "").strip()
            if not review_text and not summary_text:
                continue

            text_combined = f"{review_text} [SEP] {summary_text}".strip()
            n_words = len(text_combined.split())
            if n_words < MIN_WORDS or n_words > MAX_WORDS:
                continue

            rating = float(overall)
            if not 1.0 <= rating <= 5.0:
                continue

            timestamp = int(obj.get("unixReviewTime", 0))
            year = pd.to_datetime(timestamp, unit="s").year if timestamp > 0 else 0
            verified = bool(obj.get("verified", False))
            style = obj.get("style")

            rows.append({
                "user_id": str(reviewer),
                "item_id": str(asin),
                "rating": float(rating),
                "timestamp": timestamp,
                "year": year,
                "verified": verified,
                "item_format": extract_format(style),
                "text_combined": text_combined,
                "review_len_words": n_words,
            })

            if lines_read % 5_000_000 == 0:
                print(f"  {lines_read:,} lines, {len(rows):,} reviews ({time.time() - t0:.0f}s)")

    print(f"  Done: {len(rows):,} reviews from {lines_read:,} lines ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    print(f"\nRaw: {len(df):,} rows, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    # K-core filtering
    print(f"\nK-core filtering (user >= {MIN_USER_REVIEWS}, item >= {MIN_ITEM_REVIEWS})...")
    df = k_core_filter(df, MIN_USER_REVIEWS, MIN_ITEM_REVIEWS)

    n_items = df["item_id"].nunique()
    if n_items > TARGET_ITEMS:
        print(f"\n{n_items:,} items > target {TARGET_ITEMS}, taking top by review count...")
        item_counts = df["item_id"].value_counts()
        top_items = set(item_counts.head(TARGET_ITEMS).index)
        df = df[df["item_id"].isin(top_items)]
        # Re-run k-core after item reduction
        print("Re-running k-core after item cap...")
        df = k_core_filter(df, MIN_USER_REVIEWS, MIN_ITEM_REVIEWS)

    rng = np.random.RandomState(SEED)
    capped = []
    for item_id, group in df.groupby("item_id"):
        if len(group) > MAX_REVIEWS_PER_ITEM:
            group = group.sample(MAX_REVIEWS_PER_ITEM, random_state=rng)
        capped.append(group)
    df = pd.concat(capped, ignore_index=True)

    df = k_core_filter(df, MIN_USER_REVIEWS, MIN_ITEM_REVIEWS)

    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")
    df["year"] = df["year"].astype("int16")
    df["review_len_words"] = df["review_len_words"].astype("int32")
    df["item_format"] = df["item_format"].astype("category")

    n_u = df["user_id"].nunique()
    n_i = df["item_id"].nunique()
    density = len(df) / (n_u * n_i) * 100

    print(f"\n{'='*55}")
    print(f"Rows:                {len(df):>10,}")
    print(f"Users:               {n_u:>10,}")
    print(f"Items:               {n_i:>10,}")
    print(f"Density:             {density:>10.4f}%")
    print(f"Avg reviews/user:    {len(df)/n_u:>10.1f}")
    print(f"Avg reviews/item:    {len(df)/n_i:>10.1f}")
    print(f"Median reviews/user: {df.groupby('user_id').size().median():>10.1f}")
    print(f"Median reviews/item: {df.groupby('item_id').size().median():>10.1f}")
    print(f"Rating mean:         {df['rating'].mean():>10.3f}")
    print(f"Year range:          {df['year'].min()}–{df['year'].max()}")

    print(f"\nRating distribution:")
    for r in [1, 2, 3, 4, 5]:
        pct = (df["rating"] == r).mean() * 100
        print(f"  {int(r)}: {(df['rating']==r).sum():>8,} ({pct:>5.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
