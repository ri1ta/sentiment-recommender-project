from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "books_big.parquet"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 256
MAX_LENGTH = 128


def compute_sentiment(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=max_length,
        batch_size=batch_size,
    )

    scores = []
    for out in tqdm(clf(texts, batch_size=batch_size), total=len(texts), desc="Sentiment"):
        label = out["label"]
        conf = out["score"]
        scores.append(conf if label == "POSITIVE" else -conf)

    return np.array(scores, dtype=np.float32)


def main():
    df = pd.read_parquet(DATA_PATH)

    df = df.sort_values(["user_id", "timestamp"])
    df["_rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=True)
    df["_size"] = df.groupby("user_id")["user_id"].transform("count")
    train_mask = ~(
        (df["_rank"] == df["_size"]) | (df["_rank"] == df["_size"] - 1)
    )
    df = df.drop(columns=["_rank", "_size"])

    train_idx = df[train_mask].index

    texts = (
        df.loc[train_idx, "text_combined"]
        .str.split(" [SEP] ").str[0]
        .fillna("").tolist()
    )

    scores = compute_sentiment(texts)
    df["sentiment_score"] = float("nan")
    df.loc[train_idx, "sentiment_score"] = scores

    df.to_parquet(DATA_PATH, index=False)


if __name__ == "__main__":
    main()
