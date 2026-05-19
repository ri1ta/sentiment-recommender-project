# Sentiment-Aware Review-Based Recommendation

This repository contains the experimental code and result files for the project:

**A Recommendation System Based on Sentiment Analysis of User Reviews**

The project studies whether textual reviews can improve Top-N recommendation quality. In particular, it compares three types of review-derived signals:

- semantic review representations;
- overall review sentiment;
- NLI-based aspect-level sentiment.

The main experimental domain is **Amazon Books**. Additional cross-domain experiments are conducted on **Amazon Movies & TV** and **Amazon Video Games**.

## Project Overview

The thesis focuses on a controlled comparison of collaborative, text-aware, and aspect-aware recommendation models under a temporal leave-one-out Top-N evaluation protocol.

The main proposed models are:

- **Aspect-CF** — an aspect-aware collaborative filtering model using NLI-based item aspect profiles;
- **Aspect-MTL** — a warm-started multi-task extension of Aspect-CF with sentiment prediction as an auxiliary objective.

Additional strong collaborative baselines, including **EASE** and **MultiVAE**, are used to evaluate the absolute ranking strength of the proposed approach.

## Main Research Question

The project investigates which type of review-derived textual signal is most useful for Top-N recommendation:

1. broad semantic embeddings;
2. document-level sentiment;
3. aspect-level sentiment extracted from reviews.

The main finding is that aspect-level sentiment is the most effective textual signal within the controlled NCF-based comparison. At the same time, strong interaction-only baselines such as EASE and MultiVAE achieve higher absolute ranking performance, which motivates future work on integrating aspect features into stronger collaborative backbones.

## Repository Structure

```text
sentiment-recommender-project/
│
├── data/
│   ├── books_big_sample.csv
│   ├── books_big_nli_sample.csv
│   ├── games_big_sample.csv
│   ├── games_big_nli_sample.csv
│   ├── games_item_nli_aspects_sample.csv
│   ├── movies_big_sample.csv
│   └── movies_item_nli_aspects_sample.csv
│
├── prep/
│   ├── build_bigger_dataset.py
│   └── add_sentiment.py
│
├── models/
│   ├── nli-extraction.ipynb
│   ├── cf-bert-sentiment-cf.ipynb
│   ├── lightgcn + ncf.ipynb
│   ├── aspect-cf.ipynb
│   ├── aspect-mtl.ipynb
│   ├── cross-domain.ipynb
│   ├── sasrec-ease-multivae.ipynb
│   └── ease-aspect-profile-reranking.ipynb
│
├── results/
│   ├── ncf_results.json
│   ├── lightgcn_results.json
│   ├── cf_bert_results.json
│   ├── aspect_cf_(nli_raw)_results.json
│   ├── aspect_mtl_results.json
│   ├── mtl_sent_only_results.json
│   ├── mtl_asp_only_results.json
│   └── mtl_both_soft_results.json
│
└── README.md
```

## Data

The original Amazon review data can be downloaded from:

[Amazon Review Data](https://nijianmo.github.io/amazon/index.html)

## Main Notebooks

| Notebook | Description |
|---|---|
| `nli-extraction.ipynb` | Extracts aspect-level sentiment features from review text using zero-shot NLI. |
| `cf-bert-sentiment-cf.ipynb` | Runs semantic and overall sentiment text-aware baselines. |
| `lightgcn + ncf.ipynb` | Runs collaborative filtering baselines: NCF and LightGCN. |
| `aspect-cf.ipynb` | Trains and evaluates the Aspect-CF model. |
| `aspect-mtl.ipynb` | Trains and evaluates the Aspect-MTL model and auxiliary-objective variants. |
| `cross-domain.ipynb` | Runs cross-domain validation on Movies & TV and Video Games. |
| `sasrec-ease-multivae.ipynb` | Contains additional baseline experiments, including EASE and MultiVAE. |
| `ease-aspect-profile-reranking.ipynb` | Contains the EASE + aspect-profile reranking diagnostic. |
