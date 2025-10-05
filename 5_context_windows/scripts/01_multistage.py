#!/usr/bin/env python3
"""
01_multistage.py — Multi-Stage Retrieval (blog-support demo)

What this shows
---------------
Two-stage retrieval that improves precision before building the context:
  Stage 1: Broad, fast candidate fetch with a hashing vectorizer
  Stage 2: Precise reranking with TF-IDF cosine similarity

Why it matters
--------------
Large windows are costly and noisy. Multi-stage retrieval narrows candidates
so the final context is smaller, cleaner, and more likely to fit tight budgets.

Usage
-----
python scripts/01_multistage.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k_broad 10 \
  --k_final 2
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import tiktoken


# ----------------------------
# Tiny utilities (blog-support)
# ----------------------------

def load_documents(data_dir: str) -> Dict[str, str]:
    """Load .txt files into {doc_id: text}."""
    p = Path(data_dir)
    if not p.exists():
        raise SystemExit(f"[ERROR] data_dir not found: {data_dir}")
    docs = {}
    for fp in p.glob("*.txt"):
        docs[fp.stem] = fp.read_text(encoding="utf-8").strip()
    if not docs:
        raise SystemExit(f"[ERROR] No .txt files in {data_dir}")
    return docs


def count_tokens(text: str, model_hint: str = "gpt-4o-mini") -> int:
    """Rough token estimate using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def topk_indices_from_cosine(query_vec, matrix, k: int) -> List[int]:
    """Cosine similarity ranking; returns indices of top-k rows."""
    sims = cosine_similarity(query_vec, matrix)[0]  # shape: (n_docs,)
    order = np.argsort(-sims)[:k]
    return order.tolist()


# ----------------------------
# Stage 1: Broad retrieval
# ----------------------------

def stage1_broad_hashing(
    corpus: Dict[str, str],
    question: str,
    k_broad: int = 10,
    n_features: int = 2**16
) -> List[Tuple[str, float]]:
    """
    Fast, memory-light candidate fetch via HashingVectorizer + NearestNeighbors.
    Returns [(doc_id, similarity), ...] for the top candidates.
    """
    ids = list(corpus.keys())
    texts = [corpus[i] for i in ids]

    vect = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        stop_words="english"
    )
    X = vect.transform(texts)
    q = vect.transform([question])

    # Nearest neighbors by cosine distance (1 - cosine similarity)
    nn = NearestNeighbors(metric="cosine").fit(X)
    n = min(k_broad, len(ids))
    dists, idx = nn.kneighbors(q, n_neighbors=n)

    out = []
    for i, d in zip(idx[0], dists[0]):
        out.append((ids[i], 1.0 - float(d)))  # convert to similarity
    return out


# ----------------------------
# Stage 2: Precise reranking
# ----------------------------

def stage2_precise_tfidf(
    corpus: Dict[str, str],
    question: str,
    candidates: List[str],
    k_final: int = 5
) -> List[Tuple[str, float]]:
    """
    Rerank candidates with TF-IDF + cosine similarity for higher precision.
    Returns top-k [(doc_id, similarity), ...].
    """
    # Fit TF-IDF on just the candidate set (tighter vocabulary for discrimination)
    tfidf = TfidfVectorizer(stop_words="english")
    cand_texts = [corpus[cid] for cid in candidates]
    X = tfidf.fit_transform(cand_texts)
    q = tfidf.transform([question])

    order = topk_indices_from_cosine(q, X, k=min(k_final, len(candidates)))
    return [(candidates[i], float(cosine_similarity(q, X)[0][i])) for i in order]


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Multi-Stage Retrieval demo")
    ap.add_argument("--question", required=True, help="User question to retrieve for")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")
    ap.add_argument("--k_broad", type=int, default=10, help="Stage-1 candidate count")
    ap.add_argument("--k_final", type=int, default=5, help="Stage-2 final count")
    args = ap.parse_args()

    # Load tiny corpus
    corpus = load_documents(args.data_dir)

    # Stage 1: Broad fetch
    broad = stage1_broad_hashing(corpus, args.question, k_broad=args.k_broad)
    broad_ids = [doc_id for doc_id, _ in broad]

    # Stage 2: Precise rerank on candidates
    final = stage2_precise_tfidf(corpus, args.question, broad_ids, k_final=args.k_final)
    final_ids = [doc_id for doc_id, _ in final]

    # Token accounting: naive (take all stage-1) vs. refined (stage-2 only)
    naive_context = "\n\n---\n\n".join(corpus[i] for i in broad_ids)
    refined_context = "\n\n---\n\n".join(corpus[i] for i in final_ids)
    naive_tokens = count_tokens(naive_context)
    refined_tokens = count_tokens(refined_context)

    # Console prints only (blog-support)
    print("\nQuestion:")
    print(args.question)

    print("\nStage 1 — Broad candidates (HashingVectorizer):")
    for i, (doc_id, sim) in enumerate(broad, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nStage 2 — Precise top-k (TF-IDF rerank):")
    for i, (doc_id, sim) in enumerate(final, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nToken impact (naive stage-1 vs refined stage-2):")
    print(f"  tokens_stage1_naive ≈ {naive_tokens}")
    print(f"  tokens_stage2_final ≈ {refined_tokens}")

    print("\nFinal context (stage-2):")
    print(refined_context)


if __name__ == "__main__":
    # Make stderr less noisy on some environments (optional)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
