#!/usr/bin/env python3
"""
03_rerank.py — Reranking with optional Cohere (blog-support demo)

What this shows
---------------
1) Retrieve a broader candidate set with TF-IDF.
2) Rerank candidates either with:
   - Cohere Rerank v3 (optional, requires COHERE_API_KEY), or
   - A lightweight lexical fallback (no API keys).

Why it matters
--------------
Precision at small k is critical for tight context windows. A strong reranker
pushes the most relevant passages to the top so fewer, better chunks enter
the prompt.

Usage
-----
python scripts/03_rerank.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k_init 10 \
  --k_final 5 \
  --use_cohere \
  --cohere_model rerank-english-v3.0
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
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


def get_encoder(model_hint: str = "gpt-4o-mini"):
    try:
        return tiktoken.encoding_for_model(model_hint)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc=None) -> int:
    enc = enc or get_encoder()
    return len(enc.encode(text or ""))


# ----------------------------
# Baseline retriever (TF-IDF)
# ----------------------------

class TfidfRetriever:
    """
    Lightweight retriever using TF-IDF + cosine (NearestNeighbors).
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.nn = None
        self.ids: List[str] = []
        self.X = None

    def build(self, corpus: Dict[str, str]):
        self.ids = list(corpus.keys())
        texts = [corpus[i] for i in self.ids]
        self.X = self.vectorizer.fit_transform(texts)
        self.nn = NearestNeighbors(metric="cosine").fit(self.X)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = self.vectorizer.transform([query])
        n = min(top_k, len(self.ids))
        dists, idx = self.nn.kneighbors(q, n_neighbors=n)
        out = []
        for i, d in zip(idx[0], dists[0]):
            out.append((self.ids[i], 1.0 - float(d)))
        return out


# ----------------------------
# Rerankers
# ----------------------------

def lexical_rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_n: int
) -> List[Tuple[str, float]]:
    """
    Very small, dependency-free reranker:
    - Score = sum of query term frequencies in doc
    - + small bonus for exact phrase matches (bigrams)
    """
    q_tokens = [t for t in query.lower().split() if t.isascii()]
    bigrams = set(zip(q_tokens, q_tokens[1:])) if len(q_tokens) > 1 else set()

    scored: List[Tuple[str, float]] = []
    for doc_id, text in candidates:
        t = text.lower()
        # term frequency score
        tf_score = sum(t.count(f" {qt} ") + t.startswith(qt + " ") + t.endswith(" " + qt) for qt in q_tokens)
        # bigram bonus
        bg_score = 0
        for a, b in bigrams:
            bg_score += t.count(f"{a} {b}")
        score = float(tf_score + 0.5 * bg_score)
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[: min(top_n, len(scored))]


def cohere_rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    top_n: int,
    model: str = "rerank-english-v3.0"
) -> List[Tuple[str, float]]:
    """
    Cohere Rerank v3 (if available). Falls back to lexical if anything fails.
    Accepts candidates as [(doc_id, text), ...].
    """
    try:
        # Cohere v4+ client
        try:
            import cohere
            # Newer SDKs may use ClientV2; handle both
            client = getattr(cohere, "ClientV2", None)
            client = client(os.getenv("COHERE_API_KEY")) if client else cohere.Client(os.getenv("COHERE_API_KEY"))
        except Exception as e:
            raise RuntimeError(f"Cohere import/init failed: {e}")

        if not os.getenv("COHERE_API_KEY"):
            raise RuntimeError("Missing COHERE_API_KEY")

        # Some SDKs use .rerank, others .rerank() under client.rerank
        texts = [text for _, text in candidates]
        try:
            # Common style: client.rerank(query=..., documents=[...], top_n=...)
            resp = client.rerank(
                query=query,
                documents=texts,
                top_n=min(top_n, len(texts)),
                model=model,
            )
            # resp may have .results with (index, relevance_score)
            pairs: List[Tuple[int, float]] = []
            for r in getattr(resp, "results", []):
                idx = getattr(r, "index", None)
                score = getattr(r, "relevance_score", None)
                if idx is None or score is None:
                    continue
                pairs.append((int(idx), float(score)))
        except Exception:
            # Fallback to older signature (rare)
            resp = client.rerank(model=model, query=query, documents=texts, top_n=min(top_n, len(texts)))
            pairs = [(int(r.index), float(r.relevance_score)) for r in resp.results]

        # Map indices back to ids
        id_map = [doc_id for doc_id, _ in candidates]
        ranked = [(id_map[i], score) for i, score in pairs]
        return ranked[: min(top_n, len(ranked))]

    except Exception as e:
        print(f"[WARN] Cohere rerank failed: {e}. Falling back to lexical rerank.")
        return lexical_rerank(query, candidates, top_n)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Reranking demo (Cohere optional)")
    ap.add_argument("--question", required=True, help="User query")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")
    ap.add_argument("--k_init", type=int, default=10, help="Initial candidate count from retriever")
    ap.add_argument("--k_final", type=int, default=5, help="Final top-k after rerank")
    ap.add_argument("--use_cohere", action="store_true", help="Use Cohere Rerank v3 if available")
    ap.add_argument("--cohere_model", default="rerank-english-v3.0", help="Cohere rerank model name")
    args = ap.parse_args()

    load_dotenv(override=True)
    corpus = load_documents(args.data_dir)

    # 1) Retrieve a broad candidate set
    ret = TfidfRetriever()
    ret.build(corpus)
    initial = ret.search(args.question, top_k=args.k_init)

    # Keep ids + texts
    cand_ids = [doc_id for doc_id, _ in initial]
    candidates = [(doc_id, corpus[doc_id]) for doc_id in cand_ids]

    # 2) Rerank
    if args.use_cohere:
        final = cohere_rerank(args.question, candidates, top_n=args.k_final, model=args.cohere_model)
    else:
        final = lexical_rerank(args.question, candidates, top_n=args.k_final)

    final_ids = [doc_id for doc_id, _ in final]

    # Token accounting: naive (pre-rerank) vs refined (post-rerank)
    enc = get_encoder()
    naive_context = "\n\n---\n\n".join(corpus[i] for i in cand_ids[: args.k_final])
    refined_context = "\n\n---\n\n".join(corpus[i] for i in final_ids)
    t_naive = count_tokens(naive_context, enc)
    t_refined = count_tokens(refined_context, enc)

    # Console prints (blog-support)
    print("\nQuestion:")
    print(args.question)

    print("\nInitial candidates (TF-IDF):")
    for i, (doc_id, sim) in enumerate(initial, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nReranked top-k:")
    for i, (doc_id, score) in enumerate(final, 1):
        tag = "cohere" if args.use_cohere and os.getenv("COHERE_API_KEY") else "lexical"
        print(f"  {i:02d}. {doc_id}  ({tag}_score≈{score:.3f})")

    print("\nToken comparison (context built from docs):")
    print(f"  tokens_before_rerank (take first k_init→k_final) ≈ {t_naive}")
    print(f"  tokens_after_rerank (final top-k)               ≈ {t_refined}")

    print("\nFinal reranked context:")
    print(refined_context)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
