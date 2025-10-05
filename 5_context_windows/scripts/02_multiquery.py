#!/usr/bin/env python3
"""
02_multiquery.py — Multi-Query Rewriting (blog-support demo)

What this shows
---------------
Generate multiple query variants, retrieve for each, then merge + dedupe results.
This improves recall when corpus wording differs from the user's phrasing.

Modes
-----
- Fallback (default): static synonym expansion (no API keys needed)
- LLM (optional): Groq-based paraphrases with --use_llm and GROQ_API_KEY in .env

Usage
-----
python scripts/02_multiquery.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k 5 \
  --paraphrases 3 \
  --use_llm
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from dotenv import load_dotenv
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
# Retrieval (TF-IDF baseline)
# ----------------------------

class TfidfRetriever:
    """
    Lightweight retriever using TF-IDF + cosine via NearestNeighbors.
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
# Multi-query rewriting
# ----------------------------

STATIC_SYNONYMS = {
    "summarize": ["outline", "synthesize", "condense"],
    "evaluation": ["assessment", "benchmark", "analysis"],
    "approach": ["method", "strategy", "procedure"],
    "results": ["findings", "outcomes", "conclusions"],
    "policy": ["guideline", "standard", "framework"],
    "design": ["architecture", "structure", "layout"],
    "feedback": ["user input", "responses", "comments"],
    "incident": ["event", "outage", "issue"],
    "timeline": ["sequence", "chronology", "history"],
}

def static_rewrites(query: str, max_variants: int) -> List[str]:
    """
    Simple synonym-based expansion: replace one matched token at a time.
    Ensures deterministic output and no external dependencies.
    """
    words = query.lower().split()
    variants: List[str] = []
    for i, w in enumerate(words):
        if w in STATIC_SYNONYMS:
            for syn in STATIC_SYNONYMS[w]:
                new = words.copy()
                new[i] = syn
                variants.append(" ".join(new))
                if len(variants) >= max_variants:
                    return variants
    return variants[:max_variants]


def llm_rewrites_with_groq(query: str, n: int, model: str = "llama-3.1-8b-instant") -> List[str]:
    """
    Generate N paraphrases with Groq (requires GROQ_API_KEY). Guarantees short, safe rewrites.
    """
    from groq import Groq  # imported lazily so file runs without this dep installed
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Put it in .env or skip --use_llm.")
    client = Groq(api_key=key)

    prompt = (
        "Rewrite the user's query into distinct, concise paraphrases that preserve meaning. "
        "Avoid adding new topics. Output each paraphrase on a new line, no numbering."
        f"\n\nUser query: {query}\n"
        f"Number of paraphrases: {n}\n"
    )
    out = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate safe, concise paraphrases."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    text = out.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Keep first n distinct lines
    uniq = []
    seen: Set[str] = set()
    for ln in lines:
        if ln.lower() not in seen:
            uniq.append(ln)
            seen.add(ln.lower())
        if len(uniq) >= n:
            break
    return uniq


def merged_retrieval(
    retriever: TfidfRetriever,
    corpus: Dict[str, str],
    queries: List[str],
    k: int
) -> List[Tuple[str, float]]:
    """
    Retrieve for each query variant, then merge results by taking the
    maximum similarity score per document.
    """
    scores: Dict[str, float] = {}
    for q in queries:
        hits = retriever.search(q, top_k=k)
        for doc_id, sim in hits:
            scores[doc_id] = max(scores.get(doc_id, 0.0), sim)
    # sort by merged score desc
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:k]


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Multi-Query Rewriting demo")
    ap.add_argument("--question", required=True, help="User query")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")
    ap.add_argument("--k", type=int, default=5, help="Top-k to return")
    ap.add_argument("--paraphrases", type=int, default=3, help="Number of rewrites to generate")
    ap.add_argument("--use_llm", action="store_true", help="Use Groq for query rewrites")
    args = ap.parse_args()

    load_dotenv(override=True)
    corpus = load_documents(args.data_dir)

    # Build retriever
    ret = TfidfRetriever()
    ret.build(corpus)

    # Build query set: original + rewrites
    queries = [args.question]
    try:
        if args.use_llm:
            queries += llm_rewrites_with_groq(args.question, n=args.paraphrases)
        else:
            queries += static_rewrites(args.question, max_variants=args.paraphrases)
    except Exception as e:
        print(f"[WARN] LLM rewriting failed: {e}. Falling back to static rewrites.")
        queries += static_rewrites(args.question, max_variants=args.paraphrases)

    # Merge + dedupe queries (case-insensitive)
    uniq = []
    seen = set()
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            uniq.append(q.strip())
            seen.add(key)

    # Retrieve and merge scores
    merged = merged_retrieval(ret, corpus, uniq, k=args.k)
    merged_ids = [doc_id for doc_id, _ in merged]

    # Token comparison: context from original vs. merged-unique top-k
    enc = get_encoder()
    orig_hits = ret.search(args.question, top_k=args.k)
    orig_context = "\n\n---\n\n".join(corpus[i] for i, _ in orig_hits)
    merged_context = "\n\n---\n\n".join(corpus[i] for i in merged_ids)
    t_orig = count_tokens(orig_context, enc)
    t_merged = count_tokens(merged_context, enc)

    # Console prints only (blog-support)
    print("\nQuestion:")
    print(args.question)

    print("\nQuery variants considered:")
    for i, q in enumerate(uniq, 1):
        print(f"  {i:02d}. {q}")

    print("\nTop-k per original query:")
    for i, (doc_id, sim) in enumerate(orig_hits, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nMerged top-k across all variants:")
    for i, (doc_id, sim) in enumerate(merged, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nToken comparison (context built from retrieved docs):")
    print(f"  tokens_original_only ≈ {t_orig}")
    print(f"  tokens_multiquery_merged ≈ {t_merged}")

    print("\nFinal merged context:")
    print(merged_context)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
