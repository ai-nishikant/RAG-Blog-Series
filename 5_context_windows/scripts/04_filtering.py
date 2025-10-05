#!/usr/bin/env python3
"""
04_filtering.py — Dynamic Metadata Filtering (blog-support demo)

What this shows
---------------
Apply simple metadata filters (year, document type) either:
  - BEFORE retrieval (pre-filter), or
  - AFTER retrieval (post-filter).
Then compare the retrieved set and token footprint with/without filters.

Why it matters
--------------
Filtering reduces noise and keeps the context window focused on what matters
(e.g., recent or in-scope docs), which improves precision and saves tokens.

Usage
-----
python scripts/04_filtering.py \
  --question "What changed recently in the policy?" \
  --data_dir data/sample_docs \
  --k 5 \
  --mode pre \
  --min_year 2024 \
  --types policy,design

# Try post-filtering:
python scripts/04_filtering.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k 5 \
  --mode post \
  --types research
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ----------------------------
# Tiny utilities (blog-support)
# ----------------------------

DOC_TYPE_FROM_PREFIX = {
    "policy": "policy",
    "research": "research",
    "system": "design",
    "design": "design",
    "support": "support",
    "incident": "incident",
    "release": "release",
}

YEAR_RE = re.compile(r"\b(20\d{2})\b")

def infer_year(text: str) -> Optional[int]:
    """Return the max 4-digit year found (e.g., 2024), else None."""
    years = [int(y) for y in YEAR_RE.findall(text)]
    return max(years) if years else None

def infer_type(doc_id: str) -> str:
    """Infer type from filename prefix; fall back to 'unknown'."""
    prefix = doc_id.split("_", 1)[0].lower()
    return DOC_TYPE_FROM_PREFIX.get(prefix, "unknown")

def load_documents(data_dir: str) -> Dict[str, str]:
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
    """Lightweight TF-IDF + cosine similarity retriever."""
    def __init__(self, corpus: Dict[str, str]):
        self.ids = list(corpus.keys())
        texts = [corpus[i] for i in self.ids]
        self.vectorizer = TfidfVectorizer(stop_words="english")
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
# Filtering helpers
# ----------------------------

def build_metadata(corpus: Dict[str, str]) -> Dict[str, dict]:
    """Create tiny metadata for each doc: {id: {type, year}}."""
    meta = {}
    for doc_id, text in corpus.items():
        meta[doc_id] = {
            "type": infer_type(doc_id),
            "year": infer_year(text),
        }
    return meta

def apply_filters(ids: List[str], meta: Dict[str, dict],
                  min_year: Optional[int], types: Optional[Set[str]]) -> List[str]:
    """Return ids that pass the (min_year, types) filters."""
    out = []
    for d in ids:
        t = meta[d]["type"]
        y = meta[d]["year"]
        if types and t not in types:
            continue
        if min_year and (y is None or y < min_year):
            continue
        out.append(d)
    return out


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Dynamic metadata filtering demo")
    ap.add_argument("--question", required=True, help="User query")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")
    ap.add_argument("--k", type=int, default=5, help="Top-k to return")
    ap.add_argument("--mode", choices=["pre", "post"], default="pre",
                    help="Apply filters before retrieval (pre) or after (post)")
    ap.add_argument("--min_year", type=int, default=None, help="Keep docs with year >= this")
    ap.add_argument("--types", type=str, default=None,
                    help="Comma-separated doc types to keep (e.g., policy,design,research)")
    args = ap.parse_args()

    # Load corpus and metadata
    corpus = load_documents(args.data_dir)
    meta = build_metadata(corpus)
    enc = get_encoder()

    # Unfiltered baseline: retrieve directly for comparison
    base_ret = TfidfRetriever(corpus)
    base_hits = base_ret.search(args.question, top_k=args.k)
    base_ids = [doc_id for doc_id, _ in base_hits]
    base_ctx = "\n\n---\n\n".join(corpus[i] for i in base_ids)
    base_tokens = count_tokens(base_ctx, enc)

    # Parse filters
    keep_types = None
    if args.types:
        keep_types = {t.strip().lower() for t in args.types.split(",") if t.strip()}
    min_year = args.min_year

    # Apply filters pre or post
    if args.mode == "pre":
        # Pre-filter corpus, then retrieve
        keep_ids = apply_filters(list(corpus.keys()), meta, min_year, keep_types)
        filtered_corpus = {i: corpus[i] for i in keep_ids}
        if not filtered_corpus:
            print("[WARN] Filters eliminated all documents. Showing baseline instead.")
            filtered_hits, filtered_ids = [], []
            filtered_ctx, filtered_tokens = "", 0
        else:
            ret = TfidfRetriever(filtered_corpus)
            filtered_hits = ret.search(args.question, top_k=args.k)
            filtered_ids = [doc_id for doc_id, _ in filtered_hits]
            filtered_ctx = "\n\n---\n\n".join(filtered_corpus[i] for i in filtered_ids)
            filtered_tokens = count_tokens(filtered_ctx, enc)
    else:
        # Post-filter the retrieved set
        ret = TfidfRetriever(corpus)
        hits = ret.search(args.question, top_k=args.k * 2)  # fetch a few extra to survive filtering
        ids = [doc_id for doc_id, _ in hits]
        filtered_ids = apply_filters(ids, meta, min_year, keep_types)[: args.k]
        filtered_hits = [(i, s) for (i, s) in hits if i in set(filtered_ids)]
        filtered_ctx = "\n\n---\n\n".join(corpus[i] for i in filtered_ids)
        filtered_tokens = count_tokens(filtered_ctx, enc)

    # Console prints (blog-support)
    print("\nQuestion:")
    print(args.question)

    print("\nFilters:")
    print(f"  mode: {args.mode}")
    print(f"  min_year: {min_year if min_year else 'None'}")
    print(f"  types: {','.join(sorted(keep_types)) if keep_types else 'None'}")

    print("\nUnfiltered baseline (top-k):")
    for i, (doc_id, sim) in enumerate(base_hits, 1):
        t = meta[doc_id]["type"]; y = meta[doc_id]["year"]
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})  [type={t}, year={y}]")
    print(f"tokens_unfiltered ≈ {base_tokens}")

    print("\nFiltered results (top-k):")
    if not filtered_hits:
        print("  (no results after filtering)")
    else:
        for i, (doc_id, sim) in enumerate(filtered_hits, 1):
            t = meta[doc_id]["type"]; y = meta[doc_id]["year"]
            print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})  [type={t}, year={y}]")
    print(f"tokens_filtered   ≈ {filtered_tokens}")

    print("\nFinal filtered context:")
    print(filtered_ctx)


if __name__ == "__main__":
    main()
