#!/usr/bin/env python3
"""
05_summarize.py — Summarization under Tight Context (flagship demo)

What this shows
---------------
Retrieve top-k passages, estimate token footprint, and if the context would
overflow a small window, compress the passages (LLM or extractive fallback)
to fit the budget.

Why it matters
--------------
When context windows are tight, naive retrieval overflows quickly. Compact,
information-dense summaries preserve answer quality while controlling cost.

Usage
-----
# Small window with LLM summarization (Groq) enabled:
python scripts/05_summarize.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k 5 \
  --plan small \
  --use_llm

# Large window, likely no compression:
python scripts/05_summarize.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --k 8 \
  --plan large

# Override budgets explicitly (tokens):
python scripts/05_summarize.py \
  --question "Outline the incident response flow from detection to lessons learned." \
  --max_ctx 1400 --margin 256
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def total_tokens(sections: List[str], enc=None) -> int:
    enc = enc or get_encoder()
    return sum(count_tokens(s, enc) for s in sections)


# ----------------------------
# Retrieval (TF-IDF baseline)
# ----------------------------

class TfidfRetriever:
    """Lightweight TF-IDF + cosine (NearestNeighbors) retriever."""
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
# Summarization
# ----------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def extractive_lead(text: str, max_sentences: int = 3) -> str:
    """
    Simple extractive fallback: take the first N sentences.
    Keeps behavior deterministic and dependency-free.
    """
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return " ".join(sents[:max_sentences]) if sents else text.strip()


def groq_summarize(text: str, bullets: int = 3, model: str = "llama-3.1-8b-instant") -> str:
    """
    Abstractive summarization via Groq (optional). Requires GROQ_API_KEY in .env.
    """
    from groq import Groq  # lazy import so file runs without Groq installed
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Create .env or drop --use_llm.")
    client = Groq(api_key=key)

    system = "You are a precise technical summarizer. Keep facts, remove fluff."
    user = f"Summarize into {bullets} concise bullet points:\n\n{text}"
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=400,
    )
    return out.choices[0].message.content.strip()


def compress_passage(text: str, use_llm: bool, bullets: int) -> str:
    if use_llm:
        try:
            return groq_summarize(text, bullets=bullets)
        except Exception as e:
            print(f"[WARN] LLM summarization failed: {e}. Falling back to extractive.")
    return extractive_lead(text, max_sentences=bullets)


# ----------------------------
# Plans (small vs large)
# ----------------------------

PLANS = {
    "small": {"max_ctx": 1400, "margin": 256, "bullets": 3},
    "large": {"max_ctx": 8000, "margin": 256, "bullets": 5},
}


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Summarization under tight context demo")
    ap.add_argument("--question", required=True, help="User query")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")
    ap.add_argument("--k", type=int, default=5, help="Top-k to retrieve")
    ap.add_argument("--plan", choices=["small", "large"], default="small",
                    help="Context plan: small=compact, large=roomy")
    ap.add_argument("--max_ctx", type=int, default=None,
                    help="Override max context tokens (prompt+context budget)")
    ap.add_argument("--margin", type=int, default=None,
                    help="Reserve tokens for instructions/answer")
    ap.add_argument("--bullets", type=int, default=None,
                    help="How many bullet points or lead sentences")
    ap.add_argument("--use_llm", action="store_true",
                    help="Use Groq summarization; otherwise extractive fallback")
    args = ap.parse_args()

    load_dotenv(override=True)

    # Plan defaults
    plan_cfg = PLANS[args.plan].copy()
    if args.max_ctx is not None:
        plan_cfg["max_ctx"] = int(args.max_ctx)
    if args.margin is not None:
        plan_cfg["margin"] = int(args.margin)
    if args.bullets is not None:
        plan_cfg["bullets"] = int(args.bullets)

    # Load corpus and retrieve
    corpus = load_documents(args.data_dir)
    ret = TfidfRetriever(corpus)
    hits = ret.search(args.question, top_k=args.k)
    ids = [doc_id for doc_id, _ in hits]

    # Build naive context (no compression)
    system = "Answer using only the provided context. If unknown, say you do not know."
    header = f"User question: {args.question}\nContext:\n"
    chunks = [corpus[i] for i in ids]
    enc = get_encoder()
    naive_sections = [system, header] + chunks
    naive_tokens = total_tokens(naive_sections, enc)

    fits_naive = naive_tokens + 0 <= plan_cfg["max_ctx"]  # header already included
    print("\nQuestion:")
    print(args.question)

    print("\nRetrieved top-k (TF-IDF):")
    for i, (doc_id, sim) in enumerate(hits, 1):
        print(f"  {i:02d}. {doc_id}  (sim≈{sim:.3f})")

    print("\nPlan:")
    print(f"  name={args.plan}  max_ctx={plan_cfg['max_ctx']}  margin={plan_cfg['margin']}  bullets={plan_cfg['bullets']}")

    print("\nToken estimate:")
    print(f"  tokens_naive_context ≈ {naive_tokens}")

    if fits_naive or args.plan == "large":
        # For large plans, accept naive unless it grossly exceeds budget
        final_chunks = chunks
        final_label = "naive (no compression)"
    else:
        # Compress each chunk and rebuild
        compressed = [compress_passage(t, use_llm=args.use_llm, bullets=plan_cfg["bullets"]) for t in chunks]
        final_sections = [system, header] + compressed
        final_tokens = total_tokens(final_sections, enc)

        # If still too large, greedily drop from the tail
        if final_tokens + plan_cfg["margin"] > plan_cfg["max_ctx"]:
            acc: List[str] = []
            for c in compressed:
                trial = [system, header] + acc + [c]
                if total_tokens(trial, enc) + plan_cfg["margin"] <= plan_cfg["max_ctx"]:
                    acc.append(c)
                else:
                    break
            final_chunks = acc
            final_label = "compressed + truncated to fit"
        else:
            final_chunks = compressed
            final_label = "compressed to fit"

    final_sections = [system, header] + final_chunks
    final_tokens = total_tokens(final_sections, enc)

    print("\nToken estimate after optimization:")
    print(f"  mode={final_label}")
    print(f"  tokens_final_context ≈ {final_tokens}")

    print("\nFinal context:")
    print(system)
    print(header)
    print("\n\n---\n\n".join(final_chunks))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
