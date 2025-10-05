#!/usr/bin/env python3
"""
06_hybrid.py — Hybrid Pipeline (blog-support demo)

What this shows
---------------
A compact end-to-end pipeline that chains the most practical techniques:
  1) Multi-query rewriting (fallback synonyms or Groq LLM)
  2) (Pre) metadata filtering (year, type)
  3) Multi-stage retrieval (broad HashingVectorizer -> TF-IDF refine)
  4) Optional reranking (Cohere) with lexical fallback
  5) Summarization under tight context windows (Groq or extractive)

Why it matters
--------------
Real systems mix methods. This script demonstrates a small, didactic composition
that improves recall and precision, then packs context to fit a target budget.

Usage
-----
python scripts/06_hybrid.py \
  --question "Summarize the evaluation approach and key results." \
  --data_dir data/sample_docs \
  --plan small \
  --k_broad 10 \
  --k_final 5 \
  --paraphrases 3 \
  --use_llm \
  --use_cohere \
  --cohere_model rerank-english-v3.0 \
  --min_year 2024 \
  --types research,policy
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


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
# Metadata inference + filtering
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
    years = [int(y) for y in YEAR_RE.findall(text)]
    return max(years) if years else None

def infer_type(doc_id: str) -> str:
    prefix = doc_id.split("_", 1)[0].lower()
    return DOC_TYPE_FROM_PREFIX.get(prefix, "unknown")

def build_metadata(corpus: Dict[str, str]) -> Dict[str, dict]:
    meta = {}
    for doc_id, text in corpus.items():
        meta[doc_id] = {"type": infer_type(doc_id), "year": infer_year(text)}
    return meta

def apply_filters(ids: List[str], meta: Dict[str, dict],
                  min_year: Optional[int], types: Optional[Set[str]]) -> List[str]:
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
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY for --use_llm rewrites.")
    client = Groq(api_key=key)
    prompt = (
        "Rewrite the user's query into distinct, concise paraphrases that preserve meaning. "
        "Avoid adding new topics. Output each paraphrase on a new line."
        f"\n\nUser query: {query}\n"
        f"Number of paraphrases: {n}\n"
    )
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You generate safe, concise paraphrases."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    text = out.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    uniq, seen = [], set()
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            uniq.append(ln)
            seen.add(key)
        if len(uniq) >= n:
            break
    return uniq


# ----------------------------
# Multi-stage retrieval
# ----------------------------

def broad_candidates_hashing(corpus: Dict[str, str], question: str,
                             k_broad: int = 10, n_features: int = 2**16) -> List[Tuple[str, float]]:
    ids = list(corpus.keys())
    texts = [corpus[i] for i in ids]
    vect = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2", stop_words="english")
    X = vect.transform(texts)
    q = vect.transform([question])
    nn = NearestNeighbors(metric="cosine").fit(X)
    n = min(k_broad, len(ids))
    dists, idx = nn.kneighbors(q, n_neighbors=n)
    out = []
    for i, d in zip(idx[0], dists[0]):
        out.append((ids[i], 1.0 - float(d)))
    return out

def refine_tfidf(corpus: Dict[str, str], question: str, candidates: List[str], k_final: int = 5) -> List[Tuple[str, float]]:
    tfidf = TfidfVectorizer(stop_words="english")
    cand_texts = [corpus[cid] for cid in candidates]
    X = tfidf.fit_transform(cand_texts)
    q = tfidf.transform([question])
    sims = cosine_similarity(q, X)[0]
    order = np.argsort(-sims)[: min(k_final, len(candidates))]
    return [(candidates[i], float(sims[i])) for i in order]


# ----------------------------
# Reranking (Cohere or lexical)
# ----------------------------

def lexical_rerank(query: str, candidates: List[Tuple[str, str]], top_n: int) -> List[Tuple[str, float]]:
    q_tokens = [t for t in query.lower().split() if t.isascii()]
    bigrams = set(zip(q_tokens, q_tokens[1:])) if len(q_tokens) > 1 else set()
    scored: List[Tuple[str, float]] = []
    for doc_id, text in candidates:
        t = text.lower()
        tf_score = sum(t.count(f" {qt} ") + t.startswith(qt + " ") + t.endswith(" " + qt) for qt in q_tokens)
        bg_score = sum(t.count(f"{a} {b}") for a, b in bigrams)
        scored.append((doc_id, float(tf_score + 0.5 * bg_score)))
    scored.sort(key=lambda x: -x[1])
    return scored[: min(top_n, len(scored))]

def cohere_rerank(query: str, candidates: List[Tuple[str, str]], top_n: int, model: str) -> List[Tuple[str, float]]:
    try:
        import cohere
        client = getattr(cohere, "ClientV2", None)
        client = client(os.getenv("COHERE_API_KEY")) if client else cohere.Client(os.getenv("COHERE_API_KEY"))
        if not os.getenv("COHERE_API_KEY"):
            raise RuntimeError("Missing COHERE_API_KEY")
        texts = [text for _, text in candidates]
        resp = client.rerank(query=query, documents=texts, top_n=min(top_n, len(texts)), model=model)
        pairs = []
        for r in getattr(resp, "results", []):
            idx = getattr(r, "index", None)
            score = getattr(r, "relevance_score", None)
            if idx is None or score is None:
                continue
            pairs.append((int(idx), float(score)))
        id_map = [doc_id for doc_id, _ in candidates]
        ranked = [(id_map[i], score) for i, score in pairs]
        return ranked[: min(top_n, len(ranked))]
    except Exception as e:
        print(f"[WARN] Cohere rerank failed: {e}. Falling back to lexical rerank.")
        return lexical_rerank(query, candidates, top_n)


# ----------------------------
# Summarization (Groq or extractive)
# ----------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def extractive_lead(text: str, max_sentences: int = 3) -> str:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return " ".join(sents[:max_sentences]) if sents else text.strip()

def groq_summarize(text: str, bullets: int = 3, model: str = "llama-3.1-8b-instant") -> str:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY for --use_llm summarization.")
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
# Main Orchestration
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Hybrid pipeline demo (multi-query + multistage + rerank + summarize)")
    ap.add_argument("--question", required=True, help="User query")
    ap.add_argument("--data_dir", default="data/sample_docs", help="Folder of .txt docs")

    # Multi-query
    ap.add_argument("--paraphrases", type=int, default=3, help="Number of LLM/static rewrites")
    ap.add_argument("--use_llm", action="store_true", help="Use Groq LLM for rewrites and summarization")

    # Filtering (pre)
    ap.add_argument("--min_year", type=int, default=None, help="Keep docs with year >= this")
    ap.add_argument("--types", type=str, default=None, help="Comma-separated doc types to keep (e.g., policy,design,research)")

    # Multi-stage retrieval
    ap.add_argument("--k_broad", type=int, default=10, help="Stage-1 candidate count")
    ap.add_argument("--k_final", type=int, default=5, help="Stage-2 final count")

    # Rerank
    ap.add_argument("--use_cohere", action="store_true", help="Use Cohere Rerank v3")
    ap.add_argument("--cohere_model", default="rerank-english-v3.0", help="Cohere model name")
    ap.add_argument("--k_after_rerank", type=int, default=5, help="Top-k after reranking")

    # Plans
    ap.add_argument("--plan", choices=["small", "large"], default="small", help="Context plan")
    ap.add_argument("--max_ctx", type=int, default=None, help="Override max context tokens")
    ap.add_argument("--margin", type=int, default=None, help="Reserve tokens for instructions/answer")
    ap.add_argument("--bullets", type=int, default=None, help="Bullet points or lead sentences per passage")

    args = ap.parse_args()
    load_dotenv(override=True)

    # Plan setup
    plan_cfg = PLANS[args.plan].copy()
    if args.max_ctx is not None:
        plan_cfg["max_ctx"] = int(args.max_ctx)
    if args.margin is not None:
        plan_cfg["margin"] = int(args.margin)
    if args.bullets is not None:
        plan_cfg["bullets"] = int(args.bullets)

    # Load corpus + metadata
    corpus = load_documents(args.data_dir)
    meta = build_metadata(corpus)
    enc = get_encoder()

    # 1) Multi-query (original + rewrites)
    queries = [args.question]
    try:
        if args.use_llm:
            queries += llm_rewrites_with_groq(args.question, n=args.paraphrases)
        else:
            queries += static_rewrites(args.question, max_variants=args.paraphrases)
    except Exception as e:
        print(f"[WARN] LLM rewriting failed: {e}. Falling back to static rewrites.")
        queries += static_rewrites(args.question, max_variants=args.paraphrases)

    # Deduplicate variants
    uniq = []
    seen = set()
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            uniq.append(q.strip())
            seen.add(key)

    # 2) (Pre) Metadata filtering on the candidate universe
    keep_types = {t.strip().lower() for t in args.types.split(",")} if args.types else None
    universe_ids = list(corpus.keys())
    filtered_ids = apply_filters(universe_ids, meta, args.min_year, keep_types)
    filtered_corpus = {i: corpus[i] for i in filtered_ids} if filtered_ids else corpus

    # 3) Multi-stage retrieval per variant, merged by max score
    def stage1(question: str) -> List[Tuple[str, float]]:
        return broad_candidates_hashing(filtered_corpus, question, k_broad=args.k_broad)

    def stage2(question: str, cand_ids: List[str]) -> List[Tuple[str, float]]:
        return refine_tfidf(filtered_corpus, question, cand_ids, k_final=args.k_final)

    scores: Dict[str, float] = {}
    for q in uniq:
        broad = stage1(q)
        cand_ids = [doc_id for doc_id, _ in broad]
        refined = stage2(q, cand_ids)
        for doc_id, s in refined:
            scores[doc_id] = max(scores.get(doc_id, 0.0), float(s))
    merged_refined = sorted(scores.items(), key=lambda x: -x[1])

    # 4) Rerank final candidate pool (optional Cohere, else lexical)
    top_for_rerank = merged_refined[: max(args.k_after_rerank * 2, args.k_final)]
    cand_ids = [doc_id for doc_id, _ in top_for_rerank]
    pairs = [(doc_id, filtered_corpus.get(doc_id, corpus[doc_id])) for doc_id in cand_ids]

    if args.use_cohere:
        reranked = cohere_rerank(args.question, pairs, top_n=args.k_after_rerank, model=args.cohere_model)
    else:
        reranked = lexical_rerank(args.question, pairs, top_n=args.k_after_rerank)

    final_ids = [doc_id for doc_id, _ in reranked]
    chunks = [filtered_corpus.get(i, corpus[i]) for i in final_ids]

    # 5) Summarize to fit plan budget if needed
    system = "Answer using only the provided context. If unknown, say you do not know."
    header = f"User question: {args.question}\nContext:\n"
    naive_sections = [system, header] + chunks
    naive_tokens = total_tokens(naive_sections, enc)

    fits_naive = (naive_tokens + plan_cfg["margin"] <= plan_cfg["max_ctx"])
    if fits_naive or args.plan == "large":
        final_chunks = chunks
        label = "naive (no compression)"
    else:
        comp = []
        for t in chunks:
            comp.append(compress_passage(t, use_llm=args.use_llm, bullets=plan_cfg["bullets"]))
        tentative = [system, header] + comp
        t_tok = total_tokens(tentative, enc)
        if t_tok + plan_cfg["margin"] > plan_cfg["max_ctx"]:
            acc: List[str] = []
            for c in comp:
                trial = [system, header] + acc + [c]
                if total_tokens(trial, enc) + plan_cfg["margin"] <= plan_cfg["max_ctx"]:
                    acc.append(c)
                else:
                    break
            final_chunks = acc
            label = "compressed + truncated to fit"
        else:
            final_chunks = comp
            label = "compressed to fit"

    final_sections = [system, header] + final_chunks
    final_tokens = total_tokens(final_sections, enc)

    # ----------------------------
    # Console output
    # ----------------------------
    print("\nQuestion:")
    print(args.question)

    print("\nQuery variants considered:")
    for i, q in enumerate(uniq, 1):
        print(f"  {i:02d}. {q}")

    print("\nFilters (pre):")
    print(f"  min_year: {args.min_year if args.min_year else 'None'}")
    print(f"  types: {','.join(sorted(keep_types)) if keep_types else 'None'}")
    print(f"  filtered_universe_size: {len(filtered_corpus)} / total: {len(corpus)}")

    print("\nMerged refined candidates (before rerank):")
    for i, (doc_id, score) in enumerate(merged_refined[: max(10, args.k_after_rerank)], 1):
        m = meta[doc_id]
        print(f"  {i:02d}. {doc_id} (score≈{score:.3f}) [type={m['type']}, year={m['year']}]")

    print("\nReranked top-k:")
    tag = "cohere" if args.use_cohere and os.getenv("COHERE_API_KEY") else "lexical"
    for i, (doc_id, score) in enumerate(reranked, 1):
        m = meta[doc_id]
        print(f"  {i:02d}. {doc_id} ({tag}_score≈{score:.3f}) [type={m['type']}, year={m['year']}]")

    print("\nPlan + token accounting:")
    print(f"  plan={args.plan}  max_ctx={plan_cfg['max_ctx']}  margin={plan_cfg['margin']}  bullets={plan_cfg['bullets']}")
    print(f"  tokens_naive_context ≈ {naive_tokens}")
    print(f"  tokens_final_context ≈ {final_tokens}  (mode={label})")

    print("\nFinal context:")
    print(system)
    print(header)
    print("\n\n---\n\n".join(final_chunks))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
