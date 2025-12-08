"""
Instrumentation metrics for RAG evaluation.

Implements three metric families:
- intrinsic metrics   (output quality)
- extrinsic metrics   (operational behavior)
- behavioral metrics  (reasoning patterns)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MetricsResult:
    intrinsic: float
    extrinsic: Dict[str, float]
    behavioral: Dict[str, float]


def compute_intrinsic_metrics(reference: str, output: str) -> float:
    if reference is None or output is None:
        raise ValueError("reference and output must not be None")

    ref_tokens = {t for t in reference.lower().split() if t.strip()}
    out_tokens = {t for t in output.lower().split() if t.strip()}

    if not ref_tokens:
        return 0.0

    overlap = ref_tokens & out_tokens
    return len(overlap) / len(ref_tokens)


def compute_extrinsic_metrics(
    latency_ms: float,
    token_count: int,
    retrieval_ms: Optional[float] = None,
) -> Dict[str, float]:
    if latency_ms < 0 or token_count < 0:
        raise ValueError("latency_ms and token_count must not be negative")
    if retrieval_ms is not None and retrieval_ms < 0:
        raise ValueError("retrieval_ms must not be negative")

    metrics: Dict[str, float] = {
        "latency_ms": float(latency_ms),
        "token_count": float(token_count),
    }
    if retrieval_ms is not None:
        metrics["retrieval_ms"] = float(retrieval_ms)
    return metrics


def compute_behavioral_metrics(output: str) -> Dict[str, float]:
    if output is None:
        raise ValueError("output must not be None")

    length = len(output.split())
    return {"length": float(length)}


def compute_all_metrics(
    reference: str,
    output: str,
    latency_ms: float,
    token_count: int,
    retrieval_ms: Optional[float] = None,
) -> MetricsResult:
    intrinsic = compute_intrinsic_metrics(reference, output)
    extrinsic = compute_extrinsic_metrics(latency_ms, token_count, retrieval_ms)
    behavioral = compute_behavioral_metrics(output)

    return MetricsResult(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        behavioral=behavioral,
    )


if __name__ == "__main__":
    ref = "The policy was updated in 2024 to include new AI auditing guidelines."
    out = "The new guidelines were introduced recently for AI auditing."

    result = compute_all_metrics(ref, out, latency_ms=123, token_count=87)
    print("Intrinsic:", result.intrinsic)
    print("Extrinsic:", result.extrinsic)
    print("Behavioral:", result.behavioral)
