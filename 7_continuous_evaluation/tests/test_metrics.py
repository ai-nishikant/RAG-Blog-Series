import pytest

from rag_eval.metrics import (
    compute_intrinsic_metrics,
    compute_extrinsic_metrics,
    compute_behavioral_metrics,
    compute_all_metrics,
    MetricsResult,
)


def test_compute_intrinsic_basic_overlap():
    reference = "The policy was updated in 2024 to include AI auditing guidelines."
    output = "The policy was updated in 2024 with new AI auditing guidelines."
    score = compute_intrinsic_metrics(reference, output)
    assert 0.0 < score <= 1.0


def test_compute_intrinsic_empty_reference():
    reference = ""
    output = "Anything here."
    score = compute_intrinsic_metrics(reference, output)
    assert score == 0.0


def test_compute_intrinsic_none_raises():
    with pytest.raises(ValueError):
        compute_intrinsic_metrics(None, "output")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        compute_intrinsic_metrics("reference", None)  # type: ignore[arg-type]


def test_compute_extrinsic_valid():
    metrics = compute_extrinsic_metrics(latency_ms=100, token_count=42, retrieval_ms=30)
    assert metrics["latency_ms"] == 100.0
    assert metrics["token_count"] == 42.0
    assert metrics["retrieval_ms"] == 30.0


def test_compute_extrinsic_negative_raises():
    with pytest.raises(ValueError):
        compute_extrinsic_metrics(latency_ms=-1, token_count=10)

    with pytest.raises(ValueError):
        compute_extrinsic_metrics(latency_ms=10, token_count=-5)

    with pytest.raises(ValueError):
        compute_extrinsic_metrics(latency_ms=10, token_count=5, retrieval_ms=-1)


def test_compute_behavioral_length():
    output = "This is a short sentence."
    metrics = compute_behavioral_metrics(output)
    assert metrics["length"] == len(output.split())


def test_compute_all_metrics_returns_metrics_result():
    reference = "The policy was updated in 2024 to include AI auditing guidelines."
    output = "The policy was updated in 2024 with new AI auditing guidelines."
    result = compute_all_metrics(
        reference=reference,
        output=output,
        latency_ms=120,
        token_count=80,
        retrieval_ms=30,
    )
    assert isinstance(result, MetricsResult)
    assert "latency_ms" in result.extrinsic
    assert "length" in result.behavioral
