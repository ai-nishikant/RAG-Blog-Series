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
    """
    Container for all three metric families from a single evaluation.
    
    This dataclass aggregates the results of computing intrinsic, extrinsic,
    and behavioral metrics, making it easy to pass around complete metric
    information as a single object.
    """
    intrinsic: float                  # Quality score (e.g., token overlap with reference)
    extrinsic: Dict[str, float]       # Operational metrics (latency, token count, etc.)
    behavioral: Dict[str, float]      # Reasoning patterns (length, structure, etc.)


def compute_intrinsic_metrics(reference: str, output: str) -> float:
    """
    Compute intrinsic quality metrics by comparing output to a reference answer.
    
    This function uses a simple token overlap approach (similar to ROUGE-1) to
    measure how well the generated output matches the reference. It's a basic
    but effective way to assess factual accuracy and content coverage.
    
    Args:
        reference: The ground truth or expected answer
        output: The actual generated response from the RAG system
    
    Returns:
        float: Score between 0.0 and 1.0 representing the fraction of reference
               tokens that appear in the output (recall-based metric)
    
    Raises:
        ValueError: If reference or output is None
    """
    # Validate inputs
    if reference is None or output is None:
        raise ValueError("reference and output must not be None")

    # Tokenize and normalize both texts (lowercase, remove empty tokens)
    ref_tokens = {t for t in reference.lower().split() if t.strip()}
    out_tokens = {t for t in output.lower().split() if t.strip()}

    # Handle edge case: empty reference
    if not ref_tokens:
        return 0.0

    # Calculate overlap: intersection of token sets
    overlap = ref_tokens & out_tokens
    
    # Return recall score: what fraction of reference tokens are in the output?
    return len(overlap) / len(ref_tokens)


def compute_extrinsic_metrics(
    latency_ms: float,
    token_count: int,
    retrieval_ms: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute extrinsic (operational) metrics about the RAG system's performance.
    
    Extrinsic metrics capture system behavior and resource usage rather than
    output quality. These are critical for production systems where latency,
    cost, and throughput matter as much as accuracy.
    
    Args:
        latency_ms: Total end-to-end latency in milliseconds
        token_count: Number of tokens in the generated response
        retrieval_ms: Optional retrieval-specific latency in milliseconds
    
    Returns:
        dict: Dictionary of operational metrics, including:
              - latency_ms: Total response time
              - token_count: Response length (affects cost)
              - retrieval_ms: Time spent on document retrieval (if provided)
    
    Raises:
        ValueError: If any metric value is negative
    """
    # Validate that operational metrics are non-negative
    if latency_ms < 0 or token_count < 0:
        raise ValueError("latency_ms and token_count must not be negative")
    if retrieval_ms is not None and retrieval_ms < 0:
        raise ValueError("retrieval_ms must not be negative")

    # Build the metrics dictionary with required fields
    metrics: Dict[str, float] = {
        "latency_ms": float(latency_ms),      # Total time to generate response
        "token_count": float(token_count),     # Output length (cost indicator)
    }
    
    # Add optional retrieval timing if available
    # This helps identify whether slowness is in retrieval or generation
    if retrieval_ms is not None:
        metrics["retrieval_ms"] = float(retrieval_ms)
    
    return metrics


def compute_behavioral_metrics(output: str) -> Dict[str, float]:
    """
    Compute behavioral metrics that capture reasoning patterns in the output.
    
    Behavioral metrics analyze how the model structures its response, which can
    indicate quality issues like verbosity, terseness, or lack of reasoning.
    Currently tracks output length, but could be extended to measure:
    - Use of hedging language ("maybe", "possibly")
    - Citation patterns
    - Step-by-step reasoning presence
    - Confidence indicators
    
    Args:
        output: The generated response to analyze
    
    Returns:
        dict: Dictionary of behavioral metrics, currently containing:
              - length: Number of tokens in the output
    
    Raises:
        ValueError: If output is None
    """
    # Validate input
    if output is None:
        raise ValueError("output must not be None")

    # Count tokens by splitting on whitespace
    # This gives a rough measure of response verbosity
    length = len(output.split())
    
    return {"length": float(length)}


def compute_all_metrics(
    reference: str,
    output: str,
    latency_ms: float,
    token_count: int,
    retrieval_ms: Optional[float] = None,
) -> MetricsResult:
    """
    Compute all three metric families in one call for convenience.
    
    This is the primary entry point for getting a complete picture of RAG
    system performance. It combines quality assessment (intrinsic), operational
    metrics (extrinsic), and reasoning analysis (behavioral) into one result.
    
    Args:
        reference: Ground truth answer for comparison
        output: Generated response from RAG system
        latency_ms: Total response time in milliseconds
        token_count: Number of tokens in the output
        retrieval_ms: Optional retrieval latency in milliseconds
    
    Returns:
        MetricsResult: Object containing all computed metrics across the three families
    """
    # Compute each metric family independently
    intrinsic = compute_intrinsic_metrics(reference, output)
    extrinsic = compute_extrinsic_metrics(latency_ms, token_count, retrieval_ms)
    behavioral = compute_behavioral_metrics(output)

    # Package everything into a single result object
    return MetricsResult(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        behavioral=behavioral,
    )


# Demo/test code when run directly
if __name__ == "__main__":
    # Example reference answer (ground truth)
    ref = "The policy was updated in 2024 to include new AI auditing guidelines."
    # Example generated output (what the RAG system produced)
    out = "The new guidelines were introduced recently for AI auditing."

    # Compute all metrics for this example
    result = compute_all_metrics(ref, out, latency_ms=123, token_count=87)
    
    # Display results
    print("Intrinsic:", result.intrinsic)      # Quality score (token overlap)
    print("Extrinsic:", result.extrinsic)      # Operational metrics
    print("Behavioral:", result.behavioral)    # Reasoning patterns
