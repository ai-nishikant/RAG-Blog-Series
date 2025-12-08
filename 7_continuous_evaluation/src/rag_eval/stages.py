"""
Multi-Stage Evaluation Loop (Stage A → Stage B → Stage C)

This module operationalizes Step 2 from the blog:
- Stage A: Pre-retrieval checks
- Stage B: Post-retrieval checks
- Stage C: Post-generation checks

Each stage returns:
- a boolean (pass/fail)
- a diagnostic dictionary
"""

from typing import Dict, Any


# -----------------------
# Stage A: Pre-retrieval
# -----------------------
def stage_a_pre_retrieval(query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect malformed queries, routing issues, missing parameters, etc.
    """

    diagnostics = {
        "query_present": bool(query.strip()),
        "has_domain": "domain" in metadata,
        "complexity_ok": len(query.split()) <= metadata.get("max_query_tokens", 50),
    }

    passed = all(diagnostics.values())
    return {"passed": passed, "diagnostics": diagnostics}


# -----------------------
# Stage B: Post-retrieval
# -----------------------
def stage_b_post_retrieval(retrieved_docs: list, top_k: int = 3) -> Dict[str, Any]:
    """
    Evaluate retrieval signals:
    - relevance
    - coverage
    - redundancy
    """

    diagnostics = {
        "docs_returned": len(retrieved_docs),
        "has_enough_docs": len(retrieved_docs) >= top_k,
        "unique_docs": len({doc["id"] for doc in retrieved_docs}) == len(retrieved_docs),
    }

    passed = diagnostics["has_enough_docs"] and diagnostics["unique_docs"]
    return {"passed": passed, "diagnostics": diagnostics}


# -----------------------
# Stage C: Post-generation
# -----------------------
def stage_c_post_generation(metrics: Dict[str, float], threshold: float = 0.4) -> Dict[str, Any]:
    """
    Uses intrinsic metrics to determine grounding quality.
    """

    intrinsic = metrics.get("intrinsic", 0.0)
    diagnostics = {
        "intrinsic_score": intrinsic,
        "above_threshold": intrinsic >= threshold,
    }

    passed = diagnostics["above_threshold"]
    return {"passed": passed, "diagnostics": diagnostics}


# -----------------------
# Combined evaluator
# -----------------------
def evaluate_all_stages(
    query: str,
    metadata: Dict[str, Any],
    retrieved_docs: list,
    gen_metrics: Dict[str, float],
) -> Dict[str, Any]:

    a = stage_a_pre_retrieval(query, metadata)
    b = stage_b_post_retrieval(retrieved_docs)
    c = stage_c_post_generation(gen_metrics)

    return {
        "stage_a": a,
        "stage_b": b,
        "stage_c": c,
        "overall_passed": a["passed"] and b["passed"] and c["passed"],
    }
