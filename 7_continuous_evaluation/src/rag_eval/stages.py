"""
Multi-Stage Evaluation Loop (Stage A → Stage B → Stage C)

This module operationalizes Step 2 from the blog:
- Stage A: Pre-retrieval checks (query quality and intent validation)
- Stage B: Post-retrieval checks (document relevance and coverage)
- Stage C: Post-generation checks (response quality and grounding)

Each stage returns:
- a boolean (pass/fail)
- a diagnostic dictionary with specific metrics and checks

This multi-stage approach allows pinpointing exactly where in the RAG pipeline
quality issues occur, enabling targeted corrective actions.
"""

from typing import Dict, Any


# -----------------------
# Stage A: Pre-retrieval
# -----------------------
def stage_a_pre_retrieval(query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect malformed queries, routing issues, missing parameters, etc.
    
    Stage A validates the incoming query BEFORE attempting retrieval. This is the
    earliest point to catch issues and the cheapest place to fail fast. Checks include:
    - Query clarity and presence
    - Required metadata/context availability
    - Query complexity within acceptable bounds
    
    If Stage A fails, the system can:
    - Request clarification from the user
    - Rewrite/expand the query
    - Add missing context from conversation history
    
    Args:
        query: The user's input query string
        metadata: Additional context like domain, user preferences, conversation history
    
    Returns:
        Dict containing:
        - passed: True if all Stage A checks pass
        - diagnostics: Detailed results of each check
    """
    # Run a suite of pre-retrieval quality checks
    diagnostics = {
        # Check 1: Query is not empty or whitespace-only
        "query_present": bool(query.strip()),
        
        # Check 2: Required domain/context metadata is provided
        # This might indicate user intent, document corpus to search, etc.
        "has_domain": "domain" in metadata,
        
        # Check 3: Query complexity is reasonable (not too verbose or convoluted)
        # Very long queries may indicate confusion or need for decomposition
        "complexity_ok": len(query.split()) <= metadata.get("max_query_tokens", 50),
    }

    # Stage passes only if ALL checks are satisfied
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
    
    Stage B validates that the retrieval step successfully found relevant,
    high-quality documents. This stage catches issues like:
    - Insufficient documents retrieved (corpus gaps)
    - Duplicate documents (poor diversity)
    - Low relevance scores (query-document mismatch)
    
    If Stage B fails, the system can:
    - Increase K (retrieve more documents)
    - Adjust similarity thresholds
    - Use query expansion or reformulation
    - Try hybrid search (keyword + semantic)
    
    Args:
        retrieved_docs: List of retrieved document dictionaries, each with an "id" field
        top_k: Minimum number of documents expected (default: 3)
    
    Returns:
        Dict containing:
        - passed: True if retrieval quality is acceptable
        - diagnostics: Detailed metrics about retrieved documents
    """
    # Evaluate the quality of retrieved documents
    diagnostics = {
        # Metric 1: How many documents were actually retrieved?
        "docs_returned": len(retrieved_docs),
        
        # Check 1: Did we get enough documents to provide good coverage?
        # Fewer than top_k may indicate corpus sparsity or overly strict filters
        "has_enough_docs": len(retrieved_docs) >= top_k,
        
        # Check 2: Are all documents unique (no duplicates)?
        # Duplicates waste context window and indicate retrieval issues
        "unique_docs": len({doc["id"] for doc in retrieved_docs}) == len(retrieved_docs),
    }

    # Stage passes if we have enough unique documents
    passed = diagnostics["has_enough_docs"] and diagnostics["unique_docs"]
    
    return {"passed": passed, "diagnostics": diagnostics}


# -----------------------
# Stage C: Post-generation
# -----------------------
def stage_c_post_generation(metrics: Dict[str, float], threshold: float = 0.4) -> Dict[str, Any]:
    """
    Uses intrinsic metrics to determine grounding quality.
    
    Stage C evaluates the FINAL generated response after the LLM has processed
    the query and retrieved documents. This stage catches issues like:
    - Hallucinations (content not grounded in retrieved docs)
    - Poor quality responses despite good retrieval
    - Factual errors or inconsistencies
    
    If Stage C fails, the system can:
    - Adjust prompt engineering (add constraints, examples)
    - Switch from zero-shot to few-shot prompting
    - Add explicit grounding instructions
    - Change LLM temperature or other generation parameters
    
    Args:
        metrics: Dictionary containing computed quality metrics (intrinsic, extrinsic, behavioral)
        threshold: Minimum acceptable intrinsic quality score (default: 0.4)
    
    Returns:
        Dict containing:
        - passed: True if generation quality meets threshold
        - diagnostics: Quality score and threshold comparison
    """
    # Extract the intrinsic quality metric (e.g., token overlap with reference)
    intrinsic = metrics.get("intrinsic", 0.0)
    
    # Evaluate generation quality
    diagnostics = {
        # The computed quality score for this response
        "intrinsic_score": intrinsic,
        
        # Check: Does quality meet our minimum acceptable threshold?
        # Scores below threshold indicate hallucination, poor grounding, or irrelevance
        "above_threshold": intrinsic >= threshold,
    }

    # Stage passes if quality score is acceptable
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
    """
    Run all three evaluation stages and aggregate results.
    
    This function orchestrates the complete multi-stage evaluation process,
    running each stage in sequence and combining their results. The stages
    are independent - all three run regardless of individual failures, which
    provides a complete diagnostic picture.
    
    The overall_passed flag indicates whether the entire RAG pipeline produced
    acceptable output. If it's False, the individual stage results tell us
    exactly where the problem occurred:
    - Stage A failure → query quality issue
    - Stage B failure → retrieval quality issue  
    - Stage C failure → generation quality issue
    
    Args:
        query: User's input query
        metadata: Additional context and parameters
        retrieved_docs: Documents returned by retrieval
        gen_metrics: Computed metrics for the generated response
    
    Returns:
        Dict containing:
        - stage_a: Complete Stage A results (passed + diagnostics)
        - stage_b: Complete Stage B results (passed + diagnostics)
        - stage_c: Complete Stage C results (passed + diagnostics)
        - overall_passed: True only if ALL stages passed
    """
    # Run Stage A: Validate query quality before retrieval
    a = stage_a_pre_retrieval(query, metadata)
    
    # Run Stage B: Validate retrieval quality
    b = stage_b_post_retrieval(retrieved_docs)
    
    # Run Stage C: Validate generation quality
    c = stage_c_post_generation(gen_metrics)

    # Aggregate all results with an overall pass/fail flag
    # Overall passes only if every stage passed
    return {
        "stage_a": a,
        "stage_b": b,
        "stage_c": c,
        "overall_passed": a["passed"] and b["passed"] and c["passed"],
    }
