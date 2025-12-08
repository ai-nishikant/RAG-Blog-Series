from rag_eval.stages import (
    stage_a_pre_retrieval,
    stage_b_post_retrieval,
    stage_c_post_generation,
    evaluate_all_stages,
)


def test_stage_a_pre_retrieval_passes_on_valid_query_and_metadata():
    query = "What changed in the AI auditing policy in 2024?"
    metadata = {"domain": "policy", "max_query_tokens": 50}
    result = stage_a_pre_retrieval(query, metadata)
    assert result["passed"] is True
    assert result["diagnostics"]["query_present"] is True
    assert result["diagnostics"]["has_domain"] is True
    assert result["diagnostics"]["complexity_ok"] is True


def test_stage_a_pre_retrieval_fails_on_missing_domain():
    query = "What changed in the AI auditing policy in 2024?"
    metadata = {"max_query_tokens": 50}
    result = stage_a_pre_retrieval(query, metadata)
    assert result["passed"] is False
    assert result["diagnostics"]["has_domain"] is False


def test_stage_b_post_retrieval_passes_with_sufficient_unique_docs():
    retrieved_docs = [
        {"id": "doc1", "content": "content 1"},
        {"id": "doc2", "content": "content 2"},
        {"id": "doc3", "content": "content 3"},
    ]
    result = stage_b_post_retrieval(retrieved_docs, top_k=3)
    assert result["passed"] is True
    assert result["diagnostics"]["has_enough_docs"] is True
    assert result["diagnostics"]["unique_docs"] is True


def test_stage_b_post_retrieval_fails_with_duplicates_or_too_few_docs():
    retrieved_docs = [
        {"id": "doc1", "content": "content 1"},
        {"id": "doc1", "content": "duplicate content"},
    ]
    result = stage_b_post_retrieval(retrieved_docs, top_k=3)
    assert result["passed"] is False
    assert result["diagnostics"]["unique_docs"] is False
    assert result["diagnostics"]["has_enough_docs"] is False


def test_stage_c_post_generation_respects_threshold():
    metrics_good = {"intrinsic": 0.6}
    metrics_bad = {"intrinsic": 0.2}

    result_good = stage_c_post_generation(metrics_good, threshold=0.4)
    result_bad = stage_c_post_generation(metrics_bad, threshold=0.4)

    assert result_good["passed"] is True
    assert result_good["diagnostics"]["above_threshold"] is True

    assert result_bad["passed"] is False
    assert result_bad["diagnostics"]["above_threshold"] is False


def test_evaluate_all_stages_combines_results_correctly():
    query = "What changed in the AI auditing policy in 2024?"
    metadata = {"domain": "policy", "max_query_tokens": 50}
    retrieved_docs = [
        {"id": "doc1", "content": "content 1"},
        {"id": "doc2", "content": "content 2"},
        {"id": "doc3", "content": "content 3"},
    ]
    gen_metrics = {"intrinsic": 0.8}

    result = evaluate_all_stages(query, metadata, retrieved_docs, gen_metrics)

    assert "stage_a" in result
    assert "stage_b" in result
    assert "stage_c" in result
    assert "overall_passed" in result
    assert result["overall_passed"] is True
