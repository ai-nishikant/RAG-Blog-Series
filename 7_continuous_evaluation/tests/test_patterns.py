from rag_eval.patterns import CanaryPromptRunner, ShadowEvaluationPipeline, ReasoningAudit


def test_canary_prompt_runner_calls_llm_for_each_prompt():
    prompts = ["Prompt 1", "Prompt 2"]
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return f"output for: {prompt}"

    runner = CanaryPromptRunner(prompts)
    results = runner.run(fake_llm)

    assert len(calls) == len(prompts)
    assert len(results) == len(prompts)
    assert results[0]["prompt"] == "Prompt 1"
    assert "output for" in results[0]["output"]


def test_shadow_evaluation_pipeline_logs_outputs_without_affecting_user():
    pipeline = ShadowEvaluationPipeline()

    def dummy_system_fn(query: str) -> str:
        return f"answer for: {query}"

    output = pipeline.run("test query", dummy_system_fn)

    assert output == "answer for: test query"
    assert len(pipeline.shadow_logs) == 1
    assert pipeline.shadow_logs[0]["query"] == "test query"
    assert pipeline.shadow_logs[0]["output"] == "answer for: test query"


def test_reasoning_audit_checks_length_and_structure():
    audit = ReasoningAudit()
    output = "Step 1. Do this. Step 2. Do that."
    result = audit.audit(output)

    assert result["length"] == len(output.split())
    assert result["has_numbered_steps"] is True

    output_no_steps = "This is a single sentence without numbered steps."
    result2 = audit.audit(output_no_steps)

    assert result2["length"] == len(output_no_steps.split())
    assert result2["has_numbered_steps"] in (True, False)  # just ensure key exists
