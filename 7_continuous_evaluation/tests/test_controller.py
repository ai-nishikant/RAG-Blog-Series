from rag_eval.controller import EvaluationController


def _make_stage_results(a_pass: bool, b_pass: bool, c_pass: bool):
    return {
        "stage_a": {"passed": a_pass, "diagnostics": {}},
        "stage_b": {"passed": b_pass, "diagnostics": {}},
        "stage_c": {"passed": c_pass, "diagnostics": {}},
        "overall_passed": a_pass and b_pass and c_pass,
    }


def test_controller_prefers_fix_query_when_stage_a_fails():
    controller = EvaluationController()
    stage_results = _make_stage_results(a_pass=False, b_pass=True, c_pass=True)

    action = controller.choose_action(stage_results)
    assert action == "fix_query"

    correction = controller.execute(action)
    assert correction["action"] == "fix_query"


def test_controller_adjusts_retrieval_when_stage_b_fails():
    controller = EvaluationController()
    stage_results = _make_stage_results(a_pass=True, b_pass=False, c_pass=True)

    action = controller.choose_action(stage_results)
    assert action == "adjust_retrieval"

    correction = controller.execute(action)
    assert correction["action"] == "adjust_retrieval"


def test_controller_adjusts_prompt_when_stage_c_fails():
    controller = EvaluationController()
    stage_results = _make_stage_results(a_pass=True, b_pass=True, c_pass=False)

    action = controller.choose_action(stage_results)
    assert action == "adjust_prompt"

    correction = controller.execute(action)
    assert correction["action"] == "adjust_prompt"


def test_controller_noop_when_all_stages_pass():
    controller = EvaluationController()
    stage_results = _make_stage_results(a_pass=True, b_pass=True, c_pass=True)

    action = controller.choose_action(stage_results)
    assert action == "noop"

    correction = controller.execute(action)
    assert correction is None
