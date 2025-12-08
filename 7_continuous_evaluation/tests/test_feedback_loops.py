from rag_eval.feedback_loops import OfflineFeedbackLoop, OnlineFeedbackLoop


def test_offline_feedback_loop_records_and_summarizes():
    loop = OfflineFeedbackLoop()

    loop.record({"overall_passed": True})
    loop.record({"overall_passed": False})
    loop.record({"overall_passed": False})

    summary = loop.summarize()

    assert summary["runs"] == 3
    assert summary["failures"] == 2


def test_online_feedback_loop_reports_and_detects_drift():
    loop = OnlineFeedbackLoop()
    assert loop.has_drift() is False

    loop.report_issue({"scenario": "retrieval_drift", "action": "adjust_retrieval"})
    assert loop.has_drift() is True

    assert len(loop.live_issues) == 1
    assert loop.live_issues[0]["scenario"] == "retrieval_drift"
