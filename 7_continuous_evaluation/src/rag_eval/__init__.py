"""
RAG Evaluation Toolkit

This package implements the components described in the blog:
“Why RAG Systems Drift — And How Continuous Evaluation Keeps Them Stable”.

Modules included:
- metrics: instrumentation (intrinsic, extrinsic, behavioral)
- stages: multi-stage evaluation loop (Stage A/B/C)
- controller: automated evaluation controller
- feedback_loops: offline/online feedback utilities
- patterns: canary prompts, shadow evaluation, reasoning audits
- scenarios: synthetic drift examples for demos and notebooks
"""

from .metrics import (
    compute_intrinsic_metrics,
    compute_extrinsic_metrics,
    compute_behavioral_metrics,
    compute_all_metrics,
)

from .stages import (
    stage_a_pre_retrieval,
    stage_b_post_retrieval,
    stage_c_post_generation,
    evaluate_all_stages,
)

from .controller import EvaluationController
from .feedback_loops import OfflineFeedbackLoop, OnlineFeedbackLoop
from .patterns import (
    CanaryPromptRunner,
    ShadowEvaluationPipeline,
    ReasoningAudit,
)
from .scenarios import DriftScenario, load_scenarios

__all__ = [
    # metrics
    "compute_intrinsic_metrics",
    "compute_extrinsic_metrics",
    "compute_behavioral_metrics",
    "compute_all_metrics",
    # stages
    "stage_a_pre_retrieval",
    "stage_b_post_retrieval",
    "stage_c_post_generation",
    "evaluate_all_stages",
    # controller
    "EvaluationController",
    # feedback
    "OfflineFeedbackLoop",
    "OnlineFeedbackLoop",
    # patterns
    "CanaryPromptRunner",
    "ShadowEvaluationPipeline",
    "ReasoningAudit",
    # scenarios
    "DriftScenario",
    "load_scenarios",
]
