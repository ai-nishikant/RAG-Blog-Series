#!/usr/bin/env python
"""
run_demo.py

Command-line entry point for the Continuous Evaluation demo.

This script:

1. Loads evaluation configuration (paths, feature flags)
2. Loads synthetic drift scenarios from YAML
3. For each scenario:
   - computes metrics (intrinsic, extrinsic, behavioral)
   - runs Stage A/B/C evaluation
   - asks the EvaluationController for a corrective action
   - records the results in offline and online feedback loops
4. Prints a human-readable summary
5. Optionally writes evaluation traces to JSONL logs

Usage (from repo root: 7_continuous_evaluation/):

    python scripts/run_demo.py

You can override paths via CLI flags or environment variables:

    EVAL_CONFIG=./config/evaluation_config.yaml \
    python scripts/run_demo.py --write-logs

"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from rag_eval.metrics import compute_all_metrics
from rag_eval.stages import evaluate_all_stages
from rag_eval.controller import EvaluationController
from rag_eval.feedback_loops import OfflineFeedbackLoop, OnlineFeedbackLoop
from rag_eval.scenarios import load_scenarios, DriftScenario


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, raising a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(
    config_path: Optional[str],
    scenarios_path: Optional[str],
    log_dir: Optional[str],
) -> Dict[str, Path]:
    """
    Resolve config, scenario, and log paths using:

    1. CLI flags (highest priority)
    2. Environment variables (EVAL_CONFIG, SCENARIO_CONFIG, EVAL_LOG_DIR)
    3. Defaults under the current repo
    """
    repo_root = Path.cwd()

    # Evaluation config
    cfg_path = (
        Path(config_path)
        if config_path
        else Path(os.environ.get("EVAL_CONFIG", repo_root / "config" / "evaluation_config.yaml"))
    )

    # Scenario config
    scen_path = (
        Path(scenarios_path)
        if scenarios_path
        else Path(os.environ.get("SCENARIO_CONFIG", repo_root / "config" / "demo_scenarios.yaml"))
    )

    # Log dir
    log_path = (
        Path(log_dir)
        if log_dir
        else Path(os.environ.get("EVAL_LOG_DIR", repo_root / "data" / "logs"))
    )

    return {
        "config": cfg_path,
        "scenarios": scen_path,
        "log_dir": log_path,
    }


def _timestamp() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def run_demo(
    config_path: Optional[str] = None,
    scenarios_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    write_logs: bool = False,
) -> Dict[str, Any]:
    """
    Run the full continuous evaluation demo over all configured scenarios.

    Parameters
    ----------
    config_path : Optional[str]
        Path to evaluation_config.yaml. If None, environment or defaults are used.
    scenarios_path : Optional[str]
        Path to demo_scenarios.yaml. If None, environment or defaults are used.
    log_dir : Optional[str]
        Directory to write JSONL logs. If None, environment or defaults are used.
    write_logs : bool
        If True, write JSONL traces to the log directory.

    Returns
    -------
    Dict[str, Any]
        Summary object with basic statistics.
    """
    paths = _resolve_paths(config_path, scenarios_path, log_dir)
    cfg_file = paths["config"]
    scen_file = paths["scenarios"]
    log_path = paths["log_dir"]

    print(f"[info] Using evaluation config: {cfg_file}")
    print(f"[info] Using scenario file:     {scen_file}")
    print(f"[info] Log directory:          {log_path}")

    # Load configuration (thresholds, paths, feature flags)
    try:
        eval_config = _load_yaml(cfg_file)
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    # Load scenarios
    try:
        scenarios: List[DriftScenario] = load_scenarios(str(scen_file))
    except FileNotFoundError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    if not scenarios:
        raise SystemExit("[error] No scenarios found in demo_scenarios.yaml")

    # Initialize controller and feedback loops
    controller = EvaluationController()
    offline = OfflineFeedbackLoop()
    online = OnlineFeedbackLoop()

    results: List[Dict[str, Any]] = []

    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario.name} ===")
        params = scenario.parameters

        query = params["query"]
        metadata = params.get("metadata", {})
        retrieved_docs = params.get("retrieved_docs", [])
        reference = params["reference"]
        output = params["output"]

        # Compute metrics (uses intrinsic overlap + simple extrinsic/behavioral)
        metrics_result = compute_all_metrics(
            reference=reference,
            output=output,
            latency_ms=metadata.get("latency_ms", 120.0),
            token_count=metadata.get("token_count", 80),
            retrieval_ms=metadata.get("retrieval_ms", 30.0),
        )

        # Evaluate all stages
        stage_results = evaluate_all_stages(
            query=query,
            metadata=metadata,
            retrieved_docs=retrieved_docs,
            gen_metrics={"intrinsic": metrics_result.intrinsic},
        )

        # Choose and execute action
        action_name = controller.choose_action(stage_results)
        correction = controller.execute(action_name)

        record = {
            "timestamp": _timestamp(),
            "scenario": scenario.name,
            "description": scenario.description,
            "intrinsic": metrics_result.intrinsic,
            "stage_a_passed": stage_results["stage_a"]["passed"],
            "stage_b_passed": stage_results["stage_b"]["passed"],
            "stage_c_passed": stage_results["stage_c"]["passed"],
            "overall_passed": stage_results["overall_passed"],
            "action": action_name,
            "correction": correction,
        }

        # Add to feedback loops
        offline.record(record)
        if not stage_results["overall_passed"]:
            online.report_issue(
                {
                    "scenario": scenario.name,
                    "action": action_name,
                    "timestamp": record["timestamp"],
                }
            )

        results.append(record)

        # Print a brief human-readable summary
        print(f"[result] intrinsic={record['intrinsic']:.3f} "
              f"overall_passed={record['overall_passed']} action={record['action']}")

    # Summarize offline and online perspectives
    offline_summary = offline.summarize()
    has_drift = online.has_drift()
    issues = online.live_issues

    print("\n=== Summary ===")
    print(f"Total scenarios:         {offline_summary['runs']}")
    print(f"Total failures:          {offline_summary['failures']}")
    print(f"Drift detected (online): {has_drift}")
    if issues:
        print("Drift issues (first few):")
        for issue in issues[:5]:
            print(f"  - scenario={issue['scenario']} action={issue['action']}")

    # Optionally write logs
    if write_logs:
        log_path.mkdir(parents=True, exist_ok=True)
        eval_log_file = log_path / "sample_evaluation_runs.jsonl"
        print(f"\n[info] Writing evaluation logs to: {eval_log_file}")
        with eval_log_file.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

    return {
        "offline_summary": offline_summary,
        "has_drift": has_drift,
        "issues": issues,
        "results": results,
        "config_used": str(cfg_file),
        "scenarios_used": str(scen_file),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Continuous Evaluation demo over synthetic drift scenarios.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation_config.yaml (overrides EVAL_CONFIG/env/default).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Path to demo_scenarios.yaml (overrides SCENARIO_CONFIG/env/default).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for JSONL logs (overrides EVAL_LOG_DIR/env/default).",
    )
    parser.add_argument(
        "--write-logs",
        action="store_true",
        help="If set, write evaluation traces to data/logs/sample_evaluation_runs.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entry point. Wraps run_demo with argument parsing and basic error handling.
    """
    args = parse_args(argv)
    try:
        run_demo(
            config_path=args.config,
            scenarios_path=args.scenarios,
            log_dir=args.log_dir,
            write_logs=args.write_logs,
        )
    except SystemExit as exc:
        # Re-raise system exit (for clean CLI error messages)
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        # Catch-all to avoid ugly stack traces for simple usage errors.
        print(f"[error] Unexpected error during demo: {exc}")
        raise


if __name__ == "__main__":
    main()
