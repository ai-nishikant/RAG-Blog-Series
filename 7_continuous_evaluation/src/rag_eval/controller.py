"""
Automated Evaluation Controller

Takes stage outputs (A/B/C) and decides what corrective action to apply.
"""

from typing import Dict


class EvaluationController:
    """
    Basic controller implementing Step 3 from the blog.
    """

    def __init__(self):
        # Actions can be swapped for real modules later.
        self.actions = {
            "fix_query": self._fix_query,
            "adjust_retrieval": self._adjust_retrieval,
            "adjust_prompt": self._adjust_prompt,
            "noop": lambda: None,
        }

    def choose_action(self, stage_results: Dict[str, Dict]) -> str:
        """
        Maps Stage A/B/C failures → corrective actions.
        """

        if not stage_results["stage_a"]["passed"]:
            return "fix_query"

        if not stage_results["stage_b"]["passed"]:
            return "adjust_retrieval"

        if not stage_results["stage_c"]["passed"]:
            return "adjust_prompt"

        return "noop"

    def execute(self, action_name: str):
        """
        Execute the actual correction function.
        """

        return self.actions[action_name]()

    # --------------------
    # Action definitions
    # --------------------
    def _fix_query(self):
        return {"action": "fix_query", "detail": "Rewriting or enriching query."}

    def _adjust_retrieval(self):
        return {"action": "adjust_retrieval", "detail": "Increasing K, expanding filters."}

    def _adjust_prompt(self):
        return {"action": "adjust_prompt", "detail": "Switching to few-shot or applying structure."}
