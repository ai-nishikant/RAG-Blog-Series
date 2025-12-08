"""
Offline and Online Feedback Loops

Implements Step 4 from the blog:
- offline loops validate changes before deployment
- online loops monitor production signals
"""

from typing import Dict, List


class OfflineFeedbackLoop:
    """
    Stores evaluation runs for regression testing.
    """

    def __init__(self):
        self.history: List[Dict] = []

    def record(self, evaluation: Dict):
        self.history.append(evaluation)

    def summarize(self):
        return {
            "runs": len(self.history),
            "failures": sum(not run["overall_passed"] for run in self.history),
        }


class OnlineFeedbackLoop:
    """
    Simulates real-time monitoring of drift signals.
    """

    def __init__(self):
        self.live_issues: List[Dict] = []

    def report_issue(self, issue: Dict):
        self.live_issues.append(issue)

    def has_drift(self) -> bool:
        return len(self.live_issues) > 0
