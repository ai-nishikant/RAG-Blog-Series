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
    
    This class maintains a historical record of all evaluation runs performed
    during development and testing. It acts as a regression testing mechanism
    to ensure that changes to the RAG pipeline don't degrade performance on
    previously successful queries.
    """

    def __init__(self):
        """
        Initialize the offline feedback loop with an empty history.
        
        The history list stores all evaluation runs, allowing developers to
        track performance over time and detect regressions before deployment.
        """
        # List of evaluation dictionaries, each containing metrics and results from a run
        self.history: List[Dict] = []

    def record(self, evaluation: Dict):
        """
        Record a single evaluation run for future analysis.
        
        Each evaluation typically contains:
        - Timestamp of the run
        - Query and response pairs tested
        - Stage-by-stage metrics (A/B/C)
        - Overall pass/fail status
        - Any corrective actions taken
        
        Args:
            evaluation: Dictionary containing complete evaluation results
        """
        # Append the evaluation to our historical record
        self.history.append(evaluation)

    def summarize(self):
        """
        Generate a summary of all recorded evaluation runs.
        
        Provides high-level statistics about the evaluation history, useful for:
        - Understanding overall system stability
        - Identifying trends in failure rates
        - Validating that changes improve (or at least don't harm) performance
        
        Returns:
            dict: Summary statistics containing:
                - runs: Total number of evaluation runs recorded
                - failures: Number of runs that failed overall evaluation
        """
        return {
            "runs": len(self.history),  # Total evaluation runs performed
            # Count evaluations where overall_passed is False
            "failures": sum(not run["overall_passed"] for run in self.history),
        }


class OnlineFeedbackLoop:
    """
    Simulates real-time monitoring of drift signals.
    
    This class represents production monitoring that detects when the RAG system's
    performance degrades in real-world usage. It captures signals like:
    - User dissatisfaction (thumbs down, low ratings)
    - Increased latency or errors
    - Semantic drift in queries or document corpus
    - Changes in user behavior patterns
    """

    def __init__(self):
        """
        Initialize the online feedback loop with an empty issue tracker.
        
        In production, this would connect to monitoring systems, logging
        infrastructure, and user feedback channels.
        """
        # List of issues detected in production (e.g., user complaints, quality drops)
        self.live_issues: List[Dict] = []

    def report_issue(self, issue: Dict):
        """
        Report a new issue detected in production.
        
        Issues can come from various sources:
        - User feedback (ratings, complaints)
        - Automated quality checks on live traffic
        - Anomaly detection in metrics
        - A/B test results showing degradation
        
        Args:
            issue: Dictionary describing the issue, typically containing:
                   - issue_type: Category of problem (e.g., "low_rating", "timeout")
                   - timestamp: When the issue occurred
                   - query: The user query that triggered the issue
                   - details: Additional context about the problem
        """
        # Add the issue to our tracking list for analysis
        self.live_issues.append(issue)

    def has_drift(self) -> bool:
        """
        Check if any drift or quality issues have been detected.
        
        Drift detection indicates that the RAG system's performance has degraded
        in production, potentially requiring:
        - Re-evaluation of the pipeline
        - Model retraining or updates
        - Prompt adjustments
        - Document corpus refresh
        
        Returns:
            bool: True if any issues have been reported, False otherwise
        """
        # Simple check: any reported issues indicate potential drift
        # In production, this might use more sophisticated thresholds or patterns
        return len(self.live_issues) > 0
