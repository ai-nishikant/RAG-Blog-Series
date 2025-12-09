"""
Automated Evaluation Controller

Takes stage outputs (A/B/C) and decides what corrective action to apply.
"""

from typing import Dict


class EvaluationController:
    """
    Basic controller implementing Step 3 from the blog.
    
    This controller acts as the decision-making component in the continuous
    evaluation loop. It examines the results from three evaluation stages
    (A, B, and C) and determines the appropriate corrective action to take
    when quality issues are detected.
    """

    def __init__(self):
        """
        Initialize the controller with a mapping of action names to handler functions.
        
        The actions dictionary provides a registry of all possible corrective actions
        that can be taken based on evaluation results. Each action is mapped to a
        private method that implements the correction logic.
        """
        # Actions can be swapped for real modules later.
        self.actions = {
            "fix_query": self._fix_query,              # Stage A failure handler
            "adjust_retrieval": self._adjust_retrieval, # Stage B failure handler
            "adjust_prompt": self._adjust_prompt,       # Stage C failure handler
            "noop": lambda: None,                       # No action needed (all stages passed)
        }

    def choose_action(self, stage_results: Dict[str, Dict]) -> str:
        """
        Maps Stage A/B/C failures → corrective actions.
        
        Implements a priority-based decision tree:
        1. Stage A (query quality) is checked first - if it fails, fix the query
        2. Stage B (retrieval quality) is checked next - if it fails, adjust retrieval
        3. Stage C (generation quality) is checked last - if it fails, adjust prompt
        4. If all stages pass, return "noop" (no operation needed)
        
        Args:
            stage_results: Dictionary containing results from each evaluation stage.
                          Expected structure:
                          {
                              "stage_a": {"passed": bool, ...},
                              "stage_b": {"passed": bool, ...},
                              "stage_c": {"passed": bool, ...}
                          }
        
        Returns:
            str: The name of the action to execute ("fix_query", "adjust_retrieval",
                 "adjust_prompt", or "noop")
        """
        # Check Stage A: Query Clarity/Intent Detection
        # If the user query is ambiguous or unclear, fix it before proceeding
        if not stage_results["stage_a"]["passed"]:
            return "fix_query"

        # Check Stage B: Retrieval Quality
        # If relevant documents weren't retrieved, adjust the retrieval strategy
        if not stage_results["stage_b"]["passed"]:
            return "adjust_retrieval"

        # Check Stage C: Generation Quality
        # If the final response is poor quality, adjust the prompt engineering
        if not stage_results["stage_c"]["passed"]:
            return "adjust_prompt"

        # All stages passed - no corrective action needed
        return "noop"

    def execute(self, action_name: str):
        """
        Execute the actual correction function.
        
        Looks up the action in the actions registry and invokes it.
        
        Args:
            action_name: Name of the action to execute (must exist in self.actions)
        
        Returns:
            The result of the action function (typically a dict with action details)
        """
        return self.actions[action_name]()

    # --------------------
    # Action definitions
    # --------------------
    def _fix_query(self):
        """
        Corrective action for Stage A (query quality) failures.
        
        Triggered when the user query is ambiguous, unclear, or poorly formed.
        In a production system, this might:
        - Rewrite the query using an LLM
        - Add clarifying context from conversation history
        - Expand abbreviations or resolve pronouns
        - Extract structured intent from natural language
        
        Returns:
            dict: Description of the action taken
        """
        return {"action": "fix_query", "detail": "Rewriting or enriching query."}

    def _adjust_retrieval(self):
        """
        Corrective action for Stage B (retrieval quality) failures.
        
        Triggered when the retrieved documents are not relevant to the query.
        In a production system, this might:
        - Increase K (number of retrieved documents)
        - Adjust similarity thresholds
        - Expand metadata filters
        - Use query expansion or reformulation
        - Switch to a different retrieval strategy (e.g., hybrid search)
        
        Returns:
            dict: Description of the action taken
        """
        return {"action": "adjust_retrieval", "detail": "Increasing K, expanding filters."}

    def _adjust_prompt(self):
        """
        Corrective action for Stage C (generation quality) failures.
        
        Triggered when the LLM's final response is poor quality despite having
        good query understanding and relevant documents retrieved.
        In a production system, this might:
        - Switch from zero-shot to few-shot prompting
        - Add more specific instructions or constraints
        - Change the prompt structure or format
        - Adjust temperature or other generation parameters
        - Add chain-of-thought reasoning steps
        
        Returns:
            dict: Description of the action taken
        """
        return {"action": "adjust_prompt", "detail": "Switching to few-shot or applying structure."}
