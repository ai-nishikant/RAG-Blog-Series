"""
Engineering Patterns:
- Canary prompts
- Shadow evaluation pipelines
- Reasoning audits
"""

from typing import List, Dict


class CanaryPromptRunner:
    """
    Runs diagnostic "canary" prompts to detect model degradation or drift.
    
    Canary prompts are carefully crafted test queries with known expected outputs.
    They act as an early warning system - if these well-understood prompts start
    producing incorrect or degraded outputs, it signals a problem with the model,
    prompt engineering, or retrieval system.
    
    Similar to canaries in coal mines that detected dangerous gases, these prompts
    detect problems before they affect real users.
    """
    
    def __init__(self, prompts: List[str]):
        """
        Initialize with a set of diagnostic prompts.
        
        Args:
            prompts: List of canary prompt strings. These should be prompts with
                    well-known, stable expected outputs that can be used to detect
                    system degradation over time.
        """
        # Store the diagnostic prompts to be run periodically
        self.prompts = prompts

    def run(self, llm_fn) -> List[Dict]:
        """
        Runs LLM on diagnostic prompts.
        
        Execute all canary prompts through the provided LLM function and collect
        the results. These results can then be compared against expected outputs
        or historical baselines to detect drift or degradation.
        
        Typical usage:
        - Run daily or after each deployment
        - Compare outputs to known good responses
        - Alert if outputs deviate significantly
        
        Args:
            llm_fn: Callable that takes a prompt string and returns the LLM output
        
        Returns:
            List of dictionaries containing each prompt and its corresponding output
        """
        results = []
        # Execute each canary prompt through the LLM
        for p in self.prompts:
            output = llm_fn(p)  # Generate response using the provided LLM function
            # Store both prompt and output for comparison/analysis
            results.append({"prompt": p, "output": output})
        return results


class ShadowEvaluationPipeline:
    """
    Parallel eval pipeline whose outputs are logged but not surfaced to the user.
    
    Shadow evaluation runs an alternative version of the RAG pipeline in parallel
    with the production system. The shadow version processes real user queries but:
    - Its outputs are NOT shown to users
    - Results are logged for offline analysis
    - Used to safely test new models, prompts, or retrieval strategies
    
    This pattern enables A/B testing and gradual rollout of changes without
    risking user experience.
    """

    def __init__(self):
        """
        Initialize the shadow pipeline with empty logs.
        
        The shadow_logs will accumulate all queries and outputs processed by
        the shadow system for later analysis and comparison with production.
        """
        # Storage for all shadow evaluation results (query + output pairs)
        self.shadow_logs = []

    def run(self, query: str, system_fn):
        """
        Run the shadow system on a query and log the results.
        
        This method processes a query through the shadow pipeline (system_fn),
        logs the result for analysis, but returns the output as if it were from
        the production system. In practice, you'd typically discard the shadow
        output and only use logs for evaluation.
        
        Workflow:
        1. User query comes in
        2. Production system handles it (user sees this)
        3. Shadow system also processes it (logged only)
        4. Compare shadow vs production offline
        
        Args:
            query: User's query string
            system_fn: The shadow RAG pipeline function to evaluate
        
        Returns:
            The output from the shadow system (in practice, you'd return
            production output instead)
        """
        # Execute the shadow pipeline on the user query
        output = system_fn(query)
        
        # Log the query-output pair for offline evaluation
        # This data can be analyzed to compare shadow vs production performance
        self.shadow_logs.append({"query": query, "output": output})
        
        # Return the output (in real usage, return production output instead)
        return output


class ReasoningAudit:
    """
    Simple reasoning audit that checks for verbosity and structure.
    
    Audits the model's reasoning process by analyzing structural and stylistic
    properties of the output. This helps detect issues like:
    - Overly verbose or terse responses
    - Lack of step-by-step reasoning
    - Missing citations or structure
    - Hedging or uncertainty patterns
    
    Can be extended to check for chain-of-thought, citations, confidence
    indicators, and other reasoning quality signals.
    """

    def audit(self, output: str) -> Dict:
        """
        Perform a structural audit of the model's output.
        
        Analyzes the output to extract signals about reasoning quality and
        structure. Current checks are basic but demonstrate the pattern:
        - Length: Detects overly verbose or terse responses
        - Numbered steps: Indicates structured, step-by-step reasoning
        
        Could be extended to detect:
        - Presence of citations (e.g., "[1]", "[Source: ...]")
        - Hedging language ("maybe", "possibly", "might")
        - Confidence indicators
        - Logical connectors ("therefore", "because", "however")
        
        Args:
            output: The generated response to audit
        
        Returns:
            Dictionary containing audit metrics:
            - length: Number of tokens in the output
            - has_numbered_steps: Whether output contains numbered steps (rough proxy)
        """
        return {
            # Token count - helps identify verbose or terse responses
            "length": len(output.split()),
            
            # Simple heuristic: presence of tokens ending with "." might indicate
            # numbered steps like "1.", "2.", etc. (not perfect but demonstrates concept)
            # A better check would use regex like r'\d+\.' or r'Step \d+'
            "has_numbered_steps": any(token.endswith(".") for token in output.split()),
        }
