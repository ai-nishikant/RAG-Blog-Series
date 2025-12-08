"""
Engineering Patterns:
- Canary prompts
- Shadow evaluation pipelines
- Reasoning audits
"""

from typing import List, Dict


class CanaryPromptRunner:
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def run(self, llm_fn) -> List[Dict]:
        """
        Runs LLM on diagnostic prompts.
        """
        results = []
        for p in self.prompts:
            output = llm_fn(p)
            results.append({"prompt": p, "output": output})
        return results


class ShadowEvaluationPipeline:
    """
    Parallel eval pipeline whose outputs are logged but not surfaced to the user.
    """

    def __init__(self):
        self.shadow_logs = []

    def run(self, query: str, system_fn):
        output = system_fn(query)
        self.shadow_logs.append({"query": query, "output": output})
        return output


class ReasoningAudit:
    """
    Simple reasoning audit that checks for verbosity and structure.
    """

    def audit(self, output: str) -> Dict:
        return {
            "length": len(output.split()),
            "has_numbered_steps": any(token.endswith(".") for token in output.split()),
        }
