"""
Synthetic drift scenarios for demos and notebooks.

Used in Step 4 and Step 5 to simulate:
- retrieval drift (documents become stale or irrelevant)
- prompt drift (prompt engineering changes affect outputs)
- grounding drift (model produces ungrounded or hallucinated content)

These synthetic scenarios help test and demonstrate how the continuous evaluation
system responds to various types of quality degradation without needing to wait
for real production issues.
"""

import yaml
from typing import Dict, List


class DriftScenario:
    """
    Represents a single drift scenario configuration.
    
    Each scenario defines a specific type of quality degradation or system change
    that the continuous evaluation system should be able to detect and respond to.
    This is useful for:
    - Testing the robustness of evaluation logic
    - Demonstrating evaluation capabilities in notebooks/demos
    - Regression testing before deployment
    """
    
    def __init__(self, name: str, description: str, parameters: Dict):
        """
        Initialize a drift scenario with metadata and configuration.
        
        Args:
            name: Short identifier for the scenario (e.g., "retrieval_drift_old_docs")
            description: Human-readable explanation of what this scenario simulates
            parameters: Configuration dict controlling how the drift is simulated,
                       such as:
                       - k_reduced: Lower K value for retrieval drift
                       - noise_level: Amount of irrelevant content to inject
                       - prompt_version: Which prompt variant to use
                       - confidence_threshold: Modified threshold for filtering
        """
        self.name = name                    # Scenario identifier
        self.description = description      # What drift/issue this simulates
        self.parameters = parameters        # Knobs to control the scenario


def load_scenarios(path: str) -> List[DriftScenario]:
    """
    Load drift scenarios from a YAML configuration file.
    
    Parses a YAML file containing multiple scenario definitions and creates
    DriftScenario objects for each one. The YAML file should have structure:
    
    scenarios:
      - name: "scenario_name"
        description: "what it simulates"
        parameters:
          key1: value1
          key2: value2
    
    This allows keeping scenario configurations separate from code, making it
    easy to add new test cases or modify existing ones without code changes.
    
    Args:
        path: File path to the YAML configuration file
    
    Returns:
        List of DriftScenario objects parsed from the file
    """
    # Open and parse the YAML file safely
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Build list of scenario objects from parsed data
    scenarios = []
    # Iterate through each scenario definition in the "scenarios" key
    for item in data.get("scenarios", []):
        # Create a DriftScenario object from the YAML entry
        scenarios.append(
            DriftScenario(
                name=item["name"],                          # Required: scenario name
                description=item["description"],            # Required: what it does
                parameters=item.get("parameters", {}),      # Optional: config params
            )
        )
    return scenarios
