"""
Synthetic drift scenarios for demos and notebooks.

Used in Step 4 and Step 5 to simulate:
- retrieval drift
- prompt drift
- grounding drift
"""

import yaml
from typing import Dict, List


class DriftScenario:
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters


def load_scenarios(path: str) -> List[DriftScenario]:
    """
    Load scenarios from YAML file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    scenarios = []
    for item in data.get("scenarios", []):
        scenarios.append(
            DriftScenario(
                name=item["name"],
                description=item["description"],
                parameters=item.get("parameters", {}),
            )
        )
    return scenarios
