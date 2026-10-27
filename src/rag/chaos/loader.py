"""Scenario configuration loader from YAML files."""

from pathlib import Path

import yaml

from rag.chaos.faults import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load and validate chaos scenarios from YAML configuration.

    Args:
        path: Path to scenarios YAML file.

    Returns:
        List of validated Scenario models.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Scenarios file not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_scenarios = data.get("scenarios", []) if isinstance(data, dict) else data
    scenarios: list[Scenario] = []

    for item in raw_scenarios:
        scenarios.append(Scenario.model_validate(item))

    return scenarios
