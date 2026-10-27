"""Defensive JSON parser extracting clean data structures from noisy LLM outputs."""

import json
import re
from typing import Any

from rag.errors import MalformedOutputError


def extract_balanced_json_string(text: str) -> str:
    """Find the first substring with balanced braces or brackets."""
    first_curly = text.find("{")
    first_square = text.find("[")

    if first_curly == -1 and first_square == -1:
        raise MalformedOutputError(f"No JSON object or array found in text: {text[:100]}...")

    start_idx = (
        first_curly
        if (first_curly != -1 and (first_square == -1 or first_curly < first_square))
        else first_square
    )
    open_char = text[start_idx]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1]

        escape = bool(char == "\\" and not escape)

    raise MalformedOutputError(f"Unbalanced JSON string starting at position {start_idx}")


def parse_defensive_json(text: str) -> dict[str, Any]:
    """Parse JSON defensively from LLM response text.

    Strips Markdown code fences and extracts balanced JSON objects from conversational text.

    Args:
        text: Raw output string from LLM.

    Returns:
        Parsed dictionary.

    Raises:
        MalformedOutputError: If JSON structure cannot be extracted or parsed.
    """
    cleaned = text.strip()

    # Strip code block fences if present
    fence_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    fence_match = re.search(fence_pattern, cleaned, re.MULTILINE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise MalformedOutputError(f"Parsed JSON is not an object/dictionary: {type(parsed)}")
    except json.JSONDecodeError:
        pass

    # Extract balanced JSON object
    try:
        substring = extract_balanced_json_string(cleaned)
        parsed = json.loads(substring)
        if isinstance(parsed, dict):
            return parsed
        raise MalformedOutputError(f"Extracted JSON is not a dictionary: {type(parsed)}")
    except (json.JSONDecodeError, MalformedOutputError) as e:
        raise MalformedOutputError(f"Failed to parse balanced JSON from text: {e}") from e
