import json
import re


def extract_json(text: str) -> dict:
    """Extract the first JSON object or array from an LLM response string."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting any JSON object in the text
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM response:\n{text}")


def safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
