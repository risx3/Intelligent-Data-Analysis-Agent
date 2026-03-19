import json
import re
from utils.logger import get_logger

logger = get_logger(__name__)


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
            logger.debug("Failed to parse JSON from markdown code block")

    # Try extracting a JSON object — non-greedy to avoid merging multiple objects
    match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.debug("Failed to parse extracted JSON object from text")

    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


def safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
