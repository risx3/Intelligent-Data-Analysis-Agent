"""
Ollama LLM client.
Replaces the previous Anthropic client — no API key required,
just a running Ollama instance (ollama serve).
"""
import requests
from config import OLLAMA_BASE_URL, DEFAULT_MODEL, OLLAMA_TIMEOUT
from utils.logger import get_logger

logger = get_logger(__name__)


def call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single-turn call: system + user message."""
    return call_llm_with_history(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=model,
    )


def call_llm_with_history(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    """
    Multi-turn call supporting full conversation history.
    messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":   model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": 0.1,    # deterministic for analysis tasks
                "num_predict": 4096,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def check_ollama_health() -> bool:
    """Return True if Ollama is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        logger.warning("Ollama health check failed: connection refused at %s", OLLAMA_BASE_URL)
        return False
    except requests.Timeout:
        logger.warning("Ollama health check timed out at %s", OLLAMA_BASE_URL)
        return False
    except requests.RequestException as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return False


def list_available_models() -> list[str]:
    """Return names of models pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.ConnectionError:
        logger.warning("Could not list Ollama models: connection refused at %s", OLLAMA_BASE_URL)
        return []
    except requests.RequestException as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        return []
