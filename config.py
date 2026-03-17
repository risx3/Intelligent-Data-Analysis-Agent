import os
from dotenv import load_dotenv

load_dotenv()

# ── Ollama LLM ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL", "qwen2.5:3b")   # fast + excellent JSON
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "120"))     # seconds per request

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR  = os.path.join(OUTPUT_DIR, "reports")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
SESSIONS_DIR = os.path.join(OUTPUT_DIR, "sessions")
UPLOADS_DIR  = os.path.join(OUTPUT_DIR, "uploads")

for _d in [REPORTS_DIR, PLOTS_DIR, SESSIONS_DIR, UPLOADS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Analysis thresholds ───────────────────────────────────────────────────────
MAX_CATEGORICAL_UNIQUE   = 20
CORRELATION_THRESHOLD    = 0.7
MISSING_VALUE_THRESHOLD  = 0.3

# ── Reflection (Level 3) ──────────────────────────────────────────────────────
REFLECTION_ENABLED     = True
REFLECTION_MIN_SCORE   = 7    # accept insights if score >= this (out of 10)
REFLECTION_MAX_RETRIES = 2

# ── API server (Level 4) ──────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT  = int(os.getenv("UI_PORT",  "8501"))
