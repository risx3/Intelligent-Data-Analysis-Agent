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

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins. Use "*" only for local development.
_cors_env   = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]

# ── API input limits ──────────────────────────────────────────────────────────
MAX_QUERY_LENGTH    = int(os.getenv("MAX_QUERY_LENGTH", "5000"))   # characters
MAX_UPLOAD_SIZE_MB  = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))  # megabytes

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_ANALYSE = os.getenv("RATE_LIMIT_ANALYSE", "10/minute")  # slowapi format
RATE_LIMIT_QUERY   = os.getenv("RATE_LIMIT_QUERY",   "30/minute")

# ── Visualization constants ───────────────────────────────────────────────────
VIZ_MAX_HIST_COLS   = 6   # max numeric columns in distribution subplot grid
VIZ_MAX_BOX_COLS    = 8   # max columns in box-plot panel
VIZ_MAX_BAR_ROWS    = 15  # max top-N rows in bar/count charts
VIZ_MAX_PIE_SLICES  = 8   # max slices in pie chart
VIZ_TOP_CAT_COLS    = 3   # max categorical columns to produce bar charts for
VIZ_TOP_VALUES      = 10  # top-N values shown in categorical bar chart

# ── Memory / conversation ─────────────────────────────────────────────────────
MEMORY_HISTORY_WINDOW = 10  # number of prior conversation turns sent to LLM
