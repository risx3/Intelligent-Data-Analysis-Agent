# Intelligent Data Analysis Agent

An AI-powered multi-agent system that takes a dataset, plans analysis autonomously, runs EDA in parallel, generates self-critiqued insights, creates visualisations, and produces a full report — all running locally via **Ollama** (no API key needed).

```
User Input (CSV / Question)
        ↓
MultiAgent Orchestrator
        ├─ PlannerAgent      → LLM creates tailored analysis plan
        ├─ EDAAgent          → parallel stats + quality analysis
        ├─ VizAgent          → standard + NL query-driven plots
        ├─ InsightAgent      → business insights generation
        ├─ CriticAgent       → reflection loop (self-critique & retry)
        └─ ReportGenerator   → full Markdown report
```

---

## Features

| Level | Feature |
|-------|---------|
| **Core** | Smart planning, full EDA, auto visualisations, Markdown report |
| **L2** | NL query-driven dynamic plots, FastAPI upload endpoint, query-aware column selection |
| **L3** | Multi-turn conversation memory, agent reflection loop, multi-agent specialist system |
| **L4** | Async parallel execution, Streamlit dashboard, Docker microservices |

---

## Quick Start

### 1. Start Ollama + pull a model

```bash
ollama serve
ollama pull qwen2.5:3b   # recommended: fast + excellent JSON output
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run via CLI

```bash
# Basic analysis
python app.py data/your_dataset.csv

# With a natural-language goal
python app.py data/sales.csv --query "What drives revenue?"

# Different model
python app.py data/data.csv --model llama3.2:3b --query "Find anomalies"

# List available models
python app.py data/x.csv --list-models
```

### 4. Run as API + UI (recommended)

```bash
# Terminal 1 — API service
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. Run everything with Docker

```bash
cd docker
docker compose up --build
```

Services: Ollama (11434) · API (8000) · UI (8501)

---

## Project Structure

```
├── app.py                      # CLI entry point
├── config.py                   # Ollama URL, model, paths, thresholds
│
├── agents/
│   ├── orchestrator.py         # Multi-agent coordinator (main brain)
│   ├── planner.py              # LLM plan creator (query-aware)
│   ├── eda_agent.py            # Specialist: parallel EDA
│   ├── viz_agent.py            # Specialist: NL-driven visualisations
│   ├── insight_agent.py        # Specialist: business insights (+ reflection)
│   └── critic_agent.py         # Reflection loop: score + critique
│
├── tools/
│   ├── data_loader.py          # CSV / Excel / JSON / Parquet loader
│   ├── eda_tools.py            # describe, missing values, stats, correlation
│   └── viz_tools.py            # standard EDA plots + dynamic plot renderer
│
├── engine/
│   ├── async_executor.py       # Async wrapper around orchestrator
│   ├── executor.py             # Sync shim (backward compat)
│   └── report_generator.py     # Markdown report assembler
│
├── memory/
│   ├── session_store.py        # File-backed session persistence (JSON)
│   └── memory_manager.py       # Multi-turn NL query with history
│
├── api/
│   ├── main.py                 # FastAPI app + CORS
│   ├── schemas.py              # Pydantic request/response models
│   └── routes/
│       ├── analysis.py         # POST /analyse, GET /status, GET /result
│       └── query.py            # POST /sessions/{id}/query (NL chat)
│
├── ui/
│   └── streamlit_app.py        # Full dashboard: upload → results → chat
│
├── docker/
│   ├── docker-compose.yml      # Ollama + API + UI services
│   ├── Dockerfile.api
│   └── Dockerfile.ui
│
├── outputs/
│   ├── reports/                # Generated Markdown reports
│   ├── plots/                  # PNG visualisations
│   ├── sessions/               # Session JSON files (memory)
│   └── uploads/                # API-uploaded datasets
│
└── requirements.txt
```

---

## Analysis Pipeline

```
load_data  →  create_plan
                    ├── EDAAgent  ──────────────── ┐  (asyncio.gather)
                    │   ├── describe_data           │
                    │   ├── check_missing_values    │
                    │   ├── run_basic_stats         │
                    │   └── correlation_matrix      │
                    │                               │
                    └── VizAgent ──────────────── ──┘
                        ├── standard EDA plots
                        └── NL query-driven plots
                                    ↓
                           InsightAgent (v1)
                                    ↓
                           CriticAgent → score/10
                            score < 7? → InsightAgent (v2) → CriticAgent → ...
                                    ↓
                           ReportGenerator → report.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyse` | Upload file + start analysis (multipart) |
| `GET`  | `/analyse/{id}/status` | Poll status (queued/running/done/failed) |
| `GET`  | `/analyse/{id}/result` | Get full results when done |
| `GET`  | `/analyse/{id}/report` | Download Markdown report |
| `POST` | `/sessions/{id}/query` | NL follow-up question (multi-turn) |
| `GET`  | `/sessions/{id}/history` | Conversation history |
| `GET`  | `/health` | Ollama health + available models |

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Supported File Formats

CSV · Excel (`.xls`, `.xlsx`) · JSON · Parquet

---

## Configuration

Edit [config.py](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `qwen2.5:3b` | LLM model for all agents |
| `REFLECTION_ENABLED` | `True` | Enable insight self-critique loop |
| `REFLECTION_MIN_SCORE` | `7` | Minimum score to accept insights |
| `REFLECTION_MAX_RETRIES` | `2` | Max critique-and-revise cycles |
| `CORRELATION_THRESHOLD` | `0.7` | High-correlation flag threshold |
| `MISSING_VALUE_THRESHOLD` | `0.3` | High-severity missing value threshold |

Recommended Ollama models (fast, good JSON):

| Model | Size | Best for |
|-------|------|---------|
| `qwen2.5:3b` | 2 GB | Default — fast + excellent structured output |
| `llama3.2:3b` | 2 GB | Good general reasoning |
| `phi3:mini` | 2.3 GB | Very fast, solid instructions |
| `mistral:7b` | 4 GB | Higher quality, slower |

---

## Roadmap — All Completed ✅

### Level 2 — Natural Language Interface
- [x] NL query-driven dynamic plotting
- [x] FastAPI CSV upload endpoint
- [x] Query-aware column selection

### Level 3 — Advanced Agent
- [x] Conversation memory (multi-turn analysis)
- [x] Agent reflection loop (self-critique & retry)
- [x] Multi-agent extension (EDA, Viz, Insight, Critic specialists)

### Level 4 — Production Architecture
- [x] Async step execution (asyncio.gather for parallel EDA)
- [x] Interactive UI dashboard (Streamlit)
- [x] Microservices decomposition (Docker Compose: Ollama + API + UI)
