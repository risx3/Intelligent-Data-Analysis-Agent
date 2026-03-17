"""
FastAPI application — Level 2/4 service entry point.

Run:  uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analysis import router as analysis_router
from api.routes.query    import router as query_router
from api.schemas import HealthResponse
from utils.llm import check_ollama_health, list_available_models
from config import DEFAULT_MODEL

app = FastAPI(
    title="Intelligent Data Analysis Agent API",
    description="AI-powered EDA, insights and NL query over uploaded datasets.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)
app.include_router(query_router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return HealthResponse(
        status="ok",
        ollama_reachable=check_ollama_health(),
        available_models=list_available_models(),
    )


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "Intelligent Data Analysis Agent",
        "docs":    "/docs",
        "health":  "/health",
    }
