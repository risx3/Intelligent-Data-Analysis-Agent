"""
FastAPI application — Level 2/4 service entry point.

Run:  uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes.analysis import router as analysis_router
from api.routes.query    import router as query_router
from api.schemas import HealthResponse
from utils.llm import check_ollama_health, list_available_models
from utils.logger import get_logger
from config import DEFAULT_MODEL, CORS_ORIGINS

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Intelligent Data Analysis Agent API",
    description="AI-powered EDA, insights and NL query over uploaded datasets.",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(analysis_router)
app.include_router(query_router)

logger.info("CORS allowed origins: %s", CORS_ORIGINS)


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
