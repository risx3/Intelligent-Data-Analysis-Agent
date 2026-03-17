from pydantic import BaseModel, Field
from typing import Any


class AnalyseRequest(BaseModel):
    query: str = Field("", description="Optional natural-language analysis goal")
    model: str = Field("qwen2.5:3b", description="Ollama model to use")


class SessionStatus(BaseModel):
    session_id: str
    status: str           # "queued" | "running" | "done" | "failed"
    progress: list[dict]  # execution log so far
    error: str = ""


class AnalysisResult(BaseModel):
    session_id: str
    report_path: str | None
    plots: list[str]
    insights: str
    execution_log: list[dict]


class QueryRequest(BaseModel):
    question: str
    model: str = "qwen2.5:3b"


class QueryResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    history_length: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    available_models: list[str]
