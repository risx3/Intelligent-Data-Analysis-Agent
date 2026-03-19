from pydantic import BaseModel, Field, field_validator
from typing import Any
from config import DEFAULT_MODEL, MAX_QUERY_LENGTH


class AnalyseRequest(BaseModel):
    query: str = Field("", description="Optional natural-language analysis goal")
    model: str = Field(DEFAULT_MODEL, description="Ollama model to use")

    @field_validator("query")
    @classmethod
    def query_length(cls, v: str) -> str:
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")
        return v


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
    question: str = Field(..., description="Follow-up question about the analysis")
    model: str = Field(DEFAULT_MODEL, description="Ollama model to use")

    @field_validator("question")
    @classmethod
    def question_length(cls, v: str) -> str:
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"question must be at most {MAX_QUERY_LENGTH} characters")
        return v


class QueryResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    history_length: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    available_models: list[str]
