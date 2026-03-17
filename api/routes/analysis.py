"""
Analysis routes — file upload, async analysis, status polling.
"""
import os
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from api.schemas import SessionStatus, AnalysisResult
from memory.session_store import SessionStore
from config import UPLOADS_DIR, DEFAULT_MODEL

router = APIRouter(prefix="/analyse", tags=["analysis"])


# ── background task ────────────────────────────────────────────────────────────

async def _run_analysis(session_id: str, filepath: str, query: str, model: str):
    store = SessionStore(session_id)
    store.set_status("running")
    try:
        from engine.async_executor import AsyncAnalysisExecutor
        executor = AsyncAnalysisExecutor(
            filepath=filepath,
            user_query=query,
            session_id=session_id,
            model=model,
        )
        result = await executor.run()
        store.update_result(result)
        store.set_status("done")
    except Exception as exc:
        store.set_status("failed", error=str(exc))


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("", summary="Upload dataset and start analysis")
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str = Form(""),
    model: str = Form(DEFAULT_MODEL),
):
    session_id = str(uuid.uuid4())

    # Save uploaded file
    ext = os.path.splitext(file.filename or "data.csv")[1] or ".csv"
    save_path = os.path.join(UPLOADS_DIR, f"{session_id}{ext}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Create session
    store = SessionStore(session_id)
    store.create(dataset_path=save_path, query=query, model=model)

    # Schedule analysis in background
    background_tasks.add_task(_run_analysis, session_id, save_path, query, model)

    return {"session_id": session_id, "status": "queued"}


@router.get("/{session_id}/status", response_model=SessionStatus)
async def get_status(session_id: str):
    store = SessionStore(session_id)
    data = store.load()
    if data is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return SessionStatus(
        session_id=session_id,
        status=data.get("status", "unknown"),
        progress=data.get("execution_log", []),
        error=data.get("error", ""),
    )


@router.get("/{session_id}/result", response_model=AnalysisResult)
async def get_result(session_id: str):
    store = SessionStore(session_id)
    data = store.load()
    if data is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if data.get("status") != "done":
        raise HTTPException(400, f"Analysis not complete (status: {data.get('status')})")
    return AnalysisResult(
        session_id=session_id,
        report_path=data.get("report"),
        plots=data.get("plots", []),
        insights=data.get("insights", ""),
        execution_log=data.get("execution_log", []),
    )


@router.get("/{session_id}/report", summary="Download the Markdown report")
async def download_report(session_id: str):
    store = SessionStore(session_id)
    data = store.load()
    if data is None:
        raise HTTPException(404, "Session not found")
    path = data.get("report")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Report not generated yet")
    return FileResponse(path, media_type="text/markdown", filename=os.path.basename(path))
