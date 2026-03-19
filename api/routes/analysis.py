"""
Analysis routes — file upload, async analysis, status polling.
"""
import os
import re
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import SessionStatus, AnalysisResult
from memory.session_store import SessionStore
from utils.logger import get_logger
from config import UPLOADS_DIR, DEFAULT_MODEL, MAX_UPLOAD_SIZE_MB, RATE_LIMIT_ANALYSE

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/analyse", tags=["analysis"])

_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
_MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _safe_ext(filename: str | None) -> str:
    """Return a validated file extension or raise HTTPException."""
    raw = os.path.splitext(filename or "data.csv")[1].lower() or ".csv"
    if raw not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{raw}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )
    return raw


def _sanitize_filename(name: str) -> str:
    """Remove all characters that are not alphanumeric, dash, or underscore."""
    return re.sub(r"[^\w\-]", "_", name)


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
        logger.info("Session %s completed successfully", session_id)
    except Exception as exc:
        logger.error("Session %s failed: %s", session_id, exc, exc_info=True)
        store.set_status("failed", error=str(exc))
        # Clean up uploaded file to avoid orphaned files on disk
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info("Cleaned up uploaded file for failed session %s", session_id)
            except OSError as rm_err:
                logger.warning("Could not remove upload file %s: %s", filepath, rm_err)


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("", summary="Upload dataset and start analysis")
@limiter.limit(RATE_LIMIT_ANALYSE)
async def start_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str = Form(""),
    model: str = Form(DEFAULT_MODEL),
):
    ext = _safe_ext(file.filename)

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit "
            f"({len(content) / 1024 / 1024:.1f} MB received)",
        )

    session_id = str(uuid.uuid4())
    safe_base  = _sanitize_filename(os.path.splitext(file.filename or "data")[0])
    save_path  = os.path.join(UPLOADS_DIR, f"{session_id}_{safe_base}{ext}")

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(
        "Session %s created — file: %s (%d bytes), query: %.80r, model: %s",
        session_id, file.filename, len(content), query, model,
    )

    store = SessionStore(session_id)
    store.create(dataset_path=save_path, query=query, model=model)

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
