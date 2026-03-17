"""
NL follow-up query route — multi-turn conversation on top of an existing session.
"""
from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse
from memory.session_store import SessionStore
from memory.memory_manager import MemoryManager

router = APIRouter(prefix="/sessions", tags=["query"])


@router.post("/{session_id}/query", response_model=QueryResponse)
async def nl_query(session_id: str, body: QueryRequest):
    store = SessionStore(session_id)
    data = store.load()
    if data is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if data.get("status") != "done":
        raise HTTPException(400, "Analysis must be complete before querying")

    manager = MemoryManager(session_id)
    answer = await manager.answer_question(body.question, model=body.model)

    return QueryResponse(
        session_id=session_id,
        question=body.question,
        answer=answer,
        history_length=len(store.load().get("messages", [])),
    )


@router.get("/{session_id}/history")
async def get_history(session_id: str):
    store = SessionStore(session_id)
    data = store.load()
    if data is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return {"session_id": session_id, "messages": data.get("messages", [])}


@router.delete("/{session_id}/history")
async def clear_history(session_id: str):
    store = SessionStore(session_id)
    store.clear_messages()
    return {"session_id": session_id, "cleared": True}
