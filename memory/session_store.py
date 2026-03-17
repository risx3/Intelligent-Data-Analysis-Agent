"""
File-backed session store.
Each session lives at outputs/sessions/<session_id>.json.
"""
import json
import os
from datetime import datetime, timezone
from config import SESSIONS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.path = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, dataset_path: str, query: str, model: str) -> dict:
        data = {
            "session_id":   self.session_id,
            "status":       "queued",
            "dataset_path": dataset_path,
            "query":        query,
            "model":        model,
            "messages":     [],
            "execution_log": [],
            "report":       None,
            "plots":        [],
            "insights":     "",
            "error":        "",
            "created_at":   _now(),
            "updated_at":   _now(),
        }
        self._write(data)
        return data

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict):
        data["updated_at"] = _now()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── status helpers ─────────────────────────────────────────────────────────

    def set_status(self, status: str, error: str = ""):
        data = self.load() or {}
        data["status"] = status
        if error:
            data["error"] = error
        self._write(data)

    def update_result(self, result: dict):
        data = self.load() or {}
        data["report"]        = result.get("report")
        data["plots"]         = result.get("plots", [])
        data["insights"]      = result.get("insights", "")
        data["execution_log"] = result.get("execution_log", [])
        self._write(data)

    def update_execution_log(self, log: list[dict]):
        data = self.load() or {}
        data["execution_log"] = log
        self._write(data)

    # ── message / memory helpers ───────────────────────────────────────────────

    def add_message(self, role: str, content: str):
        data = self.load() or {}
        data.setdefault("messages", []).append({
            "role":      role,
            "content":   content,
            "timestamp": _now(),
        })
        self._write(data)

    def get_messages(self) -> list[dict]:
        data = self.load() or {}
        return data.get("messages", [])

    def clear_messages(self):
        data = self.load() or {}
        data["messages"] = []
        self._write(data)
