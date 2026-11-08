"""SQLite persistence for justimagine chat history."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "justimagine.db"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL DEFAULT 'New chat',
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              context TEXT,
              has_context INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
              role TEXT NOT NULL,
              job_id TEXT,
              payload TEXT NOT NULL,
              ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat_ts
              ON messages(chat_id, ts);
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        conn.commit()


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _title_from_prompt(prompt: str) -> str:
    text = " ".join((prompt or "").strip().split())
    if not text:
        return "New chat"
    if len(text) <= 42:
        return text
    return text[:39].rstrip() + "…"


def create_chat(title: str = "New chat") -> dict[str, Any]:
    chat_id = str(uuid.uuid4())
    now = time.time()
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO chats(id, title, created_at, updated_at, context, has_context) "
            "VALUES(?, ?, ?, ?, NULL, 0)",
            (chat_id, title, now, now),
        )
        _set_meta(conn, "active_chat_id", chat_id)
        conn.commit()
    return {
        "id": chat_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "context": None,
        "has_context": False,
        "message_count": 0,
    }


def ensure_active_chat() -> dict[str, Any]:
    with _lock:
        conn = _connect()
        active = _get_meta(conn, "active_chat_id")
        if active:
            row = conn.execute("SELECT * FROM chats WHERE id = ?", (active,)).fetchone()
            if row:
                return _chat_row(conn, row)
        row = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row:
            _set_meta(conn, "active_chat_id", row["id"])
            conn.commit()
            return _chat_row(conn, row)
    return create_chat()


def _chat_row(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?", (row["id"],)
    ).fetchone()["n"]
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "context": row["context"],
        "has_context": bool(row["has_context"]),
        "message_count": int(count),
    }


def list_chats(limit: int = 100) -> list[dict[str, Any]]:
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT c.*, "
            "(SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count "
            "FROM chats c ORDER BY c.updated_at DESC LIMIT ?",
            (max(1, min(500, int(limit))),),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "context": r["context"],
                "has_context": bool(r["has_context"]),
                "message_count": int(r["message_count"]),
            }
            for r in rows
        ]


def chat_count() -> int:
    with _lock:
        conn = _connect()
        return int(conn.execute("SELECT COUNT(*) AS n FROM chats").fetchone()["n"])


def get_chat(chat_id: str) -> dict[str, Any] | None:
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not row:
            return None
        return _chat_row(conn, row)


def set_active_chat(chat_id: str) -> dict[str, Any] | None:
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not row:
            return None
        _set_meta(conn, "active_chat_id", chat_id)
        conn.commit()
        return _chat_row(conn, row)


def get_active_chat_id() -> str | None:
    with _lock:
        return _get_meta(_connect(), "active_chat_id")


def load_messages(chat_id: str) -> list[dict[str, Any]]:
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT payload FROM messages WHERE chat_id = ? ORDER BY ts ASC, rowid ASC",
            (chat_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            msg = json.loads(row["payload"])
            if isinstance(msg, dict):
                out.append(msg)
        except Exception:
            continue
    return out


def append_message(chat_id: str, msg: dict[str, Any]) -> None:
    payload = dict(msg)
    msg_id = str(payload.get("id") or uuid.uuid4())
    payload["id"] = msg_id
    ts = float(payload.get("ts") or time.time())
    payload["ts"] = ts
    role = str(payload.get("role") or "assistant")
    job_id = payload.get("job_id")
    title_bump = None
    if role == "user":
        text = (payload.get("text") or "").strip()
        if text:
            title_bump = _title_from_prompt(text)

    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO messages(id, chat_id, role, job_id, payload, ts) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (msg_id, chat_id, role, job_id, json.dumps(payload, ensure_ascii=False), ts),
        )
        if title_bump:
            existing = conn.execute(
                "SELECT title, "
                "(SELECT COUNT(*) FROM messages m WHERE m.chat_id = chats.id AND m.role = 'user') AS uc "
                "FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
            # Name the chat from the first user prompt.
            if existing and (existing["title"] == "New chat" or int(existing["uc"] or 0) <= 1):
                conn.execute(
                    "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                    (title_bump, ts, chat_id),
                )
            else:
                conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (ts, chat_id))
        else:
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (ts, chat_id))
        conn.commit()


def update_chat_context(chat_id: str, context: str | None, has_context: bool) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "UPDATE chats SET context = ?, has_context = ?, updated_at = ? WHERE id = ?",
            (context, 1 if has_context else 0, time.time(), chat_id),
        )
        conn.commit()


def delete_chat(chat_id: str) -> bool:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        deleted = cur.rowcount > 0
        active = _get_meta(conn, "active_chat_id")
        if deleted and active == chat_id:
            nxt = conn.execute(
                "SELECT id FROM chats ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if nxt:
                _set_meta(conn, "active_chat_id", nxt["id"])
            else:
                conn.execute("DELETE FROM meta WHERE key = ?", ("active_chat_id",))
        conn.commit()
        return deleted


def touch_chat(chat_id: str) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (time.time(), chat_id),
        )
        conn.commit()
