"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> object | None:
    """Return a LangGraph checkpointer.

    Memory is the default for fast local tests. SQLite is used for the persistence
    extension and keeps checkpoints on disk across process restarts.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "outputs/checkpoints.sqlite"
        if db_path.startswith("sqlite:///"):
            db_path = db_path.removeprefix("sqlite:///")

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn=conn)
        if hasattr(saver, "setup"):
            saver.setup()
        return saver
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension and is not configured."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
