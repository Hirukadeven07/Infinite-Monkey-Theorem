"""SQLite persistence for runs, samples, per-question results and completed bench items."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from app.metrics import ResultRow
from app.schema import SampleRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, kind TEXT, status TEXT,
    config_json TEXT, lock_json TEXT);
CREATE TABLE IF NOT EXISTS samples (
    run_id INTEGER, question_id TEXT, method TEXT, budget INTEGER, idx INTEGER, text TEXT,
    answer TEXT, correct INTEGER, prm_json TEXT, tokens INTEGER, seconds REAL);
CREATE INDEX IF NOT EXISTS samples_run ON samples(run_id, question_id);
CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER, question_id TEXT, method TEXT, budget INTEGER, accuracy REAL,
    tokens INTEGER, seconds REAL, PRIMARY KEY (run_id, question_id, method, budget));
CREATE TABLE IF NOT EXISTS items_done (
    run_id INTEGER, question_id TEXT, item TEXT, budget INTEGER, gpu TEXT, detail_json TEXT,
    PRIMARY KEY (run_id, question_id, item, budget));
"""


class Store:
    def __init__(self, path: Path = Path("runs/tts.db")):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _exec(self, sql: str, params=(), many: bool = False) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.executemany(sql, params) if many else self._db.execute(sql, params)
            self._db.commit()
            return cur

    def _query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # -- runs ----------------------------------------------------------------
    def create_run(self, kind: str, config: dict, lock: dict) -> int:
        cur = self._exec("INSERT INTO runs (created, kind, status, config_json, lock_json) VALUES (?,?,?,?,?)",
                         (time.time(), kind, "created", json.dumps(config), json.dumps(lock)))
        return int(cur.lastrowid)

    @staticmethod
    def _run(row: sqlite3.Row) -> dict:
        return {"id": row["id"], "created": row["created"], "kind": row["kind"], "status": row["status"],
                "config": json.loads(row["config_json"]), "lock": json.loads(row["lock_json"])}

    def get_run(self, run_id: int) -> dict | None:
        rows = self._query("SELECT * FROM runs WHERE id = ?", (run_id,))
        return self._run(rows[0]) if rows else None

    def list_runs(self, kind: str) -> list[dict]:
        return [self._run(r) for r in self._query("SELECT * FROM runs WHERE kind = ? ORDER BY id DESC", (kind,))]

    def set_status(self, run_id: int, status: str) -> None:
        self._exec("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    # -- samples -------------------------------------------------------------
    def add_samples(self, run_id: int, qid: str, method: str, budget: int, samples: list[SampleRecord]) -> None:
        self._exec("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            (run_id, qid, method, budget, s.idx, s.text, s.answer, int(s.correct),
             json.dumps(s.prm_steps), s.tokens, s.seconds) for s in samples], many=True)

    def samples(self, run_id: int, qid: str) -> list[SampleRecord]:
        rows = self._query("SELECT * FROM samples WHERE run_id = ? AND question_id = ? ORDER BY rowid",
                           (run_id, qid))
        return [SampleRecord(r["idx"], r["text"], r["answer"], bool(r["correct"]), json.loads(r["prm_json"]),
                             r["tokens"], r["seconds"]) for r in rows]

    # -- results -------------------------------------------------------------
    def save_rows(self, run_id: int, rows: list[ResultRow]) -> None:
        self._exec("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?)", [
            (run_id, r.question_id, r.method, r.budget, r.accuracy, r.tokens, r.seconds) for r in rows], many=True)

    def rows(self, run_id: int) -> list[ResultRow]:
        return [ResultRow(r["question_id"], r["method"], r["budget"], r["accuracy"], r["tokens"], r["seconds"])
                for r in self._query("SELECT * FROM results WHERE run_id = ? ORDER BY rowid", (run_id,))]

    # -- items ---------------------------------------------------------------
    def mark_done(self, run_id: int, qid: str, item: str, budget: int, gpu: str | None, detail: dict) -> None:
        self._exec("INSERT OR REPLACE INTO items_done VALUES (?,?,?,?,?,?)",
                   (run_id, qid, item, budget, gpu, json.dumps(detail)))

    def failed_count(self, run_id: int) -> int:
        rows = self._query("SELECT detail_json FROM items_done WHERE run_id = ?", (run_id,))
        return sum(1 for r in rows if "failed" in json.loads(r["detail_json"] or "{}"))

    def done_items(self, run_id: int) -> set[tuple[str, str, int]]:
        return {(r["question_id"], r["item"], r["budget"])
                for r in self._query("SELECT question_id, item, budget FROM items_done WHERE run_id = ?", (run_id,))}
