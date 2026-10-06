from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from mlbproj.config import DATA_DIR

LEDGER = DATA_DIR / "ledger.sqlite"
TZ = ZoneInfo("America/Mexico_City")
GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LEDGER, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=8000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS picks (
              pick_id TEXT PRIMARY KEY,
              game_pk INTEGER NOT NULL,
              card_date TEXT NOT NULL,
              payload TEXT NOT NULL,
              locked_at TEXT NOT NULL,
              prev_hash TEXT NOT NULL,
              row_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settled (
              pick_id TEXT PRIMARY KEY,
              won INTEGER NOT NULL,
              settled_at TEXT NOT NULL,
              actual TEXT NOT NULL,
              prev_hash TEXT NOT NULL,
              row_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chain_tip (
              k TEXT PRIMARY KEY,
              h TEXT NOT NULL
            );
            INSERT OR IGNORE INTO chain_tip(k, h) VALUES ('picks', '"""
            + GENESIS
            + """');
            INSERT OR IGNORE INTO chain_tip(k, h) VALUES ('settled', '"""
            + GENESIS
            + """');
            CREATE TRIGGER IF NOT EXISTS picks_no_update BEFORE UPDATE ON picks
              BEGIN SELECT RAISE(ABORT, 'picks are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS picks_no_delete BEFORE DELETE ON picks
              BEGIN SELECT RAISE(ABORT, 'picks are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS settled_no_update BEFORE UPDATE ON settled
              BEGIN SELECT RAISE(ABORT, 'settled are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS settled_no_delete BEFORE DELETE ON settled
              BEGIN SELECT RAISE(ABORT, 'settled are immutable'); END;
            CREATE TABLE IF NOT EXISTS ticks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tick_key TEXT NOT NULL,
              game_pk INTEGER NOT NULL,
              ts TEXT NOT NULL,
              implied_p REAL NOT NULL,
              momio_decimal REAL NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS ticks_no_update BEFORE UPDATE ON ticks
              BEGIN SELECT RAISE(ABORT, 'ticks are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS ticks_no_delete BEFORE DELETE ON ticks
              BEGIN SELECT RAISE(ABORT, 'ticks are immutable'); END;
            """
        )


def _hash(prev: str, blob: str) -> str:
    return hashlib.sha256((prev + "\n" + blob).encode("utf-8")).hexdigest()


@contextmanager
def db():
    init()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def lock_pick(pick: dict) -> dict:
    """Insert-only. If pick_id exists, return the frozen row (payload never changes)."""
    init()
    pick_id = pick["pick_id"]
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT payload, locked_at, row_hash FROM picks WHERE pick_id=?", (pick_id,)).fetchone()
        if row:
            frozen = json.loads(row["payload"])
            frozen["pick_id"] = pick_id
            frozen["locked_at"] = row["locked_at"]
            frozen["row_hash"] = row["row_hash"]
            frozen["locked"] = True
            return frozen
        locked_at = _now()
        payload = json.dumps(pick, ensure_ascii=False, sort_keys=True)
        prev = conn.execute("SELECT h FROM chain_tip WHERE k='picks'").fetchone()["h"]
        row_hash = _hash(prev, pick_id + locked_at + payload)
        conn.execute(
            "INSERT INTO picks(pick_id, game_pk, card_date, payload, locked_at, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?)",
            (pick_id, int(pick["game_pk"]), pick["card_date"], payload, locked_at, prev, row_hash),
        )
        conn.execute("UPDATE chain_tip SET h=? WHERE k='picks'", (row_hash,))
        pick = json.loads(payload)
        pick["pick_id"] = pick_id
        pick["locked_at"] = locked_at
        pick["row_hash"] = row_hash
        pick["locked"] = True
        return pick


def get_pick(pick_id: str) -> dict | None:
    init()
    with db() as conn:
        row = conn.execute("SELECT payload, locked_at, row_hash FROM picks WHERE pick_id=?", (pick_id,)).fetchone()
    if not row:
        return None
    p = json.loads(row["payload"])
    p["pick_id"] = pick_id
    p["locked_at"] = row["locked_at"]
    p["row_hash"] = row["row_hash"]
    p["locked"] = True
    return p


def settle_pick(pick_id: str, won: bool, actual: dict) -> dict | None:
    """Insert-only grade. Second call is a no-op."""
    init()
    frozen = get_pick(pick_id)
    if not frozen:
        return None
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        exists = conn.execute("SELECT won, settled_at, actual, row_hash FROM settled WHERE pick_id=?", (pick_id,)).fetchone()
        if exists:
            frozen["won"] = bool(exists["won"])
            frozen["settled_at"] = exists["settled_at"]
            frozen["actual"] = json.loads(exists["actual"])
            frozen["settle_hash"] = exists["row_hash"]
            frozen["status"] = "ganada" if exists["won"] else "perdida"
            return frozen
        settled_at = _now()
        actual_s = json.dumps(actual, ensure_ascii=False, sort_keys=True)
        prev = conn.execute("SELECT h FROM chain_tip WHERE k='settled'").fetchone()["h"]
        blob = f"{pick_id}{int(won)}{settled_at}{actual_s}"
        row_hash = _hash(prev, blob)
        conn.execute(
            "INSERT INTO settled(pick_id, won, settled_at, actual, prev_hash, row_hash) VALUES (?,?,?,?,?,?)",
            (pick_id, 1 if won else 0, settled_at, actual_s, prev, row_hash),
        )
        conn.execute("UPDATE chain_tip SET h=? WHERE k='settled'", (row_hash,))
    frozen["won"] = won
    frozen["settled_at"] = settled_at
    frozen["actual"] = actual
    frozen["settle_hash"] = row_hash
    frozen["status"] = "ganada" if won else "perdida"
    return frozen


def settled_map() -> dict[str, dict]:
    init()
    with db() as conn:
        rows = conn.execute("SELECT pick_id, won, settled_at, actual, row_hash FROM settled").fetchall()
    out = {}
    for r in rows:
        out[r["pick_id"]] = {
            "won": bool(r["won"]),
            "settled_at": r["settled_at"],
            "actual": json.loads(r["actual"]),
            "settle_hash": r["row_hash"],
            "status": "ganada" if r["won"] else "perdida",
        }
    return out


def history() -> list[dict]:
    init()
    with db() as conn:
        rows = conn.execute(
            """
            SELECT p.pick_id, p.payload, p.locked_at, p.row_hash,
                   s.won, s.settled_at, s.actual, s.row_hash AS settle_hash
            FROM picks p
            LEFT JOIN settled s ON s.pick_id = p.pick_id
            ORDER BY p.locked_at DESC
            """
        ).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload"])
        p["pick_id"] = r["pick_id"]
        p["locked_at"] = r["locked_at"]
        p["row_hash"] = r["row_hash"]
        p["locked"] = True
        if r["settled_at"]:
            p["won"] = bool(r["won"])
            p["settled_at"] = r["settled_at"]
            p["actual"] = json.loads(r["actual"])
            p["settle_hash"] = r["settle_hash"]
            p["status"] = "ganada" if r["won"] else "perdida"
        else:
            p["status"] = "locked"
        out.append(p)
    return out


def summary() -> dict:
    rows = [r for r in history() if r.get("status") in {"ganada", "perdida"}]
    n = len(rows)
    w = sum(1 for r in rows if r.get("won"))
    pnl = 0.0
    for r in rows:
        # $100 MXN a momio decimal locked
        dec = float(r.get("momio_decimal") or 0)
        if r.get("won"):
            pnl += 100.0 * (dec - 1.0)
        else:
            pnl -= 100.0
    return {
        "n_settled": n,
        "won": w,
        "lost": n - w,
        "hit_rate": round(w / n, 3) if n else None,
        "pnl_100mxn": round(pnl, 2),
        "n_locked": len(history()),
    }


def add_tick(tick_key: str, game_pk: int, implied_p: float, momio_decimal: float, ts: str | None = None) -> None:
    init()
    ts = ts or _now()
    with db() as conn:
        last = conn.execute(
            "SELECT ts, implied_p FROM ticks WHERE tick_key=? ORDER BY id DESC LIMIT 1",
            (tick_key,),
        ).fetchone()
        if last and last["ts"] == ts:
            return
        conn.execute(
            "INSERT INTO ticks(tick_key, game_pk, ts, implied_p, momio_decimal) VALUES (?,?,?,?,?)",
            (tick_key, int(game_pk), ts, float(implied_p), float(momio_decimal)),
        )


def ticks_for(tick_key: str, limit: int = 48) -> list[dict]:
    init()
    with db() as conn:
        rows = conn.execute(
            "SELECT ts, implied_p, momio_decimal FROM ticks WHERE tick_key=? ORDER BY id DESC LIMIT ?",
            (tick_key, limit),
        ).fetchall()
    out = [{"ts": r["ts"], "implied_p": r["implied_p"], "momio_decimal": r["momio_decimal"]} for r in rows]
    out.reverse()
    return out


def tick_count(tick_key: str) -> int:
    init()
    with db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM ticks WHERE tick_key=?", (tick_key,)).fetchone()[0]
    return int(n)
