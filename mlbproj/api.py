from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mlbproj.config import (
    CACHE_DB,
    MLB_API,
    PEOPLE_TTL,
    SAVANT_EXPECTED,
    SAVANT_TTL,
    SCHEDULE_TTL,
    STATS_TTL,
    USER_AGENT,
)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(CACHE_DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, ts REAL NOT NULL, v TEXT NOT NULL)"
    )
    return con


def cache_get(key: str, ttl: float) -> Any | None:
    con = _conn()
    row = con.execute("SELECT ts, v FROM kv WHERE k = ?", (key,)).fetchone()
    con.close()
    if not row:
        return None
    ts, raw = row
    if time.time() - ts > ttl:
        return None
    return json.loads(raw)


def cache_set(key: str, value: Any) -> None:
    con = _conn()
    con.execute(
        "INSERT OR REPLACE INTO kv (k, ts, v) VALUES (?, ?, ?)",
        (key, time.time(), json.dumps(value)),
    )
    con.commit()
    con.close()


def http_json(url: str, ttl: float, timeout: float = 30.0) -> Any:
    cached = cache_get(url, ttl)
    if cached is not None:
        return cached
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} {url}") from exc
    cache_set(url, data)
    return data


def http_text(url: str, ttl: float, timeout: float = 45.0) -> str:
    cached = cache_get("text:" + url, ttl)
    if cached is not None:
        return cached
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8-sig")
    cache_set("text:" + url, text)
    return text


def schedule(date_iso: str, hydrate: str | None = None) -> dict[str, Any]:
    params = {"sportId": "1", "date": date_iso}
    if hydrate:
        params["hydrate"] = hydrate
    url = MLB_API + "/schedule?" + urllib.parse.urlencode(params)
    return http_json(url, SCHEDULE_TTL)


def people_hydrate(ids: list[int], season: int) -> list[dict[str, Any]]:
    if not ids:
        return []
    out: list[dict[str, Any]] = []
    hydrate = (
        f"stats(group=[hitting,pitching],type=[season,statSplits],"
        f"sitCodes=[vl,vr],season={season})"
    )
    uniq = []
    seen: set[int] = set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    for i in range(0, len(uniq), 40):
        chunk = uniq[i : i + 40]
        ids_s = ",".join(str(x) for x in chunk)
        url = (
            MLB_API
            + "/people?"
            + urllib.parse.urlencode({"personIds": ids_s, "hydrate": hydrate})
        )
        data = http_json(url, PEOPLE_TTL)
        out.extend(data.get("people") or [])
    return out


def season_leaders(group: str, season: int, limit: int = 2000) -> list[dict[str, Any]]:
    url = (
        MLB_API
        + "/stats?"
        + urllib.parse.urlencode(
            {
                "stats": "season",
                "group": group,
                "season": str(season),
                "sportIds": "1",
                "limit": str(limit),
                "playerPool": "all",
            }
        )
    )
    data = http_json(url, STATS_TTL)
    stats = data.get("stats") or []
    if not stats:
        return []
    return stats[0].get("splits") or []


def date_range_stats(group: str, season: int, start: str, end: str, limit: int = 2000) -> list[dict[str, Any]]:
    url = (
        MLB_API
        + "/stats?"
        + urllib.parse.urlencode(
            {
                "stats": "byDateRange",
                "group": group,
                "season": str(season),
                "sportIds": "1",
                "startDate": start,
                "endDate": end,
                "limit": str(limit),
                "playerPool": "all",
            }
        )
    )
    data = http_json(url, STATS_TTL)
    stats = data.get("stats") or []
    if not stats:
        return []
    return stats[0].get("splits") or []


def savant_expected(kind: str, year: int) -> dict[int, dict[str, float]]:
    url = SAVANT_EXPECTED.format(kind=kind, year=year)
    try:
        text = http_text(url, SAVANT_TTL)
    except Exception:
        return {}
    reader = csv.DictReader(io.StringIO(text))
    out: dict[int, dict[str, float]] = {}
    for row in reader:
        pid = row.get("player_id")
        if not pid:
            continue
        try:
            i = int(pid)
        except ValueError:
            continue
        def f(key: str) -> float | None:
            v = row.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        out[i] = {
            "pa": f("pa") or 0.0,
            "ba": f("ba") or 0.0,
            "est_ba": f("est_ba") or 0.0,
            "slg": f("slg") or 0.0,
            "est_slg": f("est_slg") or 0.0,
            "woba": f("woba") or 0.0,
            "est_woba": f("est_woba") or 0.0,
        }
    return out
