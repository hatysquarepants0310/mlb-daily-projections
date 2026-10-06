from __future__ import annotations

import json
from typing import Any

from mlbproj import api

GAMMA = "https://gamma-api.polymarket.com"
POLY_TTL = 60

# MLB Stats API abbr → Polymarket slug token
POLY_ABBR = {
    "ARI": "az",
    "AZ": "az",
    "ATL": "atl",
    "BAL": "bal",
    "BOS": "bos",
    "CHC": "chc",
    "CWS": "cws",
    "CHW": "cws",
    "CIN": "cin",
    "CLE": "cle",
    "COL": "col",
    "DET": "det",
    "HOU": "hou",
    "KC": "kc",
    "LAA": "laa",
    "LAD": "lad",
    "MIA": "mia",
    "MIL": "mil",
    "MIN": "min",
    "NYM": "nym",
    "NYY": "nyy",
    "OAK": "oak",
    "ATH": "ath",
    "PHI": "phi",
    "PIT": "pit",
    "SD": "sd",
    "SEA": "sea",
    "SF": "sf",
    "STL": "stl",
    "TB": "tb",
    "TBR": "tb",
    "TEX": "tex",
    "TOR": "tor",
    "WSH": "wsh",
    "WAS": "wsh",
}


def _parse_list(raw: Any) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
    return list(raw)


def parse_prices(raw: Any) -> list[float]:
    out = []
    for x in _parse_list(raw):
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return []
    return out


def event_slugs(away_abbr: str, home_abbr: str, date_iso: str) -> list[str]:
    a = POLY_ABBR.get(away_abbr.upper(), away_abbr.lower())
    h = POLY_ABBR.get(home_abbr.upper(), home_abbr.lower())
    return [
        f"mlb-{a}-{h}-{date_iso}",
        f"mlb-{h}-{a}-{date_iso}",
    ]


def fetch_event(slug: str) -> dict | None:
    url = f"{GAMMA}/events?slug={slug}"
    try:
        data = api.http_json(url, POLY_TTL)
    except Exception:
        return None
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and data.get("slug"):
        return data
    return None


def fetch_game_event(away_abbr: str, home_abbr: str, date_iso: str) -> dict | None:
    for slug in event_slugs(away_abbr, home_abbr, date_iso):
        ev = fetch_event(slug)
        if ev:
            return ev
    return None


def markets(ev: dict) -> list[dict]:
    out = []
    slug = ev.get("slug") or ""
    for m in ev.get("markets") or []:
        prices = parse_prices(m.get("outcomePrices"))
        outcomes = [str(x) for x in _parse_list(m.get("outcomes"))]
        if len(prices) < 2 or len(outcomes) < 2:
            continue
        out.append(
            {
                "id": str(m.get("id") or ""),
                "smt": (m.get("sportsMarketType") or ""),
                "question": m.get("question") or "",
                "slug": m.get("slug") or slug,
                "outcomes": outcomes,
                "prices": prices,
                "closed": bool(m.get("closed") or ev.get("closed")),
                "url": f"https://polymarket.com/event/{slug}",
                "event_slug": slug,
            }
        )
    return out
