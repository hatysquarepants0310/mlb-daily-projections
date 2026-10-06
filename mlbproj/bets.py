from __future__ import annotations

import math
import re
from typing import Any

from mlbproj import api, ledger, poly
from mlbproj.config import MLB_API

PX_LO = 0.18
PX_HI = 0.82
SHRINK = 0.70
MIN_EDGE = 0.03
PREGAME = {"scheduled", "pre-game", "pregame", "preview"}
FINAL = {"final", "game over", "completed"}


def momio_decimal(p: float) -> float:
    if p <= 0:
        return 99.0
    return round(1.0 / p, 3)


def momio_americano(p: float) -> int:
    p = min(max(p, 0.01), 0.99)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def shrink(p: float) -> float:
    return 0.5 + SHRINK * (p - 0.5)


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    logp = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(logp)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def p_over(line: float, mu: float) -> float:
    # O/U 6.5 → P(total >= 7)
    need = math.floor(line) + 1
    return max(0.01, min(0.99, 1.0 - poisson_cdf(need - 1, mu)))


def p_win(ra: float, rh: float, away: bool) -> float:
    ra = max(ra, 0.3)
    rh = max(rh, 0.3)
    exp = 1.83
    pa = (ra**exp) / (ra**exp + rh**exp)
    return pa if away else 1.0 - pa


def p_cover(margin: float, sd: float, line: float) -> float:
    """P(team_runs - opp_runs > -line) with normal approx. line=-1.5 means must win by 2."""
    # cover -1.5 iff margin >= 2 iff margin > 1.5
    z = (1.5 - margin) / max(sd, 0.8) if line == -1.5 else ((-line) - margin) / max(sd, 0.8)
    # P(M > 1.5) = 1-Phi((1.5-mu)/sd)
    return max(0.02, min(0.98, 0.5 * math.erfc(z / math.sqrt(2.0))))


def _status_key(status: str) -> str:
    return (status or "").strip().lower()


def can_lock(status: str) -> bool:
    return _status_key(status) in PREGAME


def is_final(status: str) -> bool:
    s = _status_key(status)
    return s in FINAL or s.startswith("final")


def team_runs(game: dict[str, Any]) -> tuple[float, float]:
    away = game["away"]["abbr"]
    home = game["home"]["abbr"]
    ra = sum(b["r"] for b in game.get("batters") or [] if b.get("team") == away)
    rh = sum(b["r"] for b in game.get("batters") or [] if b.get("team") == home)
    if ra <= 0:
        ra = 4.3
    if rh <= 0:
        rh = 4.4
    return ra, rh


def _name_match(a: str, b: str) -> bool:
    def n(s: str) -> str:
        s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
        return " ".join(s.split())

    na, nb = n(a), n(b)
    return na == nb or na in nb or nb in na


def reliability(model_p: float, implied_p: float, conf: float) -> float:
    sh = shrink(model_p)
    edge = sh - implied_p
    # 0-100: mix of shrunk p and whether we actually have edge + sample conf
    base = 100.0 * sh
    if edge < 0:
        base *= 0.85
    base = 0.65 * base + 0.35 * conf
    return round(min(max(base, 15.0), 82.0), 1)


def _row(
    *,
    game: dict,
    date_iso: str,
    market: dict,
    selection: str,
    implied: float,
    model_p: float,
    kind: str,
    settle: dict,
    take: bool,
    note: str,
    conf: float,
) -> dict:
    sh = shrink(model_p)
    implied = float(implied)
    return {
        "pick_id": f"{market['id']}:{kind}:{selection}",
        "game_pk": game["gamePk"],
        "card_date": date_iso,
        "market_id": market["id"],
        "event_slug": market["event_slug"],
        "url": market["url"],
        "smt": market["smt"],
        "question": market["question"],
        "kind": kind,
        "selection": selection,
        "implied_p": round(implied, 4),
        "momio_decimal": momio_decimal(implied),
        "momio_americano": momio_americano(implied),
        "model_p": round(model_p, 4),
        "model_shrunk": round(sh, 4),
        "edge": round(sh - implied, 4),
        "reliability": reliability(model_p, implied, conf),
        "in_band": PX_LO <= implied <= PX_HI,
        "take": take,
        "note": note,
        "settle": settle,
        "locked": False,
        "status": "live",
    }


def build_quotes(game: dict, date_iso: str, markets: list[dict]) -> list[dict]:
    ra, rh = team_runs(game)
    mu = ra + rh
    sd = math.sqrt(max(mu, 1.0))
    away_name = game["away"]["name"]
    home_name = game["home"]["name"]
    conf = 0.0
    n = 0
    for b in game.get("batters") or []:
        conf += float(b.get("confidence") or 0)
        n += 1
    conf = conf / n if n else 50.0
    pa = p_win(ra, rh, True)
    ph = 1.0 - pa
    rows: list[dict] = []

    for m in markets:
        smt = m["smt"]
        if smt == "moneyline":
            for name, px, mp, side in (
                (m["outcomes"][0], m["prices"][0], pa if _name_match(m["outcomes"][0], away_name) else ph, "away" if _name_match(m["outcomes"][0], away_name) else "home"),
                (m["outcomes"][1], m["prices"][1], pa if _name_match(m["outcomes"][1], away_name) else ph, "away" if _name_match(m["outcomes"][1], away_name) else "home"),
            ):
                take = False
                rows.append(
                    _row(
                        game=game,
                        date_iso=date_iso,
                        market=m,
                        selection=name,
                        implied=px,
                        model_p=mp,
                        kind="ml",
                        settle={"kind": "ml", "side": side},
                        take=take,
                        note="moneyline",
                        conf=conf,
                    )
                )
        elif smt == "totals" and "6pt5" in (m.get("slug") or ""):
            po = p_over(6.5, mu)
            for name, px, mp, side in (
                ("Over", m["prices"][0] if m["outcomes"][0].lower().startswith("over") else m["prices"][1], po, "over"),
                ("Under", m["prices"][1] if m["outcomes"][0].lower().startswith("over") else m["prices"][0], 1.0 - po, "under"),
            ):
                # map price to outcome name
                idx = 0 if m["outcomes"][0].lower().startswith(name.lower()) else 1
                px = m["prices"][idx]
                rows.append(
                    _row(
                        game=game,
                        date_iso=date_iso,
                        market=m,
                        selection=m["outcomes"][idx],
                        implied=px,
                        model_p=mp,
                        kind="total",
                        settle={"kind": "total", "side": side, "line": 6.5},
                        take=False,
                        note="O/U 6.5",
                        conf=conf,
                    )
                )
        elif smt == "spreads" and m.get("slug", "").endswith("1pt5"):
            # outcomes[0] is the listed team at that spread
            listed = m["outcomes"][0]
            listed_away = _name_match(listed, away_name)
            margin = (ra - rh) if listed_away else (rh - ra)
            # slug away-1pt5 means away -1.5
            slug = m.get("slug") or ""
            if "away-1pt5" in slug or "home-1pt5" in slug:
                p_list = p_cover(margin, sd, -1.5)
                px = m["prices"][0]
                side = "away" if listed_away else "home"
                rows.append(
                    _row(
                        game=game,
                        date_iso=date_iso,
                        market=m,
                        selection=f"{listed} -1.5",
                        implied=px,
                        model_p=p_list,
                        kind="spread",
                        settle={"kind": "spread", "side": side, "line": -1.5},
                        take=False,
                        note="spread -1.5",
                        conf=conf,
                    )
                )
        elif smt == "nrfi":
            p_yes = 1.0 - math.exp(-mu / 8.2)
            p_yes = max(0.15, min(0.75, p_yes))
            for i, name in enumerate(m["outcomes"]):
                side = "yes" if name.lower().startswith("yes") else "no"
                mp = p_yes if side == "yes" else 1.0 - p_yes
                rows.append(
                    _row(
                        game=game,
                        date_iso=date_iso,
                        market=m,
                        selection=name,
                        implied=m["prices"][i],
                        model_p=mp,
                        kind="nrfi",
                        settle={"kind": "nrfi", "side": side},
                        take=False,
                        note="run in 1st",
                        conf=conf,
                    )
                )

    # mark at most one take per kind: best edge in-band, model_shrunk>=0.52, edge>=MIN_EDGE
    by_kind: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_kind.setdefault(r["kind"], []).append(i)
    for kind, idxs in by_kind.items():
        ranked = sorted(idxs, key=lambda i: rows[i]["edge"], reverse=True)
        for i in ranked:
            r = rows[i]
            if r["in_band"] and r["model_shrunk"] >= 0.52 and r["edge"] >= MIN_EDGE:
                r["take"] = True
                break
    return rows


def linescore(game_pk: int) -> dict | None:
    url = f"{MLB_API}/game/{game_pk}/linescore"
    try:
        return api.http_json(url, 30)
    except Exception:
        return None


def grade(settle: dict, ls: dict) -> bool | None:
    teams = ls.get("teams") or {}
    away = int((teams.get("away") or {}).get("runs") or 0)
    home = int((teams.get("home") or {}).get("runs") or 0)
    kind = settle.get("kind")
    if kind == "ml":
        if away == home:
            return None
        if settle.get("side") == "away":
            return away > home
        return home > away
    if kind == "total":
        line = float(settle.get("line") or 6.5)
        total = away + home
        if abs(total - line) < 1e-9:
            return None
        over = total > line
        return over if settle.get("side") == "over" else (not over)
    if kind == "spread":
        line = float(settle.get("line") or -1.5)
        margin = (away - home) if settle.get("side") == "away" else (home - away)
        # -1.5 cover iff margin >= 2
        return margin + line > 0
    if kind == "nrfi":
        inn = (ls.get("innings") or [{}])
        first = inn[0] if inn else {}
        fr = int((first.get("away") or {}).get("runs") or 0) + int((first.get("home") or {}).get("runs") or 0)
        yes = fr > 0
        return yes if settle.get("side") == "yes" else (not yes)
    return None


def enrich(payload: dict) -> dict:
    date_iso = payload["date"]
    settled = ledger.settled_map()
    for g in payload.get("games") or []:
        ev = poly.fetch_game_event(g["away"]["abbr"], g["home"]["abbr"], date_iso)
        if not ev:
            g["bets"] = []
            g["poly_url"] = None
            g["bets_note"] = "Sin evento Polymarket para este juego."
            continue
        mk = poly.markets(ev)
        quotes = build_quotes(g, date_iso, mk)
        g["poly_url"] = f"https://polymarket.com/event/{ev.get('slug')}"
        lockable = can_lock(g.get("status") or "")
        final = is_final(g.get("status") or "")
        ls = linescore(int(g["gamePk"])) if final else None
        out = []
        for q in quotes:
            pid = q["pick_id"]
            existing = ledger.get_pick(pid)
            if existing:
                # freeze model/momio; keep live poly as overlay
                frozen = dict(existing)
                frozen["live_implied_p"] = q["implied_p"]
                frozen["live_momio_decimal"] = q["momio_decimal"]
                if pid in settled:
                    frozen.update({k: settled[pid][k] for k in settled[pid]})
                elif final and ls is not None:
                    won = grade(frozen.get("settle") or q["settle"], ls)
                    if won is not None:
                        graded = ledger.settle_pick(pid, won, {"away": (ls.get("teams") or {}).get("away"), "home": (ls.get("teams") or {}).get("home")})
                        if graded:
                            frozen = graded
                else:
                    frozen["status"] = "locked"
                out.append(frozen)
                continue
            if lockable and q["take"]:
                frozen = ledger.lock_pick(q)
                frozen["live_implied_p"] = q["implied_p"]
                out.append(frozen)
                continue
            q["status"] = "sin-lock" if not lockable else "live"
            if not lockable:
                q["note"] = (q.get("note") or "") + " · no entra al historial (ya empezó o ya terminó)"
            out.append(q)
        g["bets"] = out
        takes = sum(1 for r in out if r.get("take") or r.get("locked"))
        g["bets_note"] = (
            "Lock solo en pregame. Historial append-only, no se reescribe. No hay orden a Polymarket."
            if takes or out
            else "Sin quotes."
        )
    payload["history"] = ledger.history()
    payload["history_summary"] = ledger.summary()
    return payload
