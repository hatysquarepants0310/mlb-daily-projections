from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from mlbproj import api
from mlbproj.config import DATA_DIR, SLOT_PA
from mlbproj.saber import (
    binomial_at_least_one,
    confidence,
    decompose_counting,
    fip,
    interval,
    matchup_rates,
    park_adjust,
    rates_from_stat,
    shrink_rates,
    talent_blend,
    to_f,
    weather_hr_mult,
    woba_from_events,
)

ET = ZoneInfo("America/New_York")

# 3-year-ish run/HR park factors indexed by MLB team abbreviation.
# Source mix: FG 2024-25 public PFs, used as a prior; scaled around 1.00.
PARK_PF: dict[str, tuple[float, float]] = {
    "COL": (1.15, 1.18),
    "CIN": (1.08, 1.16),
    "BOS": (1.07, 1.05),
    "PHI": (1.05, 1.12),
    "NYY": (1.04, 1.14),
    "CHC": (1.04, 1.08),
    "TEX": (1.03, 1.06),
    "BAL": (1.03, 1.10),
    "MIL": (1.02, 1.08),
    "HOU": (1.02, 1.06),
    "ATL": (1.02, 1.07),
    "MIN": (1.01, 1.05),
    "TOR": (1.01, 1.04),
    "LAA": (1.01, 1.03),
    "WSH": (1.01, 1.02),
    "CWS": (1.00, 1.04),
    "CHW": (1.00, 1.04),
    "AZ": (1.00, 1.03),
    "ARI": (1.00, 1.03),
    "NYM": (0.99, 1.02),
    "DET": (0.99, 0.97),
    "KC": (0.99, 0.96),
    "CLE": (0.98, 0.94),
    "TB": (0.97, 0.92),
    "TBR": (0.97, 0.92),
    "MIA": (0.97, 0.90),
    "PIT": (0.97, 0.93),
    "STL": (0.98, 0.95),
    "LAD": (0.97, 1.04),
    "SF": (0.94, 0.86),
    "SD": (0.95, 0.92),
    "SEA": (0.94, 0.90),
    "OAK": (0.96, 0.90),
    "ATH": (0.96, 0.90),
}


def mlb_today() -> str:
    return datetime.now(ET).date().isoformat()


def _hand(side: dict[str, Any] | None) -> str:
    if not side:
        return "R"
    return (side.get("code") or "R")[0].upper()


def _split_map(person: dict[str, Any], group: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for block in person.get("stats") or []:
        g = (block.get("group") or {}).get("displayName")
        t = (block.get("type") or {}).get("displayName")
        if g != group:
            continue
        for sp in block.get("splits") or []:
            st = rates_from_stat(sp.get("stat") or {})
            if t == "season" and "season" not in out:
                out["season"] = st
            if t == "statSplits":
                code = ((sp.get("split") or {}).get("code") or "").lower()
                if code in {"vl", "vr"}:
                    out[code] = st
    return out


def _league_from_splits(splits: list[dict[str, Any]], min_pa: float = 80.0) -> dict[str, float]:
    acc = defaultdict(float)
    n = 0.0
    for sp in splits:
        st = sp.get("stat") or {}
        pa = to_f(st.get("plateAppearances") or st.get("battersFaced"))
        if pa < min_pa:
            continue
        r = rates_from_stat(st)
        w = pa
        n += w
        for k, v in r.items():
            acc[k] += v * w
    if n <= 0:
        return {
            "avg": 0.245,
            "obp": 0.315,
            "slg": 0.400,
            "iso": 0.155,
            "babip": 0.295,
            "k": 0.225,
            "bbp": 0.083,
            "hr_pa": 0.030,
            "woba": 0.315,
            "sb_pa": 0.018,
            "hbp_pa": 0.010,
            "pa": 1.0,
        }
    out = {k: acc[k] / n for k in acc}
    out["pa"] = n
    return out


def _pick_batter_split(splits: dict[str, dict[str, float]], pitcher_throws: str, batter_bats: str) -> dict[str, float]:
    season = splits.get("season") or {}
    if batter_bats == "S":
        code = "vl" if pitcher_throws == "L" else "vr"
    else:
        code = "vl" if pitcher_throws == "L" else "vr"
    split = splits.get(code)
    if split and split.get("pa", 0) >= 40:
        return split
    return season


def _pick_pitcher_split(splits: dict[str, dict[str, float]], batter_bats: str, pitcher_throws: str) -> dict[str, float]:
    season = splits.get("season") or {}
    # vs the side the batter will hit from
    if batter_bats == "S":
        side = "R" if pitcher_throws == "L" else "L"
    else:
        side = batter_bats
    code = "vl" if side == "L" else "vr"
    split = splits.get(code)
    if split and split.get("bf", split.get("pa", 0)) >= 50:
        return split
    return season


def weather_observed(game: dict[str, Any]) -> bool:
    w = game.get("weather") or {}
    temp = w.get("temp")
    if temp is None or temp == "":
        return False
    return bool(w.get("wind") or w.get("condition"))


def _weather(game: dict[str, Any]) -> dict[str, Any]:
    w = game.get("weather") or {}
    temp = to_f(w.get("temp"), 70.0)
    wind = w.get("wind") or ""
    return {
        "condition": w.get("condition") or "",
        "temp": temp,
        "wind": wind,
        "hr_mult": round(weather_hr_mult(temp, wind), 3),
        "observed": weather_observed(game),
    }


_INPUT_LABELS = {
    "sp_away": "SP visitante",
    "sp_home": "SP local",
    "lineup_away": "lineup visitante",
    "lineup_home": "lineup local",
    "weather": "weather",
}


def inputs_status(
    *,
    sp_away: bool,
    sp_home: bool,
    lineup_away: bool,
    lineup_home: bool,
    weather: bool,
) -> dict[str, Any]:
    checks = {
        "sp_away": bool(sp_away),
        "sp_home": bool(sp_home),
        "lineup_away": bool(lineup_away),
        "lineup_home": bool(lineup_home),
        "weather": bool(weather),
    }
    missing = [k for k, v in checks.items() if not v]
    if not missing:
        state = "listo"
        label = "Listo: lineup + SP + weather"
    elif any(checks.values()):
        state = "parcial"
        label = "Esperando: " + ", ".join(_INPUT_LABELS[k] for k in missing)
    else:
        state = "esperando"
        label = "Esperando: lineup + SP + weather"
    return {
        "ready": not missing,
        "state": state,
        "checks": checks,
        "missing": missing,
        "label": label,
    }


def _recency_index(splits: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for sp in splits:
        pid = (sp.get("player") or {}).get("id")
        if not pid:
            continue
        out[int(pid)] = rates_from_stat(sp.get("stat") or {})
    return out


def project_date(date_iso: str | None = None, force: bool = False) -> dict[str, Any]:
    date_iso = date_iso or mlb_today()
    cache_path = DATA_DIR / f"projections-{date_iso}.json"
    if cache_path.exists() and not force:
        age = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        if age < 180:
            return json.loads(cache_path.read_text())

    season = int(date_iso[:4])
    hyd = "probablePitcher,lineups,weather,venue,team"
    sched = api.schedule(date_iso, hydrate=hyd)
    games_raw = []
    for d in sched.get("dates") or []:
        games_raw.extend(d.get("games") or [])

    rec_start = (date.fromisoformat(date_iso) - timedelta(days=21)).isoformat()
    hit_season = api.season_leaders("hitting", season)
    pit_season = api.season_leaders("pitching", season)
    hit_rec = api.date_range_stats("hitting", season, rec_start, date_iso)
    pit_rec = api.date_range_stats("pitching", season, rec_start, date_iso)
    lg_hit = _league_from_splits(hit_season, 100)
    lg_pit = _league_from_splits(pit_season, 80)
    rec_hit = _recency_index(hit_rec)
    rec_pit = _recency_index(pit_rec)
    sav_bat = api.savant_expected("batter", season)
    sav_pit = api.savant_expected("pitcher", season)

    ids: list[int] = []
    for g in games_raw:
        for side in ("home", "away"):
            pp = (g.get("teams") or {}).get(side, {}).get("probablePitcher") or {}
            if pp.get("id"):
                ids.append(int(pp["id"]))
        lu = g.get("lineups") or {}
        for key in ("homePlayers", "awayPlayers"):
            for p in lu.get(key) or []:
                if p.get("id"):
                    ids.append(int(p["id"]))

    people = {int(p["id"]): p for p in api.people_hydrate(ids, season) if p.get("id")}

    games_out: list[dict[str, Any]] = []
    for g in games_raw:
        home = g["teams"]["home"]["team"]
        away = g["teams"]["away"]["team"]
        home_abbr = home.get("abbreviation") or home.get("teamName") or ""
        venue = (g.get("venue") or {}).get("name") or ""
        weather = _weather(g)
        pf_runs, pf_hr = PARK_PF.get(home_abbr, (1.0, 1.0))
        pf_hr *= weather["hr_mult"]
        status = (g.get("status") or {}).get("detailedState") or ""
        pp_home = (g["teams"]["home"].get("probablePitcher") or {})
        pp_away = (g["teams"]["away"].get("probablePitcher") or {})
        lu = g.get("lineups") or {}

        def pitcher_card(pp: dict[str, Any], team_abbr: str, opp_lineup: list[dict[str, Any]], opp_abbr: str) -> dict[str, Any] | None:
            pid = pp.get("id")
            if not pid:
                return None
            person = people.get(int(pid), {})
            splits = _split_map(person, "pitching")
            season_r = splits.get("season") or rates_from_stat({})
            rec = rec_pit.get(int(pid))
            sav = sav_pit.get(int(pid))
            shrunk = shrink_rates(season_r, lg_pit, n_key="bf" if season_r.get("bf") else "pa")
            rec_s = shrink_rates(rec, lg_pit, n_key="bf" if rec and rec.get("bf") else "pa") if rec else None
            xw = sav["est_woba"] if sav else None
            talent = talent_blend(shrunk, rec_s, xw, sav.get("est_ba") if sav else None, sav.get("est_slg") if sav else None, season_r.get("bf") or season_r.get("pa") or 0)
            throws = _hand(person.get("pitchHand"))
            # lineup mix vs this pitcher
            mix_l = 0
            mix_n = 0
            for b in opp_lineup:
                bats = _hand((people.get(int(b["id"]), {}) or {}).get("batSide"))
                if bats == "S":
                    bats = "R" if throws == "L" else "L"
                mix_n += 1
                if bats == "L":
                    mix_l += 1
            platoon = mix_l / mix_n if mix_n else 0.4
            vs_l = splits.get("vl") or talent
            vs_r = splits.get("vr") or talent
            k_rate = platoon * (vs_l.get("k") or talent["k"]) + (1 - platoon) * (vs_r.get("k") or talent["k"])
            bb_rate = platoon * (vs_l.get("bbp") or talent["bbp"]) + (1 - platoon) * (vs_r.get("bbp") or talent["bbp"])
            hr_rate = (platoon * (vs_l.get("hr_pa") or talent["hr_pa"]) + (1 - platoon) * (vs_r.get("hr_pa") or talent["hr_pa"])) * pf_hr
            avg_vs = platoon * (vs_l.get("avg") or talent["avg"]) + (1 - platoon) * (vs_r.get("avg") or talent["avg"])
            ip = 5.4 if (season_r.get("gs") or 0) >= 5 else 1.2
            if season_r.get("gs") and season_r.get("g"):
                if season_r["gs"] / max(season_r["g"], 1) > 0.6:
                    ip = 5.4
                else:
                    ip = 1.15
            bf = ip * 4.15
            k = bf * k_rate
            bb = bf * bb_rate
            h = bf * avg_vs * 0.92
            hr = bf * hr_rate
            hbp = bf * talent.get("hbp_pa", 0.01)
            fip_v = fip(hr, bb, hbp, k, ip)
            er = max(ip * fip_v / 9.0, 0.0)
            conf = confidence(
                season_r.get("bf") or season_r.get("pa") or 0,
                season_r.get("bf") or 0,
                bool(sav),
                bool(splits.get("vl")),
                xw,
                talent.get("woba"),
                (splits.get("vl") or {}).get("bf") or (splits.get("vl") or {}).get("pa") or 0,
            )
            return {
                "id": int(pid),
                "name": person.get("fullName") or pp.get("fullName") or "",
                "team": team_abbr,
                "opp": opp_abbr,
                "throws": throws,
                "role": "SP" if ip >= 3 else "RP",
                "ip": round(ip, 2),
                "k": round(k, 2),
                "bb": round(bb, 2),
                "h": round(h, 2),
                "hr": round(hr, 2),
                "er": round(er, 2),
                "fip": round(fip_v, 2),
                "k_rate": round(k_rate, 3),
                "bb_rate": round(bb_rate, 3),
                "xwoba_against": round(xw, 3) if xw else None,
                "confidence": conf,
                "platoon_lhb": round(platoon, 3),
            }

        def batter_rows(players: list[dict[str, Any]], team_abbr: str, opp_pp: dict[str, Any], opp_abbr: str) -> list[dict[str, Any]]:
            rows = []
            opp_id = opp_pp.get("id")
            opp_person = people.get(int(opp_id), {}) if opp_id else {}
            opp_throws = _hand(opp_person.get("pitchHand"))
            opp_splits = _split_map(opp_person, "pitching") if opp_person else {}
            opp_season = opp_splits.get("season") or (lg_pit)
            opp_name = opp_person.get("fullName") or opp_pp.get("fullName") or "TBD"
            for slot, pl in enumerate(players[:9], start=1):
                pid = int(pl["id"])
                person = people.get(pid, {})
                bats = _hand(person.get("batSide"))
                b_splits = _split_map(person, "hitting")
                season_r = b_splits.get("season") or rates_from_stat({})
                rec = rec_hit.get(pid)
                sav = sav_bat.get(pid)
                raw_split = _pick_batter_split(b_splits, opp_throws, bats)
                shrunk_season = shrink_rates(season_r, lg_hit)
                shrunk_split = shrink_rates(raw_split, lg_hit) if raw_split else shrunk_season
                rec_s = shrink_rates(rec, lg_hit) if rec else None
                xw = sav["est_woba"] if sav else None
                xba = sav.get("est_ba") if sav else None
                xslg = sav.get("est_slg") if sav else None
                talent = talent_blend(shrunk_split, rec_s, xw, xba, xslg, season_r.get("pa") or 0)
                p_split = _pick_pitcher_split(opp_splits, bats, opp_throws)
                p_shrunk = shrink_rates(p_split, lg_pit, n_key="bf" if p_split.get("bf") else "pa") if p_split else lg_pit
                matched = matchup_rates(talent, p_shrunk, lg_hit)
                parked = park_adjust(matched, pf_runs, pf_hr)
                pa = SLOT_PA[slot - 1]
                counting = decompose_counting(pa, parked)
                # R/RBI from lineup context: team run env * slot weights
                run_env = 4.55 * pf_runs * weather["hr_mult"] ** 0.3
                slot_r = (0.145, 0.135, 0.130, 0.125, 0.115, 0.105, 0.095, 0.085, 0.065)[slot - 1]
                slot_rbi = (0.100, 0.125, 0.145, 0.150, 0.130, 0.115, 0.095, 0.080, 0.060)[slot - 1]
                r = run_env * slot_r * (parked["woba"] / max(lg_hit["woba"], 0.01))
                rbi = run_env * slot_rbi * (parked["woba"] / max(lg_hit["woba"], 0.01))
                p_hit = binomial_at_least_one(counting["avg"], counting["ab"])
                p_hr = binomial_at_least_one(counting["hr"] / max(counting["pa"], 0.1), counting["pa"])
                conf = confidence(
                    season_r.get("pa") or 0,
                    opp_season.get("bf") or opp_season.get("pa") or 0,
                    bool(sav),
                    bool(b_splits.get("vl") or b_splits.get("vr")),
                    xw,
                    parked["woba"],
                    (raw_split or {}).get("pa") or 0,
                )
                lo_h, hi_h = interval(counting["h"], counting["ab"], counting["avg"])
                pos = ((pl.get("primaryPosition") or {}).get("abbreviation")) or ""
                rows.append(
                    {
                        "id": pid,
                        "name": person.get("fullName") or pl.get("fullName") or "",
                        "team": team_abbr,
                        "opp": opp_abbr,
                        "slot": slot,
                        "pos": pos,
                        "bats": bats,
                        "vs": opp_throws,
                        "pitcher": opp_name,
                        "pa": round(counting["pa"], 2),
                        "ab": round(counting["ab"], 2),
                        "h": round(counting["h"], 2),
                        "h_lo": lo_h,
                        "h_hi": hi_h,
                        "2b": round(counting["2b"], 2),
                        "3b": round(counting["3b"], 2),
                        "hr": round(counting["hr"], 2),
                        "r": round(r, 2),
                        "rbi": round(rbi, 2),
                        "bb": round(counting["bb"], 2),
                        "k": round(counting["so"], 2),
                        "sb": round(counting["sb"], 2),
                        "avg": round(counting["avg"], 3),
                        "obp": round(counting["obp"], 3),
                        "slg": round(counting["slg"], 3),
                        "ops": round(counting["ops"], 3),
                        "woba": round(parked["woba"], 3),
                        "xwoba": round(xw, 3) if xw else None,
                        "p_hit": round(p_hit, 3),
                        "p_hr": round(p_hr, 3),
                        "confidence": conf,
                        "park_runs": round(pf_runs, 3),
                        "weather_hr": weather["hr_mult"],
                    }
                )
            return rows

        away_abbr = away.get("abbreviation") or ""
        home_players = lu.get("homePlayers") or []
        away_players = lu.get("awayPlayers") or []
        home_p = pitcher_card(pp_home, home_abbr, away_players, away_abbr)
        away_p = pitcher_card(pp_away, away_abbr, home_players, home_abbr)
        batters = batter_rows(away_players, away_abbr, pp_home, home_abbr) + batter_rows(
            home_players, home_abbr, pp_away, away_abbr
        )
        pitchers = [p for p in (away_p, home_p) if p]
        inputs = inputs_status(
            sp_away=bool(pp_away.get("id")),
            sp_home=bool(pp_home.get("id")),
            lineup_away=len(away_players) >= 9,
            lineup_home=len(home_players) >= 9,
            weather=bool(weather.get("observed")),
        )
        games_out.append(
            {
                "gamePk": g.get("gamePk"),
                "gameType": g.get("gameType"),
                "status": status,
                "gameDate": g.get("gameDate"),
                "venue": venue,
                "weather": weather,
                "away": {"name": away.get("name"), "abbr": away_abbr, "id": away.get("id")},
                "home": {"name": home.get("name"), "abbr": home_abbr, "id": home.get("id")},
                "batters": batters,
                "pitchers": pitchers,
                "inputs": inputs,
            }
        )

    payload = {
        "date": date_iso,
        "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
        "model": {
            "version": "1.0.0",
            "stack": [
                "MLB Stats API lineups + probable pitchers + vl/vr splits",
                "Tango shrinkage (Marcel-style priors)",
                "Bill James Log5 matchup",
                "Statcast expected BA/SLG/wOBA (Baseball Savant)",
                "21-day recency blend",
                "Park factor + MLB weather (temp/wind) HR multiplier",
                "Binomial P(H>=1) / P(HR>=1)",
                "80% predictive interval from Bernoulli variance",
            ],
            "honest": (
                "Un juego es ruido. MAE típico de hits ~0.9–1.1. "
                "Confidence mide calidad de muestra, no probabilidad de clavar el boxscore."
            ),
            "league": {k: round(v, 3) for k, v in lg_hit.items() if k in {"avg", "obp", "slg", "iso", "k", "bbp", "hr_pa", "woba"}},
        },
        "games": games_out,
        "n_games": len(games_out),
        "n_batters": sum(len(g["batters"]) for g in games_out),
        "n_pitchers": sum(len(g["pitchers"]) for g in games_out),
    }
    cache_path.write_text(json.dumps(payload))
    return payload


# keep import used (woba helper available for tests via saber)
_ = woba_from_events
