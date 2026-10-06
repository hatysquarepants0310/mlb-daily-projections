from __future__ import annotations

import math
from typing import Any

from mlbproj.config import (
    K_AVG,
    K_BABIP,
    K_BB,
    K_HR,
    K_ISO,
    K_K,
    K_OBP,
    K_SB,
    K_WOBA,
    W_1B,
    W_2B,
    W_3B,
    W_BB,
    W_HBP,
    W_HR,
)


def to_f(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace("%", "").strip()
    if s in {".---", "-.--", "-"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def clamp(x: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return lo if x < lo else hi if x > hi else x


def log5(batter: float, pitcher: float, league: float) -> float:
    """Bill James Log5. batter/pitcher/league are rates in (0,1)."""
    b = clamp(batter)
    p = clamp(pitcher)
    lg = clamp(league, 0.02, 0.98)
    num = (b * p) / lg
    den = num + ((1.0 - b) * (1.0 - p) / (1.0 - lg))
    if den <= 0:
        return lg
    return num / den


def shrink(obs: float, n: float, league: float, k: float) -> float:
    if n <= 0:
        return league
    return (n * obs + k * league) / (n + k)


def woba_from_events(
    bb: float,
    hbp: float,
    singles: float,
    doubles: float,
    triples: float,
    hr: float,
    ab: float,
    ibb: float,
    sf: float,
) -> float:
    den = ab + bb - ibb + sf + hbp
    if den <= 0:
        return 0.0
    num = (
        W_BB * max(bb - ibb, 0.0)
        + W_HBP * hbp
        + W_1B * singles
        + W_2B * doubles
        + W_3B * triples
        + W_HR * hr
    )
    return num / den


def rates_from_stat(st: dict[str, Any]) -> dict[str, float]:
    pa = to_f(st.get("plateAppearances") or st.get("battersFaced"))
    ab = to_f(st.get("atBats"))
    h = to_f(st.get("hits"))
    doubles = to_f(st.get("doubles"))
    triples = to_f(st.get("triples"))
    hr = to_f(st.get("homeRuns"))
    bb = to_f(st.get("baseOnBalls"))
    ibb = to_f(st.get("intentionalWalks"))
    hbp = to_f(st.get("hitByPitch") or st.get("hitBatsmen"))
    so = to_f(st.get("strikeOuts"))
    sf = to_f(st.get("sacFlies"))
    sb = to_f(st.get("stolenBases"))
    singles = max(h - doubles - triples - hr, 0.0)
    avg = h / ab if ab else to_f(st.get("avg"))
    obp = to_f(st.get("obp"))
    slg = to_f(st.get("slg"))
    iso = max(slg - avg, 0.0) if slg or avg else 0.0
    babip_den = ab - so - hr + sf
    babip = (h - hr) / babip_den if babip_den > 0 else to_f(st.get("babip"))
    k_rate = so / pa if pa else 0.0
    bb_rate = bb / pa if pa else 0.0
    hr_pa = hr / pa if pa else 0.0
    hbp_pa = hbp / pa if pa else 0.0
    sb_pa = sb / pa if pa else 0.0
    woba = woba_from_events(bb, hbp, singles, doubles, triples, hr, ab, ibb, sf)
    ip = to_f(st.get("inningsPitched"))
    bf = to_f(st.get("battersFaced"))
    er = to_f(st.get("earnedRuns"))
    return {
        "pa": pa,
        "ab": ab,
        "h": h,
        "doubles": doubles,
        "triples": triples,
        "hr": hr,
        "bb": bb,
        "ibb": ibb,
        "hbp": hbp,
        "so": so,
        "sf": sf,
        "sb": sb,
        "singles": singles,
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "iso": iso,
        "babip": babip,
        "k": k_rate,
        "bbp": bb_rate,
        "hr_pa": hr_pa,
        "hbp_pa": hbp_pa,
        "sb_pa": sb_pa,
        "woba": woba,
        "ip": ip,
        "bf": bf,
        "er": er,
        "era": to_f(st.get("era")),
        "whip": to_f(st.get("whip")),
        "gs": to_f(st.get("gamesStarted")),
        "g": to_f(st.get("gamesPlayed") or st.get("gamesPitched")),
    }


def shrink_rates(r: dict[str, float], lg: dict[str, float], n_key: str = "pa") -> dict[str, float]:
    n = r.get(n_key, 0.0)
    out = dict(r)
    out["avg"] = shrink(r["avg"], n, lg["avg"], K_AVG)
    out["obp"] = shrink(r["obp"], n, lg["obp"], K_OBP)
    out["iso"] = shrink(r["iso"], n, lg["iso"], K_ISO)
    out["slg"] = out["avg"] + out["iso"]
    out["babip"] = shrink(r["babip"], n, lg["babip"], K_BABIP)
    out["k"] = shrink(r["k"], n, lg["k"], K_K)
    out["bbp"] = shrink(r["bbp"], n, lg["bbp"], K_BB)
    out["hr_pa"] = shrink(r["hr_pa"], n, lg["hr_pa"], K_HR)
    out["woba"] = shrink(r["woba"], n, lg["woba"], K_WOBA)
    out["sb_pa"] = shrink(r["sb_pa"], n, lg["sb_pa"], K_SB)
    out["hbp_pa"] = shrink(r["hbp_pa"], n, lg.get("hbp_pa", 0.01), 200)
    return out


def blend(a: float, b: float, w_a: float) -> float:
    return w_a * a + (1.0 - w_a) * b


def talent_blend(
    season: dict[str, float],
    recency: dict[str, float] | None,
    xwoba: float | None,
    xba: float | None,
    xslg: float | None,
    pa_season: float,
) -> dict[str, float]:
    """Season (shrunk) + recency + Statcast expected. More xStats when sample is thin."""
    t = dict(season)
    if recency and recency.get("pa", 0) >= 25:
        w_season = pa_season / (pa_season + 80.0)
        w_season = clamp(w_season, 0.45, 0.82)
        for k in ("avg", "obp", "iso", "slg", "babip", "k", "bbp", "hr_pa", "woba", "sb_pa"):
            t[k] = blend(season[k], recency[k], w_season)
    if xwoba and xwoba > 0:
        w_x = 0.62 if pa_season < 150 else 0.50
        t["woba"] = blend(xwoba, t["woba"], w_x)
    if xba and xba > 0:
        t["avg"] = blend(xba, t["avg"], 0.55)
    if xslg and xslg > 0:
        t["slg"] = blend(xslg, t["slg"], 0.55)
        t["iso"] = max(t["slg"] - t["avg"], 0.0)
    return t


def matchup_rates(
    batter: dict[str, float],
    pitcher: dict[str, float],
    lg: dict[str, float],
) -> dict[str, float]:
    keys = ("avg", "obp", "slg", "iso", "babip", "k", "bbp", "hr_pa", "woba")
    out = {}
    for k in keys:
        out[k] = log5(batter[k], pitcher[k], lg[k])
    out["sb_pa"] = batter.get("sb_pa", 0.0) * (0.85 + 0.3 * (pitcher.get("sb_pa", 0.02) / max(lg.get("sb_pa", 0.02), 1e-6)))
    out["hbp_pa"] = batter.get("hbp_pa", 0.01)
    return out


def parse_wind(wind: str) -> tuple[float, float]:
    """Return (mph, hr_mult) from MLB strings like '13 mph, In From CF'."""
    if not wind:
        return 0.0, 1.0
    mph = 0.0
    parts = wind.replace(",", " ").split()
    for i, p in enumerate(parts):
        if p.lower() == "mph" and i:
            try:
                mph = float(parts[i - 1])
            except ValueError:
                pass
    w = wind.lower()
    direction = 0.0
    if "out" in w and ("cf" in w or "center" in w):
        direction = 1.0
    elif "in from" in w and ("cf" in w or "center" in w):
        direction = -1.0
    elif "out to lf" in w or "out from lf" in w:
        direction = 0.4
    elif "out to rf" in w or "out from rf" in w:
        direction = 0.4
    hr_mult = 1.0 + 0.012 * mph * direction
    return mph, hr_mult


def weather_hr_mult(temp_f: float, wind: str) -> float:
    temp_adj = 1.0 + 0.007 * ((temp_f - 70.0) / 10.0)
    _, wind_adj = parse_wind(wind)
    return max(0.82, min(1.22, temp_adj * wind_adj))


def park_adjust(rates: dict[str, float], pf_runs: float, pf_hr: float) -> dict[str, float]:
    out = dict(rates)
    run_mult = 1.0 + 0.55 * (pf_runs - 1.0)
    out["woba"] = rates["woba"] * run_mult
    out["obp"] = rates["obp"] * (1.0 + 0.35 * (pf_runs - 1.0))
    out["avg"] = rates["avg"] * (1.0 + 0.30 * (pf_runs - 1.0))
    out["iso"] = rates["iso"] * (1.0 + 0.70 * (pf_hr - 1.0))
    out["hr_pa"] = rates["hr_pa"] * pf_hr
    out["slg"] = out["avg"] + out["iso"]
    return out


def binomial_at_least_one(p: float, n: float) -> float:
    """P(X>=1) with non-integer n via Poisson-binomial approximation."""
    p = clamp(p, 0.0, 0.6)
    n = max(n, 0.0)
    if n <= 0:
        return 0.0
    return 1.0 - math.exp(n * math.log(max(1.0 - p, 1e-9)))


def decompose_counting(pa: float, rates: dict[str, float]) -> dict[str, float]:
    pa = max(pa, 0.0)
    bb = pa * rates.get("bbp", 0.08)
    hbp = pa * rates.get("hbp_pa", 0.01)
    sf = pa * 0.01
    ab = max(pa - bb - hbp - sf, 0.0)
    avg = clamp(rates.get("avg", 0.25), 0.08, 0.45)
    h = ab * avg
    hr = pa * rates.get("hr_pa", 0.03)
    hr = min(hr, h)
    iso = max(rates.get("iso", 0.15), 0.0)
    extra = max(iso * ab, 0.0)
    triples = min(0.02 * ab, extra * 0.08)
    doubles = min(max((extra - 3.0 * hr - 2.0 * triples) / 1.0, 0.0), h - hr)
    # extra bases ≈ 1*2B + 2*3B + 3*HR; solve leftover as 2B
    singles = max(h - doubles - triples - hr, 0.0)
    so = pa * rates.get("k", 0.22)
    sb = pa * rates.get("sb_pa", 0.02)
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    slg = tb / ab if ab else 0.0
    obp = (h + bb + hbp) / pa if pa else 0.0
    return {
        "pa": pa,
        "ab": ab,
        "h": h,
        "1b": singles,
        "2b": doubles,
        "3b": triples,
        "hr": hr,
        "bb": bb,
        "hbp": hbp,
        "so": so,
        "sb": sb,
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": obp + slg,
        "woba": rates.get("woba", 0.0),
    }


def fip(hr: float, bb: float, hbp: float, so: float, ip: float, c: float = 3.20) -> float:
    if ip <= 0:
        return 99.0
    return c + (13.0 * hr + 3.0 * (bb + hbp) - 2.0 * so) / ip


def confidence(
    pa_batter: float,
    bf_pitcher: float,
    has_x: bool,
    has_split: bool,
    xwoba: float | None = None,
    woba: float | None = None,
    split_pa: float = 0.0,
) -> float:
    """0-100 quality of the *expected value*, not P(boxscore matches). One game is still noisy."""
    sample = 18.0 * (1.0 - math.exp(-pa_batter / 220.0))
    opp = 10.0 * (1.0 - math.exp(-bf_pitcher / 280.0))
    split_q = 8.0 * (1.0 - math.exp(-max(split_pa, 0.0) / 90.0)) if has_split else 0.0
    x_bonus = 7.0 if has_x else 0.0
    disagree = 0.0
    if xwoba and woba:
        disagree = min(12.0, 80.0 * abs(xwoba - woba))
    raw = 38.0 + sample + opp + split_q + x_bonus - disagree
    return round(clamp(raw, 22.0, 78.0), 1)


def interval(mean: float, pa: float, p_event: float) -> tuple[float, float]:
    sd = math.sqrt(max(pa, 1.0) * p_event * (1.0 - p_event))
    lo = max(0.0, mean - 1.28 * sd)
    hi = mean + 1.28 * sd
    return round(lo, 2), round(hi, 2)
