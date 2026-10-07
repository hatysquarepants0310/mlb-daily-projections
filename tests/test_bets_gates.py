from __future__ import annotations

import unittest

from mlbproj.bets import LEAGUE_YRFI, ML_MIN_GAP, build_quotes, p_over, p_yrfi


def _batter(team: str, slot: int, woba: float, runs: float) -> dict:
    return {"team": team, "slot": slot, "woba": woba, "r": runs}


def _sp(team: str, er: float, ip: float = 5.4, xw: float = 0.315) -> dict:
    return {
        "team": team,
        "role": "SP",
        "ip": ip,
        "er": er,
        "xwoba_against": xw,
    }


def _lineup(team: str, woba: float, runs_each: float) -> list[dict]:
    return [_batter(team, s, woba, runs_each) for s in range(1, 10)]


def _game(*, ready: bool, away_r: float, home_r: float, away_sp: dict, home_sp: dict, away_w: float = 0.315, home_w: float = 0.315) -> dict:
    return {
        "gamePk": 1,
        "away": {"name": "Los Angeles Dodgers", "abbr": "LAD"},
        "home": {"name": "Atlanta Braves", "abbr": "ATL"},
        "inputs": {"ready": ready},
        "batters": _lineup("LAD", away_w, away_r / 9.0) + _lineup("ATL", home_w, home_r / 9.0),
        "pitchers": [away_sp, home_sp],
    }


def _markets(yes: float = 0.45, over: float = 0.50, away_ml: float = 0.48) -> list[dict]:
    def m(mid: str, smt: str, slug: str, outcomes: list[str], prices: list[float]) -> dict:
        return {
            "id": mid,
            "smt": smt,
            "question": slug,
            "slug": slug,
            "outcomes": outcomes,
            "prices": prices,
            "tokens": [],
            "url": "https://polymarket.com/event/x",
            "event_slug": "mlb-lad-atl-2026-10-06",
        }

    return [
        m("ml", "moneyline", "mlb-lad-atl", ["Los Angeles Dodgers", "Atlanta Braves"], [away_ml, 1 - away_ml]),
        m("tot", "totals", "mlb-lad-atl-6pt5", ["Over", "Under"], [over, 1 - over]),
        m("nr", "nrfi", "mlb-lad-atl-nrfi", ["Yes", "No"], [yes, 1 - yes]),
    ]


class YRFITests(unittest.TestCase):
    def test_average_starters_near_league(self):
        g = _game(
            ready=True,
            away_r=4.4,
            home_r=4.4,
            away_sp=_sp("LAD", er=2.64),
            home_sp=_sp("ATL", er=2.64),
        )
        p = p_yrfi(g)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, LEAGUE_YRFI, delta=0.02)

    def test_aces_lower_than_league(self):
        avg = _game(ready=True, away_r=4.4, home_r=4.4, away_sp=_sp("LAD", 2.64), home_sp=_sp("ATL", 2.64))
        aces = _game(
            ready=True,
            away_r=4.4,
            home_r=4.4,
            away_sp=_sp("LAD", er=2.15, xw=0.266),
            home_sp=_sp("ATL", er=2.20, xw=0.270),
        )
        self.assertLess(p_yrfi(aces), p_yrfi(avg))

    def test_missing_starter_has_no_price(self):
        g = _game(ready=True, away_r=4.4, home_r=4.4, away_sp=_sp("LAD", 2.64), home_sp=_sp("ATL", 2.64, ip=1.15))
        g["pitchers"][1]["role"] = "RP"
        self.assertIsNone(p_yrfi(g))

    def test_no_lineup_has_no_price(self):
        g = _game(ready=False, away_r=4.4, home_r=4.4, away_sp=_sp("LAD", 2.64), home_sp=_sp("ATL", 2.64))
        g["batters"] = []
        self.assertIsNone(p_yrfi(g))


class DispersionTests(unittest.TestCase):
    def test_negbin_cuts_over_when_mean_above_line(self):
        poi = p_over(6.5, 8.6, disp=1.0)
        nb = p_over(6.5, 8.6)
        self.assertGreater(poi, nb)
        self.assertGreater(nb, 0.5)


class TakeGateTests(unittest.TestCase):
    def test_not_ready_shows_nothing(self):
        g = _game(
            ready=False,
            away_r=6.2,
            home_r=3.2,
            away_sp=_sp("LAD", 2.0, xw=0.250),
            home_sp=_sp("ATL", 4.5, xw=0.380),
        )
        rows = build_quotes(g, "2026-10-07", _markets(away_ml=0.35, yes=0.40, over=0.40))
        self.assertFalse(any(r.get("take") for r in rows))

    def test_aces_vs_cheap_yes_is_not_a_take(self):
        g = _game(
            ready=True,
            away_r=4.37,
            home_r=3.99,
            away_sp=_sp("LAD", er=2.15, xw=0.266),
            home_sp=_sp("ATL", er=2.20, xw=0.270),
        )
        self.assertGreater(abs(4.37 - 3.99), ML_MIN_GAP)
        rows = build_quotes(g, "2026-10-06", _markets(yes=0.39, over=0.52, away_ml=0.50))
        nrfi = [r for r in rows if r["kind"] == "nrfi" and r.get("take")]
        self.assertEqual(len(nrfi), 1)
        self.assertTrue(nrfi[0]["selection"].lower().startswith("no"))
        self.assertLess(nrfi[0]["edge"], 0.08)

    def test_bad_starters_can_take_yes(self):
        g = _game(
            ready=True,
            away_r=4.4,
            home_r=4.4,
            away_sp=_sp("LAD", er=4.2, xw=0.380),
            home_sp=_sp("ATL", er=4.2, xw=0.380),
        )
        rows = build_quotes(g, "2026-10-07", _markets(yes=0.42, over=0.55, away_ml=0.50))
        yes = [r for r in rows if r["kind"] == "nrfi" and r.get("take")]
        self.assertEqual(len(yes), 1)
        self.assertTrue(yes[0]["selection"].lower().startswith("yes"))

    def test_league_total_is_not_a_take(self):
        g = _game(
            ready=True,
            away_r=4.3,
            home_r=4.3,
            away_sp=_sp("LAD", 2.64),
            home_sp=_sp("ATL", 2.64),
        )
        rows = build_quotes(g, "2026-10-07", _markets(yes=0.50, over=0.45, away_ml=0.50))
        self.assertFalse(any(r.get("take") for r in rows if r["kind"] == "total"))

    def test_ml_take_needs_a_real_gap(self):
        tight = _game(
            ready=True,
            away_r=4.40,
            home_r=4.35,
            away_sp=_sp("LAD", 2.64),
            home_sp=_sp("ATL", 2.64),
        )
        wide = _game(
            ready=True,
            away_r=5.4,
            home_r=3.6,
            away_sp=_sp("LAD", 2.2, xw=0.270),
            home_sp=_sp("ATL", 3.4, xw=0.340),
        )
        cheap = _markets(yes=0.50, over=0.55, away_ml=0.40)
        self.assertFalse(any(r.get("take") for r in build_quotes(tight, "2026-10-07", cheap) if r["kind"] == "ml"))
        takes = [r for r in build_quotes(wide, "2026-10-07", cheap) if r["kind"] == "ml" and r.get("take")]
        self.assertEqual(len(takes), 1)
        self.assertIn("Dodgers", takes[0]["selection"])

    def test_negative_poly_edge_still_shows_model_side(self):
        g = _game(
            ready=True,
            away_r=5.4,
            home_r=3.6,
            away_sp=_sp("LAD", 2.2, xw=0.270),
            home_sp=_sp("ATL", 3.4, xw=0.340),
        )
        # Poly ya tiene a LAD de favorito fuerte: edge negativo, igual se muestra.
        rows = build_quotes(g, "2026-10-07", _markets(yes=0.50, over=0.55, away_ml=0.70))
        takes = [r for r in rows if r["kind"] == "ml" and r.get("take")]
        self.assertEqual(len(takes), 1)
        self.assertIn("Dodgers", takes[0]["selection"])
        self.assertLess(takes[0]["edge"], 0)


if __name__ == "__main__":
    unittest.main()
