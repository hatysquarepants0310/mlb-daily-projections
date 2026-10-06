# MLB Daily Projections

Daily MLB player projections. Live data, not a 2017 DFS dump.

Stack:

- MLB Stats API: slate, lineups, probable pitchers, season + vs L/R splits, 21-day recency
- Baseball Savant expected BA / SLG / wOBA
- Tango / Marcel shrinkage on every rate
- Bill James Log5 batter × pitcher
- Park prior + MLB weather (temp / wind) as HR multiplier
- Binomial P(H≥1) and P(HR≥1), 80% interval on hits
- FIP-style pitcher card

The 2017 Hart repo (Python 2.7, MySQL, FanGraphs scrape, DraftKings optimizer) is years dead. Glouberman *MLB-hit-predictor* is a 2014–2019 Beat-the-Streak classifier (weather, venue, next-game hit). This system keeps the parts that still have edge — Log5, platoon, park, weather, P(hit) — and replaces the rest with 2026 public feeds.

Apuestas: momio de Polymarket (gamma) por juego (ML, O/U 6.5, -1.5, NRFI). El servicio patea cada 60s: ticks de precio (y CLOB 1h al primer sample) y **lock en T-10** del `gameDate` MLB. Al Final, linescore liquida. Ledger sqlite append-only + hash chain; UPDATE/DELETE abortan. No manda órdenes a Poly.

Foquito por juego (`inputs`): verde = lineup 9+9 + SP ambos + weather observado; ámbar = algo falta; rojo = nada posteado. No es el T-10. Hover muestra qué falta.

## Honesty

A single MLB game is high variance. Typical MAE on hits is around 0.9–1.1 even for good public systems. `confidence` is sample quality (PA, BF, whether xwOBA and splits exist), not “this will happen.” Intervals are Bernoulli 80%, not magic.

## Dashboard

Binds Tailscale IP only (not `0.0.0.0`):

```
http://100.118.48.64:8765
```

Same URL from haty-home and from tuf-laptop on the tailnet.

```bash
cd /home/haty/mlb-daily-projections
uv sync
uv run python -m mlbproj.web
```

systemd user unit: `mlbproj.service` (enable with `systemctl --user enable --now mlbproj`).

## API

- `GET /api/health`
- `GET /api/projections?date=YYYY-MM-DD&force=true`

## Model path (one game, one batter)

1. Season rates vs the handedness he will see, shrunk to league.
2. Blend 21-day recency (sample-size weighted).
3. Blend Statcast expected (more weight when PA is thin).
4. Log5 vs starter’s matching split (also shrunk).
5. Park × weather on extra-base/HR rates.
6. Lineup-slot PA → counting stats, R/RBI from run environment.

Pitchers: platoon mix of the posted lineup, expected IP by role, FIP from projected events.
