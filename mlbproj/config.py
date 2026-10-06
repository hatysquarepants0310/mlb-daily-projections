from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"
SAVANT_EXPECTED = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type={kind}&year={year}&min=1&csv=true"
)
USER_AGENT = "mlb-daily-projections/1.0 (haty-home; personal research)"

HOST = "100.118.48.64"
PORT = 8765

# Tango-style regression-to-mean priors (PA or BF).
K_WOBA = 200
K_AVG = 200
K_OBP = 180
K_ISO = 160
K_K = 60
K_BB = 120
K_BABIP = 200
K_HR = 170
K_SB = 80

# 2024 FanGraphs wOBA weights (stable enough for 2026 talent scale).
W_BB = 0.690
W_HBP = 0.722
W_1B = 0.888
W_2B = 1.271
W_3B = 1.616
W_HR = 2.101
W_OBA_SCALE = 1.24

SLOT_PA = (4.65, 4.55, 4.45, 4.32, 4.20, 4.08, 3.95, 3.82, 3.70)

CACHE_DB = DATA_DIR / "cache.sqlite"
SCHEDULE_TTL = 180
STATS_TTL = 3600
SAVANT_TTL = 12 * 3600
PEOPLE_TTL = 3600
