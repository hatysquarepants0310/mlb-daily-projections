from __future__ import annotations

import argparse
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mlbproj import ledger
from mlbproj.bets import LAST_CYCLE, enrich, run_cycle
from mlbproj.config import HOST, PORT, ROOT
from mlbproj.engine import mlb_today, project_date

_stop = threading.Event()


def _loop() -> None:
    # first pass immediately, then every 60s
    while True:
        try:
            run_cycle()
        except Exception:
            pass
        if _stop.wait(60):
            break


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t = threading.Thread(target=_loop, name="mlbproj-cycle", daemon=True)
    t.start()
    yield
    _stop.set()


app = FastAPI(title="MLB Daily Projections", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "today": mlb_today(), "cycle": LAST_CYCLE or None}


@app.get("/api/projections")
def projections(
    date: str | None = Query(default=None, description="YYYY-MM-DD, default MLB today ET"),
    force: bool = Query(default=False),
) -> JSONResponse:
    try:
        payload = enrich(project_date(date, force=force), record_ticks=False)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse(payload)


@app.get("/api/history")
def history() -> dict:
    return {"ok": True, "summary": ledger.summary(), "rows": ledger.history()}


@app.get("/api/ticks")
def ticks(pick_id: str = Query(...)) -> dict:
    return {"ok": True, "pick_id": pick_id, "ticks": ledger.ticks_for(pick_id, 80)}


@app.post("/api/tick")
def tick_now() -> dict:
    return {"ok": True, **run_cycle()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("mlbproj.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
