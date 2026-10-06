from __future__ import annotations

import argparse

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mlbproj.config import HOST, PORT, ROOT
from mlbproj.engine import mlb_today, project_date

app = FastAPI(title="MLB Daily Projections", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "today": mlb_today()}


@app.get("/api/projections")
def projections(
    date: str | None = Query(default=None, description="YYYY-MM-DD, default MLB today ET"),
    force: bool = Query(default=False),
) -> JSONResponse:
    try:
        payload = project_date(date, force=force)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("mlbproj.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
