from __future__ import annotations

from dataclasses import asdict

from .strategies import TOP5_STRATEGIES


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("install project with [api] extras") from exc
    app = FastAPI(title="AIOS Track 2", version="0.2.0")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/strategies")
    def strategies():
        return [asdict(s) for s in TOP5_STRATEGIES]

    @app.get("/")
    def root():
        return FileResponse("ui/index.html")

    return app


app = None
try:
    app = create_app()
except RuntimeError:
    pass
