"""EarlyDX FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router
from backend.app.config import get_settings
from backend.app.database import get_session, init_db, sync_registries
from backend.app.services.model_loader import STORE
from backend.app.api.routes import _dataset_registry, _model_registry

settings = get_settings()
logging.basicConfig(level=settings.log_level)
log = logging.getLogger("earlydx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STORE.load_all()
    log.info("models loaded: %s (errors: %s)", STORE.available(), STORE.errors())
    if init_db():
        db = next(get_session())
        try:
            sync_registries(db, _model_registry(), _dataset_registry())
        finally:
            db.close()
        log.info("database ready and registries synced")
    else:
        log.warning("database unavailable; predictions will not be persisted")
    yield


app = FastAPI(
    title="EarlyDX API",
    version="0.1.0",
    description=(
        "Multi-disease early risk assessment. Research prototype, not a medical "
        "device, not clinically validated."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"service": "EarlyDX API", "docs": "/docs", "health": "/health"}
