"""FastAPI application factory."""
from __future__ import annotations

import logging
import os

import fastapi as fa

from mausoleo.index import loader as index_loader
from mausoleo.server.db import Db, DbConfig
from mausoleo.server.routes import router

log = logging.getLogger(__name__)


def create_app(db_cfg: DbConfig | None = None) -> fa.FastAPI:
    app = fa.FastAPI(title="Mausoleo Search API", version="0.1.0")
    db = Db(db_cfg or DbConfig.from_env())
    app.state.db = db

    @app.on_event("startup")
    async def _on_startup() -> None:
        # Best-effort schema install on startup. The loader's ``setup_schema``
        # is idempotent so repeated boots are safe.
        if os.environ.get("MAUSOLEO_AUTO_SCHEMA", "1") == "1":
            try:
                index_loader.setup_schema(
                    host=db.cfg.host,
                    port=db.cfg.port,
                    database=db.cfg.database,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("schema setup failed on startup: %s", exc)

    app.include_router(router)
    return app
