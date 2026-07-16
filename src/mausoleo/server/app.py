"""FastAPI application factory."""

from __future__ import annotations

import contextlib
import logging
import os
import typing as tp

import fastapi as fa

from mausoleo.index import loader as index_loader
from mausoleo.server.db import Db, DbConfig
from mausoleo.server.routes import router

log = logging.getLogger(__name__)


def create_app(db_cfg: DbConfig | None = None) -> fa.FastAPI:
    db = Db(db_cfg or DbConfig.from_env())

    @contextlib.asynccontextmanager
    async def _lifespan(_: fa.FastAPI) -> tp.AsyncIterator[None]:
        if os.environ.get("MAUSOLEO_AUTO_SCHEMA", "1") == "1":
            try:
                index_loader.setup_schema(
                    host=db.cfg.host,
                    port=db.cfg.port,
                    database=db.cfg.database,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("schema setup failed on startup: %s", exc)
        yield

    app = fa.FastAPI(title="Mausoleo Search API", version="0.1.0", lifespan=_lifespan)
    app.state.db = db
    app.include_router(router)
    return app
