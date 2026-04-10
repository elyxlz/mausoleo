from __future__ import annotations

import fastapi as fa

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.server.routes import router


def create_app(config: OcrPipelineConfig | None = None) -> fa.FastAPI:
    app = fa.FastAPI(title="Mausoleo", version="0.1.0")
    app.state.pipeline_config = config
    app.include_router(router)
    return app
