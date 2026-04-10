from __future__ import annotations

import dataclasses as dc
import typing as tp

import fastapi as fa

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.pipeline import run_pipeline

router = fa.APIRouter()


@router.post("/ocr")
async def ocr_endpoint(
    request: fa.Request,
    files: list[fa.UploadFile],
    date: str = fa.Form(...),
    source: str = fa.Form(default="il_messaggero"),
) -> dict[str, tp.Any]:
    config: OcrPipelineConfig | None = request.app.state.pipeline_config
    if config is None:
        raise fa.HTTPException(status_code=500, detail="no pipeline config set")
    images = [await f.read() for f in files]
    issue = run_pipeline(config, images, date, source)
    return dc.asdict(issue)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
