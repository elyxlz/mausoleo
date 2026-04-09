from __future__ import annotations

import fastapi as fa


def create_app() -> fa.FastAPI:
    app = fa.FastAPI(title="Mausoleo", version="0.1.0")
    return app
