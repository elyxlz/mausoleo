"""Query embedding for semantic search.

The plan calls for BGE-M3 (multilingual; good for Italian). Loading
``sentence-transformers`` is heavy and the sentence-transformers package is
not in the base dependency set, so we degrade gracefully:

1. If ``MAUSOLEO_EMBED_BACKEND=zero`` (or the model can't be loaded), we
   return a zero vector. The semantic search endpoint will still work
   structurally (it will return rows ordered by L2 distance to the zero
   vector), which is enough for the loader/CLI smoke tests.

2. Otherwise we lazily import ``sentence_transformers`` and load the model.
   First call pays the cost; subsequent calls reuse the cached encoder.

Embedding dim defaults to 1024 (BGE-M3 dense). It's overridable via
``MAUSOLEO_EMBED_DIM``.
"""
from __future__ import annotations

import logging
import os
import typing as tp

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


class Embedder:
    def __init__(self, model_name: str | None = None, dim: int | None = None) -> None:
        self.model_name = model_name or os.environ.get("MAUSOLEO_EMBED_MODEL", DEFAULT_MODEL)
        self.dim = dim or int(os.environ.get("MAUSOLEO_EMBED_DIM", DEFAULT_DIM))
        self._model: tp.Any = None
        self._failed = False

    def _load(self) -> tp.Any:
        if self._model is not None or self._failed:
            return self._model
        backend = os.environ.get("MAUSOLEO_EMBED_BACKEND", "auto")
        if backend == "zero":
            self._failed = True
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:
            log.warning("sentence-transformers unavailable, falling back to zeros: %s", exc)
            self._failed = True
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            log.warning("failed to load embedding model %s: %s", self.model_name, exc)
            self._failed = True
            return None
        return self._model

    def encode(self, query: str) -> list[float]:
        model = self._load()
        if model is None:
            return [0.0] * self.dim
        vec = model.encode(query, normalize_embeddings=True)
        return [float(x) for x in vec]


# Singleton — created lazily by FastAPI startup or first call.
_DEFAULT: Embedder | None = None


def default_embedder() -> Embedder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Embedder()
    return _DEFAULT
