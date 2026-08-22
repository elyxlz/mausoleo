from __future__ import annotations

import dataclasses as dc


@dc.dataclass(frozen=True, kw_only=True)
class BaseOperatorConfig:
    mock: bool = False
