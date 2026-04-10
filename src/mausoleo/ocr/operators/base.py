from __future__ import annotations

import dataclasses as dc
import inspect
import typing as tp
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import partial

import ray.data


class OperatorType(StrEnum):
    MAP = "map"
    MAP_BATCHES = "map_batches"
    FILTER = "filter"
    FLAT_MAP = "flat_map"


@dc.dataclass(frozen=True, kw_only=True)
class BaseOperatorConfig:
    gpu_fraction: float = 0.0
    batch_size: int = 1
    min_actors: int = 1
    mock: bool = False
    runtime_env: dict[str, tp.Any] | None = None


ConfigT = tp.TypeVar("ConfigT", bound=BaseOperatorConfig)


class StatefulOperator(ABC, tp.Generic[ConfigT]):
    @abstractmethod
    def __init__(self, config: ConfigT) -> None: ...

    @abstractmethod
    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]: ...


@dc.dataclass(frozen=True)
class OperatorEntry:
    operation: OperatorType
    impl: type[tp.Any] | tp.Callable[..., tp.Any]


OPERATOR_REGISTRY: dict[type[tp.Any], OperatorEntry] = {}


def register_operator(
    config_class: type[tp.Any],
    *,
    operation: OperatorType,
) -> tp.Callable[[tp.Any], tp.Any]:
    def decorator(op_impl: tp.Any) -> tp.Any:
        OPERATOR_REGISTRY[config_class] = OperatorEntry(operation=operation, impl=op_impl)
        return op_impl

    return decorator


def apply_operator(
    ds: ray.data.Dataset,
    *,
    step_config: BaseOperatorConfig,
    n_gpu: int,
    n_gpu_operators: int = 1,
) -> ray.data.Dataset:
    config_type = type(step_config)
    if config_type not in OPERATOR_REGISTRY:
        raise ValueError(f"unregistered operator: {config_type.__name__}")

    entry = OPERATOR_REGISTRY[config_type]
    is_stateful = inspect.isclass(entry.impl)

    if entry.operation == OperatorType.MAP_BATCHES:
        if is_stateful:
            gpu_budget = max(1, n_gpu // max(n_gpu_operators, 1)) if step_config.gpu_fraction > 0 else 0
            max_actors = max(1, int(gpu_budget / step_config.gpu_fraction)) if step_config.gpu_fraction > 0 else 4
            kwargs: dict[str, tp.Any] = {
                "fn_constructor_args": (step_config,),
                "batch_size": step_config.batch_size,
                "num_gpus": step_config.gpu_fraction if step_config.gpu_fraction > 0 else 0,
                "compute": ray.data.ActorPoolStrategy(min_size=step_config.min_actors, max_size=max_actors),
            }
            if step_config.runtime_env is not None:
                kwargs["runtime_env"] = step_config.runtime_env
            return ds.map_batches(entry.impl, **kwargs)
        return ds.map_batches(partial(entry.impl, config=step_config), batch_size=step_config.batch_size)

    if entry.operation == OperatorType.MAP:
        return ds.map(partial(entry.impl, config=step_config))

    if entry.operation == OperatorType.FILTER:
        return ds.filter(partial(entry.impl, config=step_config))

    if entry.operation == OperatorType.FLAT_MAP:
        return ds.flat_map(partial(entry.impl, config=step_config))

    raise ValueError(f"unsupported operation: {entry.operation}")
