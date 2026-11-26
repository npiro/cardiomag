"""Model registry: built-in models register by name; any external torch.nn.Module can be used with
`target: "package.module:ClassName"` in the training config.

Contract for a model class:
    Model(n_sensors: int, n_times: int, n_outputs: int, **kwargs) -> torch.nn.Module
    forward(x: Tensor[B, S, T]) -> logits Tensor[B, n_outputs]            (task: classify)
                                   reconstruction Tensor[B, S, T]          (task: reconstruct)
    class attribute `task` in {"classify", "reconstruct"} (default "classify")
"""
from __future__ import annotations

import importlib

_REGISTRY: dict = {}


def register_model(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_model_class(spec: dict):
    if "target" in spec:
        mod, _, cls = spec["target"].partition(":")
        return getattr(importlib.import_module(mod), cls)
    from . import models  # noqa: F401  (registers built-ins)
    name = spec["name"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; built-in: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_model(spec: dict, n_sensors: int, n_times: int, n_outputs: int):
    cls = get_model_class(spec)
    return cls(n_sensors=n_sensors, n_times=n_times, n_outputs=n_outputs, **spec.get("kwargs", {}))


def available():
    from . import models  # noqa: F401
    return sorted(_REGISTRY)
