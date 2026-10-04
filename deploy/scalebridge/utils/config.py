"""Small config helpers for deployment paths that do not require Hydra."""

from __future__ import annotations

import importlib
from functools import partial
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access used by ScaleBridge classes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = to_config(value)


def to_config(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode({key: to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def load_yaml(path: Path) -> ConfigNode:
    with path.open("r", encoding="utf-8") as handle:
        return to_config(yaml.safe_load(handle) or {})


def deep_merge(base: dict, update: dict) -> ConfigNode:
    result = to_config(base.copy())
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = to_config(value)
    return result


def resolve_target(target: str):
    module_name, attribute = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


def instantiate(spec: dict, **overrides):
    spec = to_config(spec)
    if "_target_" not in spec:
        raise ValueError(f"Config has no _target_: {spec}")
    target = resolve_target(str(spec["_target_"]))
    kwargs = {
        key: value
        for key, value in spec.items()
        if not key.startswith("_")
    }
    kwargs.update(overrides)
    if spec.get("_partial_", False):
        return partial(target, **kwargs)
    return target(**kwargs)


def set_dotted(config: ConfigNode, path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("Config override path cannot be empty")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Unknown config override path: {path}")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config override path: {path}")
    node[parts[-1]] = to_config(value)
