"""Re-exports of the store backends that need an optional extra, imported on first use: importing a
store package then needs no extra, and a backend whose extra is missing names the one to install."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from agent_env.config.loader import install_hint


def lazy_backends(package: str, backends: dict[str, str]) -> Callable[[str], Any]:
    """The module ``__getattr__`` of ``package``, serving each ``name: module`` of ``backends``."""

    def __getattr__(name: str) -> Any:
        if name not in backends:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            module = importlib.import_module(backends[name])
        except ImportError as e:
            hint = install_hint(e)
            if not hint:
                raise
            raise ModuleNotFoundError(f"{package}.{name} cannot be imported: {e}{hint}") from e
        return getattr(module, name)

    return __getattr__
