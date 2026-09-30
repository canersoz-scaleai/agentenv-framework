"""A plugin's own settings: its ``[plugins.<distribution name>]`` table.

Internal: the public reader is ``agent_env.plugins.settings``. ``config show`` and ``config explain``
find a table with the same lookup, so the report and the plugin agree about which table it reads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import distributions
from typing import Any, Optional

from packaging.utils import canonicalize_name

from agent_env.config import loader as config_loader
from agent_env.config import runtime
from agent_env.config.errors import ConfigError

TABLE = "plugins"
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Installed:
    """An installed distribution: its version, whether it is agent-env itself, and whether it is a
    plugin, which is any other distribution that declares an ``agent_env.*`` entry point."""

    version: str
    plugin: bool
    core: bool = False


def spellings(table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each canonical distribution name in ``[plugins]``, and the keys that spell it, in file order."""
    found: dict[str, list[str]] = {}
    for key in table:
        found.setdefault(canonicalize_name(key), []).append(key)
    return found


def where(key: str) -> str:
    """The table as written in TOML, a dotted name quoted so it does not read as nested."""
    return f"[{TABLE}.{toml_key(key)}]"


def toml_key(key: str) -> str:
    """A key as TOML writes it: bare where it can be, quoted otherwise."""
    return key if _BARE_KEY.fullmatch(key) else _quoted(key)


def _quoted(key: str) -> str:
    return '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'


def duplicate(name: str, keys: list[str]) -> str:
    tables = ", ".join(where(k) for k in keys)
    return f"config.toml sets {name}'s settings more than once, as {tables}; keep one"


def installed() -> dict[str, Installed]:
    """Every installed distribution, by canonical name. One whose metadata cannot be read is left out."""
    found: dict[str, Installed] = {}
    with suppress(Exception):
        for dist in distributions():
            try:
                name = dist.name
                if name:
                    core = canonicalize_name(name) == config_loader.DISTRIBUTION
                    plugin = not core and any(ep.group.startswith("agent_env.") for ep in dist.entry_points)
                    found.setdefault(canonicalize_name(name), Installed(dist.version or "", plugin, core))
            except Exception:
                continue
    return found


def settings(name: str, *, config: Optional[runtime.Config] = None) -> dict[str, Any]:
    """The plugin ``name``'s own settings: its ``[plugins.<name>]`` table, ``{}`` when there is none.

    ``name`` is the plugin's distribution name. Keys are matched by canonical name (PEP 503), so
    ``[plugins.agentenv_browser]`` is ``agentenv-browser``'s table. ``env:`` and ``secret:``
    references are resolved, as in every table agent-env reads. Reads ``config``'s document
    (default: the process Config) on every call, so call it where the value is used, not at import.
    Raises ``agent_env.config.ConfigError`` when ``[plugins]`` or the table is not a table, when two
    keys name the same distribution, or when a reference without a default cannot be resolved.
    """
    config = config or runtime.get_config()
    wanted = canonicalize_name(name)
    keys = spellings(config.section(TABLE)).get(wanted, [])
    if not keys:
        return {}
    if len(keys) > 1:
        raise ConfigError(duplicate(wanted, keys))
    try:
        return config_loader.interpolate(
            config.section(TABLE, keys[0]), secret_resolver=lambda key: config.get_secret_store().get(key)
        )
    except ConfigError as e:
        raise ConfigError(f"{where(keys[0])}: {e}") from e
