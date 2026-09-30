"""``agent_env.plugins.inventory``: every installed plugin's contributions, and what became of each.

Internal: the public names are re-exported from ``agent_env.plugins``. The registries are built on
a throwaway Config that reads the caller's document, so the Config in use keeps its registries and
its ``load_failures()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Optional

from agent_env.config import runtime
from agent_env.config import snapshot as config_snapshot
from agent_env.config.errors import ConfigError
from agent_env.plugins import _discovery, _registration, _report, _requirements

Status = Literal["active", "replaced", "failed", "skipped", "conflict", "blocked", "unloaded"]

# Statuses a group-level error does not override: each already says why the name is missing.
_SETTLED = frozenset({"failed", "skipped", "conflict"})


@dataclass(frozen=True)
class Diagnostic:
    """Why a group, the config or entry-point discovery failed."""

    code: str
    reason: str


@dataclass(frozen=True)
class Claimant:
    """Another entry point that claims the same name."""

    package: str
    version: str
    value: str


@dataclass(frozen=True)
class Replacement:
    """The config that registers a different class under a contribution's name."""

    file: Optional[Path]
    table: str
    impl: str


@dataclass(frozen=True)
class Contribution:
    """One entry point a distribution declares, and what became of it. ``code`` says why it has
    its status, and is set exactly when ``reason`` is."""

    group: str
    name: str
    value: str
    status: Status
    _: KW_ONLY
    code: Optional[str] = None
    reason: Optional[str] = None
    replaced_by: Optional[Replacement] = None
    conflicts_with: tuple[Claimant, ...] = ()


@dataclass(frozen=True)
class Distribution:
    """An installed distribution that declares entry points in the type groups."""

    name: str
    version: str
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class Inventory:
    """Every installed plugin distribution, and the error each group's real build would raise."""

    distributions: tuple[Distribution, ...]
    # With load=False, only what metadata proves: that no config file was found.
    group_errors: Mapping[str, Diagnostic]
    # Groups whose installed entry points could not be read, so none of their plugins is listed.
    discovery_errors: Mapping[str, Diagnostic]
    config_path: Optional[Path]
    config_error: Optional[Diagnostic]
    loaded: bool


for _public in (Claimant, Contribution, Diagnostic, Distribution, Inventory, Replacement):
    _public.__module__ = "agent_env.plugins"


def _groups() -> dict[str, tuple[Callable[[], Any], Callable[[runtime.Config], Any]]]:
    """Per group: its built-in names, and how the probe builds it."""
    # Imported here, not at the top: every registry imports agent_env.plugins, and so this module.
    from agent_env.artifact.registry import ARTIFACT_REGISTRY
    from agent_env.env import registry as env_registry
    from agent_env.explorer.plugin import load_plugins
    from agent_env.providers.env_providers import env_provider
    from agent_env.providers.sandbox_providers.sandbox_provider import _BUILTIN_SANDBOX_PROVIDERS
    from agent_env.providers.env_state.env_state_provider import _BUILTIN_STATE_PROVIDERS
    from agent_env.task_step import registry as task_step_registry

    return {
        _registration.ENVS: (env_registry._builtin_registry, lambda probe: probe.env_registry()),
        _registration.ARTIFACTS: (lambda: ARTIFACT_REGISTRY, lambda probe: probe.artifact_registry()),
        _registration.TASK_STEPS: (task_step_registry._builtin_registry, lambda probe: probe.task_step_registry()),
        _registration.SANDBOX_PROVIDERS: (lambda: _BUILTIN_SANDBOX_PROVIDERS, lambda probe: probe.sandbox_registry()),
        _registration.STATE_PROVIDERS: (lambda: _BUILTIN_STATE_PROVIDERS, lambda probe: probe.state_registry()),
        _registration.ENV_PROVIDERS: (env_provider._builtin_env_providers, lambda probe: probe.env_provider_registry()),
        _registration.EXPLORER_PLUGINS: (dict, lambda probe: load_plugins(source=probe)),
    }


def inventory(config: Optional[runtime.Config] = None, *, load: bool = True) -> Inventory:
    """Every installed distribution declaring entry points in the type groups, with a status per
    contribution.

    ``load=False`` reads installed metadata only: no plugin code runs, and a status that needs a
    build (``active``, ``replaced``, ``failed``) is reported as ``unloaded``. ``load=True`` imports
    the plugins, constructs the explorer plugins, and builds each group on a throwaway Config that
    reads ``config``'s document (default: the process Config). ``config`` keeps its registries and
    its ``load_failures()``; like any registry build, this pins its document first if nothing has
    yet. Importing or constructing a plugin can still have process-wide effects, such as a package
    that sets ``AGENT_ENV_CONFIG`` on import.
    """
    base = config or runtime.get_config()
    try:
        document = base._document()
        missing = None
    except ConfigError as exc:
        # No config file was found: every real build raises this before merging anything, so the
        # plugins are checked against an empty document and then blocked.
        document, missing = config_snapshot.Snapshot(path=None, _document={}), exc
    if missing is not None:
        config_error = Diagnostic(_report.CONFIG_NOT_FOUND, str(missing))
    elif document.error is not None:
        config_error = Diagnostic(_report.CONFIG_UNREADABLE, str(document.error))
    else:
        config_error = None
    # A malformed document is used as it is, so each build raises its parse error where a real one does.
    probe = _registration.InventoryProbe(_snapshot=document) if load else None
    group_errors: dict[str, Diagnostic] = {}
    discovery_errors: dict[str, Diagnostic] = {}
    by_distribution: dict[tuple[str, str], list[Contribution]] = {}
    groups = _groups()
    for group, (builtins_of, build) in groups.items():
        try:
            claims = _discovery.claims(group)
        except Exception as exc:
            discovery_errors[group] = Diagnostic(_report.ENTRY_POINTS_UNREADABLE, _discovery.discovery_error(exc))
            continue
        if not claims:
            continue
        builtins = frozenset(builtins_of())
        registrations = None
        if probe is not None:
            try:
                build(probe)
            except (Exception, SystemExit) as exc:
                group_errors[group] = Diagnostic(_build_error_code(exc, document), f"{type(exc).__name__}: {exc}")
            registrations = probe.registrations.get(group)
        if missing is not None:
            group_errors[group] = Diagnostic(_report.CONFIG_NOT_FOUND, f"{type(missing).__name__}: {missing}")
        for name, found in claims.items():
            for plugin, ep in found:
                contribution = _classify(group, name, plugin, ep, found, builtins, registrations, document.path)
                if group in group_errors and contribution.status not in _SETTLED:
                    error = group_errors[group]
                    contribution = Contribution(
                        group, name, plugin.value, "blocked", code=error.code, reason=error.reason
                    )
                by_distribution.setdefault(_discovery.identity(ep), []).append(contribution)
    order = list(groups)
    distributions = tuple(
        Distribution(dist, version, tuple(sorted(found, key=lambda c: (order.index(c.group), c.name))))
        for (dist, version), found in sorted(by_distribution.items(), key=lambda item: (item[0][0].lower(), item[0][1]))
    )
    return Inventory(
        distributions=distributions,
        group_errors=MappingProxyType(group_errors),
        discovery_errors=MappingProxyType(discovery_errors),
        config_path=document.path,
        config_error=config_error,
        loaded=load,
    )


def _classify(
    group: str,
    name: str,
    plugin: _discovery.Plugin,
    ep: Any,
    found: list[tuple[_discovery.Plugin, Any]],
    builtins: frozenset[str],
    registrations: Optional[_registration.Registrations],
    config_path: Optional[Path],
) -> Contribution:
    # Same precedence as merge: a built-in name is skipped for every claimant, before conflicts.
    if name in builtins:
        return Contribution(
            group, name, plugin.value, "skipped", code=_report.BUILTIN_NAME, reason="clashes with a built-in"
        )
    # Read from metadata, so it is reported without loading. A claim that cannot load claims nothing,
    # so it takes no part in a conflict, as in a build.
    if (unmet := _requirements.incompatibility(getattr(ep, "dist", None))) is not None:
        return Contribution(group, name, plugin.value, "failed", code=_report.INCOMPATIBLE_CORE, reason=unmet)
    found = _usable(found)
    if len(found) > 1:
        others = [(other, other_ep) for other, other_ep in found if other != plugin]
        alone = all((other.dist, other.version) == (plugin.dist, plugin.version) for other, _ in found)
        who = "; ".join(str(other) for other, _ in others)
        return Contribution(
            group, name, plugin.value, "conflict", code=_report.NAME_CONFLICT,
            reason=(f"this package declares the name more than once: {who}; report it to its author" if alone
                    else f"also registered by {who}; remove all but one with `agent-env plugin remove <package>`"),
            conflicts_with=tuple(_claimant(other, other_ep) for other, other_ep in others),
        )
    if registrations is None:
        return Contribution(
            group, name, plugin.value, "unloaded", code=_report.NOT_LOADED, reason="not imported (installed metadata only)"
        )
    if name in registrations.added:
        return Contribution(group, name, plugin.value, "active")
    if name in registrations.released:
        table, impl, replacement = registrations.released[name]
        if replacement is registrations.classes.get(name):
            return Contribution(group, name, plugin.value, "active")
        where = f"[{table}] of {config_path}" if config_path else f"[{table}]"
        return Contribution(
            group, name, plugin.value, "replaced", code=_report.REPLACED_BY_CONFIG,
            reason=f"config names {impl!r} in {where}", replaced_by=Replacement(config_path, table, impl),
        )
    if name in registrations.failures:
        reason = registrations.failures[name].removeprefix(f"registered by {plugin} but ")
        return Contribution(group, name, plugin.value, "failed", code=registrations.codes[name], reason=reason)
    # Unreachable while every registry records what it did; visible, not silent, if not.
    return Contribution(
        group, name, plugin.value, "unloaded", code=_report.STATUS_UNKNOWN, reason="its status could not be determined"
    )


def _usable(found: list[tuple[_discovery.Plugin, Any]]) -> list[tuple[_discovery.Plugin, Any]]:
    """The claims whose requirements admit this agent-env; the others cannot load."""
    return [(p, ep) for p, ep in found if _requirements.incompatibility(getattr(ep, "dist", None)) is None]


def _claimant(plugin: _discovery.Plugin, ep: Any) -> Claimant:
    package, version = _discovery.identity(ep)
    return Claimant(package, version, plugin.value)


def _build_error_code(exc: BaseException, document: config_snapshot.Snapshot) -> str:
    """The code for what a group's build raised."""
    if document.error is not None and exc is document.error:
        return _report.CONFIG_UNREADABLE
    if isinstance(exc, ConfigError):
        return _report.CONFIG_INVALID
    return _report.GROUP_BUILD_FAILED
