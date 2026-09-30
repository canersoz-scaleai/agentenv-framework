"""``ExplorerPlugin`` — a class wrapping an ``APIRouter`` so an installed package can add routes
to the explorer, through an ``agent_env.explorer_plugins`` entry point or ``[explorer.plugins]``.

Plugins mount *before* the core routers (see ``create_app``), so a plugin may add a
literal path under a core prefix (e.g. ``/api/v1/agents/trajectory-url``) that the core
``/{entity_id}`` catch-all would otherwise shadow. ``[explorer.plugins].impls`` is a flat list of
``module:Class`` strings, so this seam contributes routes; a plugin reads its own settings from
``[plugins.<distribution name>]`` with ``agent_env.plugins.settings``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from agent_env.plugins import _registration, _report

if TYPE_CHECKING:
    from fastapi import APIRouter
    from agent_env.config.runtime import Config


class ExplorerPlugin(ABC):
    """A named bundle of routes contributed to the explorer by an installed package."""

    type: ClassVar[str] = "explorer_plugin"

    @classmethod
    def from_config(cls) -> Self:
        # [explorer.plugins].impls is a flat list of 'module:Class' strings, so a plugin is
        # constructed with no arguments and reads its settings with agent_env.plugins.settings.
        return cls()

    @property
    @abstractmethod
    def router(self) -> "APIRouter":
        """The router to mount. Its prefix should be namespaced under /api/v1."""


def load_plugins(*, source: Config | None = None) -> list[ExplorerPlugin]:
    """Instantiate every ``agent_env.explorer_plugins`` entry point, then every
    ``[explorer.plugins] impls`` entry, raising on a missing ``type``, a duplicate, or a bad spec.
    An impl whose ``type`` a plugin already registered replaces it, with a warning. A plugin that
    fails to construct is skipped and recorded, like one that fails to load; an impl raises.
    ``router`` is left unread: ``create_app`` reads it once, to mount it."""
    from agent_env.config import runtime
    from agent_env.config import ConfigError, load_impl

    config = source or runtime.get_config()
    section = config.section("explorer", "plugins")
    impls = section.get("impls", [])
    if not isinstance(impls, list):
        raise ConfigError(
            f"[explorer.plugins] impls must be a list of 'module:Class' strings, got {type(impls).__name__}"
        )

    classes: dict[str, type[ExplorerPlugin]] = {}
    validate = _registration.typed_validator(ExplorerPlugin)
    from_plugins = _registration.merge(classes, _registration.EXPLORER_PLUGINS, validate, source=config)
    # Placeholders keep each plugin's mount position while impls are constructed in list order.
    built: dict[str, ExplorerPlugin | None] = dict.fromkeys(classes)
    seen: dict[str, str] = {}
    for impl in impls:
        if not isinstance(impl, str):
            raise ConfigError(
                f"[explorer.plugins] impl must be a 'module:Class' string, got {type(impl).__name__}: {impl!r}"
            )
        cls = load_impl(impl, ExplorerPlugin)
        if cls.type == ExplorerPlugin.type:
            raise ConfigError(
                f"[explorer.plugins] impl {impl!r} does not define its own 'type' "
                f"(inherits the base default {ExplorerPlugin.type!r}); set a unique 'type' ClassVar"
            )
        if cls.type in seen:
            raise ConfigError(
                f"[explorer.plugins] impl {impl!r} type {cls.type!r} is already registered by {seen[cls.type]!r}"
            )
        if problem := _registration.unimplemented(cls, ExplorerPlugin):
            raise ConfigError(f"[explorer.plugins] impl {impl!r}: {problem}")
        seen[cls.type] = impl
        if from_plugins.refuse(cls.type, f"[explorer.plugins] impl {impl!r}"):
            continue
        from_plugins.release(cls.type, "explorer.plugins", impl, cls)
        classes[cls.type] = cls
        built[cls.type] = cls.from_config()
    for name in from_plugins:
        try:
            instance = classes[name].from_config()
        except (Exception, SystemExit) as exc:
            from_plugins.reject(name, _report.LOAD_FAILED, f"failed to construct: {exc!r}")
            del built[name]
            continue
        built[name] = instance
    return [instance for instance in built.values() if instance is not None]

