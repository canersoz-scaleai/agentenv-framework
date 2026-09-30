"""On-disk state directories for the local (no-infra) stores."""

from __future__ import annotations

from pathlib import Path

from agent_env.config.errors import ConfigError


def ensure_state_dir(path: Path) -> None:
    """Create ``path`` if missing. A directory this creates is private to the user (``0o700``)
    and gets a ``.gitignore`` of ``*`` so generated state is never committed; a directory that
    already existed is left as is. A directory that cannot be created is a ``ConfigError``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            if path.is_dir():
                return
            raise
        (path / ".gitignore").write_text("*\n")
    except OSError as e:
        raise ConfigError(
            f"cannot create the local store directory {path} ({e}); set XDG_STATE_HOME to a "
            "writable directory, or point the store somewhere writable in config.toml"
        ) from e
