"""Store error types shared across the persistence layer."""

# Re-exported for back-compat; ConfigError now lives with the config surface.
from agent_env.config.errors import ConfigError  # noqa: F401


class NotFoundError(Exception):
    """Raised when an entity is not found in the store."""

    pass


class ObjectAlreadyExistsError(Exception):
    """Raised when attempting to create an object that already exists."""

    pass


class ObjectNotFoundError(NotFoundError, FileNotFoundError):
    """Raised when an object store read addresses no object, on every backend. It is a
    FileNotFoundError too, so filesystem-style handlers keep catching it."""


class GrantUnavailableError(RuntimeError):
    """Raised when an object store cannot issue a transfer grant for the requested lifetime."""


class ConcurrentModificationError(Exception):
    """Raised when an update fails due to a concurrent modification."""

    pass


