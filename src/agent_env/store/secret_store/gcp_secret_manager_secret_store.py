"""Google Cloud Secret Manager secret store backend (the ``gcp`` extra)."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator, ItemsView, KeysView, Mapping, ValuesView
from typing import Any, Optional

import google.auth
import yaml
from google.api_core.retry import Retry, if_transient_error
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import secretmanager

from agent_env.config.errors import ConfigError
from agent_env.store import _google
from agent_env.store.secret_store.secret_store import SecretStore

logger = logging.getLogger(__name__)

# A rotation reaches every process within five minutes, at one access per process each time.
_DEFAULT_TTL_SECONDS = 300.0

# Floor between forced refresh() calls; doubles as the retry backoff after a failed re-fetch.
_DEFAULT_MIN_REFRESH_INTERVAL = 10.0

# Re-fetches run on the read path: fail in seconds, not after the client's minute of retries.
_RETRY = Retry(predicate=if_transient_error, initial=0.5, maximum=4.0, timeout=15.0)
_TIMEOUT_SECONDS = 10.0

# The client library logs request and response payloads at DEBUG; the payload is the bundle.
# The transport's logger is named as well as its package because the SDK's logging scope
# (GOOGLE_SDK_PYTHON_LOGGING_SCOPE) can set DEBUG on it directly, which its package's level
# would not override.
_PAYLOAD_LOGGERS = (
    "google.cloud.secretmanager_v1",
    "google.cloud.secretmanager_v1.services.secret_manager_service.transports.rest",
    "google.api_core",
    "google.auth",
    "urllib3",
)


class _BundleView(Mapping):
    """Live read-only view of the store's bundle: consumers that memoize ``_load()``'s
    return for the process lifetime (``Config._get_secret()`` does) still observe TTL
    re-fetches. Accessors bind to one snapshot each; not json/pickle/deepcopy-serializable.
    """

    def __init__(self, store: "GcpSecretManagerSecretStore") -> None:
        self._store = store

    def __getitem__(self, key: str) -> Any:
        return self._store._current()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store._current())

    def __len__(self) -> int:
        return len(self._store._current())

    def get(self, key: str, default: Any = None) -> Any:
        return self._store._current().get(key, default)

    def keys(self) -> KeysView:
        return self._store._current().keys()

    def items(self) -> ItemsView:
        return self._store._current().items()

    def values(self) -> ValuesView:
        return self._store._current().values()

    def __eq__(self, other: Any) -> bool:
        return self._store._current() == other

    def __repr__(self) -> str:  # never render values
        return f"<_BundleView of {self._store._secret_name!r}>"


#: Shared by every caller, because the loggers being changed are process-global.
_suppress_guard = threading.Lock()
_suppress_depth = 0
_suppress_saved: list = []


@contextlib.contextmanager
def _suppress_payload_logging() -> Iterator[None]:
    """Keep a Secret Manager payload out of the logs for the duration of a call.

    Raising the level only around the fetch does not fight the caller's logging config or
    silence the library anywhere else, and the previous levels are restored on the error
    path too. Reference-counted, because fetches from different stores overlap: only the
    outermost context captured the levels worth restoring, so only it restores them.
    """
    global _suppress_depth, _suppress_saved

    loggers = [logging.getLogger(name) for name in _PAYLOAD_LOGGERS]
    with _suppress_guard:
        if _suppress_depth == 0:
            _suppress_saved = [(lg, lg.level) for lg in loggers]
            for lg in loggers:
                if lg.level < logging.INFO:
                    lg.setLevel(logging.INFO)
        _suppress_depth += 1
    try:
        yield
    finally:
        with _suppress_guard:
            _suppress_depth -= 1
            if _suppress_depth == 0:
                for lg, level in _suppress_saved:
                    lg.setLevel(level)
                _suppress_saved = []


class GcpSecretManagerSecretStore(SecretStore):
    """Reads secrets from one Secret Manager secret version holding a YAML/JSON mapping (a
    combined bundle secret). Authenticates through Application Default Credentials.

    The bundle is cached (thread-safe) and re-fetched once older than ``ttl_seconds``
    (0 = cache for the process lifetime); ``refresh()`` forces a re-fetch, rate-limited
    to one per ``min_refresh_interval``. A failed TTL re-fetch serves the last-good
    bundle; only the first-ever fetch and an explicit ``refresh()`` raise.
    """

    def __init__(
        self,
        secret_name: str,
        project: str | None = None,
        version: str = "latest",
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_refresh_interval: float = _DEFAULT_MIN_REFRESH_INTERVAL,
    ) -> None:
        self._secret_name = secret_name
        self._project = project
        self._version = version
        self._ttl_seconds = float(ttl_seconds)
        self._min_refresh_interval = float(min_refresh_interval)
        self._client: secretmanager.SecretManagerServiceClient | None = None
        self._values: Optional[dict] = None
        self._fetched_at: Optional[float] = None
        self._last_forced_refresh_at: Optional[float] = None
        self._last_failed_fetch_at: Optional[float] = None
        self._lock = threading.Lock()
        self._view = _BundleView(self)

    @classmethod
    def from_config(
        cls,
        *,
        secret_name: str,
        project: str | None = None,
        version: str = "latest",
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_refresh_interval: float = _DEFAULT_MIN_REFRESH_INTERVAL,
    ) -> GcpSecretManagerSecretStore:
        """Build from a ``[stores.secret.config]`` table (literal values only). ``project``
        defaults to the one Application Default Credentials resolve; ``version`` to the latest."""
        return cls(
            secret_name,
            project=project,
            version=version,
            ttl_seconds=ttl_seconds,
            min_refresh_interval=min_refresh_interval,
        )

    def _fetch(self) -> dict:
        """Fetch and parse the bundle. Caller holds ``self._lock``."""
        client = self._connect()
        name = client.secret_version_path(self._project, self._secret_name, self._version)
        with _suppress_payload_logging():
            response = client.access_secret_version(name=name, retry=_RETRY, timeout=_TIMEOUT_SECONDS)
            data = response.payload.data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                f"Secret Manager secret {self._secret_name!r} holds binary data; expected a YAML/JSON mapping"
            ) from None
        try:
            loaded = yaml.load(text, Loader=yaml.BaseLoader) or {}
        except yaml.YAMLError as exc:
            # YAML error marks embed the raw secret line: re-raise sanitized, context dropped.
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            raise ValueError(
                f"Secret Manager secret {self._secret_name!r} is not valid YAML/JSON{where}"
            ) from None
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Secret Manager secret {self._secret_name!r} must be a YAML/JSON "
                f"mapping, got {type(loaded).__name__}"
            )
        return loaded

    def _connect(self) -> secretmanager.SecretManagerServiceClient:
        """The client, built on first use. REST rather than gRPC: a process that forks after a
        fetch hands its client to the child, and a gRPC channel does not survive that. A REST
        connection the two then share fails once, and the retry opens a new one."""
        if self._client is None:
            try:
                credentials, default_project = google.auth.default(scopes=_google.SCOPES)
            except (DefaultCredentialsError, OSError) as e:
                raise ConfigError(
                    f"No Google credentials for Secret Manager secret {self._secret_name!r}: {e}"
                ) from e
            self._project = self._project or default_project
            if not self._project:
                raise ConfigError(
                    f"Secret Manager secret {self._secret_name!r} needs a project: set one, "
                    "or GOOGLE_CLOUD_PROJECT for the credentials"
                )
            self._client = secretmanager.SecretManagerServiceClient(credentials=credentials, transport="rest")
        return self._client

    def _needs_fetch(self) -> bool:
        if self._values is None or self._fetched_at is None:
            return True
        if self._ttl_seconds <= 0:
            return False  # TTL disabled: cache for the process lifetime
        if (time.monotonic() - self._fetched_at) < self._ttl_seconds:
            return False
        # Outage backoff: serve stale without re-probing Secret Manager on every read.
        if self._last_failed_fetch_at is not None and (
            time.monotonic() - self._last_failed_fetch_at
        ) < self._min_refresh_interval:
            return False
        return True

    def _current(self) -> dict:
        """Freshest available bundle. Only the first-ever load blocks: a reader that
        finds a re-fetch already in flight serves the stale bundle immediately."""
        if self._needs_fetch():
            first_load = self._values is None
            acquired = self._lock.acquire(blocking=first_load)
            if acquired:
                try:
                    if self._needs_fetch():
                        self._refetch_locked()
                finally:
                    self._lock.release()
        assert self._values is not None  # first fetch either populated it or raised
        return self._values

    def _refetch_locked(self) -> None:
        """One TTL re-fetch attempt. Caller holds ``self._lock``."""
        try:
            values = self._fetch()
        except ValueError:
            if self._values is None:
                raise
            self._last_failed_fetch_at = time.monotonic()
            # Config rot, not an outage — escalate so monitoring can tell them apart.
            logger.error(
                "Secret Manager secret %r was rotated to invalid content; serving the cached "
                "bundle (age %.0fs) until the secret is fixed.",
                self._secret_name,
                time.monotonic() - (self._fetched_at or 0.0),
                exc_info=True,
            )
        except Exception:
            if self._values is None:
                raise
            self._last_failed_fetch_at = time.monotonic()
            logger.warning(
                "Re-fetching Secret Manager secret %r failed; serving the cached bundle "
                "(age %.0fs). Will retry in %.0fs.",
                self._secret_name,
                time.monotonic() - (self._fetched_at or 0.0),
                self._min_refresh_interval,
                exc_info=True,
            )
        else:
            # Post-fetch clock read: the TTL measures from arrival, not request send.
            self._values = values
            self._fetched_at = time.monotonic()
            self._last_failed_fetch_at = None

    def _load(self) -> Mapping:
        """The bundle as a live view. Primes eagerly so a first-ever fetch failure
        raises at the call site."""
        self._current()
        return self._view

    def refresh(self) -> Mapping:
        """Force a re-fetch (rate-limited to one per ``min_refresh_interval``) so a key
        added to the secret resolves on the first request that references it. Raises on
        failure — the caller asked for fresh data — leaving the cached bundle intact."""
        with self._lock:
            now = time.monotonic()
            within_cooldown = (
                self._last_forced_refresh_at is not None
                and (now - self._last_forced_refresh_at) < self._min_refresh_interval
            )
            if self._values is not None and within_cooldown:
                return self._view
            # Cooldown and TTL are stamped only after a successful fetch, off a post-fetch
            # clock read: a failed attempt must not suppress its own retry, and a fetch
            # slower than the cooldown must not hand back an already-expired one.
            try:
                values = self._fetch()
            except Exception:
                if self._values is not None:
                    self._last_failed_fetch_at = time.monotonic()  # arm the read-path backoff
                raise
            fetched_at = time.monotonic()
            self._values = values
            self._last_forced_refresh_at = fetched_at
            self._fetched_at = fetched_at
            self._last_failed_fetch_at = None
            return self._view

    def get(self, name: str) -> str | None:
        value = self._current().get(name)
        return None if value is None else str(value)
