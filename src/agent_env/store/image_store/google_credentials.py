"""Artifact Registry credentials for OciRegistryImageStore (the ``gcp`` extra)."""

from __future__ import annotations

import copy
import json
import threading
import time
from datetime import UTC, datetime, timedelta

import google.auth
from google.auth import impersonated_credentials
from google.api_core.exceptions import RetryError
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError

from agent_env.config.errors import ConfigError
from agent_env.store import _google
from agent_env.store.image_store.oci_registry_credentials import (
    OciRegistryCredentials,
    RegistryAuth,
    normalize_registry_host,
)

# A token must outlast what it is handed to: an image build logs in with it to push at its end.
_MIN_REMAINING = timedelta(minutes=45)
_USERNAME = "oauth2accesstoken"
_TRANSIENT_IAM_CODES = frozenset({429, 500, 502, 503, 504})
# A mint that found IAM unreachable spent a whole retry window; callers that come just after it
# share its failure rather than each spending another.
_OUTAGE_REUSE_SECONDS = 10.0


def _transient(error: Exception) -> bool:
    """A dropped connection, or IAM declining for now. google-auth reports an IAM refusal as a
    RefreshError carrying the response body: a 429 or 5xx, or a body that is no JSON at all, as a
    front end's error page is. A token endpoint's refusal carries a parsed mapping, and is final."""
    if _google.dropped_connection(error):
        return True
    if not isinstance(error, RefreshError) or len(error.args) < 2 or not isinstance(error.args[1], str):
        return False
    try:
        body = json.loads(error.args[1])
    except ValueError:
        return True
    status = body.get("error") if isinstance(body, dict) else None
    return isinstance(status, dict) and status.get("code") in _TRANSIENT_IAM_CODES


_MINT_RETRY = _google.RETRY.with_predicate(_transient)


class GoogleAccessTokenCredentials(OciRegistryCredentials):
    """Docker-login material for Artifact Registry: an access token of ``service_account``, which
    Application Default Credentials impersonate. The token is handed to sandboxes, so the account
    should hold nothing but write access to the registry's repository."""

    def __init__(self, service_account: str) -> None:
        if not service_account.strip():
            raise ConfigError("GoogleAccessTokenCredentials needs a service_account")
        self._service_account = service_account.strip()
        self._credentials: impersonated_credentials.Credentials | None = None
        self._request = None
        self._lock = threading.Lock()
        self._failure: tuple[float, Exception] | None = None

    def mint(self, host: str) -> RegistryAuth:
        arrived = time.monotonic()
        with self._lock:
            # A caller that queued behind a failed attempt, or arrives just after IAM proved
            # unreachable, gets its failure rather than another retry window.
            if self._failure is not None:
                failed_at, failure = self._failure
                outage = isinstance(failure, TransportError) and time.monotonic() - failed_at < _OUTAGE_REUSE_SECONDS
                if failed_at > arrived or outage:
                    raise copy.copy(failure).with_traceback(None) from failure
            try:
                token = self._token()
            except Exception as e:
                self._failure = (time.monotonic(), e)
                raise
            self._failure = None
        return RegistryAuth(registry=normalize_registry_host(host), username=_USERNAME, password=token)

    def _token(self) -> str:
        if self._credentials is None:
            self._credentials, self._request = self._impersonate()
        if _remaining(self._credentials) < _MIN_REMAINING:
            try:
                _MINT_RETRY(self._credentials.refresh)(self._request)
            except RefreshError as e:
                # Look the credentials up again next time, in case they were replaced.
                self._credentials = self._request = None
                raise ConfigError(f"Cannot mint an access token as {self._service_account}: {e}") from e
            except RetryError as e:
                raise TransportError(f"IAM minted no access token as {self._service_account} within the retry window: {e.cause}") from e
        return self._credentials.token

    def _impersonate(self):
        try:
            source, _ = google.auth.default(scopes=_google.SCOPES)
        except (DefaultCredentialsError, OSError) as e:
            raise ConfigError(f"No Google credentials to impersonate {self._service_account} with: {e}") from e
        credentials = impersonated_credentials.Credentials(
            source_credentials=source, target_principal=self._service_account, target_scopes=_google.SCOPES
        )
        return credentials, _google.pooled_request()


def _remaining(credentials) -> timedelta:
    if credentials.token is None:
        return timedelta(0)
    # google-auth keeps expiry as naive UTC.
    return credentials.expiry - datetime.now(UTC).replace(tzinfo=None)
