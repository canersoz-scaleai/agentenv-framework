"""What the Google Cloud store backends share (the ``gcp`` extra): the scope they request, and how
they call Google's IAM API."""

from __future__ import annotations

import ssl
from collections.abc import Iterator

import requests
from google.api_core.retry import Retry
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import MutualTLSChannelError, TransportError
from google.auth.transport.requests import AuthorizedSession, Request

from agent_env.config.errors import ConfigError

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_TIMEOUT_SECONDS = 10
_LOST_IN_TRANSIT = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)
# The TLS failures that mean the connection was cut, not that the other side refused it.
_CUT_TLS = (ssl.SSLEOFError, ssl.SSLZeroReturnError)


def dropped_connection(error: Exception) -> bool:
    """A google-auth request lost in transit, before or during the response. A TLS failure counts
    only when the handshake was cut short: a refused certificate would repeat."""
    cause = error.__cause__
    if not isinstance(error, TransportError) or not isinstance(cause, _LOST_IN_TRANSIT):
        return False
    tls = [e for e in _chain(cause) if isinstance(e, ssl.SSLError)]
    return not tls or any(isinstance(e, _CUT_TLS) for e in tls)


def _chain(error: BaseException) -> Iterator[BaseException]:
    """Every exception in the chain requests and urllib3 wrap a transport failure in."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        links = [getattr(current, "reason", None), *current.args, current.__cause__, current.__context__]
        pending += [link for link in links if isinstance(link, BaseException)]


RETRY = Retry(predicate=dropped_connection, initial=0.5, maximum=4.0, timeout=30.0)


class _Request(Request):
    """google-auth's request with a short default timeout. It stays a Request, with its session in
    view, so google-auth can mount its metadata-server mTLS adapter on it."""

    def __call__(self, url, method="GET", body=None, headers=None, timeout=_TIMEOUT_SECONDS, **kwargs):
        return super().__call__(url, method=method, body=body, headers=headers, timeout=timeout, **kwargs)


def pooled_request() -> Request:
    """A google-auth request over one pooled session. It adds no credentials of its own: google-auth
    refreshes credentials over it, and those requests may go to non-Google token endpoints.
    google.auth.iam calls the mTLS endpoint where client certificates are configured, so the session
    presents them, as google-auth's own IAM calls do."""
    session = AuthorizedSession(AnonymousCredentials(), refresh_status_codes=())
    try:
        session.configure_mtls_channel()
    except MutualTLSChannelError as e:
        raise ConfigError(f"Cannot set up mutual TLS for Google's IAM API: {e}") from e
    return _Request(session)
