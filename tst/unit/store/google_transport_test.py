"""The Google backends' IAM calls, with real google-auth credentials over a stubbed HTTP adapter:
the source credentials refresh without anyone's bearer token, and only the IAM call carries theirs."""

from __future__ import annotations

import json
import ssl
import sys
from datetime import UTC, datetime, timedelta

import google.auth
import pytest
import requests
import urllib3.exceptions
from google.auth import iam
from google.auth.exceptions import MutualTLSChannelError, TransportError
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import credentials as oauth2_credentials

from agent_env.config.errors import ConfigError
from agent_env.store import _google
from agent_env.store.image_store.google_credentials import GoogleAccessTokenCredentials
from agent_env.store.object_store.gcs_object_store import _IamSigner

ACCOUNT = "registry@example-project.iam.gserviceaccount.com"


@pytest.fixture
def sent(monkeypatch) -> list[requests.PreparedRequest]:
    """Every request google-auth sends, answered as the token endpoint and IAM would."""
    seen: list[requests.PreparedRequest] = []

    def send(adapter, request, **kwargs):
        seen.append(request)
        request.timeout = kwargs.get("timeout")
        if "oauth2" in request.url and request.url.endswith("/token"):
            body = {"access_token": f"source-{len(seen)}", "expires_in": 3600, "token_type": "Bearer"}
        elif request.url.endswith(":generateAccessToken") and _refuse_minting:
            return _response(request, 401, {"error": {"code": 401, "status": "UNAUTHENTICATED"}})
        elif request.url.endswith(":generateAccessToken"):
            expire = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            body = {"accessToken": "registry-token", "expireTime": expire}
        elif request.url.endswith(":signBlob"):
            body = {"keyId": "k", "signedBlob": "c2lnbmVk"}
        else:
            pytest.fail(f"unexpected request to {request.url}")
        return _response(request, 200, body)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return seen


_refuse_minting = False


def _response(request, status: int, body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code, response._content, response.url = status, json.dumps(body).encode(), request.url
    response.headers["content-type"] = "application/json"
    response.request = request
    return response


def _user_login() -> oauth2_credentials.Credentials:
    return oauth2_credentials.Credentials(
        token=None, refresh_token="refresh", client_id="id", client_secret="secret",
        token_uri="https://oauth2.example.test/token",
    )


def _authorization(request: requests.PreparedRequest) -> str | None:
    return request.headers.get("authorization") or request.headers.get("Authorization")


def _issued_bearers(sent: list[requests.PreparedRequest]) -> set[str]:
    """The tokens the stub's token endpoint handed out, as bearer headers."""
    return {f"Bearer source-{i + 1}" for i, r in enumerate(sent) if r.url.endswith("/token")}


def test_minting_refreshes_the_source_without_a_bearer_token(monkeypatch, sent):
    source = _user_login()
    monkeypatch.setattr(google.auth, "default", lambda scopes: (source, None))
    auth = GoogleAccessTokenCredentials(ACCOUNT).mint("us-west1-docker.pkg.dev")
    assert auth.password == "registry-token"
    [token_request] = [r for r in sent if r.url.endswith("/token")]
    [iam_request] = [r for r in sent if r.url.endswith(":generateAccessToken")]
    assert _authorization(token_request) is None
    assert _authorization(iam_request) in _issued_bearers(sent)
    assert iam_request.timeout == 10


def test_an_iam_refusal_is_a_config_error_not_a_refresh_of_no_credentials(monkeypatch, sent):
    """The pooled session holds no credentials, so it must not try to refresh them on a 401."""
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_user_login(), None))
    monkeypatch.setattr(sys.modules[__name__], "_refuse_minting", True)
    with pytest.raises(ConfigError, match=f"Cannot mint an access token as {ACCOUNT}.*401"):
        GoogleAccessTokenCredentials(ACCOUNT).mint("us-west1-docker.pkg.dev")


def test_signing_refreshes_the_source_without_a_bearer_token(sent):
    source = _user_login()
    assert _IamSigner(source, ACCOUNT).sign_bytes(b"payload") == b"signed"
    [token_request] = [r for r in sent if r.url.endswith("/token")]
    [iam_request] = [r for r in sent if r.url.endswith(":signBlob")]
    assert _authorization(token_request) is None
    assert _authorization(iam_request) in _issued_bearers(sent)


def _wrapped(cause: BaseException) -> TransportError:
    """How google-auth's Request reports a requests failure."""
    error = TransportError(cause)
    error.__cause__ = cause
    return error


def _ssl(cls: type, reason: str) -> ssl.SSLError:
    """An ssl error as the ssl module raises it, with the OpenSSL reason it names."""
    error = cls(1, reason.lower().replace("_", " "))
    error.reason, error.library = reason, "SSL"
    return error


def _tls_error(reason: BaseException) -> requests.exceptions.SSLError:
    """How requests reports a TLS failure: urllib3's retry error around urllib3's SSLError."""
    return requests.exceptions.SSLError(
        urllib3.exceptions.MaxRetryError(None, "/", reason=urllib3.exceptions.SSLError(reason))
    )


def _through_a_proxy(reason: BaseException) -> requests.exceptions.ProxyError:
    """How requests reports a failure to reach a proxy, which carries the cause second."""
    return requests.exceptions.ProxyError(
        urllib3.exceptions.ProxyError("Unable to connect to proxy", urllib3.exceptions.SSLError(reason))
    )


@pytest.mark.parametrize(
    ("cause", "retried"),
    [
        (requests.ConnectionError("reset"), True),
        (requests.ReadTimeout("slow"), True),
        (requests.exceptions.ChunkedEncodingError("IncompleteRead(20 bytes read, 187 more expected)"), True),
        (_tls_error(_ssl(ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING")), True),
        (_tls_error(_ssl(ssl.SSLCertVerificationError, "CERTIFICATE_VERIFY_FAILED")), False),
        (_tls_error(_ssl(ssl.SSLError, "TLSV13_ALERT_CERTIFICATE_REQUIRED")), False),
        (_through_a_proxy(_ssl(ssl.SSLCertVerificationError, "CERTIFICATE_VERIFY_FAILED")), False),
        (requests.HTTPError("403"), False),
    ],
    ids=[
        "reset", "timeout", "cut-mid-response", "cut-mid-handshake",
        "certificate-rejected", "client-certificate-refused", "certificate-rejected-via-proxy", "not-transport",
    ],
)
def test_what_counts_as_a_dropped_connection(cause, retried):
    assert _google.dropped_connection(_wrapped(cause)) is retried


def test_mutual_tls_that_cannot_be_set_up_is_a_config_error(monkeypatch):
    def refuse(self):
        raise MutualTLSChannelError("No module named 'OpenSSL'")

    monkeypatch.setattr(AuthorizedSession, "configure_mtls_channel", refuse)
    with pytest.raises(ConfigError, match="mutual TLS.*OpenSSL"):
        _google.pooled_request()


def test_a_signer_iam_never_answers_is_a_transport_error_naming_the_account(monkeypatch):
    monkeypatch.setattr(_google, "RETRY", _google.RETRY.with_delay(initial=0.01, maximum=0.02).with_timeout(0.05))

    def drop(self, message):
        raise _wrapped(requests.ConnectionError("reset"))

    monkeypatch.setattr(iam.Signer, "sign", drop)
    with pytest.raises(TransportError, match=f"IAM signed nothing as {ACCOUNT} within the retry window.*reset"):
        _IamSigner(_user_login(), ACCOUNT).sign_bytes(b"payload")
