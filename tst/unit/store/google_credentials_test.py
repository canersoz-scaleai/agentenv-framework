"""GoogleAccessTokenCredentials against stand-ins for google-auth's ADC and impersonation."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import google.auth
import pytest
import requests
from google.api_core.retry import Retry
from google.auth import impersonated_credentials
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError

from agent_env.config.errors import ConfigError
from agent_env.store import RegistryAuth, _google
from agent_env.store.image_store import google_credentials
from agent_env.store.image_store.google_credentials import GoogleAccessTokenCredentials
from agent_env.store.image_store.image_store import OciRegistryImageStore

ACCOUNT = "registry@example-project.iam.gserviceaccount.com"
HOST = "us-west1-docker.pkg.dev"


class _Impersonated:
    """Stands in for impersonated_credentials.Credentials: each refresh mints a new token."""

    made: list[dict] = []
    scopes: list = []
    lifetime = timedelta(hours=1)
    failures: list[Exception] = []
    refresh_seconds = 0.0
    requests: list = []

    def __init__(self, **kwargs) -> None:
        _Impersonated.made.append(kwargs)
        self.token = None
        self.expiry = None
        self.refreshes = 0

    def refresh(self, request) -> None:
        _Impersonated.requests.append(request)
        time.sleep(_Impersonated.refresh_seconds)
        if _Impersonated.failures:
            raise _Impersonated.failures.pop(0)
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + _Impersonated.lifetime


def _fast(retry: Retry, timeout: float = 1) -> Retry:
    """The production retry policy with short waits, so a test exercises its predicate."""
    return retry.with_delay(initial=0.01, maximum=0.02).with_timeout(timeout)


def _dropped() -> TransportError:
    error = TransportError(requests.ConnectionError("reset"))
    error.__cause__ = requests.ConnectionError("reset")
    return error


@pytest.fixture
def adc(monkeypatch):
    source = object()
    monkeypatch.setattr(google.auth, "default", lambda scopes: _Impersonated.scopes.append(scopes) or (source, "adc-project"))
    monkeypatch.setattr(impersonated_credentials, "Credentials", _Impersonated)
    monkeypatch.setattr(_google, "pooled_request", lambda: "pooled")
    monkeypatch.setattr(google_credentials, "_MINT_RETRY", _fast(google_credentials._MINT_RETRY))
    _Impersonated.scopes = []
    _Impersonated.made, _Impersonated.failures, _Impersonated.requests = [], [], []
    _Impersonated.lifetime, _Impersonated.refresh_seconds = timedelta(hours=1), 0.0
    return source


@pytest.mark.parametrize("account", ["", "  "])
def test_a_service_account_is_required(account):
    with pytest.raises(ConfigError, match="needs a service_account"):
        GoogleAccessTokenCredentials.from_config(service_account=account)


def test_building_the_credentials_looks_up_no_identity(monkeypatch):
    monkeypatch.setattr(google.auth, "default", lambda scopes: pytest.fail("looked up ADC while building"))
    GoogleAccessTokenCredentials.from_config(service_account=ACCOUNT)


def test_mint_impersonates_the_account_from_adc(adc):
    auth = GoogleAccessTokenCredentials(ACCOUNT).mint("US-WEST1-docker.pkg.dev")
    assert auth == RegistryAuth(registry=HOST, username="oauth2accesstoken", password="token-1")
    [made] = _Impersonated.made
    assert made == {"source_credentials": adc, "target_principal": ACCOUNT, "target_scopes": _google.SCOPES}
    assert _Impersonated.scopes == [_google.SCOPES]
    assert _Impersonated.requests == ["pooled"]


def test_a_token_is_reused_while_most_of_its_life_is_left(adc):
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    assert credentials.mint(HOST).password == credentials.mint(HOST).password == "token-1"
    assert len(_Impersonated.made) == 1


@pytest.mark.parametrize("lifetime", [timedelta(minutes=44), timedelta(seconds=-1)], ids=["nearly-spent", "expired"])
def test_a_token_without_enough_life_left_is_replaced(adc, lifetime):
    """An image build mints at its start and pushes at its end, up to half an hour later."""
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    _Impersonated.lifetime = lifetime
    credentials.mint(HOST)
    _Impersonated.lifetime = timedelta(hours=1)
    assert credentials.mint(HOST).password == "token-2"


def test_concurrent_mints_refresh_once(adc):
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    _Impersonated.refresh_seconds = 0.05
    start = threading.Barrier(8)
    passwords: list[str] = []

    def mint() -> None:
        start.wait()
        passwords.append(credentials.mint(HOST).password)

    threads = [threading.Thread(target=mint) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert passwords == ["token-1"] * 8
    assert len(_Impersonated.made) == 1 and len(_Impersonated.requests) == 1


def _iam_refusal(code: int) -> RefreshError:
    return RefreshError("Unable to acquire impersonated credentials", json.dumps({"error": {"code": code, "status": "X"}}))


def test_a_refused_mint_is_a_config_error_and_the_next_one_looks_up_credentials_again(adc):
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    _Impersonated.failures = [_iam_refusal(403)]
    with pytest.raises(ConfigError, match=f"Cannot mint an access token as {ACCOUNT}.*403"):
        credentials.mint(HOST)
    assert credentials.mint(HOST).password == "token-1"
    assert len(_Impersonated.made) == 2 and len(_Impersonated.requests) == 2


@pytest.mark.parametrize(
    "refusal",
    [
        *[_iam_refusal(code) for code in (429, 500, 502, 503, 504)],
        RefreshError("Unable to acquire impersonated credentials", "<html><title>502 Bad Gateway</title></html>"),
    ],
    ids=["429", "500", "502", "503", "504", "front-end-page"],
)
def test_iam_declining_for_now_is_retried(adc, refusal):
    _Impersonated.failures = [refusal]
    assert GoogleAccessTokenCredentials(ACCOUNT).mint(HOST).password == "token-1"


@pytest.mark.parametrize(
    "refusal",
    [
        RefreshError("invalid_grant: Bad Request", {"error": "invalid_grant", "error_description": "Bad Request"}),
        RefreshError("Failed to retrieve the metadata server's token"),
        RefreshError("Unable to acquire impersonated credentials", json.dumps({"error": "unexpected"})),
    ],
    ids=["expired-login", "no-body", "no-status"],
)
def test_other_refusals_are_final(adc, refusal):
    _Impersonated.failures = [refusal]
    with pytest.raises(ConfigError, match=f"Cannot mint an access token as {ACCOUNT}"):
        GoogleAccessTokenCredentials(ACCOUNT).mint(HOST)
    assert len(_Impersonated.requests) == 1


def test_the_account_name_is_used_without_surrounding_whitespace(adc):
    GoogleAccessTokenCredentials(f" {ACCOUNT}\n").mint(HOST)
    assert _Impersonated.made[0]["target_principal"] == ACCOUNT


@pytest.mark.parametrize(
    ("failure", "raised"), [(_iam_refusal(403), ConfigError), (_dropped(), TransportError)], ids=["refused", "unreachable"]
)
def test_callers_queued_behind_a_failed_mint_get_its_failure_without_retrying(adc, monkeypatch, failure, raised):
    """Otherwise each queued caller waits out a retry window of its own, one after another."""
    monkeypatch.setattr(google_credentials, "_MINT_RETRY", _fast(google_credentials._MINT_RETRY, timeout=0.2))
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    _Impersonated.refresh_seconds = 0.05
    _Impersonated.failures = [failure] * 1000
    start = threading.Barrier(4)
    errors: list[Exception] = []

    def mint() -> None:
        start.wait()
        try:
            credentials.mint(HOST)
        except raised as e:
            errors.append(e)

    threads = [threading.Thread(target=mint) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(errors) == 4
    assert len({id(e) for e in errors}) == 4, "each caller gets an exception of its own"
    attempts = 1000 - len(_Impersonated.failures)
    assert attempts < (4 if raised is ConfigError else 12)


def test_callers_arriving_just_after_an_outage_share_its_failure(adc, monkeypatch):
    """A caller queued for a worker thread, not on the lock, arrives after the failure was recorded."""
    monkeypatch.setattr(google_credentials, "_MINT_RETRY", _fast(google_credentials._MINT_RETRY, timeout=0.05))
    clock = [1000.0]
    monkeypatch.setattr(google_credentials, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    credentials = GoogleAccessTokenCredentials(ACCOUNT)
    _Impersonated.failures = [_dropped()] * 1000
    with pytest.raises(TransportError):
        credentials.mint(HOST)
    attempts = len(_Impersonated.requests)
    clock[0] += 5
    with pytest.raises(TransportError, match="within the retry window"):
        credentials.mint(HOST)
    assert len(_Impersonated.requests) == attempts
    clock[0] += 6
    _Impersonated.failures = []
    assert credentials.mint(HOST).password == "token-1"


def test_an_unreachable_iam_names_the_account(adc, monkeypatch):
    monkeypatch.setattr(google_credentials, "_MINT_RETRY", _fast(google_credentials._MINT_RETRY, timeout=0.05))
    _Impersonated.failures = [_dropped()] * 100
    with pytest.raises(TransportError, match=f"IAM minted no access token as {ACCOUNT} within the retry window.*reset"):
        GoogleAccessTokenCredentials(ACCOUNT).mint(HOST)


def test_a_dropped_connection_is_retried(adc):
    _Impersonated.failures = [_dropped()]
    assert GoogleAccessTokenCredentials(ACCOUNT).mint(HOST).password == "token-1"


@pytest.mark.parametrize(
    "error",
    [DefaultCredentialsError("Your default credentials were not found."), IsADirectoryError(21, "Is a directory")],
    ids=["none", "a-directory"],
)
def test_missing_adc_is_a_config_error(monkeypatch, error):
    def no_adc(scopes):
        raise error

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(ConfigError, match=f"No Google credentials to impersonate {ACCOUNT} with"):
        GoogleAccessTokenCredentials(ACCOUNT).mint(HOST)


@pytest.fixture
def store(adc) -> OciRegistryImageStore:
    return OciRegistryImageStore.from_config(
        registry_host=HOST,
        repository_prefix="example-project/agentenv",
        credentials={
            "impl": "agent_env.store.image_store.google_credentials:GoogleAccessTokenCredentials",
            "service_account": ACCOUNT,
        },
    )


def test_the_image_store_mints_only_for_its_own_registry(store):
    assert store.auth(store.image_ref("app", "v1")).password == "token-1"
    assert store.auth("ghcr.io/example/app:v1") is None
