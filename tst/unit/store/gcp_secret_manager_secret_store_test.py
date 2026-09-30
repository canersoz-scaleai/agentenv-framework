"""GcpSecretManagerSecretStore against a stand-in Secret Manager client (no network)."""

from __future__ import annotations

import base64
import json
import logging
import traceback
from types import SimpleNamespace

import google.auth
import pytest
import requests
from google.api_core.exceptions import ServiceUnavailable
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import DefaultCredentialsError

from agent_env.config.errors import ConfigError
from agent_env.store.secret_store import gcp_secret_manager_secret_store as gsm
from agent_env.store.secret_store.gcp_secret_manager_secret_store import GcpSecretManagerSecretStore
from tst.store import secret_conformance

_BUNDLE = b'conformance_present: value-1\nconformance_present_empty: ""\n'
_REST_LOGGER = "google.cloud.secretmanager_v1.services.secret_manager_service.transports.rest"


class _Client:
    """Stands in for SecretManagerServiceClient: serves one payload, records the calls."""

    instances: list["_Client"] = []

    def __init__(self, *, credentials, transport) -> None:
        self.credentials, self.transport = credentials, transport
        self.payload: bytes | Exception = _BUNDLE
        self.accessed: list[str] = []
        self.deadlines: list[tuple] = []
        self.levels_during_access: dict[str, int] = {}
        _Client.instances.append(self)

    @staticmethod
    def secret_version_path(project, secret, secret_version) -> str:
        return f"projects/{project}/secrets/{secret}/versions/{secret_version}"

    def access_secret_version(self, *, name, retry, timeout):
        self.accessed.append(name)
        self.deadlines.append((retry, timeout))
        self.levels_during_access = {n: logging.getLogger(n).getEffectiveLevel() for n in gsm._PAYLOAD_LOGGERS}
        if isinstance(self.payload, Exception):
            raise self.payload
        return SimpleNamespace(payload=SimpleNamespace(data=self.payload))


@pytest.fixture
def adc(monkeypatch):
    _Client.instances = []
    monkeypatch.setattr(google.auth, "default", lambda scopes: ("adc-credentials", "adc-project"))
    monkeypatch.setattr(gsm.secretmanager, "SecretManagerServiceClient", _Client)


def _store(**config) -> GcpSecretManagerSecretStore:
    return GcpSecretManagerSecretStore.from_config(secret_name="bundle", **config)


@pytest.mark.parametrize("case", secret_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(adc, case):
    case(_store(project="p"))


def test_building_the_store_looks_up_no_credentials(monkeypatch):
    monkeypatch.setattr(google.auth, "default", lambda scopes: pytest.fail("looked up ADC while building"))
    _store()


def test_it_reads_the_named_version_over_rest_with_adc(adc):
    store = _store(project="p", version="7")
    assert store.get("conformance_present") == "value-1"
    [client] = _Client.instances
    assert (client.credentials, client.transport) == ("adc-credentials", "rest")
    assert client.accessed == ["projects/p/secrets/bundle/versions/7"]
    assert client.deadlines == [(gsm._RETRY, gsm._TIMEOUT_SECONDS)]


def test_the_project_defaults_to_the_credentials_one(adc):
    store = _store()
    store.get("conformance_present")
    assert _Client.instances[0].accessed == ["projects/adc-project/secrets/bundle/versions/latest"]


def test_no_project_anywhere_is_a_config_error(adc, monkeypatch):
    monkeypatch.setattr(google.auth, "default", lambda scopes: ("adc-credentials", None))
    with pytest.raises(ConfigError, match="needs a project"):
        _store().get("x")


def test_missing_adc_is_a_config_error(monkeypatch):
    def no_adc(scopes):
        raise DefaultCredentialsError("Your default credentials were not found.")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(ConfigError, match="No Google credentials for Secret Manager secret 'bundle'"):
        _store().get("x")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\xfe", "holds binary data"),
        (b"token: [hunter2", "is not valid YAML/JSON at line 1"),
        (b"- hunter2", "must be a YAML/JSON mapping, got list"),
    ],
    ids=["binary", "invalid", "not-a-mapping"],
)
def test_a_bad_payload_names_the_secret_and_never_its_content(adc, payload, message):
    store = _store(project="p")
    store._connect().payload = payload
    with pytest.raises(ValueError, match=message) as raised:
        store.get("token")
    assert "hunter2" not in "".join(traceback.format_exception(raised.value))


def test_a_failed_re_fetch_serves_the_last_good_bundle(adc, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(gsm.time, "monotonic", lambda: clock[0])
    store = _store(project="p", ttl_seconds=60)
    assert store.get("conformance_present") == "value-1"
    store._client.payload = ServiceUnavailable("Secret Manager is down")
    clock[0] += 61
    assert store.get("conformance_present") == "value-1"
    assert len(store._client.accessed) == 2
    with pytest.raises(ServiceUnavailable):
        store.refresh()


def test_the_payload_loggers_are_quiet_while_it_fetches(adc):
    for name in gsm._PAYLOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)
    try:
        store = _store(project="p")
        store.get("conformance_present")
        assert all(level >= logging.INFO for level in store._client.levels_during_access.values())
        assert all(logging.getLogger(name).level == logging.DEBUG for name in gsm._PAYLOAD_LOGGERS)
    finally:
        for name in gsm._PAYLOAD_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)


def _serve_bundle(adapter, request, **kwargs) -> requests.Response:
    response = requests.Response()
    response.status_code, response.url, response.request = 200, request.url, request
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(
        {"name": "projects/p/secrets/bundle/versions/1", "payload": {"data": base64.b64encode(_BUNDLE).decode()}}
    ).encode()
    return response


def test_a_real_client_logs_nothing_from_its_transport_while_the_store_fetches(monkeypatch):
    """The transport logger set to DEBUG on its own, as the SDK's logging scope does: the same
    client logs the call outside the store, and nothing from inside it. The handler sits on the
    transport logger because the client stops the ``google`` logger propagating to the root."""
    monkeypatch.setattr(google.auth, "default", lambda scopes: (AnonymousCredentials(), "p"))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _serve_bundle)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    transport = logging.getLogger(_REST_LOGGER)
    transport.addHandler(handler)
    transport.setLevel(logging.DEBUG)
    try:
        store = _store()
        assert store.get("conformance_present") == "value-1"
        assert records == []
        store._client.access_secret_version(name="projects/p/secrets/bundle/versions/latest")
        assert records
    finally:
        transport.removeHandler(handler)
        transport.setLevel(logging.NOTSET)


def test_overlapping_fetches_keep_the_loggers_quiet_until_the_last_one_ends():
    transport = logging.getLogger(_REST_LOGGER)
    transport.setLevel(logging.DEBUG)
    first, second = gsm._suppress_payload_logging(), gsm._suppress_payload_logging()
    try:
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        assert transport.level == logging.INFO
        second.__exit__(None, None, None)
        assert transport.level == logging.DEBUG
    finally:
        transport.setLevel(logging.NOTSET)


def test_the_logger_levels_come_back_when_the_fetch_raises(adc):
    transport = logging.getLogger(_REST_LOGGER)
    transport.setLevel(logging.DEBUG)
    try:
        store = _store(project="p")
        store._connect().payload = ServiceUnavailable("Secret Manager is down")
        with pytest.raises(ServiceUnavailable):
            store.get("conformance_present")
        assert store._client.levels_during_access[_REST_LOGGER] == logging.INFO
        assert transport.level == logging.DEBUG
    finally:
        transport.setLevel(logging.NOTSET)
