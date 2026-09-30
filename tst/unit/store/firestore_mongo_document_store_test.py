"""FirestoreMongoDocumentStore against a stand-in collection that answers as Firestore does."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import google.auth
import pytest
import requests
from google.auth.exceptions import DefaultCredentialsError, TransportError
from pymongo import _csot
from pymongo.errors import DuplicateKeyError as PyMongoDuplicateKeyError
from pymongo.errors import OperationFailure, PyMongoError

from agent_env.config.errors import ConfigError
from agent_env.store import DuplicateKeyError, Filter, UpdateSpec
from agent_env.store.document_store import firestore_mongo_document_store as fs
from agent_env.store.document_store.firestore_mongo_document_store import (
    WRITE_STAMP_FIELD,
    FirestoreMongoDocumentStore,
)

_X = Filter.of(id="x")


def _result(matched=0, modified=0, upserted_id=None):
    return SimpleNamespace(matched_count=matched, modified_count=modified, upserted_id=upserted_id)


class _Collection:
    """Answers each write with the next scripted reply, recording what it was sent. A reply that is
    an exception is raised; a callable is called with the write's spec."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.writes: list[tuple[str, dict]] = []
        self.found: list = []
        self.index_deadlines: list = []

    def _reply(self, op, spec):
        self.writes.append((op, spec))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(spec) if callable(reply) else reply

    def update_one(self, filter, spec, upsert=False, session=None):
        return self._reply("update_one", spec)

    def replace_one(self, filter, body, upsert=False):
        return self._reply("replace_one", body)

    def find_one_and_update(self, filter, spec, return_document=None, upsert=False):
        return self._reply("find_one_and_update", spec)

    def find_one(self, filter, sort=None, session=None):
        reply = self.found.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def create_index(self, keys, **kwargs):
        self.index_deadlines.append(_csot.get_timeout())


class _Session:
    @contextmanager
    def start_transaction(self):
        yield


class _Database:
    def __init__(self, collection, name="db"):
        self._collection = collection
        self.name = name

        @contextmanager
        def start_session():
            yield _Session()

        self.client = SimpleNamespace(start_session=start_session)

    def __getitem__(self, name):
        return self._collection


def _store(collection) -> FirestoreMongoDocumentStore:
    return FirestoreMongoDocumentStore(_Database(collection))


def _own_image(extra=None):
    """A find_one_and_update reply carrying the stamp the store sent."""
    return lambda spec: {"_id": 1, "id": "x", **(extra or {}), WRITE_STAMP_FIELD: spec["$set"][WRITE_STAMP_FIELD]}


def test_a_matched_update_that_did_not_apply_counts_as_a_miss():
    coll = _Collection(_result(matched=1, modified=0), _result(matched=1, modified=1))
    store = _store(coll)
    assert store.update("c", _X, UpdateSpec(set={"owner": 1})) == 0
    assert store.update("c", _X, UpdateSpec(set={"owner": 1})) == 1
    stamps = [spec["$set"][WRITE_STAMP_FIELD] for _, spec in coll.writes]
    assert len(set(stamps)) == 2 and all(stamps)


def test_update_one_and_get_returns_none_for_another_writer_s_document():
    coll = _Collection({"_id": 1, "id": "x", "owner": 2, WRITE_STAMP_FIELD: "theirs"}, _own_image({"owner": 1}), None)
    store = _store(coll)
    assert store.update_one_and_get("c", _X, UpdateSpec(set={"owner": 1})) is None
    assert store.update_one_and_get("c", _X, UpdateSpec(set={"owner": 1})) == {"id": "x", "owner": 1}
    assert store.update_one_and_get("c", _X, UpdateSpec(set={"owner": 1})) is None


def test_a_replace_is_stamped_and_counted_by_what_it_modified():
    coll = _Collection(_result(matched=1, modified=0), _result(upserted_id="new"))
    store = _store(coll)
    assert store.replace("c", _X, {"id": "x", "v": 2}) == 0
    assert store.replace("c", _X, {"id": "x", "v": 2}, upsert=True) == 1
    assert all(WRITE_STAMP_FIELD in body for _, body in coll.writes)


def test_reads_never_return_the_stamp():
    coll = _Collection()
    coll.found = [{"_id": 1, "id": "x", WRITE_STAMP_FIELD: "s"}]
    coll.find = lambda filter: iter([{"_id": 2, "id": "y", WRITE_STAMP_FIELD: "s"}])
    store = _store(coll)
    assert store.find_one("c", _X) == {"id": "x"}
    assert store.query("c", Filter.of()) == [{"id": "y"}]


def test_an_upsert_that_loses_the_insert_race_is_retried_once():
    collision = PyMongoDuplicateKeyError("E11000 duplicate key error")
    store = _store(_Collection(collision, _result(matched=1, modified=1)))
    assert store.update("c", _X, UpdateSpec(set={"a": 1}), upsert=True) == 1
    with pytest.raises(DuplicateKeyError):
        _store(_Collection(collision, collision)).update("c", _X, UpdateSpec(set={"a": 1}), upsert=True)
    with pytest.raises(DuplicateKeyError):
        _store(_Collection(collision)).update("c", _X, UpdateSpec(set={"a": 1}))


def test_a_pre_image_update_retries_a_write_conflict_in_its_transaction(monkeypatch):
    monkeypatch.setattr(fs.time, "sleep", lambda seconds: None)
    coll = _Collection(_result(matched=1, modified=1))
    coll.found = [OperationFailure("Too much contention", code=112), {"_id": 1, "id": "x", "n": 1, WRITE_STAMP_FIELD: "s"}]
    assert _store(coll).update_one_and_get("c", _X, UpdateSpec(inc={"n": 1}), return_after=False) == {"id": "x", "n": 1}
    assert [op for op, _ in coll.writes] == ["update_one"]


def test_a_pre_image_update_raises_what_is_not_a_conflict():
    coll = _Collection()
    coll.found = [OperationFailure("unauthorized", code=13)]
    with pytest.raises(OperationFailure):
        _store(coll).update_one_and_get("c", _X, UpdateSpec(inc={"n": 1}), return_after=False)


def test_an_index_build_is_given_minutes_not_the_socket_timeout():
    coll = _Collection()
    _store(coll).ensure_index("c", ["id", "version"], unique=True)
    assert coll.index_deadlines and coll.index_deadlines[0] > 300


def test_threads_asking_for_one_index_wait_for_its_first_build():
    """Firestore answers a createIndex re-issued during a build before the index enforces, so
    only the first caller in a process may issue it; the rest wait for it."""
    building, release = threading.Event(), threading.Event()
    coll = _Collection()

    def slow_create_index(keys, **kwargs):
        coll.index_deadlines.append(_csot.get_timeout())
        building.set()
        release.wait(5)

    coll.create_index = slow_create_index
    store = _store(coll)
    first = threading.Thread(target=store.ensure_index, args=("c", ["id"]), kwargs={"unique": True})
    first.start()
    building.wait(5)
    second = threading.Thread(target=store.ensure_index, args=("c", ["id"]), kwargs={"unique": True})
    second.start()
    second.join(0.2)
    assert second.is_alive()
    release.set()
    first.join(5)
    second.join(5)
    assert len(coll.index_deadlines) == 1


def test_stores_on_one_backend_share_its_index_builds_and_other_backends_do_not():
    first, same, other = _Collection(), _Collection(), _Collection()
    store = FirestoreMongoDocumentStore(_Database(first), backend="host-a/db")
    store.ensure_index("c", ["id"], unique=True)
    store.ensure_index("c", ["id"], unique=True)
    store.ensure_index("c", ["id"], unique=True, ttl_seconds=60)
    store.ensure_index("c", ["id"])
    FirestoreMongoDocumentStore(_Database(same), backend="host-a/db").ensure_index("c", ["id"], unique=True)
    FirestoreMongoDocumentStore(_Database(other), backend="host-b/db").ensure_index("c", ["id"], unique=True)
    assert [len(c.index_deadlines) for c in (first, same, other)] == [3, 0, 1]


def test_a_pre_image_update_retries_a_transient_transaction_error(monkeypatch):
    monkeypatch.setattr(fs.time, "sleep", lambda seconds: None)
    transient = PyMongoError("connection reset during the transaction")
    transient._add_error_label("TransientTransactionError")
    coll = _Collection(_result(matched=1, modified=1))
    coll.found = [transient, {"_id": 1, "id": "x", "n": 1}]
    assert _store(coll).update_one_and_get("c", _X, UpdateSpec(inc={"n": 1}), return_after=False) == {"id": "x", "n": 1}


class _Credentials:
    def __init__(self, valid, drops=0):
        self.valid, self.token, self.refreshed, self.drops = valid, "token-1", 0, drops

    def refresh(self, request):
        self.refreshed += 1
        if self.drops:
            self.drops -= 1
            raise TransportError("connection dropped") from requests.ConnectionError("reset by peer")
        self.valid, self.token = True, "token-2"


def test_the_oidc_callback_hands_over_an_access_token_refreshing_it_when_stale():
    fresh, stale = _Credentials(valid=True), _Credentials(valid=False)
    assert fs._AccessTokenCallback(fresh).fetch(None).access_token == "token-1" and fresh.refreshed == 0
    assert fs._AccessTokenCallback(stale).fetch(None).access_token == "token-2" and stale.refreshed == 1


def test_the_oidc_callback_retries_a_dropped_token_request(monkeypatch):
    monkeypatch.setattr(fs._google, "RETRY", fs._google.RETRY.with_delay(initial=0, maximum=0))
    dropped_once = _Credentials(valid=False, drops=1)
    assert fs._AccessTokenCallback(dropped_once).fetch(None).access_token == "token-2"
    assert dropped_once.refreshed == 2


class _Client:
    """Stands in for MongoClient, recording how it was built."""

    built: list = []
    ping_error: Exception | None = None

    def __init__(self, uri, **options):
        self.uri, self.options = uri, options
        _Client.built.append(self)
        self.admin = SimpleNamespace(command=self._command)

    def _command(self, name):
        if _Client.ping_error is not None:
            raise _Client.ping_error

    def __getitem__(self, name):
        return SimpleNamespace(name=name, client=self)


@pytest.fixture
def client(monkeypatch):
    _Client.built, _Client.ping_error = [], None
    monkeypatch.setattr(fs, "MongoClient", _Client)
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_Credentials(valid=True), "p"))
    return _Client


def test_from_config_connects_over_oidc_without_retryable_writes(client):
    store = FirestoreMongoDocumentStore.from_config(host="uid.us-west1.firestore.goog", database="db")
    [built] = client.built
    assert built.uri == "mongodb://uid.us-west1.firestore.goog:443/db"
    assert built.options["loadBalanced"] and built.options["tls"]
    assert built.options["authMechanism"] == "MONGODB-OIDC" and built.options["retryWrites"] is False
    assert isinstance(built.options["authMechanismProperties"]["OIDC_CALLBACK"], fs._AccessTokenCallback)
    assert store.database.name == "db" and store._backend == "uid.us-west1.firestore.goog/db"


def test_from_config_names_the_database_when_it_cannot_connect(client, monkeypatch):
    client.ping_error = RuntimeError("unreachable")
    with pytest.raises(ConnectionError, match="Firestore database 'db'"):
        FirestoreMongoDocumentStore.from_config(host="h", database="db")

    def no_adc(scopes):
        raise DefaultCredentialsError("not found")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(ConfigError, match="No Google credentials for Firestore database 'db'"):
        FirestoreMongoDocumentStore.from_config(host="h", database="db")
