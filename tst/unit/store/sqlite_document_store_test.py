"""LocalSqliteDocumentStore runs the full DocumentStore conformance suite.

Fast tier — no Mongo, no network (stdlib sqlite over a temp file), so the same
assertions that prove Mongo parity also give quick backend-neutral coverage.
"""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.store import Filter, LocalSqliteDocumentStore, UpdateSpec, compare_and_swap
from agent_env.store.document_store import sqlite_document_store
from tst.store import conformance


@pytest.fixture
def store_coll(tmp_path):
    return LocalSqliteDocumentStore(str(tmp_path / "docstore.db")), "coll"


@pytest.mark.parametrize("case", conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store_coll):
    case(*store_coll)


def test_threaded_cas_converges(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["id"], unique=True)
    store.insert(coll, {"id": "x", "rev": 0, "steps": []})
    n = 20

    def add(k):
        return compare_and_swap(
            store, coll, Filter.of(id="x"),
            lambda doc: UpdateSpec(add_to_set={"steps": [k]}),
            counter_field="rev",
        )

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(add, range(n)))

    doc = store.find_one(coll, Filter.of(id="x"))
    assert sorted(doc["steps"]) == list(range(n))  # every writer landed, none lost
    assert doc["rev"] == n


def test_reader_sees_a_collection_created_by_another_connection(tmp_path):
    """A long-lived reader must see a collection another connection creates after it opened,
    without reopening. Covers several primitives, not one."""
    path = str(tmp_path / "shared.db")
    reader = LocalSqliteDocumentStore(path)
    writer = LocalSqliteDocumentStore(path)   # a separate connection to the same file

    for coll in ("a2a_agents", "tasks", "evals"):
        assert reader.find_one(coll, Filter.of(id="x")) is None       # collection absent so far
        writer.insert(coll, {"id": "x", "version": 1})                # creates docs_<coll> + a row
        got = reader.find_one(coll, Filter.of(id="x"))
        assert got is not None and got["id"] == "x", f"{coll}: cross-connection write not seen"


def test_processes_opening_a_new_database_at_once_all_write(tmp_path):
    path = tmp_path / "new" / "shared.db"
    code = (
        "import sys\n"
        "from agent_env.store import LocalSqliteDocumentStore\n"
        "store = LocalSqliteDocumentStore(sys.argv[1])\n"
        "for i in range(50):\n"
        "    store.insert('coll', {'id': f'{sys.argv[2]}-{i}'})\n"
    )
    # nosemgrep: dangerous-subprocess-use-audit -- argv is sys.executable + the literal `code`; no external input
    writers = [subprocess.Popen([sys.executable, "-c", code, str(path), name], stderr=subprocess.PIPE) for name in "abcd"]
    for writer in writers:
        _, stderr = writer.communicate(timeout=120)
        assert writer.returncode == 0, stderr.decode()
    assert len(LocalSqliteDocumentStore(str(path)).query("coll", Filter.of())) == 200


class _Connection:
    """Fails its first ``failures`` statements with ``code``."""

    def __init__(self, failures, code):
        self.failures, self.code, self.calls = failures, code, 0

    def execute(self, sql):
        self.calls += 1
        if self.calls <= self.failures:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = self.code
            raise error


def test_the_wal_switch_waits_out_a_connection_holding_the_lock():
    conn = _Connection(failures=3, code=sqlite3.SQLITE_BUSY)
    sqlite_document_store._enable_wal(conn)
    assert conn.calls == 4


def test_the_wal_switch_raises_other_errors_and_a_lock_held_too_long(monkeypatch):
    conn = _Connection(failures=1, code=sqlite3.SQLITE_IOERR)
    with pytest.raises(sqlite3.OperationalError):
        sqlite_document_store._enable_wal(conn)
    assert conn.calls == 1

    monkeypatch.setattr(sqlite_document_store, "_BUSY_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(sqlite3.OperationalError):
        sqlite_document_store._enable_wal(_Connection(failures=10**6, code=sqlite3.SQLITE_BUSY))
