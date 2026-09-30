"""SQLite implementation of the DocumentStore interface (stdlib, no infra).

Stores each document as a JSON row in a per-collection table; SQLite provides
uniqueness (via unique indexes on ``json_extract``), atomicity (transactions),
and durability. ``Filter`` / ``UpdateSpec`` / ``Sort`` semantics are evaluated in
Python, mirroring ``mongo_document_store`` as the oracle. Datetimes are
serialized as ISO strings.

SQL safety note: every user-controlled *value* is bound as a ``?`` placeholder.
The only interpolated tokens are SQL *identifiers* (table / index / column
names), which cannot be passed as bound parameters. Those are derived from a
strict allowlist (see ``_safe_ident`` / ``_safe_path``) before interpolation, so
the ``execute(f"...")`` sites carry ``# nosemgrep`` for the raw-query rule.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from agent_env.store.document_store.document_store import (
    DocumentStore,
    DuplicateKeyError,
    Filter,
    Sort,
    UpdateSpec,
)
from agent_env.store.document_store import evaluation
from agent_env.store.ids import aliases_local_encoding
from agent_env.store.local_state import ensure_state_dir

# SQL identifiers (table/index names) must be a bare word; JSON-path field
# references additionally allow dots for nested paths (e.g. ``metadata.rev``).
_IDENT_RE = re.compile(r"[A-Za-z0-9_]+")
_PATH_RE = re.compile(r"[A-Za-z0-9_.]+")

_BUSY_TIMEOUT_SECONDS = 5.0
_WAL_SWITCH_RETRY_DELAY_SECONDS = 0.01


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch the database to WAL. Processes opening a new database at the same time all need
    the exclusive lock this takes, and SQLite fails the losers at once instead of waiting, so
    retry until the winner is done."""
    deadline = time.monotonic() + _BUSY_TIMEOUT_SECONDS
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if e.sqlite_errorcode != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                raise
            time.sleep(_WAL_SWITCH_RETRY_DELAY_SECONDS)


class LocalSqliteDocumentStore(DocumentStore):
    """DocumentStore backed by a single stdlib-``sqlite3`` database file."""

    def __init__(self, path: str) -> None:
        # Nothing touches disk until the first operation: constructing a store creates nothing.
        self._path = Path(path)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._tables: set[str] = set()

    @property
    def path(self) -> Path:
        return self._path

    def check_id(self, entity_id: str) -> None:
        super().check_id(entity_id)
        if aliases_local_encoding(entity_id):
            raise ValueError(
                f"id {entity_id!r} is spelled like an encoded @local id, so it would share that id's objects, "
                "images and files; a bare id in a local store can't start with 'local/' or 'local-<name>-<12 hex>'"
            )

    @property
    def _conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                ensure_state_dir(self._path.parent)
                conn = sqlite3.connect(
                    str(self._path), timeout=_BUSY_TIMEOUT_SECONDS, check_same_thread=False, isolation_level=None
                )
                _enable_wal(conn)
                self._connection = conn
            return self._connection

    def find_one(
        self, collection: str, filter: Filter, sort: Optional[Sort] = None
    ) -> Optional[dict]:
        with self._lock:
            matches = [doc for _, doc in self._load(collection) if evaluation.matches(doc, filter)]
            matches = evaluation.sort_docs(matches, sort)
            return matches[0] if matches else None

    def query(
        self,
        collection: str,
        filter: Filter,
        sort: Optional[Sort] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        with self._lock:
            docs = [doc for _, doc in self._load(collection) if evaluation.matches(doc, filter)]
            docs = evaluation.sort_docs(docs, sort)
            if offset:
                docs = docs[offset:]
            if limit:
                docs = docs[:limit]
            return docs

    def count(self, collection: str, filter: Filter) -> int:
        with self._lock:
            return sum(1 for _, doc in self._load(collection) if evaluation.matches(doc, filter))

    def insert(self, collection: str, doc: dict) -> None:
        with self._lock:
            tbl = self._ensure_table(collection)
            self._insert_raw(tbl, dict(doc))

    def update(
        self, collection: str, filter: Filter, update: UpdateSpec, upsert: bool = False
    ) -> int:
        update.validate()
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, doc = match
                evaluation.apply_update(doc, update)
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(doc), rowid),
                )
                return 1
            if upsert:
                self._insert_raw(self._ensure_table(collection), evaluation.synthesize(filter, update))
                return 1
            return 0

    def update_one_and_get(
        self,
        collection: str,
        filter: Filter,
        update: UpdateSpec,
        return_after: bool = True,
        upsert: bool = False,
    ) -> Optional[dict]:
        update.validate()
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, doc = match
                before = json.loads(self._dumps(doc))
                evaluation.apply_update(doc, update)
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(doc), rowid),
                )
                return doc if return_after else before
            if upsert:
                new_doc = evaluation.synthesize(filter, update)
                self._insert_raw(self._ensure_table(collection), new_doc)
                return new_doc if return_after else None
            return None

    def replace(
        self, collection: str, filter: Filter, doc: dict, upsert: bool = False
    ) -> int:
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is not None:
                rowid, _ = match
                self._conn.execute(
                    # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; doc/rowid are bound
                    f'UPDATE "{self._table(collection)}" SET doc=? WHERE rowid=?',
                    (self._dumps(dict(doc)), rowid),
                )
                return 1
            if upsert:
                self._insert_raw(self._ensure_table(collection), dict(doc))
                return 1
            return 0

    def delete(self, collection: str, filter: Filter) -> int:
        with self._lock, self._immediate():
            match = self._first_match(collection, filter)
            if match is None:
                return 0
            rowid, _ = match
            # nosemgrep: sqlalchemy-execute-raw-query -- table is a validated identifier; rowid is bound
            self._conn.execute(f'DELETE FROM "{self._table(collection)}" WHERE rowid=?', (rowid,))
            return 1

    def ensure_index(
        self,
        collection: str,
        fields: list[str],
        unique: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if ttl_seconds is not None:
            logger.warning(
                "LocalSqliteDocumentStore has no TTL support; ttl_seconds=%s on %s%s ignored — rows will not auto-expire.",
                ttl_seconds, collection, fields,
            )
        with self._lock:
            tbl = self._ensure_table(collection)
            raw_name = tbl + "_" + "_".join(fields) + ("_unique" if unique else "_index")
            name = self._safe_ident(re.sub(r"\W", "_", raw_name))
            cols = ", ".join(f"json_extract(doc, '$.{self._safe_path(f)}')" for f in fields)
            uniq = "UNIQUE " if unique else ""
            self._conn.execute(
                # nosemgrep: sqlalchemy-execute-raw-query -- name/tbl/cols are validated identifiers
                f'CREATE {uniq}INDEX IF NOT EXISTS "{name}" ON "{tbl}" ({cols})'
            )

    # --- sqlite plumbing ---

    @staticmethod
    def _safe_ident(name: str) -> str:
        """Validate a SQL identifier (table/index name) against a strict allowlist.

        Identifiers cannot be passed as bound parameters, so we assert they are a
        bare ``[A-Za-z0-9_]`` word before interpolating them into SQL. User input
        only ever reaches SQL as a bound ``?`` value, never as an identifier.
        """
        if not _IDENT_RE.fullmatch(name):
            raise ValueError(f"unsafe SQL identifier: {name!r}")
        return name

    @staticmethod
    def _safe_path(field: str) -> str:
        """Validate a JSON-path field reference used inside ``json_extract('$.…')``."""
        if not _PATH_RE.fullmatch(field):
            raise ValueError(f"unsafe JSON field path: {field!r}")
        return field

    def _table(self, collection: str) -> str:
        # The table name is an identifier (not bindable): sanitize the collection
        # to a bare word, then assert the allowlist as defense-in-depth.
        return self._safe_ident("docs_" + re.sub(r"\W", "_", collection))

    def _ensure_table(self, collection: str) -> str:
        tbl = self._table(collection)
        if tbl not in self._tables:
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier
            self._conn.execute(f'CREATE TABLE IF NOT EXISTS "{tbl}" (doc TEXT NOT NULL)')
            self._tables.add(tbl)
        return tbl

    def _adopt_if_created(self, tbl: str) -> bool:
        """Adopt a table that exists on disk but is not yet in ``self._tables``."""
        if self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
        ).fetchone():
            self._tables.add(tbl)
            return True
        return False

    def _load(self, collection: str) -> list[tuple[int, dict]]:
        tbl = self._table(collection)
        if tbl not in self._tables and not self._adopt_if_created(tbl):
            return []
        return [
            (row[0], json.loads(row[1]))
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier
            for row in self._conn.execute(f'SELECT rowid, doc FROM "{tbl}"')
        ]

    def _first_match(self, collection: str, filter: Filter) -> Optional[tuple[int, dict]]:
        for rowid, doc in self._load(collection):
            if evaluation.matches(doc, filter):
                return rowid, doc
        return None

    @contextmanager
    def _immediate(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def _insert_raw(self, tbl: str, doc: dict) -> None:
        try:
            # nosemgrep: sqlalchemy-execute-raw-query -- tbl is a validated identifier; doc is bound
            self._conn.execute(f'INSERT INTO "{tbl}"(doc) VALUES(?)', (self._dumps(doc),))
        except sqlite3.IntegrityError as e:
            raise DuplicateKeyError(str(e)) from e

    def _json_default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")

    def _dumps(self, doc: dict) -> str:
        return json.dumps(doc, default=self._json_default)
