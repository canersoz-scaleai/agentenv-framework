"""MongoDocumentStore reports pymongo's duplicate-key error as the store's own, on every write."""

from __future__ import annotations

import pytest
from pymongo.errors import DuplicateKeyError as PyMongoDuplicateKeyError

from agent_env.store import DuplicateKeyError, Filter, UpdateSpec
from agent_env.store.document_store.mongo_document_store import MongoDocumentStore


class _Colliding:
    """A pymongo collection whose every write collides with a unique index."""

    def _collide(self, *args, **kwargs):
        raise PyMongoDuplicateKeyError("E11000 duplicate key error collection: db.coll index: id_version_unique")

    insert_one = update_one = find_one_and_update = replace_one = _collide


_MISSING = Filter.of(id="a", version=1, state="free")


@pytest.mark.parametrize(
    "write",
    [
        lambda store: store.insert("coll", {"id": "a", "version": 1}),
        lambda store: store.update("coll", _MISSING, UpdateSpec(set={"x": 1}), upsert=True),
        lambda store: store.update_one_and_get("coll", _MISSING, UpdateSpec(set={"x": 1}), upsert=True),
        lambda store: store.replace("coll", _MISSING, {"id": "a", "version": 1}, upsert=True),
    ],
    ids=["insert", "update-upsert", "update-one-and-get-upsert", "replace-upsert"],
)
def test_a_unique_violation_is_the_stores_duplicate_key_error(write):
    with pytest.raises(DuplicateKeyError, match="E11000"):
        write(MongoDocumentStore({"coll": _Colliding()}))
