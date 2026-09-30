"""TaskInstanceStore.file_artifact_universe_ids_for_batch orders its answer itself."""

from __future__ import annotations

from agent_env.config import set_document_store
from agent_env.task.store import TaskInstanceStore


class _UnorderedDocStore:
    """Answers a query with its matches reversed, as a store with no natural order may."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def ensure_index(self, *args, **kwargs) -> None:
        pass

    def query(self, collection, filter, sort=None, limit=None, offset=None) -> list[dict]:
        return list(reversed(self.docs))


def _instance(instance_id: str, created_at_utc: str, universe_id: str | None) -> dict:
    metadata = {"batch_id": "batch-1"}
    if universe_id is not None:
        metadata["file_artifact_universe"] = {"id": universe_id}
    return {"instance_id": instance_id, "created_at_utc": created_at_utc, "context": {"metadata": metadata}}


def test_universes_follow_instance_creation_order_whatever_order_the_store_answers_in():
    set_document_store(_UnorderedDocStore([
        _instance("i-3", "2026-09-28 10:00 UTC", "u-a"),
        _instance("i-1", "2026-09-28 10:01 UTC", "u-c"),
        _instance("i-2", "2026-09-28 10:01 UTC", "u-b"),
        _instance("i-4", "2026-09-28 10:02 UTC", None),
        _instance("i-5", "2026-09-28 10:03 UTC", "u-d"),
        _instance("i-6", "2026-09-28 10:04 UTC", "u-a"),
    ]))
    assert TaskInstanceStore().file_artifact_universe_ids_for_batch("batch-1") == ["u-a", "u-c", "u-b", "u-d"]
