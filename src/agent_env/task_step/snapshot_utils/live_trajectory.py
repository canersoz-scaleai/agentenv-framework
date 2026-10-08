"""A turn's live trajectory, stored as it is read: ``<prefix><turn>/live/<after>-<next>.jsonl`` per
page, ``meta.json`` naming the events' format, and ``end.json`` once the turn has ended.

Written from the worker with its own credentials, like the final trajectory. Every object is
written once: one already there holds the same bytes, so a repeated write is a no-op."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from agent_env.a2a_agent.trajectory_follower import TrajectoryBatch, follow_trajectory
from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

_WRITE_ATTEMPTS = 3
# The task has ended when the turn's wait returns, so draining its last events is normally one read.
_DRAIN_SECONDS = 60


class LiveTrajectoryChunks:
    """The ``follow_trajectory`` sink that stores one turn's pages."""

    def __init__(self, store: ObjectStore, prefix_key: str, turn_id: str) -> None:
        self._store = store
        self._directory = f"{prefix_key}{turn_id}/live/"
        self._format_written = False

    async def __call__(self, batch: TrajectoryBatch) -> None:
        if batch.events:
            body = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in batch.events
            )
            await self._put(f"{batch.after:08d}-{batch.next:08d}.jsonl", body, "application/x-ndjson")
        if batch.format and not self._format_written:
            await self._put("meta.json", json.dumps({"format": batch.format}).encode(), "application/json")
            self._format_written = True
        if batch.last:
            end = {"state": batch.state.value, "next": batch.next}
            await self._put("end.json", json.dumps(end).encode(), "application/json")

    async def _put(self, name: str, body: bytes, content_type: str) -> None:
        key = self._directory + name
        for attempt in range(1, _WRITE_ATTEMPTS + 1):
            try:
                await asyncio.to_thread(self._store.put, key, body, content_type=content_type)
                return
            except ObjectAlreadyExistsError:
                return
            except Exception as exc:
                if attempt == _WRITE_ATTEMPTS:
                    raise
                logger.warning("Live trajectory write of %s failed (attempt %d, will retry): %s", key, attempt, exc)
                await asyncio.sleep(attempt)


@asynccontextmanager
async def following_live_trajectory(
    endpoint: str,
    store: ObjectStore,
    prefix_key: str,
    turn_id: str,
    *,
    poll_interval_seconds: float,
) -> AsyncIterator[Callable[[str], None]]:
    """Yield the callback that starts following the agent's task once the turn's message is sent.

    When the body returns, the follower drains what remains, for at most ``_DRAIN_SECONDS``; when it
    raises, the follower is stopped at once, since the agent may be gone. A follower's own failure
    is logged and never fails the turn."""
    stop = asyncio.Event()
    followers: list[asyncio.Task] = []

    def follow(task_id: str) -> None:
        sink = LiveTrajectoryChunks(store, prefix_key, turn_id)
        followers.append(
            asyncio.create_task(
                follow_trajectory(
                    endpoint, task_id, sink, poll_interval_seconds=poll_interval_seconds, stop=stop
                )
            )
        )

    try:
        yield follow
        stop.set()
        if followers:
            _, draining = await asyncio.wait(followers, timeout=_DRAIN_SECONDS)
            if draining:
                logger.warning(
                    "Live trajectory of turn %s did not finish draining in %ds (continuing)",
                    turn_id,
                    _DRAIN_SECONDS,
                )
    finally:
        stop.set()
        for follower in followers:
            follower.cancel()
        if followers:
            await asyncio.wait(followers)
        for follower in followers:
            if not follower.cancelled() and follower.exception() is not None:
                logger.warning(
                    "Live trajectory of turn %s stopped (continuing): %r", turn_id, follower.exception()
                )
