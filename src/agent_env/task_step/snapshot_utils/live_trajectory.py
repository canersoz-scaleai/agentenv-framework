"""A turn's live trajectory, stored as it is read: ``<prefix><turn>/live/<after>-<next>.jsonl`` per
page, ``meta.json`` naming the events' format, and ``end.json`` once the turn has ended: with the
agent's final state when the follower read it, else ``canceled`` or ``failed`` when the step stopped
following first.

Written from the worker with its own credentials, like the final trajectory. Every object is
written once: one already there holds the same bytes, so a repeated write is a no-op."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from agentenv_protocol.a2a_agent import TrajectoryState

from agent_env.a2a_agent.trajectory_follower import TrajectoryBatch, follow_trajectory
from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

_WRITE_ATTEMPTS = 3
# The task has ended when the turn's wait returns, so draining its last events is normally one read.
_DRAIN_SECONDS = 60
# A stopped step marks its turn ended in passing: the write must not hold up the cancel.
_END_WRITE_SECONDS = 5


class LiveTrajectoryChunks:
    """The ``follow_trajectory`` sink that stores one turn's pages."""

    def __init__(self, store: ObjectStore, prefix_key: str, turn_id: str) -> None:
        self._store = store
        self._directory = f"{prefix_key}{turn_id}/live/"
        self._format_written = False
        self._stored = 0
        self._ended = False

    async def __call__(self, batch: TrajectoryBatch) -> None:
        if batch.events:
            body = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in batch.events
            )
            await self._put(f"{batch.after:08d}-{batch.next:08d}.jsonl", body, "application/x-ndjson")
            self._stored = batch.next
        if batch.format and not self._format_written:
            await self._put("meta.json", json.dumps({"format": batch.format}).encode(), "application/json")
            self._format_written = True
        if batch.last:
            await self.end(batch.state)

    async def end(self, state: TrajectoryState) -> None:
        """Mark the turn ended in ``state`` after the events stored so far; the first mark stays."""
        if self._ended:
            return
        end = {"state": state.value, "next": self._stored}
        await self._put("end.json", json.dumps(end).encode(), "application/json")
        self._ended = True

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
    raises, the follower is stopped at once, since the agent may be gone, and the turn is marked
    ended (``canceled`` for a cancelled step, else ``failed``) so readers do not wait on it. A
    follower's own failure is logged and never fails the turn."""
    stop = asyncio.Event()
    followers: list[asyncio.Task] = []
    sinks: list[LiveTrajectoryChunks] = []
    stopped_as: TrajectoryState | None = None

    def follow(task_id: str) -> None:
        sink = LiveTrajectoryChunks(store, prefix_key, turn_id)
        sinks.append(sink)
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
    except asyncio.CancelledError:
        stopped_as = TrajectoryState.CANCELED
        raise
    except BaseException:
        stopped_as = TrajectoryState.FAILED
        raise
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
        if stopped_as is not None and sinks:
            await _mark_ended(sinks[-1], stopped_as, turn_id)


async def _mark_ended(sink: LiveTrajectoryChunks, state: TrajectoryState, turn_id: str) -> None:
    marking = asyncio.create_task(sink.end(state))
    done, _ = await asyncio.wait({marking}, timeout=_END_WRITE_SECONDS)
    if not done:
        marking.cancel()
        logger.warning("Live trajectory of turn %s was not marked %s in time (continuing)", turn_id, state.value)
    elif marking.exception() is not None:
        logger.warning(
            "Live trajectory of turn %s was not marked %s (continuing): %r", turn_id, state.value, marking.exception()
        )
