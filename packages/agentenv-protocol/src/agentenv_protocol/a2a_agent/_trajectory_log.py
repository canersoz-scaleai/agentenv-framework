"""The framework's in-process store of task trajectories, served by the trajectory extension."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

from .extensions import TrajectoryState
from .tasks.v1 import TrajectoryLog

logger = logging.getLogger(__name__)

MAX_CACHED_TRAJECTORY_BYTES = 256 * 1024 * 1024
MAX_CACHED_CONTEXTS = 1_024
MAX_TRAJECTORY_READ_EVENTS = 1_000
MAX_TRAJECTORY_READ_BYTES = 4 * 1024 * 1024


@dataclass(slots=True)
class _TaskRecord:
    context_id: str
    log: TrajectoryLog
    started: bool = False
    cancel_requested: bool = False
    final_state: TrajectoryState | None = None
    native: bytes | None = None

    @property
    def state(self) -> TrajectoryState:
        if self.final_state is not None:
            return self.final_state
        return TrajectoryState.RUNNING if len(self.log) else TrajectoryState.PENDING

    @property
    def size_bytes(self) -> int:
        return self.log._size_bytes + len(self.native or b"")


@dataclass(frozen=True, slots=True)
class TrajectoryPage:
    context_id: str
    task_id: str
    state: TrajectoryState
    format: str | None
    events: list[bytes]
    total: int


class TaskTrajectories:
    """Each task's log and final record, grouped by context in execution order.

    A whole context is the unit of eviction, least recently used first, so positions in a
    context read never shift; a context with a task pending or running is never evicted."""

    def __init__(
        self,
        max_bytes: int = MAX_CACHED_TRAJECTORY_BYTES,
        max_contexts: int = MAX_CACHED_CONTEXTS,
    ) -> None:
        if max_bytes < 1 or max_contexts < 1:
            raise ValueError("trajectory cache bounds must be positive")
        self._max_bytes = max_bytes
        self._max_contexts = max_contexts
        self._tasks: dict[str, _TaskRecord] = {}
        self._contexts: OrderedDict[str, list[str]] = OrderedDict()

    def register(self, task_id: str, context_id: str) -> TrajectoryLog:
        """The task's log, created ``pending`` the first time the task is seen."""
        record = self._tasks.get(task_id)
        if record is None:
            log = TrajectoryLog()
            log._task_id = task_id
            record = self._tasks[task_id] = _TaskRecord(context_id=context_id, log=log)
            self._contexts.setdefault(context_id, []).append(task_id)
        self._touch(record.context_id)
        return record.log

    def start(self, task_id: str) -> None:
        """Place the task after the context's started tasks, as it begins to run.

        Only tasks that have not started move; they have no events, so no position shifts."""
        record = self._tasks[task_id]
        if record.started:
            return
        order = self._contexts[record.context_id]
        order.remove(task_id)
        started = sum(1 for other in order if self._tasks[other].started)
        order.insert(started, task_id)
        record.started = True
        self._touch(record.context_id)

    def request_cancel(self, task_id: str) -> None:
        """Make ``canceled`` the outcome the task is sealed with, whatever ``run`` returns."""
        record = self._tasks.get(task_id)
        if record is not None:
            record.cancel_requested = True

    def seal(
        self, task_id: str, state: TrajectoryState, *, native: bytes | None = None
    ) -> None:
        """End the task's trajectory; the first seal wins."""
        record = self._tasks[task_id]
        if record.final_state is not None:
            return
        if native is not None:
            if len(record.log):
                logger.warning(
                    "task %s appended trajectory events and returned a native trajectory; "
                    "the native trajectory is its final record",
                    task_id,
                )
            record.native = native
        record.log._sealed = True
        record.final_state = TrajectoryState.CANCELED if record.cancel_requested else state
        self._evict()

    def final(self, task_id: str) -> bytes | None:
        """The ended task's final trajectory as a JSON array, or None if there is none yet."""
        record = self._tasks.get(task_id)
        if record is None or record.final_state is None:
            return None
        self._touch(record.context_id)
        if record.native is not None:
            return record.native
        if not len(record.log):
            return None
        return _json_array(record.log._events)

    def context_events(self, context_id: str) -> bytes | None:
        """Every event of the context's tasks in order, as a JSON array."""
        records = self._context_records(context_id)
        if records is None:
            return None
        return _json_array([event for record in records for event in record.log._events])

    def task_page(
        self, task_id: str, after: int, max_events: int, max_bytes: int
    ) -> TrajectoryPage | None:
        record = self._tasks.get(task_id)
        if record is None:
            return None
        self._touch(record.context_id)
        # State before length: a terminal state means the length can no longer grow.
        state = record.state
        events = record.log._events
        total = len(events)
        return TrajectoryPage(
            context_id=record.context_id,
            task_id=task_id,
            state=state,
            format=record.log.format,
            events=_page(events, after, total, max_events, max_bytes),
            total=total,
        )

    def context_page(
        self, context_id: str, after: int, max_events: int, max_bytes: int
    ) -> TrajectoryPage | None:
        records = self._context_records(context_id)
        if records is None:
            return None
        task_ids = self._contexts[context_id]
        states = [record.state for record in records]
        lengths = [len(record.log) for record in records]
        current = next(
            (index for index, record in enumerate(records) if record.final_state is None),
            len(records) - 1,
        )
        if TrajectoryState.RUNNING in states:
            state = TrajectoryState.RUNNING
        elif TrajectoryState.PENDING in states:
            state = TrajectoryState.PENDING
        else:
            state = states[-1]
        page: list[bytes] = []
        offset = 0
        for record, length in zip(records, lengths):
            if len(page) >= max_events:
                break
            if after < offset + length:
                start = max(after - offset, 0)
                page.extend(
                    _page(
                        record.log._events,
                        start,
                        length,
                        max_events - len(page),
                        max_bytes - sum(map(len, page)),
                        first=not page,
                    )
                )
            offset += length
        return TrajectoryPage(
            context_id=context_id,
            task_id=task_ids[current],
            state=state,
            format=records[current].log.format,
            events=page,
            total=sum(lengths),
        )

    def _context_records(self, context_id: str) -> list[_TaskRecord] | None:
        task_ids = self._contexts.get(context_id)
        if not task_ids:
            return None
        self._touch(context_id)
        return [self._tasks[task_id] for task_id in task_ids]

    def _touch(self, context_id: str) -> None:
        if context_id in self._contexts:
            self._contexts.move_to_end(context_id)

    def _evict(self) -> None:
        total = sum(record.size_bytes for record in self._tasks.values())
        for context_id in tuple(self._contexts):
            if total <= self._max_bytes and len(self._contexts) <= self._max_contexts:
                return
            task_ids = self._contexts[context_id]
            if any(self._tasks[task_id].final_state is None for task_id in task_ids):
                continue
            del self._contexts[context_id]
            for task_id in task_ids:
                total -= self._tasks.pop(task_id).size_bytes


def _page(
    events: list[bytes],
    start: int,
    stop: int,
    max_events: int,
    max_bytes: int,
    *,
    first: bool = True,
) -> list[bytes]:
    """Events from ``start``, within both budgets; the first event of a read is always
    returned, so one larger than ``max_bytes`` cannot stall a reader."""
    page: list[bytes] = []
    used = 0
    for encoded in events[start:min(stop, start + max_events)]:
        if used + len(encoded) > max_bytes and (page or not first):
            break
        page.append(encoded)
        used += len(encoded)
    return page


def _json_array(events: list[bytes]) -> bytes:
    return b"[" + b",".join(events) + b"]"
