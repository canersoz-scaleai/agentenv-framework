import asyncio
import json
import time

import httpx
import pytest
from agentenv_protocol.a2a_agent import TrajectoryState

from agent_env.a2a_agent.trajectory_follower import (
    TrajectoryBatch,
    follow_trajectory,
    live_trajectory_endpoint,
)
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.task_step.snapshot_utils import live_trajectory
from agent_env.task_step.snapshot_utils.live_trajectory import (
    LiveTrajectoryChunks,
    following_live_trajectory,
)

_ENDPOINT = "http://agent.test/ext/trajectory"


def _page(after: int, events: list, state: str, *, has_more: bool = False) -> dict:
    return {
        "context_id": "ctx",
        "task_id": "task-1",
        "state": state,
        "format": "test-events/1",
        "events": events,
        "next": after + len(events),
        "has_more": has_more,
    }


class _Agent:
    """Answers each cursor read from a script of responses, recording the ``after`` it was sent."""

    def __init__(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)
        self.afters: list[int] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.afters.append(json.loads(request.content)["after"])
        return self.answers.pop(0)


async def _follow(agent: _Agent) -> list[TrajectoryBatch]:
    batches: list[TrajectoryBatch] = []

    async def sink(batch: TrajectoryBatch) -> None:
        batches.append(batch)

    async with agent.client() as client:
        await follow_trajectory(
            _ENDPOINT, "task-1", sink, poll_interval_seconds=0, stop=asyncio.Event(), client=client
        )
    return batches


def test_the_live_endpoint_comes_from_the_card():
    def card(request: dict) -> dict:
        return {
            "capabilities": {
                "extensions": [
                    {
                        "uri": "urn:agentenv:trajectory/v1",
                        "params": {"endpoint": "/ext/trajectory", "methods": {"get": {"method": "POST", "request": request}}},
                    }
                ]
            }
        }

    live = card({"oneOf": [{"required": ["task_id"]}, {"required": ["task_id", "after"], "optional": ["limit"]}]})
    assert live_trajectory_endpoint(live, "http://agent.test") == _ENDPOINT
    assert live_trajectory_endpoint(card({"required": ["task_id"]}), "http://agent.test") is None
    assert live_trajectory_endpoint({}, "http://agent.test") is None


@pytest.mark.asyncio
async def test_pages_are_followed_until_the_task_ends_and_every_event_is_read():
    agent = _Agent(
        httpx.Response(200, json=_page(0, [], "pending")),
        httpx.Response(200, json=_page(0, [{"i": 0}], "running")),
        httpx.Response(200, json=_page(1, [{"i": 1}], "completed", has_more=True)),
        httpx.Response(200, json=_page(2, [{"i": 2}], "completed")),
    )

    batches = await _follow(agent)

    assert agent.afters == [0, 0, 1, 2]
    assert [(b.after, b.next, b.events, b.last) for b in batches] == [
        (0, 1, [{"i": 0}], False),
        (1, 2, [{"i": 1}], False),
        (2, 3, [{"i": 2}], True),
    ]
    assert batches[-1].state is TrajectoryState.COMPLETED


@pytest.mark.asyncio
async def test_an_ended_task_with_nothing_new_still_gets_a_last_batch():
    agent = _Agent(httpx.Response(200, json=_page(0, [], "failed")))

    batches = await _follow(agent)

    assert [(b.events, b.state, b.last) for b in batches] == [([], TrajectoryState.FAILED, True)]


@pytest.mark.asyncio
async def test_a_server_error_is_retried(monkeypatch):
    monkeypatch.setattr("agent_env.a2a_agent.trajectory_follower._MAX_BACKOFF_SECONDS", 0)
    agent = _Agent(
        httpx.Response(503),
        httpx.Response(200, json=_page(0, [{"i": 0}], "completed")),
    )

    batches = await _follow(agent)

    assert agent.afters == [0, 0]
    assert batches[-1].events == [{"i": 0}]


@pytest.mark.asyncio
async def test_a_task_the_agent_no_longer_knows_ends_following():
    agent = _Agent(httpx.Response(404, json={"detail": "Unknown task"}))

    assert await _follow(agent) == []


@pytest.mark.asyncio
async def test_a_client_error_is_raised():
    agent = _Agent(httpx.Response(400, json={"detail": "bad"}))

    with pytest.raises(httpx.HTTPStatusError):
        await _follow(agent)


@pytest.mark.asyncio
async def test_chunks_meta_and_end_are_stored_once(tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    chunks = LiveTrajectoryChunks(store, "prefix/", "turn")
    batch = TrajectoryBatch(
        after=0, next=2, events=[{"i": 0}, {"i": 1}], format="test-events/1",
        state=TrajectoryState.COMPLETED, last=True,
    )

    await chunks(batch)
    await LiveTrajectoryChunks(store, "prefix/", "turn")(batch)

    assert sorted(store.list("prefix/")) == [
        "prefix/turn/live/00000000-00000002.jsonl",
        "prefix/turn/live/end.json",
        "prefix/turn/live/meta.json",
    ]
    assert store.get(store.object_url("prefix/turn/live/00000000-00000002.jsonl")) == b'{"i":0}\n{"i":1}\n'
    assert json.loads(store.get(store.object_url("prefix/turn/live/end.json"))) == {"state": "completed", "next": 2}


@pytest.mark.asyncio
async def test_a_failing_write_is_retried_then_raised(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    attempts: list[str] = []

    def failing_put(key, body, *, content_type):
        attempts.append(key)
        raise OSError("disk full")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(store, "put", failing_put)
    monkeypatch.setattr(live_trajectory.asyncio, "sleep", no_sleep)
    batch = TrajectoryBatch(after=0, next=1, events=[{"i": 0}], format=None, state=TrajectoryState.RUNNING, last=False)

    with pytest.raises(OSError):
        await LiveTrajectoryChunks(store, "prefix/", "turn")(batch)
    assert len(attempts) == live_trajectory._WRITE_ATTEMPTS


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("turn failed"), asyncio.CancelledError()])
async def test_a_failed_turn_stops_its_follower_without_draining(monkeypatch, tmp_path, error):
    """An agent that never ends: only a drain would keep the follower reading."""
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, real=httpx.AsyncClient, **kw: real(
            *a, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_page(0, [], "running"))), **kw
        ),
    )
    store = LocalFilesystemObjectStore(str(tmp_path))
    started = time.monotonic()

    with pytest.raises(type(error)):
        async with following_live_trajectory(
            _ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0
        ) as on_sent:
            on_sent("task-1")
            await asyncio.sleep(0.05)
            raise error

    assert time.monotonic() - started < 2
    assert store.list("prefix/") == []
