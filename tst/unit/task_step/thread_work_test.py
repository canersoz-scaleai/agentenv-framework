"""finish_on_thread: work handed to a thread runs to its end when its caller is cancelled."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.task_step.thread_work import finish_on_thread


@pytest.mark.asyncio
async def test_a_cancelled_caller_leaves_the_work_to_finish_and_logs_how_it_ended(caplog):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(5)
        finished.set()
        return "done"

    caller = asyncio.create_task(finish_on_thread(work, "Uploading x"))
    await asyncio.to_thread(started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not finished.is_set()
    with caplog.at_level(logging.INFO, logger="agent_env.task_step.thread_work"):
        release.set()
        await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0.05)
    assert "Uploading x finished after its caller was cancelled" in caplog.text


@pytest.mark.asyncio
async def test_a_failure_after_the_cancel_is_logged(caplog):
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(5)
        raise OSError("connection reset")

    caller = asyncio.create_task(finish_on_thread(work, "Uploading x"))
    await asyncio.to_thread(started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    for _ in range(50):
        if "failed after its caller was cancelled: connection reset" in caplog.text:
            break
        await asyncio.sleep(0.02)
    assert "Uploading x failed after its caller was cancelled: connection reset" in caplog.text


def test_work_that_never_got_a_thread_before_the_loop_closed_is_released():
    ran, released, blocker = [], [], threading.Event()

    async def main():
        busy = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(busy)
        busy.submit(blocker.wait, 5)
        asyncio.create_task(finish_on_thread(lambda: ran.append(1), "Uploading x", if_never_run=lambda: released.append(1)))
        await asyncio.sleep(0.05)
        threading.Timer(0.2, blocker.set).start()

    asyncio.run(main())

    assert ran == [] and released == [1]


def test_work_still_running_when_the_loop_closes_is_not_reported_as_never_run(caplog):
    started, release, finished, released = threading.Event(), threading.Event(), threading.Event(), []

    def work():
        started.set()
        release.wait(5)
        finished.set()

    async def main():
        asyncio.create_task(finish_on_thread(work, "Uploading x", if_never_run=lambda: released.append(1)))
        await asyncio.to_thread(started.wait, 5)
        threading.Timer(0.2, release.set).start()

    with caplog.at_level(logging.INFO, logger="agent_env.task_step.thread_work"):
        asyncio.run(main())

    assert finished.is_set() and released == []
    assert "Uploading x was still running when the event loop closed" in caplog.text
    assert "did not run" not in caplog.text
