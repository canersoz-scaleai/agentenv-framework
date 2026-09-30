"""Blocking work handed to a worker thread and not abandoned halfway when its caller is cancelled."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from typing import Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def finish_on_thread(
    work: Callable[[], T], what: str, *, if_never_run: Optional[Callable[[], None]] = None
) -> T:
    """Run ``work`` on a worker thread and return its result. Cancelling the caller raises at once,
    as for ``asyncio.to_thread``, but does not stop ``work``: it runs to its end, so whatever it
    owns (a temp file it uploads, say) is released by it and not removed under it, and its outcome
    is logged. ``if_never_run`` releases what ``work`` would have, if the loop shuts down before
    a thread takes it."""
    claim = threading.Lock()

    def claimed() -> T:
        if not claim.acquire(blocking=False):
            raise asyncio.CancelledError()
        return work()

    job = asyncio.ensure_future(asyncio.to_thread(claimed))
    job.add_done_callback(functools.partial(_if_the_loop_shut_down, what, claim, if_never_run))
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        job.add_done_callback(functools.partial(_log_outcome, what))
        raise


def _if_the_loop_shut_down(
    what: str, claim: threading.Lock, release: Optional[Callable[[], None]], job: asyncio.Future
) -> None:
    """A loop that shuts down cancels the job whether or not a thread has taken ``work``."""
    if not job.cancelled():
        return
    if claim.acquire(blocking=False):
        logger.warning("%s did not run: its caller was cancelled and the event loop closed", what)
        if release is not None:
            release()
    else:
        logger.warning("%s was still running when the event loop closed; its thread runs it to its end", what)


def _log_outcome(what: str, job: asyncio.Future) -> None:
    if job.cancelled():
        return
    if job.exception() is not None:
        logger.warning("%s failed after its caller was cancelled: %s", what, job.exception())
    else:
        logger.info("%s finished after its caller was cancelled", what)
