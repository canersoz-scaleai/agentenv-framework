"""Ctrl-C and SIGTERM for a command that tears down what its runs deployed.

``Interrupts`` counts both signals from when it is entered until it exits: inside its event loops and in the
code between them, so none is lost at a hand-off. Inside ``gather``, the first signal cancels the runs that
haven't begun their teardown and a second cancels the rest, stopping the teardown. It takes over the process's
handlers, so it has an effect only in the main thread, and a signal the process inherited as ignored stays
ignored.
"""

from __future__ import annotations

import asyncio
import signal
import threading
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

SIGNALS = (signal.SIGINT, signal.SIGTERM)

_tearing_down: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()


def tearing_down() -> None:
    """Spare the current task from the first signal: it is tearing down what it deployed."""
    task = asyncio.current_task()
    if task is not None:
        _tearing_down.add(task)


class Interrupts:
    def __init__(self) -> None:
        self.signum: int | None = None  # the first signal
        self.count = 0
        self._previous: dict[int, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_signal: Callable[[int], None] | None = None

    def __enter__(self) -> Interrupts:
        if threading.current_thread() is threading.main_thread():
            for signum in SIGNALS:
                if signal.getsignal(signum) is not signal.SIG_IGN:
                    self._previous[signum] = signal.signal(signum, self._record)
        return self

    def __exit__(self, *exc) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, signal.SIG_DFL if previous is None else previous)

    @property
    def reason(self) -> str:
        return "cancelled by Ctrl-C" if self.signum in (None, signal.SIGINT) else (
            f"cancelled by {signal.Signals(self.signum).name}")

    def run(self, main: Coroutine) -> Any:
        """``asyncio.run(main)``, taking the handlers back from the loop once it closes."""
        try:
            return asyncio.run(main)
        finally:
            for signum in self._previous:
                signal.signal(signum, self._record)

    async def gather(self, runs: Iterable[Awaitable], on_signal: Callable[[int], None]) -> list[asyncio.Future]:
        """Run ``runs`` at once until each is done, and return them. Each signal calls ``on_signal`` with the
        count so far; the first cancels the runs not tearing down, and every later one cancels them all."""
        futures = [asyncio.ensure_future(run) for run in runs]

        def cancel(count: int) -> None:
            on_signal(count)
            for future in futures:
                if count > 1 or future not in _tearing_down:
                    future.cancel()

        with self._in_loop(cancel):
            await asyncio.gather(*futures, return_exceptions=True)
        return futures

    async def stopping(self, runs: Iterable[Awaitable]) -> list:
        """Await ``runs`` at once; any signal cancels them all."""
        futures = [asyncio.ensure_future(run) for run in runs]
        with self._in_loop(lambda count: [future.cancel() for future in futures]):
            return list(await asyncio.gather(*futures))

    @contextmanager
    def _in_loop(self, on_signal: Callable[[int], None]) -> Iterator[None]:
        """Pass each signal to ``on_signal`` in the running loop. The loop's own handlers wake it from any
        thread, and they stay until it closes, counting what comes after, rather than dropping a signal that
        is already on its way."""
        loop = asyncio.get_running_loop()
        self._loop, self._on_signal = loop, on_signal
        for signum in self._previous:
            try:
                loop.add_signal_handler(signum, self._record, signum)
            except (NotImplementedError, RuntimeError):
                pass  # the process's handler reaches the loop instead
        for count in range(1, self.count + 1):
            loop.call_soon(on_signal, count)
        try:
            yield
        finally:
            self._loop = self._on_signal = None

    def _record(self, signum: int, frame=None) -> None:
        self.count += 1
        if self.signum is None:
            self.signum = signum
        if self._on_signal is not None:
            self._loop.call_soon_threadsafe(self._on_signal, self.count)
