"""Whether code runs on an event loop's thread, for tests that check blocking I/O is moved off it."""

import asyncio


def on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
