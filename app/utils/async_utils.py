import asyncio
import concurrent.futures


def run_coroutine_in_loop(coro):
    """Run an async coroutine synchronously, avoiding nested loop issues thread-safely."""
    import threading

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Check if we are running in the event loop's thread
        loop_thread_id = getattr(loop, "_thread_id", None)
        if loop_thread_id is not None and loop_thread_id == threading.get_ident():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, coro).result()
        else:
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)
