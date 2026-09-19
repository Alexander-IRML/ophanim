"""Shared laptop resource gate, independent of any model or downstream project."""

from contextlib import contextmanager
from pathlib import Path
import threading


class WorkCancelled(InterruptedError):
    """Cooperative cancellation at a safe processing boundary."""


_registry_lock = threading.Lock()
_locks = {}


@contextmanager
def heavy_work(data_directory, cancelled=lambda: False):
    """Serialize heavy jobs across subsystems (and POSIX application processes).

    Cancellation while waiting never interrupts another job. Computational
    stages must additionally check their own cancellation at safe boundaries.
    """
    root = Path(data_directory).resolve()
    with _registry_lock:
        lock = _locks.setdefault(str(root), threading.Lock())
    while not lock.acquire(timeout=0.1):
        if cancelled():
            raise WorkCancelled("Cancelled while waiting for the compute slot")
    stream = None
    try:
        if cancelled():
            raise WorkCancelled("Job cancelled")
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            directory = root / "runtime"
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / "compute.lock").open("a+b")
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if cancelled():
                        raise WorkCancelled("Cancelled while waiting for the compute slot")
                    threading.Event().wait(0.1)
        yield
    finally:
        if stream is not None:
            stream.close()
        lock.release()
