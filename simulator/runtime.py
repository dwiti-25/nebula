"""Scoped wall-clock deadlines; execution limits never enter model schemas."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
import time

DEADLINE = ContextVar("simulation_deadline", default=None)


def expired():
    deadline = DEADLINE.get()
    return deadline is not None and time.monotonic() >= deadline


def remaining_timeout(default):
    deadline = DEADLINE.get()
    return default if deadline is None else max(0.0, min(default, deadline - time.monotonic()))


@contextmanager
def time_budget(seconds=None):
    if seconds is not None and (isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("wall-clock budget must be finite and positive")
    parent = DEADLINE.get()
    deadline = None if seconds is None else time.monotonic() + seconds
    if parent is not None:
        deadline = parent if deadline is None else min(parent, deadline)
    token = DEADLINE.set(deadline)
    try:
        yield
    finally:
        DEADLINE.reset(token)


def bounded_search(function):
    @wraps(function)
    def wrapped(*args, search_seconds=None, **kwargs):
        with time_budget(search_seconds):
            return function(*args, **kwargs)
    return wrapped
