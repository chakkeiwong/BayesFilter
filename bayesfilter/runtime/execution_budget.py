"""Host-side cooperative execution limits, separate from numerical identity.

Scopes compose by intersection. A controller can stop at a safe boundary
without adding wall-clock values to its candidate identities or random seeds.
Running TensorFlow calls cannot be interrupted here; a process supervisor owns
the final hard limit.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time


_checks = ContextVar("bayesfilter_execution_budget_checks", default=())
_supervised_wall = ContextVar("bayesfilter_supervised_wall_budget", default=False)
_progress = ContextVar("bayesfilter_execution_progress", default=None)


class ExecutionBudgetExceeded(TimeoutError):
    """A scoped execution allowance ended at a cooperative boundary."""


@contextmanager
def execution_budget(*, deadline=None, check=None, replace_wall_time_limit=False):
    """Intersect allowances; explicitly supervised time may replace a config clock.

    Replacement affects only a controller's cumulative wall limit. Enclosing
    execution scopes, numerical work limits, identities and seeds still apply.
    The caller must supply a live check owned by a bounded process supervisor.
    """
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("execution deadline must be a finite monotonic timestamp")
    if check is not None and not callable(check):
        raise TypeError("execution budget check must be callable")
    if type(replace_wall_time_limit) is not bool or (replace_wall_time_limit and check is None):
        raise ValueError("supervised wall replacement requires an explicit live check")
    added = (() if deadline is None else (lambda: time.monotonic() < deadline,))
    if check is not None:
        added += (check,)
    token = _checks.set(_checks.get() + added)
    wall_token = _supervised_wall.set(_supervised_wall.get() or replace_wall_time_limit)
    try:
        yield
    finally:
        _supervised_wall.reset(wall_token)
        _checks.reset(token)


def execution_budget_available(*, elapsed_seconds=None, wall_time_limit_seconds=None):
    if not all(check() for check in _checks.get()):
        return False
    if wall_time_limit_seconds is not None and not _supervised_wall.get():
        if elapsed_seconds is None:
            raise ValueError("a cumulative wall limit requires elapsed time")
        return elapsed_seconds < wall_time_limit_seconds
    return True


def configured_wall_time_limit(seconds):
    """Resolve a legacy internal clock without changing serialized config."""
    return None if _supervised_wall.get() else seconds


@contextmanager
def execution_progress(callback):
    """Observe completed host-side work without changing numerical decisions."""
    if not callable(callback):
        raise TypeError("execution progress callback must be callable")
    token = _progress.set(callback)
    try:
        yield
    finally:
        _progress.reset(token)


def report_execution_progress(phase, identity):
    callback = _progress.get()
    if callback is not None:
        callback(phase, identity)


def require_execution_budget():
    if not execution_budget_available():
        raise ExecutionBudgetExceeded("execution allowance exhausted at a cooperative boundary")
