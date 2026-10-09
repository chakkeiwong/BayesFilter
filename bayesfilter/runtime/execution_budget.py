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


class ExecutionBudgetExceeded(TimeoutError):
    """A scoped execution allowance ended at a cooperative boundary."""


@contextmanager
def execution_budget(*, deadline=None, check=None):
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("execution deadline must be a finite monotonic timestamp")
    if check is not None and not callable(check):
        raise TypeError("execution budget check must be callable")
    added = (() if deadline is None else (lambda: time.monotonic() < deadline,))
    if check is not None:
        added += (check,)
    token = _checks.set(_checks.get() + added)
    try:
        yield
    finally:
        _checks.reset(token)


def execution_budget_available():
    return all(check() for check in _checks.get())


def require_execution_budget():
    if not execution_budget_available():
        raise ExecutionBudgetExceeded("execution allowance exhausted at a cooperative boundary")
