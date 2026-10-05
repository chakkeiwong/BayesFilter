"""Bounded no-update recovery for invalid NeuTra target batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


class TargetValidityRecoveryError(RuntimeError):
    """Raised when a rejected target batch mutates trainer state."""


@dataclass(frozen=True)
class TargetValidityAttempt:
    """One completed target evaluation before any optimizer update."""

    admitted: bool
    payload: Any
    diagnostic: Mapping[str, Any]


@dataclass(frozen=True)
class TargetValidityRecovery:
    """Terminal result of bounded target-batch admission attempts."""

    admitted: bool
    accepted_attempt: int | None
    payload: Any | None
    rejection_receipts: tuple[Mapping[str, Any], ...]


def bounded_target_validity_recovery(
    *,
    max_retries: int,
    state_snapshot: Callable[[], Mapping[str, Any]],
    evaluate_attempt: Callable[[int], TargetValidityAttempt],
    archive_rejection: Callable[[int, TargetValidityAttempt], Mapping[str, Any]],
) -> TargetValidityRecovery:
    """Try fresh batches while proving rejected attempts apply no update.

    ``evaluate_attempt`` may transport and evaluate a proposal batch but must
    not call an optimizer. The admitted payload is returned to the caller,
    which owns the single optimizer update after admission.
    """

    retries = int(max_retries)
    if retries < 0:
        raise ValueError("max_retries must be nonnegative")
    receipts = []
    for attempt in range(retries + 1):
        before = dict(state_snapshot())
        result = evaluate_attempt(attempt)
        if not isinstance(result, TargetValidityAttempt):
            raise TypeError("evaluate_attempt must return TargetValidityAttempt")
        if result.admitted:
            return TargetValidityRecovery(
                admitted=True,
                accepted_attempt=attempt,
                payload=result.payload,
                rejection_receipts=tuple(receipts),
            )
        receipts.append(dict(archive_rejection(attempt, result)))
        after = dict(state_snapshot())
        for field in ("step", "state_hash"):
            if after.get(field) != before.get(field):
                raise TargetValidityRecoveryError(
                    f"trainer state field {field!r} changed on rejected target batch"
                )
    return TargetValidityRecovery(
        admitted=False,
        accepted_attempt=None,
        payload=None,
        rejection_receipts=tuple(receipts),
    )


__all__ = [
    "TargetValidityAttempt",
    "TargetValidityRecovery",
    "TargetValidityRecoveryError",
    "bounded_target_validity_recovery",
]
