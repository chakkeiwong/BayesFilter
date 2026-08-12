from __future__ import annotations

import json

import pytest

from bayesfilter.inference.neutra_target_validity_recovery import (
    TargetValidityAttempt,
    TargetValidityRecoveryError,
    bounded_target_validity_recovery,
)


def test_invalid_attempt_is_archived_without_optimizer_update_then_recovers(tmp_path) -> None:
    state = {"step": 17, "state_hash": "finite-state"}
    attempted = []

    def evaluate(attempt: int) -> TargetValidityAttempt:
        attempted.append(attempt)
        return TargetValidityAttempt(
            admitted=attempt == 1,
            payload={"attempt": attempt, "values": [1.0]},
            diagnostic={"invalid_rows": [3] if attempt == 0 else []},
        )

    def archive(attempt: int, result: TargetValidityAttempt):
        path = tmp_path / f"failure-{attempt}.json"
        path.write_text(json.dumps(result.diagnostic), encoding="ascii")
        return {"attempt": attempt, "path": str(path)}

    result = bounded_target_validity_recovery(
        max_retries=3,
        state_snapshot=lambda: state,
        evaluate_attempt=evaluate,
        archive_rejection=archive,
    )

    assert result.admitted is True
    assert result.accepted_attempt == 1
    assert result.payload == {"attempt": 1, "values": [1.0]}
    assert attempted == [0, 1]
    assert state == {"step": 17, "state_hash": "finite-state"}
    assert len(result.rejection_receipts) == 1
    assert json.loads((tmp_path / "failure-0.json").read_text(encoding="ascii")) == {
        "invalid_rows": [3]
    }


def test_retry_exhaustion_is_controlled_and_preserves_all_receipts() -> None:
    state = {"step": 5, "state_hash": "unchanged"}
    result = bounded_target_validity_recovery(
        max_retries=2,
        state_snapshot=lambda: state,
        evaluate_attempt=lambda attempt: TargetValidityAttempt(
            admitted=False,
            payload={"attempt": attempt},
            diagnostic={"invalid_rows": [attempt]},
        ),
        archive_rejection=lambda attempt, _result: {"attempt": attempt},
    )

    assert result.admitted is False
    assert result.accepted_attempt is None
    assert result.payload is None
    assert result.rejection_receipts == (
        {"attempt": 0},
        {"attempt": 1},
        {"attempt": 2},
    )
    assert state == {"step": 5, "state_hash": "unchanged"}


def test_rejected_attempt_that_mutates_optimizer_state_fails_closed() -> None:
    state = {"step": 9, "state_hash": "before"}

    def evaluate(_attempt: int) -> TargetValidityAttempt:
        state.update(step=10, state_hash="after")
        return TargetValidityAttempt(False, None, {"invalid_rows": [0]})

    with pytest.raises(TargetValidityRecoveryError, match="changed on rejected"):
        bounded_target_validity_recovery(
            max_retries=0,
            state_snapshot=lambda: state,
            evaluate_attempt=evaluate,
            archive_rejection=lambda _attempt, _result: {},
        )
