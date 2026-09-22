from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_stage_resume import (
    HMC_STAGE_RESUME_SCHEMA,
    HMCStageOutcome,
    HMCStageResumeError,
    HMCStageSpec,
    completed_hmc_stage,
    load_hmc_stage_resume_checkpoint,
    run_hmc_stage_sequence,
    write_hmc_stage_resume_checkpoint,
)


ROUTE = "ordinary_hmc_tuning"
CONTRACT = {
    "target_signature": "target-v1",
    "adapter_signature": "adapter-v1",
    "config_signature": "config-v1",
    "source_revision": "source-v1",
}
NAMES = ("geometry", "bootstrap", "windowed_mass", "verification")


def _digest(value: int) -> str:
    return hashlib.sha256(str(value).encode("ascii")).hexdigest()


def _stages(
    calls: list[str],
    artifacts: dict[str, int],
    *,
    terminal: bool = True,
) -> tuple[HMCStageSpec, ...]:
    def stage(name: str, index: int):
        def run(previous_handoff):
            assert previous_handoff == (None if index == 0 else index)
            calls.append(name)
            value = index + 1
            reference = f"{name}.json"
            artifacts[reference] = value
            return completed_hmc_stage(
                handoff=value,
                handoff_kind=f"{name}_handoff",
                handoff_ref=reference,
                handoff_sha256=_digest(value),
                summary={"stage": name, "value": value},
                terminal=terminal and index == len(NAMES) - 1,
            )

        return run

    return tuple(
        HMCStageSpec(
            name=name,
            index=index,
            handoff_kind=f"{name}_handoff",
            run=stage(name, index),
        )
        for index, name in enumerate(NAMES)
    )


def _loader(artifacts: dict[str, int]):
    def load(checkpoint):
        value = artifacts[checkpoint.handoff_ref]
        assert checkpoint.handoff_sha256 == _digest(value)
        return value

    return load


def test_uninterrupted_sequence_threads_typed_handoffs() -> None:
    calls: list[str] = []
    artifacts: dict[str, int] = {}
    checkpoints = []
    result = run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages(calls, artifacts),
        checkpoint_callback=checkpoints.append,
    )

    assert calls == list(NAMES)
    assert result.completed_stage_names == NAMES
    assert result.final_handoff == 4
    assert result.terminal is True
    assert len(checkpoints) == 4
    assert checkpoints[-1].terminal is True
    assert checkpoints[-1].payload()["schema"] == HMC_STAGE_RESUME_SCHEMA
    assert "final_handoff" not in result.payload()


def test_interruption_after_boundary_resumes_with_loaded_handoff(
    tmp_path: Path,
) -> None:
    baseline_calls: list[str] = []
    baseline_artifacts: dict[str, int] = {}
    baseline_checkpoints = []
    baseline = run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages(baseline_calls, baseline_artifacts),
        checkpoint_callback=baseline_checkpoints.append,
    )

    checkpoint_path = tmp_path / "latest.json"
    calls: list[str] = []
    artifacts: dict[str, int] = {}

    def persist_then_interrupt(checkpoint) -> None:
        write_hmc_stage_resume_checkpoint(checkpoint_path, checkpoint)
        if checkpoint.stage_name == "bootstrap":
            raise RuntimeError("simulated stream interruption")

    with pytest.raises(RuntimeError, match="stream interruption"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=_stages(calls, artifacts),
            checkpoint_callback=persist_then_interrupt,
        )
    assert calls == ["geometry", "bootstrap"]

    checkpoint = load_hmc_stage_resume_checkpoint(
        checkpoint_path,
        route=ROUTE,
        run_contract=CONTRACT,
    )
    resumed_calls: list[str] = []
    resumed_checkpoints = []
    resumed = run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages(resumed_calls, artifacts),
        resume_checkpoint=checkpoint,
        handoff_loader=_loader(artifacts),
        checkpoint_callback=resumed_checkpoints.append,
    )

    assert resumed_calls == ["windowed_mass", "verification"]
    assert resumed.completed_stage_names == baseline.completed_stage_names
    assert resumed.final_handoff == baseline.final_handoff
    assert resumed_checkpoints[-1].payload() == baseline_checkpoints[-1].payload()


def test_terminal_checkpoint_is_idempotently_recoverable() -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    calls: list[str] = []
    result = run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages(calls, artifacts),
        resume_checkpoint=checkpoints[-1],
        handoff_loader=_loader(artifacts),
    )

    assert calls == []
    assert result.completed_stage_names == NAMES
    assert result.emitted_boundaries == ()
    assert result.final_handoff == 4
    assert result.terminal is True


def test_resume_requires_loader_before_any_stage_runs() -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    calls: list[str] = []
    with pytest.raises(HMCStageResumeError, match="handoff_loader"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=_stages(calls, artifacts),
            resume_checkpoint=checkpoints[0],
        )
    assert calls == []


def test_loader_failure_precedes_remaining_stage_calls() -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    calls: list[str] = []

    def fail_loader(checkpoint):
        raise HMCStageResumeError("handoff digest mismatch")

    with pytest.raises(HMCStageResumeError, match="digest mismatch"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=_stages(calls, artifacts),
            resume_checkpoint=checkpoints[0],
            handoff_loader=fail_loader,
        )
    assert calls == []


def test_route_or_contract_mismatch_precedes_loader_and_stage_calls() -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    calls: list[str] = []
    loader_calls: list[str] = []

    def loader(checkpoint):
        loader_calls.append(checkpoint.stage_name)
        return 1

    with pytest.raises(HMCStageResumeError, match="route"):
        run_hmc_stage_sequence(
            route="fixed_transport_hmc_tuning",
            run_contract=CONTRACT,
            stages=_stages(calls, artifacts),
            resume_checkpoint=checkpoints[0],
            handoff_loader=loader,
        )
    with pytest.raises(HMCStageResumeError, match="run contract"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract={**CONTRACT, "target_signature": "changed"},
            stages=_stages(calls, artifacts),
            resume_checkpoint=checkpoints[0],
            handoff_loader=loader,
        )
    assert loader_calls == []
    assert calls == []


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    payload = dict(checkpoints[0].payload())
    payload["stage_name"] = "changed"
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(HMCStageResumeError, match="hash mismatch"):
        load_hmc_stage_resume_checkpoint(path)


def test_runtime_handoff_is_not_required_to_be_json_safe() -> None:
    runtime_handoff = object()
    outcome = completed_hmc_stage(
        handoff=runtime_handoff,
        handoff_kind="geometry_handoff",
        handoff_ref="geometry.json",
        handoff_sha256=_digest(1),
    )
    assert outcome.handoff is runtime_handoff


def test_non_json_or_nonfinite_summary_is_rejected() -> None:
    with pytest.raises(HMCStageResumeError, match="unsupported object"):
        completed_hmc_stage(
            handoff=1,
            handoff_kind="geometry_handoff",
            handoff_ref="geometry.json",
            handoff_sha256=_digest(1),
            summary={"runtime_state": object()},
        )
    with pytest.raises(HMCStageResumeError, match="non-finite"):
        completed_hmc_stage(
            handoff=1,
            handoff_kind="geometry_handoff",
            handoff_ref="geometry.json",
            handoff_sha256=_digest(1),
            summary={"elapsed": float("nan")},
        )


def test_exception_or_wrong_outcome_never_gets_checkpointed() -> None:
    callbacks = []

    def fail(previous):
        raise RuntimeError("stage failed")

    failing = HMCStageSpec(
        name="geometry",
        index=0,
        handoff_kind="geometry_handoff",
        run=fail,
    )
    with pytest.raises(RuntimeError, match="stage failed"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=(failing,),
            checkpoint_callback=callbacks.append,
        )
    assert callbacks == []

    wrong = HMCStageSpec(
        name="geometry",
        index=0,
        handoff_kind="geometry_handoff",
        run=lambda previous: {"complete": True},
    )
    with pytest.raises(HMCStageResumeError, match="HMCStageOutcome"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=(wrong,),
            checkpoint_callback=callbacks.append,
        )
    assert callbacks == []


def test_wrong_handoff_kind_or_terminal_position_is_rejected() -> None:
    wrong_kind = HMCStageSpec(
        name="geometry",
        index=0,
        handoff_kind="geometry_handoff",
        run=lambda previous: completed_hmc_stage(
            handoff=1,
            handoff_kind="wrong_handoff",
            handoff_ref="geometry.json",
            handoff_sha256=_digest(1),
            terminal=True,
        ),
    )
    with pytest.raises(HMCStageResumeError, match="handoff kind"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=(wrong_kind,),
        )

    artifacts: dict[str, int] = {}
    early_terminal = list(_stages([], artifacts))
    early_terminal[0] = HMCStageSpec(
        name="geometry",
        index=0,
        handoff_kind="geometry_handoff",
        run=lambda previous: completed_hmc_stage(
            handoff=1,
            handoff_kind="geometry_handoff",
            handoff_ref="geometry.json",
            handoff_sha256=_digest(1),
            terminal=True,
        ),
    )
    with pytest.raises(HMCStageResumeError, match="terminal flag"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=early_terminal,
        )


def test_atomic_writer_round_trips_and_refuses_invalid_updates(tmp_path: Path) -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    path = tmp_path / "nested" / "checkpoint.json"
    write_hmc_stage_resume_checkpoint(path, checkpoints[1])
    loaded = load_hmc_stage_resume_checkpoint(
        path,
        route=ROUTE,
        run_contract=CONTRACT,
    )
    assert loaded.payload() == checkpoints[1].payload()
    assert not list(path.parent.glob("*.tmp"))

    with pytest.raises(HMCStageResumeError, match="rewound"):
        write_hmc_stage_resume_checkpoint(path, checkpoints[0])
    changed = type(checkpoints[1])(
        route=ROUTE,
        run_contract=CONTRACT,
        stage_name="bootstrap",
        stage_index=1,
        handoff_kind="bootstrap_handoff",
        handoff_ref="different.json",
        handoff_sha256=_digest(99),
    )
    with pytest.raises(HMCStageResumeError, match="different same-stage"):
        write_hmc_stage_resume_checkpoint(path, changed)
    with pytest.raises(HMCStageResumeError, match="stage gap"):
        write_hmc_stage_resume_checkpoint(path, checkpoints[3])
    write_hmc_stage_resume_checkpoint(path, checkpoints[2])
    write_hmc_stage_resume_checkpoint(path, checkpoints[3])
    with pytest.raises(HMCStageResumeError, match="terminal checkpoint"):
        write_hmc_stage_resume_checkpoint(path, checkpoints[2])


def test_final_nonterminal_outcome_is_not_checkpointed() -> None:
    calls: list[str] = []
    artifacts: dict[str, int] = {}
    checkpoints = []
    with pytest.raises(HMCStageResumeError, match="terminal flag"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=_stages(calls, artifacts, terminal=False),
            checkpoint_callback=checkpoints.append,
        )
    assert calls == list(NAMES)
    assert len(checkpoints) == len(NAMES) - 1


def test_nonterminal_final_checkpoint_is_rejected_before_loader() -> None:
    artifacts: dict[str, int] = {}
    checkpoints = []
    run_hmc_stage_sequence(
        route=ROUTE,
        run_contract=CONTRACT,
        stages=_stages([], artifacts),
        checkpoint_callback=checkpoints.append,
    )
    final = checkpoints[-1]
    nonterminal_final = type(final)(
        route=final.route,
        run_contract=final.run_contract,
        stage_name=final.stage_name,
        stage_index=final.stage_index,
        handoff_kind=final.handoff_kind,
        handoff_ref=final.handoff_ref,
        handoff_sha256=final.handoff_sha256,
        terminal=False,
    )
    loader_calls = []

    def loader(checkpoint):
        loader_calls.append(checkpoint.stage_name)
        return 4

    with pytest.raises(HMCStageResumeError, match="terminal flag"):
        run_hmc_stage_sequence(
            route=ROUTE,
            run_contract=CONTRACT,
            stages=_stages([], artifacts),
            resume_checkpoint=nonterminal_final,
            handoff_loader=loader,
        )
    assert loader_calls == []


def test_public_exports_are_lazy_and_available() -> None:
    import bayesfilter
    import bayesfilter.inference as inference

    assert bayesfilter.HMCStageOutcome is HMCStageOutcome
    assert inference.HMCStageOutcome is HMCStageOutcome
