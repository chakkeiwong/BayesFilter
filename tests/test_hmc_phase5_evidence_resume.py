from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from bayesfilter.inference import hmc_phase5_evidence_resume as resume


def _ledger() -> dict:
    return {
        "schema": "bayesfilter.hmc_fixed_trajectory_private_evidence_ledger.v1",
        "bounded_selection_signature": "bounded-signature",
        "terminal_disposition": "budget_exhausted_valid",
        "attempts": [
            {
                "attempt_index": 0,
                "initial_matrix": {"selection_signature": "initial"},
                "extensions": [
                    {
                        "round_index": 0,
                        "checkpoint": 128,
                        "matrix": {"selection_signature": "terminal"},
                    }
                ],
                "terminal_matrix": {"selection_signature": "terminal"},
            }
        ],
        "attempt_count": 1,
        "private_handoff_only": True,
        "raw_samples_exposed": False,
        "raw_start_bank_exposed": False,
    }


def _contract(**changes) -> resume.Phase5EvidenceReplayContract:
    values = {
        "evidence_ledger": _ledger(),
        "adapted_mass_artifact_signature": "mass",
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "active_start_bank_signature": "active-bank",
        "source_start_bank_signature": "source-bank",
        "fixed_hmc_adapter_signature": "fixed-adapter",
        "initial_step_size": 0.25,
        "selection_root_seed": (101, 202),
        "source_checkpoint": 128,
        "extension_checkpoint": 256,
        "target_scope": "synthetic-target",
        "verification_num_results": 64,
        "verification_num_burnin_steps": 16,
        "verification_start_count": 3,
    }
    values.update(changes)
    return resume.Phase5EvidenceReplayContract(**values)


def _source_result(*, ledger=None, loop_signature="bounded-signature"):
    observed_ledger = _ledger() if ledger is None else ledger
    selection_loop = SimpleNamespace(
        private_evidence_ledger=lambda: observed_ledger,
        signature=loop_signature,
        evidence_extension_checkpoints=(128,),
        terminal_disposition="budget_exhausted_valid",
        selection=SimpleNamespace(disposition="inconclusive_evidence"),
    )
    operational = SimpleNamespace(
        final_kernel_state=SimpleNamespace(
            transform=SimpleNamespace(signature="coordinate"),
            momentum_metric=SimpleNamespace(signature="metric"),
        ),
        private_start_bank_signature="source-bank",
    )
    fixed = SimpleNamespace(
        _operational_selection_loop=selection_loop,
        adapted_mass_artifact_signature="mass",
        diagnostics={"frozen_start_bank_signature": "active-bank"},
        ladder_hmc_adapter_signature="fixed-adapter",
        initial_step_size=0.25,
        config=SimpleNamespace(seed=(101, 202), target_scope="synthetic-target"),
    )
    windowed = SimpleNamespace(operational_warmup_result=operational)
    attempt = SimpleNamespace(
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
    )
    return SimpleNamespace(
        tune_verify_repair_loop=SimpleNamespace(attempts=(attempt,))
    )


def test_replay_contract_requires_next_exact_doubling() -> None:
    with pytest.raises(ValueError, match="next exact doubling"):
        _contract(extension_checkpoint=384)


def test_replay_gate_accepts_json_list_tuple_normalization() -> None:
    observed = _ledger()
    observed["attempts"] = tuple(observed["attempts"])
    observed["attempts"][0]["extensions"] = tuple(
        observed["attempts"][0]["extensions"]
    )
    gate = resume.validate_phase5_evidence_replay(
        source_tuning_result=_source_result(ledger=observed),
        contract=_contract(),
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())
    assert gate["extension_transition_count_before_gate"] == 0
    assert gate["runtime_bearing_artifact_hashes_compared"] is False


def test_replay_gate_rejects_signature_drift() -> None:
    with pytest.raises(ValueError, match="bounded_selection_signature"):
        resume.validate_phase5_evidence_replay(
            source_tuning_result=_source_result(loop_signature="changed"),
            contract=_contract(),
        )


def test_public_resume_stops_before_controller_on_replay_mismatch(monkeypatch) -> None:
    calls = []

    class FakeTuningResult:
        adapter_signature = "adapter"
        geometry = object()
        bootstrap = object()

    monkeypatch.setattr(resume, "HMCKernelTuningResult", FakeTuningResult)
    monkeypatch.setattr(resume, "stable_adapter_signature", lambda _adapter: "adapter")

    def reject_replay(**_kwargs):
        raise ValueError("replay mismatch")

    monkeypatch.setattr(resume, "validate_phase5_evidence_replay", reject_replay)
    monkeypatch.setattr(
        resume,
        "run_hmc_tune_verify_repair_loop",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="replay mismatch"):
        resume.run_replay_gated_phase5_evidence_extension(
            adapter=object(),
            source_tuning_result=FakeTuningResult(),
            contract=_contract(),
        )
    assert calls == []
