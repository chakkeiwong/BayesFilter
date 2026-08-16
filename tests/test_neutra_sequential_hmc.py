from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_hmc import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    NeuTraHMCError,
    SequentialNeuTraHMCConfig,
    load_sequential_neutra_hmc_xla_receipt,
    qualify_sequential_neutra_hmc_xla,
    run_sequential_neutra_hmc,
    sequential_chunk_seed,
    validate_sequential_neutra_hmc_xla_receipt,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference import neutra_hmc as neutra_hmc_module


class GaussianStatusAdapter:
    parameter_dim = 2

    def __init__(self, *, status_valid: bool = True) -> None:
        self.status_valid = bool(status_valid)
        self.status_batch_sizes = []

    def log_prob_and_grad(self, z):
        values = tf.convert_to_tensor(z, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    def latent_to_position(self, z):
        return tf.convert_to_tensor(z, tf.float64)

    def log_prob_and_grad_status(self, z):
        value, score = self.log_prob_and_grad(z)
        count = tf.shape(value)[0]
        self.status_batch_sizes.append(int(z.shape[0]))
        valid = tf.fill((count,), self.status_valid)
        return value, score, {
            "status_code": tf.where(valid, tf.zeros_like(valid, tf.int32), tf.ones_like(valid, tf.int32)),
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros((count,), tf.int32),
            "min_innovation_eigenvalue": tf.ones((count,), tf.float64),
            "innovation_condition_estimate": tf.ones((count,), tf.float64),
        }


class GaussianStatusWithoutConditionAdapter(GaussianStatusAdapter):
    def log_prob_and_grad_status(self, z):
        value, score, status = super().log_prob_and_grad_status(z)
        status = dict(status)
        status.pop("innovation_condition_estimate")
        return value, score, status


class QualificationGaussianAdapter(GaussianStatusAdapter):
    parameter_dim = 9
    target_scope = "sequential_xla_qualification_fixture"

    def __init__(self, *, full_chain_ready: bool = False) -> None:
        super().__init__()
        self.full_chain_ready = bool(full_chain_ready)

    def adapter_signature(self) -> str:
        return "sequential-xla-qualification-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=self.full_chain_ready,
            runtime_backend="tensorflow_gaussian_test_fixture",
            evidence_path="tests/test_neutra_sequential_hmc.py",
            target_scope=self.target_scope,
            nonclaims=("qualification fixture only",),
        )


def tiny_config() -> SequentialNeuTraHMCConfig:
    return SequentialNeuTraHMCConfig(
        step_size=0.05,
        num_leapfrog_steps=2,
        seed=(20260722, 1),
        warmup_chunk_size=4,
        warmup_min_results=4,
        warmup_window_results=4,
        warmup_max_results=4,
        retained_chunk_size=4,
        retained_min_results=4,
        retained_max_results=4,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_max=1.0,
        chain_count=2,
        use_xla=False,
    )


def test_policy_defaults_match_repository_sequential_contract() -> None:
    config = SequentialNeuTraHMCConfig(
        step_size=0.1,
        num_leapfrog_steps=4,
        seed=(1, 2),
    )
    assert config.payload()["policy_id"] == NEUTRA_SEQUENTIAL_HMC_POLICY_ID
    assert config.warmup_min_results == 2000
    assert config.warmup_window_results == 1000
    assert config.warmup_max_results == 10000
    assert config.retained_min_results == 1000
    assert config.retained_max_results == 10000
    assert config.warmup_rhat_max == pytest.approx(1.05)
    assert config.retained_rhat_max == pytest.approx(1.01)
    assert config.delta_h_abs_max == pytest.approx(1000.0)
    assert config.acceptance_min == pytest.approx(0.35)
    assert config.acceptance_max == pytest.approx(0.95)


def test_sequential_hmc_forbids_one_leapfrog_step() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        SequentialNeuTraHMCConfig(
            step_size=0.1,
            num_leapfrog_steps=1,
            seed=(1, 2),
        )


def test_chunk_policy_uses_only_declared_mechanics_vetoes() -> None:
    unavailable = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        chain_moved=(True, True),
        native_divergence_status="not_exposed_by_kernel",
        native_divergence_count=None,
    )
    assert unavailable == ()

    divergent = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        chain_moved=(True, True),
        native_divergence_status="available",
        native_divergence_count=1,
    )
    assert divergent == ("positive_native_divergence",)

    movement_veto = neutra_hmc_module._chunk_policy_vetoes(
        samples_finite=True,
        log_accept_finite=True,
        target_finite=True,
        proposed_finite=True,
        target_score_finite=True,
        delta_h_finite=True,
        target_status_passed=True,
        chain_moved=(True, False),
        native_divergence_status="not_exposed_by_kernel",
        native_divergence_count=None,
    )
    assert movement_veto == ("chain_without_movement",)


def test_sequential_controller_accepts_exact_external_chunk_callback(tmp_path) -> None:
    calls = []

    def run_chunk(state, seed, config):
        state = tf.convert_to_tensor(state, tf.float64)
        calls.append(
            {
                "state_shape": tuple(state.shape),
                "seed": tuple(seed),
                "leapfrog": config.num_leapfrog_steps,
            }
        )
        offsets = tf.reshape(tf.range(1, 5, dtype=tf.float64), (4, 1, 1))
        samples = state[tf.newaxis, :, :] + offsets * 0.01
        shape = (4, 2)
        trace = {
            "is_accepted": tf.ones(shape, tf.bool),
            "log_accept_ratio": tf.fill(
                shape, tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_log_prob": tf.zeros(shape, tf.float64),
            "proposed_target_log_prob": tf.zeros(shape, tf.float64),
            "target_score": tf.zeros((4, 2, 2), tf.float64),
            "delta_h": tf.fill(
                shape, -tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_status_code": tf.zeros(shape, tf.int32),
            "target_valid_pre_regularized_score": tf.ones(shape, tf.bool),
            "target_floor_count_value": tf.zeros(shape, tf.int32),
            "target_min_innovation_eigenvalue": tf.ones(shape, tf.float64),
        }
        return samples, trace

    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.zeros((2, 2), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "external",
        archive_label="external",
        run_chunk=run_chunk,
    )
    assert calls
    assert {call["state_shape"] for call in calls} == {(2, 2)}
    assert {call["leapfrog"] for call in calls} == {2}
    assert Path(result.archive["manifest_path"]).is_file()
    assert result.diagnostics["hard_vetoes"] == []


def test_target_status_from_trace_vetoes_invalid_transition() -> None:
    diagnostics = neutra_hmc_module._target_status_from_trace(
        {
            "target_status_code": tf.constant(((0, 1),), tf.int32),
            "target_valid_pre_regularized_score": tf.constant(((True, False),)),
            "target_floor_count_value": tf.constant(((0, 1),), tf.int32),
            "target_min_innovation_eigenvalue": tf.constant(((1.0, 0.0),), tf.float64),
        }
    )
    assert diagnostics is not None
    assert diagnostics["passed"] is False
    assert diagnostics["status_nonvalid_count"] == 1


def test_chunk_seeds_are_phase_separated_and_deterministic() -> None:
    first = sequential_chunk_seed((4, 5), phase_index=0, chunk_index=0)
    assert first == sequential_chunk_seed((4, 5), phase_index=0, chunk_index=0)
    assert first != sequential_chunk_seed((4, 5), phase_index=0, chunk_index=1)
    assert first != sequential_chunk_seed((4, 5), phase_index=1, chunk_index=0)


def test_movement_is_measured_from_the_immediate_pre_chunk_state() -> None:
    pre_chunk = tf.constant(((3.0, 4.0), (5.0, 6.0)), tf.float64)
    frozen = tf.repeat(pre_chunk[None, ...], repeats=4, axis=0)
    assert neutra_hmc_module._chain_moved(pre_chunk, frozen).numpy().tolist() == [
        False,
        False,
    ]


def test_tiny_run_archives_warmup_separately_and_excludes_it(tmp_path) -> None:
    adapter = GaussianStatusAdapter()
    result = run_sequential_neutra_hmc(
        adapter,
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="tiny",
    )
    assert result.warmup_results_per_chain == 4
    assert result.retained_results_per_chain in {0, 4}
    assert result.metadata["warmup_excluded_from_posterior"] is True
    assert set(result.diagnostics["warmup"]) >= {
        "hmc_coordinates",
        "model_parameters",
    }
    assert result.diagnostics["acceptance_role"] == (
        "explanatory_only_not_a_convergence_veto"
    )
    assert result.archive["warmup_chunk_count"] == 1
    assert list((tmp_path / "run" / "warmup").glob("*samples.tftensor"))
    assert adapter.status_batch_sizes
    assert set(adapter.status_batch_sizes) == {8}
    assert list((tmp_path / "run" / "warmup").glob("*delta_h.tftensor"))
    if result.retained_results_per_chain:
        assert list((tmp_path / "run" / "retained").glob("*samples.tftensor"))


def test_target_status_failure_is_hard_veto(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(status_valid=False),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="status-veto",
    )
    assert result.passed is False
    assert result.stop_reason == "hard_veto"
    assert "target_status_veto" in result.diagnostics["hard_vetoes"]


def test_target_status_accepts_q20_schema_without_condition_estimate() -> None:
    diagnostics = neutra_hmc_module._target_status(
        GaussianStatusWithoutConditionAdapter(),
        tf.zeros((4, 2, 2), tf.float64),
    )
    assert diagnostics["passed"] is True
    assert diagnostics["maximum_innovation_condition_estimate"] is None
    assert diagnostics["innovation_condition_estimate_status"] == (
        "not_exposed_by_target"
    )


def test_run_rejects_nonempty_output_root(tmp_path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(NeuTraHMCError, match="new or empty"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            tiny_config(),
            archive_root=root,
            archive_label="occupied",
        )


def test_budget_refusal_is_archived_as_a_resource_cap(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="budget-cap",
        budget_check=lambda _requested_work: False,
    )
    assert result.passed is False
    assert result.stop_reason == "hard_veto"
    assert "campaign_resource_cap" in result.diagnostics["hard_vetoes"]
    assert result.archive["warmup_chunk_count"] == 0


def test_archived_result_payload_uses_defined_schema(tmp_path) -> None:
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "run",
        archive_label="payload-schema",
        budget_check=lambda _requested_work: False,
    )
    payload = result.payload()
    assert payload["schema"] == "bayesfilter.neutra.sequential_hmc_result.v1"
    assert payload["stop_reason"] == "hard_veto"


def _fixed_diagnostics(max_rhat: float) -> dict[str, object]:
    coordinate = {
        "all_finite": True,
        "max_rhat": max_rhat,
        "min_bulk_ess": 100.0,
        "min_tail_ess": 100.0,
        "rhat_by_parameter": [max_rhat, max_rhat],
        "bulk_ess_by_parameter": [100.0, 100.0],
        "tail_ess_by_parameter": [100.0, 100.0],
        "rhat_threshold": max_rhat,
    }
    return {
        "hmc_coordinates": coordinate,
        "model_parameters": coordinate,
        "primary_diagnostic_coordinate": "hmc_coordinates_z",
        "physical_coordinate_role": "explanatory_only",
        "all_finite": True,
        "max_rhat": max_rhat,
        "min_bulk_ess": 100.0,
        "min_tail_ess": 100.0,
        "rhat_threshold": max_rhat,
    }


def test_archived_rhat_promotion_thresholds_are_strict(
    tmp_path, monkeypatch
) -> None:
    warmup_equal = iter((_fixed_diagnostics(1.05),))
    monkeypatch.setattr(
        neutra_hmc_module,
        "_diagnostics",
        lambda *_args, **_kwargs: next(warmup_equal),
    )
    warmup_result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "warmup-equality",
        archive_label="warmup-equality",
        run_chunk=_checkpoint_run_chunk()[1],
    )
    assert warmup_result.stop_reason == "warmup_cap_not_ready"
    assert warmup_result.warmup_results_per_chain == 4
    assert warmup_result.retained_results_per_chain == 0

    warmup_then_retained = iter(
        (_fixed_diagnostics(1.049), _fixed_diagnostics(1.01))
    )
    monkeypatch.setattr(
        neutra_hmc_module,
        "_diagnostics",
        lambda *_args, **_kwargs: next(warmup_then_retained),
    )
    retained_result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=tmp_path / "retained-equality",
        archive_label="retained-equality",
        run_chunk=_checkpoint_run_chunk()[1],
    )
    assert retained_result.stop_reason == "retained_cap_not_passed"
    assert retained_result.retained_results_per_chain == 4
    assert retained_result.diagnostics["movement_role"] == "hard_validity_veto"


def _checkpoint_run_chunk(*, fail_on_call: int | None = None):
    calls = []

    def run_chunk(state, seed, config):
        del seed
        state = tf.convert_to_tensor(state, tf.float64)
        calls.append(tf.identity(state))
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("simulated process interruption")
        draws = tf.cast(
            tf.range(1, config.warmup_chunk_size + 1), tf.float64
        )[:, None, None]
        chain_offsets = tf.reshape(
            tf.cast(tf.range(config.chain_count), tf.float64), (1, -1, 1)
        )
        samples = state[None, :, :] + 0.01 * draws + 0.001 * chain_offsets
        shape = (config.warmup_chunk_size, config.chain_count)
        trace = {
            "is_accepted": tf.ones(shape, tf.bool),
            "log_accept_ratio": tf.fill(
                shape, tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_log_prob": tf.zeros(shape, tf.float64),
            "proposed_target_log_prob": tf.zeros(shape, tf.float64),
            "target_score": tf.zeros(
                (*shape, int(state.shape[-1])), tf.float64
            ),
            "delta_h": tf.fill(
                shape, -tf.math.log(tf.constant(0.7, tf.float64))
            ),
            "target_status_code": tf.zeros(shape, tf.int32),
            "target_valid_pre_regularized_score": tf.ones(shape, tf.bool),
            "target_floor_count_value": tf.zeros(shape, tf.int32),
            "target_min_innovation_eigenvalue": tf.ones(shape, tf.float64),
        }
        return samples, trace

    return calls, run_chunk


def _interrupted_checkpoint(tmp_path, *, label: str = "resume"):
    root = tmp_path / label
    config = SequentialNeuTraHMCConfig(
        **{
            **tiny_config().__dict__,
            "warmup_min_results": 8,
            "warmup_window_results": 8,
            "warmup_max_results": 8,
        }
    )
    calls, run_chunk = _checkpoint_run_chunk(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            config,
            archive_root=root,
            archive_label=label,
            run_chunk=run_chunk,
        )
    checkpoint = root / f"{label}-checkpoint.json"
    assert checkpoint.is_file()
    assert len(calls) == 2
    return root, checkpoint, config


def test_resume_continues_from_hash_bound_checkpoint(tmp_path) -> None:
    root, _checkpoint, config = _interrupted_checkpoint(tmp_path)
    resumed_calls, resumed_chunk = _checkpoint_run_chunk()
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        config,
        archive_root=root,
        archive_label="resume",
        run_chunk=resumed_chunk,
        resume=True,
    )
    assert result.archive["resumed"] is True
    assert resumed_calls
    archived = tf.io.parse_tensor(
        next((root / "warmup").glob("*samples.tftensor")).read_bytes(),
        out_type=tf.float64,
    )
    tf.debugging.assert_equal(resumed_calls[0], archived[-1])


def test_resume_rejects_config_mismatch(tmp_path) -> None:
    root, _checkpoint, _config = _interrupted_checkpoint(tmp_path, label="mismatch")
    changed = SequentialNeuTraHMCConfig(
        **{**tiny_config().__dict__, "step_size": 0.051}
    )
    with pytest.raises(NeuTraHMCError, match="run contract mismatch"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            changed,
            archive_root=root,
            archive_label="mismatch",
            run_chunk=_checkpoint_run_chunk()[1],
            resume=True,
        )


def test_resume_rejects_corrupted_tensor_receipt(tmp_path) -> None:
    root, checkpoint, config = _interrupted_checkpoint(tmp_path, label="corrupt")
    payload = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    sample = Path(payload["phase_rows"]["warmup"][0]["sample_receipt"]["path"])
    sample.write_bytes(sample.read_bytes() + b"corrupt")
    with pytest.raises(NeuTraHMCError, match="tensor receipt hash mismatch"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            config,
            archive_root=root,
            archive_label="corrupt",
            run_chunk=_checkpoint_run_chunk()[1],
            resume=True,
        )


def test_resume_rejects_terminal_root(tmp_path) -> None:
    root = tmp_path / "terminal"
    result = run_sequential_neutra_hmc(
        GaussianStatusAdapter(),
        tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
        tiny_config(),
        archive_root=root,
        archive_label="terminal",
        run_chunk=_checkpoint_run_chunk()[1],
    )
    assert Path(result.archive["manifest_path"]).is_file()
    with pytest.raises(NeuTraHMCError, match="already terminal"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            tiny_config(),
            archive_root=root,
            archive_label="terminal",
            run_chunk=_checkpoint_run_chunk()[1],
            resume=True,
        )


def test_resume_rejects_orphan_partial_block(tmp_path) -> None:
    root, _checkpoint, config = _interrupted_checkpoint(tmp_path, label="orphan")
    (root / "warmup" / "unledgered-partial.tftensor").write_bytes(b"partial")
    with pytest.raises(NeuTraHMCError, match="orphan partial-block"):
        run_sequential_neutra_hmc(
            GaussianStatusAdapter(),
            tf.constant(((0.0, 0.0), (1.0, -1.0)), tf.float64),
            config,
            archive_root=root,
            archive_label="orphan",
            run_chunk=_checkpoint_run_chunk()[1],
            resume=True,
        )


def test_exact_sequential_xla_qualification_and_receipt_validation(tmp_path) -> None:
    state = tf.reshape(tf.linspace(-0.4, 0.4, 36), (4, 9))
    receipt = qualify_sequential_neutra_hmc_xla(
        adapter=QualificationGaussianAdapter(),
        initial_state=state,
        step_size=0.1,
        num_leapfrog_steps=2,
        seed=(20260809, 10),
        evidence_path=tmp_path / "qualification.json",
        chunk_results=4,
        value_score_atol=1.0e-12,
    )
    assert receipt.tracing_count == 1
    assert receipt.all_chains_moved is True
    loaded = load_sequential_neutra_hmc_xla_receipt(receipt.evidence_path)
    assert loaded.payload() == receipt.payload()
    config = SequentialNeuTraHMCConfig(
        step_size=0.1,
        num_leapfrog_steps=2,
        seed=(20260809, 10),
        warmup_chunk_size=4,
        warmup_min_results=4,
        warmup_window_results=4,
        warmup_max_results=4,
        retained_chunk_size=4,
        retained_min_results=4,
        retained_max_results=4,
        chain_count=4,
        use_xla=True,
        target_status_required=True,
        primary_diagnostic_coordinate="hmc_coordinates_z",
        retained_ess_required=False,
        xla_qualification_required=True,
    )
    validate_sequential_neutra_hmc_xla_receipt(
        loaded,
        adapter=QualificationGaussianAdapter(full_chain_ready=True),
        initial_state=state,
        config=config,
    )
