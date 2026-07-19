from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_hmc_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_coordinate_report() -> dict[str, Any]:
    return {
        "rank_normalized_split_rhat": {
            "bulk": [1.0] * 4,
            "folded": [1.0] * 4,
            "maximum": [1.0] * 4,
        },
        "rank_normalized_ess": {
            "bulk": [200.0] * 4,
            "lower_5pct": [180.0] * 4,
            "upper_95pct": [170.0] * 4,
            "tail": [170.0] * 4,
        },
        "mean": {"mcse_sd_ratio": [0.05] * 4},
        "initialization_memory": {},
    }


def _manifest(draws: int, rates: tuple[float, float, float, float]) -> dict[str, Any]:
    accepted = [int(round(rate * draws)) for rate in rates]
    return {
        "retained_sample_count": draws,
        "diagnostics_private_metadata": {
            "native_divergence_status": "not_exposed_by_kernel",
            "divergence_count": None,
            "sampler_health_diagnostics": {
                "acceptance_rate_by_chain": list(rates),
                "accepted_decision_count": sum(accepted),
                "acceptance_decision_count": 4 * draws,
                "log_accept_ratio": {"nonfinite_count": 0},
                "target_log_prob": {"nonfinite_count": 0},
            },
        },
    }


class _IdentityAdapter:
    geometry_report = {"role": "test_identity"}

    @staticmethod
    def free_from_latent(value: Any) -> Any:
        return tf.convert_to_tensor(value, tf.float64)


def _moving_draws(draws: int) -> tf.Tensor:
    base = tf.reshape(tf.cast(tf.range(draws), tf.float64), (draws, 1, 1))
    chain = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    parameter = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 1, 4))
    return 0.01 * base + chain + 0.1 * parameter


def test_plan_constants_and_existing_artifact_audit(harness: ModuleType) -> None:
    assert harness.GPU_BUDGET_SECONDS == 8.0 * 60.0 * 60.0
    assert harness.KERNEL_CANDIDATES[0] == ("balanced", 0.3925, 4)
    assert harness.SEGMENT_DRAWS == (250, 250, 500, 1000)
    audit = harness.build_existing_artifact_audit()
    assert audit["qualifying_artifact_count"] == 0
    assert audit["reusable_locked_a1_retained_archives"] == []
    assert audit["decision"].startswith("NO_EXISTING_ARTIFACT_QUALIFIES")
    scopes = [row["observed"].get("target_scope") for row in audit["near_misses"]]
    assert "minimal_ssl_lstm_zhaocui_hmc_ladder:zhaocui_fixed:phase1" in scopes


def test_adapter_authority_is_bounded_and_a1_remains_target_only(
    harness: ModuleType,
) -> None:
    adapter = harness.A4CalibrationHMCAdapter()
    scoped = adapter.value_score_capability()
    base = adapter.base.value_score_capability()
    assert scoped.target_scope == harness.ACQUISITION_SCOPE
    assert scoped.full_chain_xla_diagnostic_ready is True
    assert scoped.xla_hmc_ready is True
    assert base.target_scope == harness.TARGET_SCOPE
    assert base.full_chain_xla_diagnostic_ready is False
    assert base.xla_hmc_ready is False
    covariance = adapter.factor @ tf.transpose(adapter.factor)
    assert tuple(covariance.shape) == (4, 4)
    assert bool(tf.reduce_all(tf.linalg.eigvalsh(covariance) > 0.0))


def test_affine_value_score_delegation_and_chain_rule(harness: ModuleType) -> None:
    adapter = harness.A4CalibrationHMCAdapter()
    latent = tf.constant(harness.INITIAL_STATES, tf.float64)
    free = adapter.free_from_latent(latent)
    values, scores = adapter.log_prob_and_grad(latent)
    base_values, base_scores = adapter.base.batch_value_and_score(free)
    tf.debugging.assert_near(values, base_values, atol=1.0e-10, rtol=0.0)
    tf.debugging.assert_near(scores, base_scores @ adapter.factor, atol=1.0e-8, rtol=0.0)
    scalar_free = adapter.free_from_latent(latent[1])
    scalar_value, scalar_score = adapter.log_prob_and_grad(latent[1])
    base_scalar_value, base_scalar_score = adapter.base.value_and_score(scalar_free)
    tf.debugging.assert_near(
        scalar_free,
        adapter.center + tf.linalg.matvec(adapter.factor, latent[1]),
        atol=1.0e-12,
        rtol=0.0,
    )
    tf.debugging.assert_near(scalar_value, base_scalar_value, atol=1.0e-10, rtol=0.0)
    tf.debugging.assert_near(
        scalar_score,
        tf.linalg.matvec(adapter.factor, base_scalar_score, transpose_a=True),
        atol=1.0e-8,
        rtol=0.0,
    )


def test_private_archive_readback_verifies_hash_and_shape(
    harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    archive = Path("archive")
    absolute = tmp_path / archive
    absolute.mkdir()
    samples = tf.reshape(tf.cast(tf.range(8 * 4 * 4), tf.float64), (8, 4, 4))
    final_state = samples[-1]
    sample_bytes = bytes(tf.io.serialize_tensor(samples).numpy())
    state_bytes = bytes(tf.io.serialize_tensor(final_state).numpy())
    sample_path = absolute / "unit_retained_samples.tftensor"
    state_path = absolute / "unit_final_state.tftensor"
    sample_path.write_bytes(sample_bytes)
    state_path.write_bytes(state_bytes)
    manifest = {
        "artifact_type": "bayesfilter_private_retained_sample_hmc_archive",
        "sample_shards": [
            {
                "path": str(archive / sample_path.name),
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
                "shape": [8, 4, 4],
            }
        ],
        "sidecars": {
            "final_state": {
                "path": str(archive / state_path.name),
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
            }
        },
    }
    (absolute / "unit_private_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    observed_samples, observed_state, _ = harness._read_archive(archive, "unit")
    tf.debugging.assert_equal(observed_samples, samples)
    tf.debugging.assert_equal(observed_state, final_state)
    sample_path.write_bytes(sample_bytes + b"corrupt")
    with pytest.raises(harness.AcquisitionError, match="hash mismatch"):
        harness._read_archive(archive, "unit")


def test_admission_aggregates_segments_and_checks_both_coordinates(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, ...]] = []

    def report(value: Any) -> dict[str, Any]:
        calls.append(tuple(value.shape))
        return _passing_coordinate_report()

    monkeypatch.setattr(harness, "_coordinate_diagnostics", report)
    result = harness._admission_diagnostics(
        latent_draw_major=_moving_draws(500),
        adapter=_IdentityAdapter(),
        segment_manifests=(
            _manifest(250, (0.4, 0.5, 0.6, 0.7)),
            _manifest(250, (0.6, 0.5, 0.4, 0.3)),
        ),
    )
    assert result["admitted"] is True
    assert result["acceptance_rate_by_chain"] == pytest.approx([0.5] * 4)
    assert result["acceptance_rate"] == pytest.approx(0.5)
    assert calls == [(4, 500, 4), (4, 500, 4)]
    assert result["native_divergence_interpretation"].endswith(
        "not_zero_divergences"
    )


def test_promotion_failure_extends_but_does_not_hard_veto(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = _passing_coordinate_report()
    failed["rank_normalized_split_rhat"]["maximum"][2] = 1.06
    monkeypatch.setattr(harness, "_coordinate_diagnostics", lambda _value: failed)
    result = harness._admission_diagnostics(
        latent_draw_major=_moving_draws(250),
        adapter=_IdentityAdapter(),
        segment_manifests=(_manifest(250, (0.5, 0.5, 0.5, 0.5)),),
    )
    assert result["admitted"] is False
    assert result["hard_vetoes"] == []
    assert result["decision"] == "PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS"
    assert "latent:rank_normalized_split_rhat_above_threshold" in result[
        "promotion_vetoes"
    ]
    assert "free:rank_normalized_split_rhat_above_threshold" in result[
        "promotion_vetoes"
    ]


def test_hard_veto_short_circuits_undefined_coordinate_diagnostics(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_value: Any) -> Any:
        raise AssertionError("coordinate diagnostics must not run after hard veto")

    monkeypatch.setattr(harness, "_coordinate_diagnostics", forbidden)
    result = harness._admission_diagnostics(
        latent_draw_major=tf.zeros((250, 4, 4), tf.float64),
        adapter=_IdentityAdapter(),
        segment_manifests=(_manifest(250, (0.5, 0.5, 0.5, 0.5)),),
    )
    assert result["decision"] == "HARD_VETO_STOP"
    assert "unmoved_chain" in result["hard_vetoes"]
    assert result["coordinate_diagnostics"]["status"].startswith("not_computed")


def test_canary_partial_movement_is_tuning_trigger_not_acquisition_admission(
    harness: ModuleType,
) -> None:
    hard, repair = harness._classify_canary_movement(
        tf.constant([False, True, True, True])
    )
    assert hard == []
    assert repair == ["subset_of_canary_chains_unmoved_tuning_attention"]
    hard, repair = harness._classify_canary_movement(tf.zeros([4], tf.bool))
    assert hard == ["all_canary_chains_unmoved"]
    assert repair == []
    hard, repair = harness._classify_canary_movement(tf.ones([4], tf.bool))
    assert hard == []
    assert repair == []


def test_budget_accounting_fails_closed(
    harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    path = Path("gpu.json")
    (tmp_path / path).write_text(
        json.dumps(
            {
                "run_manifest": {
                    "wall_time_seconds": 12.5,
                    "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
                    "cpu_gpu_status": "trusted_gpu_xla",
                }
            }
        ),
        encoding="utf-8",
    )
    assert harness._load_prior_gpu_seconds((path,)) == 12.5
    with pytest.raises(harness.AcquisitionError, match="counted twice"):
        harness._load_prior_gpu_seconds((path, path))
    with pytest.raises(harness.AcquisitionError, match="does not exist"):
        harness._load_prior_gpu_seconds((Path("missing.json"),))


def test_budget_lineage_requires_canary_tuning_and_prior_segments(
    harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bindings = harness._source_bindings()
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "_source_bindings", lambda: source_bindings)

    def write(
        path: Path,
        schema: str,
        status: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": schema,
            "status": status,
            "run_manifest": {
                "wall_time_seconds": 10.0,
                "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
                "cpu_gpu_status": "trusted_gpu_xla",
            },
        }
        payload["source_files"] = harness._source_bindings()
        if extra:
            payload.update(extra)
        (tmp_path / path).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    canary = Path("canary.json")
    tuning = Path("tuning.json")
    segment = Path("segment.json")
    write(canary, "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1", "PASSED")
    write(
        tuning,
        "bayesfilter.ssl_lstm.a4_hmc_tuning_screen.v1",
        "SELECTED",
        extra={"budget_lineage_artifacts": [canary.as_posix()]},
    )
    write(segment, "bayesfilter.ssl_lstm.a4_hmc_acquisition_segment.v1", "EXTEND")
    assert harness._validate_tuning_budget_lineage(
        (canary,), candidate_index=0
    ) == 10.0
    assert harness._validate_segment_budget_lineage(
        (canary, tuning, segment),
        selected_tuning=tuning,
        previous_segment_outputs=(segment,),
    ) == 30.0
    with pytest.raises(harness.AcquisitionError, match="omits required"):
        harness._validate_segment_budget_lineage(
            (canary, tuning),
            selected_tuning=tuning,
            previous_segment_outputs=(segment,),
        )


def test_repaired_canary_inherits_and_charges_failed_attempt(
    harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_sources = harness._source_bindings()
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "_source_bindings", lambda: current_sources)
    failed = Path("failed.json")
    repaired = Path("repaired.json")
    common_manifest = {
        "wall_time_seconds": 10.0,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "cpu_gpu_status": "trusted_gpu_xla",
    }
    (tmp_path / failed).write_text(
        json.dumps(
            {
                "schema_version": "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1",
                "status": "FAILED",
                "run_manifest": common_manifest,
                "source_files": [{"historical": True}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / repaired).write_text(
        json.dumps(
            {
                "schema_version": "bayesfilter.ssl_lstm.a4_hmc_gpu_canary.v1",
                "status": "PASSED_WITH_TUNING_REPAIR_TRIGGER",
                "run_manifest": common_manifest,
                "source_files": current_sources,
                "budget_lineage_artifacts": [failed.as_posix()],
            }
        ),
        encoding="utf-8",
    )
    assert harness._validate_tuning_budget_lineage(
        (failed, repaired), candidate_index=0
    ) == 20.0
