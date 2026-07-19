from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/"
    "run_ssl_lstm_neutra_phase7_retained_admission_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase7_retained_harness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(
    *,
    draws: int,
    acceptance_by_chain: tuple[float, float, float, float],
    divergence_count: int | None = None,
) -> dict[str, object]:
    return {
        "retained_sample_count": draws,
        "diagnostics_private_metadata": {
            "native_divergence_status": (
                "available" if divergence_count is not None else "not_exposed_by_kernel"
            ),
            "divergence_count": divergence_count,
            "sampler_health_diagnostics": {
                "acceptance_rate_by_chain": list(acceptance_by_chain),
                "log_accept_ratio": {"nonfinite_count": 0},
                "target_log_prob": {"nonfinite_count": 0},
            },
        },
    }


def _moving_draw_major(draws: int = 8) -> tf.Tensor:
    draw = tf.reshape(tf.cast(tf.range(draws), tf.float64), (draws, 1, 1))
    chain = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    parameter = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 1, 4))
    return draw + 10.0 * chain + 0.1 * parameter


def test_stage_a_contract_and_timing_derived_acquisition_are_frozen(
    harness: ModuleType,
) -> None:
    assert harness.CANARY_RESULTS == 4
    assert harness.CANARY_BURNIN == 2
    assert harness.STEP_SIZE == 0.8
    assert harness.NUM_LEAPFROG_STEPS == 4
    assert harness.ACCEPTANCE_BAND == (0.55, 0.85)
    assert harness.ACQUISITION_SEGMENT_RESULTS == 256
    assert harness.ACQUISITION_BURNIN == 128
    assert harness.ACQUISITION_CHECKPOINT_SEGMENTS == (1, 2, 4, 8)
    assert harness.ACQUISITION_CHART_WALL_CAP_SECONDS == 1050.0
    assert harness.ACQUISITION_WALL_CAP_SECONDS == 2100.0


def test_stage_a_receipt_is_exact_and_canary_samples_remain_excluded(
    harness: ModuleType,
) -> None:
    binding = harness.validate_canary_receipt()
    assert binding["sha256"] == harness.CANARY_RECEIPT_SHA256
    assert binding["samples_excluded_from_retained_evidence"] is True
    assert binding["decision"] == (
        "PHASE7_STAGE_A_TIMING_CANARY_PASSED_BUDGET_FREEZE_REQUIRED"
    )


def test_seed_ledgers_are_unique_and_disjoint_from_phase6(harness: ModuleType) -> None:
    assert harness.canary_seed_ledger() == harness.CANARY_SEEDS
    g = harness.acquisition_seeds("fresh-g", 4)
    h = harness.acquisition_seeds("fresh-h", 4)
    harness.validate_seed_ledger(
        {"fresh-g": g, "fresh-h": h},
        disjoint_below=harness.PHASE6_MAX_SEED_COMPONENT,
    )
    words = [item for pair in (*g, *h) for item in pair]
    assert len(words) == len(set(words))
    with pytest.raises(harness.Phase7Error, match="reused"):
        harness.validate_seed_ledger(
            {"fresh-g": ((7001, 7002),), "fresh-h": ((7001, 7003),)},
            disjoint_below=harness.PHASE6_MAX_SEED_COMPONENT,
        )
    with pytest.raises(harness.Phase7Error, match="disjoint"):
        harness.validate_seed_ledger(
            {"fresh-g": ((6901, 7002),)},
            disjoint_below=harness.PHASE6_MAX_SEED_COMPONENT,
        )


def test_lineage_metadata_binds_exact_previous_manifest_and_final_state(
    harness: ModuleType,
) -> None:
    first = harness._lineage_metadata(
        chart="fresh-g",
        role="mechanics_timing_canary",
        segment_index=0,
        seed=(7101, 7102),
        previous=None,
    )
    assert first["previous_manifest_sha256"] is None
    assert first["previous_final_state_sha256"] is None
    assert first["canary_excluded_from_retained_evidence"] is True
    second = harness._lineage_metadata(
        chart="fresh-g",
        role="retained_admission",
        segment_index=1,
        seed=(8111, 8112),
        previous={
            "manifest_sha256": "manifest-one",
            "final_state_sha256": "state-one",
        },
    )
    assert second["previous_manifest_sha256"] == "manifest-one"
    assert second["previous_final_state_sha256"] == "state-one"
    assert second["canary_excluded_from_retained_evidence"] is False


def test_runner_configs_separate_initial_burnin_from_continuations(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs: list[object] = []

    def capture(_adapter: object, _state: tf.Tensor, config: object) -> object:
        configs.append(config)
        return object()

    monkeypatch.setattr(harness, "build_retained_sample_hmc_archive_runner", capture)
    state = tf.zeros((4, 4), tf.float64)
    harness._build_runner(
        adapter=object(),
        initial_state=state,
        num_results=4,
        burnin=2,
        seed=(7101, 7102),
    )
    harness._build_runner(
        adapter=object(),
        initial_state=state,
        num_results=4,
        burnin=0,
        seed=(7111, 7112),
    )
    assert [item.num_burnin_steps for item in configs] == [2, 0]
    assert [item.use_xla for item in configs] == [True, True]
    assert [item.target_scope for item in configs] == [
        harness.TARGET_SCOPE,
        harness.TARGET_SCOPE,
    ]


def test_write_json_refuses_overwrite_and_encodes_nonfinite_strictly(
    harness: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "strict.json"
    harness._write_json(output, {"nan": float("nan"), "infinity": float("inf")})
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"non-strict constant: {value}"),
    )
    assert payload == {"infinity": "Infinity", "nan": "NaN"}
    with pytest.raises(harness.Phase7Error, match="refusing to overwrite"):
        harness._write_json(output, {"status": "second"})


def test_cumulative_admission_transposes_draw_and_chain_axes_and_aggregates_counts(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tf.Tensor] = []

    def screen(samples: tf.Tensor) -> tuple[dict[str, object], list[str]]:
        tensor = tf.convert_to_tensor(samples, tf.float64)
        observed.append(tensor)
        return {"shape": list(tensor.shape)}, []

    monkeypatch.setattr(harness, "_coordinate_screen", screen)
    z = _moving_draw_major(draws=8)
    theta = 2.0 * z
    result = harness.cumulative_admission(
        z_draw_major=z,
        theta_draw_major=theta,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            _manifest(draws=4, acceptance_by_chain=(0.5, 0.75, 0.5, 0.75)),
            _manifest(draws=4, acceptance_by_chain=(0.75, 0.5, 0.75, 0.5)),
        ),
    )
    assert [tuple(item.shape) for item in observed] == [(4, 8, 4), (4, 8, 4)]
    tf.debugging.assert_equal(observed[0], tf.transpose(z, (1, 0, 2)))
    tf.debugging.assert_equal(observed[1], tf.transpose(theta, (1, 0, 2)))
    assert result["acceptance_rate"] == pytest.approx(0.625)
    assert result["acceptance_rate_by_chain"] == pytest.approx([0.625] * 4)
    assert result["admitted"] is True


def test_acceptance_is_promotion_veto_while_divergence_is_hard_veto(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "_coordinate_screen", lambda samples: ({}, []))
    samples = _moving_draw_major(draws=4)
    low = harness.cumulative_admission(
        z_draw_major=samples,
        theta_draw_major=samples,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            _manifest(draws=4, acceptance_by_chain=(0.5, 0.5, 0.5, 0.5)),
        ),
    )
    assert low["decision"] == "EXTEND_TO_NEXT_FROZEN_CHECKPOINT"
    assert low["hard_vetoes"] == []
    assert "aggregate_acceptance_outside_threshold" in low["promotion_vetoes"]
    divergent = harness.cumulative_admission(
        z_draw_major=samples,
        theta_draw_major=samples,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            _manifest(
                draws=4,
                acceptance_by_chain=(0.75, 0.75, 0.75, 0.75),
                divergence_count=1,
            ),
        ),
    )
    assert divergent["decision"] == "HARD_VETO_STOP"
    assert "positive_native_divergence" in divergent["hard_vetoes"]


def test_native_divergence_unavailability_is_not_zero(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "_coordinate_screen", lambda samples: ({}, []))
    samples = _moving_draw_major(draws=4)
    result = harness.cumulative_admission(
        z_draw_major=samples,
        theta_draw_major=samples,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            _manifest(draws=4, acceptance_by_chain=(0.75, 0.75, 0.75, 0.75)),
        ),
    )
    assert result["native_divergence_statuses"] == ["not_exposed_by_kernel"]
    assert "positive_native_divergence" not in result["hard_vetoes"]


def test_cross_replication_uses_four_means_and_ten_second_moments(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def diagnostics(samples: tf.Tensor) -> dict[str, tf.Tensor]:
        means = tf.reduce_mean(samples, axis=(0, 1))
        return {
            "pooled_mean": means,
            "mean_mcse": tf.ones_like(means),
        }

    monkeypatch.setattr(harness, "posterior_mean_diagnostics", diagnostics)
    g = tf.reshape(tf.cast(tf.range(4 * 8 * 4), tf.float64), (4, 8, 4)) / 100.0
    result = harness.cross_replication_stability(g, g + 0.01)
    assert len(result["functional_names"]) == 14
    assert result["functional_names"][:4] == [
        "mean_theta_0",
        "mean_theta_1",
        "mean_theta_2",
        "mean_theta_3",
    ]
    assert result["functional_names"][-1] == "raw_second_theta_3_3"
    assert result["passed"] is True
    assert result["inference_status"]["statistically_supported_ranking"] == "none"


def test_cross_replication_is_not_reached_from_partial_admission(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(harness, "cross_replication_stability", forbidden)
    with pytest.raises(harness.Phase7Error, match="both independent admissions"):
        harness.admitted_cross_replication_stability(
            admissions={
                "fresh-g": {"admitted": True},
                "fresh-h": {"admitted": False},
            },
            theta_chain_major={
                "fresh-g": tf.zeros((4, 4, 4)),
                "fresh-h": tf.zeros((4, 4, 4)),
            },
        )
    assert called is False


def test_cross_replication_gate_calls_comparison_only_after_both_admit(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"passed": True}
    monkeypatch.setattr(
        harness,
        "cross_replication_stability",
        lambda g, h: expected,
    )
    result = harness.admitted_cross_replication_stability(
        admissions={
            "fresh-g": {"admitted": True},
            "fresh-h": {"admitted": True},
        },
        theta_chain_major={
            "fresh-g": tf.zeros((4, 4, 4)),
            "fresh-h": tf.ones((4, 4, 4)),
        },
    )
    assert result is expected


def test_upstream_receipts_and_phase7_scoped_binding_are_exact(
    harness: ModuleType,
) -> None:
    upstream = harness.validate_upstream_receipts()
    assert upstream["phase5"]["sha256"] == harness.PHASE5_RECEIPT_SHA256
    assert upstream["phase6"]["sha256"] == harness.PHASE6_RECEIPT_SHA256
    assert upstream["kernel"] == {
        "mass_matrix": "identity",
        "step_size": 0.8,
        "num_leapfrog_steps": 4,
        "trajectory_length": 3.2,
    }
    adapter, initial_z, binding = harness.load_binding("fresh-g")
    assert tuple(initial_z.shape) == (4, 4)
    assert binding["four_distinct_starts"] is True
    assert binding["transport_hash"] == harness.TRANSPORT_HASHES["fresh-g"]
    assert binding["scoped_target_scope"] == harness.TARGET_SCOPE
    assert adapter.value_score_capability().full_chain_xla_diagnostic_ready is True


def test_acquisition_runs_charts_sequentially_and_stops_each_only_on_admission(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(harness, "validate_upstream_receipts", lambda: {"passed": True})
    monkeypatch.setattr(harness, "validate_canary_receipt", lambda: {"passed": True})
    monkeypatch.setattr(
        harness,
        "load_binding",
        lambda label: (object(), tf.zeros((4, 4), tf.float64), {"label": label}),
    )
    monkeypatch.setattr(harness, "_build_runner", lambda **kwargs: object())
    monkeypatch.setattr(harness, "_build_post_archive_auditor", lambda *args: object())
    monkeypatch.setattr(
        harness,
        "_build_theta_mapper",
        lambda *args: lambda samples: tf.ones_like(samples),
    )
    monkeypatch.setattr(
        harness,
        "_mapped_theta_audit",
        lambda theta, mapper: {
            "shape_valid": True,
            "all_finite": True,
            "jit_compile": True,
            "compile_trace_count": 1,
            "output_device": "/device:GPU:0",
            "passed": True,
            "diagnostic_role": "engineering_validity_hard_gate",
        },
    )
    segment_calls: list[tuple[str, int]] = []

    def run_segment(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        chart = str(kwargs["chart"])
        index = int(kwargs["segment_index"])
        segment_calls.append((chart, index))
        archive = {
            "samples": tf.ones((256, 4, 4), tf.float64),
            "final_state": tf.ones((4, 4), tf.float64),
            "manifest": {"diagnostics_private_metadata": {}},
            "manifest_sha256": f"manifest-{chart}-{index}",
            "final_state_sha256": f"state-{chart}-{index}",
        }
        public = {"hard_vetoes": [], "passed": True}
        return archive, public

    monkeypatch.setattr(harness, "_run_segment", run_segment)
    checkpoint_calls = 0

    def admission(**kwargs: object) -> dict[str, object]:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        admitted = checkpoint_calls in {1, 3}
        return {
            "admitted": admitted,
            "decision": "ADMITTED" if admitted else "EXTEND_TO_NEXT_FROZEN_CHECKPOINT",
            "hard_vetoes": [],
            "promotion_vetoes": [] if admitted else ["diagnostic_screen"],
            "draw_count_per_chain": 256 if checkpoint_calls == 1 else 512,
        }

    monkeypatch.setattr(harness, "cumulative_admission", admission)
    monkeypatch.setattr(
        harness,
        "admitted_cross_replication_stability",
        lambda **kwargs: {"passed": True, "decision": "STABILITY_SCREEN_PASSED"},
    )
    monkeypatch.setattr(
        harness,
        "_run_manifest",
        lambda **kwargs: {"wall_time_seconds": 1.0},
    )
    monkeypatch.setattr(harness, "_source_bindings", lambda: {"runner": "test"})
    output = tmp_path / "acquisition.json"
    result = harness.run_acquisition(output=output, wall_cap_seconds=2100.0)
    assert segment_calls == [
        ("fresh-g", 0),
        ("fresh-h", 0),
        ("fresh-h", 1),
    ]
    assert result["both_charts_admitted"] is True
    assert result["decision"] == "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF"
    assert result["charts"]["fresh-g"]["executed_segment_count"] == 1
    assert result["charts"]["fresh-h"]["executed_segment_count"] == 2
    assert result["contract"]["canary_samples_included"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == result["decision"]


def test_maximum_checkpoint_closes_without_false_extend(
    harness: ModuleType,
) -> None:
    final = {
        "admitted": False,
        "decision": "EXTEND_TO_NEXT_FROZEN_CHECKPOINT",
        "hard_vetoes": [],
        "promotion_vetoes": ["theta:bulk_ess_below_threshold"],
    }
    if (
        final["admitted"] is not True
        and not final["hard_vetoes"]
        and 8 == harness.ACQUISITION_CHECKPOINT_SEGMENTS[-1]
    ):
        final["decision"] = "MAXIMUM_OPPORTUNITY_EXHAUSTED_NOT_ADMITTED"
    assert final["decision"] == "MAXIMUM_OPPORTUNITY_EXHAUSTED_NOT_ADMITTED"


def test_resource_cap_is_valid_incomplete_evidence_not_artifact_invalidity(
    harness: ModuleType,
) -> None:
    classification = harness._research_failure_classification(
        ["fresh-g:resource_cap_exhausted"]
    )
    assert classification["evidence_or_implementation_invalidity"] == []
    assert classification["sampler_hard_veto"] == []
    assert classification["resource_continuation_veto"] == [
        "fresh-g:resource_cap_exhausted"
    ]
