from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/"
    "run_ssl_lstm_neutra_phase6_transformed_hmc_tuning_2026_07_16.py"
)
QUEUE_SCRIPT = (
    ROOT
    / "docs/benchmarks/"
    "queue_ssl_lstm_neutra_phase6_canary_2026_07_16.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6_hmc_harness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def queue_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase6_queue_harness", QUEUE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(
    accepted_by_chain: tuple[float, float, float, float],
    *,
    draws: int = 4,
) -> dict[str, tf.Tensor]:
    accepted = tf.stack(
        [
            tf.concat(
                (
                    tf.ones((round(draws * rate),), tf.bool),
                    tf.zeros((draws - round(draws * rate),), tf.bool),
                ),
                axis=0,
            )
            for rate in accepted_by_chain
        ],
        axis=1,
    )
    return {
        "is_accepted": accepted,
        "log_accept_ratio": tf.zeros((draws, 4), tf.float64),
        "target_log_prob": -tf.ones((draws, 4), tf.float64),
    }


def _moving_samples(*, draws: int = 4) -> tf.Tensor:
    time = tf.reshape(tf.cast(tf.range(draws), tf.float64), (draws, 1, 1))
    chains = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    parameters = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 1, 4))
    return 0.2 * time + chains + 0.1 * parameters


def _row(step: float, viable: bool, acceptance: float) -> dict[str, object]:
    return {
        "step_size": step,
        "num_leapfrog_steps": 4,
        "diagnostics": {
            "viable": viable,
            "acceptance_rate_by_chain": [acceptance] * 4,
        },
    }


def test_plan_constants_and_seed_roles_are_prospective(harness: ModuleType) -> None:
    assert harness.PILOT_ACCEPTANCE_BAND == (0.50, 0.90)
    assert harness.CONFIRMATION_ACCEPTANCE_BAND == (0.55, 0.85)
    assert harness.TARGET_ACCEPTANCE == 0.70
    assert harness.H_REPAIR_SEED == 6901
    assert harness.LADDER_R2_RUNNER_SHA256 == (
        "ea903b2af5cdd8476aea1bc38841c6379c226ce51a0668999525f426015f97c3"
    )
    assert harness.INITIAL_SCALE_GRID == (0.05, 0.10, 0.20, 0.40)
    assert harness.TRAJECTORY_GRID == (2, 4, 8, 16)
    assert harness.TRAJECTORY_PRIORITY == (8, 4, 16, 2)
    seed_words = [
        *harness.CANARY_SEEDS["fresh-g"],
        *harness.CANARY_SEEDS["fresh-h"],
        *range(6300, 6308),
        *range(6400, 6408),
        *range(6500, 6504),
        *range(6600, 6604),
        *harness.CONFIRMATION_SEEDS.values(),
        harness.H_REPAIR_SEED,
    ]
    assert len(seed_words) == len(set(seed_words))


def test_identity_mass_fixture_has_correct_covariance_and_kinetic_energy(
    harness: ModuleType,
) -> None:
    fixture = harness.identity_mass_fixture()
    assert fixture["passed"] is True
    assert fixture["mass_matrix"] == tf.eye(4, dtype=tf.float64).numpy().tolist()
    assert fixture["precision_matrix"] == fixture["mass_matrix"]
    assert fixture["momentum_covariance"] == fixture["mass_matrix"]
    assert fixture["explicit_residual_max_abs"] == 0.0


def test_per_chain_diagnostics_do_not_allow_aggregate_masking(
    harness: ModuleType,
) -> None:
    samples = _moving_samples()
    samples = tf.concat((samples[:, :3], tf.zeros((4, 1, 4), tf.float64)), axis=1)
    diagnostics = harness.diagnose_run(
        samples=samples,
        initial_state=samples[0],
        trace=_trace((1.0, 1.0, 1.0, 0.0)),
        acceptance_band=(0.50, 0.90),
        min_movement=0.25,
        min_rms_jump=0.05,
    )
    assert diagnostics["acceptance_rate"] == pytest.approx(0.75)
    assert diagnostics["viable"] is False
    assert "unmoved_chain" in diagnostics["hard_vetoes"]
    assert "per_chain_acceptance_above_band" in diagnostics["acceptance_vetoes"]
    assert "per_chain_acceptance_below_band" in diagnostics["acceptance_vetoes"]


def test_native_divergence_unavailability_is_not_zero(harness: ModuleType) -> None:
    diagnostics = harness.diagnose_run(
        samples=_moving_samples(),
        initial_state=tf.zeros((4, 4), tf.float64),
        trace=_trace((0.75, 0.75, 0.75, 0.75)),
        acceptance_band=(0.50, 0.90),
        min_movement=0.25,
        min_rms_jump=0.05,
    )
    assert diagnostics["native_divergence_status"] == "unavailable_not_zero"
    assert diagnostics["native_divergence_count"] is None


def test_positive_native_divergence_is_a_hard_veto(harness: ModuleType) -> None:
    trace = _trace((0.75, 0.75, 0.75, 0.75))
    trace["divergence"] = tf.constant(
        [[False] * 4, [False] * 4, [False, True, False, False], [False] * 4]
    )
    diagnostics = harness.diagnose_run(
        samples=_moving_samples(),
        initial_state=tf.zeros((4, 4), tf.float64),
        trace=trace,
        acceptance_band=(0.50, 0.90),
        min_movement=0.25,
        min_rms_jump=0.05,
    )
    assert diagnostics["native_divergence_status"] == "available"
    assert diagnostics["native_divergence_count"] == 1
    assert "positive_native_divergence" in diagnostics["hard_vetoes"]


def test_nonfinite_proposed_target_is_a_hard_veto(harness: ModuleType) -> None:
    trace = _trace((0.75, 0.75, 0.75, 0.75))
    trace["proposed_target_log_prob"] = tf.constant(
        [[-1.0] * 4, [-1.0] * 4, [-1.0, float("nan"), -1.0, -1.0], [-1.0] * 4],
        tf.float64,
    )
    diagnostics = harness.diagnose_run(
        samples=_moving_samples(),
        initial_state=tf.zeros((4, 4), tf.float64),
        trace=trace,
        acceptance_band=(0.50, 0.90),
        min_movement=0.25,
        min_rms_jump=0.05,
    )
    assert diagnostics["finite"]["proposed_target_log_prob"] is False
    assert "nonfinite_hmc_telemetry" in diagnostics["hard_vetoes"]


def test_scale_expansion_is_symmetric_across_g_and_h(harness: ModuleType) -> None:
    all_high = {
        "fresh-g": [_row(0.1, False, 0.95)],
        "fresh-h": [_row(0.1, False, 0.70)],
    }
    assert harness._scale_expansion(all_high) == harness.HIGH_SCALE_EXPANSION
    all_low = {
        "fresh-g": [_row(0.1, False, 0.70)],
        "fresh-h": [_row(0.1, False, 0.40)],
    }
    assert harness._scale_expansion(all_low) == harness.LOW_SCALE_EXPANSION
    opposed = {
        "fresh-g": [_row(0.1, False, 0.95)],
        "fresh-h": [_row(0.1, False, 0.40)],
    }
    assert harness._scale_expansion(opposed) == (
        *harness.HIGH_SCALE_EXPANSION,
        *harness.LOW_SCALE_EXPANSION,
    )
    mixed = {
        "fresh-g": [_row(0.1, False, 0.95), _row(0.2, False, 0.40)],
        "fresh-h": [_row(0.1, True, 0.70)],
    }
    assert harness._scale_expansion(mixed) == ()


def test_selection_rules_are_deterministic_not_metric_rankings(
    harness: ModuleType,
) -> None:
    scale_rows = [_row(0.1, True, 0.70), _row(0.4, True, 0.85)]
    assert harness._select_scale(scale_rows)["step_size"] == 0.1
    tied_rows = [_row(0.1, True, 0.60), _row(0.4, True, 0.80)]
    assert harness._select_scale(tied_rows)["step_size"] == 0.4
    trajectory_rows = []
    for leapfrog in (2, 4, 8, 16):
        row = _row(0.4, leapfrog in (4, 8, 16), 0.70)
        row["num_leapfrog_steps"] = leapfrog
        trajectory_rows.append(row)
    assert harness._select_trajectory(trajectory_rows)["num_leapfrog_steps"] == 8


def test_phase5_receipt_binding_is_exact(harness: ModuleType) -> None:
    binding = harness.validate_phase5_receipt()
    assert binding["decision"] == "PHASE5_EXACT_TRANSFORMED_TARGET_PASSED"
    assert binding["sha256"] == harness.PHASE5_RECEIPT_SHA256
    assert binding["source_bindings_revalidated"] is True


def test_scoped_binding_preserves_four_starts_and_global_capability(
    harness: ModuleType,
) -> None:
    adapter, initial_z, binding = harness.load_binding("fresh-g")
    assert tuple(initial_z.shape) == (4, 4)
    assert binding["four_distinct_starts"] is True
    assert binding["original_start_roundtrip_max_abs"] <= 1.0e-9
    assert binding["original_start_coordinate_system"] == "A0_affine_latent_z"
    assert binding["initial_z_radii"] == pytest.approx(
        [0.06814970004555097, 1.601309822842936, 1.2319851088185074, 1.0755935490216897]
    )
    assert binding["base_target_capability_unchanged"] == {
        "xla_hmc_ready": False,
        "full_chain_xla_diagnostic_ready": False,
    }
    assert adapter.value_score_capability().full_chain_xla_diagnostic_ready is True


def test_write_json_refuses_overwrite(harness: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    harness._write_json(output, {"status": "one"})
    with pytest.raises(harness.Phase6Error, match="refusing to overwrite"):
        harness._write_json(output, {"status": "two"})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "one"


def test_write_json_encodes_nonfinite_explanatory_values_explicitly(
    harness: ModuleType,
    tmp_path: Path,
) -> None:
    output = tmp_path / "nonfinite.json"
    harness._write_json(
        output,
        {
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
        },
    )
    payload = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"non-strict constant: {value}"),
    )
    assert payload == {
        "nan": "NaN",
        "positive": "Infinity",
        "negative": "-Infinity",
    }


def test_queue_pid_identity_and_gpu_parser(
    queue_harness: ModuleType,
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    command_path = proc_root / "123" / "cmdline"
    command_path.parent.mkdir(parents=True)
    command_path.write_bytes(b"python\0run_other_lane.py\0--flag\0")
    assert queue_harness.pid_command(123, proc_root=proc_root) == (
        "python run_other_lane.py --flag"
    )
    assert queue_harness.pid_command(456, proc_root=proc_root) is None
    rows = (("GPU-one", "10"), ("GPU-two", "20"))
    assert queue_harness.gpu_is_busy("GPU-two", rows) is True
    assert queue_harness.gpu_is_busy("GPU-three", rows) is False


def test_h_repair_contract_validates_exact_ladder_and_freezes_both_or_neither(
    harness: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ladder = Path(
        "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
        "phase-6-trial0-gh/ladder-r2.json"
    )
    output = tmp_path / "repair.json"
    fake_binding = {
        "transport_hash": "h-transport",
        "payload_sha256": "h-payload",
    }
    monkeypatch.setattr(
        harness,
        "load_binding",
        lambda label: (
            object(),
            tf.zeros((4, 4), tf.float64),
            fake_binding,
        ),
    )
    monkeypatch.setattr(harness, "_build_runner", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        harness,
        "_run_arm",
        lambda *args, **kwargs: {
            "step_size": 0.8,
            "num_leapfrog_steps": 4,
            "trajectory_length": 3.2,
            "diagnostics": {
                "viable": True,
                "hard_vetoes": [],
                "acceptance_vetoes": [],
            },
        },
    )
    monkeypatch.setattr(
        harness,
        "_run_manifest",
        lambda **kwargs: {"wall_time_seconds": 1.0},
    )
    payload = harness.run_h_confirmation_repair(
        output=output,
        wall_cap_seconds=600.0,
        ladder_receipt=ladder,
        ladder_sha256="6065d862f7dd6aeaea5db57a10f7d4a06be7292a93ffac4e4320e689f7533c51",
    )
    assert payload["decision"] == (
        "PHASE6_IDENTITY_MASS_KERNELS_FROZEN_AFTER_H_REPAIR"
    )
    assert set(payload["selected_kernels"]) == {"fresh-g", "fresh-h"}
    assert payload["repair_contract"]["seed"] == [20260716, 6901]
