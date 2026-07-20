from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_complexity_retained_hmc_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_neutra_complexity_retained_hmc_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def moving_draws(draws: int = 8) -> tf.Tensor:
    draw = tf.reshape(tf.cast(tf.range(draws), tf.float64), (draws, 1, 1))
    chain = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    coordinate = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 1, 4))
    return 0.1 * draw + chain + 0.01 * coordinate


def manifest(
    *,
    draws: int,
    acceptance: tuple[float, float, float, float],
    divergence_count: int | None = None,
):
    return {
        "retained_sample_count": draws,
        "diagnostics_private_metadata": {
            "native_divergence_status": (
                "available" if divergence_count is not None else "not_exposed_by_kernel"
            ),
            "divergence_count": divergence_count,
            "sampler_health_diagnostics": {
                "acceptance_rate_by_chain": list(acceptance),
                "log_accept_ratio": {"nonfinite_count": 0},
                "target_log_prob": {"nonfinite_count": 0},
            },
        },
    }


def test_contract_is_q_general_and_freezes_phase5_thresholds() -> None:
    args = runner.parse_args(["--mode", "contract-smoke", "--q", "20"])
    runner.validate_args(args)
    payload = runner.contract_payload(args)
    assert payload["status"] == "PASSED"
    assert payload["segment_results"] == 256
    assert payload["initial_burnin"] == 256
    assert payload["checkpoint_draws_per_chain"] == [512, 1024, 2048, 4096]
    assert payload["thresholds"] == {
        "rhat_max": 1.01,
        "bulk_ess_min": 400.0,
        "tail_ess_min": 400.0,
        "mcse_sd_ratio_max": 0.05,
        "cross_replication_combined_mcse_multiplier": 3.0,
    }
    assert payload["material_execution_authorized"] is False


def test_material_mode_fails_closed_without_authority_and_phase4_receipt() -> None:
    args = runner.parse_args(["--mode", "acquire", "--q", "1"])
    with pytest.raises(runner.RetainedHMCError, match="authorize-material-run"):
        runner.validate_args(args)
    args.authorize_material_run = True
    args.cap_seconds = 60.0
    args.output_root = Path("docs/plans/artifacts/test-retained")
    with pytest.raises(runner.RetainedHMCError, match="phase4-summary"):
        runner.validate_args(args)


def test_phase4_receipt_requires_two_frozen_kernels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "sha256", lambda path: "hash")
    path = tmp_path / "phase4.json"
    path.write_text(
        json.dumps(
            {
                "schema": runner.PHASE4_SCHEMA,
                "status": "TUNING_REPAIR_REQUIRED",
                "q": 20,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.RetainedHMCError, match="freeze both kernels"):
        runner.load_phase4_contract(20, Path("phase4.json"))


def test_cumulative_admission_uses_both_z_and_theta_without_acceptance_veto(
    monkeypatch,
) -> None:
    observed = []

    def pass_screen(values):
        observed.append(tf.convert_to_tensor(values))
        return {"shape": list(values.shape)}, []

    monkeypatch.setattr(runner, "coordinate_screen", pass_screen)
    z = moving_draws(8)
    result = runner.cumulative_admission(
        z_draw_major=z,
        theta_draw_major=2.0 * z,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            manifest(draws=4, acceptance=(0.0, 0.25, 0.5, 1.0)),
            manifest(draws=4, acceptance=(1.0, 0.75, 0.5, 0.0)),
        ),
    )
    assert [tuple(item.shape) for item in observed] == [(4, 8, 4), (4, 8, 4)]
    assert result["admitted"] is True
    assert result["acceptance_rate_by_chain"] == pytest.approx([0.5] * 4)


def test_cumulative_admission_hard_vetoes_divergence_and_unmoved_chain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "coordinate_screen", lambda values: ({}, []))
    z = moving_draws(4)
    z = tf.concat((z[:, :3], tf.zeros((4, 1, 4), tf.float64)), axis=1)
    result = runner.cumulative_admission(
        z_draw_major=z,
        theta_draw_major=z,
        initial_state=tf.zeros((4, 4), tf.float64),
        segment_manifests=(
            manifest(draws=4, acceptance=(0.5, 0.5, 0.5, 0.5), divergence_count=1),
        ),
    )
    assert result["admitted"] is False
    assert "positive_native_divergence" in result["hard_vetoes"]
    assert "unmoved_chain" in result["hard_vetoes"]


def test_cross_replication_screen_uses_four_means_and_ten_second_moments() -> None:
    generator = tf.random.Generator.from_seed(20260719)
    a = generator.normal((4, 512, 4), dtype=tf.float64)
    same = runner.cross_replication_stability(a, a)
    assert same["passed"] is True
    assert len(same["functional_names"]) == 14
    shifted = runner.cross_replication_stability(a, a + 1.0)
    assert shifted["passed"] is False


def test_budget_reserve_scales_with_q_l_equivalent_observation() -> None:
    budget = runner.Budget(100_000.0)
    assert budget.segment_reserve(
        transitions=512, leapfrog_steps=8, cold_runner=True
    ) == runner.FIRST_COMPILED_SEGMENT_RESERVE_SECONDS
    budget.observe(2.0)
    assert budget.segment_reserve(
        transitions=256, leapfrog_steps=8, cold_runner=False
    ) == pytest.approx(2.0 * 256 * 8 * runner.SEGMENT_COST_MARGIN)


def test_phase4_cost_observations_seed_first_retained_reserve() -> None:
    arm = {"timing": {"seconds_per_transition_leapfrog": 2.5}}
    phase4 = {
        "tuning": {
            "chart-a": {
                "scale_rows": [arm],
                "trajectory_rows": [],
                "confirmation": None,
                "adjacent_repair": None,
            },
            "chart-b": {
                "scale_rows": [],
                "trajectory_rows": [arm],
                "confirmation": None,
                "adjacent_repair": None,
            },
        }
    }
    assert runner.phase4_cost_observations(phase4) == (2.5, 2.5)
    with pytest.raises(runner.RetainedHMCError, match="no usable HMC cost"):
        runner.phase4_cost_observations(
            {
                "tuning": {
                    label: {
                        "scale_rows": [],
                        "trajectory_rows": [],
                        "confirmation": None,
                        "adjacent_repair": None,
                    }
                    for label in runner.CHARTS
                }
            }
        )


def test_runner_uses_immutable_archive_and_no_material_contract_smoke_side_effects() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "build_retained_sample_hmc_archive_runner" in source
    assert "overwrite=False" in source
    assert "execution_source_drift" not in source
    assert "samples_retained_as_posterior_evidence" in source
    assert "native divergence unavailability is not zero divergences" not in source
