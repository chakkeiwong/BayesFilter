"""Static contract tests for the paper d100 HMC runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_paper_d100_target import (
    make_paper_funnel_spec,
    sample_paper_d100_exact,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_paper_d100_hmc_2026_08_13.py"
ADJUDICATOR = ROOT / (
    "docs/benchmarks/"
    "adjudicate_neutra_paper_d100_gaussian_archive_2026_08_13.py"
)
GENERIC_ADJUDICATOR = ROOT / "docs/benchmarks/adjudicate_neutra_paper_d100_archive_interval_2026_08_14.py"
CALIBRATOR = ROOT / "docs/benchmarks/calibrate_neutra_paper_d100_gaussian_intervals_2026_08_14.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("paper_d100_hmc_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_shared_sequential_policy_and_forbids_l1() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "run_sequential_neutra_hmc" in source
    assert "SequentialNeuTraHMCConfig" in source
    assert "leapfrog_grid=(3, 5, 10, 15, 20, 25, 32)" in source
    assert "if leapfrog < 2:" in source
    assert "warmup_min_results=2000" in source
    assert "retained_rhat_max=1.01" in source
    assert "bulk_ess_min=400.0" in source
    assert "tail_ess_min=400.0" in source
    assert "xla_qualification_required=False" in source
    assert "NoUTurnSampler" not in source


def test_runner_binds_frozen_state_and_exact_analytic_diagnostics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "training state semantic hash mismatch" in source
    assert "training objective mismatch" in source
    assert "training target mismatch" in source
    assert "exact_normalized_gaussian_whitened_diagnostics" in source
    assert "exact_paper_funnel_structural_diagnostics" in source
    assert "all_individual_intervals_contain_exact" in source
    assert "all_individual_intervals_contain_exact_probability" in source
    assert "chain-aware CDF-at-exact-quantile equivalence" in source
    assert 'parser.add_argument("--hmc-repair", action="store_true")' in source
    assert 'parser.add_argument("--interval-level", type=float, default=0.99)' in source
    assert "only interval levels 0.99 and 0.999 are reviewed" in source
    assert "full coordinate table is descriptive" in source
    assert "objective_ranking" in source
    assert "artifact_hashes" in source
    assert "trust_basis" in source


def test_runner_does_not_use_numpy_or_samplewise_target_fallback() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.while_loop" not in source


def test_funnel_diagnostics_compute_covariance_ratio_mcse_and_quantile_screen() -> None:
    runner = _load_runner()
    target = make_paper_funnel_spec()
    rows = sample_paper_d100_exact(target, 4000, seed=(20260813, 91001))
    chains = tf.reshape(rows, (1000, 4, 100))
    diagnostic = runner._funnel_diagnostics(tf, target, chains)
    structural = diagnostic["structural_screen"]
    quantile = diagnostic["quantile_screen"]
    assert len(structural["names"]) == 9
    assert tuple(structural["mcse"].shape) == (9,)
    assert bool(tf.reduce_all(tf.math.is_finite(structural["mcse"])))
    assert tuple(quantile["probabilities"].shape) == (5,)
    assert tuple(quantile["empirical_quantiles"].shape) == (5,)
    assert bool(tf.reduce_all(tf.math.is_finite(quantile["candidate_cdf_batch_means_mcse"])))
    assert abs(float(diagnostic["cov_y_residual_square"])) < 0.03
    assert abs(float(diagnostic["tail_low_residual_second_moment"]) - 1.0) < 0.03
    assert abs(float(diagnostic["tail_high_residual_second_moment"]) - 1.0) < 0.03


def test_funnel_diagnostics_reject_shifted_y_law() -> None:
    runner = _load_runner()
    target = make_paper_funnel_spec()
    rows = sample_paper_d100_exact(target, 4000, seed=(20260813, 91002))
    shifted_y = rows[:, :1] + tf.constant(0.5, tf.float64)
    shifted_x = rows[:, 1:] * tf.exp(tf.constant(0.5, tf.float64))
    shifted = tf.reshape(tf.concat((shifted_y, shifted_x), axis=1), (1000, 4, 100))
    diagnostic = runner._funnel_diagnostics(tf, target, shifted)
    structural = diagnostic["structural_screen"]
    quantile = diagnostic["quantile_screen"]
    assert not bool(structural["individual_interval_contains_exact"][0])
    assert not bool(quantile["all_individual_intervals_contain_exact_probability"])


def test_gaussian_adjudicator_is_hash_bound_and_does_not_rerun_hmc() -> None:
    source = ADJUDICATOR.read_text(encoding="utf-8")
    assert "_verify_hmc_hashes" in source
    assert "HMC artifact hash mismatch" in source
    assert "source HMC result did not pass its sampler gates" in source
    assert 'sequential.get("passed") is True' in source
    assert "historical_sampler_statuses" in source
    assert "_load_retained" in source
    assert "_gaussian_diagnostics" in source
    assert "source_sampler_passed_corrected_analytic_rejected" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "run_sequential_neutra_hmc" not in source
    assert "tune_fixed_transport_hmc_kernel" not in source
    assert "import numpy" not in source


def test_generic_interval_adjudicator_is_uniform_and_hash_bound() -> None:
    source = GENERIC_ADJUDICATOR.read_text(encoding="utf-8")
    assert 'choices=(0.99, 0.999)' in source
    assert '"uniform_interval_policy": True' in source
    assert "source HMC target mismatch" in source
    assert "_funnel_diagnostics" in source
    assert "_gaussian_diagnostics" in source
    assert "import numpy" not in source


def test_gaussian_calibrator_is_iid_diagnostic_only() -> None:
    source = CALIBRATOR.read_text(encoding="utf-8")
    assert "known_correct_iid_target" in source
    assert "pass_rate_99" in source
    assert "pass_rate_999" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "import numpy" not in source
