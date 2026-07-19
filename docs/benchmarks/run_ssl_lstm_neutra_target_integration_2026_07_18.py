#!/usr/bin/env python3
"""Execute the bounded SSL-LSTM target-adapter/preflight lane.

This runner never reads HMC/NeuTra/retained artifacts and never performs a G/H
comparison. It writes one immutable JSON receipt under the target-integration
artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.predictive_equivalence import (
    chain_bartlett_long_run_covariance,
    proper_score_loss,
    split_quadratic_loss_confidence_bounds,
)
from bayesfilter.inference.ssl_lstm_target_integration import (
    HORIZON,
    calibrate_horizon_scales,
    compare_path_and_conditional_moments,
    conditional_observation_moments,
)
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
)


ARTIFACT_DIR = ROOT / "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/target-integration"
DEFAULT_OUTPUT = ARTIFACT_DIR / "target-integration-preflight.json"
PLAN_PATH = Path("docs/plans/bayesfilter-ssl-lstm-neutra-target-integration-plan-2026-07-18.md")
RUNNER_SOURCE = Path("docs/benchmarks/run_ssl_lstm_neutra_target_integration_2026_07_18.py")
ADAPTER_SOURCE = Path("bayesfilter/inference/ssl_lstm_target_integration.py")
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
FORECAST_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py")
PARAMETER_ADAPTER_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py")
POINTS_HEX = (
    ("0x1.6666666666666p-2", "-0x1.47ae147ae147bp-4", "0x1.4cccccccccccdp-1", "0x1.999999999999ap-5"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.ee87ac2b0ee48p-2", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.50dd6faf210bep-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.b19cbccaf903cp-3", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.2cd959924a756p-5", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.25964cacd3b9fp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.7f2fe6466d539p-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.8891e0688b5c0p-5"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.c88ade80893d6p-3"),
)
CALIBRATION_SEEDS = ((20260718, 4101), (20260718, 4102), (20260718, 4103), (20260718, 4104))
EVALUATION_SEEDS = ((20260718, 4201), (20260718, 4202), (20260718, 4203), (20260718, 4204))
ROBUSTNESS_SEEDS = ((20260718, 4301), (20260718, 4302), (20260718, 4303), (20260718, 4304))
CHAIN_COUNT = 4
DRAW_COUNT = 10
REPLICATION_COUNT = 2
HAC_MULTIPLIER = 3.0
RIDGE_LADDER = (0.0,)
CONDITION_NUMBER_MAX = 1.0e8
K_THRESHOLD = 0.0068491


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_values(tensor: tf.Tensor) -> list[float]:
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy()]


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _paths_for_seeds(config: SSLLSTMForecastConfig, seeds: tuple[tuple[int, int], ...]):
    points = tf.constant([[float.fromhex(value) for value in row] for row in POINTS_HEX], tf.float64)
    paths = []
    bank_signatures = []
    for chain, seed in enumerate(seeds):
        bank = make_ssl_lstm_innovation_bank(
            config,
            DRAW_COUNT,
            tf.constant(seed, tf.int32),
            "independent_arm",
            chain + 1,
        )
        paths.append(
            forecast_ssl_lstm_paths(
                points,
                bank,
                config,
                draw_chunk_size=DRAW_COUNT,
                runtime_execution_role="cpu_hidden_xla_reference",
                trust_basis="cpu_hidden_reference_exception_not_gpu_evidence",
            )
        )
        bank_signatures.append(bank.content_signature)
    return tuple(paths), tuple(bank_signatures)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config = SSLLSTMForecastConfig()
    calibration_paths, calibration_banks = _paths_for_seeds(config, CALIBRATION_SEEDS)
    evaluation_paths, evaluation_banks = _paths_for_seeds(config, EVALUATION_SEEDS)
    robustness_paths, robustness_banks = _paths_for_seeds(config, ROBUSTNESS_SEEDS)
    calibration = calibrate_horizon_scales(calibration_paths, seed_roots=CALIBRATION_SEEDS)
    comparison = compare_path_and_conditional_moments(
        evaluation_paths,
        calibration,
        config,
        jit_compile=False,
        independent_paths_by_chain=robustness_paths,
    )
    values = comparison.conditional.influence_values
    covariance = chain_bartlett_long_run_covariance(
        values,
        bandwidth_multiplier=HAC_MULTIPLIER,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
        jit_compile=False,
    )
    average_loss = proper_score_loss(tf.fill([HORIZON], tf.constant(0.1, tf.float64)) )
    bounds = split_quadratic_loss_confidence_bounds(
        comparison.conditional.feature_estimate,
        covariance.pooled_mean_covariance,
        average_loss,
        jit_compile=False,
    )
    hard_checks = {
        "conditional_variance_finite_positive": bool(tf.reduce_all(comparison.standardized_conditional_variances > 0.0)),
        "calibration_scales_finite_positive": bool(tf.reduce_all(calibration.scale > 0.0)),
        "calibration_evaluation_seed_disjoint": not (set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS)),
        "all_seed_domains_disjoint": len(set(CALIBRATION_SEEDS + EVALUATION_SEEDS + ROBUSTNESS_SEEDS)) == 12,
        "paired_six_mcse": comparison.paired_pass,
        "covariance_numerically_admissible": covariance.numerically_admissible,
        "covariance_zero_ridge_admissible": covariance.inference_admissible,
        "split_bounds_admissible": bounds.inference_admissible,
        "alpha_allocation": float(bounds.allocated_familywise_alpha) == 0.05,
    }
    receipt = {
        "schema": "bayesfilter.ssl_lstm.target_integration.preflight.v1",
        "status": "TARGET_ADAPTER_PREFLIGHT_PASSED_GH_CONFIRMATION_CLOSED" if all(hard_checks.values()) else "TARGET_PREFLIGHT_FAILED_GH_CONFIRMATION_CLOSED",
        "plan": str(PLAN_PATH),
        "plan_sha256": _hash_file(ROOT / PLAN_PATH),
        "source_hashes": {
            str(RUNNER_SOURCE): _hash_file(ROOT / RUNNER_SOURCE),
            str(ADAPTER_SOURCE): _hash_file(ROOT / ADAPTER_SOURCE),
            str(PREDICTIVE_SOURCE): _hash_file(ROOT / PREDICTIVE_SOURCE),
            str(FORECAST_SOURCE): _hash_file(ROOT / FORECAST_SOURCE),
            str(PARAMETER_ADAPTER_SOURCE): _hash_file(ROOT / PARAMETER_ADAPTER_SOURCE),
        },
        "execution": {
            "command": "CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_ssl_lstm_neutra_target_integration_2026_07_18.py",
            "environment": "tfgpu",
            "git_commit": _git_commit(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "forecast_jit_compile": True,
            "forecast_execution_role": "cpu_hidden_xla_reference",
            "statistical_adapter_jit_compile": False,
            "statistical_adapter_execution_role": "eager_debug_reference",
            "trust_basis": "cpu_hidden_reference_exception_not_gpu_evidence",
            "cpu_sample_worker_count": 1,
            "data_version": "A2_frozen_ten_row_engineering_fixture",
            "wall_seconds": time.time() - started,
        },
        "fixture": {
            "point_names": ["truth_free", "phase2s_center", "shell_0_minus", "shell_0_plus", "shell_1_minus", "shell_1_plus", "shell_2_minus", "shell_2_plus", "shell_3_minus", "shell_3_plus"],
            "draw_count": DRAW_COUNT,
            "replication_count": REPLICATION_COUNT,
            "chain_count": CHAIN_COUNT,
            "horizon": HORIZON,
            "forecast_config_signature": config.signature(),
        },
        "calibration": {
            "seed_roots": [list(seed) for seed in CALIBRATION_SEEDS],
            "bank_signatures": list(calibration_banks),
            "center": _tensor_values(calibration.center),
            "scale": _tensor_values(calibration.scale),
            "center_sha256": _hash_file(ROOT / PLAN_PATH) if False else hashlib.sha256(tf.io.serialize_tensor(calibration.center).numpy()).hexdigest(),
            "scale_sha256": hashlib.sha256(tf.io.serialize_tensor(calibration.scale).numpy()).hexdigest(),
            "calibration_signature": calibration.calibration_signature,
        },
        "evaluation": {
            "seed_roots": [list(seed) for seed in EVALUATION_SEEDS],
            "bank_signatures": list(evaluation_banks),
            "robustness_seed_roots": [list(seed) for seed in ROBUSTNESS_SEEDS],
            "robustness_bank_signatures": list(robustness_banks),
            "adapter_source_signature": conditional_observation_moments(evaluation_paths[0], config).source_signature,
            "paired_feature_difference": _tensor_values(comparison.paired_feature_difference),
            "paired_standard_error": _tensor_values(comparison.paired_standard_error),
            "paired_mcse_multiplier": comparison.paired_mcse_multiplier,
            "paired_pass": comparison.paired_pass,
            "independent_feature_difference": _tensor_values(comparison.independent_feature_difference),
            "conditional_feature_estimate": _tensor_values(comparison.conditional.feature_estimate),
            "path_feature_estimate": _tensor_values(comparison.path.feature_estimate),
        },
        "covariance": {
            "bandwidth": covariance.bandwidth,
            "bandwidth_multiplier": HAC_MULTIPLIER,
            "ridge_ladder": list(RIDGE_LADDER),
            "condition_number": float(covariance.condition_number),
            "minimum_eigenvalue": float(tf.reduce_min(covariance.eigenvalues)),
            "numerically_admissible": covariance.numerically_admissible,
            "inference_admissible": covariance.inference_admissible,
            "interpretation": "engineering_shape_conditioning_only_nonstationary_A2_fixture",
        },
        "split_bounds": {
            "threshold": K_THRESHOLD,
            "average_lower": float(bounds.average_lower_bound),
            "average_upper": float(bounds.average_upper_bound),
            "horizon_lower": _tensor_values(bounds.horizon_lower_bounds),
            "horizon_upper": _tensor_values(bounds.horizon_upper_bounds),
            "allocated_familywise_alpha": float(bounds.allocated_familywise_alpha),
            "average_lower_kkt_residual": float(bounds.average_lower_kkt_residual),
            "average_upper_kkt_residual": float(bounds.average_upper_kkt_residual),
            "maximum_horizon_lower_kkt_residual": float(tf.reduce_max(bounds.horizon_lower_kkt_residuals)),
            "maximum_horizon_upper_kkt_residual": float(tf.reduce_max(bounds.horizon_upper_kkt_residuals)),
            "inference_admissible": bounds.inference_admissible,
        },
        "hard_checks": hard_checks,
        "nonclaims": [
            "no HMC or NeuTra execution",
            "no retained posterior artifact read",
            "no G/H confirmation or equivalence/material-difference claim",
            "no posterior correctness, sampler ranking, or model adequacy claim",
            "HAC on fixed A2 rows is not target long-run covariance evidence",
        ],
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
