#!/usr/bin/env python3
"""Retune and run canonical fixed-length HMC for a frozen paper d100 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)
CHAIN_COUNT = 4
STATE_SCHEMA = "bayesfilter.neutra.paper_d100_training_state.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--gaussian-constants", type=Path, required=True)
    parser.add_argument("--target", choices=("paper_funnel", "paper_ill_cond_gaussian"), required=True)
    parser.add_argument("--objective", choices=("reverse_kl", "forward_kl"), required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--cap-seconds", type=float, default=7200.0)
    parser.add_argument("--hmc-repair", action="store_true")
    parser.add_argument("--interval-level", type=float, default=0.99)
    return parser.parse_args()


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "as_list"):
        return _ready(value.as_list())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _load_replay_tensor(tf: Any, root: Path, receipt: Mapping[str, Any]) -> Any:
    path = root / str(receipt["path"])
    if _sha256(path) != str(receipt["sha256"]):
        raise RuntimeError(f"replay receipt hash mismatch: {path.name}")
    return tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)


def _load_frozen_transport(tf: Any, training_root: Path, expected_target: str, expected_objective: str) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )

    state_path = training_root / "trainer_state.json"
    manifest_path = training_root / "run_manifest.json"
    hashes_path = training_root / "artifact_hashes.json"
    state = _load_json(state_path)
    manifest = _load_json(manifest_path)
    hashes = _load_json(hashes_path)
    artifacts = hashes.get("artifacts", {})
    if artifacts.get(state_path.name) != _sha256(state_path):
        raise RuntimeError("training state artifact hash mismatch")
    if artifacts.get(manifest_path.name) != _sha256(manifest_path):
        raise RuntimeError("training manifest artifact hash mismatch")
    if state.get("schema") != STATE_SCHEMA:
        raise RuntimeError("training state schema mismatch")
    if state.get("objective") != expected_objective:
        raise RuntimeError("training objective mismatch")
    target = state.get("target")
    if not isinstance(target, Mapping) or target.get("name") != expected_target:
        raise RuntimeError("training target mismatch")
    if manifest.get("objective") != expected_objective:
        raise RuntimeError("training manifest objective mismatch")
    state_hash = str(state.get("state_hash", ""))
    semantic = {key: value for key, value in state.items() if key != "state_hash"}
    if len(state_hash) != 64 or _stable_hash(semantic) != state_hash:
        raise RuntimeError("training state semantic hash mismatch")
    config_payload = dict(state["config"])
    config_payload.pop("schema", None)
    config_payload["hidden_layers"] = tuple(config_payload["hidden_layers"])
    config_payload["initialization_seed"] = tuple(config_payload["initialization_seed"])
    config_payload["stage_s_max"] = tuple(config_payload.get("stage_s_max", ()))
    config = WeightedNeuTraConfig(**config_payload)
    transport = WeightedDenseIAFTransport(config)
    variables = state.get("variables")
    if not isinstance(variables, list) or len(variables) != len(transport.trainable_variables):
        raise RuntimeError("training state variable count mismatch")
    for variable, raw in zip(transport.trainable_variables, variables, strict=True):
        tensor = tf.convert_to_tensor(raw, tf.float64)
        if tensor.shape != variable.shape:
            raise RuntimeError("training state variable shape mismatch")
        tf.debugging.assert_all_finite(tensor, "frozen transport variable")
        variable.assign(tensor)
    tensor_hash = _stable_hash([variable.numpy().tolist() for variable in transport.trainable_variables])
    transport.bind_frozen_identity(
        {
            "checkpoint_sha256": _sha256(state_path),
            "training_state_hash": state_hash,
            "transport_tensor_hash": tensor_hash,
        }
    )
    return transport, config, {
        "state": state,
        "manifest": manifest,
        "state_path": state_path,
        "manifest_path": manifest_path,
        "state_sha256": _sha256(state_path),
        "state_hash": state_hash,
    }


def _batch_means_mcse(tf: Any, values: Any, batch_count: int = 10) -> Any:
    draws = int(values.shape[0])
    chains = int(values.shape[1])
    batch_size = draws // int(batch_count)
    if batch_size < 2:
        raise RuntimeError("too few retained draws for batch-means MCSE")
    trimmed = values[: batch_size * int(batch_count)]
    chain_major = tf.transpose(trimmed, (1, 0) + tuple(range(2, trimmed.shape.rank)))
    batches = tf.reshape(
        chain_major,
        (chains * int(batch_count), batch_size) + tuple(int(v) for v in values.shape[2:]),
    )
    means = tf.reduce_mean(batches, axis=1)
    return tf.sqrt(tf.math.reduce_variance(means, axis=0) / tf.cast(tf.shape(means)[0], tf.float64))


def _load_retained(tf: Any, archive_root: Path, expected_dimension: int) -> Any:
    paths = sorted((archive_root / "retained").glob("*-samples.tftensor"))
    if not paths:
        raise RuntimeError("retained sample archive is missing")
    tensors = [tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64) for path in paths]
    if any(tensor.shape.rank != 3 or tensor.shape[1] != CHAIN_COUNT or tensor.shape[2] != expected_dimension for tensor in tensors):
        raise RuntimeError("retained sample archive shape mismatch")
    return tf.concat(tensors, axis=0)


def _critical_value(interval_level: float) -> float:
    if float(interval_level) == 0.99:
        return 2.5758293035489004
    if float(interval_level) == 0.999:
        return 3.2905267314919255
    raise ValueError("only interval levels 0.99 and 0.999 are reviewed")


def _gaussian_diagnostics(
    tf: Any, spec: Any, physical: Any, interval_level: float = 0.99
) -> Mapping[str, Any]:
    centered = physical - tf.constant(spec.mean, tf.float64)[tf.newaxis, tf.newaxis, :]
    flat = tf.reshape(centered, (-1, spec.dimension))
    whitened = tf.transpose(
        tf.linalg.triangular_solve(
            tf.constant(spec.cholesky, tf.float64), tf.transpose(flat), lower=True
        )
    )
    whitened = tf.reshape(whitened, tf.shape(centered))
    mean = tf.reduce_mean(whitened, axis=(0, 1))
    square = tf.square(whitened)
    second = tf.reduce_mean(square, axis=(0, 1))
    radius = tf.reduce_sum(square, axis=-1)
    projection_mean = tf.reduce_mean(whitened[:, :, :4], axis=(0, 1))
    projection_second = tf.reduce_mean(tf.square(whitened[:, :, :4]), axis=(0, 1))
    mean_mcse = _batch_means_mcse(tf, whitened)
    square_mcse = _batch_means_mcse(tf, square)
    radius_mcse = _batch_means_mcse(tf, radius[:, :, tf.newaxis])[0]
    projection_mean_mcse = _batch_means_mcse(tf, whitened[:, :, :4])
    projection_second_mcse = _batch_means_mcse(tf, tf.square(whitened[:, :, :4]))
    structural_series = tf.concat(
        (
            tf.reduce_mean(whitened, axis=-1)[:, :, tf.newaxis],
            tf.reduce_mean(square, axis=-1)[:, :, tf.newaxis],
            radius[:, :, tf.newaxis],
            whitened[:, :, :4],
            tf.square(whitened[:, :, :4]),
        ),
        axis=2,
    )
    structural_mcse = _batch_means_mcse(tf, structural_series)
    structural_values = tf.concat(
        (
            tf.reshape(tf.reduce_mean(whitened), (1,)),
            tf.reshape(tf.reduce_mean(square), (1,)),
            tf.reshape(tf.reduce_mean(radius), (1,)),
            projection_mean,
            projection_second,
        ),
        axis=0,
    )
    structural_exact = tf.constant(
        (0.0, 1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        tf.float64,
    )
    z_critical = tf.constant(_critical_value(interval_level), tf.float64)
    structural_lower = structural_values - z_critical * structural_mcse
    structural_upper = structural_values + z_critical * structural_mcse
    structural_passed = tf.logical_and(
        structural_exact >= structural_lower, structural_exact <= structural_upper
    )
    return {
        "authority": "exact_normalized_gaussian_whitened_diagnostics",
        "coordinate": "whitened=(Sigma^-1/2)(theta-mu)",
        "draws_per_chain": int(physical.shape[0]),
        "chain_count": int(physical.shape[1]),
        "whitened_mean": mean,
        "whitened_second_moment": second,
        "mahalanobis_radius_mean": tf.reduce_mean(radius),
        "projection_mean_first4": projection_mean,
        "projection_second_moment_first4": projection_second,
        "whitened_mean_mcse": mean_mcse,
        "whitened_second_moment_mcse": square_mcse,
        "mahalanobis_radius_mcse": radius_mcse,
        "projection_mean_mcse_first4": projection_mean_mcse,
        "projection_second_moment_mcse_first4": projection_second_mcse,
        "structural_screen": {
            "names": (
                "grand_whitened_mean",
                "grand_whitened_second_moment",
                "mahalanobis_radius_mean",
                "projection_0_mean",
                "projection_1_mean",
                "projection_2_mean",
                "projection_3_mean",
                "projection_0_second_moment",
                "projection_1_second_moment",
                "projection_2_second_moment",
                "projection_3_second_moment",
            ),
            "values": structural_values,
            "exact_values": structural_exact,
            "mcse": structural_mcse,
            "interval_level": float(interval_level),
            "lower": structural_lower,
            "upper": structural_upper,
            "individual_interval_contains_exact": structural_passed,
            "all_individual_intervals_contain_exact": tf.reduce_all(structural_passed),
            "decision_role": "separate_diagnostics_no_omnibus_p_value",
        },
        "exact_values": {
            "whitened_mean": tf.zeros((100,), tf.float64),
            "whitened_second_moment": tf.ones((100,), tf.float64),
            "mahalanobis_radius_mean": tf.constant(100.0, tf.float64),
            "projection_mean_first4": tf.zeros((4,), tf.float64),
            "projection_second_moment_first4": tf.ones((4,), tf.float64),
        },
        "nonclaims": (
            "full coordinate table is descriptive and not an omnibus joint test",
            "analytic diagnostics do not rank reverse versus forward",
        ),
    }


def _funnel_diagnostics(
    tf: Any, spec: Any, physical: Any, interval_level: float = 0.99
) -> Mapping[str, Any]:
    y = physical[:, :, 0]
    residual = physical[:, :, 1:] * tf.exp(-y[:, :, tf.newaxis])
    residual_square = tf.square(residual)
    residual_mean_by_draw = tf.reduce_mean(residual, axis=-1)
    residual_square_by_draw = tf.reduce_mean(residual_square, axis=-1)
    y_square = tf.square(y)
    tail_low = tf.cast(y < -2.0, tf.float64)
    tail_high = tf.cast(y > 2.0, tf.float64)
    low_probability = tf.reduce_mean(tail_low)
    high_probability = tf.reduce_mean(tail_high)
    if bool(low_probability <= 0.0) or bool(high_probability <= 0.0):
        raise RuntimeError("retained funnel archive contains no observations in a required tail")
    low_residual_square = tf.reduce_sum(residual_square_by_draw * tail_low) / tf.reduce_sum(tail_low)
    high_residual_square = tf.reduce_sum(residual_square_by_draw * tail_high) / tf.reduce_sum(tail_high)
    y_mean = tf.reduce_mean(y)
    residual_square_mean = tf.reduce_mean(residual_square_by_draw)
    covariance = tf.reduce_mean(
        (y - y_mean) * (residual_square_by_draw - residual_square_mean)
    )
    mean_values = tf.stack(
        (
            y_mean,
            tf.reduce_mean(y_square),
            tf.reduce_mean(residual_mean_by_draw),
            residual_square_mean,
            covariance,
            low_probability,
            high_probability,
            low_residual_square,
            high_residual_square,
        )
    )
    covariance_influence = (
        (y - y_mean) * (residual_square_by_draw - residual_square_mean)
    )
    low_ratio_influence = (
        tail_low * (residual_square_by_draw - low_residual_square) / low_probability
    )
    high_ratio_influence = (
        tail_high * (residual_square_by_draw - high_residual_square) / high_probability
    )
    scalar_series = tf.stack(
        (
            y,
            y_square,
            residual_mean_by_draw,
            residual_square_by_draw,
            covariance_influence,
            tail_low,
            tail_high,
            low_ratio_influence,
            high_ratio_influence,
        ),
        axis=-1,
    )
    mcse = _batch_means_mcse(tf, scalar_series)
    exact_values = tf.constant(
        (
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.02275013194817921,
            0.02275013194817921,
            1.0,
            1.0,
        ),
        tf.float64,
    )
    z_critical = tf.constant(_critical_value(interval_level), tf.float64)
    structural_lower = mean_values - z_critical * mcse
    structural_upper = mean_values + z_critical * mcse
    structural_passed = tf.logical_and(
        exact_values >= structural_lower, exact_values <= structural_upper
    )

    quantile_probabilities = tf.constant((0.01, 0.10, 0.50, 0.90, 0.99), tf.float64)
    exact_quantiles = tf.constant(
        (
            -2.3263478740408408,
            -1.2815515655446004,
            0.0,
            1.2815515655446004,
            2.3263478740408408,
        ),
        tf.float64,
    )
    cdf_series = tf.cast(
        y[:, :, tf.newaxis] <= exact_quantiles[tf.newaxis, tf.newaxis, :],
        tf.float64,
    )
    cdf_values = tf.reduce_mean(cdf_series, axis=(0, 1))
    cdf_mcse = _batch_means_mcse(tf, cdf_series)
    cdf_lower = cdf_values - z_critical * cdf_mcse
    cdf_upper = cdf_values + z_critical * cdf_mcse
    cdf_passed = tf.logical_and(
        quantile_probabilities >= cdf_lower,
        quantile_probabilities <= cdf_upper,
    )
    sorted_y = tf.sort(tf.reshape(y, (-1,)))
    positions = quantile_probabilities * tf.cast(tf.size(sorted_y) - 1, tf.float64)
    lower_index = tf.cast(tf.floor(positions), tf.int32)
    upper_index = tf.cast(tf.math.ceil(positions), tf.int32)
    fractions = positions - tf.floor(positions)
    empirical_quantiles = (
        tf.gather(sorted_y, lower_index) * (1.0 - fractions)
        + tf.gather(sorted_y, upper_index) * fractions
    )
    return {
        "authority": "exact_paper_funnel_structural_diagnostics",
        "draws_per_chain": int(physical.shape[0]),
        "chain_count": int(physical.shape[1]),
        "y_mean": mean_values[0],
        "y_second_moment": mean_values[1],
        "standardized_residual_mean": mean_values[2],
        "standardized_residual_second_moment": mean_values[3],
        "cov_y_residual_square": mean_values[4],
        "prob_y_below_minus2": mean_values[5],
        "prob_y_above_plus2": mean_values[6],
        "tail_low_residual_second_moment": mean_values[7],
        "tail_high_residual_second_moment": mean_values[8],
        "scalar_mcse": mcse,
        "structural_screen": {
            "names": (
                "y_mean",
                "y_second_moment",
                "standardized_residual_mean",
                "standardized_residual_second_moment",
                "cov_y_residual_square",
                "prob_y_below_minus2",
                "prob_y_above_plus2",
                "tail_low_residual_second_moment",
                "tail_high_residual_second_moment",
            ),
            "values": mean_values,
            "exact_values": exact_values,
            "mcse": mcse,
            "interval_level": float(interval_level),
            "lower": structural_lower,
            "upper": structural_upper,
            "individual_interval_contains_exact": structural_passed,
            "all_individual_intervals_contain_exact": tf.reduce_all(structural_passed),
            "decision_role": "separate_diagnostics_no_omnibus_p_value",
        },
        "quantile_screen": {
            "probabilities": quantile_probabilities,
            "exact_quantiles": exact_quantiles,
            "empirical_quantiles": empirical_quantiles,
            "candidate_cdf_at_exact_quantiles": cdf_values,
            "candidate_cdf_batch_means_mcse": cdf_mcse,
            "interval_level": float(interval_level),
            "cdf_interval_lower": cdf_lower,
            "cdf_interval_upper": cdf_upper,
            "individual_interval_contains_exact_probability": cdf_passed,
            "all_individual_intervals_contain_exact_probability": tf.reduce_all(cdf_passed),
            "decision_role": (
                "chain-aware CDF-at-exact-quantile equivalence; separate diagnostics, "
                "no omnibus p-value"
            ),
        },
        "exact_values": {
            "y_mean": tf.constant(0.0, tf.float64),
            "y_second_moment": tf.constant(1.0, tf.float64),
            "standardized_residual_mean": tf.constant(0.0, tf.float64),
            "standardized_residual_second_moment": tf.constant(1.0, tf.float64),
            "cov_y_residual_square": tf.constant(0.0, tf.float64),
            "prob_y_below_minus2": tf.constant(0.02275013194817921, tf.float64),
            "prob_y_above_plus2": tf.constant(0.02275013194817921, tf.float64),
            "tail_low_residual_second_moment": tf.constant(1.0, tf.float64),
            "tail_high_residual_second_moment": tf.constant(1.0, tf.float64),
        },
        "nonclaims": (
            "empirical quantile differences are descriptive; the chain-aware CDF intervals decide",
            "structural diagnostics are separate screens, not a joint p-value",
        ),
    }


def _run_tuning(base: Any, transport: Any, initial: Any, output: Path, target_name: str, objective: str, repair: bool) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
        tune_fixed_transport_hmc_kernel,
    )

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.02 if repair else 0.05,
        leapfrog_grid=(3, 5, 10, 15, 20, 25, 32),
        chain_count=CHAIN_COUNT,
        initial_state_bank=tuple(tuple(float(value) for value in row) for row in initial.numpy().tolist()),
        target_accept_prob=0.85 if repair else 0.70,
        acceptance_band=(0.70, 0.95) if repair else (0.55, 0.90),
        repair_band=(0.55, 0.98) if repair else (0.40, 0.95),
        fixed_grid_fallback_acceptance_max=0.95,
        budget_schedule=(64, 128, 256) if repair else (32, 64, 128),
        tune_num_results=16,
        screen_num_results=64,
        screen_num_burnin_steps=16,
        verification_num_results=2000,
        verification_num_burnin_steps=64,
        require_modern_rank_normalized_verification=True,
        verification_coordinate_system="hmc_coordinates",
        verification_min_retained_results_per_chain=2000,
        tune_seed_base=(20260813, 64001 if repair else 54001),
        screen_seed_base=(20260813, 65001 if repair else 55001),
        verification_seed_base=(20260813, 66001 if repair else 56001),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=f"weighted_neutra_paper_d100:{target_name}:{objective}:tuning_v1",
        output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=initial[0],
        config=config,
        output_dir=output / "tuning",
    ).payload()


def _run_sequential(adapter: Any, initial: Any, tuning: Mapping[str, Any], output: Path, cap: float, target_name: str, objective: str) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc

    kernel = tuning.get("final_kernel_payload")
    if tuning.get("passed") is not True or not isinstance(kernel, Mapping):
        raise RuntimeError("HMC tuning produced no viable fixed kernel")
    leapfrog = int(kernel.get("num_leapfrog_steps", 0))
    if leapfrog < 2:
        raise RuntimeError("L=1 is forbidden")
    step_size = float(kernel.get("step_size", 0.0))
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise RuntimeError("HMC step size is invalid")
    started = time.perf_counter()
    config = SequentialNeuTraHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        seed=(20260813, 57001),
        warmup_chunk_size=500,
        warmup_min_results=2000,
        warmup_window_results=1000,
        warmup_max_results=10000,
        retained_chunk_size=500,
        retained_min_results=1000,
        retained_max_results=10000,
        retained_check_interval_results=1000,
        warmup_rhat_max=1.05,
        retained_rhat_max=1.01,
        bulk_ess_min=400.0,
        tail_ess_min=400.0,
        delta_h_abs_max=1000.0,
        acceptance_min=0.35,
        acceptance_max=0.95,
        chain_count=CHAIN_COUNT,
        use_xla=True,
        target_status_required=True,
        retained_ess_required=True,
        xla_qualification_required=False,
    )
    result = run_sequential_neutra_hmc(
        adapter,
        initial,
        config,
        archive_root=output / "archive",
        archive_label=f"paper-d100-{target_name}-{objective}",
        budget_check=lambda _transitions: time.perf_counter() - started < cap,
    )
    return {"schema": "bayesfilter.neutra.sequential_hmc_result.v1", **result.__dict__}


def main() -> int:
    args = _parse_args()
    output = args.output_root.resolve()
    training_root = args.training_root.resolve()
    replay_root = args.replay_root.resolve()
    constants_path = args.gaussian_constants.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if float(args.interval_level) not in (0.99, 0.999):
        raise ValueError("only interval levels 0.99 and 0.999 are reviewed")
    if not float(args.cap_seconds) > 0.0:
        raise ValueError("cap-seconds must be positive")
    required = (
        PLAN,
        constants_path,
        training_root / "trainer_state.json",
        training_root / "run_manifest.json",
        training_root / "artifact_hashes.json",
        replay_root / "replay_manifest.json",
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("paper d100 HMC inputs are missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_paper_d100_target import (
        PaperD100ValueScoreAdapter,
        load_paper_gaussian_spec,
        make_paper_funnel_spec,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical_gpus}")
    spec = make_paper_funnel_spec() if args.target == "paper_funnel" else load_paper_gaussian_spec(constants_path)
    transport, config, frozen = _load_frozen_transport(tf, training_root, spec.name, args.objective)
    base = PaperD100ValueScoreAdapter(spec)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=f"weighted_neutra_paper_d100:{spec.name}:{args.objective}:hmc_v1",
        runtime_backend="tensorflow_exact_paper_d100_frozen_iaf_hmc",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "exact paper d100 target and frozen transport only",
            "no objective ranking or default promotion",
        ),
    )
    initial = tf.random.stateless_normal(
        (CHAIN_COUNT, spec.dimension), seed=(20260813, 58001), dtype=tf.float64
    ) * tf.constant(0.05, tf.float64)
    manifest = {
        "schema": "bayesfilter.neutra.paper_d100_hmc_manifest.v1",
        "plan": PLAN.as_posix(),
        "training_root": training_root.as_posix(),
        "training_state_sha256": frozen["state_sha256"],
        "training_state_hash": frozen["state_hash"],
        "target": spec.manifest_payload(),
        "objective": args.objective,
        "adapter_signature": adapter.adapter_signature(),
        "transport_manifest": transport.manifest_payload(),
        "memory_policy": _ready(memory_policy),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": False,
        "initial_state": initial,
        "hmc_grid": {"leapfrog": [3, 5, 10, 15, 20, 25, 32], "l1_forbidden": True},
        "hmc_repair": bool(args.hmc_repair),
        "interval_level": float(args.interval_level),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    _write(output / "run_manifest.json", manifest)
    tuning = _run_tuning(base, transport, initial, output, spec.name, args.objective, args.hmc_repair)
    if tuning.get("passed") is not True:
        result = {
            "schema": "bayesfilter.neutra.paper_d100_hmc_result.v1",
            "manifest": manifest,
            "tuning": tuning,
            "decision": {"status": "hmc_candidate_rejected_at_tuning", "promotion": False, "nonclaims": ["no posterior correctness claim"]},
            "wall_seconds": time.perf_counter() - started,
        }
        _write(output / "result.json", result)
        _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.paper_d100_hmc_hashes.v1", "artifacts": {path.relative_to(output).as_posix(): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}})
        print(json.dumps({"passed": False, "output_root": output.as_posix()}, sort_keys=True))
        return 0
    sequential = _run_sequential(adapter, initial, tuning, output, min(float(args.cap_seconds), 5400.0), spec.name, args.objective)
    retained = _load_retained(tf, output / "archive", spec.dimension)
    flat = tf.reshape(retained, (-1, spec.dimension))
    physical_flat = transport.forward_batch(flat)
    physical = tf.reshape(physical_flat, tf.shape(retained))
    analytic = (
        _gaussian_diagnostics(tf, spec, physical, args.interval_level)
        if spec.name == "paper_ill_cond_gaussian"
        else _funnel_diagnostics(tf, spec, physical, args.interval_level)
    )
    passed = bool(sequential.get("passed"))
    structural_passed = bool(
        analytic.get("structural_screen", {}).get(
            "all_individual_intervals_contain_exact", False
        )
    )
    quantile_passed = (
        bool(
            analytic.get("quantile_screen", {}).get(
                "all_individual_intervals_contain_exact_probability", False
            )
        )
        if spec.name == "paper_funnel"
        else True
    )
    analytic_passed = bool(structural_passed and quantile_passed)
    candidate_passed = bool(passed and analytic_passed)
    result = {
        "schema": "bayesfilter.neutra.paper_d100_hmc_result.v1",
        "manifest": manifest,
        "tuning": tuning,
        "sequential": sequential,
        "analytic_diagnostics": analytic,
        "decision": {
            "status": (
                "candidate_sampler_and_analytic_passed"
                if candidate_passed
                else "candidate_sampler_passed_analytic_rejected"
                if passed
                else "candidate_sampler_rejected"
            ),
            "candidate_passed": candidate_passed,
            "sampler_passed": passed,
            "analytic_individual_intervals_passed": analytic_passed,
            "analytic_structural_intervals_passed": structural_passed,
            "analytic_quantile_intervals_passed": quantile_passed,
            "promotion": False,
            "primary_criterion": "canonical sequential R-hat/ESS plus target-specific exact analytic diagnostics",
            "objective_ranking": "not_supported",
            "default_promotion": False,
            "nonclaims": ["no objective superiority", "no original-paper replication", "no universal funnel guarantee"],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.paper_d100_hmc_hashes.v1", "artifacts": {path.relative_to(output).as_posix(): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}})
    print(json.dumps({"passed": candidate_passed, "sampler_passed": passed, "output_root": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
