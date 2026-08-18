#!/usr/bin/env python3
"""Focused terminal continuation for Austria-SIR GenUT derivative localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

REQUIRE_GPU = os.environ.get("BAYESFILTER_GENUT_AD_REQUIRE_GPU", "1") != "0"
if not REQUIRE_GPU and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU reference requires CUDA_VISIBLE_DEVICES=-1")
GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf, require_gpu=REQUIRE_GPU
)

from bayesfilter.highdim import cubature_genut_filter as genut_filter
from bayesfilter.highdim import ledh_contract_e_reset_tf as reset
from bayesfilter.highdim.cubature_genut_adapters import (
    parameterized_austria_sir_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.independent_score import sir_observation_simulator_tf as sir
from docs.benchmarks.run_genut_sir_ad_root_cause_20260817 import (
    EXPECTED_OBSERVATION_HASH,
    PARAMETER_COUNT,
    PLAN,
    SEED,
    STATE_DIMENSION,
    _controls,
    _metrics,
    _noise,
    _safe,
    _sha,
    _tensor_sha,
)


LOCAL_CHECKPOINT = ROOT / (
    "docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/"
    "attempt05/checkpoint.json"
)
PRIOR_FULL_CHECKPOINT = ROOT / (
    "docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/"
    "attempt05/full_program_checkpoint.json"
)


def _pure_sinkhorn_value(
    particles: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    particle_count = particles.shape[0]
    if particle_count is None:
        raise ValueError("static particle count required")
    deltas = particles[:, None, :] - particles[None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    cost_scale = tf.maximum(
        tf.reduce_mean(cost), tf.cast(1.0e-3, particles.dtype)
    )
    kernel = tf.exp(-cost / (cost_scale * tf.cast(8.0, particles.dtype)))
    uniform = tf.fill(
        [particle_count],
        tf.cast(1.0 / particle_count, particles.dtype),
    )
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    tiny = tf.cast(1.0e-7, particles.dtype)
    for _ in range(32):
        left = uniform / (tf.linalg.matvec(kernel, right) + tiny)
        right = weights / (
            tf.linalg.matvec(tf.transpose(kernel), left) + tiny
        )
    coupling = left[:, None] * kernel * right[None, :]
    row_mass = tf.reduce_sum(coupling, axis=1)
    return (coupling @ particles) / row_mass[:, None], row_mass


def _pure_restore_value(
    particles: tf.Tensor, weights: tf.Tensor, design: tf.Tensor
) -> tf.Tensor:
    barycentric, row_mass = _pure_sinkhorn_value(particles, weights)
    target_mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered_source = particles - target_mean[None, :]
    target_covariance = tf.einsum(
        "n,ni,nj->ij", weights, centered_source, centered_source
    )
    centered_transport = barycentric - tf.reduce_mean(
        barycentric, axis=0, keepdims=True
    )
    transport_covariance = tf.einsum(
        "ni,nj->ij", centered_transport, centered_transport
    ) / tf.cast(tf.shape(particles)[0], particles.dtype)
    minimum_gap = tf.reduce_min(
        tf.linalg.eigvalsh(
            0.5
            * (
                target_covariance
                - transport_covariance
                + tf.transpose(target_covariance - transport_covariance)
            )
        )
    )
    valid = (
        tf.reduce_all(tf.math.is_finite(row_mass))
        & tf.reduce_all(row_mass > tf.cast(1.0e-7, particles.dtype))
        & (minimum_gap + tf.cast(1.0e-5, particles.dtype) > 0.0)
    )
    safe_barycentric = tf.where(
        valid,
        barycentric,
        tf.broadcast_to(target_mean[None, :], tf.shape(barycentric)),
    )
    forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        particles[None, :, :],
        weights[None, :],
        safe_barycentric[None, :, :],
        design[None, :, :],
        tf.constant([1.0e-5], particles.dtype),
    )
    return forward["particles"][0]


def restore_composition_audit() -> dict[str, Any]:
    particle_count = 36
    particles = tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION], [1201, 1], dtype=tf.float32
    )
    logits = tf.random.stateless_normal(
        [particle_count], [1201, 2], dtype=tf.float32
    )
    weights = tf.nn.softmax(logits)
    particle_tangent = 0.05 * tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION, PARAMETER_COUNT],
        [1201, 3],
        dtype=tf.float32,
    )
    raw_weight_tangent = tf.random.stateless_normal(
        [particle_count, PARAMETER_COUNT], [1201, 4], dtype=tf.float32
    )
    weight_tangent = weights[:, None] * (
        raw_weight_tangent
        - tf.reduce_sum(
            weights[:, None] * raw_weight_tangent, axis=0, keepdims=True
        )
    )
    design = cubature_design(
        dim=STATE_DIMENSION, num_particles=particle_count
    )

    @tf.function(jit_compile=False)
    def kernel(particles_value, weights_value, particles_dot, weights_dot):
        manual = genut_filter._restore_cloud_jvp_core(  # noqa: SLF001
            particles_value,
            weights_value,
            particles_dot,
            weights_dot,
            design,
            epsilon=8.0,
            sinkhorn_steps=16,
            balance_steps=16,
            ridge=1.0e-5,
            parameter_count=PARAMETER_COUNT,
        )
        ad = []
        for parameter_index in range(PARAMETER_COUNT):
            with tf.autodiff.ForwardAccumulator(
                (particles_value, weights_value),
                (
                    particles_dot[:, :, parameter_index],
                    weights_dot[:, parameter_index],
                ),
            ) as accumulator:
                value = _pure_restore_value(
                    particles_value, weights_value, design
                )
            ad.append(accumulator.jvp(value))
        return (
            manual["particles_tangent"],
            tf.stack(ad, axis=-1),
            manual["reset_valid"],
            manual["minimum_gap_eigenvalue"],
        )

    manual, ad, valid, gap = kernel(
        particles, weights, particle_tangent, weight_tangent
    )
    return {
        "particles": _metrics(manual, ad),
        "valid": bool(valid.numpy()),
        "minimum_gap_eigenvalue": float(gap.numpy()),
    }


def graph_case(
    adapter: Any,
    observations: tf.Tensor,
    *,
    particle_count: int,
    horizon: int,
    arm: str,
) -> dict[str, Any]:
    initial, process = _noise(SEED, particle_count, horizon)
    design = cubature_design(
        dim=STATE_DIMENSION, num_particles=particle_count
    )
    theta = tf.zeros([PARAMETER_COUNT], tf.float32)
    controls = _controls(arm)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def kernel(theta_value):
        with tf.GradientTape() as tape:
            tape.watch(theta_value)
            value, score, diagnostics = genut_filter.finite_value_score(
                adapter,
                theta_value,
                observations[:horizon],
                initial,
                process,
                design,
                **controls,
            )
        ad_score = tape.gradient(value, theta_value)
        return (
            value,
            score,
            ad_score,
            diagnostics["program_valid"],
            diagnostics["minimum_covariance_gap_eigenvalue"],
            diagnostics["maximum_normalized_shape_displacement"],
        )

    started = time.perf_counter()
    (
        value,
        manual_score,
        ad_score,
        valid,
        gap,
        shape_displacement,
    ) = kernel(theta)
    return {
        "particle_count": particle_count,
        "horizon": horizon,
        "arm": arm,
        "value": float(value.numpy()),
        "manual_score": _safe(manual_score),
        "ad_score": _safe(ad_score),
        "score_comparison": _metrics(manual_score, ad_score),
        "j0_comparison": _metrics(manual_score[0], ad_score[0]),
        "program_valid": bool(valid.numpy()),
        "minimum_covariance_gap_eigenvalue": float(gap.numpy()),
        "maximum_normalized_shape_displacement": float(
            shape_displacement.numpy()
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }


def _write_markdown(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "# GenUT Austria-SIR AD Root-Cause Result",
        "",
        f"- Classification: `{payload['decision']['root_cause_classification']}`",
        f"- Exact SIR score established: `{payload['decision']['exact_sir_score_established']}`",
        "",
        "| N | T | Arm | Manual graph j0 | AD graph j0 | Relative error | Valid |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["graph_horizon_ladder"]:
        lines.append(
            f"| {row['particle_count']} | {row['horizon']} | {row['arm']} | "
            f"{row['manual_score'][0]:.8g} | {row['ad_score'][0]:.8g} | "
            f"{row['j0_comparison']['symmetric_relative_l2']:.3g} | "
            f"{row['program_valid']} |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Item | Status |",
            "|---|---|",
        ]
    )
    for key, value in payload["inference_status"].items():
        lines.append(f"| `{key}` | `{value}` |")
    (output / "result.md").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if not LOCAL_CHECKPOINT.exists() or not PRIOR_FULL_CHECKPOINT.exists():
        raise RuntimeError("attempt05 checkpoints are required")
    local = json.loads(LOCAL_CHECKPOINT.read_text(encoding="utf-8"))
    prior_full = json.loads(PRIOR_FULL_CHECKPOINT.read_text(encoding="utf-8"))
    adapter = parameterized_austria_sir_candidate_adapter(
        latent_preclip=False
    )
    with tf.device("/CPU:0"):
        observations64 = sir.fixed_observed_path(81120, 20)
    if _tensor_sha(observations64) != EXPECTED_OBSERVATION_HASH:
        raise RuntimeError("fixed observation hash mismatch")
    observations = tf.cast(observations64, tf.float32)
    logical = tf.config.list_logical_devices("GPU")
    if REQUIRE_GPU and not logical:
        raise RuntimeError("GPU required")

    restore = restore_composition_audit()
    (output / "restore_composition.json").write_text(
        json.dumps(_safe(restore), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    graph_rows = []
    cases = (
        (36, 1, "none"),
        (36, 2, "none"),
        (36, 3, "none"),
        (36, 4, "none"),
        (36, 5, "none"),
        (36, 1, "diagonal"),
        (36, 2, "diagonal"),
    )
    for particle_count, horizon, arm in cases:
        row = graph_case(
            adapter,
            observations,
            particle_count=particle_count,
            horizon=horizon,
            arm=arm,
        )
        graph_rows.append(row)
        (output / "graph_checkpoint.json").write_text(
            json.dumps(_safe(graph_rows), indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        print(
            f"graph N={particle_count} T={horizon} arm={arm} "
            f"manual={row['manual_score'][0]:.9g} ad={row['ad_score'][0]:.9g} "
            f"rel={row['j0_comparison']['symmetric_relative_l2']:.3g}",
            flush=True,
        )

    local_comparisons = [
        local["law_and_callbacks"]["transition_total_tangent"],
        local["law_and_callbacks"]["observation_total_tangent"],
        local["transport"]["particles"],
        local["transport"]["coupling"],
        restore["particles"],
    ]
    local_comparisons.extend(
        row["comparisons"][name]
        for row in local["reset_condition_ladder"]
        for name in row["comparisons"]
    )
    local_comparisons.extend(
        row["particles"] for row in local["shape_maps"]
    )
    local_max_relative = max(
        float(row["symmetric_relative_l2"])
        for row in local_comparisons
    )
    graph_max_relative = max(
        float(row["score_comparison"]["symmetric_relative_l2"])
        for row in graph_rows
    )
    graph_j0_max_relative = max(
        float(row["j0_comparison"]["symmetric_relative_l2"])
        for row in graph_rows
    )
    invalid_graph_rows = sum(not row["program_valid"] for row in graph_rows)

    if invalid_graph_rows:
        classification = "finite_program_invalid_at_localization_rung"
    elif graph_j0_max_relative > 1.0e-2:
        classification = "manual_graph_jvp_mismatch"
    elif local_max_relative > 1.0e-2:
        classification = "manual_local_map_jvp_mismatch"
    else:
        classification = (
            "manual_jvp_consistent_fp32_xla_graph_instability_and_"
            "particle_variance_remain"
        )

    source_paths = (
        Path(__file__).resolve(),
        ROOT / "docs/benchmarks/run_genut_sir_ad_root_cause_20260817.py",
        PLAN,
        ROOT / "bayesfilter/highdim/cubature_genut_filter.py",
        ROOT / "bayesfilter/highdim/cubature_genut_adapters.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
    )
    payload = {
        "schema": "bayesfilter.genut_sir_ad_root_cause_terminal.v1",
        "status": "COMPLETE",
        "plan": str(PLAN.relative_to(ROOT)),
        "local_evidence": {
            "source": str(LOCAL_CHECKPOINT.relative_to(ROOT)),
            "source_sha256": _sha(LOCAL_CHECKPOINT),
            "law_and_callbacks": local["law_and_callbacks"],
            "transport": local["transport"],
            "reset_condition_ladder": local["reset_condition_ladder"],
            "shape_maps": local["shape_maps"],
            "restore_composition": restore,
        },
        "prior_xla_graph_rows": {
            "source": str(PRIOR_FULL_CHECKPOINT.relative_to(ROOT)),
            "source_sha256": _sha(PRIOR_FULL_CHECKPOINT),
            "rows": prior_full,
        },
        "graph_horizon_ladder": graph_rows,
        "decision": {
            "root_cause_classification": classification,
            "local_max_symmetric_relative_l2": local_max_relative,
            "graph_score_max_symmetric_relative_l2": graph_max_relative,
            "graph_j0_max_symmetric_relative_l2": graph_j0_max_relative,
            "invalid_graph_rows": invalid_graph_rows,
            "exact_sir_score_established": False,
            "statistical_ranking_supported": False,
            "default_or_hmc_readiness": False,
        },
        "decision_table": [
            {
                "decision": "classify manual JVP correctness",
                "primary_criterion_status": classification,
                "veto_diagnostic_status": {
                    "invalid_graph_rows": invalid_graph_rows,
                    "restore_valid": restore["valid"],
                },
                "main_uncertainty": (
                    "No admitted independent observed-data SIR score exists; "
                    "three seeds only describe particle variance"
                ),
                "next_justified_action": (
                    "If graph manual/AD agree, quantify FP32 XLA sensitivity "
                    "and particle variance separately; do not repair algebra"
                ),
                "not_concluded": "exact score, ranking, default, or HMC readiness",
            }
        ],
        "inference_status": {
            "hard_veto_screen": (
                "passed for valid graph rungs and local maps"
                if not invalid_graph_rows
                else "failed at one or more graph rungs"
            ),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "prior three-seed arm means/SD and XLA-versus-graph drift"
            ),
            "default_readiness": "not supported",
            "next_evidence_needed": (
                "FP64/no-TF32 or conditioned XLA ladder plus many-seed "
                "variance decomposition, followed by an admitted independent oracle"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "Graph AD may agree with graph manual JVP while XLA implements "
                "a numerically different FP32 finite program at long horizon"
            ),
            "overturning_result": (
                "A scale-material graph manual/AD mismatch on a valid rung, or "
                "a local-map mismatch that persists in FP64"
            ),
            "weakest_evidence": (
                "Full XLA AD remains unavailable and seed-level variance has only three replicates"
            ),
        },
        "manifest": {
            "command": sys.argv,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "environment": "/home/chakwong/anaconda3/envs/tftwogpu",
            "python": sys.executable,
            "tensorflow": tf.__version__,
            "host": platform.node(),
            "logical_gpus": [device.name for device in logical],
            "execution_target": (
                "gpu" if REQUIRE_GPU else "cpu_reference_cuda_intentionally_hidden"
            ),
            "physical_gpu_details": [
                tf.config.experimental.get_device_details(device)
                for device in tf.config.list_physical_devices("GPU")
            ],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "memory_policy": dict(GPU_MEMORY_POLICY),
            "jit_compile": {
                "terminal_graph_authority": False,
                "prior_primal_manual_comparator": True,
            },
            "cpu_reference_nonclaim": (
                None
                if REQUIRE_GPU
                else "debug/reference derivative evidence only; not GPU/XLA production evidence"
            ),
            "tf32": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "seed": SEED,
            "wall_time_seconds": time.perf_counter() - started,
            "output": str(output.relative_to(ROOT)),
            "source_hashes": {
                str(path.relative_to(ROOT)): _sha(path)
                for path in source_paths
            },
            "trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
                if REQUIRE_GPU
                else "cpu_reference_cuda_intentionally_hidden"
            ),
        },
        "upstream_vetoes": {
            "classifier_gaussian_exact_oracle": "FAILED_8_OF_9_CELLS",
            "genut_lgssm_kalman_oracle": "FAILED",
        },
        "nonclaims": [
            "no exact observed-data SIR score",
            "no classifier oracle admission",
            "no algorithm ranking",
            "no default or HMC readiness",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    _write_markdown(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.output.resolve())


if __name__ == "__main__":
    main()
