#!/usr/bin/env python3
"""Emit tiny CPU-hidden raw-route values for Phase 6 preservation checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

from docs.benchmarks import benchmark_ledh_same_target_actual_sv_score as actual_sv
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_ledh_same_target_generalized_sv_score as generalized_sv
from docs.benchmarks import benchmark_ledh_same_target_ksc_sv_score as ksc_sv
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm
from docs.benchmarks import benchmark_ledh_same_target_predator_prey_score as predator_prey


def _base_args(**overrides: Any):
    chunks = select_transport_chunks(2)

    class Args:
        batch_seeds = [81120]
        time_steps = 1
        num_particles = 2
        transport_policy = "active-all"
        sinkhorn_iterations = 1
        sinkhorn_epsilon = 1.0
        annealed_scaling = 0.9
        annealed_convergence_threshold = 1.0e-3
        transport_plan_mode = "streaming"
        transport_ad_mode = "full"
        row_chunk_size = chunks.row_chunk_size
        col_chunk_size = chunks.col_chunk_size
        particle_chunk_size = 2
        dtype = "float64"
        tf32_mode = "disabled"
        flow_observation_variance = 2.0
        historical_raw_diagnostic = True

    args = Args()
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _record(result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "objective": float(tf.convert_to_tensor(result["objective"])),
        "log_likelihood": tf.convert_to_tensor(result["log_likelihood"]).numpy().tolist(),
        "gradient_tensor": tf.convert_to_tensor(result["gradient_tensor"]).numpy().tolist(),
        "score_route": str(result["score_route"]),
    }
    for name in ("fixed_resampling_mask", "resampling_mask", "reset_mask"):
        if name in result:
            record[name] = tf.convert_to_tensor(result[name]).numpy().tolist()
    return record


def _lgssm() -> dict[str, Any]:
    args = _base_args(
        transport_gradient_mode=lgssm.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        score_mode="compact-sensitivity",
        score_fd_step=1.0e-5,
        score_fd_atol=5.0e-3,
        score_fd_rtol=5.0e-3,
        score_fd_tf32_mode="match",
    )
    lgssm._configure_precision(args)  # noqa: SLF001
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE)
    tensors = lgssm._build_lgssm_manual_tensors(args, theta)  # noqa: SLF001
    return _record(lgssm._compact_value_and_score_from_components(tensors, args, theta))  # noqa: SLF001


def _model(module, theta, *, particles: int, time_steps: int, chunks: int = 2):
    policy_chunks = select_transport_chunks(particles)
    gradient_mode = module.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
    args = _base_args(
        num_particles=particles,
        time_steps=time_steps,
        row_chunk_size=policy_chunks.row_chunk_size,
        col_chunk_size=policy_chunks.col_chunk_size,
        particle_chunk_size=chunks,
        transport_gradient_mode=gradient_mode,
    )
    module._configure_precision(args)  # noqa: SLF001
    return _record(module._compact_value_and_score_across_seeds(args, list(theta)))  # noqa: SLF001


def _fixed_sir() -> dict[str, Any]:
    args = _base_args(
        num_particles=2,
        time_steps=1,
        transport_gradient_mode=fixed_sir.p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        theta_values=[0.0, 0.0, 0.0],
    )
    fixed_sir._configure_precision(args)  # noqa: SLF001
    return _record(fixed_sir._compact_value_and_score_across_seeds(args, [0.0, 0.0, 0.0]))  # noqa: SLF001


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    routes = {
        "lgssm": _lgssm(),
        "fixed_sir": _fixed_sir(),
        "predator_prey": _model(predator_prey, predator_prey.TRUTH_THETA, particles=2, time_steps=1),
        "actual_sv": _model(actual_sv, actual_sv.TRUTH_THETA, particles=2, time_steps=1),
        "generalized_sv": _model(generalized_sv, generalized_sv.TRUTH_THETA, particles=2, time_steps=1),
        "ksc_sv": _model(ksc_sv, ksc_sv.TRUTH_THETA, particles=2, time_steps=1),
    }
    canonical = json.dumps(routes, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema_version": "bayesfilter.contract_e_phase6_raw_diagnostic_baseline.v1",
        "cpu_hidden": True,
        "routes": routes,
        "routes_sha256": hashlib.sha256(canonical).hexdigest(),
        "comparison_baseline": str(args.compare) if args.compare else None,
        "bitwise_json_identity": None,
    }
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        payload["bitwise_json_identity"] = routes == baseline["routes"]
        if not payload["bitwise_json_identity"]:
            raise SystemExit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
