"""Audit the q=20 SSL-LSTM state-space interface for a LEDH adapter.

This is a CPU diagnostic.  It exercises the actual batched structural target
callbacks and records the terms that are, and are not, available for a
Li-Coates LEDH-PFPF proposal.  It does not call a parameter-space affine map
and does not claim LEDH admission.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("q20 LEDH adapter audit requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError(
        "q20 LEDH adapter audit requires TF_FORCE_GPU_ALLOW_GROWTH=true"
    )

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("q20 LEDH adapter audit found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


TARGET = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
STRUCTURAL = ROOT / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-particle-authority-phase24-q20-ledh-adapter-"
    "audit-subplan-2026-08-25.md"
)
RUNNER = Path(__file__).resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_safe(item) for item in value)
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _shape(value: Any) -> list[int | None] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [None if dim is None else int(dim) for dim in shape.as_list()]


def _finite(value: Any) -> bool:
    tensor = tf.convert_to_tensor(value)
    return bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _callback_record(
    name: str,
    path: str,
    value: Any,
    *,
    expected_rank: int,
) -> dict[str, Any]:
    shape = _shape(value)
    return {
        "name": name,
        "path": path,
        "present": True,
        "shape": shape,
        "expected_rank": expected_rank,
        "rank_matches": shape is not None and len(shape) == expected_rank,
        "finite": _finite(value),
    }


def _source_symbols(source: str) -> dict[str, bool]:
    symbols = (
        "transition_log_prob",
        "observation_log_prob",
        "proposal_log_prob",
        "pre_flow",
        "post_flow",
        "covariance_lifecycle",
        "pseudo_time",
        "determinant",
        "log_density",
    )
    return {symbol: symbol in source.lower() for symbol in symbols}


def build_audit() -> dict[str, Any]:
    started = time.perf_counter()
    target = batch_native_complexity_posterior_target(
        20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh"
    )
    free = tf.constant(
        [[-0.35, -0.20, 0.15, 0.25], [0.40, 0.10, -0.25, -0.10]],
        dtype=tf.float64,
    )
    model, derivatives = target._batched_components(free)
    batch_size = int(free.shape[0])
    point_count = 3
    state_dim = int(model.state_dim)
    innovation_dim = int(model.innovation_dim)
    observation_dim = int(model.observation_dim)
    previous = tf.zeros([batch_size, point_count, state_dim], tf.float64)
    innovation = tf.zeros([batch_size, point_count, innovation_dim], tf.float64)
    next_state = model.transition(previous, innovation)
    observed = model.observe(next_state)
    structural_residual = model.deterministic_residual(
        previous, innovation, next_state
    )
    transition_jacobian = derivatives.transition_state_jacobian_fn(
        previous, innovation
    )
    innovation_jacobian = derivatives.transition_innovation_jacobian_fn(
        previous, innovation
    )
    parameter_transition = derivatives.d_transition_fn(previous, innovation)
    observation_jacobian = derivatives.observation_state_jacobian_fn(next_state)
    parameter_observation = derivatives.d_observation_fn(next_state)
    value, score, status = target.neutra_batch_log_prob_and_grad_status(free)

    target_source = inspect.getsource(type(target))
    structural_source = STRUCTURAL.read_text(encoding="utf-8")
    public_names = sorted(
        name
        for name in dir(target)
        if not name.startswith("_")
    )
    required_terms = {
        "initial_state_mean": {
            "status": "bound",
            "path": "model.initial_mean",
            "quantity": "batched initial state mean",
        },
        "initial_state_covariance": {
            "status": "bound",
            "path": "model.initial_covariance",
            "quantity": "batched initial state covariance",
        },
        "innovation_covariance": {
            "status": "bound",
            "path": "model.innovation_covariance",
            "quantity": "batched process/innovation covariance",
        },
        "observation_covariance": {
            "status": "bound",
            "path": "model.observation_covariance",
            "quantity": "batched observation covariance",
        },
        "transition_map": {
            "status": "bound_private_structural",
            "path": "target._batched_components(...)[0].transition_fn",
            "quantity": "transition(previous, innovation)",
        },
        "observation_map": {
            "status": "bound_private_structural",
            "path": "target._batched_components(...)[0].observation_fn",
            "quantity": "observation(state)",
        },
        "transition_state_jacobian": {
            "status": "bound_private_structural",
            "path": "target._batched_components(...)[1].transition_state_jacobian_fn",
            "quantity": "state Jacobian of transition map",
        },
        "transition_innovation_jacobian": {
            "status": "bound_private_structural",
            "path": "target._batched_components(...)[1].transition_innovation_jacobian_fn",
            "quantity": "innovation Jacobian of transition map",
        },
        "transition_log_density": {
            "status": "missing",
            "path": "missing explicit callback",
            "quantity": "log p(x_t | x_{t-1}) for a realized transition",
        },
        "observation_log_density": {
            "status": "missing",
            "path": "missing explicit callback",
            "quantity": "log p(y_t | x_t) for a realized state",
        },
        "pre_flow_proposal_density": {
            "status": "missing",
            "path": "missing explicit proposal object/callback",
            "quantity": "log density before the LEDH flow",
        },
        "post_flow_target_terms": {
            "status": "aggregate_only",
            "path": "target.neutra_batch_log_prob_and_grad_status",
            "quantity": "aggregate posterior value and score, not decomposed flow terms",
        },
        "per_step_covariance_lifecycle": {
            "status": "missing_explicit_contract",
            "path": "UKF internals are consumed by value/score, not returned as LEDH callbacks",
            "quantity": "per-particle covariance/linearization state across pseudo-time",
        },
        "pseudo_time_flow_matrices": {
            "status": "missing",
            "path": "missing LEDH pseudo-time coefficient/flow API",
            "quantity": "A_t, b_t, and determinant product for the proposal flow",
        },
    }
    hard_checks = {
        "target_instantiated": True,
        "model_shape_contract": all(
            shape is not None and all(dim is not None for dim in shape)
            for shape in (
                _shape(model.initial_mean),
                _shape(model.initial_covariance),
                _shape(model.innovation_covariance),
                _shape(model.observation_covariance),
            )
        ),
        "transition_finite": _finite(next_state),
        "observation_finite": _finite(observed),
        "derivatives_finite": all(
            _finite(value)
            for value in (
                transition_jacobian,
                innovation_jacobian,
                parameter_transition,
                observation_jacobian,
                parameter_observation,
            )
        ),
        "aggregate_target_finite": _finite(value) and _finite(score),
        "aggregate_target_status_present": bool(status),
    }
    missing_required = [
        name
        for name, record in required_terms.items()
        if record["status"] in {"missing", "missing_explicit_contract", "aggregate_only"}
    ]
    status_code = (
        "ADAPTER_NOT_READY_REPAIRABLE"
        if all(hard_checks.values())
        and missing_required
        else "ADAPTER_AUDIT_FAIL"
    )
    return {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.ledh_adapter_audit.v1",
        "status": status_code,
        "role": "q20_interface_admissibility_diagnostic",
        "target": {
            "q": 20,
            "target_scope": target.target_scope,
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "parameter_dim": target.parameter_dim,
            "parameter_names": list(target.parameter_names),
            "public_names": public_names,
            "source_symbols_target": _source_symbols(target_source),
            "source_symbols_structural": _source_symbols(structural_source),
        },
        "dimensions": {
            "batch": batch_size,
            "points": point_count,
            "state": state_dim,
            "innovation": innovation_dim,
            "observation": observation_dim,
        },
        "callback_checks": {
            "transition": _callback_record(
                "transition", "model.transition_fn", next_state, expected_rank=3
            ),
            "observation": _callback_record(
                "observation", "model.observation_fn", observed, expected_rank=3
            ),
            "deterministic_residual": _callback_record(
                "deterministic_residual",
                "model.deterministic_residual_fn",
                structural_residual,
                expected_rank=3,
            ),
            "transition_state_jacobian": _callback_record(
                "transition_state_jacobian",
                "derivatives.transition_state_jacobian_fn",
                transition_jacobian,
                expected_rank=4,
            ),
            "transition_innovation_jacobian": _callback_record(
                "transition_innovation_jacobian",
                "derivatives.transition_innovation_jacobian_fn",
                innovation_jacobian,
                expected_rank=4,
            ),
            "parameter_transition": _callback_record(
                "parameter_transition",
                "derivatives.d_transition_fn",
                parameter_transition,
                expected_rank=4,
            ),
            "observation_state_jacobian": _callback_record(
                "observation_state_jacobian",
                "derivatives.observation_state_jacobian_fn",
                observation_jacobian,
                expected_rank=4,
            ),
            "parameter_observation": _callback_record(
                "parameter_observation",
                "derivatives.d_observation_fn",
                parameter_observation,
                expected_rank=4,
            ),
        },
        "required_terms": required_terms,
        "hard_checks": hard_checks,
        "missing_required_terms": missing_required,
        "repair_boundary": {
            "smallest_next_step": (
                "add a repository-owned q20 adapter exposing frozen proposal, "
                "transition/observation log densities, per-step covariance state, "
                "and LEDH pseudo-time matrices; bind each term to this target signature"
            ),
            "classification": "extension_or_invention_until_source_and_measure_are_bound",
            "do_not_do": (
                "do not call the private structural UKF object or an affine parameter "
                "map source-faithful LEDH without the missing density lifecycle"
            ),
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_intentionally_hidden": True,
            "jit_compile": False,
            "random_seed": "deterministic_fixed_tensor_no_rng",
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "runner": _sha(RUNNER),
                "target": _sha(TARGET),
                "structural": _sha(STRUCTURAL),
                "plan": _sha(PLAN),
            },
        },
        "nonclaims": [
            "private structural callbacks do not establish a source-faithful LEDH proposal",
            "aggregate value/score does not provide proposal or observation density terms",
            "this audit does not establish q20 whitening, mode coverage, posterior correctness, or HMC readiness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise RuntimeError("output root must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    result = build_audit()
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    status = result["status"]
    summary = (
        "The target has finite batched structural transition, observation, and "
        "Jacobian callbacks, but it lacks an explicit LEDH proposal/density and "
        "per-step covariance lifecycle. The gap is repairable and LEDH admission "
        "is not granted."
    )
    (output / "result.md").write_text(
        "# Phase 24 q20 LEDH Adapter Audit\n\n"
        f"Status: `{status}`\n\n{summary}\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status == "ADAPTER_NOT_READY_REPAIRABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
