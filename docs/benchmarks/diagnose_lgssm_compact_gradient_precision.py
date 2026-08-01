"""Localize LGSSM compact-score failures across GPU precision modes.

This is a diagnostic-only wrapper around the production LGSSM compact adapter.
It does not alter transport behavior or emit leaderboard-admissible evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


_PRE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE.add_argument("--cuda-visible-devices", default="0")
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_PRE_ARGS.cuda_visible_devices)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from docs.benchmarks import (
    benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter as adapter,
)
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm


TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--expect-device-kind", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--tf32-mode", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--num-particles", type=int, default=256)
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=81120)
    parser.add_argument("--transport-policy", choices=("active-all", "no-resampling"), default="active-all")
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--row-chunk-size", type=int, default=256)
    parser.add_argument("--col-chunk-size", type=int, default=256)
    parser.add_argument("--particle-chunk-size", type=int, default=256)
    parser.add_argument(
        "--fd-steps",
        type=float,
        nargs="+",
        default=(0.01, 0.00492156660115185, 0.002, 0.001),
    )
    parser.add_argument("--compact-fd", action="store_true")
    parser.add_argument("--skip-center-decomposition", action="store_true")
    parser.add_argument("--skip-one-step-decomposition", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device_scope == "cpu" and args.expect_device_kind != "cpu":
        raise ValueError("CPU scope requires --expect-device-kind=cpu")
    if args.device_scope == "visible" and args.expect_device_kind != "gpu":
        raise ValueError("visible scope requires --expect-device-kind=gpu")
    if args.num_particles <= 1 or args.time_steps <= 0:
        raise ValueError("particle count and time steps must be positive")
    if any(step <= 0.0 for step in args.fd_steps):
        raise ValueError("FD steps must be positive")
    return args


def _module_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        batch_seeds=[args.seed],
        num_particles=args.num_particles,
        time_steps=args.time_steps,
        transport_policy=args.transport_policy,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        transport_gradient_mode=lgssm.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="full",
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        score_fd_step=None,
        score_fd_atol=5.0e-3,
        score_fd_rtol=5.0e-3,
        score_fd_tf32_mode="match",
        score_mode="compact-sensitivity",
        score_diagnostic_stage="score-only",
        score_reference_json=None,
        history_mode="value-only",
        warmups=0,
        repeats=1,
        dtype="float32",
        tf32_mode=args.tf32_mode,
        device=args.device,
        device_scope=args.device_scope,
        cuda_visible_devices=args.cuda_visible_devices,
        expect_device_kind=args.expect_device_kind,
        output=args.output,
        markdown_output=None,
        aggregate_score_shards=None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _tensor_fingerprint(tensors: Mapping[str, tf.Tensor]) -> dict[str, Any]:
    leaves = []
    aggregate = hashlib.sha256()
    for name in sorted(tensors):
        value = tf.convert_to_tensor(tensors[name])
        raw = bytes(tf.io.serialize_tensor(value).numpy())
        digest = hashlib.sha256(raw).hexdigest()
        aggregate.update(name.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))
        leaves.append(
            {
                "name": name,
                "shape": [int(dim) for dim in value.shape],
                "dtype": value.dtype.name,
                "sha256": digest,
            }
        )
    return {"aggregate_sha256": aggregate.hexdigest(), "leaves": leaves}


def _kalman_oracle(time_steps: int) -> dict[str, Any]:
    observations = tf.convert_to_tensor(
        lgssm._lgssm_dataset(lgssm.DATASET_SEED)["observations"][:time_steps],  # noqa: SLF001
        dtype=tf.float64,
    )
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=tf.float64)

    def value(candidate: tf.Tensor) -> tf.Tensor:
        phi = candidate[:3]
        q_scale = candidate[3]
        r_scale = candidate[4]
        return tf_kalman_log_likelihood(
            observations=observations,
            transition_offset=tf.zeros([lgssm.STATE_DIM], dtype=tf.float64),
            transition_matrix=tf.linalg.diag(phi),
            transition_covariance=tf.square(q_scale) * tf.eye(lgssm.STATE_DIM, dtype=tf.float64),
            observation_offset=tf.zeros([lgssm.OBS_DIM], dtype=tf.float64),
            observation_matrix=tf.constant(
                [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
                dtype=tf.float64,
            ),
            observation_covariance=tf.square(r_scale) * tf.eye(lgssm.OBS_DIM, dtype=tf.float64),
            initial_state_mean=tf.zeros([lgssm.STATE_DIM], dtype=tf.float64),
            initial_state_covariance=tf.linalg.diag(
                tf.square(q_scale) / (1.0 - tf.square(phi))
            ),
        )

    with tf.GradientTape() as tape:
        tape.watch(theta)
        objective = value(theta)
    score = tape.gradient(objective, theta)
    return {"objective": float(objective.numpy()), "score": score.numpy().tolist()}


def _validate_devices(values: tuple[tf.Tensor, ...], expected: str) -> list[str]:
    devices = sorted({str(value.device) for value in values})
    token = "GPU" if expected == "gpu" else "CPU"
    if not devices or not all(token in device.upper() for device in devices):
        raise ValueError(f"expected {expected} outputs, got {devices}")
    return devices


def _one_step_primal_decomposition(
    tensors: Mapping[str, tf.Tensor],
    candidate: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Compare the shared flow primal with the compact constant-H shortcut."""

    candidate = tf.reshape(tf.cast(candidate, lgssm.DTYPE), [len(lgssm.PARAMETER_NAMES)])
    initial_particles = tf.cast(tensors["initial_particles"], lgssm.DTYPE)
    transition_noise = tf.cast(tensors["transition_noise"], lgssm.DTYPE)
    observations = tf.cast(tensors["observations"], lgssm.DTYPE)
    batch_size, num_particles, state_dim = lgssm.core_tf._static_shape(  # noqa: SLF001
        initial_particles, "initial_particles"
    )
    components = lgssm._lgssm_components(candidate, batch_size)  # noqa: SLF001
    transition_matrix = components["transition_matrix"]
    transition_covariance = components["transition_covariance"]
    observation_covariance = components["observation_covariance"]
    observation_matrix = components["observation_matrix"]
    prior_means = tf.einsum("bnj,bdj->bnd", initial_particles, transition_matrix)
    pre_flow = prior_means + transition_noise[:, 0] * components["q_scale"]
    h_jac = tf.tile(
        observation_matrix[None, None, :, :],
        [batch_size, num_particles, 1, 1],
    )
    predicted_pre_flow = tf.einsum("md,bnd->bnm", observation_matrix, pre_flow)
    residual = observations[0][None, None, :] - predicted_pre_flow
    shared_flow, shared_aux = lgssm.core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
        pre_flow_particles=pre_flow,
        prior_means=prior_means,
        observation_jacobian=h_jac,
        observation_residual=residual,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
    )

    prior_chol = tf.linalg.cholesky(transition_covariance)
    prior_precision = tf.linalg.cholesky_solve(
        prior_chol, tf.eye(state_dim, dtype=lgssm.DTYPE)[None, :, :]
    )
    obs_chol = tf.linalg.cholesky(observation_covariance)
    obs_precision = tf.linalg.cholesky_solve(
        obs_chol, tf.eye(lgssm.OBS_DIM, dtype=lgssm.DTYPE)[None, :, :]
    )
    compact_pseudo = predicted_pre_flow + residual
    compact_base_post_precision = prior_precision[:, None, :, :] + tf.einsum(
        "od,boq,qe->bde",
        observation_matrix,
        obs_precision,
        observation_matrix,
    )[:, None, :, :]
    compact_post_covariance = tf.linalg.inv(compact_base_post_precision)
    compact_post_covariance = tf.tile(
        compact_post_covariance, [1, num_particles, 1, 1]
    )
    compact_info = tf.einsum("bde,bne->bnd", prior_precision, prior_means) + tf.einsum(
        "bnod,boq,bnq->bnd", h_jac, obs_precision, compact_pseudo
    )
    compact_post_mean = tf.einsum(
        "bnde,bne->bnd", compact_post_covariance, compact_info
    )
    compact_post_chol = tf.linalg.cholesky(compact_post_covariance)
    prior_inv = tf.linalg.triangular_solve(
        prior_chol, tf.eye(state_dim, dtype=lgssm.DTYPE)[None, :, :]
    )
    compact_affine = tf.einsum("bnij,bjk->bnik", compact_post_chol, prior_inv)
    delta = pre_flow - prior_means
    compact_post_flow = compact_post_mean + tf.einsum(
        "bnij,bnj->bni", compact_affine, delta
    )
    compact_logdet = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(compact_post_chol)), axis=-1
    ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)), axis=-1)[:, None]

    shaped_post_precision = prior_precision[:, None, :, :] + tf.einsum(
        "bnod,boq,bnqe->bnde", h_jac, obs_precision, h_jac
    )
    shaped_post_covariance = tf.linalg.inv(shaped_post_precision)
    shaped_post_mean = tf.einsum("bnde,bne->bnd", shaped_post_covariance, compact_info)
    shaped_post_chol = tf.linalg.cholesky(shaped_post_covariance)
    shaped_affine = tf.einsum("bnij,bjk->bnik", shaped_post_chol, prior_inv)
    shaped_post_flow = shaped_post_mean + tf.einsum(
        "bnij,bnj->bni", shaped_affine, delta
    )
    shaped_logdet = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(shaped_post_chol)), axis=-1
    ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)), axis=-1)[:, None]

    uniform = lgssm.core_tf.uniform_log_weights(batch_size, num_particles)

    def increment(post_flow: tf.Tensor, pre_density: tf.Tensor, logdet: tf.Tensor) -> tf.Tensor:
        transition_density = lgssm._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - prior_means, transition_covariance
        )
        predicted = tf.einsum("md,bnd->bnm", observation_matrix, post_flow)
        observation_density = lgssm._batched_gaussian_logpdf(  # noqa: SLF001
            predicted - observations[0][None, None, :], observation_covariance
        )
        _weights, value = lgssm.core_tf._normalize_log_weights(  # noqa: SLF001
            uniform + transition_density + observation_density - pre_density + logdet
        )
        return value

    shared_increment = increment(
        shared_flow.post_flow_particles,
        shared_flow.pre_flow_log_density,
        shared_flow.forward_log_det,
    )
    compact_pre_density = lgssm._batched_gaussian_logpdf(  # noqa: SLF001
        pre_flow - prior_means, transition_covariance
    )
    compact_increment = increment(compact_post_flow, compact_pre_density, compact_logdet)
    shaped_increment = increment(shaped_post_flow, compact_pre_density, shaped_logdet)

    def max_abs(lhs: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_max(tf.abs(lhs - rhs))

    return {
        "predicted_pre_flow_vs_shared_hx0_max_abs": max_abs(
            predicted_pre_flow,
            tf.einsum("bnod,bnd->bno", h_jac, pre_flow),
        ),
        "pseudo_observation_max_abs": max_abs(compact_pseudo, shared_aux.pseudo_observation),
        "compact_post_precision_vs_shared_max_abs": max_abs(
            compact_base_post_precision, shared_aux.post_precision
        ),
        "shaped_post_precision_vs_shared_max_abs": max_abs(
            shaped_post_precision, shared_aux.post_precision
        ),
        "compact_post_covariance_vs_shared_max_abs": max_abs(
            compact_post_covariance, shared_aux.post_covariance
        ),
        "shaped_post_covariance_vs_shared_max_abs": max_abs(
            shaped_post_covariance, shared_aux.post_covariance
        ),
        "compact_info_vs_shared_max_abs": max_abs(compact_info, shared_aux.info),
        "compact_post_flow_vs_shared_max_abs": max_abs(
            compact_post_flow, shared_flow.post_flow_particles
        ),
        "shaped_post_flow_vs_shared_max_abs": max_abs(
            shaped_post_flow, shared_flow.post_flow_particles
        ),
        "compact_logdet_vs_shared_max_abs": max_abs(
            compact_logdet, shared_flow.forward_log_det
        ),
        "shaped_logdet_vs_shared_max_abs": max_abs(
            shaped_logdet, shared_flow.forward_log_det
        ),
        "shared_increment": shared_increment[0],
        "compact_increment": compact_increment[0],
        "shaped_increment": shaped_increment[0],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    module_args = _module_args(args)
    precision = lgssm._configure_precision(module_args)  # noqa: SLF001
    physical = tf.config.list_physical_devices("GPU")
    for gpu in physical:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical = tf.config.list_logical_devices("GPU")
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=tf.float32)
    prepared = adapter._prepare_compact_xla_inputs(module_args)  # noqa: SLF001

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_score(candidate: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return adapter._compact_score_tensor_outputs(  # noqa: SLF001
            module_args, candidate, prepared
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_value(candidate: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return adapter._value_tensor_outputs(module_args, candidate, prepared)  # noqa: SLF001

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_value_original_particles(candidate: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = lgssm._same_target_value_from_components(  # noqa: SLF001
            prepared["tensors"], module_args, candidate
        )
        return result["objective"], result["log_likelihood"]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_score_reconstructed_particles(
        candidate: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        candidate_tensors = adapter._candidate_tensors(  # noqa: SLF001
            prepared["tensors"], candidate
        )
        result = lgssm._compact_value_and_score_from_components(  # noqa: SLF001
            candidate_tensors, module_args, candidate
        )
        return (
            result["objective"],
            result["log_likelihood"],
            result["gradient_tensor"],
            result["per_seed_gradient"],
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_one_step_decomposition(candidate: tf.Tensor) -> dict[str, tf.Tensor]:
        return _one_step_primal_decomposition(prepared["tensors"], candidate)

    started = time.perf_counter()
    with tf.device(args.device):
        score_started = time.perf_counter()
        score_outputs = compiled_score(theta)
        for value in score_outputs:
            value.numpy()
        score_seconds = time.perf_counter() - score_started
        center_value = compiled_value(theta)
        for value in center_value:
            value.numpy()
        center_value_original_particles = ()
        center_score_reconstructed_particles = ()
        if not args.skip_center_decomposition:
            center_value_original_particles = compiled_value_original_particles(theta)
            for value in center_value_original_particles:
                value.numpy()
            center_score_reconstructed_particles = compiled_score_reconstructed_particles(theta)
            for value in center_score_reconstructed_particles:
                value.numpy()
        one_step_decomposition: dict[str, tf.Tensor] = {}
        if not args.skip_one_step_decomposition:
            one_step_decomposition = compiled_one_step_decomposition(theta)
            for value in one_step_decomposition.values():
                value.numpy()
        direction = tf.one_hot(3, len(lgssm.PARAMETER_NAMES), dtype=tf.float32)
        fd_records = []
        for raw_step in args.fd_steps:
            step = tf.constant(raw_step, dtype=tf.float32)
            minus_theta = theta - step * direction
            plus_theta = theta + step * direction
            endpoint_started = time.perf_counter()
            minus = compiled_value(minus_theta)
            plus = compiled_value(plus_theta)
            for value in (*minus, *plus):
                value.numpy()
            denominator = plus_theta[3] - minus_theta[3]
            fd = (plus[0] - minus[0]) / denominator
            record = {
                "nominal_step": raw_step,
                "effective_step": float((denominator / 2.0).numpy()),
                "minus_q_scale": float(minus_theta[3].numpy()),
                "plus_q_scale": float(plus_theta[3].numpy()),
                "minus_objective": float(minus[0].numpy()),
                "plus_objective": float(plus[0].numpy()),
                "finite_difference": float(fd.numpy()),
                "gap_to_compact_jvp": float((fd - score_outputs[2][3]).numpy()),
            }
            if args.compact_fd:
                compact_minus = compiled_score_reconstructed_particles(minus_theta)
                compact_plus = compiled_score_reconstructed_particles(plus_theta)
                for value in (*compact_minus, *compact_plus):
                    value.numpy()
                compact_fd = (compact_plus[0] - compact_minus[0]) / denominator
                record.update(
                    {
                        "compact_minus_objective": float(compact_minus[0].numpy()),
                        "compact_plus_objective": float(compact_plus[0].numpy()),
                        "compact_finite_difference": float(compact_fd.numpy()),
                        "compact_fd_gap_to_compact_jvp": float(
                            (compact_fd - score_outputs[2][3]).numpy()
                        ),
                    }
                )
            record["endpoint_seconds"] = time.perf_counter() - endpoint_started
            fd_records.append(record)
    devices = _validate_devices(
        (
            *score_outputs,
            *center_value,
            *center_value_original_particles,
            *center_score_reconstructed_particles,
            *one_step_decomposition.values(),
            *minus,
            *plus,
        ),
        args.expect_device_kind,
    )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py",
        ROOT / "docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter.py",
        ROOT / "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    )
    ledh_payload: dict[str, Any] = {
        "objective": float(score_outputs[0].numpy()),
        "log_likelihood": float(score_outputs[1].numpy()[0]),
        "value_route_center_objective": float(center_value[0].numpy()),
        "score_value_center_gap": float((score_outputs[0] - center_value[0]).numpy()),
        "compact_score": score_outputs[2].numpy().tolist(),
        "per_seed_score": score_outputs[3].numpy().tolist(),
        "q_scale_fd_ladder": fd_records,
    }
    if center_value_original_particles:
        ledh_payload.update(
            {
                "value_route_original_particles_center_objective": float(
                    center_value_original_particles[0].numpy()
                ),
                "value_initial_particle_roundtrip_center_effect": float(
                    (center_value[0] - center_value_original_particles[0]).numpy()
                ),
                "score_route_reconstructed_particles_center_objective": float(
                    center_score_reconstructed_particles[0].numpy()
                ),
                "score_initial_particle_roundtrip_center_effect": float(
                    (center_score_reconstructed_particles[0] - score_outputs[0]).numpy()
                ),
                "duplicated_primal_center_gap_with_original_particles": float(
                    (score_outputs[0] - center_value_original_particles[0]).numpy()
                ),
                "duplicated_primal_center_gap_with_reconstructed_particles": float(
                    (center_score_reconstructed_particles[0] - center_value[0]).numpy()
                ),
            }
        )
    if one_step_decomposition:
        ledh_payload["one_step_primal_decomposition"] = {
            name: float(value.numpy()) for name, value in one_step_decomposition.items()
        }

    payload = {
        "schema_version": "bayesfilter.lgssm_compact_gradient_precision_diagnostic.v1",
        "artifact_status": "completed",
        "terminal_artifact": True,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "question": "Does the XLA compact q_scale JVP match stable FD of the identical fixed-noise LEDH scalar?",
        "evidence_class": (
            TRUST_BASIS if args.device_scope == "visible" else "cpu_hidden_debug_only"
        ),
        "diagnostic_only": True,
        "row_id": lgssm.ROW_ID,
        "parameter_names": list(lgssm.PARAMETER_NAMES),
        "theta": [float(value) for value in theta.numpy()],
        "shape": {
            "num_particles": args.num_particles,
            "time_steps": args.time_steps,
            "seed": args.seed,
        },
        "transport": {
            "policy": args.transport_policy,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "precision": precision,
        "jit_compile": True,
        "devices": devices,
        "physical_gpus": [str(device) for device in physical],
        "logical_gpus": [str(device) for device in logical],
        "prepared_tensor_fingerprint": _tensor_fingerprint(prepared["tensors"]),
        "ledh": ledh_payload,
        "kalman_oracle": _kalman_oracle(args.time_steps),
        "run_manifest": {
            "git_commit": _git_commit(),
            "command_argv": sys.argv,
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
            },
        },
        "nonclaims": [
            "not leaderboard admission",
            "not HMC or posterior correctness evidence",
            "not multi-seed uncertainty evidence",
            "scalar value proximity is not gradient correctness",
        ],
    }
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
