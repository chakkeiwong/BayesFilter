"""Streaming LGSSM scale benchmark for default GPU LEDH-PFPF-OT TF32.

This harness measures the streaming value path for the repository default DPF
transport target.  It is designed to separate avoidable storage from
algorithmic all-pairs OT compute:

* time recursion is a TensorFlow ``while_loop``;
* streaming OT returns no dense ``[B,N,N]`` matrix;
* likelihood-only mode avoids filtered-history output;
* ``--proposal-mode callback`` avoids storing full ``[B,T,N,D]`` pre-flow input.

Timings are descriptive only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters import (
    batched_annealed_warmstart_student_tf as warmstart_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners.common_tf import load_json


DEFAULT_DTYPE_NAME = "float32"
DEFAULT_TF32_MODE = "enabled"
DTYPE = tf.float32


NONCLAIMS = (
    "single synthetic LGSSM-shaped fixture only",
    "production/default target by owner directive",
    "no CPU/GPU ranking claim",
    "no posterior validity claim",
    "no active transport gradient validation claim",
    "no HMC readiness claim",
    "streaming removes dense transport storage but not all-pairs OT compute",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=200)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--obs-dim", type=int, default=20)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("callback", "tensor"),
        default="callback",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--particle-chunk-size", type=int, default=256)
    parser.add_argument("--warmstart-mode", choices=("none", "heuristic", "learned"), default="none")
    parser.add_argument("--warmstart-hidden-dim", type=int, default=32)
    parser.add_argument("--warmstart-checkpoint", default=None)
    parser.add_argument("--target-object-convention-id", default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--allow-random-learned-stub", action="store_true")
    parser.add_argument("--return-history", action="store_true")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--dtype", choices=("float64", "float32"), default=DEFAULT_DTYPE_NAME)
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default=DEFAULT_TF32_MODE,
    )
    parser.add_argument("--include-output-arrays", action="store_true")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state_dim and obs_dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.include_output_arrays and not args.return_history:
        raise ValueError("--include-output-arrays requires --return-history")
    learned_lineage_flags = {
        "--target-object-convention-id": args.target_object_convention_id,
        "--dataset-manifest": args.dataset_manifest,
        "--checkpoint-manifest": args.checkpoint_manifest,
    }
    if args.warmstart_mode != "learned":
        provided = [flag for flag, value in learned_lineage_flags.items() if value is not None]
        if provided:
            raise ValueError(
                "learned-mode provenance flags are only valid with --warmstart-mode learned: "
                + ", ".join(provided)
            )
    elif args.allow_random_learned_stub and args.warmstart_checkpoint is None:
        provided = [flag for flag, value in learned_lineage_flags.items() if value is not None]
        if provided:
            raise ValueError(
                "random learned stub mode cannot carry dataset/checkpoint provenance flags: "
                + ", ".join(provided)
            )
    else:
        missing = [flag for flag, value in learned_lineage_flags.items() if value is None]
        if missing:
            raise ValueError(
                "checkpoint-backed learned mode requires provenance flags: "
                + ", ".join(missing)
            )
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update({
        "dtype": args.dtype,
        "tf_dtype": DTYPE.name,
        "tf32_mode": args.tf32_mode,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
    })
    return metadata


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


def _fixture(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    batch_size = args.batch_size
    time_steps = args.time_steps
    num_particles = args.num_particles
    state_dim = args.state_dim
    obs_dim = args.obs_dim

    batch = np.arange(batch_size, dtype=np.float64)
    state_grid = np.linspace(-1.0, 1.0, state_dim, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    time_grid = np.arange(time_steps, dtype=np.float64)

    initial_particles = (
        0.08 * rng.standard_normal((batch_size, num_particles, state_dim))
        + 0.03 * state_grid[None, None, :]
        + 0.01 * particle_grid[None, :, None]
        + 0.0001 * batch[:, None, None]
    )

    diagonal = 0.86 + 0.08 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    transition_matrix = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        transition_matrix[row] = np.diag(diagonal + 0.00001 * row)
        transition_matrix[row] += 0.006 * np.eye(state_dim, k=1, dtype=np.float64)
        transition_matrix[row] += -0.004 * np.eye(state_dim, k=-1, dtype=np.float64)

    q_diag = 0.04 + 0.004 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    r_diag = 0.06 + 0.006 * np.linspace(0.0, 1.0, obs_dim, dtype=np.float64)
    transition_covariance = np.tile(np.diag(q_diag)[None, :, :], (batch_size, 1, 1))
    observation_covariance = np.tile(np.diag(r_diag)[None, :, :], (batch_size, 1, 1))

    observation_matrix = np.zeros((batch_size, obs_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        for obs_index in range(obs_dim):
            state_index = obs_index % state_dim
            observation_matrix[row, obs_index, state_index] = 1.0
            if state_dim > 1:
                observation_matrix[row, obs_index, (state_index + 1) % state_dim] = 0.025

    observations = 0.05 * np.sin(
        0.023 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0)
    )
    observations += 0.02 * np.cos(
        0.011 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0)
    )

    time_wave = 0.012 * np.sin(0.017 * time_grid[:, None] * (np.arange(state_dim) + 1))
    particle_wave = 0.006 * np.cos(0.11 * particle_grid[:, None] * (np.arange(state_dim) + 1))

    if args.transport_policy == "active-all":
        fixed_resampling_mask = np.ones((batch_size, time_steps), dtype=bool)
    elif args.transport_policy == "active-odd":
        mask = (np.arange(time_steps)[None, :] % 2) == 1
        fixed_resampling_mask = np.broadcast_to(mask, (batch_size, time_steps)).copy()
    else:
        fixed_resampling_mask = np.zeros((batch_size, time_steps), dtype=bool)

    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "observation_matrix": observation_matrix,
        "time_wave": time_wave,
        "particle_wave": particle_wave,
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _to_tensors(fixture: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        dtype = tf.bool if value.dtype == np.bool_ else DTYPE
        tensors[name] = tf.constant(value, dtype=dtype)
    if args.proposal_mode == "tensor":
        transitioned_initial = np.einsum(
            "bnj,bdj->bnd",
            fixture["initial_particles"],
            fixture["transition_matrix"],
        )
        pre_flow = (
            transitioned_initial[:, None, :, :]
            + fixture["time_wave"][None, :, None, :]
            + fixture["particle_wave"][None, None, :, :]
        )
        tensors["pre_flow_particles"] = tf.constant(pre_flow, dtype=DTYPE)
    return tensors


def _make_pre_flow_step_fn(tensors: dict[str, tf.Tensor]):
    def _pre_flow_step(particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        transitioned = tf.einsum("bnj,bdj->bnd", particles, tensors["transition_matrix"])
        time_wave = tensors["time_wave"][time_index][None, None, :]
        return transitioned + time_wave + tensors["particle_wave"][None, :, :]

    return _pre_flow_step


def _make_observation_fn(observation_matrix: tf.Tensor):
    def _observation(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    return _observation


def _make_observation_jacobian_fn(observation_matrix: tf.Tensor):
    def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
        batch_size = points.shape[0]
        num_particles = points.shape[1]
        if batch_size is None or num_particles is None:
            raise ValueError("benchmark fixture requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], DTYPE)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * np.pi, DTYPE))
        + logdet[:, None]
        + quad
    )


def _make_transition_log_density(
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
):
    def _transition_log_density(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance)

    return _transition_log_density


def _make_observation_log_density(
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
):
    def _observation_log_density(
        x: tf.Tensor,
        observation: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(
            predicted - observation[None, None, :],
            observation_covariance,
        )

    return _observation_log_density


def _prepare_warmstart_fn(
    args: argparse.Namespace,
    tensors: dict[str, tf.Tensor],
    *,
    warmstart_mode_override: str | None = None,
) -> tuple[callable | None, dict[str, Any]]:
    mode = args.warmstart_mode if warmstart_mode_override is None else warmstart_mode_override
    if mode == "none":
        return None, {
            "mode": "none",
            "checkpoint_path": None,
            "checkpoint_loaded": False,
            "random_stub_allowed": False,
            **_manifest_placeholders(),
        }
    if mode == "heuristic":
        def _heuristic(scaled_particles: tf.Tensor, log_weights: tf.Tensor, mask: tf.Tensor, epsilon: tf.Tensor):
            model = _HeuristicWarmstartModel()
            return model.predict_warmstart_state(
                scaled_particles,
                log_weights,
                epsilon,
                valid_mask=mask,
            )
        return _heuristic, {
            "mode": "heuristic",
            "checkpoint_path": None,
            "checkpoint_loaded": False,
            "random_stub_allowed": False,
            **_manifest_placeholders(),
        }
    if args.warmstart_checkpoint is None and not args.allow_random_learned_stub:
        raise ValueError(
            "learned warmstart mode requires --warmstart-checkpoint unless "
            "--allow-random-learned-stub is set"
        )
    config = warmstart_tf.BatchedAnnealedWarmstartConfigTF(
        particle_hidden_dim=args.warmstart_hidden_dim,
        pooled_hidden_dim=args.warmstart_hidden_dim,
    )
    model = warmstart_tf.BatchedAnnealedWarmstartStudentTF(config)
    # Build variables on a representative tensor shape before optionally loading weights.
    batch_size = int(tensors["initial_particles"].shape[0])
    num_particles = int(tensors["initial_particles"].shape[1])
    dummy_log_weights = tf.fill(
        [batch_size, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    _ = model(
        tf.cast(tensors["initial_particles"], DTYPE),
        dummy_log_weights,
        epsilon=tf.cast(args.sinkhorn_epsilon, DTYPE),
        training=False,
    )
    checkpoint_loaded = False
    checkpoint_path = None
    lineage_metadata = _manifest_placeholders()
    if args.warmstart_checkpoint is not None:
        checkpoint_path = str(Path(args.warmstart_checkpoint).expanduser().resolve())
        model.load_weights(checkpoint_path)
        checkpoint_loaded = True
        lineage_metadata = _assemble_learned_lineage_metadata(
            target_object_convention_id=args.target_object_convention_id,
            dataset_manifest=args.dataset_manifest,
            checkpoint_manifest=args.checkpoint_manifest,
            checkpoint_path=checkpoint_path,
        )

    def _learned(scaled_particles: tf.Tensor, log_weights: tf.Tensor, mask: tf.Tensor, epsilon: tf.Tensor):
        return warmstart_tf.predict_batched_annealed_warmstart_state_tf(
            model,
            scaled_particles,
            log_weights,
            epsilon,
            valid_mask=mask,
        )

    return _learned, {
        "mode": "learned",
        "checkpoint_path": checkpoint_path,
        "checkpoint_loaded": checkpoint_loaded,
        "random_stub_allowed": bool(args.allow_random_learned_stub),
        "model_config": {
            "particle_hidden_dim": args.warmstart_hidden_dim,
            "pooled_hidden_dim": args.warmstart_hidden_dim,
        },
        **lineage_metadata,
    }


def _manifest_placeholders() -> dict[str, Any]:
    return {
        "target_object_convention_id": None,
        "dataset_manifest_path": None,
        "dataset_manifest_reproducibility_digest": None,
        "checkpoint_manifest_path": None,
        "checkpoint_manifest_reproducibility_digest": None,
        "checkpoint_reference": None,
    }


def _resolve_repo_reference(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_required_manifest(path_str: str, *, manifest_label: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path_str).expanduser().resolve()
    payload = load_json(manifest_path)
    reproducibility_digest = payload.get("reproducibility_digest")
    if reproducibility_digest is None:
        raise ValueError(f"{manifest_label} manifest must contain top-level reproducibility_digest")
    return manifest_path, payload


def _require_matching_target_object_convention(
    payload: dict[str, Any],
    *,
    expected: str,
    manifest_label: str,
) -> None:
    observed = payload.get("target_object_convention_id")
    if observed != expected:
        raise ValueError(
            f"{manifest_label} manifest target_object_convention_id mismatch: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _assemble_learned_lineage_metadata(
    *,
    target_object_convention_id: str,
    dataset_manifest: str,
    checkpoint_manifest: str,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    dataset_manifest_path, dataset_payload = _load_required_manifest(
        dataset_manifest,
        manifest_label="dataset",
    )
    checkpoint_manifest_path, checkpoint_payload = _load_required_manifest(
        checkpoint_manifest,
        manifest_label="checkpoint",
    )
    _require_matching_target_object_convention(
        dataset_payload,
        expected=target_object_convention_id,
        manifest_label="dataset",
    )
    _require_matching_target_object_convention(
        checkpoint_payload,
        expected=target_object_convention_id,
        manifest_label="checkpoint",
    )
    expected_dataset_path = _resolve_repo_reference(dataset_manifest_path)
    expected_dataset_digest = dataset_payload["reproducibility_digest"]
    observed_dataset_path = checkpoint_payload.get("dataset_manifest_path")
    observed_dataset_digest = checkpoint_payload.get("dataset_manifest_reproducibility_digest")
    if observed_dataset_path != expected_dataset_path:
        raise ValueError(
            "checkpoint manifest dataset_manifest_path mismatch: "
            f"expected {expected_dataset_path!r}, observed {observed_dataset_path!r}"
        )
    if observed_dataset_digest != expected_dataset_digest:
        raise ValueError(
            "checkpoint manifest dataset_manifest_reproducibility_digest mismatch: "
            f"expected {expected_dataset_digest!r}, observed {observed_dataset_digest!r}"
        )
    checkpoint_reference = checkpoint_payload.get("checkpoint_reference")
    if checkpoint_reference is None:
        checkpoint_reference = checkpoint_payload.get("checkpoint_path")
    if checkpoint_reference is None and checkpoint_path is not None:
        checkpoint_reference = _resolve_repo_reference(Path(checkpoint_path))
    return {
        "target_object_convention_id": target_object_convention_id,
        "dataset_manifest_path": expected_dataset_path,
        "dataset_manifest_reproducibility_digest": expected_dataset_digest,
        "checkpoint_manifest_path": _resolve_repo_reference(checkpoint_manifest_path),
        "checkpoint_manifest_reproducibility_digest": checkpoint_payload["reproducibility_digest"],
        "checkpoint_reference": checkpoint_reference,
    }


class _HeuristicWarmstartModel:
    def predict_warmstart_state(self, scaled_particles, log_weights, epsilon, *, valid_mask):
        del scaled_particles, epsilon
        zeros = tf.zeros_like(log_weights)
        return warmstart_tf.build_annealed_transport_warmstart_state_tf(
            log_weights,
            zeros,
            log_weights,
            zeros,
            valid_mask,
        )


def _value_from_tensors(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    *,
    retained_teacher_warmstart_fn: callable | None = None,
):
    kwargs = dict(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=tensors["transition_matrix"],
        transition_covariance=tensors["transition_covariance"],
        observation_covariance=tensors["observation_covariance"],
        observation_fn=_make_observation_fn(tensors["observation_matrix"]),
        observation_jacobian_fn=_make_observation_jacobian_fn(tensors["observation_matrix"]),
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            tensors["transition_matrix"],
            tensors["transition_covariance"],
        ),
        observation_log_density_fn=_make_observation_log_density(
            tensors["observation_matrix"],
            tensors["observation_covariance"],
        ),
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=args.return_history,
        retained_teacher_warmstart_fn=retained_teacher_warmstart_fn,
    )
    if args.proposal_mode == "tensor":
        kwargs["pre_flow_particles"] = tensors["pre_flow_particles"]
    else:
        kwargs["pre_flow_step_fn"] = _make_pre_flow_step_fn(tensors)
    return streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(**kwargs)


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _validate_device(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "max": max(timings),
    }


def _preview(values: np.ndarray, *, limit: int = 8) -> list[float]:
    return [float(v) for v in values.reshape(-1)[: min(limit, values.size)]]


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Streaming Default GPU LEDH-PFPF-OT TF32 LGSSM Benchmark",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Shape: `{result['shape']}`",
        f"- Device request: `{result['device']}`",
        f"- Output devices: `{result['output_devices']}`",
        f"- Precision: `{result['precision']}`",
        f"- Proposal mode: `{result['proposal_mode']}`",
        f"- Warm-start mode: `{result['warmstart_mode']}`",
        f"- Return history: `{result['return_history']}`",
        f"- Transport: `{result['transport']}`",
        f"- Compile plus first call seconds: `{result['compile_and_first_call_seconds']}`",
        f"- Warm-call timing summary seconds: `{result['warm_call_timing_summary_seconds']}`",
        f"- Finite output: `{result['finite_output']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    fixture = _fixture(args)
    tensors = _to_tensors(fixture, args)

    include_cold_reference = args.warmstart_mode != "none"
    warmstart_fn, warmstart_metadata = _prepare_warmstart_fn(args, tensors, warmstart_mode_override=args.warmstart_mode)
    cold_warmstart_fn, _cold_warmstart_metadata = _prepare_warmstart_fn(args, tensors, warmstart_mode_override="none")

    if include_cold_reference:
        @tf.function(jit_compile=True, reduce_retracing=True)
        def compiled_outputs() -> tuple[tf.Tensor, ...]:
            result = _value_from_tensors(tensors, args, retained_teacher_warmstart_fn=warmstart_fn)
            cold_result = _value_from_tensors(tensors, args, retained_teacher_warmstart_fn=cold_warmstart_fn)
            return (
                result.log_likelihood,
                result.filtered_means,
                result.filtered_variances,
                result.ess_by_time,
                result.final_particles,
                result.max_row_residual,
                result.max_column_residual,
                cold_result.log_likelihood,
                cold_result.final_particles,
                cold_result.max_row_residual,
                cold_result.max_column_residual,
            )
    else:
        @tf.function(jit_compile=True, reduce_retracing=True)
        def compiled_outputs() -> tuple[tf.Tensor, ...]:
            result = _value_from_tensors(tensors, args, retained_teacher_warmstart_fn=warmstart_fn)
            return (
                result.log_likelihood,
                result.filtered_means,
                result.filtered_variances,
                result.ess_by_time,
                result.final_particles,
                result.max_row_residual,
                result.max_column_residual,
            )

    with tf.device(args.device):
        memory_before = _gpu_memory_info()
        start = time.perf_counter()
        outputs = compiled_outputs()
        _materialize(*outputs)
        compile_and_first = time.perf_counter() - start

        for _ in range(args.warmups):
            _materialize(*compiled_outputs())

        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = compiled_outputs()
            _materialize(*outputs)
            timings.append(time.perf_counter() - start)
        memory_after = _gpu_memory_info()

    if include_cold_reference:
        (
            value,
            filtered_means,
            filtered_variances,
            ess_by_time,
            final_particles,
            max_row_residual,
            max_column_residual,
            cold_value,
            cold_final_particles,
            cold_max_row_residual,
            cold_max_column_residual,
        ) = outputs
        cold_value_np = cold_value.numpy()
        cold_final_particles_np = cold_final_particles.numpy()
    else:
        value, filtered_means, filtered_variances, ess_by_time, final_particles, max_row_residual, max_column_residual = outputs
        cold_value_np = None
        cold_final_particles_np = None
        cold_max_row_residual = None
        cold_max_column_residual = None
    value_np = value.numpy()
    means_np = filtered_means.numpy()
    variances_np = filtered_variances.numpy()
    ess_np = ess_by_time.numpy()
    final_particles_np = final_particles.numpy()
    output_devices = _validate_device((value,), args.expect_device_kind)
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "precision": precision,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
        },
        "transport_policy": args.transport_policy,
        "seed": args.seed,
        "transport": {
            "plan_mode": "streaming",
            "gradient_mode": "raw",
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "dense_transport_matrix_materialized": False,
        },
        "proposal_mode": args.proposal_mode,
        "warmstart_mode": args.warmstart_mode,
        "warmstart_metadata": warmstart_metadata,
        "stores_full_pre_flow_particles": args.proposal_mode == "tensor",
        "particle_chunk_size": args.particle_chunk_size,
        "return_history": bool(args.return_history),
        "jit_compile": True,
        "compiled_unit": "streaming_batched_ledh_pfpf_ot_lgssm_value",
        "compile_and_first_call_seconds": compile_and_first,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "output_devices": output_devices,
        "output_shape": list(value_np.shape),
        "history_shapes": {
            "filtered_means": list(means_np.shape),
            "filtered_variances": list(variances_np.shape),
            "ess_by_time": list(ess_np.shape),
            "final_particles": list(final_particles_np.shape),
        },
        "finite_output": bool(
            np.isfinite(value_np).all()
            and np.isfinite(means_np).all()
            and np.isfinite(variances_np).all()
            and np.isfinite(ess_np).all()
            and np.isfinite(final_particles_np).all()
        ),
        "log_likelihood_preview": _preview(value_np),
        "filtered_means_preview": _preview(means_np),
        "filtered_variances_preview": _preview(variances_np),
        "ess_by_time_preview": _preview(ess_np),
        "final_particles_preview": _preview(final_particles_np),
        "teacher_replay_rmse_proxy": float(np.sqrt(np.mean(np.square(final_particles_np)))),
        "teacher_replay_max_abs_proxy": float(np.max(np.abs(final_particles_np))),
        "max_row_residual": float(max_row_residual.numpy()),
        "max_column_residual": float(max_column_residual.numpy()),
        "nonclaims": list(NONCLAIMS),
    }
    if include_cold_reference:
        delta = final_particles_np - cold_final_particles_np
        result.update({
            "cold_log_likelihood_preview": _preview(cold_value_np),
            "teacher_replay_rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "teacher_replay_max_abs": float(np.max(np.abs(delta))),
            "cold_max_row_residual": float(cold_max_row_residual.numpy()),
            "cold_max_column_residual": float(cold_max_column_residual.numpy()),
        })
    if args.include_output_arrays:
        result["output_arrays"] = {
            "log_likelihood": value_np.tolist(),
            "filtered_means": means_np.tolist(),
            "filtered_variances": variances_np.tolist(),
            "ess_by_time": ess_np.tolist(),
            "final_particles": final_particles_np.tolist(),
        }
    if not result["finite_output"]:
        raise FloatingPointError("streaming compiled benchmark emitted non-finite value")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
