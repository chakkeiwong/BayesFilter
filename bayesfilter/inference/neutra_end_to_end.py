"""Reusable end-to-end NeuTra campaign composition.

The module deliberately delegates target evaluation, training, native kernel
tuning, sequential sampling, and convergence to existing BayesFilter APIs. It
only composes those APIs and records an evidence-bound result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.hmc_kernel_tuning import (
    admitted_kernel_mechanics_payload_from_tuning_result,
    build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload,
    HMCKernelTuningConfig,
    build_retained_frozen_kernel_hmc_adapter_from_tuning_result,
    tune_hmc_kernel,
)
from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_batching import (
    batch_native_value_status_target_fn,
    bind_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_hmc import (
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    run_sequential_neutra_hmc,
)
from bayesfilter.inference.neutra_training import (
    NeuTraTrainingError,
    PlainDenseIAFTrainingConfig,
    restore_plain_dense_iaf_flow,
    train_plain_dense_iaf,
    train_plain_dense_iaf_infrastructure_segments,
)
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.runtime import atomic_write_json
from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)
from bayesfilter.testing.neutra_model_registry_tf import CellSpec, RecipeSpec


PASS_THRESHOLD = 0.05
SEVERE_THRESHOLD = 0.003
TRAINING_BATCH_SIZE = 128
SCREEN_STEPS = 500
FINAL_STEPS = 5000
FINAL_SEGMENT_STEPS = 1000
CHAIN_COUNT = 4


class NeuTraEndToEndError(RuntimeError):
    """Raised when one campaign cell violates the execution contract."""


class _RunState:
    """Persist diagnostic phase state without making it scientific evidence."""

    def __init__(self, root: Path, *, cell_id: str, output_root: Path) -> None:
        self.path = Path(root) / "run_state.json"
        self.cell_id = str(cell_id)
        self.output_root = str(output_root)
        self.phase = "created"
        self.terminal = False

    def update(self, phase: str, status: str = "running", **details: Any) -> None:
        self.phase = str(phase)
        payload = {
            "schema": "bayesfilter.neutra.all_models.run_state.v1",
            "role": "diagnostic_execution_state_not_scientific_evidence",
            "cell_id": self.cell_id,
            "output_root": self.output_root,
            "pid": os.getpid(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "status": str(status),
            "terminal_result_authority": "result.json_and_run_manifest",
            "details": dict(details),
        }
        atomic_write_json(self.path, payload)

    def complete(self, result: Mapping[str, Any]) -> None:
        self.terminal = True
        self.update(
            "terminal_result_written",
            status="completed",
            result_path=str(Path(self.output_root) / self.cell_id / "result.json"),
            decision=result.get("decision"),
            passed=result.get("passed"),
        )

    def __enter__(self) -> "_RunState":
        self.update("launch", status="started")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, _traceback: Any) -> bool:
        if exc_type is not None:
            self.update(
                self.phase,
                status="exception",
                exception_type=getattr(exc_type, "__name__", str(exc_type)),
                exception_message=str(exc_value),
            )
        elif not self.terminal:
            self.update(self.phase, status="returned_without_terminal_result")
        return False


class BatchNativeBoundAdapter:
    """Expose one inspected NeuTra batch target to all downstream consumers."""

    def __init__(self, base_adapter: Any, *, target_signature: str) -> None:
        self.base_adapter = base_adapter
        self.binding = bind_batch_native_neutra_target(
            base_adapter, target_signature=target_signature
        )
        self._value_status_target = batch_native_value_status_target_fn(self.binding)
        self.parameter_dim = int(base_adapter.parameter_dim)
        names = getattr(base_adapter, "parameter_names", None)
        self.parameter_names = (
            tuple(names()) if callable(names) else tuple(names or ())
        )
        self.target_signature = str(target_signature)
        self.supports_retained_flat_batch = True
        self.supports_retained_value_score_status = bool(
            callable(getattr(base_adapter, "neutra_batch_log_prob_and_grad_status", None))
        )

    def adapter_signature(self) -> str:
        return self.binding.adapter_signature

    def value_score_capability(self) -> Any:
        return value_score_capability(self.base_adapter)

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self.log_prob_and_grad_status(theta)
        return value, score

    def log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        values = tf.convert_to_tensor(theta, tf.float64)
        if values.shape.rank == 1:
            # HMC supplies one state at a time; rank-2-required targets still
            # run through the repository-issued batch-native callable.
            value, score, status = self.binding.invoke(values[tf.newaxis, :])
            return (
                tf.convert_to_tensor(value, tf.float64)[0],
                tf.convert_to_tensor(score, tf.float64)[0],
                {str(name): tf.convert_to_tensor(item)[0] for name, item in status.items()},
            )
        if values.shape.rank != 2:
            raise ValueError("end-to-end NeuTra target requires rank-1 or rank-2 positions")
        value, score, status = self.binding.invoke(values)
        return value, score, dict(status)

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        values = tf.convert_to_tensor(theta, tf.float64)
        if values.shape.rank == 1:
            value, _score, status = self.binding.invoke(values[tf.newaxis, :])
            normalized = {
                str(name): tf.convert_to_tensor(item)[0]
                for name, item in status.items()
            }
            condition_available = "innovation_condition_estimate" in status
            if not condition_available:
                normalized["innovation_condition_estimate"] = tf.ones_like(
                    tf.convert_to_tensor(value, tf.float64)[0]
                )
            normalized["innovation_condition_estimate_available"] = tf.constant(
                condition_available
            )
            return normalized
        if values.shape.rank != 2:
            raise ValueError("end-to-end target telemetry requires rank-1 or rank-2 positions")
        _value, status = self._value_status_target(values)
        return dict(status)


class TensorArchive:
    """Write disjoint warm-up and retained source-coordinate tensors."""

    def __init__(self, output_root: Path, target_signature: str) -> None:
        self.output_root = Path(output_root)
        self.target_signature = str(target_signature)

    def __call__(self, *, stage: str, chunk_index: int | None, latent_samples: Any,
                 model_samples: Any, seed: tuple[int, int] | None,
                 cumulative: bool) -> Mapping[str, Any]:
        label = "cumulative" if cumulative else f"chunk-{int(chunk_index):04d}"
        destination = self.output_root / stage / label
        if destination.exists():
            raise NeuTraEndToEndError(f"archive destination exists: {destination}")
        destination.mkdir(parents=True)
        latent = tf.convert_to_tensor(latent_samples, tf.float64)
        model = tf.convert_to_tensor(model_samples, tf.float64)
        latent_path = destination / "latent.tensor"
        model_path = destination / "model.tensor"
        tf.io.write_file(str(latent_path), tf.io.serialize_tensor(latent))
        tf.io.write_file(str(model_path), tf.io.serialize_tensor(model))
        metadata = {
            "schema": "bayesfilter.neutra.all_models.tensor_archive.v1",
            "stage": stage, "chunk_index": chunk_index, "cumulative": cumulative,
            "seed": seed, "target_signature": self.target_signature,
            "sample_shape": tuple(int(item) for item in latent.shape),
            "latent_path": str(latent_path), "model_path": str(model_path),
            "warmup_excluded_from_posterior": True,
        }
        atomic_write_json(destination / "metadata.json", metadata)
        return metadata


@dataclass(frozen=True)
class EndToEndConfig:
    output_root: Path
    screen_steps: int = SCREEN_STEPS
    final_steps: int = FINAL_STEPS
    final_segment_steps: int = FINAL_SEGMENT_STEPS
    screen_only: bool = False
    screen_result_path: Path | None = None
    expected_screen_result_sha256: str | None = None
    require_gpu: bool = True
    jit_compile: bool = True
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if int(self.screen_steps) <= 0 or int(self.final_steps) <= 0:
            raise ValueError("training steps must be positive")
        if int(self.final_segment_steps) <= 0:
            raise ValueError("final_segment_steps must be positive")
        if bool(self.require_gpu) and not bool(self.jit_compile):
            raise ValueError("GPU NeuTra training requires XLA")
        if (self.screen_result_path is None) != (
            self.expected_screen_result_sha256 is None
        ):
            raise ValueError(
                "screen_result_path and expected_screen_result_sha256 are required together"
            )
        if self.screen_result_path is not None and bool(self.screen_only):
            raise ValueError("a preserved screen handoff cannot be screen-only")
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.screen_result_path is not None:
            digest = str(self.expected_screen_result_sha256).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("expected_screen_result_sha256 must be SHA-256 hex")
            object.__setattr__(
                self, "screen_result_path", Path(self.screen_result_path)
            )
            object.__setattr__(self, "expected_screen_result_sha256", digest)


@dataclass(frozen=True)
class FrozenTransportValidationConfig:
    output_root: Path
    frozen_transport_path: Path
    expected_frozen_transport_sha256: str
    admitted_kernel_replay_path: Path | None = None
    tuning_only: bool = False
    require_gpu: bool = True
    jit_compile: bool = True
    seed_offset: int = 0

    def __post_init__(self) -> None:
        digest = str(self.expected_frozen_transport_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("expected_frozen_transport_sha256 must be SHA-256 hex")
        if bool(self.require_gpu) and not bool(self.jit_compile):
            raise ValueError("serious NeuTra validation requires GPU/XLA")
        if bool(self.tuning_only) and self.admitted_kernel_replay_path is not None:
            raise ValueError(
                "tuning_only validation cannot use an admitted kernel replay"
            )
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "frozen_transport_path",
            Path(self.frozen_transport_path),
        )
        object.__setattr__(self, "expected_frozen_transport_sha256", digest)
        object.__setattr__(self, "tuning_only", bool(self.tuning_only))
        if self.admitted_kernel_replay_path is not None:
            object.__setattr__(
                self, "admitted_kernel_replay_path", Path(self.admitted_kernel_replay_path)
            )
        object.__setattr__(self, "seed_offset", int(self.seed_offset))


@dataclass(frozen=True)
class FrozenTransportBroadGridConfig:
    """Broad-grid tuning inputs for one preserved transport."""

    output_root: Path
    frozen_transport_path: Path
    expected_frozen_transport_sha256: str
    root_seed: tuple[int, int]
    initial_step_size: float | None = None
    screen_results: int = 128
    require_gpu: bool = True
    jit_compile: bool = True

    def __post_init__(self) -> None:
        digest = str(self.expected_frozen_transport_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("expected_frozen_transport_sha256 must be SHA-256 hex")
        seed = tuple(int(item) for item in self.root_seed)
        if len(seed) != 2 or any(item < 0 for item in seed):
            raise ValueError("root_seed must contain two nonnegative integers")
        step = self.initial_step_size
        if step is not None and (not math.isfinite(float(step)) or float(step) <= 0.0):
            raise ValueError("initial_step_size must be positive and finite")
        if bool(self.require_gpu) and not bool(self.jit_compile):
            raise ValueError("serious broad-grid tuning requires GPU/XLA")
        if int(self.screen_results) <= 64:
            raise ValueError("broad-grid screen_results must exceed 64")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "frozen_transport_path", Path(self.frozen_transport_path))
        object.__setattr__(self, "expected_frozen_transport_sha256", digest)
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(
            self,
            "initial_step_size",
            None if step is None else float(step),
        )
        object.__setattr__(self, "screen_results", int(self.screen_results))


@dataclass(frozen=True)
class BroadGridSequentialConfig:
    """Inputs for sequential HMC from one unique broad-grid survivor."""

    output_root: Path
    frozen_transport_path: Path
    expected_frozen_transport_sha256: str
    broad_grid_result_path: Path
    expected_broad_grid_result_sha256: str
    chunk_results: int = 65
    require_gpu: bool = True
    jit_compile: bool = True

    def __post_init__(self) -> None:
        for name in (
            "expected_frozen_transport_sha256",
            "expected_broad_grid_result_sha256",
        ):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be SHA-256 hex")
            object.__setattr__(self, name, digest)
        if int(self.chunk_results) <= 1:
            raise ValueError("sequential HMC chunk_results must exceed one")
        if bool(self.require_gpu) and not bool(self.jit_compile):
            raise ValueError("serious sequential HMC requires GPU/XLA")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self, "frozen_transport_path", Path(self.frozen_transport_path)
        )
        object.__setattr__(
            self, "broad_grid_result_path", Path(self.broad_grid_result_path)
        )
        object.__setattr__(self, "chunk_results", int(self.chunk_results))


def run_neutra_frozen_transport_broad_grid_cell(
    *,
    spec: CellSpec,
    config: FrozenTransportBroadGridConfig,
) -> Mapping[str, Any]:
    """Tune the approved primary and one-hop grid without retained sampling.

    Compatibility wrapper: the procedure itself is owned by
    ``bayesfilter.inference.neutra_shared_procedure`` and runs the explicit
    ``operational_broad_grid_v1`` variant with ``sampling_launched``: False.
    """

    from bayesfilter.inference.neutra_shared_procedure import (
        OPERATIONAL_BROAD_GRID_V1,
        SharedNeuTraProcedureConfig,
        run_shared_neutra_procedure,
    )

    return run_shared_neutra_procedure(
        spec=spec,
        config=SharedNeuTraProcedureConfig(
            output_root=config.output_root,
            frozen_transport_path=config.frozen_transport_path,
            expected_frozen_transport_sha256=config.expected_frozen_transport_sha256,
            root_seed=config.root_seed,
            variant=OPERATIONAL_BROAD_GRID_V1,
            launch_sequential=False,
            initial_step_size=config.initial_step_size,
            screen_results=config.screen_results,
            require_gpu=config.require_gpu,
            jit_compile=config.jit_compile,
        ),
    )


def run_neutra_broad_grid_sequential_cell(
    *, spec: CellSpec, config: BroadGridSequentialConfig
) -> Mapping[str, Any]:
    """Run shared sequential HMC for the unique broad-grid survivor."""

    root = config.output_root / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"cell output root must be fresh: {root}")
    root.mkdir(parents=True)
    run_state = _RunState(root, cell_id=spec.cell_id, output_root=config.output_root)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=config.require_gpu
    )
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    for path, expected, label in (
        (
            config.frozen_transport_path,
            config.expected_frozen_transport_sha256,
            "frozen transport",
        ),
        (
            config.broad_grid_result_path,
            config.expected_broad_grid_result_sha256,
            "broad-grid result",
        ),
    ):
        if not path.is_file() or _file_sha256(path) != expected:
            raise NeuTraEndToEndError(f"{label} identity mismatch")

    adapter = spec.adapter_factory()
    if _target_signature(adapter) != spec.target_signature:
        raise NeuTraEndToEndError("broad-grid sampling target signature mismatch")
    bound = BatchNativeBoundAdapter(adapter, target_signature=spec.target_signature)
    loaded = load_frozen_neutra_artifact(
        _read_mapping(config.frozen_transport_path),
        expected_target_signature=spec.target_signature,
    )
    grid = _read_mapping(config.broad_grid_result_path)
    from bayesfilter.inference.neutra_shared_procedure import (
        extract_sequential_handoff_from_broad_grid_result,
    )

    try:
        handoff = extract_sequential_handoff_from_broad_grid_result(grid)
    except ValueError as error:
        raise NeuTraEndToEndError(str(error)) from error
    candidate = handoff["candidate"]
    step_size = float(handoff["step_size"])
    leapfrog = int(handoff["num_leapfrog_steps"])

    target_scope = f"{spec.cell_id}:fixed_neutra_broad_grid_unique_pair"
    tuned_adapter = _fixed_transport_adapter(bound, loaded.transport, target_scope)
    sequential_config = _sequential_config_from_pair(
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        spec=spec,
        chunk_results=config.chunk_results,
    )
    initial = _initial_state(tf, spec, spec.initial_seed)
    thresholds = RankNormalizedHMCThresholds(1.01, 1000.0, 400.0)
    archive = TensorArchive(root / "samples", spec.target_signature)

    def retained_diagnostic(draws: tf.Tensor) -> Mapping[str, Any]:
        return rank_normalized_hmc_diagnostics(
            draws, parameter_names=spec.parameter_names, thresholds=thresholds
        )

    def model_transform(samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(samples)
        flat = tf.reshape(samples, (-1, spec.parameter_dim))
        raw = loaded.transport.forward_batch(flat)
        physical = spec.physical_transform(tf, raw)
        return tf.reshape(physical, shape)

    run_state.update("sequential_sampling")
    sequential = run_sequential_neutra_hmc(
        adapter=tuned_adapter,
        initial_state=initial,
        model_transform=model_transform,
        parameter_names=spec.parameter_names,
        config=sequential_config,
        archive_callback=archive,
        retained_diagnostic_fn=retained_diagnostic,
    )
    truth_tail = (
        _truth_tail(spec, sequential["private_retained_raw"])
        if sequential.get("passed") is True
        else {
            "status": "NOT_EVALUATED_INVALID_SAMPLER",
            "minimum_p_truth": None,
            "parameter_rows": (),
            "reason": "sampler health or convergence gate failed",
        }
    )
    result = _base_result(
        spec,
        root,
        {
            "passed": bool(
                sequential.get("passed") is True and truth_tail["status"] == "PASS"
            ),
            "decision": _decision(sequential, truth_tail),
            "frozen_transport_input": {
                "path": str(config.frozen_transport_path),
                "sha256": config.expected_frozen_transport_sha256,
                "retrained": False,
            },
            "broad_grid_handoff": {
                "path": str(config.broad_grid_result_path),
                "sha256": config.expected_broad_grid_result_sha256,
                "unique_viable_pair": candidate,
                "procedure_variant": handoff["procedure_variant"],
                "stochastic_ranking_performed": False,
            },
            "sequential": {
                key: value
                for key, value in sequential.items()
                if not key.startswith("private_")
            },
            "truth_tail": truth_tail,
            "nonclaims": (
                "unique viable pair is not a statistically best pair",
                "one-seed truth-tail diagnostic only",
                "no exact-filter, superiority, or default-readiness claim",
            ),
        },
        memory_policy,
        started,
    )
    atomic_write_json(root / "result.json", result)
    atomic_write_json(
        root / "run_manifest.json",
        {
            "schema": "bayesfilter.neutra.broad_grid_sequential_manifest.v1",
            "cell_id": spec.cell_id,
            "target_signature": spec.target_signature,
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": tuple(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu_memory_policy": memory_policy,
            "device": "/GPU:0",
            "jit_compile": config.jit_compile,
            "tf32_execution_enabled": True,
            "chunk_results": config.chunk_results,
            "warmup_root_seed": sequential_config.warmup_seed,
            "retained_root_seed": sequential_config.retained_seed,
            "data_version": "bound_by_target_signature",
            "frozen_transport_sha256": config.expected_frozen_transport_sha256,
            "broad_grid_result_sha256": config.expected_broad_grid_result_sha256,
            "wall_time_seconds": time.monotonic() - started,
            "plan_path": spec.plan_path,
            "result_path": str(root / "result.json"),
        },
    )
    run_state.complete(result)
    return result


def run_neutra_end_to_end_cell(
    *,
    spec: CellSpec,
    config: EndToEndConfig,
) -> Mapping[str, Any]:
    """Train, tune, sample, and diagnose one registry cell."""

    root = config.output_root / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"cell output root must be fresh: {root}")
    root.mkdir(parents=True)
    run_state = _RunState(root, cell_id=spec.cell_id, output_root=config.output_root)
    run_state.update("launch", status="started")
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=config.require_gpu
    )
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    _write_manifest(root, spec, config, memory_policy, started)

    adapter = spec.adapter_factory()
    run_state.update("target_validation")
    observed_signature = _target_signature(adapter)
    if observed_signature != spec.target_signature:
        raise NeuTraEndToEndError(
            f"target signature mismatch for {spec.cell_id}: "
            f"{observed_signature} != {spec.target_signature}"
        )
    if int(getattr(adapter, "parameter_dim", -1)) != spec.parameter_dim:
        raise NeuTraEndToEndError(f"parameter dimension mismatch: {spec.cell_id}")
    bound_adapter = BatchNativeBoundAdapter(
        adapter, target_signature=spec.target_signature
    )

    center, factor, geometry_reference = spec.geometry_factory(tf)
    _validate_geometry(center, factor, spec.parameter_dim)
    geometry_payload = {
        "center": _json_ready(center),
        "factor": _json_ready(factor),
        "reference": geometry_reference,
    }
    atomic_write_json(root / "geometry.json", geometry_payload)
    run_state.update("geometry_validated", artifact_path=str(root / "geometry.json"))

    if config.screen_result_path is None:
        run_state.update("recipe_screen")
        selection = _screen_recipes(
            spec=spec,
            adapter=adapter,
            bound_adapter=bound_adapter,
            center=center,
            factor=factor,
            root=root,
            config=config,
            memory_policy=memory_policy,
        )
    else:
        run_state.update("recipe_screen_handoff")
        selection = _load_preserved_screen_selection(spec=spec, config=config)
        atomic_write_json(
            root / "screen_handoff.json",
            {
                "schema": "bayesfilter.neutra.screen_handoff.v1",
                "cell_id": spec.cell_id,
                "target_signature": spec.target_signature,
                "screen_result_path": str(config.screen_result_path),
                "screen_result_sha256": config.expected_screen_result_sha256,
                "selected_recipe_id": selection["selected_recipe_id"],
                "screen_steps": config.screen_steps,
            },
        )
    if not selection["passed"]:
        result = _base_result(spec, root, selection, memory_policy, started)
        atomic_write_json(root / "result.json", result)
        _write_manifest(root, spec, config, memory_policy, started)
        run_state.complete(result)
        return result

    if config.screen_only:
        result = _base_result(
            spec,
            root,
            {
                "passed": True,
                "decision": "SCREEN_COMPLETED_REPRESENTATIVE_NOMINATED",
                "selection": selection,
                "final_training_launched": False,
                "hmc_tuning_launched": False,
                "hmc_sampling_launched": False,
            },
            memory_policy,
            started,
        )
        atomic_write_json(root / "result.json", result)
        _write_manifest(root, spec, config, memory_policy, started)
        run_state.complete(result)
        return result

    recipe = _recipe_by_id(spec.recipes, str(selection["selected_recipe_id"]))
    run_state.update("final_training", selected_recipe_id=recipe.recipe_id)
    final = _train_final(
        spec=spec,
        adapter=adapter,
        bound_adapter=bound_adapter,
        center=center,
        factor=factor,
        recipe=recipe,
        root=root,
        config=config,
        memory_policy=memory_policy,
    )
    loaded = load_frozen_neutra_artifact(
        _read_mapping(Path(final["payload"]["path"])),
        expected_target_signature=spec.target_signature,
    )
    parity = final["frozen_trainable_parity"]
    run_state.update("tuning_admission")
    tuning = _native_tune(
        spec=spec,
        adapter=bound_adapter,
        loaded=loaded,
        root=root,
        config=config,
    )
    if tuning.passed is not True:
        result = _base_result(
            spec, root,
            {"passed": False, "selection": selection, "final": final, "parity": parity,
             "tuning": tuning.payload(), "decision": "TUNING_FAILED"},
            memory_policy, started,
        )
        atomic_write_json(root / "result.json", result)
        _write_manifest(root, spec, config, memory_policy, started)
        run_state.complete(result)
        return result

    if tuning.final_kernel_payload is None:
        raise NeuTraEndToEndError("public tuner passed without final kernel handoff")
    tuned_adapter = _fixed_transport_adapter(
        bound_adapter, loaded.transport, f"{spec.cell_id}:fixed_neutra_native_tuning"
    )
    replay = build_retained_frozen_kernel_hmc_adapter_from_tuning_result(
        adapter=tuned_adapter,
        tuning_result=tuning,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        target_scope=f"{spec.cell_id}:fixed_neutra_native_tuning",
    )
    _assert_public_tuning_contract(tuning, replay)
    run_state.update("sequential_sampling")
    initial = _initial_state(tf, spec, spec.initial_seed)
    thresholds = RankNormalizedHMCThresholds(1.01, 1000.0, 400.0)
    archive = TensorArchive(root / "samples", spec.target_signature)

    def retained_diagnostic(draws: tf.Tensor) -> Mapping[str, Any]:
        return rank_normalized_hmc_diagnostics(
            draws, parameter_names=spec.parameter_names, thresholds=thresholds
        )

    def model_transform(samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(samples)
        flat = tf.reshape(samples, (-1, spec.parameter_dim))
        raw = loaded.transport.forward_batch(flat)
        physical = spec.physical_transform(tf, raw)
        return tf.reshape(physical, shape)

    sequential = run_sequential_neutra_hmc(
        adapter=replay.adapter,
        initial_state=initial,
        model_transform=model_transform,
        parameter_names=spec.parameter_names,
        config=_sequential_config(replay, spec),
        archive_callback=archive,
        retained_diagnostic_fn=retained_diagnostic,
    )
    run_state.update("terminal_diagnostics")
    truth_tail = (
        _truth_tail(spec, sequential["private_retained_raw"])
        if sequential.get("passed") is True
        else {
            "status": "NOT_EVALUATED_INVALID_SAMPLER",
            "minimum_p_truth": None,
            "parameter_rows": (),
            "reason": "sampler health or convergence gate failed",
        }
    )
    result = _base_result(
        spec, root,
        {
            "passed": bool(sequential.get("passed") is True and truth_tail["status"] == "PASS"),
            "decision": _decision(sequential, truth_tail),
            "selection": selection,
            "final": final,
            "parity": parity,
            "tuning": tuning.payload(),
            "sequential": {key: value for key, value in sequential.items() if not key.startswith("private_")},
            "truth_tail": truth_tail,
            "primary_criterion": "all parameters p_truth >= 0.05 after valid converged NeuTra HMC",
            "nonclaims": (
                "one-seed truth-tail diagnostic only",
                "no distributional equivalence or filter exactness claim",
                "no sampler superiority or default-readiness claim",
            ),
        },
        memory_policy,
        started,
    )
    atomic_write_json(root / "result.json", result)
    _write_manifest(root, spec, config, memory_policy, started)
    run_state.complete(result)
    return result


def run_neutra_frozen_transport_validation_cell(
    *,
    spec: CellSpec,
    config: FrozenTransportValidationConfig,
) -> Mapping[str, Any]:
    """Tune and validate one preserved transport without retraining it."""

    root = config.output_root / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"cell output root must be fresh: {root}")
    root.mkdir(parents=True)
    run_state = _RunState(root, cell_id=spec.cell_id, output_root=config.output_root)
    run_state.update("launch", status="started")
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=config.require_gpu
    )
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    frozen_path = config.frozen_transport_path
    if not frozen_path.is_file():
        raise NeuTraEndToEndError(f"frozen transport does not exist: {frozen_path}")
    observed_sha256 = _file_sha256(frozen_path)
    if observed_sha256 != config.expected_frozen_transport_sha256:
        raise NeuTraEndToEndError(
            "frozen transport SHA-256 mismatch: "
            f"{observed_sha256} != {config.expected_frozen_transport_sha256}"
        )

    adapter = spec.adapter_factory()
    run_state.update("target_validation")
    observed_signature = _target_signature(adapter)
    if observed_signature != spec.target_signature:
        raise NeuTraEndToEndError(
            f"target signature mismatch for {spec.cell_id}: "
            f"{observed_signature} != {spec.target_signature}"
        )
    if int(getattr(adapter, "parameter_dim", -1)) != spec.parameter_dim:
        raise NeuTraEndToEndError(f"parameter dimension mismatch: {spec.cell_id}")
    bound_adapter = BatchNativeBoundAdapter(
        adapter, target_signature=spec.target_signature
    )
    loaded = load_frozen_neutra_artifact(
        _read_mapping(frozen_path),
        expected_target_signature=spec.target_signature,
    )
    transport_input = {
        "path": str(frozen_path),
        "sha256": observed_sha256,
        "target_signature": loaded.manifest.target_signature,
        "training_state_hash": loaded.manifest.training_state_hash,
        "retrained": False,
    }
    atomic_write_json(root / "frozen_transport_input.json", transport_input)
    run_state.update("transport_validated", artifact_path=str(root / "frozen_transport_input.json"))

    execution = _admitted_kernel_execution(config)
    replay_path = config.admitted_kernel_replay_path
    tuning = None
    if replay_path is None:
        tuning = _native_tune(
            spec=spec,
            adapter=bound_adapter,
            loaded=loaded,
            root=root,
            config=EndToEndConfig(
                output_root=config.output_root,
                seed_offset=config.seed_offset,
            ),
        )
    else:
        if not replay_path.is_file():
            raise NeuTraEndToEndError(
                f"admitted kernel replay artifact does not exist: {replay_path}"
            )
    if tuning is not None and tuning.passed is not True:
        result = _base_result(
            spec,
            root,
            {
                "passed": False,
                "decision": "TUNING_FAILED",
                "frozen_transport_input": transport_input,
                "tuning": tuning.payload(),
                "nonclaims": (
                    "preserved-transport public-tuner validation only",
                    "no convergence, truth recovery, or NeuTra validity claim",
                ),
            },
            memory_policy,
            started,
        )
        atomic_write_json(root / "result.json", result)
        _write_frozen_validation_manifest(
            root=root,
            spec=spec,
            config=config,
            memory_policy=memory_policy,
            started=started,
            transport_input=transport_input,
            tuning_seed=tuning.config.seed,
            sequential_config=None,
            admitted_kernel_replay=None,
        )
        run_state.complete(result)
        return result

    if config.tuning_only:
        assert tuning is not None
        result = _base_result(
            spec,
            root,
            {
                "passed": bool(tuning.passed),
                "decision": (
                    "TUNING_ONLY_PASS" if tuning.passed else "TUNING_FAILED"
                ),
                "frozen_transport_input": transport_input,
                "tuning": tuning.payload(),
                "sampling_launched": False,
                "tuning_only": True,
                "nonclaims": (
                    "preserved-transport public-tuner validation only",
                    "no sequential HMC sampling claim",
                    "no convergence, truth recovery, or NeuTra validity claim",
                ),
            },
            memory_policy,
            started,
        )
        atomic_write_json(root / "result.json", result)
        _write_frozen_validation_manifest(
            root=root,
            spec=spec,
            config=config,
            memory_policy=memory_policy,
            started=started,
            transport_input=transport_input,
            tuning_seed=tuning.config.seed,
            sequential_config=None,
            admitted_kernel_replay=None,
        )
        run_state.complete(result)
        return result

    tuned_adapter = _fixed_transport_adapter(
        bound_adapter,
        loaded.transport,
        f"{spec.cell_id}:fixed_neutra_native_tuning",
    )
    target_scope = f"{spec.cell_id}:fixed_neutra_native_tuning"
    if replay_path is None:
        assert tuning is not None
        mechanics = admitted_kernel_mechanics_payload_from_tuning_result(
            adapter=tuned_adapter,
            tuning_result=tuning,
            initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
            target_signature=spec.target_signature,
            target_scope=target_scope,
            execution=execution,
        )
        replay_path = root / "admitted_kernel_mechanics.json"
        atomic_write_json(replay_path, mechanics)
    else:
        mechanics = _read_mapping(replay_path)
    replay = build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
        adapter=tuned_adapter,
        mechanics_payload=mechanics,
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        target_signature=spec.target_signature,
        target_scope=target_scope,
        execution=execution,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
    )
    if tuning is not None:
        _assert_public_tuning_contract(tuning, replay)
    sequential_config = _sequential_config(
        replay,
        spec,
        seed_offset=config.seed_offset,
    )
    initial = _initial_state(tf, spec, spec.initial_seed)
    thresholds = RankNormalizedHMCThresholds(1.01, 1000.0, 400.0)
    archive = TensorArchive(root / "samples", spec.target_signature)

    def retained_diagnostic(draws: tf.Tensor) -> Mapping[str, Any]:
        return rank_normalized_hmc_diagnostics(
            draws, parameter_names=spec.parameter_names, thresholds=thresholds
        )

    def model_transform(samples: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(samples)
        flat = tf.reshape(samples, (-1, spec.parameter_dim))
        raw = loaded.transport.forward_batch(flat)
        physical = spec.physical_transform(tf, raw)
        return tf.reshape(physical, shape)

    run_state.update("sequential_sampling")
    sequential = run_sequential_neutra_hmc(
        adapter=replay.adapter,
        initial_state=initial,
        model_transform=model_transform,
        parameter_names=spec.parameter_names,
        config=sequential_config,
        archive_callback=archive,
        retained_diagnostic_fn=retained_diagnostic,
    )
    run_state.update("terminal_diagnostics")
    truth_tail = (
        _truth_tail(spec, sequential["private_retained_raw"])
        if sequential.get("passed") is True
        else {
            "status": "NOT_EVALUATED_INVALID_SAMPLER",
            "minimum_p_truth": None,
            "parameter_rows": (),
            "reason": "sampler health or convergence gate failed",
        }
    )
    result = _base_result(
        spec,
        root,
        {
            "passed": bool(
                sequential.get("passed") is True and truth_tail["status"] == "PASS"
            ),
            "decision": _decision(sequential, truth_tail),
            "frozen_transport_input": transport_input,
            "tuning": None if tuning is None else tuning.payload(),
            "admitted_kernel_replay": {
                "path": str(replay_path),
                "mechanics_sha256": mechanics.get("mechanics_sha256"),
                "tuning_provenance": mechanics.get("tuning_provenance"),
                "source": "tuner" if config.admitted_kernel_replay_path is None else "persisted_artifact",
            },
            "sequential": {
                key: value
                for key, value in sequential.items()
                if not key.startswith("private_")
            },
            "truth_tail": truth_tail,
            "primary_criterion": (
                "all parameters p_truth >= 0.05 after valid converged NeuTra HMC"
            ),
            "nonclaims": (
                "one-seed truth-tail diagnostic only",
                "no distributional equivalence or filter exactness claim",
                "no sampler superiority or default-readiness claim",
            ),
        },
        memory_policy,
        started,
    )
    atomic_write_json(root / "result.json", result)
    _write_frozen_validation_manifest(
        root=root,
        spec=spec,
        config=config,
        memory_policy=memory_policy,
        started=started,
        transport_input=transport_input,
        tuning_seed=(None if tuning is None else tuning.config.seed),
        sequential_config=sequential_config,
        admitted_kernel_replay={
            "path": str(replay_path),
            "mechanics_sha256": mechanics.get("mechanics_sha256"),
            "source": (
                "tuner"
                if config.admitted_kernel_replay_path is None
                else "persisted_artifact"
            ),
        },
    )
    run_state.complete(result)
    return result


def run_neutra_preflight_cell(
    *,
    spec: CellSpec,
    recipe_id: str,
    output_root: Path,
    steps: int = 1,
) -> Mapping[str, Any]:
    """Exercise one fresh GPU/XLA training and native-tuner route."""

    root = Path(output_root) / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"preflight output must be fresh: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    adapter = spec.adapter_factory()
    if _target_signature(adapter) != spec.target_signature:
        raise NeuTraEndToEndError("preflight target signature mismatch")
    bound = BatchNativeBoundAdapter(adapter, target_signature=spec.target_signature)
    center, factor, geometry_reference = spec.geometry_factory(tf)
    _validate_geometry(center, factor, spec.parameter_dim)
    recipe = _recipe_by_id(spec.recipes, recipe_id)
    atomic_write_json(
        root / "preflight_progress.json",
        {
            "schema": "bayesfilter.neutra.all_models.preflight_progress.v1",
            "stage": "training_started",
            "cell_id": spec.cell_id,
            "recipe_id": recipe_id,
        },
    )
    training = _train_and_score(
        spec=spec,
        adapter=adapter,
        bound_adapter=bound,
        center=center,
        factor=factor,
        recipe=recipe,
        output_root=root / "training-job",
        steps=int(steps),
        seed=spec.initial_seed,
        memory_policy=memory_policy,
    )
    atomic_write_json(
        root / "preflight_progress.json",
        {
            "schema": "bayesfilter.neutra.all_models.preflight_progress.v1",
            "stage": "training_parity_heldout_passed",
            "cell_id": spec.cell_id,
            "recipe_id": recipe_id,
            "training_state_hash": training["training_state_hash"],
        },
    )
    loaded = load_frozen_neutra_artifact(
        _read_mapping(Path(training["payload"]["path"])),
        expected_target_signature=spec.target_signature,
    )
    atomic_write_json(
        root / "preflight_progress.json",
        {
            "schema": "bayesfilter.neutra.all_models.preflight_progress.v1",
            "stage": "native_tuner_contract_started",
            "cell_id": spec.cell_id,
            "recipe_id": recipe_id,
        },
    )
    tiny_tuning = tune_hmc_kernel(
        adapter=_fixed_transport_adapter(
            bound, loaded.transport, f"{spec.cell_id}:preflight_native_tuning"
        ),
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        config=HMCKernelTuningConfig.smoke(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            mass_policy="fixed_identity",
            chain_execution_mode="tf_function",
            use_xla=True,
            target_scope=f"{spec.cell_id}:preflight_native_tuning",
            target_status_trace_policy="per_chain_step",
            source="bayesfilter.neutra_all_models_preflight.public_tuner",
        ),
        output_dir=root / "tuning",
    )
    atomic_write_json(
        root / "preflight_progress.json",
        {
            "schema": "bayesfilter.neutra.all_models.preflight_progress.v1",
            "stage": "native_tuner_contract_completed",
            "cell_id": spec.cell_id,
            "recipe_id": recipe_id,
            "tuner_passed": tiny_tuning.passed,
        },
    )
    if tiny_tuning.geometry is None:
        raise NeuTraEndToEndError("preflight public tuner did not emit geometry")
    tuning_payload = _read_mapping(root / "tuning" / "hmc_kernel_tuning_result.json")
    bootstrap_mechanics_passed = _preflight_bootstrap_mechanics_passed(
        tuning_payload
    )
    if not bootstrap_mechanics_passed:
        raise NeuTraEndToEndError(
            "preflight public tuner did not prove real HMC mechanics: "
            f"{tiny_tuning.hard_vetoes} / {tiny_tuning.repair_triggers}"
        )
    successful_rows = (tiny_tuning,)
    result = {
        "schema": "bayesfilter.neutra.all_models.preflight_result.v1",
        "cell_id": spec.cell_id,
        "passed": True,
        "decision": "PASS_ENGINEERING_PREFLIGHT",
        "target_signature": spec.target_signature,
        "recipe_id": recipe_id,
        "steps": int(steps),
        "geometry_reference": geometry_reference,
        "training": training,
        "native_tuning_route_executed": True,
        "native_hmc_runner_executed": True,
        "bootstrap_hmc_mechanics_passed": bootstrap_mechanics_passed,
        "candidate_tuning_passed": tiny_tuning.passed is True,
        "candidate_tuning_final_status": tiny_tuning.final_status,
        "candidate_rejection_is_engineering_preflight_compatible": True,
        "successful_real_tuning_run_count": len(successful_rows),
        "native_tuning_scientific_pass_required": False,
        "tiny_tuning": tuning_payload,
        "gpu_memory_policy": memory_policy,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": (
            "one-step engineering smoke only",
            "the one-step transport may be rejected by scientific tuning gates",
            "tiny tuning budgets are not kernel admission evidence",
            "no convergence, truth recovery, or scientific claim",
        ),
    }
    atomic_write_json(root / "result.json", result)
    _write_manifest(
        root,
        spec,
        EndToEndConfig(output_root=Path(output_root), screen_steps=steps, final_steps=steps, screen_only=True),
        memory_policy,
        started,
    )
    return result


def run_neutra_training_throughput_cell(
    *,
    spec: CellSpec,
    recipe_id: str,
    output_root: Path,
    repeated_steps: int = 25,
) -> Mapping[str, Any]:
    """Measure target-specific training throughput without entering HMC."""

    if int(repeated_steps) < 2:
        raise ValueError("training throughput requires at least two repeated steps")
    root = Path(output_root) / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"throughput output must be fresh: {root}")
    root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    adapter = spec.adapter_factory()
    if _target_signature(adapter) != spec.target_signature:
        raise NeuTraEndToEndError("throughput target signature mismatch")
    bound = BatchNativeBoundAdapter(adapter, target_signature=spec.target_signature)
    center, factor, geometry_reference = spec.geometry_factory(tf)
    _validate_geometry(center, factor, spec.parameter_dim)
    recipe = _recipe_by_id(spec.recipes, recipe_id)

    rung_rows = []
    for rung_id, steps in (("compile_plus_first", 1), ("repeated", int(repeated_steps))):
        rung_started = time.monotonic()
        try:
            training = _train_and_score(
                spec=spec,
                adapter=adapter,
                bound_adapter=bound,
                center=center,
                factor=factor,
                recipe=recipe,
                output_root=root / rung_id,
                steps=steps,
                seed=spec.initial_seed,
                memory_policy=memory_policy,
            )
        except NeuTraTrainingError as exc:
            failed_rung_elapsed = time.monotonic() - rung_started
            failure = {
                "schema": "bayesfilter.neutra.training_throughput_failure.v1",
                "cell_id": spec.cell_id,
                "passed": False,
                "decision": "TRAINING_CANDIDATE_HARD_VETO",
                "classification": "target_status_or_training_numerics_failure",
                "target_signature": spec.target_signature,
                "recipe_id": recipe_id,
                "failed_rung": rung_id,
                "failed_rung_steps": steps,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_rung_end_to_end_seconds": failed_rung_elapsed,
                "completed_rung_rows": tuple(rung_rows),
                "gpu_memory_policy": memory_policy,
                "hmc_tuning_launched": False,
                "hmc_sampling_launched": False,
                "elapsed_seconds": time.monotonic() - started,
                "output_root": str(root),
                "nonclaims": (
                    "candidate rejection is not a GPU or throughput-harness failure",
                    "no recipe-family, HMC, posterior, or default-readiness conclusion",
                ),
            }
            atomic_write_json(root / "failure.json", failure)
            _write_training_throughput_manifest(
                root=root,
                spec=spec,
                recipe=recipe,
                repeated_steps=int(repeated_steps),
                memory_policy=memory_policy,
                allocator={},
                timing=None,
                started=started,
                passed=False,
                decision=failure["decision"],
                result_path=root / "failure.json",
            )
            return failure
        rung_rows.append(
            {
                "rung_id": rung_id,
                "steps": steps,
                "seed": spec.initial_seed,
                "training_program_seconds": training[
                    "training_program_elapsed_seconds"
                ],
                "end_to_end_seconds": time.monotonic() - rung_started,
                "heldout_reverse_kl": training["mean_reverse_kl"],
                "candidate_passed": training["passed"],
                "hard_vetoes": training["hard_vetoes"],
                "heldout_target_status": training["heldout_target_status"],
                "frozen_trainable_parity": training[
                    "frozen_trainable_parity"
                ],
                "runtime_metadata": training["runtime_metadata"],
                "artifacts": {
                    "frozen_transport": training["payload"]["path"],
                    "training_state": training["state_path"],
                },
            }
        )

    timing = _throughput_timing_summary(rung_rows)
    allocator = {
        key: int(value)
        for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
    }
    candidate_passed = all(row["candidate_passed"] is True for row in rung_rows)
    result = {
        "schema": "bayesfilter.neutra.training_throughput_result.v1",
        "cell_id": spec.cell_id,
        "passed": True,
        "decision": (
            "PASS_TRAINING_THROUGHPUT_MEASUREMENT"
            if candidate_passed
            else "PASS_THROUGHPUT_MEASUREMENT_WITH_CANDIDATE_HARD_VETO"
        ),
        "target_signature": spec.target_signature,
        "recipe_id": recipe_id,
        "batch_size": TRAINING_BATCH_SIZE,
        "jit_compile": True,
        "device": "/GPU:0",
        "geometry_reference": geometry_reference,
        "rung_rows": tuple(rung_rows),
        "candidate_passed": candidate_passed,
        "timing": timing,
        "gpu_allocator_bytes": allocator,
        "gpu_memory_policy": memory_policy,
        "hmc_tuning_launched": False,
        "hmc_sampling_launched": False,
        "elapsed_seconds": time.monotonic() - started,
        "output_root": str(root),
        "nonclaims": (
            "throughput and engineering hard gates only",
            "training loss is descriptive and not recipe promotion evidence",
            "no HMC tuning, convergence, posterior, or default-readiness claim",
        ),
    }
    atomic_write_json(root / "result.json", result)
    _write_training_throughput_manifest(
        root=root,
        spec=spec,
        recipe=recipe,
        repeated_steps=int(repeated_steps),
        memory_policy=memory_policy,
        allocator=allocator,
        timing=timing,
        started=started,
        passed=True,
        decision=result["decision"],
        result_path=root / "result.json",
    )
    return result


def _throughput_timing_summary(
    rung_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(rung_rows) != 2:
        raise ValueError("throughput timing requires exactly two rungs")
    first, repeated = rung_rows
    repeated_steps = int(repeated["steps"])
    if int(first["steps"]) != 1 or repeated_steps < 2:
        raise ValueError("invalid throughput timing rung step counts")
    first_program = float(first["training_program_seconds"])
    repeated_program = float(repeated["training_program_seconds"])
    first_wall = float(first["end_to_end_seconds"])
    repeated_wall = float(repeated["end_to_end_seconds"])
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (first_program, repeated_program, first_wall, repeated_wall)
    ):
        raise ValueError("throughput timing values must be positive and finite")
    program_slope = max(0.0, (repeated_program - first_program) / (repeated_steps - 1))
    wall_slope = max(0.0, (repeated_wall - first_wall) / (repeated_steps - 1))
    # Fixed compile/scoring/parity cost is already charged once per projected
    # job. Retaining repeated program average is conservative without counting
    # that fixed end-to-end overhead again for every additional step.
    conservative_step = max(
        repeated_program / repeated_steps,
        program_slope,
        wall_slope,
    )
    return {
        "method": "one_step_plus_repeated_graph_difference_v1",
        "compile_plus_first_program_seconds": first_program,
        "compile_plus_first_end_to_end_seconds": first_wall,
        "repeated_program_seconds": repeated_program,
        "repeated_end_to_end_seconds": repeated_wall,
        "repeated_steps": repeated_steps,
        "program_difference_step_seconds": program_slope,
        "end_to_end_difference_step_seconds": wall_slope,
        "conservative_additional_step_seconds": conservative_step,
        "projection_formula": (
            "jobs * compile_plus_first_end_to_end_seconds + "
            "sum(max(steps_per_job - 1, 0)) * "
            "conservative_additional_step_seconds"
        ),
    }


def _preflight_bootstrap_mechanics_passed(
    tuning_payload: Mapping[str, Any],
) -> bool:
    if tuning_payload.get("schema") != "bayesfilter.hmc_kernel_tuning_public_artifact.v1":
        return False
    bootstrap = tuning_payload.get("bootstrap_public_summary")
    return bool(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("preflight_passed") is True
        and bootstrap.get("round_timing_available") is True
        and int(bootstrap.get("round_count", 0)) > 0
        and math.isfinite(float(bootstrap.get("max_round_runtime_s", float("nan"))))
        and bool(bootstrap.get("observed_acceptance_relations"))
    )


def _load_preserved_screen_selection(
    *, spec: CellSpec, config: EndToEndConfig
) -> Mapping[str, Any]:
    path = config.screen_result_path
    expected_sha256 = config.expected_screen_result_sha256
    if path is None or expected_sha256 is None or not path.is_file():
        raise NeuTraEndToEndError("preserved screen result is unavailable")
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise NeuTraEndToEndError("preserved screen result SHA-256 mismatch")
    result = _read_mapping(path)
    if (
        result.get("cell_id") != spec.cell_id
        or result.get("target_signature") != spec.target_signature
        or result.get("passed") is not True
        or result.get("decision")
        != "SCREEN_COMPLETED_REPRESENTATIVE_NOMINATED"
        or result.get("final_training_launched") is not False
    ):
        raise NeuTraEndToEndError("preserved screen result contract mismatch")
    selection = result.get("selection")
    if not isinstance(selection, Mapping) or selection.get("passed") is not True:
        raise NeuTraEndToEndError("preserved screen lacks a passing selection")
    rows = selection.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise NeuTraEndToEndError("preserved screen recipe rows are malformed")
    expected_recipe_ids = {recipe.recipe_id for recipe in spec.recipes}
    observed_recipe_ids = {str(row.get("recipe_id")) for row in rows}
    if observed_recipe_ids != expected_recipe_ids:
        raise NeuTraEndToEndError("preserved screen recipe inventory mismatch")
    for row in rows:
        replicates = row.get("replicate_rows")
        if row.get("passed") is not True or not isinstance(replicates, Sequence):
            raise NeuTraEndToEndError("preserved screen contains a failed recipe")
        for replicate in replicates:
            runtime = replicate.get("runtime_metadata", {})
            if (
                replicate.get("passed") is not True
                or int(runtime.get("program_step_count", -1)) != config.screen_steps
                or int(runtime.get("training_batch_size", -1)) != TRAINING_BATCH_SIZE
                or runtime.get("jit_compile") is not True
                or runtime.get("sample_axis_python_loop_used") is not False
                or runtime.get("scalar_fallback_used") is not False
                or runtime.get("row_mapped_scalar_target_used") is not False
            ):
                raise NeuTraEndToEndError(
                    "preserved screen recipe runtime contract mismatch"
                )
    selected_recipe_id = str(selection.get("selected_recipe_id"))
    if selected_recipe_id not in expected_recipe_ids:
        raise NeuTraEndToEndError("preserved screen selected recipe is unknown")
    return selection


def _screen_recipes(*, spec: CellSpec, adapter: Any, bound_adapter: Any,
                    center: Any, factor: Any, root: Path,
                    config: EndToEndConfig,
                    memory_policy: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = []
    screen_seeds = tuple(spec.screen_seeds or (spec.initial_seed,))
    for recipe in spec.recipes:
        recipe_root = root / "screen" / recipe.recipe_id
        replicate_rows = []
        for replicate_index, seed in enumerate(screen_seeds):
            output_root = (
                recipe_root
                if len(screen_seeds) == 1
                else recipe_root / f"replicate-{replicate_index + 1:02d}"
            )
            try:
                replicate = _train_and_score(
                    spec=spec,
                    adapter=adapter,
                    bound_adapter=bound_adapter,
                    center=center,
                    factor=factor,
                    recipe=recipe,
                    output_root=output_root,
                    steps=config.screen_steps,
                    seed=seed,
                    memory_policy=memory_policy,
                )
            except NeuTraTrainingError as exc:
                replicate = {
                    "recipe_id": recipe.recipe_id,
                    "passed": False,
                    "decision": "TRAINING_CANDIDATE_HARD_VETO",
                    "hard_vetoes": (str(exc),),
                    "training_seed": seed,
                    "mean_reverse_kl": None,
                    "affine_nonworse": False,
                    "output_root": str(output_root),
                }
                atomic_write_json(
                    output_root / "candidate_failure.json",
                    {
                        "schema": "bayesfilter.neutra.screen_candidate_failure.v1",
                        **replicate,
                    },
                )
            replicate_rows.append(replicate)
        recipe_passed = all(row["passed"] is True for row in replicate_rows)
        replication_means = (
            tuple(float(row["mean_reverse_kl"]) for row in replicate_rows)
            if recipe_passed
            else ()
        )
        rows.append(
            {
                "recipe_id": recipe.recipe_id,
                "passed": recipe_passed,
                "mean_reverse_kl": (
                    _mean(replication_means) if recipe_passed else None
                ),
                "reverse_kl_mcse": (
                    _mcse(replication_means) if recipe_passed else None
                ),
                "reverse_kl_replication_means": replication_means,
                "screen_seeds": screen_seeds,
                "replicate_rows": tuple(replicate_rows),
                "affine_nonworse": all(
                    row["affine_nonworse"] is True for row in replicate_rows
                ),
                "statistical_unit": "independent_training_seed",
            }
        )
    survivors = [row for row in rows if row["passed"]]
    if not survivors:
        selection = {
            "passed": False,
            "selected_recipe_id": None,
            "rows": rows,
            "decision": "NO_SURVIVING_RECIPE",
            "selection_role": "hard_gate_rejection_only",
        }
        atomic_write_json(root / "screen" / "selection.json", selection)
        return selection
    means = {row["recipe_id"]: float(row["mean_reverse_kl"]) for row in survivors}
    nominal = min(means, key=means.get)
    viable = []
    nominal_row = next(item for item in survivors if item["recipe_id"] == nominal)
    for row in survivors:
        if len(screen_seeds) == 1:
            # One stochastic fit cannot support a recipe ranking.
            viable.append(row["recipe_id"])
            continue
        differences = tuple(
            left - right
            for left, right in zip(
                row["reverse_kl_replication_means"],
                nominal_row["reverse_kl_replication_means"],
                strict=True,
            )
        )
        if _mean(differences) <= float(spec.selection_mcse_multiplier) * _mcse(
            differences
        ):
            viable.append(row["recipe_id"])
    if spec.require_affine_nonworse:
        viable = [item for item in viable if next(row for row in survivors if row["recipe_id"] == item)["affine_nonworse"]]
    if not viable:
        selection = {
            "passed": False,
            "selected_recipe_id": None,
            "nominal_lowest_mean_recipe": nominal,
            "rows": rows,
            "decision": "NO_RECIPE_PASSED_TARGET_SPECIFIC_PROXY_VETO",
            "selection_role": "proxy_nomination_or_veto_only",
        }
        atomic_write_json(root / "screen" / "selection.json", selection)
        return selection
    selected = spec.preferred_recipe_id if spec.preferred_recipe_id in viable else min(
        viable,
        key=lambda item: (
            _parameter_count(
                spec.parameter_dim, _recipe_by_id(spec.recipes, item)
            ),
            item,
        ),
    )
    selection = {
        "passed": True, "selected_recipe_id": selected, "nominal_lowest_mean_recipe": nominal,
        "rows": rows, "selection_role": "proxy_nomination_only",
        "statistically_supported_ranking": False,
        "statistical_unit": "independent_training_seed",
        "screen_seeds_paired_across_recipes": True,
    }
    atomic_write_json(root / "screen" / "selection.json", selection)
    return selection


def _train_and_score(*, spec: CellSpec, adapter: Any, bound_adapter: Any,
                     center: Any, factor: Any, recipe: RecipeSpec,
                     output_root: Path, steps: int, seed: tuple[int, int],
                     memory_policy: Mapping[str, Any]) -> Mapping[str, Any]:
    if output_root.exists():
        raise NeuTraEndToEndError(f"training output must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    training_config = _training_config(
        spec=spec,
        center=center,
        factor=factor,
        recipe=recipe,
        output_dir=output_root / "training",
        steps=steps,
        seed=seed,
    )
    _require_memory_growth(memory_policy)
    training = train_plain_dense_iaf(
        adapter=adapter,
        config=training_config,
        freeze_transport_id=f"e2e-{spec.cell_id.lower()}-{recipe.recipe_id}-{steps}",
    )
    return _score_training(
        spec=spec,
        bound_adapter=bound_adapter,
        center=center,
        factor=factor,
        recipe=recipe,
        training_config=training_config,
        training=training,
        seed=seed,
    )


def _training_config(*, spec: CellSpec, center: Any, factor: Any,
                     recipe: RecipeSpec, output_dir: Path, steps: int,
                     seed: tuple[int, int]) -> PlainDenseIAFTrainingConfig:
    return PlainDenseIAFTrainingConfig(
        target_signature=spec.target_signature, dimension=spec.parameter_dim,
        affine_center=center, affine_factor=factor,
        output_dir=output_dir, seed=seed,
        hidden_layers=recipe.hidden_layers, stage_count=recipe.stage_count,
        activation="elu", s_max=1.0,
        init_scale=0.02, steps=steps, batch_size=TRAINING_BATCH_SIZE,
        learning_rate=recipe.learning_rate, final_learning_rate_fraction=recipe.final_learning_rate_fraction,
        clip_norm=10.0, checkpoint_every=steps, heartbeat_every=max(1, min(10, steps)),
        jit_compile=True, device="/GPU:0", require_gpu=True,
    )


def _score_training(*, spec: CellSpec, bound_adapter: Any, center: Any,
                    factor: Any, recipe: RecipeSpec,
                    training_config: PlainDenseIAFTrainingConfig,
                    training: Any, seed: tuple[int, int]) -> Mapping[str, Any]:
    if training.frozen_payload_path is None:
        raise NeuTraEndToEndError("training did not emit frozen transport")
    loaded = load_frozen_neutra_artifact(_read_mapping(training.frozen_payload_path), expected_target_signature=spec.target_signature)
    heldout_seed = (int(seed[0]), int(seed[1]) + 500)
    values = tf.random.stateless_normal(
        (8, TRAINING_BATCH_SIZE, spec.parameter_dim),
        seed=heldout_seed,
        dtype=tf.float64,
    )
    transformed = FixedTransportValueScoreAdapter(base_adapter=bound_adapter, transport=loaded.transport, target_scope=f"{spec.cell_id}:heldout", evidence_path=__file__, xla_hmc_ready=True, full_chain_xla_diagnostic_ready=True, require_batch_native=True)
    @tf.function(jit_compile=True, reduce_retracing=True)
    def heldout_program(z_batch: tf.Tensor):
        value, _score, status = transformed.log_prob_and_grad_status(z_batch)
        return (
            -value,
            tf.convert_to_tensor(status["status_code"], tf.int32),
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
            tf.convert_to_tensor(status["floor_count_value"], tf.int32),
            tf.convert_to_tensor(status["min_innovation_eigenvalue"], tf.float64),
        )

    heldout_rows = []
    with tf.device("/GPU:0"):
        # Preserve the eight independent diagnostic batches.  Flattening them
        # into one 1,024-row target call is unnecessary and exceeds memory for
        # expensive batch-native targets such as SVX-ZC.
        for heldout_batch in tf.unstack(values, axis=0):
            heldout_rows.append(heldout_program(heldout_batch))
    objective, status_code, status_valid, floor_count, min_eigenvalue = tuple(
        tf.stack(items, axis=0) for items in zip(*heldout_rows, strict=True)
    )
    invalid = tf.logical_or(tf.not_equal(status_code, 0), tf.logical_not(status_valid))
    invalid_count = int(tf.reduce_sum(tf.cast(invalid, tf.int32)).numpy())
    heldout_status = {
        "all_valid": invalid_count == 0,
        "row_count": 8 * TRAINING_BATCH_SIZE,
        "invalid_row_count": invalid_count,
        "nonzero_status_code_count": int(
            tf.reduce_sum(tf.cast(tf.not_equal(status_code, 0), tf.int32)).numpy()
        ),
        "invalid_pre_regularized_score_count": int(
            tf.reduce_sum(tf.cast(tf.logical_not(status_valid), tf.int32)).numpy()
        ),
        "floor_count_total": int(tf.reduce_sum(floor_count).numpy()),
        "minimum_innovation_eigenvalue": float(
            tf.reduce_min(min_eigenvalue).numpy()
        ),
    }
    batches = objective
    means = tuple(float(item) for item in tf.reduce_mean(batches, axis=1).numpy().tolist())
    affine_means = _affine_reverse_kl_batches(bound_adapter, center, factor, values)
    affine_differences = tuple(left - right for left, right in zip(means, affine_means, strict=True))
    flow = restore_plain_dense_iaf_flow(config=training_config, state_path=training.state_path)
    parity = _frozen_trainable_parity(flow, loaded, spec, seed)
    program_elapsed = float(training.records[-1]["program_elapsed_seconds"])
    if not math.isfinite(program_elapsed) or program_elapsed <= 0.0:
        raise NeuTraEndToEndError("training program timing is unavailable")
    hard_vetoes = (
        ()
        if heldout_status["all_valid"]
        else ("heldout_target_status_nonvalid",)
    )
    return {"recipe_id": recipe.recipe_id, "passed": not hard_vetoes,
            "decision": ("PASS_TRAINING_HARD_GATES" if not hard_vetoes else "TRAINING_CANDIDATE_HARD_VETO"),
            "hard_vetoes": hard_vetoes,
            "mean_reverse_kl": _mean(means), "reverse_kl_mcse": _mcse(means),
            "reverse_kl_batch_means": means, "affine_reverse_kl_batch_means": affine_means,
            "affine_nonworse": bool(_mean(affine_differences) <= 2.0 * _mcse(affine_differences)),
            "training_state_hash": training.state_hash,
            "payload": {"path": str(training.frozen_payload_path)},
            "state_path": str(training.state_path),
            "training_seed": seed, "heldout_seed": heldout_seed,
            "heldout_target_status": heldout_status,
            "frozen_trainable_parity": parity,
            "training_program_elapsed_seconds": program_elapsed,
            "training_program_step_count": int(
                training.runtime_metadata["program_step_count"]
            ),
            "runtime_metadata": training.runtime_metadata}


def _train_final(*, spec: CellSpec, adapter: Any, bound_adapter: Any,
                 center: Any, factor: Any, recipe: RecipeSpec, root: Path,
                 config: EndToEndConfig,
                 memory_policy: Mapping[str, Any]) -> Mapping[str, Any]:
    output_root = root / "final"
    if output_root.exists():
        raise NeuTraEndToEndError(f"final training output must be fresh: {output_root}")
    seed = (spec.initial_seed[0] + 1, spec.initial_seed[1])
    training_config = _training_config(
        spec=spec,
        center=center,
        factor=factor,
        recipe=recipe,
        output_dir=output_root / "segments",
        steps=config.final_steps,
        seed=seed,
    )
    _require_memory_growth(memory_policy)
    segmented = train_plain_dense_iaf_infrastructure_segments(
        adapter=adapter,
        config=training_config,
        segment_steps=min(config.final_segment_steps, config.final_steps),
        freeze_transport_id=(
            f"e2e-{spec.cell_id.lower()}-{recipe.recipe_id}-{config.final_steps}"
        ),
    )
    result = dict(
        _score_training(
            spec=spec,
            bound_adapter=bound_adapter,
            center=center,
            factor=factor,
            recipe=recipe,
            training_config=training_config,
            training=segmented.final_result,
            seed=seed,
        )
    )
    result["steps"] = config.final_steps
    result["segmented_training"] = {
        "segment_steps": min(config.final_segment_steps, config.final_steps),
        "segment_rows": segmented.segment_rows,
        "progress_path": str(segmented.progress_path),
        "result_path": str(segmented.result_path),
        "terminal_only_freeze": True,
    }
    return result


def _native_tune(*, spec: CellSpec, adapter: Any, loaded: Any, root: Path, config: EndToEndConfig) -> Any:
    return tune_hmc_kernel(
        adapter=_fixed_transport_adapter(
            adapter, loaded.transport, f"{spec.cell_id}:fixed_neutra_native_tuning"
        ),
        initial_position=tf.zeros((spec.parameter_dim,), tf.float64),
        config=HMCKernelTuningConfig.serious(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            mass_policy="fixed_identity",
            chain_execution_mode="tf_function",
            use_xla=True,
            target_scope=f"{spec.cell_id}:fixed_neutra_native_tuning",
            target_status_trace_policy="per_chain_step",
            source="bayesfilter.neutra_all_models_end_to_end.public_tuner",
        ),
        output_dir=root / "tuning",
    )


def _assert_public_tuning_contract(result: Any, replay: Any) -> None:
    if result.config.mass_policy != "fixed_identity":
        raise NeuTraEndToEndError("public tuner mass policy is not fixed identity")
    if result.config.target_accept_prob != 0.70 or result.config.acceptance_band != (0.65, 0.75):
        raise NeuTraEndToEndError("public tuner acceptance policy drifted")
    if replay.final_kernel_payload.get("mass_policy") != "fixed_identity":
        raise NeuTraEndToEndError("replay handoff mass policy drifted")


def _fixed_transport_adapter(
    base_adapter: Any, transport: Any, target_scope: str
) -> FixedTransportValueScoreAdapter:
    return FixedTransportValueScoreAdapter(
        base_adapter=base_adapter,
        transport=transport,
        target_scope=target_scope,
        evidence_path=__file__,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )


def _sequential_config(
    replay: Any,
    spec: CellSpec,
    *,
    seed_offset: int = 0,
) -> Any:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig

    return SequentialNeuTraHMCConfig(
        step_size=float(replay.step_size), num_leapfrog_steps=int(replay.num_leapfrog_steps),
        warmup_seed=(20260718, spec.initial_seed[1] + 100 + int(seed_offset)),
        retained_seed=(20260718, spec.initial_seed[1] + 101 + int(seed_offset)),
        warmup_chunk_results=1000, warmup_min_results=2000, warmup_check_window_results=1000,
        warmup_max_results=10000, warmup_rhat_max=1.05,
        retained_chunk_results=1000, retained_min_results=1000, retained_max_results=10000,
        retained_rhat_max=1.01, minimum_chain_count=4, jit_compile=True,
    )


def _sequential_config_from_pair(
    *,
    step_size: float,
    num_leapfrog_steps: int,
    spec: CellSpec,
    chunk_results: int,
) -> Any:
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig

    return SequentialNeuTraHMCConfig(
        step_size=float(step_size),
        num_leapfrog_steps=int(num_leapfrog_steps),
        warmup_seed=(20260802, spec.initial_seed[1] + 200),
        retained_seed=(20260802, spec.initial_seed[1] + 201),
        warmup_chunk_results=int(chunk_results),
        warmup_min_results=2000,
        warmup_check_window_results=1000,
        warmup_max_results=10000,
        warmup_rhat_max=1.05,
        retained_chunk_results=int(chunk_results),
        retained_min_results=1000,
        retained_max_results=10000,
        retained_rhat_max=1.01,
        minimum_chain_count=4,
        jit_compile=True,
    )


def _initial_state(tf_module: Any, spec: CellSpec, seed: tuple[int, int]) -> Any:
    del seed
    offsets = tf_module.constant((0.0, 0.1, -0.1, 0.16), tf_module.float64)[:, None]
    return tf_module.broadcast_to(offsets, (4, spec.parameter_dim))


def _target_signature(adapter: Any) -> str:
    signature = getattr(adapter, "target_signature", None)
    if signature is not None:
        return str(signature)
    contract = getattr(adapter, "contract", None)
    if contract is None:
        raise NeuTraEndToEndError("adapter exposes neither target signature nor contract")
    from bayesfilter.ssm import stable_ssm_target_signature

    return stable_ssm_target_signature(contract)


def _require_memory_growth(memory_policy: Mapping[str, Any]) -> None:
    required = {
        "schema": "bayesfilter.tensorflow.gpu_memory_policy.v1",
        "mode": "memory_growth",
        "all_physical_devices_memory_growth": True,
        "configured_before_logical_device_initialization": True,
    }
    for key, expected in required.items():
        if memory_policy.get(key) != expected:
            raise NeuTraEndToEndError(f"GPU memory-growth contract failed: {key}")


def _affine_reverse_kl_batches(
    adapter: Any, center: Any, factor: Any, base_values: tf.Tensor
) -> tuple[float, ...]:
    center_tensor = tf.convert_to_tensor(center, tf.float64)
    factor_tensor = tf.convert_to_tensor(factor, tf.float64)
    _sign, logdet = tf.linalg.slogdet(factor_tensor)
    batch_means = []
    for base_batch in tf.unstack(base_values, axis=0):
        raw = center_tensor + tf.matmul(
            base_batch, factor_tensor, transpose_b=True
        )
        value, _score = adapter.log_prob_and_grad(raw)
        objective = -(tf.convert_to_tensor(value, tf.float64) + logdet)
        batch_means.append(tf.reduce_mean(objective))
    return tuple(float(item) for item in tf.stack(batch_means).numpy().tolist())


def _frozen_trainable_parity(
    flow: Any, loaded: Any, spec: CellSpec, seed: tuple[int, int]
) -> Mapping[str, Any]:
    probes = tf.random.stateless_normal(
        (2, spec.parameter_dim), seed=(seed[0], seed[1] + 700), dtype=tf.float64
    )
    theta_score = tf.random.stateless_normal(
        (2, spec.parameter_dim), seed=(seed[0], seed[1] + 701), dtype=tf.float64
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg: tf.Tensor, score_arg: tf.Tensor):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_arg)
            train_theta, train_logdet = flow.forward_and_logdet(z_arg)
            theta_objective = tf.reduce_sum(train_theta * score_arg)
            logdet_objective = tf.reduce_sum(train_logdet)
        train_pullback = tape.gradient(theta_objective, z_arg)
        train_logdet_score = tape.gradient(logdet_objective, z_arg)
        frozen_theta = loaded.transport.forward_batch(z_arg)
        frozen_logdet = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        frozen_pullback = loaded.transport.pullback_score_batch(z_arg, score_arg)
        frozen_logdet_score = loaded.transport.log_abs_det_jacobian_score_batch(z_arg)
        return (
            tf.reduce_max(tf.abs(train_theta - frozen_theta)),
            tf.reduce_max(tf.abs(train_logdet - frozen_logdet)),
            tf.reduce_max(tf.abs(train_pullback - frozen_pullback)),
            tf.reduce_max(tf.abs(train_logdet_score - frozen_logdet_score)),
        )

    with tf.device("/GPU:0"):
        gaps = compiled(probes, theta_score)
    numeric = tuple(float(item.numpy()) for item in gaps)
    passed = all(value <= 1.0e-10 for value in numeric)
    if not passed:
        raise NeuTraEndToEndError(f"frozen/trainable parity failed: {spec.cell_id}")
    return {
        "passed": True,
        "transport_max_abs": numeric[0],
        "logdet_max_abs": numeric[1],
        "pullback_score_max_abs": numeric[2],
        "logdet_score_max_abs": numeric[3],
        "jit_compile": True,
    }


def _mean(values: Sequence[float]) -> float:
    numeric = tuple(float(item) for item in values)
    if not numeric:
        raise NeuTraEndToEndError("mean requires at least one value")
    return math.fsum(numeric) / len(numeric)


def _mcse(values: Sequence[float]) -> float:
    numeric = tuple(float(item) for item in values)
    if len(numeric) <= 1:
        return 0.0
    mean = _mean(numeric)
    return math.sqrt(
        math.fsum((item - mean) ** 2 for item in numeric)
        / ((len(numeric) - 1) * len(numeric))
    )


def _truth_tail(spec: CellSpec, samples: Any) -> Mapping[str, Any]:
    truth = spec.truth_factory(tf)
    physical = tf.convert_to_tensor(samples, tf.float64)
    pooled = tf.reshape(physical, (-1, spec.parameter_dim))
    less = tf.reduce_sum(tf.cast(pooled < truth[None, :], tf.float64), axis=0)
    equal = tf.reduce_sum(tf.cast(pooled == truth[None, :], tf.float64), axis=0)
    count = tf.cast(tf.shape(pooled)[0], tf.float64)
    cdf = (less + 0.5 * equal + 0.5) / (count + 1.0)
    p_values = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    rows = tuple({"parameter": name, "truth": float(truth[i].numpy()), "p_truth": float(p_values[i].numpy()), "status": _tail_status(float(p_values[i].numpy()))} for i, name in enumerate(spec.parameter_names))
    minimum = min(row["p_truth"] for row in rows)
    return {"parameter_rows": rows, "minimum_p_truth": minimum, "status": "PASS" if minimum >= PASS_THRESHOLD else ("MARGINAL_RERUN" if minimum >= SEVERE_THRESHOLD else "SEVERE_FAILURE")}


def _tail_status(value: float) -> str:
    return "PASS" if value >= PASS_THRESHOLD else ("MARGINAL_RERUN" if value >= SEVERE_THRESHOLD else "SEVERE_FAILURE")


def _decision(sequential: Mapping[str, Any], truth_tail: Mapping[str, Any]) -> str:
    if sequential.get("passed") is not True:
        return "HMC_CONVERGENCE_OR_HEALTH_FAILURE"
    if truth_tail["status"] == "PASS":
        return "PASS_ONE_SEED_TRUTH_TAIL"
    return str(truth_tail["status"])


def _admitted_kernel_execution(config: FrozenTransportValidationConfig) -> Mapping[str, Any]:
    return {
        "dtype": "float64",
        "backend": "tensorflow_probability",
        "jit_compile": bool(config.jit_compile),
        "tf32_execution_enabled": True,
        "mass_policy": "fixed_identity",
    }


def _base_result(spec: CellSpec, root: Path, payload: Mapping[str, Any], memory_policy: Mapping[str, Any], started: float) -> Mapping[str, Any]:
    return {"schema": "bayesfilter.neutra.all_models.cell_result.v1", "cell_id": spec.cell_id, "target_signature": spec.target_signature, "parameter_names": spec.parameter_names, "output_root": str(root), "gpu_memory_policy": memory_policy, "elapsed_seconds": time.monotonic() - started, **dict(payload)}


def _write_manifest(root: Path, spec: CellSpec, config: EndToEndConfig, memory_policy: Mapping[str, Any], started: float) -> None:
    commit = subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
    terminal = (root / "result.json").is_file()
    atomic_write_json(root / "run_manifest.json", {"schema": "bayesfilter.neutra.all_models.run_manifest.v1", "status": "terminal" if terminal else "started", "cell_id": spec.cell_id, "target_signature": spec.target_signature, "data_version": "bound_by_target_signature", "git_commit": commit, "command": tuple(sys.argv), "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"), "python_executable": sys.executable, "python_version": platform.python_version(), "tensorflow_version": tf.__version__, "gpu_memory_policy": memory_policy, "device": "/GPU:0" if config.require_gpu else "explicit_cpu_exception", "training_batch_size": TRAINING_BATCH_SIZE, "jit_compile": config.jit_compile, "tf32_execution_enabled": True, "screen_only": config.screen_only, "screen_steps": config.screen_steps, "final_steps": config.final_steps, "final_segment_steps": config.final_segment_steps, "output_root": str(root), "wall_time_seconds": time.monotonic() - started, "plan_path": spec.plan_path, "result_file": str(root / "result.json"), "screen_seeds": spec.screen_seeds or (spec.initial_seed,), "random_seed_policy": "paired_common_training_seeds_across_recipes", "nonclaims": ("screen ranking is not statistically supported", "no sampler superiority or default-readiness claim")})


def _write_training_throughput_manifest(
    *,
    root: Path,
    spec: CellSpec,
    recipe: RecipeSpec,
    repeated_steps: int,
    memory_policy: Mapping[str, Any],
    allocator: Mapping[str, int],
    timing: Mapping[str, Any] | None,
    started: float,
    passed: bool,
    decision: str,
    result_path: Path,
) -> None:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    atomic_write_json(
        root / "run_manifest.json",
        {
            "schema": "bayesfilter.neutra.training_throughput_manifest.v1",
            "cell_id": spec.cell_id,
            "target_signature": spec.target_signature,
            "passed": bool(passed),
            "decision": str(decision),
            "data_version": "bound_by_target_signature",
            "git_commit": commit,
            "command": tuple(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "recipe": recipe.payload(),
            "training_seed": spec.initial_seed,
            "batch_size": TRAINING_BATCH_SIZE,
            "rung_steps": (1, repeated_steps),
            "device": "/GPU:0",
            "gpu_memory_policy": memory_policy,
            "gpu_allocator_bytes": allocator,
            "jit_compile": True,
            "tf32_execution_enabled": True,
            "scalar_fallback_used": False,
            "sample_axis_python_loop_used": False,
            "timing": timing,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(root),
            "artifact_paths": {
                "result": str(result_path),
                "manifest": str(root / "run_manifest.json"),
            },
            "plan_path": spec.plan_path,
            "result_file": str(result_path),
            "hmc_tuning_launched": False,
            "hmc_sampling_launched": False,
            "nonclaims": (
                "throughput and engineering hard gates only",
                "no recipe ranking, HMC, posterior, or default-readiness claim",
            ),
        },
    )


def _write_frozen_validation_manifest(
    *,
    root: Path,
    spec: CellSpec,
    config: FrozenTransportValidationConfig,
    memory_policy: Mapping[str, Any],
    started: float,
    transport_input: Mapping[str, Any],
    tuning_seed: tuple[int, int] | None,
    sequential_config: Any | None,
    admitted_kernel_replay: Mapping[str, Any] | None,
) -> None:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    atomic_write_json(
        root / "run_manifest.json",
        {
            "schema": "bayesfilter.neutra.frozen_transport_validation_manifest.v1",
            "cell_id": spec.cell_id,
            "target_signature": spec.target_signature,
            "git_commit": commit,
            "command": tuple(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu_memory_policy": memory_policy,
            "jit_compile": config.jit_compile,
            "tuning_only": config.tuning_only,
            "tf32_execution_enabled": True,
            "output_root": str(root),
            "wall_time_seconds": time.monotonic() - started,
            "frozen_transport_input": dict(transport_input),
            "admitted_kernel_replay_path": (
                None
                if config.admitted_kernel_replay_path is None
                else str(config.admitted_kernel_replay_path)
            ),
            "admitted_kernel_replay": (
                None
                if admitted_kernel_replay is None
                else dict(admitted_kernel_replay)
            ),
            "random_seeds": {
                "public_tuner": (
                    None
                    if tuning_seed is None
                    else tuple(int(item) for item in tuning_seed)
                ),
                "sequential_warmup": (
                    None
                    if sequential_config is None
                    else sequential_config.warmup_seed
                ),
                "sequential_retained": (
                    None
                    if sequential_config is None
                    else sequential_config.retained_seed
                ),
            },
            "plan_path": (
                "docs/plans/bayesfilter-public-tuner-fixed-identity-mass-plan-"
                "2026-07-19.md"
            ),
            "result_path": str(root / "result.json"),
            "nonclaims": (
                "one-seed diagnostic only",
                "no sampler superiority or default-readiness claim",
                "no sequential HMC sampling claim"
                if config.tuning_only
                else "sequential sampling may be executed after tuning",
            ),
        },
    )


def _validate_geometry(center: Any, factor: Any, dimension: int) -> None:
    center = tf.convert_to_tensor(center, tf.float64)
    factor = tf.convert_to_tensor(factor, tf.float64)
    if center.shape != (dimension,) or factor.shape != (dimension, dimension):
        raise NeuTraEndToEndError("affine geometry shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(center)).numpy() and tf.reduce_all(tf.math.is_finite(factor)).numpy()):
        raise NeuTraEndToEndError("affine geometry is nonfinite")
    if float(tf.abs(tf.linalg.det(factor)).numpy()) <= 0.0:
        raise NeuTraEndToEndError("affine geometry is singular")


def _parameter_count(dimension: int, recipe: RecipeSpec) -> int:
    sizes = (int(dimension), *recipe.hidden_layers, 2 * int(dimension))
    per_stage = sum(
        left * right + right for left, right in zip(sizes[:-1], sizes[1:])
    )
    return int(recipe.stage_count) * per_stage


def _recipe_by_id(recipes: Sequence[RecipeSpec], recipe_id: str) -> RecipeSpec:
    for recipe in recipes:
        if recipe.recipe_id == recipe_id:
            return recipe
    raise NeuTraEndToEndError(f"unknown recipe: {recipe_id}")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise NeuTraEndToEndError(f"expected JSON mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy().tolist())
    return value
