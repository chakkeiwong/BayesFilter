"""NumPy-free, graph-native target-specific LGSSM NeuTra training harness."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/lgssm_neutra_target_specific_protocol_2026_07_14"
)
CONTRACT_PATH = DEFAULT_ARTIFACT_ROOT / "campaign_contract.json"
MASS_PATH = ROOT / (
    "docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/mass.json"
)
EXPECTED_CONTRACT_FILE_SHA256 = (
    "6815261d1c66fe85160b2aa76b5bd97bfc9872efe21da3da31e94b36d539e2d9"
)
EXPECTED_MASS_FILE_SHA256 = (
    "54549c9156821536bc4780f0406a7716b0d3fa39a5b5900fa2893cbef2968a95"
)
EXPECTED_MASS_ARTIFACT_HASH = (
    "sha256:2e41adfdebb47e9b949a675671c12ad1261588d6932c27c3c795724abaa355ad"
)
EXPECTED_TARGET_SIGNATURE = (
    "f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30"
)
EXPECTED_ADAPTER_SIGNATURE = (
    "42dc7bad0137fd9c31aa1d618bb4e560f68d1bbe3a7ab4f5ef95e458b2abc985"
)
DIMENSION = 18
BATCH_SIZE = 128
HEARTBEAT_EVERY = 10
SMOKE_STEPS = 5
SCREEN_STEPS = 500
FINAL_STEPS = 5000
SCREEN_SEED = (20260714, 1401)
FINAL_SEEDS = {
    "dense_seed1201": (20260713, 1201),
    "dense_seed1202": (20260713, 1202),
    "dense_seed1203": (20260715, 1203),
}
HELDOUT_SEEDS = tuple((20260714, 1501 + index) for index in range(8))
SOURCE_ANCHOR = "source_anchor_lr5e3"
SCREEN_RECIPE_ORDER = (
    SOURCE_ANCHOR,
    "lower_lr1e3",
    "shallow_2stage_lr5e3",
    "wide_2x_lr5e3",
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"

NONCLAIMS = (
    "screen held-out reverse KL is nomination-only",
    "500-step behavior is not serious training evidence",
    "one favorably truth-centered 18D LGSSM fixture only",
    "no posterior correctness, HMC convergence, or method superiority claim",
)


class StrictLGSSMNeuTraTrainingError(RuntimeError):
    """Raised when the strict training closure or execution fails closed."""


@dataclass(frozen=True)
class TrainingRecipe:
    recipe_id: str
    stage_count: int
    hidden_layers: tuple[int, ...]
    learning_rate: float

    def payload(self) -> Mapping[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage_count": self.stage_count,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": 1.0,
            "batch_size": BATCH_SIZE,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_constant_learning_rate",
            "composition": "T_phi(z)=affine(dense_iaf_stack(z))",
        }


SCREEN_RECIPES = {
    SOURCE_ANCHOR: TrainingRecipe(SOURCE_ANCHOR, 3, (18, 18), 5.0e-3),
    "lower_lr1e3": TrainingRecipe("lower_lr1e3", 3, (18, 18), 1.0e-3),
    "shallow_2stage_lr5e3": TrainingRecipe(
        "shallow_2stage_lr5e3", 2, (18, 18), 5.0e-3
    ),
    "wide_2x_lr5e3": TrainingRecipe("wide_2x_lr5e3", 3, (36, 36), 5.0e-3),
}


def run_gpu_training_job(
    *,
    job_kind: str,
    job_id: str,
    artifact_root: str | Path | None = None,
    selected_recipe_path: str | Path | None = None,
    step_override: int | None = None,
) -> Mapping[str, Any]:
    """Run one strict training job with one compiled optimization invocation."""

    recipe, seed, planned_steps, selection_reference = _job_spec(
        job_kind=job_kind,
        job_id=job_id,
        selected_recipe_path=selected_recipe_path,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise StrictLGSSMNeuTraTrainingError("GPU training cannot run with CUDA hidden")
    steps = planned_steps if step_override is None else int(step_override)
    if steps <= 0 or steps > planned_steps:
        raise StrictLGSSMNeuTraTrainingError("step override is outside the job budget")
    root_base = DEFAULT_ARTIFACT_ROOT if artifact_root is None else Path(artifact_root)
    root = _job_root(root_base, job_kind=job_kind, job_id=job_id)
    if root.exists():
        raise StrictLGSSMNeuTraTrainingError(f"training job root already exists: {root}")

    contract = _validate_immutable_inputs()
    tf, gpu_memory_policy = _trusted_tensorflow()
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraTrainingError,
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
        train_plain_dense_iaf,
    )
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    if bundle.adapter.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
        raise StrictLGSSMNeuTraTrainingError("exact-target adapter signature mismatch")
    center, factor = _load_affine_geometry(tf)
    config = PlainDenseIAFTrainingConfig(
        target_signature=bundle.target_signature,
        dimension=DIMENSION,
        affine_center=center,
        affine_factor=factor,
        output_dir=root / "training",
        seed=seed,
        hidden_layers=recipe.hidden_layers,
        stage_count=recipe.stage_count,
        activation="elu",
        s_max=1.0,
        init_scale=0.02,
        steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=recipe.learning_rate,
        final_learning_rate_fraction=1.0,
        clip_norm=10.0,
        checkpoint_every=steps,
        heartbeat_every=(steps if job_kind == "smoke" else HEARTBEAT_EVERY),
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    start = time.monotonic()
    try:
        trained = train_plain_dense_iaf(
            adapter=bundle.adapter,
            config=config,
            freeze_transport_id=(
                f"lgssm-neutra-strict-{job_kind}-{job_id}-{steps}steps"
            ),
        )
        if trained.frozen_payload_path is None or trained.completed_steps != steps:
            raise StrictLGSSMNeuTraTrainingError("training did not freeze at terminal step")
        loaded = load_frozen_neutra_artifact(
            _read_mapping(trained.frozen_payload_path, "frozen transport"),
            expected_target_signature=bundle.target_signature,
        )
        flow = restore_plain_dense_iaf_flow(
            config=config, state_path=trained.state_path
        )
        completion = {
            "schema": "bayesfilter.lgssm_neutra_strict_training_completion.v1",
            "job_kind": job_kind,
            "job_id": job_id,
            "steps": steps,
            "target_signature": bundle.target_signature,
            "adapter_signature": bundle.adapter.adapter_signature(),
            "training_state_hash": trained.state_hash,
            "compiled_training_program_invocations": trained.runtime_metadata[
                "compiled_training_program_invocations"
            ],
            "compiled_training_control_flow": trained.runtime_metadata[
                "compiled_training_control_flow"
            ],
            "checkpoint_policy": trained.runtime_metadata["checkpoint_policy"],
            "target_status_all_valid": all(
                row.get("target_status_all_valid") for row in trained.records
            ),
            "checkpoint": _file_reference(trained.state_path),
            "payload": _file_reference(trained.frozen_payload_path),
            "selected_recipe_source": selection_reference,
            "evidence_role": "durable_training_completion_before_post_training_validation",
            "nonclaims": NONCLAIMS,
        }
        _write_new_json(
            root / "training_completion.json", _with_artifact_hash(completion)
        )
        parity = _compiled_parity(tf, flow, loaded)
        heldout = (
            _compiled_heldout(tf, bundle, loaded) if job_kind == "screen" else None
        )
        closure = audit_imported_bayesfilter_closure()
        if closure["passed"] is not True:
            raise StrictLGSSMNeuTraTrainingError("repository import closure uses NumPy")
    except (NeuTraTrainingError, StrictLGSSMNeuTraTrainingError) as exc:
        result = {
            "schema": "bayesfilter.lgssm_neutra_strict_training_job.v1",
            "job_kind": job_kind,
            "job_id": job_id,
            "passed": False,
            "decision": "REJECT_STRICT_TRAINING_JOB",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_seconds": time.monotonic() - start,
            "nonclaims": NONCLAIMS,
        }
        _write_new_json(root / "result.json", _with_artifact_hash(result))
        return result

    records = tuple(trained.records)
    result = {
        "schema": "bayesfilter.lgssm_neutra_strict_training_job.v1",
        "job_kind": job_kind,
        "job_id": job_id,
        "passed": True,
        "decision": "ENGINEERING_VALID_GRAPH_NATIVE_NUMPY_FREE_TRAINING_JOB",
        "recipe": recipe.payload(),
        "seed": seed,
        "steps": steps,
        "planned_steps": planned_steps,
        "step_override_debug_only": step_override is not None,
        "target_signature": bundle.target_signature,
        "adapter_signature": bundle.adapter.adapter_signature(),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "training_state_hash": trained.state_hash,
        "payload": _file_reference(trained.frozen_payload_path),
        "checkpoint": _file_reference(trained.state_path),
        "progress": _file_reference(trained.progress_path),
        "records": records,
        "runtime_metadata": trained.runtime_metadata,
        "compiled_training_program_invocations": trained.runtime_metadata[
            "compiled_training_program_invocations"
        ],
        "compiled_training_control_flow": trained.runtime_metadata[
            "compiled_training_control_flow"
        ],
        "checkpoint_policy": trained.runtime_metadata["checkpoint_policy"],
        "frozen_reload_and_score_parity": parity,
        "heldout_common_batches": heldout,
        "repository_import_closure": closure,
        "selected_recipe_source": selection_reference,
        "campaign_contract": {
            "path": str(CONTRACT_PATH.relative_to(ROOT)),
            "file_sha256": EXPECTED_CONTRACT_FILE_SHA256,
            "contract_hash": contract["contract_hash"],
        },
        "gpu_manifest": _gpu_manifest(tf, gpu_memory_policy=gpu_memory_policy),
        "elapsed_seconds": time.monotonic() - start,
        "evidence_role": (
            "wiring_only" if step_override is not None or job_kind == "smoke" else
            "proxy_nomination_only" if job_kind == "screen" else
            "engineering_candidate_for_downstream_hmc"
        ),
        "nonclaims": NONCLAIMS,
    }
    result = _with_artifact_hash(result)
    _write_new_json(root / "result.json", result)
    return result


def audit_imported_bayesfilter_closure() -> Mapping[str, Any]:
    """Reject NumPy and Python callback bridges in imported repository modules."""

    rows = []
    violations = []
    for module_name in sorted(
        name for name in sys.modules if name == "bayesfilter" or name.startswith("bayesfilter.")
    ):
        module = sys.modules[module_name]
        source_path = getattr(module, "__file__", None)
        if not source_path or not str(source_path).endswith(".py"):
            continue
        path = Path(source_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_violations = []
        numpy_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy" or alias.name.startswith("numpy."):
                        numpy_aliases.add(alias.asname or alias.name.split(".")[0])
                        module_violations.append(f"numpy_import:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "numpy" or str(node.module).startswith("numpy."):
                    module_violations.append(f"numpy_from_import:{node.lineno}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in numpy_aliases:
                    module_violations.append(f"numpy_call:{node.lineno}")
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tf"
                    and node.func.attr in {"numpy_function", "py_function"}
                ):
                    module_violations.append(f"tensorflow_host_callback:{node.lineno}")
        relative = str(path.relative_to(ROOT))
        rows.append({"module": module_name, "path": relative})
        violations.extend(f"{relative}:{item}" for item in module_violations)
    return {
        "passed": not violations,
        "policy": "no_repository_numpy_import_or_call_and_no_tf_host_callback",
        "module_count": len(rows),
        "modules": tuple(rows),
        "violations": tuple(violations),
        "tensorflow_third_party_numpy_dependency_out_of_scope": True,
    }


def _compiled_parity(tf: Any, flow: Any, loaded: Any) -> Mapping[str, Any]:
    probes = _common_probe_points(tf)
    theta_score = tf.random.stateless_normal(
        tf.shape(probes), seed=(20260714, 1601), dtype=tf.float64
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg, theta_score_arg):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_arg)
            theta_train, logdet_train = flow.forward_and_logdet(z_arg)
            pullback_objective = tf.reduce_sum(theta_train * theta_score_arg)
            logdet_objective = tf.reduce_sum(logdet_train)
        pullback_train = tape.gradient(pullback_objective, z_arg)
        logdet_score_train = tape.gradient(logdet_objective, z_arg)
        theta_frozen = loaded.transport.forward_batch(z_arg)
        logdet_frozen = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        pullback_frozen = loaded.transport.pullback_score_batch(
            z_arg, theta_score_arg
        )
        logdet_score_frozen = loaded.transport.log_abs_det_jacobian_score_batch(
            z_arg
        )
        return (
            tf.reduce_max(tf.abs(theta_train - theta_frozen)),
            tf.reduce_max(tf.abs(logdet_train - logdet_frozen)),
            tf.reduce_max(tf.abs(pullback_train - pullback_frozen)),
            tf.reduce_max(tf.abs(logdet_score_train - logdet_score_frozen)),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(probes, theta_score)
    devices = tuple(str(item.device) for item in outputs)
    deltas = tuple(float(item.numpy()) for item in outputs)
    passed = bool(
        deltas[0] <= 1.0e-12
        and deltas[1] <= 1.0e-12
        and deltas[2] <= 1.0e-12
        and deltas[3] <= 1.0e-12
        and all("GPU" in device.upper() for device in devices)
    )
    if not passed:
        raise StrictLGSSMNeuTraTrainingError("frozen TensorFlow parity failed")
    return {
        "passed": True,
        "transport_max_abs": deltas[0],
        "logdet_max_abs": deltas[1],
        "pullback_score_max_abs": deltas[2],
        "logdet_score_max_abs": deltas[3],
        "target_status_source": "all_compiled_training_records",
        "output_devices": devices,
        "jit_compile": True,
    }


def _compiled_heldout(tf: Any, bundle: Any, loaded: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    fixed = FixedTransportValueScoreAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
        target_scope="lgssm_neutra_strict_training_heldout",
        evidence_path="docs/plans/bayesfilter-lgssm-neutra-graph-native-training-migration-plan-2026-07-14.md",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    seeds = tf.constant(HELDOUT_SEEDS, tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(seed_tensor):
        z = tf.map_fn(
            lambda seed: tf.random.stateless_normal(
                (BATCH_SIZE, DIMENSION), seed=seed, dtype=tf.float64
            ),
            seed_tensor,
            fn_output_signature=tf.TensorSpec((BATCH_SIZE, DIMENSION), tf.float64),
        )
        flat = tf.reshape(z, (-1, DIMENSION))
        theta = loaded.transport.forward_batch(flat)
        logdet = loaded.transport.log_abs_det_jacobian_batch(flat)
        value, score = fixed.log_prob_and_grad_batch(flat)
        status = bundle.adapter.target_status_telemetry(theta)
        objective = tf.reshape(-(value + logdet), (len(HELDOUT_SEEDS), BATCH_SIZE))
        force = tf.reshape(
            tf.linalg.norm(score, axis=-1), (len(HELDOUT_SEEDS), BATCH_SIZE)
        )
        return (
            tf.reduce_mean(objective, axis=1),
            tf.math.reduce_std(objective, axis=1),
            tf.reduce_mean(force, axis=1),
            tf.reduce_max(force, axis=1),
            tf.reduce_all(tf.equal(status["status_code"], 0)),
            tf.reduce_all(status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(seeds)
    means = tuple(float(value) for value in outputs[0].numpy().tolist())
    if not bool(outputs[4].numpy()) or not bool(outputs[5].numpy()):
        raise StrictLGSSMNeuTraTrainingError("held-out target status failed")
    rows = tuple(
        {
            "seed": seed,
            "reverse_kl_objective_mean": means[index],
            "reverse_kl_objective_sd": float(outputs[1][index].numpy()),
            "transformed_force_norm_mean": float(outputs[2][index].numpy()),
            "transformed_force_norm_max": float(outputs[3][index].numpy()),
            "target_status_all_valid": True,
        }
        for index, seed in enumerate(HELDOUT_SEEDS)
    )
    return {
        "batch_count": len(HELDOUT_SEEDS),
        "batch_size": BATCH_SIZE,
        "common_seed_policy": "identical_stateless_base_draws_for_every_recipe",
        "rows": rows,
        "mean_reverse_kl_objective": _finite_mean(means),
        "mcse_across_batches": _sample_mean_mcse(means),
        "target_status_all_valid": True,
        "metric_role": "proxy_nomination_only_not_transport_promotion",
        "single_compiled_heldout_invocation": True,
    }


def _validate_immutable_inputs() -> Mapping[str, Any]:
    if _file_sha256(CONTRACT_PATH) != EXPECTED_CONTRACT_FILE_SHA256:
        raise StrictLGSSMNeuTraTrainingError("campaign contract file hash mismatch")
    contract = _read_mapping(CONTRACT_PATH, "campaign contract")
    if contract.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise StrictLGSSMNeuTraTrainingError("campaign target signature mismatch")
    if contract.get("adapter_signature") != EXPECTED_ADAPTER_SIGNATURE:
        raise StrictLGSSMNeuTraTrainingError("campaign adapter signature mismatch")
    if _file_sha256(MASS_PATH) != EXPECTED_MASS_FILE_SHA256:
        raise StrictLGSSMNeuTraTrainingError("mass file hash mismatch")
    mass = _read_mapping(MASS_PATH, "mass")
    if mass.get("artifact_hash") != EXPECTED_MASS_ARTIFACT_HASH:
        raise StrictLGSSMNeuTraTrainingError("mass artifact hash mismatch")
    return contract


def _load_affine_geometry(tf: Any) -> tuple[tf.Tensor, tf.Tensor]:
    mass = _read_mapping(MASS_PATH, "mass")
    center = tf.convert_to_tensor(mass["center"], tf.float64)
    factor = tf.convert_to_tensor(mass["factor"], tf.float64)
    covariance = tf.convert_to_tensor(mass["mass_covariance"], tf.float64)
    if center.shape != (DIMENSION,) or factor.shape != (DIMENSION, DIMENSION):
        raise StrictLGSSMNeuTraTrainingError("affine geometry shape mismatch")
    residual = float(tf.reduce_max(tf.abs(factor @ tf.transpose(factor) - covariance)).numpy())
    if residual > 1.0e-12:
        raise StrictLGSSMNeuTraTrainingError("affine factor covariance mismatch")
    return center, factor


def _trusted_tensorflow():
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        TensorFlowGPUMemoryPolicyError,
        configure_tensorflow_gpu_memory_growth,
    )

    try:
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    except TensorFlowGPUMemoryPolicyError as exc:
        raise StrictLGSSMNeuTraTrainingError(str(exc)) from exc
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    return tf, memory_policy


def _job_spec(
    *,
    job_kind: str,
    job_id: str,
    selected_recipe_path: str | Path | None = None,
) -> tuple[TrainingRecipe, tuple[int, int], int, Mapping[str, Any] | None]:
    if job_kind in {"smoke", "screen"}:
        if selected_recipe_path is not None:
            raise StrictLGSSMNeuTraTrainingError(
                "selected recipe is valid only for final training jobs"
            )
        try:
            recipe = SCREEN_RECIPES[job_id]
        except KeyError as exc:
            raise StrictLGSSMNeuTraTrainingError(f"unknown recipe: {job_id}") from exc
        if job_kind == "smoke":
            seed = (20260714, 1411 + SCREEN_RECIPE_ORDER.index(job_id))
            return recipe, seed, SMOKE_STEPS, None
        return recipe, SCREEN_SEED, SCREEN_STEPS, None
    if job_kind == "final":
        try:
            seed = FINAL_SEEDS[job_id]
        except KeyError as exc:
            raise StrictLGSSMNeuTraTrainingError(f"unknown final job: {job_id}") from exc
        if selected_recipe_path is None:
            raise StrictLGSSMNeuTraTrainingError(
                "final training requires an explicit selected recipe artifact"
            )
        recipe, reference = _validate_selected_recipe(
            Path(selected_recipe_path), expected_job_id=job_id, expected_seed=seed
        )
        return recipe, seed, FINAL_STEPS, reference
    raise StrictLGSSMNeuTraTrainingError(f"unknown job kind: {job_kind}")


def _validate_selected_recipe(
    path: Path,
    *,
    expected_job_id: str,
    expected_seed: tuple[int, int],
) -> tuple[TrainingRecipe, Mapping[str, Any]]:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise StrictLGSSMNeuTraTrainingError(
            "selected recipe artifact must be inside the repository"
        ) from exc
    selected = _read_mapping(resolved, "selected recipe")
    if selected.get("schema") != "bayesfilter.lgssm_neutra_selected_training_recipe.v1":
        raise StrictLGSSMNeuTraTrainingError("selected recipe schema mismatch")
    if not _artifact_hash_matches(selected):
        raise StrictLGSSMNeuTraTrainingError("selected recipe artifact hash mismatch")
    if (
        int(selected.get("final_steps", -1)) != FINAL_STEPS
        or selected.get("screen_weights_reused") is not False
        or selected.get("final_seeds", {}).get(expected_job_id)
        != list(expected_seed)
    ):
        raise StrictLGSSMNeuTraTrainingError("selected recipe final contract mismatch")
    try:
        recipe_id = str(selected["selected_recipe"]["recipe_id"])
        recipe = SCREEN_RECIPES[recipe_id]
    except (KeyError, TypeError) as exc:
        raise StrictLGSSMNeuTraTrainingError("selected recipe identity is unknown") from exc
    if _json_ready(selected.get("selected_recipe")) != _json_ready(recipe.payload()):
        raise StrictLGSSMNeuTraTrainingError("selected recipe payload mismatch")
    result_reference = selected.get("selection_result")
    if not isinstance(result_reference, Mapping):
        raise StrictLGSSMNeuTraTrainingError("selected recipe result reference is missing")
    result_path = (ROOT / str(result_reference.get("path", ""))).resolve()
    try:
        result_path.relative_to(ROOT)
    except ValueError as exc:
        raise StrictLGSSMNeuTraTrainingError(
            "selected recipe result must be inside the repository"
        ) from exc
    if not result_path.is_file() or result_reference.get("file_sha256") != _file_sha256(
        result_path
    ):
        raise StrictLGSSMNeuTraTrainingError("selected recipe result file hash mismatch")
    result = _read_mapping(result_path, "selected recipe screen result")
    if (
        result.get("schema") != "bayesfilter.neutra.batch_native_screen_result.v1"
        or result.get("passed") is not True
        or not _artifact_hash_matches(result)
        or result.get("selection", {}).get("selected_recipe_id") != recipe_id
        or result.get("artifact_hash") != selected.get("selection_result_artifact_hash")
        or result.get("screen_weights_reused_by_final") is not False
    ):
        raise StrictLGSSMNeuTraTrainingError("selected recipe result identity mismatch")
    return recipe, {
        "selected_recipe": _file_reference(resolved),
        "selected_recipe_artifact_hash": selected["artifact_hash"],
        "selection_result": _file_reference(result_path),
        "selection_result_artifact_hash": result["artifact_hash"],
        "recipe_id": recipe_id,
        "screen_weights_reused": False,
    }


def _job_root(root: Path, *, job_kind: str, job_id: str) -> Path:
    if job_kind in {"smoke", "screen"}:
        return root / job_kind / "candidates" / job_id / "attempt_1_graph_native"
    return root / "phase4" / "training_jobs" / job_id / "attempt_1_graph_native"


def _common_probe_points(tf: Any):
    values = tf.range(4 * DIMENSION, dtype=tf.float64)
    midpoint = tf.constant(0.5 * (4 * DIMENSION - 1), tf.float64)
    return tf.reshape((values - midpoint) / tf.constant(97.0, tf.float64), (4, DIMENSION))


def _finite_mean(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise StrictLGSSMNeuTraTrainingError("mean inputs must be finite")
    return math.fsum(numeric) / len(numeric)


def _sample_mean_mcse(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    mean = _finite_mean(numeric)
    variance = math.fsum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
    return math.sqrt(variance / len(numeric))


def _gpu_manifest(
    tf: Any,
    *,
    gpu_memory_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "trust_basis": TRUST_BASIS,
        "tensorflow_version": tf.__version__,
        "physical_gpus": tuple(str(item) for item in tf.config.list_physical_devices("GPU")),
        "logical_gpus": tuple(str(item) for item in tf.config.list_logical_devices("GPU")),
        "nvidia_smi": tuple(line for line in query.stdout.splitlines() if line.strip()),
        "gpu_memory_policy": dict(gpu_memory_policy),
        "training_dtype": "float64",
        "jit_compile": True,
    }


def _read_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictLGSSMNeuTraTrainingError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise StrictLGSSMNeuTraTrainingError(f"{label} must be a JSON object")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "file_sha256": _file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _artifact_hash_matches(payload: Mapping[str, Any]) -> bool:
    normalized = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_hash", "artifact_hash_semantics"}
    }
    return payload.get("artifact_hash") == f"sha256:{_stable_json_hash(normalized)}"


def _with_artifact_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = f"sha256:{_stable_json_hash(result)}"
    result["artifact_hash_semantics"] = "stable_json_sha256_excluding_artifact_hash_fields"
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _json_ready(materialized.tolist())
        if hasattr(materialized, "item"):
            return _json_ready(materialized.item())
        return _json_ready(materialized)
    return value
