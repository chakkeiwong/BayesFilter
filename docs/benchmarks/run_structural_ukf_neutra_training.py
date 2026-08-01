#!/usr/bin/env python3
"""Screen and train a target-specific dense-IAF for structural UKF NeuTra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as common


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CAMPAIGN_ID = "structural-ukf-neutra-truth-tail-20260717"
CELL_ID = "STR-UKF"
DIMENSION = 5
EXPECTED_TYPED_SIGNATURE = (
    "e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665"
)
IDENTITY_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p5/STR-UKF/r1b-identity/cpu-attempt-02"
)
IDENTITY_RESULT_SHA256 = (
    "73fd7a10fd89999993b2b88b636774df489e984e1c589cb3efff57ce2d3ea83d"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-structural-ukf-neutra-truth-tail-campaign-"
    "2026-07-17.md"
)
CAMPAIGN_ROOT = Path(
    "docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717"
)
SCREEN_STEPS = 500
FINAL_STEPS = 5000
BATCH_SIZE = 128
HELDOUT_BATCH_COUNT = 8
HELDOUT_BATCH_SIZE = 128
RECIPE_ORDER = (
    "dim3_lr1e3",
    "dim3_lr5e3",
    "dim6_lr1e3",
    "dim6_lr5e3",
)
SCREEN_SEEDS = {
    recipe: (20260717, 41001 + index) for index, recipe in enumerate(RECIPE_ORDER)
}
HELDOUT_SEED = (20260717, 41100)
FINAL_SEED = (20260717, 41201)
NONCLAIMS = (
    "training and heldout reverse KL are nomination evidence only",
    "the zero-identity affine start is source-prior geometry, not posterior geometry",
    "screen weights are never reused by final training",
    "no HMC convergence, truth recovery, filter exactness, calibration, superiority, or readiness claim",
)


class StructuralUKFTrainingError(RuntimeError):
    """Raised when the structural training evidence contract fails."""


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    hidden_layers: tuple[int, ...]
    learning_rate: float

    def payload(self) -> Mapping[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "stage_count": 3,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": 0.1,
            "batch_size": BATCH_SIZE,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_linear_decay",
        }


RECIPES = {
    "dim3_lr1e3": Recipe("dim3_lr1e3", (15, 15), 1.0e-3),
    "dim3_lr5e3": Recipe("dim3_lr5e3", (15, 15), 5.0e-3),
    "dim6_lr1e3": Recipe("dim6_lr1e3", (30, 30), 1.0e-3),
    "dim6_lr5e3": Recipe("dim6_lr5e3", (30, 30), 5.0e-3),
}


def _audit_points(tf: Any, structural_truth_source: Any) -> Any:
    # Typed identity binds serialized audit points, so construct them on CPU.
    with tf.device("/CPU:0"):
        truth = structural_truth_source()
        eye = 0.5 * tf.eye(DIMENSION, dtype=tf.float64)
        tails = tf.constant(
            [[1.5, -1.0, 1.2, -1.4, 0.8], [-1.3, 1.4, -1.1, 1.0, -1.5]],
            tf.float64,
        )
        return tf.concat(
            [truth[None, :], tf.zeros([1, DIMENSION], tf.float64),
             truth[None, :] + eye, truth[None, :] - eye, tails],
            axis=0,
        )


def reconstruct_identity(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_campaign import (
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        StructuralUKFLikelihoodRecomposer,
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
        structural_source_probit_jacobian_value_score,
        structural_source_uniform_prior_value_score,
        structural_truth_source,
    )

    reference = common._verify_result_root(
        IDENTITY_ROOT, IDENTITY_RESULT_SHA256, require_passed=True
    )
    expected = common._read_mapping(IDENTITY_ROOT / "target_identity.json")
    registry = common._read_mapping(IDENTITY_ROOT / "repaired_registry.json")
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=_audit_points(tf, structural_truth_source),
        prior_value_score_fn=structural_source_uniform_prior_value_score,
        likelihood_value_score_fn=StructuralUKFLikelihoodRecomposer(adapter).__call__,
        jacobian_value_score_fn=structural_source_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=CELL_ID,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=registry,
        registry_artifact_sha256=common._file_sha256(
            IDENTITY_ROOT / "repaired_registry.json"
        ),
    )
    require_typed_neutra_target(identity, adapter=adapter)
    observed = common._json_ready(identity.payload())
    if observed != expected:
        observed_blob = json.dumps(
            observed, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        expected_blob = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        raise StructuralUKFTrainingError(
            "structural typed identity drift: "
            f"first_difference={first_difference(expected, observed)}; "
            f"expected_sha256={hashlib.sha256(expected_blob).hexdigest()}; "
            f"observed_sha256={hashlib.sha256(observed_blob).hexdigest()}"
        )
    if identity.target_signature != EXPECTED_TYPED_SIGNATURE:
        raise StructuralUKFTrainingError("structural target signature drift")
    return adapter, identity, reference


def first_difference(expected: Any, observed: Any, path: str = "$") -> str:
    if type(expected) is not type(observed):
        return f"{path}:type:{type(expected).__name__}!={type(observed).__name__}"
    if isinstance(expected, Mapping):
        expected_keys = tuple(sorted(expected))
        observed_keys = tuple(sorted(observed))
        if expected_keys != observed_keys:
            return f"{path}:keys:{expected_keys!r}!={observed_keys!r}"
        for key in expected_keys:
            difference = first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}:length:{len(expected)}!={len(observed)}"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if expected == observed else f"{path}:{expected!r}!={observed!r}"


def run_training_job(
    *,
    job_kind: str,
    recipe_id: str,
    output_root: Path,
    stop_after_steps: int | None = None,
    resume_infrastructure_from: Path | None = None,
) -> Mapping[str, Any]:
    if job_kind not in {"screen", "final"}:
        raise StructuralUKFTrainingError("job_kind must be screen or final")
    try:
        recipe = RECIPES[recipe_id]
    except KeyError as exc:
        raise StructuralUKFTrainingError(f"unknown recipe: {recipe_id}") from exc
    if output_root.exists():
        raise FileExistsError(f"training output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_campaign import (
        load_campaign_neutra_transport,
        require_typed_neutra_target,
        train_campaign_neutra,
    )
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
        train_plain_dense_iaf,
    )

    adapter, identity, identity_reference = reconstruct_identity(tf)
    selection = None
    if job_kind == "final":
        selection = load_selection()
        if selection["selected_recipe_id"] != recipe_id:
            raise StructuralUKFTrainingError("final recipe differs from selection")
        steps = FINAL_STEPS
        seed = FINAL_SEED
    else:
        steps = SCREEN_STEPS
        seed = SCREEN_SEEDS[recipe_id]

    terminal_step = steps if stop_after_steps is None else int(stop_after_steps)
    if terminal_step <= 0 or terminal_step > steps:
        raise StructuralUKFTrainingError("stop_after_steps is outside the run")
    if job_kind == "screen" and (
        terminal_step != steps or resume_infrastructure_from is not None
    ):
        raise StructuralUKFTrainingError("screen jobs do not support segmentation")
    parent_completed_steps = 0
    if resume_infrastructure_from is not None:
        parent_state = common._read_mapping(resume_infrastructure_from)
        parent_completed_steps = int(parent_state.get("completed_steps", -1))
        if not 0 < parent_completed_steps < terminal_step:
            raise StructuralUKFTrainingError("invalid infrastructure-resume step boundary")

    center = tf.zeros([DIMENSION], tf.float64)
    factor = tf.eye(DIMENSION, dtype=tf.float64)
    config = PlainDenseIAFTrainingConfig(
        target_signature=identity.target_signature,
        dimension=DIMENSION,
        affine_center=center,
        affine_factor=factor,
        output_dir=output_root / "training",
        seed=seed,
        hidden_layers=recipe.hidden_layers,
        stage_count=3,
        activation="elu",
        s_max=1.0,
        init_scale=0.02,
        steps=steps,
        batch_size=BATCH_SIZE,
        learning_rate=recipe.learning_rate,
        final_learning_rate_fraction=0.1,
        clip_norm=10.0,
        checkpoint_every=steps,
        heartbeat_every=10,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    freeze_transport_id = (
        f"structural-ukf-{job_kind}-{recipe_id}-{steps}-20260717"
        if terminal_step == steps
        else None
    )
    if terminal_step == steps and resume_infrastructure_from is None:
        trained = train_campaign_neutra(
            identity=identity,
            adapter=adapter,
            config=config,
            freeze_transport_id=str(freeze_transport_id),
            gpu_memory_policy=memory_policy,
        )
    else:
        require_typed_neutra_target(identity, adapter=adapter)
        trained = train_plain_dense_iaf(
            adapter=adapter,
            config=config,
            stop_after_steps=terminal_step,
            resume_infrastructure_from=resume_infrastructure_from,
            freeze_transport_id=freeze_transport_id,
        )
    if trained.completed_steps != terminal_step:
        raise StructuralUKFTrainingError("training did not reach segment terminal step")
    require_training_runtime(
        trained,
        expected_completed_steps=terminal_step,
        expected_program_steps=terminal_step - parent_completed_steps,
    )
    loaded = None
    parity = None
    heldout = None
    payload_reference = None
    transport_artifact_signature = None
    transport_hash = None
    if terminal_step == steps:
        if trained.frozen_payload_path is None:
            raise StructuralUKFTrainingError("completed training did not freeze")
        payload = common._read_mapping(trained.frozen_payload_path)
        loaded = load_campaign_neutra_transport(
            identity=identity, adapter=adapter, payload=payload
        )
        flow = restore_plain_dense_iaf_flow(config=config, state_path=trained.state_path)
        parity = compiled_parity(tf, flow, loaded, recipe_id=recipe_id)
        heldout = heldout_diagnostics(tf, adapter=adapter, loaded=loaded)
        payload_reference = common._file_reference(trained.frozen_payload_path)
        transport_artifact_signature = loaded.artifact_signature
        transport_hash = loaded.manifest.transport_hash
    passed = bool(
        terminal_step < steps
        or (
            parity is not None
            and parity["passed"]
            and heldout is not None
            and heldout["target_status_all_valid"]
        )
    )
    result = {
        "schema": "bayesfilter.structural_ukf_neutra_training_job.v1",
        "campaign_id": CAMPAIGN_ID,
        "identity_program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "passed": passed,
        "decision": (
            "CHECKPOINT_STRUCTURAL_UKF_FINAL_TRAINING_SEGMENT"
            if terminal_step < steps
            else (
            "PASS_STRUCTURAL_UKF_TRAINING_SCREEN"
            if job_kind == "screen" and passed
            else (
                "ADMIT_STRUCTURAL_UKF_FRESH_5000_STEP_TRANSPORT"
                if passed
                else "REJECT_STRUCTURAL_UKF_TRAINING_INVALID"
            )
            )
        ),
        "recipe": recipe.payload(),
        "seed": seed,
        "steps": steps,
        "completed_steps": terminal_step,
        "segmented_execution": bool(
            terminal_step < steps or resume_infrastructure_from is not None
        ),
        "resume_infrastructure_from": (
            None
            if resume_infrastructure_from is None
            else str(resume_infrastructure_from)
        ),
        "screen_weights_reused_by_final": False,
        "affine_initialization": {
            "center": [0.0] * DIMENSION,
            "factor": tf.eye(DIMENSION, dtype=tf.float64).numpy().tolist(),
            "role": "standard_normal_source_prior_warm_start_not_posterior_geometry",
        },
        "target_identity": identity.payload(),
        "identity_reference": identity_reference,
        "selection_reference": selection,
        "training_state_hash": trained.state_hash,
        "transport_artifact_signature": transport_artifact_signature,
        "transport_hash": transport_hash,
        "payload": payload_reference,
        "checkpoint": common._file_reference(trained.state_path),
        "progress": common._file_reference(trained.progress_path),
        "records": trained.records,
        "runtime_metadata": trained.runtime_metadata,
        "frozen_trainable_parity": parity,
        "heldout_common_batches": heldout,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_role": (
            "infrastructure_checkpoint_not_scientific_evidence"
            if terminal_step < steps
            else ("proxy_nomination_only"
            if job_kind == "screen"
            else "engineering_candidate_for_structural_truth_tail_hmc"
            )
        ),
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_root / "result.json", result)
    common._write_new_json(
        output_root / "run_manifest.json",
        run_manifest(
            output_root=output_root,
            job_kind=job_kind,
            recipe_id=recipe_id,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            wall_time=time.monotonic() - started,
        ),
    )
    write_recursive_hashes(output_root)
    return result


def run_canary(*, output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"canary output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    adapter, identity, identity_reference = reconstruct_identity(tf)
    probes = tf.random.stateless_normal(
        (BATCH_SIZE, DIMENSION),
        seed=tf.constant((20260717, 40901), tf.int32),
        stddev=tf.constant(0.25, tf.float64),
        dtype=tf.float64,
    )

    @tf.function(
        input_signature=[tf.TensorSpec([BATCH_SIZE, DIMENSION], tf.float64)],
        jit_compile=True,
    )
    def compiled(theta: Any):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    with tf.device("/GPU:0"):
        value, score, status = compiled(probes)
    passed = bool(
        tf.reduce_all(tf.math.is_finite(value)).numpy()
        and tf.reduce_all(tf.math.is_finite(score)).numpy()
        and tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        and all("GPU" in str(item.device).upper() for item in (value, score))
    )
    result = {
        "schema": "bayesfilter.structural_ukf_neutra_gpu_xla_canary.v1",
        "campaign_id": CAMPAIGN_ID,
        "cell_id": CELL_ID,
        "passed": passed,
        "decision": (
            "PASS_STRUCTURAL_UKF_GPU_XLA_BATCH_CANARY"
            if passed
            else "BLOCK_STRUCTURAL_UKF_GPU_XLA_BATCH_CANARY"
        ),
        "target_identity": identity.payload(),
        "identity_reference": identity_reference,
        "batch_size": BATCH_SIZE,
        "seed": (20260717, 40901),
        "jit_compile": True,
        "value_all_finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
        "score_all_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "target_status_all_valid": bool(
            tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
            and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        ),
        "value_device": str(value.device),
        "score_device": str(score.device),
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_root / "result.json", result)
    common._write_new_json(
        output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.structural_ukf_neutra_canary_manifest.v1",
            "campaign_id": CAMPAIGN_ID,
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dirty_worktree_disclosure": "shared dirty worktree; scoped new paths only",
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "tensorflow_version": tf.__version__,
            "tensorflow_probability_version": tfp.__version__,
            "gpu_memory_policy": memory_policy,
            "jit_compile": True,
            "dtype": "float64",
            "tf32_execution_enabled": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "target_signature": identity.target_signature,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output_root),
            "plan_file": str(PLAN_PATH),
            "result_file": str(output_root / "result.json"),
            "nonclaims": NONCLAIMS,
        },
    )
    write_recursive_hashes(output_root)
    if not passed:
        raise StructuralUKFTrainingError("GPU/XLA canary failed")
    return result


def require_training_runtime(
    trained: Any, *, expected_completed_steps: int, expected_program_steps: int
) -> None:
    runtime = trained.runtime_metadata
    required = {
        "jit_compile": True,
        "require_gpu": True,
        "training_batch_size": BATCH_SIZE,
        "scalar_fallback_used": False,
        "sample_axis_python_loop_used": False,
        "row_mapped_scalar_target_used": False,
        "compiled_training_program_invocations": 1,
        "compiled_training_control_flow": "tf_while_loop",
        "program_step_count": expected_program_steps,
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            raise StructuralUKFTrainingError(f"training runtime failed: {key}")
    batch_target = runtime.get("batch_native_target")
    if not isinstance(batch_target, Mapping):
        raise StructuralUKFTrainingError("missing batch-native target metadata")
    for key in (
        "scalar_fallback_used",
        "sample_axis_python_loop_used",
        "row_mapped_scalar_target_used",
    ):
        if batch_target.get(key) is not False:
            raise StructuralUKFTrainingError(f"batch-native target failed: {key}")
    for key in (
        "trainable_variable_devices",
        "adam_moment_devices",
        "compiled_output_devices",
    ):
        devices = tuple(str(item) for item in runtime.get(key, ()))
        if not devices or not all("GPU" in item.upper() for item in devices):
            raise StructuralUKFTrainingError(f"training is not GPU-only: {key}")
    if trained.completed_steps != expected_completed_steps or not trained.records or any(
        row.get("target_values_finite") is not True
        or row.get("target_status_available") is not True
        or row.get("target_status_all_valid") is not True
        or int(row.get("target_status_nonvalid_count", 1)) != 0
        for row in trained.records
    ):
        raise StructuralUKFTrainingError("training target-health record failed")


def compiled_parity(
    tf: Any, flow: Any, loaded: Any, *, recipe_id: str
) -> Mapping[str, Any]:
    probes = tf.constant(
        [[0.0] * DIMENSION, [0.1, -0.1, 0.08, -0.08, 0.06]], tf.float64
    )
    seed = SCREEN_SEEDS[recipe_id]
    theta_score = tf.random.stateless_normal(
        tf.shape(probes), seed=(seed[0], seed[1] + 700), dtype=tf.float64
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(z_arg: Any, score_arg: Any):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_arg)
            theta_train, logdet_train = flow.forward_and_logdet(z_arg)
            pullback_objective = tf.reduce_sum(theta_train * score_arg)
            logdet_objective = tf.reduce_sum(logdet_train)
        pullback_train = tape.gradient(pullback_objective, z_arg)
        logdet_score_train = tape.gradient(logdet_objective, z_arg)
        theta_frozen = loaded.transport.forward_batch(z_arg)
        logdet_frozen = loaded.transport.log_abs_det_jacobian_batch(z_arg)
        pullback_frozen = loaded.transport.pullback_score_batch(z_arg, score_arg)
        logdet_score_frozen = loaded.transport.log_abs_det_jacobian_score_batch(z_arg)
        return (
            tf.reduce_max(tf.abs(theta_train - theta_frozen)),
            tf.reduce_max(tf.abs(logdet_train - logdet_frozen)),
            tf.reduce_max(tf.abs(pullback_train - pullback_frozen)),
            tf.reduce_max(tf.abs(logdet_score_train - logdet_score_frozen)),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(probes, theta_score)
    gaps = tuple(float(item.numpy()) for item in outputs)
    passed = all(item <= 1.0e-10 for item in gaps) and all(
        "GPU" in str(item.device).upper() for item in outputs
    )
    if not passed:
        raise StructuralUKFTrainingError("frozen/trainable parity failed")
    return {
        "passed": True,
        "transport_max_abs": gaps[0],
        "logdet_max_abs": gaps[1],
        "pullback_score_max_abs": gaps[2],
        "logdet_score_max_abs": gaps[3],
        "output_devices": tuple(str(item.device) for item in outputs),
        "jit_compile": True,
    }


def heldout_diagnostics(
    tf: Any, *, adapter: Any, loaded: Any
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter

    learned = FixedTransportValueScoreAdapter(
        base_adapter=adapter,
        transport=loaded.transport,
        target_scope="structural_ukf_neutra_training_heldout",
        evidence_path=str(PLAN_PATH),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(seed: Any):
        z = tf.random.stateless_normal(
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE, DIMENSION),
            seed=seed,
            dtype=tf.float64,
        )
        flat = tf.reshape(z, (-1, DIMENSION))
        learned_value, learned_score = learned.log_prob_and_grad_batch(flat)
        learned_status = learned.target_status_telemetry(flat)
        baseline_value, baseline_score = adapter.log_prob_and_grad(flat)
        baseline_status = adapter.target_status_telemetry(flat)
        learned_objective = tf.reshape(
            -learned_value, (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE)
        )
        baseline_objective = tf.reshape(
            -baseline_value, (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE)
        )
        learned_force = tf.reshape(
            tf.linalg.norm(learned_score, axis=-1),
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE),
        )
        baseline_force = tf.reshape(
            tf.linalg.norm(baseline_score, axis=-1),
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE),
        )
        return (
            tf.reduce_mean(learned_objective, axis=1),
            tf.math.reduce_std(learned_objective, axis=1),
            tf.reduce_mean(baseline_objective, axis=1),
            tf.math.reduce_std(baseline_objective, axis=1),
            tf.reduce_mean(learned_force, axis=1),
            tf.reduce_max(learned_force, axis=1),
            tf.reduce_mean(baseline_force, axis=1),
            tf.reduce_max(baseline_force, axis=1),
            tf.reduce_all(tf.equal(learned_status["status_code"], 0)),
            tf.reduce_all(learned_status["valid_pre_regularized_score"]),
            tf.reduce_all(tf.equal(baseline_status["status_code"], 0)),
            tf.reduce_all(baseline_status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(tf.constant(HELDOUT_SEED, tf.int32))
    if not all(bool(outputs[index].numpy()) for index in range(8, 12)):
        raise StructuralUKFTrainingError("heldout target status failed")
    learned_means = tuple(float(item) for item in outputs[0].numpy().tolist())
    baseline_means = tuple(float(item) for item in outputs[2].numpy().tolist())
    if not all(math.isfinite(item) for item in (*learned_means, *baseline_means)):
        raise StructuralUKFTrainingError("heldout objective is non-finite")
    differences = tuple(
        left - right for left, right in zip(learned_means, baseline_means)
    )
    return {
        "batch_count": HELDOUT_BATCH_COUNT,
        "batch_size": HELDOUT_BATCH_SIZE,
        "root_seed": HELDOUT_SEED,
        "common_seed_policy": "identical_stateless_8x128x5_base_tensor_for_all_recipes_and_identity_baseline",
        "learned_reverse_kl_means": learned_means,
        "learned_reverse_kl_sds": tuple(float(item) for item in outputs[1].numpy().tolist()),
        "identity_baseline_reverse_kl_means": baseline_means,
        "identity_baseline_reverse_kl_sds": tuple(float(item) for item in outputs[3].numpy().tolist()),
        "learned_force_means": tuple(float(item) for item in outputs[4].numpy().tolist()),
        "learned_force_maxima": tuple(float(item) for item in outputs[5].numpy().tolist()),
        "identity_baseline_force_means": tuple(float(item) for item in outputs[6].numpy().tolist()),
        "identity_baseline_force_maxima": tuple(float(item) for item in outputs[7].numpy().tolist()),
        "mean_learned_reverse_kl": common._mean(learned_means),
        "mean_identity_baseline_reverse_kl": common._mean(baseline_means),
        "paired_mean_difference_learned_minus_identity": common._mean(differences),
        "paired_difference_mcse": common._mcse(differences),
        "target_status_all_valid": True,
        "single_compiled_heldout_invocation": True,
        "output_devices": tuple(str(item.device) for item in outputs),
        "metric_role": "proxy_nomination_only_not_transport_promotion_or_veto",
    }


def select_screen_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if tuple(row.get("recipe_id") for row in rows) != RECIPE_ORDER:
        raise StructuralUKFTrainingError("screen rows do not match recipe order")
    learned = {
        str(row["recipe_id"]): tuple(
            float(item) for item in row["heldout_common_batches"]["learned_reverse_kl_means"]
        )
        for row in rows
    }
    if any(not all(math.isfinite(item) for item in values) for values in learned.values()):
        raise StructuralUKFTrainingError("screen heldout loss is non-finite")
    nominal = min(RECIPE_ORDER, key=lambda item: common._mean(learned[item]))
    viable = []
    comparison_rows = []
    for recipe_id in RECIPE_ORDER:
        differences = tuple(
            left - right for left, right in zip(learned[recipe_id], learned[nominal])
        )
        difference = common._mean(differences)
        mcse = common._mcse(differences)
        within = bool(difference <= 2.0 * mcse)
        if within:
            viable.append(recipe_id)
        comparison_rows.append(
            {
                "recipe_id": recipe_id,
                "mean_reverse_kl": common._mean(learned[recipe_id]),
                "paired_mean_difference_from_nominal": difference,
                "paired_difference_mcse": mcse,
                "within_two_paired_mcse": within,
            }
        )
    selected = min(
        viable,
        key=lambda item: (
            parameter_count(RECIPES[item]),
            RECIPES[item].learning_rate,
            RECIPE_ORDER.index(item),
        ),
    )
    return {
        "passed": True,
        "decision": "NOMINATE_STRUCTURAL_UKF_RECIPE_FOR_FRESH_5000_STEP_TRAINING",
        "selected_recipe_id": selected,
        "nominal_lowest_mean_recipe": nominal,
        "comparison_rows": comparison_rows,
    }


def finalize_screen(*, output_path: Path) -> Mapping[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"selection path already exists: {output_path}")
    rows = []
    references = []
    for recipe_id in RECIPE_ORDER:
        root = default_job_root("screen", recipe_id)
        result_path = root / "result.json"
        common._verify_result_root(
            root, common._file_sha256(result_path), require_passed=True
        )
        row = common._read_mapping(result_path)
        if (
            row.get("cell_id") != CELL_ID
            or row.get("job_kind") != "screen"
            or row.get("recipe_id") != recipe_id
            or common._json_ready(row.get("recipe"))
            != common._json_ready(RECIPES[recipe_id].payload())
        ):
            raise StructuralUKFTrainingError(f"invalid screen row: {recipe_id}")
        rows.append(row)
        references.append(
            {
                "recipe_id": recipe_id,
                "result": common._file_reference(result_path),
                "artifact_hashes": common._file_reference(root / "artifact_hashes.json"),
            }
        )
    selection = select_screen_rows(rows)
    selected = selection["selected_recipe_id"]
    result = {
        "schema": "bayesfilter.structural_ukf_neutra_training_selection.v1",
        "campaign_id": CAMPAIGN_ID,
        "cell_id": CELL_ID,
        **selection,
        "recipe_order": RECIPE_ORDER,
        "selected_recipe": RECIPES[str(selected)].payload(),
        "screen_references": references,
        "selection_rule": "within_two_paired_mcse_of_nominal_lowest_common_heldout_loss_then_parameter_count_learning_rate_declared_order",
        "statistically_supported_ranking": False,
        "screen_weights_reused_by_final": False,
        "evidence_role": "proxy_nomination_not_transport_promotion",
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_path, result)
    return result


def load_selection() -> Mapping[str, Any]:
    path = selection_path()
    selection = common._read_mapping(path)
    if (
        selection.get("schema")
        != "bayesfilter.structural_ukf_neutra_training_selection.v1"
        or selection.get("cell_id") != CELL_ID
        or selection.get("passed") is not True
        or selection.get("selected_recipe_id") not in RECIPES
    ):
        raise StructuralUKFTrainingError("invalid structural training selection")
    return {**selection, "path": str(path), "sha256": common._file_sha256(path)}


def run_manifest(
    *,
    output_root: Path,
    job_kind: str,
    recipe_id: str,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    wall_time: float,
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.structural_ukf_neutra_training_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "cell_id": CELL_ID,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped new paths only",
        "command": " ".join(sys.argv),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "target_signature": target_signature,
        "random_seed": SCREEN_SEEDS[recipe_id] if job_kind == "screen" else FINAL_SEED,
        "heldout_seed": HELDOUT_SEED,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def default_job_root(job_kind: str, recipe_id: str) -> Path:
    stage = "screen/candidates" if job_kind == "screen" else "final"
    attempt = "attempt-01" if job_kind == "screen" else "attempt-02/segment-5000"
    return CAMPAIGN_ROOT / "training" / stage / recipe_id / attempt


def selection_path() -> Path:
    return CAMPAIGN_ROOT / "training/screen/selection.json"


def write_recursive_hashes(root: Path) -> None:
    hashes = {
        str(path.relative_to(root)): common._file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    common._write_new_json(
        root / "artifact_hashes.json",
        {"schema": "bayesfilter.structural_ukf_neutra_training_hashes.v1", "artifacts": hashes},
    )


def parameter_count(recipe: Recipe) -> int:
    sizes = (DIMENSION, *recipe.hidden_layers, 2 * DIMENSION)
    per_stage = sum(
        left * right + right for left, right in zip(sizes[:-1], sizes[1:])
    )
    return 3 * per_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("canary", "run", "finalize"), required=True)
    parser.add_argument("--job-kind", choices=("screen", "final"))
    parser.add_argument("--recipe", choices=RECIPE_ORDER)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume-infrastructure-from", type=Path)
    args = parser.parse_args(argv)
    if args.action == "canary":
        result = run_canary(
            output_root=args.output_root or CAMPAIGN_ROOT / "canary/attempt-01"
        )
    elif args.action == "finalize":
        result = finalize_screen(output_path=selection_path())
    else:
        if args.job_kind is None or args.recipe is None:
            parser.error("run requires --job-kind and --recipe")
        result = run_training_job(
            job_kind=args.job_kind,
            recipe_id=args.recipe,
            output_root=args.output_root or default_job_root(args.job_kind, args.recipe),
            stop_after_steps=args.stop_after_steps,
            resume_infrastructure_from=args.resume_infrastructure_from,
        )
    print(
        json.dumps(
            {
                "cell_id": CELL_ID,
                "passed": result["passed"],
                "decision": result["decision"],
                "recipe_id": result.get("recipe_id", result.get("selected_recipe_id")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
