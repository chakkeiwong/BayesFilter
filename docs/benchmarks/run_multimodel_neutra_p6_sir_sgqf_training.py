#!/usr/bin/env python3
"""Run and select P6 target-specific SIR-SGQF dense-IAF training jobs."""

from __future__ import annotations

import argparse
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
from docs.benchmarks import run_multimodel_neutra_p6_sir_sgqf_hmc as hmc


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "SIR-SGQF"
DIMENSION = 3
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-r3-sir-sgqf-training-subplan-2026-07-16.md"
)
PHASE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF"
)
COMPARATOR_ROOT = PHASE_ROOT / "plain-hmc-affine/attempt-02"
COMPARATOR_RESULT_SHA256 = "621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2"
SCREEN_STEPS = 500
FINAL_STEPS = 5000
BATCH_SIZE = 128
HELDOUT_BATCH_COUNT = 8
HELDOUT_BATCH_SIZE = 128
RECIPE_ORDER = (
    "dim3_lr1e3",
    "dim3_lr5e3",
    "wide_lr1e3",
    "wide_lr5e3",
)
SCREEN_SEEDS = {
    recipe: (20260716, 31001 + index) for index, recipe in enumerate(RECIPE_ORDER)
}
HELDOUT_SEED = (20260716, 31100)
FINAL_SEED = (20260716, 31201)
SCREEN_ATTEMPTS = {
    "dim3_lr1e3": "attempt-02",
    "dim3_lr5e3": "attempt-01",
    "wide_lr1e3": "attempt-01",
    "wide_lr5e3": "attempt-01",
}
NONCLAIMS = (
    "training and common-heldout reverse KL are nomination or veto evidence only",
    "screen weights are never reused by final training",
    "no transported HMC convergence or plain-HMC agreement claim",
    "no SGQF exactness, superiority, calibration, forecasting, robustness, or readiness claim",
)


class P6SIRTrainingError(RuntimeError):
    """Raised when a frozen P6 SIR training boundary fails closed."""


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
            "final_learning_rate_fraction": 1.0,
            "batch_size": BATCH_SIZE,
            "activation": "elu",
            "s_max": 1.0,
            "init_scale": 0.02,
            "clip_norm": 10.0,
            "optimizer": "manual_adam_constant_learning_rate",
        }


RECIPES = {
    "dim3_lr1e3": Recipe("dim3_lr1e3", (9, 9), 1.0e-3),
    "dim3_lr5e3": Recipe("dim3_lr5e3", (9, 9), 5.0e-3),
    "wide_lr1e3": Recipe("wide_lr1e3", (18, 18), 1.0e-3),
    "wide_lr5e3": Recipe("wide_lr5e3", (18, 18), 5.0e-3),
}


def run_training_job(
    *, job_kind: str, recipe_id: str, output_root: Path
) -> Mapping[str, Any]:
    if job_kind not in {"screen", "final"}:
        raise P6SIRTrainingError("job_kind must be screen or final")
    try:
        recipe = RECIPES[recipe_id]
    except KeyError as exc:
        raise P6SIRTrainingError(f"unknown training recipe: {recipe_id}") from exc
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
        train_campaign_neutra,
    )
    from bayesfilter.inference.neutra_training import (
        PlainDenseIAFTrainingConfig,
        restore_plain_dense_iaf_flow,
    )

    adapter, identity, identity_reference = _reconstruct_identity(tf)
    center, factor, geometry_reference = _load_geometry(tf)
    comparator_reference = common._verify_result_root(
        COMPARATOR_ROOT, COMPARATOR_RESULT_SHA256, require_passed=True
    )
    if job_kind == "final":
        selection = _load_selection()
        if selection["selected_recipe_id"] != recipe_id:
            raise P6SIRTrainingError("final recipe does not match frozen selection")
        seed = FINAL_SEED
        steps = FINAL_STEPS
    else:
        selection = None
        seed = SCREEN_SEEDS[recipe_id]
        steps = SCREEN_STEPS

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
        final_learning_rate_fraction=1.0,
        clip_norm=10.0,
        checkpoint_every=steps,
        heartbeat_every=10,
        jit_compile=True,
        device="/GPU:0",
        require_gpu=True,
    )
    trained = train_campaign_neutra(
        identity=identity,
        adapter=adapter,
        config=config,
        freeze_transport_id=f"p6-sir-sgqf-{job_kind}-{recipe_id}-{steps}",
        gpu_memory_policy=memory_policy,
    )
    if trained.completed_steps != steps or trained.frozen_payload_path is None:
        raise P6SIRTrainingError("training did not freeze at the terminal step")
    _require_training_runtime(trained, expected_steps=steps)
    payload = common._read_mapping(trained.frozen_payload_path)
    loaded = load_campaign_neutra_transport(
        identity=identity, adapter=adapter, payload=payload
    )
    flow = restore_plain_dense_iaf_flow(config=config, state_path=trained.state_path)
    parity = _compiled_parity(tf, flow, loaded, recipe_id=recipe_id)
    heldout = _heldout(
        tf,
        adapter=adapter,
        loaded=loaded,
        center=center,
        factor=factor,
    )
    final_admitted = bool(job_kind == "screen" or heldout["affine_nonworse"])
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_training_job.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "passed": final_admitted,
        "decision": (
            "PASS_P6_SIR_SGQF_TRAINING_SCREEN_JOB"
            if job_kind == "screen"
            else (
                "ADMIT_P6_SIR_SGQF_FRESH_5000_STEP_FROZEN_TRAINING"
                if final_admitted
                else "REJECT_P6_SIR_SGQF_FINAL_TRAINING_AFFINE_DEGRADATION"
            )
        ),
        "recipe": recipe.payload(),
        "seed": seed,
        "steps": steps,
        "screen_weights_reused_by_final": False,
        "target_identity": identity.payload(),
        "identity_reference": identity_reference,
        "comparator_reference": comparator_reference,
        "geometry_reference": geometry_reference,
        "selection_reference": selection,
        "training_state_hash": trained.state_hash,
        "transport_artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "payload": common._file_reference(trained.frozen_payload_path),
        "checkpoint": common._file_reference(trained.state_path),
        "progress": common._file_reference(trained.progress_path),
        "records": trained.records,
        "runtime_metadata": trained.runtime_metadata,
        "frozen_trainable_parity": parity,
        "heldout_common_batches": heldout,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_role": (
            "proxy_nomination_and_affine_veto_only"
            if job_kind == "screen"
            else "engineering_candidate_for_r4_neutra_hmc"
        ),
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_root / "result.json", result)
    common._write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            job_kind=job_kind,
            recipe_id=recipe_id,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            wall_time=time.monotonic() - started,
        ),
    )
    _write_recursive_hashes(output_root)
    return result


def select_screen_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if tuple(row.get("recipe_id") for row in rows) != RECIPE_ORDER:
        raise P6SIRTrainingError("screen rows must match the frozen recipe order")
    affine_vectors = tuple(
        tuple(float(item) for item in row["heldout_common_batches"]["affine_reverse_kl_means"])
        for row in rows
    )
    if any(vector != affine_vectors[0] for vector in affine_vectors[1:]):
        raise P6SIRTrainingError("common affine heldout control drifted across recipes")
    learned = {
        row["recipe_id"]: tuple(
            float(item)
            for item in row["heldout_common_batches"]["learned_reverse_kl_means"]
        )
        for row in rows
    }
    affine = affine_vectors[0]
    affine_viable = []
    affine_rows = []
    for recipe_id in RECIPE_ORDER:
        differences = tuple(left - right for left, right in zip(learned[recipe_id], affine))
        difference = common._mean(differences)
        mcse = common._mcse(differences)
        nonworse = bool(difference <= 2.0 * mcse)
        if nonworse:
            affine_viable.append(recipe_id)
        affine_rows.append(
            {
                "recipe_id": recipe_id,
                "paired_mean_difference_from_affine": difference,
                "paired_difference_mcse": mcse,
                "affine_nonworse": nonworse,
            }
        )
    if not affine_viable:
        return {
            "passed": False,
            "decision": "REJECT_P6_SIR_SGQF_PLAIN_DENSE_IAF_AFFINE_DEGRADATION",
            "selected_recipe_id": None,
            "nominal_lowest_mean_recipe": None,
            "affine_comparison_rows": affine_rows,
            "learned_comparison_rows": (),
        }

    nominal = min(affine_viable, key=lambda item: common._mean(learned[item]))
    learned_viable = []
    learned_rows = []
    for recipe_id in affine_viable:
        differences = tuple(
            left - right for left, right in zip(learned[recipe_id], learned[nominal])
        )
        difference = common._mean(differences)
        mcse = common._mcse(differences)
        within = bool(difference <= 2.0 * mcse)
        if within:
            learned_viable.append(recipe_id)
        learned_rows.append(
            {
                "recipe_id": recipe_id,
                "mean_reverse_kl": common._mean(learned[recipe_id]),
                "paired_mean_difference_from_nominal": difference,
                "paired_difference_mcse": mcse,
                "within_two_paired_mcse": within,
            }
        )
    selected = min(
        learned_viable,
        key=lambda item: (
            _parameter_count(RECIPES[item]),
            RECIPES[item].learning_rate,
            RECIPE_ORDER.index(item),
        ),
    )
    return {
        "passed": True,
        "decision": "NOMINATE_P6_SIR_SGQF_RECIPE_FOR_FRESH_5000_STEP_TRAINING",
        "selected_recipe_id": selected,
        "nominal_lowest_mean_recipe": nominal,
        "affine_comparison_rows": affine_rows,
        "learned_comparison_rows": learned_rows,
    }


def _require_training_runtime(trained: Any, *, expected_steps: int) -> None:
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
        "program_step_count": expected_steps,
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            raise P6SIRTrainingError(f"training runtime contract failed: {key}")
    batch_target = runtime.get("batch_native_target")
    if not isinstance(batch_target, Mapping):
        raise P6SIRTrainingError("training lacks batch-native target metadata")
    for key in (
        "scalar_fallback_used",
        "sample_axis_python_loop_used",
        "row_mapped_scalar_target_used",
    ):
        if batch_target.get(key) is not False:
            raise P6SIRTrainingError(f"batch-native target contract failed: {key}")
    device_fields = (
        "trainable_variable_devices",
        "adam_moment_devices",
        "compiled_output_devices",
    )
    for key in device_fields:
        devices = tuple(str(item) for item in runtime.get(key, ()))
        if not devices or not all("GPU" in item.upper() for item in devices):
            raise P6SIRTrainingError(f"training runtime is not GPU-only: {key}")
    if not trained.records or any(
        row.get("target_values_finite") is not True
        or row.get("target_status_available") is not True
        or row.get("target_status_all_valid") is not True
        or int(row.get("target_status_nonvalid_count", 1)) != 0
        for row in trained.records
    ):
        raise P6SIRTrainingError("training record target health failed")


def finalize_screen(*, output_path: Path) -> Mapping[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"screen selection already exists: {output_path}")
    rows = []
    references = []
    for recipe_id in RECIPE_ORDER:
        root = _default_job_root("screen", recipe_id)
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
            raise P6SIRTrainingError(f"invalid screen row: {recipe_id}")
        rows.append(row)
        references.append(
            {
                "recipe_id": recipe_id,
                "result": common._file_reference(result_path),
                "artifact_hashes": common._file_reference(root / "artifact_hashes.json"),
            }
        )
    selection = select_screen_rows(rows)
    selected_id = selection["selected_recipe_id"]
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_training_selection.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        **selection,
        "recipe_order": RECIPE_ORDER,
        "selected_recipe": None if selected_id is None else RECIPES[selected_id].payload(),
        "screen_references": references,
        "selection_rule": "affine_nonworse_within_two_paired_mcse_then_within_two_paired_mcse_of_lowest_learned_mean_then_parameter_count_learning_rate_declared_order",
        "statistically_supported_ranking": False,
        "screen_weights_reused_by_final": False,
        "evidence_role": "proxy_nomination_and_affine_veto_not_transport_promotion",
        "nonclaims": NONCLAIMS,
    }
    common._write_new_json(output_path, result)
    return result


def _reconstruct_identity(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_campaign import (
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIRSGQFLikelihoodRecomposer,
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
        sir_identity_chart_jacobian_value_score,
        sir_prior_value_score,
    )

    reference = hmc._verify_root(hmc.IDENTITY_ROOT, hmc.IDENTITY_RESULT_SHA256)
    expected = common._read_mapping(hmc.IDENTITY_ROOT / "target_identity.json")
    registry = common._read_mapping(hmc.IDENTITY_ROOT / "repaired_registry.json")
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    audit_points = tf.concat(
        [
            tf.zeros([1, DIMENSION], tf.float64),
            0.5 * tf.eye(DIMENSION, dtype=tf.float64),
            -0.5 * tf.eye(DIMENSION, dtype=tf.float64),
            tf.eye(DIMENSION, dtype=tf.float64),
            -tf.eye(DIMENSION, dtype=tf.float64),
        ],
        axis=0,
    )
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=audit_points,
        prior_value_score_fn=sir_prior_value_score,
        likelihood_value_score_fn=SIRSGQFLikelihoodRecomposer(adapter).__call__,
        jacobian_value_score_fn=sir_identity_chart_jacobian_value_score,
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
            hmc.IDENTITY_ROOT / "repaired_registry.json"
        ),
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if common._json_ready(identity.payload()) != expected:
        raise P6SIRTrainingError("SIR-SGQF training target identity drift")
    if identity.target_signature != hmc.TYPED_SIGNATURE:
        raise P6SIRTrainingError("SIR-SGQF typed target signature drift")
    return adapter, identity, reference


def _load_geometry(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    reference = hmc._verify_root(hmc.GEOMETRY_ROOT, hmc.GEOMETRY_RESULT_SHA256)
    result = common._read_mapping(hmc.GEOMETRY_ROOT / "result.json")
    if result.get("passed") is not True or result.get("geometry") is None:
        raise P6SIRTrainingError("SIR-SGQF geometry is not admitted")
    center = tf.constant(result["geometry"]["center"], tf.float64)
    factor = tf.constant(result["geometry"]["cholesky_factor"], tf.float64)
    return center, factor, reference


def _compiled_parity(
    tf: Any, flow: Any, loaded: Any, *, recipe_id: str
) -> Mapping[str, Any]:
    probes = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.1, 0.08]], tf.float64)
    score_seed = SCREEN_SEEDS[recipe_id]
    theta_score = tf.random.stateless_normal(
        tf.shape(probes),
        seed=(score_seed[0], score_seed[1] + 700),
        dtype=tf.float64,
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
        pullback_frozen = loaded.transport.pullback_score_batch(z_arg, theta_score_arg)
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
    if not all(item <= 1.0e-10 for item in gaps) or not all(
        "GPU" in str(item.device).upper() for item in outputs
    ):
        raise P6SIRTrainingError("frozen/trainable transport parity failed")
    return {
        "passed": True,
        "transport_max_abs": gaps[0],
        "logdet_max_abs": gaps[1],
        "pullback_score_max_abs": gaps[2],
        "logdet_score_max_abs": gaps[3],
        "output_devices": tuple(str(item.device) for item in outputs),
        "jit_compile": True,
    }


def _heldout(
    tf: Any,
    *,
    adapter: Any,
    loaded: Any,
    center: Any,
    factor: Any,
) -> Mapping[str, Any]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from docs.benchmarks.run_multimodel_neutra_p6_sir_sgqf_geometry import (
        AffineMassTargetAdapter,
    )

    learned = FixedTransportValueScoreAdapter(
        base_adapter=adapter,
        transport=loaded.transport,
        target_scope="p6_sir_sgqf_training_heldout",
        evidence_path=str(PLAN_PATH),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )
    affine = AffineMassTargetAdapter(
        base_adapter=adapter,
        center=center,
        factor=factor,
        target_signature=hmc.TYPED_SIGNATURE,
        mass_artifact_sha256=hmc.GEOMETRY_RESULT_SHA256,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(seed):
        z = tf.random.stateless_normal(
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE, DIMENSION),
            seed=seed,
            dtype=tf.float64,
        )
        flat = tf.reshape(z, (-1, DIMENSION))
        learned_value, learned_score = learned.log_prob_and_grad_batch(flat)
        learned_status = learned.target_status_telemetry(flat)
        affine_value, affine_score = affine.log_prob_and_grad(flat)
        affine_status = affine.target_status_telemetry(flat)
        learned_objective = tf.reshape(
            -learned_value, (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE)
        )
        affine_objective = tf.reshape(
            -affine_value, (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE)
        )
        learned_force = tf.reshape(
            tf.linalg.norm(learned_score, axis=-1),
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE),
        )
        affine_force = tf.reshape(
            tf.linalg.norm(affine_score, axis=-1),
            (HELDOUT_BATCH_COUNT, HELDOUT_BATCH_SIZE),
        )
        return (
            tf.reduce_mean(learned_objective, axis=1),
            tf.math.reduce_std(learned_objective, axis=1),
            tf.reduce_mean(affine_objective, axis=1),
            tf.math.reduce_std(affine_objective, axis=1),
            tf.reduce_mean(learned_force, axis=1),
            tf.reduce_max(learned_force, axis=1),
            tf.reduce_mean(affine_force, axis=1),
            tf.reduce_max(affine_force, axis=1),
            tf.reduce_all(tf.equal(learned_status["status_code"], 0)),
            tf.reduce_all(learned_status["valid_pre_regularized_score"]),
            tf.reduce_all(tf.equal(affine_status["status_code"], 0)),
            tf.reduce_all(affine_status["valid_pre_regularized_score"]),
        )

    with tf.device("/GPU:0"):
        outputs = compiled(tf.constant(HELDOUT_SEED, tf.int32))
    if not all(bool(outputs[index].numpy()) for index in range(8, 12)):
        raise P6SIRTrainingError("common-heldout target status failed")
    learned_means = tuple(float(item) for item in outputs[0].numpy().tolist())
    affine_means = tuple(float(item) for item in outputs[2].numpy().tolist())
    if not all(math.isfinite(item) for item in (*learned_means, *affine_means)):
        raise P6SIRTrainingError("common-heldout objective is nonfinite")
    differences = tuple(left - right for left, right in zip(learned_means, affine_means))
    paired_difference = common._mean(differences)
    paired_mcse = common._mcse(differences)
    return {
        "batch_count": HELDOUT_BATCH_COUNT,
        "batch_size": HELDOUT_BATCH_SIZE,
        "root_seed": HELDOUT_SEED,
        "common_seed_policy": "one_identical_stateless_8x128x3_base_tensor_for_affine_and_all_recipes",
        "learned_reverse_kl_means": learned_means,
        "learned_reverse_kl_sds": tuple(float(item) for item in outputs[1].numpy().tolist()),
        "affine_reverse_kl_means": affine_means,
        "affine_reverse_kl_sds": tuple(float(item) for item in outputs[3].numpy().tolist()),
        "learned_force_means": tuple(float(item) for item in outputs[4].numpy().tolist()),
        "learned_force_maxima": tuple(float(item) for item in outputs[5].numpy().tolist()),
        "affine_force_means": tuple(float(item) for item in outputs[6].numpy().tolist()),
        "affine_force_maxima": tuple(float(item) for item in outputs[7].numpy().tolist()),
        "mean_learned_reverse_kl": common._mean(learned_means),
        "mean_affine_reverse_kl": common._mean(affine_means),
        "paired_mean_difference_learned_minus_affine": paired_difference,
        "paired_difference_mcse": paired_mcse,
        "affine_nonworse": bool(paired_difference <= 2.0 * paired_mcse),
        "target_status_all_valid": True,
        "single_compiled_heldout_invocation": True,
        "output_devices": tuple(str(item.device) for item in outputs),
        "metric_role": "proxy_nomination_and_affine_veto_not_transport_promotion",
    }


def _load_selection() -> Mapping[str, Any]:
    path = _selection_path()
    selection = common._read_mapping(path)
    if (
        selection.get("schema")
        != "bayesfilter.multimodel_neutra_p6_sir_sgqf_training_selection.v1"
        or selection.get("cell_id") != CELL_ID
        or selection.get("passed") is not True
        or selection.get("selected_recipe_id") not in RECIPES
    ):
        raise P6SIRTrainingError("invalid frozen SIR-SGQF training selection")
    return {**selection, "path": str(path), "sha256": common._file_sha256(path)}


def _run_manifest(
    *,
    output_root: Path,
    job_kind: str,
    recipe_id: str,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    wall_time: float,
) -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_training_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "job_kind": job_kind,
        "recipe_id": recipe_id,
        "git_commit": commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            f"{sys.executable} {Path(__file__)} --action run --job-kind {job_kind} "
            f"--recipe {recipe_id} --output-root {output_root}"
        ),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
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


def _default_job_root(job_kind: str, recipe_id: str) -> Path:
    stage = "screen/candidates" if job_kind == "screen" else "final"
    attempt = SCREEN_ATTEMPTS[recipe_id] if job_kind == "screen" else "attempt-01"
    return PHASE_ROOT / "training" / stage / recipe_id / attempt


def _selection_path() -> Path:
    return PHASE_ROOT / "training/screen/selection.json"


def _write_recursive_hashes(root: Path) -> None:
    hashes = {
        str(path.relative_to(root)): common._file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    common._write_new_json(
        root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_training_hashes.v1",
            "artifacts": hashes,
        },
    )


def _parameter_count(recipe: Recipe) -> int:
    sizes = (DIMENSION, *recipe.hidden_layers, 2 * DIMENSION)
    per_stage = sum(
        left * right + right for left, right in zip(sizes[:-1], sizes[1:])
    )
    return 3 * per_stage


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("run", "finalize"), required=True)
    parser.add_argument("--job-kind", choices=("screen", "final"))
    parser.add_argument("--recipe", choices=RECIPE_ORDER)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.action == "finalize":
        result = finalize_screen(output_path=_selection_path())
    else:
        if args.job_kind is None or args.recipe is None:
            parser.error("run requires --job-kind and --recipe")
        result = run_training_job(
            job_kind=args.job_kind,
            recipe_id=args.recipe,
            output_root=args.output_root
            or _default_job_root(args.job_kind, args.recipe),
        )
    print(
        json.dumps(
            {
                "cell_id": CELL_ID,
                "passed": result["passed"],
                "decision": result["decision"],
                "recipe_id": result.get(
                    "recipe_id", result.get("selected_recipe_id")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
