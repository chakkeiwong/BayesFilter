"""M13 scoped numerical-control diagnostics, not posterior admission.

The force is the existing canonical TensorFlow implementation. Only diagnostic
comparisons and scope-bound offline selection are performed here.
"""
from dataclasses import replace
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
from bayesfilter.inference.surrogate_hmc_dctr_selection import (
    MULTIPLIERS, issue_selection, load_selection, protection_controls,
)
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_controls import (
    describe, finite_json, normalized_changes,
)
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
    _simulate_range_bearing, build_range_bearing_fixture_tf, stable_digest,
)

PLAN = "docs/plans/surrogate-hmc-m13-controls-2026-09-24.md"
PARTICLE_COUNT = 256
PARTITIONS = {"calibration": (1701, 0), "validation": (2701, 1), "claim": (3701, 2)}


def m13_scope():
    return dict(model="M13", attempt=1, horizon=20, state_dimension=4, observation_dimension=2,
        parameter_dimension=2, particle_count=PARTICLE_COUNT, chunk_extent=PARTICLE_COUNT,
        dtype="float32", tf32=True, jit_compile=True, route="canonical_batch_fused_value_score",
        reset_contract_id="contract_e_chol_v1", fixture="range_bearing_gaussian_tf_tfp_moderate",
        parameters=["lambda_Q", "lambda_R"], noise_law="Q=exp(2*lambda_Q)*Q0; R=exp(2*lambda_R)*R0",
        prior="independent N(0,log(2)^2)", prior_support="R^2", fixed_initial_law=True,
        draw_design="scrambled_halton_gaussian", design_seed=20260923,
        partition_seeds={name: dict(data_seed=seed, design_partition=part)
                         for name, (seed, part) in PARTITIONS.items()},
        data_regime="independent T20 calibration/validation/claim; once-rounded FP32 observations",
        control_family="full_canonical_geometric_protection_curve_v1")


def partition_fixture(partition):
    """Generate the unchanged fixture law with independent, frozen data streams."""
    seed, _ = PARTITIONS[partition]
    with tf.device("/CPU:0"):
        base = build_range_bearing_fixture_tf()
        states, observations = _simulate_range_bearing(A=base.A, Q=base.Q, R=base.R,
            m0=base.m0, P0=base.P0, horizon=base.horizon, seed=seed)
        observations = tf.cast(tf.cast(observations, tf.float32), tf.float64)
        model = {key: base.model_definition()[key] for key in ("A", "Q", "R", "m0", "P0", "horizon")}
        model.update(name=base.name, seed=seed, backend="tensorflow_tensorflow_probability",
                     observation_rounding="once_float32")
        data = dict(states=states.numpy().tolist(), observations=observations.numpy().tolist())
    return replace(base, states=states, observations=observations, fixture_generation_seed=seed,
                   model_checksum=stable_digest(model), observation_checksum=stable_digest(data))


def control_points():
    sd = math.log(2.)
    return {name: tf.constant([[i*scale*sd, j*scale*sd] for i in (-1, 0, 1) for j in (-1, 0, 1)], tf.float32)
            for name, scale in (("nominal", 1.), ("stress", 3.))}


def source_hashes(target):
    sources = dict(target.manifest["source_sha256"])
    for name in ("bayesfilter/testing/surrogate_hmc_m13_controls.py",
                 "bayesfilter/testing/surrogate_hmc_ar2_controls.py",
                 "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
                 "bayesfilter/inference/surrogate_hmc_dctr_selection.py"):
        sources[name] = file_hash(ROOT/name)
    return sources


def compare_pair(stronger, weaker):
    """Preserve numerical invalidity and assess non-harm, not posterior error."""
    diagnostics = [{name: describe(value) for name, value in arm.items()} for arm in (stronger, weaker)]
    changes = {name: normalized_changes(stronger[name], weaker[name]) for name in stronger}
    maximum = max(changes["nominal"].values())
    healthy = all(row["nominal"]["all_healthy"] for row in diagnostics)
    safe = all(row["stress"]["all_failures_flagged"] for row in diagnostics)
    strong, weak = (row["stress"]["healthy_rows"] for row in diagnostics)
    no_new_failure = all(not b or a for a, b in zip(strong, weak))
    return dict(healthy_nominal=healthy, stress_safe=safe, no_new_stress_failure=no_new_failure,
        normalized_maximum_change=maximum, tolerance=1e-3, component_changes=changes,
        non_harm_passed=healthy and safe and no_new_failure and math.isfinite(maximum) and maximum <= 1e-3)


def controls_stage(output):
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("M13 controls require the declared TF32 execution scope")
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    scope, points = m13_scope(), control_points()
    curves, paths, sources = {}, {}, None
    no_unflagged_failures = True
    for partition, (_, design_partition) in PARTITIONS.items():
        directory = out/partition
        directory.mkdir()
        fixture = partition_fixture(partition)
        data_path = directory/"data.json"
        write_json(data_path, dict(partition=partition, attempt=scope["attempt"], model_id=scope["model"],
            model=fixture.model_definition(),
            scope=scope, observations=fixture.observations.numpy().tolist(),
            states=fixture.states.numpy().tolist(), design_partition=design_partition))
        if partition == "claim":
            continue
        paths[partition] = data_path
        outputs = []
        arm_paths = []
        for index, multiplier in enumerate(MULTIPLIERS):
            target = make_range_bearing_target(fixture, particle_count=PARTICLE_COUNT,
                controls=protection_controls(multiplier), dtype=tf.float32, design_partition=design_partition)
            current_sources = source_hashes(target)
            if sources is not None and sources != current_sources:
                raise ValueError("source changed during M13 control curve")
            sources = current_sources
            evaluate = target.compile(9, filter_moments=True, all_correction_diagnostics=True)
            begin = time.monotonic()
            result = {name: evaluate(theta) for name, theta in points.items()}
            result["stress"][0].numpy()
            diagnostics = {name: describe(value) for name, value in result.items()}
            no_unflagged_failures &= all(row["all_failures_flagged"] for row in diagnostics.values())
            arm_path = directory/f"arm-{index}.json"
            write_json(arm_path, finite_json(dict(multiplier=multiplier,
                controls=protection_controls(multiplier), target=dict(target.manifest),
                points={name: theta.numpy().tolist() for name, theta in points.items()},
                situations=diagnostics, wall_seconds=time.monotonic()-begin)))
            print(partition, multiplier, {name: row["all_healthy"] for name, row in diagnostics.items()},
                  time.monotonic()-begin, flush=True)
            outputs.append(result)
            arm_paths.append(arm_path)
            if not no_unflagged_failures:
                raise ValueError("unflagged invalid M13 force or moment output")
        pairs = [compare_pair(outputs[i], outputs[i+1]) for i in range(3)]
        curves[partition] = directory/"curve.json"
        write_json(curves[partition], finite_json(dict(scope=scope, source_hashes=sources,
            data_sha256=file_hash(data_path), pairs=pairs,
            arms=[dict(path=str(path), sha256=file_hash(path)) for path in arm_paths])))
    selection_path = out/"selection.json"
    try:
        issue_selection(output=selection_path, scope=scope, evidence_paths=curves,
                        partition_data_paths=paths, source_hashes=sources)
    except ValueError as exc:
        if str(exc) != "NO_STABLE_CONTROL_PAIR":
            raise
        return dict(status="M13_CONTROL_REPAIR_REQUIRED", qualified=False, scope=scope,
            reason=str(exc), continuation_veto=False, posterior_comparisons="NOT_RUN",
            heuristic_dominance="NOT_EVALUATED", inference_launched=False, all_models_validated=False,
            next_action="Existing master: inspect conditional curves and repair controls within budget; prepare posterior reference.")
    selection = load_selection(selection_path, expected_scope=scope, expected_source_hashes=sources)
    return dict(status="M13_SCOPED_PROPOSAL_CONTROLS_SELECTED", qualified=True, scope=scope,
        qualification_scope="numerical controls for imperfect proposal-field experiment only",
        selection=str(selection_path), selected_multiplier=selection["selected_multiplier"],
        claim_data_evaluated=False, posterior_comparisons="NOT_RUN", posterior_qualified=False,
        heuristic_dominance="NOT_EVALUATED", inference_launched=False, all_models_validated=False,
        next_action="Existing master: qualify M13 independent posterior reference and endpoint before public HMC tuning.")
