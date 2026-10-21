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


def m13_scope(*, refined=False, bounded=False):
    if refined and bounded:
        raise ValueError('choose one M13 control family')
    attempt = 3 if bounded else 2 if refined else 1
    partitions = {p: (seed+(attempt-1 if p != "claim" else 0), part+(3*(attempt-1) if p != "claim" else 0))
                  for p, (seed, part) in PARTITIONS.items()}
    return dict(model="M13", attempt=attempt, horizon=20, state_dimension=4, observation_dimension=2,
        parameter_dimension=2, particle_count=PARTICLE_COUNT, chunk_extent=PARTICLE_COUNT,
        dtype="float32", tf32=True, jit_compile=True, route="canonical_batch_fused_value_score",
        reset_contract_id="contract_e_chol_v1", fixture="range_bearing_gaussian_tf_tfp_moderate",
        parameters=["lambda_Q", "lambda_R"], noise_law="Q=exp(2*lambda_Q)*Q0; R=exp(2*lambda_R)*R0",
        prior="independent N(0,log(2)^2)", prior_support="R^2", fixed_initial_law=True,
        draw_design="scrambled_halton_gaussian", design_seed=20260923,
        partition_seeds={name: dict(data_seed=seed, design_partition=part)
                         for name, (seed, part) in partitions.items()},
        data_regime="independent T20 calibration/validation/claim; once-rounded FP32 observations",
        control_family=("m13_fixed_reset_caps_lm_curve_v1" if bounded else
                        "m13_fixed_reset_geometric_protection_curve_v1" if refined else
                        "full_canonical_geometric_protection_curve_v1"))


def partition_fixture(partition, *, attempt=1):
    """Generate the unchanged fixture law with independent, frozen data streams."""
    seed, _ = PARTITIONS[partition]
    seed += attempt-1 if partition != "claim" else 0
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


def controls_stage(output, *, refined=False, bounded=False):
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("M13 controls require the declared TF32 execution scope")
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    scope, points = m13_scope(refined=refined, bounded=bounded), control_points()
    settings = fixed_protection_controls if bounded else fixed_reset_controls if refined else protection_controls
    curves, paths, sources = {}, {}, None
    no_unflagged_failures = True
    for partition, spec in scope["partition_seeds"].items():
        design_partition = spec["design_partition"]
        directory = out/partition
        directory.mkdir()
        fixture = partition_fixture(partition, attempt=scope["attempt"])
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
                controls=settings(multiplier), dtype=tf.float32, design_partition=design_partition)
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
                controls=settings(multiplier), target=dict(target.manifest),
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
        if refined or bounded:
            refined_selection(selection_path, scope=scope, sources=sources, curves=curves, data_paths=paths)
        else:
            issue_selection(output=selection_path, scope=scope, evidence_paths=curves,
                            partition_data_paths=paths, source_hashes=sources)
    except ValueError as exc:
        if str(exc) != "NO_STABLE_CONTROL_PAIR":
            raise
        return dict(status="M13_CONTROL_REPAIR_REQUIRED", qualified=False, scope=scope,
            reason=str(exc), continuation_veto=False, posterior_comparisons="NOT_RUN",
            heuristic_dominance="NOT_EVALUATED", inference_launched=False, all_models_validated=False,
            next_action="Existing master: inspect conditional curves and repair controls within budget; prepare posterior reference.")
    selection = (refined_selection(selection_path, scope=scope, sources=sources) if refined or bounded else
                 load_selection(selection_path, expected_scope=scope, expected_source_hashes=sources))
    return dict(status="M13_SCOPED_PROPOSAL_CONTROLS_SELECTED", qualified=True, scope=scope,
        qualification_scope="numerical controls for imperfect proposal-field experiment only",
        selection=str(selection_path), selected_multiplier=selection["selected_multiplier"],
        claim_data_evaluated=False, posterior_comparisons="NOT_RUN", posterior_qualified=False,
        heuristic_dominance="NOT_EVALUATED", inference_launched=False, all_models_validated=False,
        next_action="Existing master: qualify M13 independent posterior reference and endpoint before public HMC tuning.")


def localization_candidates():
    strong, weak = protection_controls(1.), protection_controls(.001)
    groups = {
        "reset_ridge": ("reset_ridge",),
        "lm": ("correction_lm_damping", "correction_lm_scale_floor"),
        "trust_caps": ("correction_trust_radius", "pairwise_rms_cap", "coordinate_cap"),
    }
    return {"strong": strong, "weak": weak, **{
        "restore_"+name: weak | {key: strong[key] for key in keys}
        for name, keys in groups.items()}}


def localization_stage(output):
    """Identify the control group behind the observed reset failure; no selection."""
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("M13 localization requires the declared TF32 scope")
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    fixture, points = partition_fixture("validation"), control_points()
    scope = m13_scope()
    write_json(out/"data.json", dict(model=fixture.model_definition(),
        observations=fixture.observations.numpy().tolist(),
        partition="validation", attempt=1, model_id="M13", scope=scope))
    diagnostics, sources = {}, None
    for name, controls in localization_candidates().items():
        target = make_range_bearing_target(fixture, particle_count=PARTICLE_COUNT,
            controls=controls, dtype=tf.float32, design_partition=PARTITIONS["validation"][1])
        current = source_hashes(target)
        if sources is not None and sources != current:
            raise ValueError("source changed during localization")
        sources = current
        evaluate = target.compile(9, filter_moments=True, all_correction_diagnostics=True)
        begin = time.monotonic()
        result = {name: evaluate(theta) for name, theta in points.items()}
        result["stress"][0].numpy()
        situations = {kind: describe(value) for kind, value in result.items()}
        row = dict(controls=controls, target=dict(target.manifest), situations=situations,
            points={kind: theta.numpy().tolist() for kind, theta in points.items()},
            wall_seconds=time.monotonic()-begin)
        write_json(out/(name+".json"), finite_json(row))
        diagnostics[name] = {kind: dict(healthy_rows=d["healthy_rows"],
            all_healthy=d["all_healthy"], all_failures_flagged=d["all_failures_flagged"],
            values=d["values"], scores=d["scores"],
            first_failure_time=d["status"].get("first_failure_time"),
            first_failure_stage=d["status"].get("first_failure_stage"))
            for kind, d in situations.items()}
        print(name, {k: d["all_healthy"] for k, d in situations.items()}, row["wall_seconds"], flush=True)
        if not all(d["all_failures_flagged"] for d in situations.values()):
            raise ValueError("unflagged nonfinite localization output")
    return dict(status="M13_CONTROL_GROUP_LOCALIZATION_COMPLETED", qualified=False,
        diagnostic_completed=True, scope=scope, source_hashes=sources, arms=finite_json(diagnostics),
        inference_launched=False, posterior_comparisons="NOT_RUN", all_models_validated=False,
        next_action="Existing master: use the measured group isolation to define a fresh scoped control curve; posterior reference remains pending.")


def fixed_reset_controls(multiplier):
    """Hold the measured reset repair fixed while calibrating other protections."""
    return protection_controls(multiplier) | {"reset_ridge": protection_controls(1.)["reset_ridge"]}



def fixed_protection_controls(multiplier):
    settings = protection_controls(1.)
    varied = protection_controls(multiplier)
    for key in ("correction_lm_damping", "correction_lm_scale_floor"):
        settings[key] = varied[key]
    return settings

def refined_selection(output, *, scope, sources, curves=None, data_paths=None):
    """Issue/load M13's fixed-reset family using measured independent curves."""
    import json
    from bayesfilter.inference.surrogate_hmc_dctr_selection import first_stable_pair
    path = Path(output)
    if scope == m13_scope(bounded=True):
        settings = fixed_protection_controls
    elif scope == m13_scope(refined=True):
        settings = fixed_reset_controls
    else:
        raise ValueError("unsupported M13 fixed-reset tuning scope")
    if curves is not None:
        if path.exists():
            raise FileExistsError("selection must not overwrite evidence")
        if set(curves) != {"calibration", "validation"} or set(data_paths) != set(curves):
            raise ValueError("independent calibration/validation required")
        rows = {p: json.loads(Path(f).read_text()) for p, f in curves.items()}
        for p, row in rows.items():
            data = json.loads(Path(data_paths[p]).read_text())
            if (row["scope"] != scope or row["source_hashes"] != sources
                or row["data_sha256"] != file_hash(data_paths[p])
                or data.get("attempt") != scope["attempt"] or data.get("partition") != p
                or data.get("model_id") != "M13"):
                raise ValueError("scope, source or partition mismatch")
        index = first_stable_pair({p: row["pairs"] for p, row in rows.items()})
        if index is None:
            raise ValueError("NO_STABLE_CONTROL_PAIR")
        write_json(path, dict(schema="bayesfilter.m13_fixed_reset_controls.v1",
            scope=scope, source_hashes=sources, issuer_sha256=file_hash(Path(__file__)),
            use="imperfect_proposal_field_experiment", exact_score_admission=False,
            default_readiness=False, selected_index=index, selected_multiplier=MULTIPLIERS[index],
            controls=settings(MULTIPLIERS[index]),
            evidence={p: dict(path=str(Path(f).resolve()), sha256=file_hash(f)) for p, f in curves.items()},
            partition_data={p: dict(path=str(Path(f).resolve()), sha256=file_hash(f)) for p, f in data_paths.items()}))
    result = json.loads(path.read_text())
    if (result.get("schema") != "bayesfilter.m13_fixed_reset_controls.v1"
        or result.get("scope") != scope or result.get("source_hashes") != sources
        or result.get("issuer_sha256") != file_hash(Path(__file__))
        or result.get("use") != "imperfect_proposal_field_experiment"
        or result.get("exact_score_admission") is not False or result.get("default_readiness") is not False):
        raise ValueError("invalid M13 fixed-reset control artifact")
    for group in ("evidence", "partition_data"):
        if set(result[group]) != {"calibration", "validation"}:
            raise ValueError("invalid selection partitions")
        for record in result[group].values():
            if file_hash(Path(record["path"])) != record["sha256"]:
                raise ValueError("selection evidence changed")
    rows = {p: json.loads(Path(r["path"]).read_text()) for p, r in result["evidence"].items()}
    index = first_stable_pair({p: row["pairs"] for p, row in rows.items()})
    if (index is None or result.get("selected_index") != index
        or result.get("selected_multiplier") != MULTIPLIERS[index]
        or result.get("controls") != settings(MULTIPLIERS[index])):
        raise ValueError("selection differs from the measured curve")
    return result
