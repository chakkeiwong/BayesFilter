"""Offline diagnostic tuning for the five-model surrogate-HMC campaign.

The shared analytical filter supplies calibration histories and the proposal
force. This module issues exploratory control selections, never canonical LEDH
admission. FP64 is a same-algorithm precision reference, not a model oracle.
"""
from dataclasses import asdict, replace
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.analytical_jet_tf import Jet, stack
from bayesfilter.highdim.covariance_proposal_tf import TRACE_FIELDS
from bayesfilter.highdim.covariance_proposal_score_tf import (
    Models, build_maps, draw_from_maps, component_densities, observation_density)
from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter, precision_gaussian_mixture
from bayesfilter.highdim.covariance_proposal_beta_tf import (
    make_beta_solver, pilot_ratios, pilot_objective)

CALIBRATION_SEEDS = (26101211, 26101212)
VALIDATION_SEEDS = (26101213, 26101214)


def make_context(spec, theta, controls, n, precision_policy="uniform"):
    """Compile pilot densities using the force's actual proposal primitives."""
    dtype = theta.dtype
    equal = tf.fill([3], tf.constant(1./3, dtype))
    d, p = spec.dimension, spec.parameter_count
    o = getattr(spec, "observation_dimension", d)
    @tf.function(input_signature=[
        tf.TensorSpec([n, d], dtype), tf.TensorSpec([o], dtype),
        tf.TensorSpec([n, d], dtype), tf.TensorSpec([n, 2], dtype)],
        jit_compile=True, autograph=False)
    def context(ancestors, observation, noise, uniforms):
        models = Models(spec, theta, tf.zeros([1, p], dtype))
        maps = build_maps(models, Jet(ancestors, tf.zeros([1, n, d], dtype)),
                          observation, controls)
        x = draw_from_maps(maps, equal, uniforms, noise)
        if precision_policy == "uniform":
            densities, valid = component_densities(
                x, maps[0], maps[1], shared_flow_covariance=models.constant_observation_jacobian,
                contraction=controls.shared_gaussian_contraction)
        else:
            parts = [precision_gaussian_mixture(
                stack([x]), stack([maps[0][j]]), stack([maps[1][j]]),
                shared_covariance=j<2 or models.constant_observation_jacobian,
                contraction=controls.shared_gaussian_contraction,
                precision_policy=precision_policy) for j in range(3)]
            densities = stack([part[0][0] for part in parts], -1)
            valid = tf.reduce_all(tf.stack([part[1] for part in parts]))
        log_target = observation_density(models, x, observation)+densities[:, 0]
        return densities.v, log_target.v, valid & maps[6]
    return context


def precision_screen(a, b, horizon, candidate_precision_policy="uniform"):
    """Fail closed on final-output precision or either execution's health."""
    a_values = tf.concat([tf.cast(a["value"][:, None], tf.float64),
                          tf.cast(a["score"], tf.float64)], axis=1)
    b_values = tf.concat([b["value"][:, None], b["score"]], axis=1)
    errors = tf.abs(a_values-b_values)
    scaled = errors/tf.maximum(tf.constant(1., tf.float64), tf.abs(b_values))
    healthy = (tf.reduce_all(a["valid"]) & tf.reduce_all(b["valid"])
               & tf.reduce_all(a["steps_completed"] == horizon)
               & tf.reduce_all(b["steps_completed"] == horizon)
               & tf.reduce_all(tf.math.is_finite(a_values))
               & tf.reduce_all(tf.math.is_finite(b_values)))
    passed = bool(healthy & tf.reduce_all(scaled <= .01))
    return dict(passed=passed, healthy=bool(healthy),
                candidate_values_and_scores=a_values, fp64_values_and_scores=b_values,
                candidate_precision_policy=candidate_precision_policy,
                reference_precision_policy="uniform", reference_dtype="float64",
                absolute_errors=errors, scaled_errors=scaled,
                tolerance=.01, scale="max(1,abs(FP64 component))",
                role="numerical veto; same-algorithm reference, not model oracle")



def trajectory_diagnostics(run):
    """Preserve already-computed health diagnostics outside the hot graph."""
    report = dict(valid=run["valid"], steps_completed=run["steps_completed"])
    for name in ("ess", "max_weight", "reset_mean_error",
                 "reset_covariance_relative_error", "reset_margin",
                 "sinkhorn_row_error", "sinkhorn_column_error", "reset_displacement",
                 "correction_displacement", "correction_coordinate_displacement"):
        values = run["trace"][:, :, TRACE_FIELDS.index(name)]
        finite = tf.math.is_finite(values)
        positive = tf.constant(float("inf"), values.dtype)
        negative = -positive
        report[name] = dict(finite_count=tf.reduce_sum(tf.cast(finite, tf.int32), axis=1),
            minimum=tf.reduce_min(tf.where(finite, values, positive), axis=1),
            maximum=tf.reduce_max(tf.where(finite, values, negative), axis=1))
    return report



def independent_references(case, positions, data, validation_seeds=VALIDATION_SEEDS):
    """Independent value/score diagnostics on the same validation inputs."""
    if case.name == "lgssm":
        evaluate = case.spec.reference_value_and_score
        refined = None
        role = "exact_Kalman_CPU_reference_exception"
    elif case.name == "ksc":
        from bayesfilter.highdim.sqmc_ksc_tf import grid_reference
        evaluate = grid_reference(1201, 48., jit_compile=True)
        refined = grid_reference(1601, 56., jit_compile=True)
        role = "refined_KSC_grid_approximation"
    else:
        return dict(status="NOT_AVAILABLE", role="no independent score oracle in this tuning scope")
    rows = []
    for seed in validation_seeds:
        obs = tf.cast(data[seed][0], tf.float64)
        for index in range(int(positions.shape[0])):
            theta = tf.cast(positions[index], tf.float64)
            value, score = evaluate(theta, obs)
            base = tf.concat([tf.reshape(value, [1]), score], 0)
            row = dict(seed=seed, row=index, value_and_score=base,
                       finite=bool(tf.reduce_all(tf.math.is_finite(base))))
            if refined is not None:
                rv, rs = refined(theta, obs)
                high = tf.concat([tf.reshape(rv, [1]), rs], 0)
                row.update(refined_value_and_score=high,
                           refinement_absolute_difference=tf.abs(high-base),
                           refinement_scaled_difference=tf.abs(high-base)/tf.maximum(tf.ones_like(high),tf.abs(high)),
                           finite=row["finite"] and bool(tf.reduce_all(tf.math.is_finite(high))))
            rows.append(row)
    return dict(status="AVAILABLE", reference=role, rows=rows,
                role="explanatory nominal/local value-score accuracy; not posterior validation")


def tune(case, output, *, smoother_reset=False):
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    start = time.monotonic()
    n, dtype = runtime.N, case.force_dtype
    calibration_seeds = tuple(seed+case.tuning_seed_offset for seed in CALIBRATION_SEEDS)
    validation_seeds = tuple(seed+case.tuning_seed_offset for seed in VALIDATION_SEEDS)
    baseline = case.controls
    candidates = ([replace(baseline, reset_epsilon=baseline.reset_epsilon*factor)
                   for factor in (1., 2., 4., 8.)] if smoother_reset else
                  [baseline, replace(baseline, flow_steps=2*baseline.flow_steps,
                                     reset_steps=2*baseline.reset_steps),
                   replace(baseline, reset_epsilon=.5*baseline.reset_epsilon)])
    theta, equal = case.theta0, tf.constant([1/3]*3, dtype)
    times = sorted(set([0, case.horizon//2, case.horizon-1]))
    data = {seed: (tf.cast(runtime.fresh_observations(case, seed), dtype),
                   tuple(tf.cast(x,dtype) for x in runtime.design(case, seed+700)))
            for seed in calibration_seeds+validation_seeds}
    positions = tf.stack([theta, tf.cast(case.chart(tf.fill([case.p], tf.constant(.01, dtype)))[0], dtype)])
    references = independent_references(case, positions, data, validation_seeds)
    spec64 = case.spec
    if case.name == "range_bearing":
        from scripts.run_range_bearing_covariance_comparison import specs
        _, spec64 = specs(tf, tf.float64, case.horizon)
    records = []
    for index, controls in enumerate(candidates):
        runtime.save(Path(output)/"progress.json", dict(status="CALIBRATING",
            candidate=index, controls=asdict(controls), completed_candidates=len(records)))
        history = make_batched_score_filter(case.spec, n, case.horizon, controls,
            dtype, jit_compile=True, batch_size=1, record_history=True, precision_policy=case.precision_policy)
        context = make_context(case.spec, theta, controls, n, precision_policy=case.precision_policy)
        splits, context_valid = [], True
        trajectory_health = []
        for seeds in (calibration_seeds, validation_seeds):
            densities, targets = [], []
            for seed in seeds:
                obs, fixed = data[seed]
                run = history(theta[None], obs, *fixed, equal)
                trajectory_health.append(dict(seed=seed, **trajectory_diagnostics(run)))
                context_valid &= bool(tf.reduce_all(run["valid"])) and int(run["steps_completed"][0]) == case.horizon
                for t in times:
                    ld, la, good = context(run["ancestor_history"][0, t], obs[t],
                        tf.random.stateless_normal([n, case.spec.dimension], [seed, t+71], dtype=dtype),
                        tf.random.stateless_uniform([n, 2], [seed, t+91], dtype=dtype))
                    context_valid &= bool(good)
                    densities.append(ld); targets.append(la)
            splits.append((tf.stack(densities), tf.stack(targets)))
        if not context_valid:
            row = dict(index=index, controls=asdict(controls), valid=False,
                       precision_pass=False, context_valid=False,
                       status="INVALID_CALIBRATION_OR_VALIDATION_TRAJECTORY",
                       trajectory_health=trajectory_health)
            records.append(row)
            runtime.save(Path(output)/("filter-candidate-"+str(index)+".json"), row)
            continue
        fit = make_beta_solver(len(times)*len(calibration_seeds), n, dtype)(
            *splits[0], tf.constant(.1, dtype), tf.constant(1e-4, dtype))
        ratios, logarithms = pilot_ratios(*splits[1])
        loss = pilot_objective(fit["beta"], ratios, logarithms)[0]
        equal_loss = pilot_objective(equal, ratios, logarithms)[0]
        fit_ok = context_valid and bool(fit["converged"]) and bool(tf.math.is_finite(loss))
        good = context_valid and bool(tf.math.is_finite(equal_loss))
        use_fit = fit_ok and good and float(loss) <= float(equal_loss)
        weights = fit["beta"] if use_fit else equal
        candidate_score = make_batched_score_filter(case.spec, n, case.horizon, controls,
            dtype, jit_compile=True, batch_size=2, precision_policy=case.precision_policy)
        timing_obs, timing_fixed = data[calibration_seeds[0]]
        timer = time.monotonic()
        first = candidate_score(positions, timing_obs, *timing_fixed, weights)
        tf.nest.map_structure(lambda value: value.numpy(), first)
        compile_seconds = time.monotonic()-timer
        timer = time.monotonic()
        repeat = candidate_score(positions, timing_obs, *timing_fixed, weights)
        tf.nest.map_structure(lambda value: value.numpy(), repeat)
        score_seconds = time.monotonic()-timer
        good = good and bool(tf.reduce_all(first["valid"])) and bool(tf.reduce_all(repeat["valid"]))
        runtime.save(Path(output)/"progress.json", dict(status="PRECISION_CHECK",
            candidate=index, completed_candidates=len(records), warmed_score_seconds=score_seconds))
        score64 = make_batched_score_filter(spec64, n, case.horizon, controls,
            tf.float64, jit_compile=True, batch_size=2, precision_policy="uniform")
        checks = []
        for seed in validation_seeds:
            obs, fixed = data[seed]
            low = candidate_score(positions, obs, *fixed, weights)
            high = score64(tf.cast(positions, tf.float64), tf.cast(obs, tf.float64),
                          *[tf.cast(x, tf.float64) for x in fixed], tf.cast(weights, tf.float64))
            check = precision_screen(low, high, case.horizon, case.precision_policy)
            check["validation_seed"] = seed
            reference_rows = [r for r in references.get("rows", []) if r["seed"] == seed]
            if reference_rows:
                reference_values = tf.stack([r.get("refined_value_and_score", r["value_and_score"])
                                             for r in reference_rows])
                check["independent_reference_value_and_score"] = reference_values
                check["candidate_reference_absolute_error"] = tf.abs(
                    check["candidate_values_and_scores"]-reference_values)
                check["reference_role"] = references["role"]
            checks.append(check)
            runtime.save(Path(output)/("precision-candidate-"+str(index)+"-seed-"+str(seed)+".json"), check)
        precision_ok = all(check["passed"] for check in checks)
        good = good and precision_ok
        selected_loss = loss if use_fit else equal_loss
        row = dict(index=index, controls=asdict(controls), valid=good, beta=weights,
                   trajectory_health=trajectory_health,
                   precision_pass=precision_ok, precision_checks=checks,
                   first_score_compile_and_execute_seconds=compile_seconds,
                   warmed_score_seconds=score_seconds, score_batch_size=2,
                   importance_variance_cost_proxy=float(selected_loss)*score_seconds,
                   fit_converged=fit["converged"], context_valid=context_valid,
                   beta_fit={k:fit[k] for k in ("beta","objective","gap","iterations","converged")},
                   calibration_candidate_valid=fit_ok, validation_loss=loss,
                   equal_validation_loss=equal_loss, selected_beta_loss=selected_loss,
                   beta_choice="calibrated" if use_fit else "equal_defensive_mixture")
        records.append(runtime.json_ready(row))
        runtime.save(Path(output)/("filter-candidate-"+str(index)+".json"), row)
    viable = [row for row in records if row["valid"]]
    selected = min(viable, key=lambda row:(row["importance_variance_cost_proxy"],row["index"])) if viable else None
    result = dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC" if selected else "FILTER_SCOPE_REPAIR_REQUIRED",
        scope=case.scope(), candidates=records, selected=selected, continuation_veto=False,
        calibration_seeds=calibration_seeds, validation_seeds=validation_seeds,
        independent_reference_diagnostics=references,
        claim_data_seed=1704 if case.name=="range_bearing" else 26100831,
        baseline_controls=asdict(baseline),
        control_family="smoother_reset_1_2_4_8" if smoother_reset else "baseline_more_steps_half_epsilon",
        calibration_implementation="shared_batched_analytical_filter_with_primal_history",
        ranking="descriptive_importance_variance_times_warmed_score_cost_nomination",
        selection_policy="finite valid candidates with <=1% precision; validation variance times warmed score cost",
        proxy_nonclaim="does not measure HMC ESS per second or establish statistical superiority",
        canonical_admission=False, wall_seconds=time.monotonic()-start)
    result["selection_hash"] = runtime.hashed(result)
    runtime.save(Path(output)/"filter-selection.json", result)
    return result
