"""Independent finite-box M13 posterior diagnostics; never HMC admission."""
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-posterior-pilot-2026-09-24.md"
ORDER, PARTICLES, REPLICATES = 33, 4096, 8
PRIOR_SD = math.log(2.)
RADIUS = 4*PRIOR_SD
SEED = (2026092402, 9101)
F64 = tf.float64
NAMES = ("lambda_Q_mean", "lambda_R_mean", "lambda_Q_sd", "lambda_R_sd",
         "lambda_Q_lambda_R_covariance", "Q_sd_multiplier_mean",
         "R_sd_multiplier_mean", "outer_region_probability")


def make_integrator(order, radius, replicates):
    """Positive composite Simpson rule, including the declared Gaussian prior.

    Replications measure importance-sampling uncertainty conditional on this
    finite quadrature rule. They cannot quantify its truncation or grid error.
    """
    if order < 5 or order % 2 != 1 or radius <= 0 or replicates < 2:
        raise ValueError("odd order>=5, positive radius and >=2 replications required")
    axis = tf.linspace(tf.constant(-radius, F64), tf.constant(radius, F64), order)
    weights = tf.constant([1 if i in (0, order-1) else 4 if i % 2 else 2
                           for i in range(order)], F64)*(2*radius/(order-1)/3)
    q, r = tf.meshgrid(axis, axis, indexing="ij")
    prior = -(q*q+r*r)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2)
    log_weights = tf.math.log(weights[:, None]*weights[None, :])+prior
    boundary = tf.cast((tf.abs(q) >= .75*radius) | (tf.abs(r) >= .75*radius), F64)

    def moments(log_values):
        weighted = log_values+log_weights[..., None]
        log_evidence = tf.reduce_logsumexp(weighted, axis=[0, 1])
        mass = tf.exp(weighted-log_evidence[None, None, :])
        mean_q = tf.reduce_sum(mass*q[..., None], axis=[0, 1])
        mean_r = tf.reduce_sum(mass*r[..., None], axis=[0, 1])
        dq, dr = q[..., None]-mean_q, r[..., None]-mean_r
        metrics = tf.stack([mean_q, mean_r,
            tf.sqrt(tf.reduce_sum(mass*dq*dq, axis=[0, 1])),
            tf.sqrt(tf.reduce_sum(mass*dr*dr, axis=[0, 1])),
            tf.reduce_sum(mass*dq*dr, axis=[0, 1]),
            tf.reduce_sum(mass*tf.exp(q)[..., None], axis=[0, 1]),
            tf.reduce_sum(mass*tf.exp(r)[..., None], axis=[0, 1]),
            tf.reduce_sum(mass*boundary[..., None], axis=[0, 1])], axis=-1)
        return log_evidence, metrics

    @tf.function(input_signature=[tf.TensorSpec([order, order, replicates], F64)],
                 jit_compile=True, autograph=False)
    def integrate(log_values):
        log_evidence, metrics = moments(log_values)
        pooled = tf.reduce_logsumexp(log_values, axis=-1, keepdims=True)-math.log(replicates)
        pooled_evidence, pooled_metrics = moments(pooled)
        centered = metrics-tf.reduce_mean(metrics, axis=0)
        mcse = tf.sqrt(tf.reduce_sum(centered*centered, axis=0)/(replicates*(replicates-1)))
        evidence_relative = tf.exp(log_evidence-tf.reduce_max(log_evidence))
        evidence_centered = evidence_relative-tf.reduce_mean(evidence_relative)
        log_evidence_mcse = tf.sqrt(tf.reduce_sum(evidence_centered**2)/
            (replicates*(replicates-1)))/tf.reduce_mean(evidence_relative)
        return dict(log_evidence=pooled_evidence[0], metrics=pooled_metrics[0],
            replicate_metrics=metrics, mcse=mcse, log_evidence_mcse=log_evidence_mcse,
            finite=tf.reduce_all(tf.math.is_finite(log_values)) &
                   tf.reduce_all(tf.math.is_finite(pooled_metrics)))
    return axis, integrate


def summarize_grid(log_values, valid, *, radius=RADIUS):
    """Refuse missing/invalid nodes instead of assigning them zero posterior mass."""
    logs = tf.convert_to_tensor(log_values, F64)
    validity = tf.convert_to_tensor(valid, tf.bool)
    if logs.shape.rank != 3 or logs.shape[0] != logs.shape[1] or validity.shape != logs.shape:
        raise ValueError("complete square grid with replicate validity required")
    order, replicates = int(logs.shape[0]), int(logs.shape[2])
    if order < 9 or (order-1) % 4:
        raise ValueError("nested Simpson grids require order=4k+1, k>=2")
    if not bool(tf.reduce_all(validity & tf.math.is_finite(logs)).numpy()):
        return dict(status="INVALID_GRID_NO_POSTERIOR_SUMMARY", complete_valid_grid=False,
                    posterior_reference_qualified=False)
    _, fine = make_integrator(order, radius, replicates)
    _, coarse = make_integrator((order+1)//2, radius, replicates)
    fine_result, coarse_result = fine(logs), coarse(logs[::2, ::2])
    differences = fine_result["replicate_metrics"]-coarse_result["replicate_metrics"]
    centered = differences-tf.reduce_mean(differences, axis=0)
    paired_mcse = tf.sqrt(tf.reduce_sum(centered**2, axis=0)/(replicates*(replicates-1)))
    def serial(result):
        return {key: value.numpy().tolist() for key, value in result.items()}
    return dict(status="FINITE_BOX_POSTERIOR_DIAGNOSTIC_ONLY", complete_valid_grid=True,
        posterior_reference_qualified=False, metric_names=list(NAMES),
        fine=serial(fine_result), coarse=serial(coarse_result),
        fine_minus_coarse=(fine_result["metrics"]-coarse_result["metrics"]).numpy().tolist(),
        paired_grid_difference_mcse=paired_mcse.numpy().tolist(),
        radius=radius, quadrature="positive composite Simpson", prior_support="R^2",
        mcse_scope="importance sampling only; no grid or omitted-tail uncertainty",
        tail_bound="NOT_ESTABLISHED", endpoint_qualified=False)


def posterior_pilot_stage(output):
    from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    fixture = partition_fixture("calibration")
    data = out/"data.json"
    write_json(data, dict(model=fixture.model_definition(),
        observations=fixture.observations.numpy().tolist(), data_seed=1701,
        role="calibration posterior localization; not untouched claim"))
    dependencies = [__file__, str(ROOT/PLAN),
        str(ROOT/"bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py"),
        str(ROOT/"bayesfilter/testing/surrogate_hmc_m13_wrapped_reference.py"),
        str(ROOT/"bayesfilter/testing/surrogate_hmc_m13_reference.py"),
        str(ROOT/"bayesfilter/testing/surrogate_hmc_m13_controls.py"),
        str(ROOT/"experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py")]
    sources = {name: file_hash(Path(name)) for name in dependencies}
    axis, _ = make_integrator(ORDER, RADIUS, REPLICATES)
    nodes = axis.numpy().tolist()
    evaluate = trajectory_importance_reference(fixture, PARTICLES, REPLICATES, full_support=True)
    logs = [[[None]*REPLICATES for _ in nodes] for _ in nodes]
    validity = [[[False]*REPLICATES for _ in nodes] for _ in nodes]
    invalid = []
    minimum_ess, maximum_weight = float("inf"), 0.
    seed = tf.constant(SEED, tf.int32)  # common random numbers across theta
    rows = sorted(range(ORDER), key=lambda i: abs(nodes[i]))
    start = time.monotonic()
    for i in rows:
        records = []
        for j in rows:
            result = evaluate(tf.constant([nodes[i], nodes[j]], F64), seed)
            values = result["log_likelihood"]
            valid = result["valid"] & tf.math.is_finite(values)
            logs[i][j] = [float(v) if math.isfinite(float(v)) else None for v in values.numpy().tolist()]
            validity[i][j] = valid.numpy().tolist()
            if not all(validity[i][j]):
                invalid.append([i, j])
            ess = float(tf.reduce_min(result["minimum_ess"]).numpy())
            weight = float(tf.reduce_max(result["proposal_maximum_normalized_weight"]).numpy())
            if math.isfinite(ess):
                minimum_ess = min(minimum_ess, ess)
            if math.isfinite(weight):
                maximum_weight = max(maximum_weight, weight)
            def finite_list(value):
                return [float(v) if math.isfinite(float(v)) else None for v in value.numpy().tolist()]
            records.append(dict(index=[i, j], theta=[nodes[i], nodes[j]],
                log_likelihood=logs[i][j], valid=validity[i][j],
                minimum_ess=finite_list(result["minimum_ess"]),
                maximum_weight=finite_list(result["proposal_maximum_normalized_weight"]),
                fit_first=finite_list(result["proposal_first_diagnostics"]),
                fit_full=finite_list(result["proposal_full_diagnostics"])))
        write_json(out/f"row-{i:02}.json", records)
        print("M13 posterior pilot row", i, "invalid so far", len(invalid),
              "wall", time.monotonic()-start, flush=True)
    if any(file_hash(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError("source changed during posterior-reference pilot")
    grid_path = out/"grid.json"
    write_json(grid_path, dict(axis=nodes, log_likelihood=logs, valid=validity, seed=list(SEED)))
    summary = (dict(status="INVALID_GRID_NO_POSTERIOR_SUMMARY", complete_valid_grid=False,
                    posterior_reference_qualified=False) if invalid else summarize_grid(logs, validity))
    write_json(out/"summary.json", summary)
    return dict(status=("M13_POSTERIOR_REFERENCE_PILOT_COMPLETED" if not invalid else
                        "M13_POSTERIOR_REFERENCE_PILOT_REPAIR_REQUIRED"),
        qualified=False, posterior_reference_qualified=False, endpoint_qualified=False,
        inference_launched=False, all_models_validated=False, continuation_veto=False,
        invalid_nodes=invalid, valid_node_count=ORDER*ORDER-len(invalid), total_nodes=ORDER*ORDER,
        minimum_ess=minimum_ess if math.isfinite(minimum_ess) else None,
        maximum_weight=maximum_weight, summary=summary, grid_sha256=file_hash(grid_path),
        data_sha256=file_hash(data), source_sha256=sources, particles=PARTICLES, replicates=REPLICATES,
        seed=list(SEED), common_random_numbers_across_theta=True, dtype="float64",
        independent_reference_exception=True, jit_compile=True, plan=PLAN,
        heuristic_dominance="integration invariants tested; sampler comparison not run",
        ranking_supported=False, posterior_comparisons="NOT_RUN",
        next_action="Existing master: inspect conditional validity, grid sensitivity and tails; repair/qualify reference before HMC endpoint and tuning.")
