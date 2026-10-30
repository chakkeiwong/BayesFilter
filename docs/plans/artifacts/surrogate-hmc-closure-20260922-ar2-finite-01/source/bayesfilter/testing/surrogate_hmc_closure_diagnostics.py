"""Bounded reference/diagnostic workers; cannot authorize HMC or runtime defaults."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def wilson(successes, trials):
    if trials < 1:
        return None
    z = 1.959963984540054
    p, z2 = successes/trials, z*z
    center = (p+z2/(2*trials))/(1+z2/trials)
    radius = z*math.sqrt(p*(1-p)/trials+z2/(4*trials*trials))/(1+z2/trials)
    return dict(count=successes, trials=trials, rate=p, lower=max(0., center-radius), upper=min(1., center+radius))


def wrapped_stage():
    import tensorflow as tf
    from types import SimpleNamespace
    from bayesfilter.wrapped_gaussian_tf import alias_policy, wrapped_gaussian_kernel
    from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import observation_log_density_tf

    @tf.function(input_signature=[tf.TensorSpec([], tf.float64)], jit_compile=True, autograph=False)
    def integrate(sigma):
        angle = (tf.cast(tf.range(8192), tf.float64)+.5)*2*math.pi/8192-math.pi
        particles = tf.broadcast_to(tf.constant([[1., 0., 0., 0.]], tf.float64), [8192, 4])
        observed = tf.stack([tf.ones_like(angle), angle], axis=-1)
        fixture = SimpleNamespace(R=tf.linalg.diag(tf.stack([tf.constant(1., tf.float64), sigma*sigma])))
        values = observation_log_density_tf(fixture, particles, observed)
        mass = tf.reduce_sum(tf.exp(values+.5*math.log(2*math.pi)))*2*math.pi/8192
        return mass

    masses = [dict(sigma=sigma, integrated_mass=float(integrate(tf.constant(sigma, tf.float64))))
              for sigma in (.04, .2, 1., math.pi)]
    kernel = wrapped_gaussian_kernel()
    x = tf.constant([[.2, -.9], [-2., 3.], [.7, 0.]], tf.float64)
    dx = tf.constant([[.1, -.3], [.2, .4], [-.5, .1]], tf.float64)
    cov = tf.constant([[.6, .25], [.25, 2.]], tf.float64)
    dc = tf.constant([[.2, -.04], [-.04, .3]], tf.float64)
    result = kernel(x, cov, dx, dc)
    eps = tf.constant(1e-5, tf.float64)
    fd = (kernel(x+eps*dx, cov+eps*dc, dx, dc)["value"]-kernel(x-eps*dx, cov-eps*dc, dx, dc)["value"])/(2*eps)
    error = float(tf.reduce_max(tf.abs(fd-result["tangent"])))
    passed = all(abs(row["integrated_mass"]-1) <= 1e-10 for row in masses) and error <= 2e-9
    return dict(status="OBSERVATION_LAW_REPAIR_QUALIFIED" if passed else "LAW_REPAIR_FAILED",
                qualified=passed, masses=masses, analytical_tangent_max_fd_error=error, alias_policy=alias_policy(),
                consumer="range_bearing_tf.observation_log_density_tf", jit_compile=True,
                remaining=["integrated likelihood and parameter-posterior reference", "canonical model consumer and tuning"],
                posterior_qualified=False, inference_launched=False)


def structural_stage():
    import tensorflow as tf
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import PerPointScoreModel, canonical_batch_fused_value_score
    from bayesfilter.highdim.ledh_failure_status_tf import FailureCode
    dtype = tf.float64
    f = tf.constant([[.45, -.15], [1., 0.]], dtype)
    q = tf.constant([[.09, 0.], [0., 0.]], dtype)
    h = tf.constant([[1., 0.]], dtype)
    model = PerPointScoreModel(
        transition_mean_fn=lambda theta, points: tf.matmul(points, f, transpose_b=True),
        transition_mean_tangent_fn=lambda theta, points, tangent, direction: tf.matmul(tangent, f, transpose_b=True),
        observation_fn=lambda points: tf.matmul(points, h, transpose_b=True),
        observation_jacobian_fn=lambda points: tf.broadcast_to(h, [tf.shape(points)[0], 1, 2]),
        observation_tangent_fn=lambda points, tangent: tf.matmul(tangent, h, transpose_b=True),
        process_covariance=q, observation_covariance=tf.constant([[.04]], dtype))
    cloud = tf.constant([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]*2, dtype)

    @tf.function(input_signature=[tf.TensorSpec([2, 1], dtype)], jit_compile=True, autograph=False)
    def evaluate(theta):
        return canonical_batch_fused_value_score(model, theta, tf.ones([2, 1, 1], dtype), cloud,
            tf.eye(2, batch_shape=[8], dtype=dtype), tf.zeros([1, 8, 2], dtype), tf.constant([[.1]], dtype),
            substeps=1, reset_policy="contract_e", reset_design=cloud, reset_sinkhorn_steps=8,
            reset_balance_steps=8, correction_steps=1, pairwise_steps=1, coordinate_cap=2., k_batch_mode="sequential")
    _, _, status = evaluate(tf.zeros([2, 1], dtype))
    expected_block = bool(tf.reduce_all(~status["value_valid"] & ~status["force_valid"]
                                      & (status["value_failure_code"] == int(FailureCode.FACTORIZATION))))
    return dict(status="BLOCKED_CANONICAL_STRUCTURAL_TRANSITION" if expected_block else "UNEXPECTED_CONSUMER_RESULT",
                qualified=False, expected_block_observed=expected_block,
                consumer_status={key: status[key].numpy().tolist() for key in
                    ("value_valid", "force_valid", "value_failure_code", "first_failure_time", "first_failure_stage")},
                process_covariance=q.numpy().tolist(), jit_compile=True,
                required_implementation=["innovation-space proposal", "same-measure transition weights",
                    "structural deterministic completion", "full canonical reset and analytical recursive score"],
                next_action="derive and implement the general structural canonical path before AR2 inference",
                inference_launched=False)


def uncertainty_protocol(confirmation=False):
    protocol = dict(scope="KSC-T2 thirteen functions on qualified reference grid; diagnostic discrete law",
        groups=[("iid", 0., 0, 512, (1000, 2000)), ("refresh_half", .5, 0, 256, (1000, 2000)),
                ("refresh_nine_tenths", .9, 0, 128, (4000, 8000)),
                ("sticky", .99, 0, 128, (1000, 2000)), ("chain_offsets", 0., 1, 128, (1000, 2000)),
                ("drift", 0., 2, 128, (1000, 2000))],
        seed_base=2026092200, functions=13, comparison_families=1, planned_looks=2, permutations=1023,
        false_veto_upper_limit=.05, coverage_failure_upper_limit=.05, minimum_admitted=64,
        rejection_lower_limit=.95, alpha=.05, false_equivalence_reference_shift_margin_multiple=1.25,
        calibration_scope="candidate diagnostic only; no runtime promotion or old-chain rescoring")
    if confirmation:
        protocol.update(groups=[("refresh_half", .5, 0, 1024, (1000, 2000)),
                                ("refresh_nine_tenths", .9, 0, 1024, (4000, 8000))],
                        seed_base=2026192200, purpose="fresh precision confirmation of unchanged dependent-chain diagnostic")
    return protocol


def uncertainty_stage(output, metadata, *, confirmation=False):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat, rank_normalized_bulk_tail_ess
    from bayesfilter.testing.surrogate_hmc_mcse_randomization import (
        batch_randomization, binary_function_rhat, diagnostic_batch_statistics)
    from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals

    protocol = uncertainty_protocol(confirmation)
    source = ROOT/"docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-posterior-01/report.json"
    reference = json.loads(source.read_text())
    if reference["qualified"] is not True:
        raise ValueError("KSC posterior reference is not qualified")
    grid = reference["grids"][-1]
    arrays = {}
    for name in ("gamma", "beta", "density", "predictive"):
        record = grid[name]
        path = Path(record["path"])
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"reference tensor changed: {name}")
        arrays[name] = tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64)
    a, b = arrays["gamma"], arrays["beta"]
    # Trapezoidal node masses define a finite diagnostic probability law.
    def weights(knots):
        delta = knots[1:]-knots[:-1]
        return tf.concat([delta[:1]/2, (delta[:-1]+delta[1:])/2, delta[-1:]/2], 0)
    probabilities = tf.reshape(arrays["density"]*weights(a)[:, None]*weights(b)[None, :], [-1])
    tf.debugging.assert_all_finite(probabilities, "reference node weights")
    tf.debugging.assert_non_negative(probabilities)
    probabilities /= tf.reduce_sum(probabilities)
    cdf = tf.cumsum(probabilities)
    cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], 0)
    aa, bb = tf.meshgrid(a, b, indexing="ij")
    table = tf.stack([tf.reshape(aa, [-1]), tf.reshape(bb, [-1])], -1)
    means, scales, cuts = [tf.constant(reference[k], tf.float64) for k in ("means", "scales", "cuts")]
    functions = scalar_functionals(table, tf.reshape(arrays["predictive"], [-1, 2]), means, scales, cuts)
    truth = tf.reduce_sum(probabilities[:, None]*functions, 0)
    margins = tf.constant(reference["margins"], tf.float64)
    critical = tfp.distributions.StudentT(tf.constant(36., tf.float64), tf.constant(0., tf.float64),
        tf.constant(1., tf.float64)).quantile(tf.constant(1-.05/(2*13*2), tf.float64))
    report = dict(metadata, status="RUNNING", qualified=False, protocol=protocol, groups={},
        reference=str(source), reference_sha256=file_hash(source), reference_tensors=grid,
        truth=truth.numpy().tolist(), function_names=reference["function_names"],
        margins=margins.numpy().tolist(), critical_value=float(critical),
        reference_error="zero for the explicitly defined discrete diagnostic law; continuous-posterior error excluded",
        generator_jit_compile=True, batch_and_randomization_jit_compile=True,
        rank_diagnostic_jit_compile=False, rank_exception="established tie-aware non-XLA diagnostic graph",
        inference_launched=False, runtime_default_promoted=False)
    if confirmation:
        basis = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-01/uncertainty/report.json"
        preceding = json.loads(basis.read_text())
        if not all(preceding["groups"][name]["passed"] for name in ("iid", "sticky", "chain_offsets", "drift")):
            raise ValueError("precision confirmation requires passing original IID and rejection controls")
        for key in ("reference_sha256", "truth", "margins", "critical_value"):
            if report[key] != preceding[key]:
                raise ValueError(f"confirmation changed diagnostic definition: {key}")
        report["preceding_calibration"] = dict(path=str(basis), sha256=file_hash(basis),
            role="retain passed IID/rejection controls; fresh dependent groups only; no pooled or rescored old outcomes")
    write_json(output, report)

    def generator(count):
        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.int32)],
                     jit_compile=True, autograph=False)
        def generate(seed, rho, regime):
            uniforms = tf.random.stateless_uniform([count, 4, 2], seed, dtype=tf.float64)
            fresh = tf.reshape(tf.searchsorted(cdf, tf.reshape(uniforms[:, :, 0], [-1]), side="right"), [count, 4])
            indices = tf.scan(lambda previous, pair: tf.where(pair[1] >= rho, pair[0], previous),
                (fresh[1:], uniforms[1:, :, 1]), initializer=fresh[0])
            indices = tf.concat([fresh[:1], indices], 0)
            points, values = tf.gather(table, indices), tf.gather(functions, indices)
            offsets = tf.constant([-.8, -.25, .25, .8], tf.float64)[None, :]
            drift = tf.linspace(tf.constant(-1.5, tf.float64), tf.constant(1.5, tf.float64), count)[:, None]
            shift = tf.cast(regime == 1, tf.float64)*offsets+tf.cast(regime == 2, tf.float64)*drift
            points += shift[:, :, None]*tf.constant([1., 0.], tf.float64)*scales
            # Recompute moments/CDFs after deliberate nonstationary perturbation.
            values = tf.reshape(scalar_functionals(tf.reshape(points, [-1, 2]),
                tf.reshape(values[:, :, -2:], [-1, 2]), means, scales, cuts), [count, 4, 13])
            return tf.transpose(points, [1, 0, 2]), tf.transpose(values, [1, 0, 2])
        return generate

    def calculator(count):
        batches = diagnostic_batch_statistics(count, 13)
        randomized = batch_randomization(count, 13)

        @tf.function(input_signature=[tf.TensorSpec([4, count, 2], tf.float64),
            tf.TensorSpec([4, count, 13], tf.float64), tf.TensorSpec([2], tf.int32)], jit_compile=False, autograph=False)
        def evaluate(points, values, seed):
            pr = rank_normalized_split_rhat(points)
            ess = rank_normalized_bulk_tail_ess(points)
            fr = rank_normalized_split_rhat(values)
            effective, binary = binary_function_rhat(values, fr)
            estimates, lag = batches(values)
            mcse = tf.reduce_max(estimates, axis=0)
            sensitivity = mcse/tf.reduce_min(estimates, axis=0)
            randomization = randomized(values, seed)
            finite = tf.reduce_all(tf.math.is_finite(points)) & tf.reduce_all(tf.math.is_finite(values))
            mixing = finite & tf.reduce_all(pr["maximum"] <= 1.01) & tf.reduce_all(ess["bulk"] >= 400.) & tf.reduce_all(ess["tail"] >= 400.)
            variance_valid = tf.reduce_all(tf.math.is_finite(mcse) & (mcse > 0.) & (sensitivity <= 2.))
            common = mixing & variance_valid
            original = common & tf.reduce_all(fr["maximum"] <= 1.01) & tf.reduce_all(tf.abs(lag) <= .3)
            candidate = common & tf.reduce_all(effective <= 1.01) & randomization["valid"] & (randomization["p_value"] > .01)
            difference = tf.reduce_mean(values, [0, 1])-truth
            radius = critical*mcse
            naive = tf.sqrt(tf.math.reduce_variance(values, [0, 1])/(4*count-1))
            false_equivalence = []
            for index in (0, 7):
                wrong_reference = tf.one_hot(index, 13, dtype=tf.float64)*1.25*margins
                false_equivalence.append(candidate & tf.reduce_all(tf.abs(difference-wrong_reference)+radius <= margins))
            return dict(candidate_valid=candidate, original_valid=original,
                all_functions_covered=tf.reduce_all(tf.abs(difference) <= radius),
                naive_all_functions_covered=tf.reduce_all(tf.abs(difference) <= critical*naive),
                false_equivalence=tf.reduce_any(tf.stack(false_equivalence)), mcse=mcse, half_width=radius,
                difference=difference, effective_function_rhat=effective, parameter_rhat=pr["maximum"],
                bulk_ess=ess["bulk"], tail_ess=ess["tail"], sensitivity=sensitivity, batch_lag=lag,
                binary_fold_uninformative=binary, randomization_p=randomization["p_value"])
        return evaluate
    generators = {n: generator(n) for n in (2000, 8000)}
    calculators = {n: calculator(n) for n in (1000, 2000, 4000, 8000)}
    start = time.monotonic()
    for group_index, (name, rho, regime, replications, looks) in enumerate(protocol["groups"]):
        rows = []
        with (output.parent/(name+"-replications.jsonl")).open("x") as stream:
            for replication in range(replications):
                seed = protocol["seed_base"]+10000*group_index+replication
                points, values = generators[looks[-1]](tf.constant([seed, 0], tf.int32), tf.constant(rho, tf.float64), tf.constant(regime, tf.int32))
                results = [finite_report({key: value.numpy().tolist() for key, value in
                    calculators[n](points[:, :n], values[:, :n], tf.constant([seed, 100+n], tf.int32)).items()})
                    for n in looks]
                row = dict(replication=replication, seed=seed, looks=results)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
                if time.monotonic()-start > 2700:
                    raise TimeoutError("prospective calibration wall reserve reached")
        admitted = [row for row in rows if any(look["candidate_valid"] for look in row["looks"])]
        any_veto = sum(any(not look["candidate_valid"] for look in row["looks"]) for row in rows)
        both_veto = sum(all(not look["candidate_valid"] for look in row["looks"]) for row in rows)
        failed_intervals = sum(any(not look["all_functions_covered"] for look in row["looks"]) for row in rows)
        admitted_failures = sum(any(look["candidate_valid"] and not look["all_functions_covered"] for look in row["looks"]) for row in admitted)
        group = dict(any_look_veto=wilson(any_veto, replications), both_look_veto=wilson(both_veto, replications),
            unconditional_interval_failure=wilson(failed_intervals, replications),
            admitted_interval_failure=wilson(admitted_failures, len(admitted)),
            false_equivalence=wilson(sum(any(look["false_equivalence"] for look in row["looks"]) for row in rows), replications),
            original_any_veto=wilson(sum(any(not look["original_valid"] for look in row["looks"]) for row in rows), replications),
            naive_interval_failure=wilson(sum(any(not look["naive_all_functions_covered"] for look in row["looks"]) for row in rows), replications))
        if name in ("iid", "refresh_half", "refresh_nine_tenths"):
            group["passed"] = (len(admitted) >= 64 and group["admitted_interval_failure"]["upper"] <= .05
                and group["unconditional_interval_failure"]["upper"] <= .05 and group["false_equivalence"]["upper"] <= .05
                and (name != "iid" or group["any_look_veto"]["upper"] <= .05))
        else:
            group["passed"] = group["both_look_veto"]["lower"] >= .95
        report["groups"][name] = group
        write_json(output, report)
    report["qualified"] = all(row["passed"] for row in report["groups"].values())
    report["status"] = "SCOPED_DIAGNOSTIC_CALIBRATION_PASSED" if report["qualified"] else "DIAGNOSTIC_CANDIDATE_NOT_QUALIFIED"
    report["statistical_scope"] = "Per-regime Wilson bounds; no simultaneous all-model guarantee or candidate ranking"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("wrapped", "structural", "uncertainty", "uncertainty-confirmation", "structural-integration", "ar2-reference", "ar2-reference-extended", "ar2-controls", "ar2-cost", "ar2-calibration", "ar2-calibration-confirm", "ar2-calibration-repair", "ar2-calibration-repair-r2", "ar2-profile", "ar2-kernel-repair", "ar2-inference-b", "ar2-inference-a", "ar2-finite-preflight"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("fresh stage output required")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from bayesfilter.inference.surrogate_hmc_program import write_json
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = (configure_tensorflow_gpu_memory_growth(tf) if args.device == "gpu" else
              dict(mode="CPU_REFERENCE_EXCEPTION", gpu_intentionally_hidden=True))
    metadata = dict(stage=args.stage, tensorflow=tf.__version__, device=args.device, memory_policy=memory,
                    tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
                    gpu_intentionally_hidden=args.device == "cpu")
    if args.device == "gpu":
        probe = tf.linalg.matmul(tf.ones([2, 2]), tf.ones([2, 2]))
        if "GPU" not in probe.device:
            raise RuntimeError("GPU stage did not place its probe on the GPU")
        metadata["probe_device"] = probe.device
    if args.stage == "structural-integration":
        from bayesfilter.testing.surrogate_hmc_structural_integration import integration_stage
        result = integration_stage()
    elif args.stage.startswith("ar2-reference"):
        from bayesfilter.testing.surrogate_hmc_ar2_reference import reference_stage
        result = reference_stage(args.output, extended=args.stage == "ar2-reference-extended")
    elif args.stage == "ar2-controls":
        from bayesfilter.testing.surrogate_hmc_ar2_controls import controls_stage
        result = controls_stage(args.output)
    elif args.stage == "ar2-cost":
        from bayesfilter.testing.surrogate_hmc_ar2_sampling import cost_stage
        result = cost_stage(args.output)
    elif args.stage in ("ar2-calibration", "ar2-calibration-confirm", "ar2-calibration-repair", "ar2-calibration-repair-r2"):
        from bayesfilter.testing.surrogate_hmc_ar2_calibration import COST_ROOT, calibration_stage
        result = calibration_stage(args.output, COST_ROOT, confirmation=args.stage.endswith("-confirm"),
                                   repair=("repair" in args.stage),
                                   repair_variant="r2" if args.stage.endswith("-r2") else "r3")
    elif args.stage == "ar2-profile":
        from bayesfilter.testing.surrogate_hmc_ar2_profile import profile_stage
        result = profile_stage(args.output)
    elif args.stage == "ar2-kernel-repair":
        from bayesfilter.testing.surrogate_hmc_ar2_kernel_repair import kernel_repair_stage
        result = kernel_repair_stage(args.output)
    elif args.stage == "ar2-finite-preflight":
        from bayesfilter.testing.surrogate_hmc_ar2_finite import preflight_stage
        result = preflight_stage(args.output)
    elif args.stage in ("ar2-inference-b", "ar2-inference-a"):
        from bayesfilter.testing.surrogate_hmc_ar2_calibration import COST_ROOT
        from bayesfilter.testing.surrogate_hmc_ar2_inference import CALIBRATION_ROOT, inference_stage
        result = inference_stage(args.output, cost_root=COST_ROOT, calibration_root=CALIBRATION_ROOT,
                                 arm="B" if args.stage.endswith("-b") else "A")
    else:
        result = (uncertainty_stage(args.output, metadata, confirmation=args.stage == "uncertainty-confirmation")
              if args.stage.startswith("uncertainty") else
              wrapped_stage() if args.stage == "wrapped" else structural_stage())
    result.update(metadata)
    write_json(args.output, result)
    print(json.dumps(dict(stage=args.stage, status=result["status"], qualified=result["qualified"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
