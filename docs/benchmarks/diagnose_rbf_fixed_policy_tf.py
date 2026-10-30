"""Diagnostic-only fixed-policy RBF normalizer and filtering checks.

Independent TF quadrature is the numerical reference. The candidate evaluation
calls the existing, snapshotted RBFControlledSMC consumer. No fitting or selection.
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, "/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
MEMORY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
spec = importlib.util.spec_from_file_location("rbf_forward_snapshot",
    Path(__file__).with_name("rbf_controlled_smc_tf.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
D = tf.float64
REPETITIONS = 128
BATCH = 8
BOOTSTRAPS = 20000


def scalar(value):
    return float(value.numpy())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Quadrature:
    """Direct integration of model densities, independent of the particle code."""
    def __init__(self, horizon, points, bound, observation_variance):
        self.h, self.n, self.bound, self.r = horizon, points, bound, observation_variance
        self.grid = tf.linspace(tf.constant(-bound, D), tf.constant(bound, D), points)
        step = tf.constant(2 * bound / (points - 1), D)
        self.integration_weight = step * tf.concat([
            tf.constant([.5], D), tf.ones([points - 2], D), tf.constant([.5], D)], 0)
        self.run = tf.function(self._run, input_signature=[tf.TensorSpec([horizon], D)],
            jit_compile=True, autograph=False)

    def _run(self, observations):
        x, weight = self.grid, self.integration_weight
        arrays = tuple(tf.TensorArray(D, self.h) for _ in range(7))
        def body(t, probability, logz, arrays):
            def transition():
                # Columns index previous states; rows index current states.
                mean = .5*x + 25*x/(1+x*x) + 8*tf.cos(1.2*tf.cast(t, D))
                density = tf.exp(-.5*(x[:, None]-mean[None, :])**2/10)
                density /= tf.sqrt(tf.constant(20*math.pi, D))
                return tf.linalg.matvec(density, probability)
            prediction = tf.cond(t == 0,
                lambda: tf.exp(-.5*x*x/5)/tf.sqrt(tf.constant(10*math.pi, D)),
                transition)
            mass = tf.reduce_sum(prediction*weight)
            logg = -.5*(tf.math.log(tf.constant(2*math.pi*self.r, D))
                +(observations[t]-x*x/20)**2/self.r)
            shift = tf.reduce_max(logg)
            unnormalized = prediction*weight*tf.exp(logg-shift)
            normalizer = tf.reduce_sum(unnormalized)
            probability = unnormalized/normalizer
            logz += tf.math.log(normalizer)+shift
            mean = tf.reduce_sum(probability*x)
            second = tf.reduce_sum(probability*x*x)
            boundary = tf.reduce_sum(tf.where(tf.abs(x)>.95*self.bound, probability, 0.))
            arrays = tuple(a.write(t, v) for a, v in zip(arrays,
                (logz, mean, second, mass, boundary, probability, prediction)))
            return t+1, probability, logz, arrays
        state = tf.while_loop(lambda t, *_: t < self.h, body,
            (tf.constant(0), tf.zeros([self.n], D), tf.constant(0., D), arrays),
            parallel_iterations=1)
        return {k: a.stack() for k, a in zip(
            ("logZ", "mean", "second_moment", "predictive_mass", "boundary_mass", "probability", "prediction"), state[3])}


class ForwardDiagnostic:
    def __init__(self, horizon, particles, components, observation_variance):
        self.h, self.n, self.m = horizon, particles, components
        self.filter = core.RBFControlledSMC(observations=horizon, particles=particles,
            replicas=BATCH, components=components, observation_variance=observation_variance)
        field = tf.TensorSpec([horizon, BATCH], D)
        policy = tf.TensorSpec([horizon, BATCH, components], D)
        cloud = tf.TensorSpec([horizon, BATCH, particles], D)
        self.run = tf.function(self._run, input_signature=[
            field, policy, policy, field, field, cloud, cloud, field],
            jit_compile=True, autograph=False)
        self.noise = tf.function(self._noise, input_signature=[tf.TensorSpec([2], tf.int32)],
            jit_compile=True, autograph=False)

    def _noise(self, seed):
        normal = tf.random.stateless_normal([self.h, BATCH, self.n],
            tf.stack([seed[0], seed[1]*3]), dtype=D)
        component = tf.random.stateless_uniform([self.h, BATCH, self.n],
            tf.stack([seed[0], seed[1]*3+1]), dtype=D)
        resampling = tf.random.stateless_uniform([self.h, BATCH],
            tf.stack([seed[0], seed[1]*3+2]), dtype=D)
        return normal, component, resampling

    def _run(self, beta, weights, knots, constant, y, normal, component, uniform):
        sim = self.filter.forward(beta, weights, knots, constant, y,
            normal, component, uniform)
        moments = tf.TensorArray(D, self.h)
        def body(t, moments, valid):
            x = sim["x"][t]
            def future():
                integral = core.rbf_gaussian_mixture(core.multimodal_transition(t+1, x),
                    tf.constant(10., D), beta[t+1], weights[t+1], knots[t+1], constant[t+1])
                return integral["log_integral"], integral["valid"]
            lookahead, okay = tf.cond(t+1 < self.h, future,
                lambda: (tf.zeros_like(x), tf.constant(True)))
            # Remove the future-policy twist to estimate the actual filtering mean.
            filtering_weights = tf.nn.softmax(sim["log_incremental_weight"][t]-lookahead, -1)
            mean = tf.reduce_sum(filtering_weights*x, -1)
            return t+1, moments.write(t, mean), valid & okay & tf.reduce_all(tf.math.is_finite(mean))
        state = tf.while_loop(lambda t, *_: t < self.h, body,
            (tf.constant(0), moments, sim["valid"]), parallel_iterations=1)
        return dict(logZ=sim["logZ"][-1], filtering_mean=state[1].stack(),
            mean_ess_fraction=tf.reduce_mean(sim["ess"]/self.n, 0), valid=state[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=(128, 4096), default=128)
    parser.add_argument("--seed-base", type=int, default=202609300)
    parser.add_argument("--full-retention", action="store_true", help="Diagnostic only: preserve every positive saved coefficient")
    parser.add_argument("--refit", type=Path, help="Diagnostic full backward-refit directory; requires --full-retention")
    args = parser.parse_args()
    if args.refit and not args.full_retention:
        parser.error("--refit requires --full-retention")
    repetitions = args.repetitions
    started = time.monotonic()
    checks, progress = [], []
    summary = dict(status="RUNNING", scope="RBF_FIXED_POLICY_DOWNSTREAM_DIAGNOSTIC",
        replication="INCOMPLETE", promotion="NOT_REQUESTED", gpu_memory_policy=MEMORY,
        numpy_runtime=False, pfor=False, python_time_loop=False, jit_compile=True,
        tf32=False, dtype="float64", device="/GPU:0", checks=checks)
    def flag(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
        if not passed:
            raise RuntimeError(name)
    try:
        fixture = json.loads(args.fixture.read_text())
        h, n, m = (int(fixture[k]) for k in ("observations", "particles", "components"))
        r = float(fixture["observation_variance"])
        flag("fixed_scope", (h, n, m, r) == (101, 512, 16, 1.))
        with tf.device("/GPU:0"):
            y = tf.constant(fixture["y"], D)
            quadrature, oracle_details = [], []
            for points, bound in ((1025, 60.), (2049, 60.), (2049, 80.)):
                model = Quadrature(h, points, bound, r)
                before = time.monotonic()
                result = model.run(y)
                finite = bool(tf.reduce_all(tf.stack([
                    tf.reduce_all(tf.math.is_finite(v)) for v in result.values()])).numpy())
                mass_error = scalar(tf.reduce_max(tf.abs(result["predictive_mass"]-1)))
                boundary = scalar(tf.reduce_max(result["boundary_mass"]))
                flag("quadrature_valid_%d_%d" % (points, bound), finite and
                    mass_error <= 1e-10 and boundary <= 1e-12,
                    predictive_mass_error=mass_error, boundary_mass=boundary)
                flag("quadrature_one_trace_%d_%d" % (points, bound),
                    model.run.experimental_get_tracing_count() == 1)
                quadrature.append(result)
                oracle_details.append(dict(points=points, bound=bound,
                    seconds=time.monotonic()-before,
                    values={k: v.numpy().tolist() for k, v in result.items()}))
            for i, j in ((0, 1), (0, 2), (1, 2)):
                ze = scalar(tf.abs(quadrature[i]["logZ"][-1]-quadrature[j]["logZ"][-1]))
                me = scalar(tf.reduce_max(tf.abs(quadrature[i]["mean"]-quadrature[j]["mean"])))
                flag("quadrature_agreement_%d_%d" % (i, j), ze <= 1e-7 and me <= 1e-6,
                    terminal_logZ_error=ze, filtering_mean_error=me)
            oracle = quadrature[-1]
            save_json(args.output/"quadrature.json", oracle_details)
            print(json.dumps(dict(stage="oracle_passed", terminal_logZ=scalar(oracle["logZ"][-1]))), flush=True)

            def saved(name):
                return tf.io.parse_tensor(tf.io.read_file(str(
                    args.candidate/("fit_"+name+".tensor"))), D)
            raw = {k: saved(k) for k in ("beta", "weights", "knots", "constant")}
            flag("saved_policy_valid", bool(tf.reduce_all(tf.stack([
                tf.reduce_all(tf.math.is_finite(v)) for v in raw.values()])).numpy())
                and bool(tf.reduce_all(raw["weights"] >= 0).numpy())
                and bool(tf.reduce_all(tf.reduce_sum(raw["weights"], -1) > 0).numpy()))
            def repeat_policy(values):
                return [tf.repeat(v[:, None, ...], BATCH, axis=1) for v in values]
            if args.full_retention:
                raw["coefficients"] = saved("coefficients")
                refitted = None
                if args.refit:
                    record = json.loads((args.refit/"summary.json").read_text())
                    if record.get("status") != "PASS" or record.get("full_backward_refit") is not True:
                        raise RuntimeError("Full-refit input did not pass its validity and recurrence checks")
                    refitted = {k: tf.io.parse_tensor(tf.io.read_file(str(args.refit/("fit_"+k+".tensor"))), D)
                        for k in ("beta", "constant", "coefficients")}
                all_coefficients = tf.stack([v for i, case in enumerate(fixture["cases"])
                    for v in (raw["coefficients"][:, i], refitted["coefficients"][:, i]
                        if refitted is not None else tf.constant(case["full_coefficients"], D))], 1)
                flag("full_coefficients_valid",
                    bool(tf.reduce_all(tf.math.is_finite(all_coefficients) & (all_coefficients >= 0)).numpy())
                    and bool(tf.reduce_all(tf.reduce_sum(all_coefficients, -1) > 0).numpy()))
                m = int(tf.reduce_max(tf.reduce_sum(tf.cast(all_coefficients > 0, tf.int32), -1)).numpy())
                summary.update(scope="RBF_FULL_RETENTION_FIXED_POLICY_DIAGNOSTIC",
                    policy_representation="all_positive_saved_coefficients_exact_zero_pruning",
                    original_basis_components=n, forward_component_slots=m,
                    full_backward_refit=False)
                if args.refit:
                    summary.update(scope="RBF_FULL_BACKWARD_REFIT_FORWARD_DIAGNOSTIC",
                        full_backward_refit=True, mechanism_control="previous_QR_full_saved_fits",
                        policy_representation="two_full_refits_and_two_full_saved_fits_exact_zero_pruning")
                @tf.function(input_signature=[tf.TensorSpec([h, n], D), tf.TensorSpec([h, n], D)],
                    jit_compile=True, autograph=False)
                def compact(coefficients, cloud):
                    values, indices = tf.math.top_k(coefficients, k=n, sorted=True)
                    weights = values[:, :m]
                    knots = tf.gather(cloud, indices[:, :m], axis=1, batch_dims=1)
                    return weights, knots, tf.reduce_all(values[:, m:] == 0)
            arms = []
            for index, case in enumerate(fixture["cases"]):
                label = int(case["seed"])
                ref = case["rounds"][1]
                if args.full_retention:
                    cloud = tf.constant(case["rounds"][0]["x"], D)
                    for prefix, coefficients, beta, constant in (
                        ("qr_full", raw["coefficients"][:, index], raw["beta"][:, index], raw["constant"][:, index]),
                        (("qr_refit", refitted["coefficients"][:, index], refitted["beta"][:, index], refitted["constant"][:, index])
                         if refitted is not None else ("svd_full", tf.constant(case["full_coefficients"], D), tf.constant(ref["beta"], D), tf.constant(ref["constant"], D)))):
                        weights, knots, exact = compact(coefficients, cloud)
                        flag("zero_only_pruning_%s_%d" % (prefix, label), bool(exact.numpy()))
                        arms.append(("%s_seed%d" % (prefix, label), repeat_policy([beta, weights, knots, constant])))
                else:
                    arms.append(("qr_seed%d" % label, repeat_policy([
                        raw[k][:, index] for k in ("beta", "weights", "knots", "constant")])))
                    arms.append(("svd_seed%d" % label, repeat_policy([
                        tf.constant(ref[k], D) for k in ("beta", "weights", "knots", "constant")])))
            if args.full_retention:
                flag("compact_one_trace", compact.experimental_get_tracing_count() == 1)
            def zero_policy(components):
                return [tf.zeros([h, BATCH], D),
                    tf.broadcast_to(tf.one_hot(0, components, dtype=D), [h, BATCH, components]),
                    tf.zeros([h, BATCH, components], D), tf.zeros([h, BATCH], D)]
            arms.extend([("bpf512", zero_policy(1)), ("bpf6553", zero_policy(1))])
            runners = {
                "policy": ForwardDiagnostic(h, n, m, r),
                "bpf512": ForwardDiagnostic(h, n, 1, r),
                "bpf6553": ForwardDiagnostic(h, 6553, 1, r)}
            yy = tf.repeat(y[:, None], BATCH, axis=1)
            control_noise = runners["policy"].noise(tf.constant([args.seed_base+2, 0], tf.int32))
            control_a = runners["policy"].run(*zero_policy(m), yy, *control_noise)
            control_b = runners["bpf512"].run(*zero_policy(1), yy, *control_noise)
            identity_error = max(scalar(tf.reduce_max(tf.abs(control_a[k]-control_b[k])))
                for k in ("logZ", "filtering_mean", "mean_ess_fraction"))
            flag("zero_policy_bpf_identity", identity_error <= 1e-10
                and bool(control_a["valid"].numpy()) and bool(control_b["valid"].numpy()),
                max_absolute_error=identity_error)

            values, means, ess_values = [], [], []
            with (args.output/"rows.jsonl").open("w") as rows:
                for name, policy in arms:
                    runner = runners[name] if name.startswith("bpf") else runners["policy"]
                    arm_values, arm_means, arm_ess = [], [], []
                    before = time.monotonic()
                    base_seed = args.seed_base+1 if name == "bpf6553" else args.seed_base
                    for batch in range(repetitions//BATCH):
                        noise = runner.noise(tf.constant([base_seed, batch], tf.int32))
                        result = runner.run(*policy, yy, *noise)
                        flag("forward_valid_%s_%d" % (name, batch),
                            bool(result["valid"].numpy()) and
                            bool(tf.reduce_all(tf.math.is_finite(result["logZ"])).numpy()))
                        arm_values.append(result["logZ"])
                        arm_means.append(tf.transpose(result["filtering_mean"]))
                        arm_ess.append(result["mean_ess_fraction"])
                        logs, fm, es = (result["logZ"].numpy().tolist(),
                            tf.transpose(result["filtering_mean"]).numpy().tolist(),
                            result["mean_ess_fraction"].numpy().tolist())
                        for row in range(BATCH):
                            rows.write(json.dumps(dict(arm=name, repetition=batch*BATCH+row,
                                seed_base=base_seed, batch=batch, batch_row=row,
                                logZ=logs[row], filtering_mean=fm[row], mean_ess_fraction=es[row]),
                                allow_nan=False)+"\n")
                        rows.flush()
                    values.append(tf.concat(arm_values, 0))
                    means.append(tf.concat(arm_means, 0))
                    ess_values.append(tf.concat(arm_ess, 0))
                    progress.append(dict(arm=name, repetitions=repetitions,
                        seconds=time.monotonic()-before))
                    save_json(args.output/"progress.json", progress)
                    print(json.dumps(progress[-1]), flush=True)
            logz = tf.stack(values)
            fm = tf.stack(means)
            @tf.function(input_signature=[tf.TensorSpec([6, repetitions], D)],
                jit_compile=True, autograph=False)
            def uncertainty(log_values):
                ratios = tf.exp(log_values-oracle["logZ"][-1])
                chunks = (BOOTSTRAPS+127)//128
                array = tf.TensorArray(D, chunks, element_shape=[6, 128])
                def body(i, output):
                    indices = tf.random.stateless_uniform([128, repetitions],
                        tf.stack([tf.constant(args.seed_base+9, tf.int32), i]),
                        minval=0, maxval=repetitions, dtype=tf.int32)
                    means = tf.reduce_mean(tf.gather(ratios, indices, axis=1), -1)
                    return i+1, output.write(i, means)
                _, array = tf.while_loop(lambda i, _: i < chunks, body,
                    (tf.constant(0), array), parallel_iterations=1)
                boot = tf.reshape(tf.transpose(array.stack(), [1, 0, 2]), [6, chunks*128])[:, :BOOTSTRAPS]
                ordered = tf.sort(boot, axis=-1)
                lower = math.floor(BOOTSTRAPS*.01/(2*6))
                upper = min(BOOTSTRAPS-1, math.ceil(BOOTSTRAPS*(1-.01/(2*6))))
                average = tf.reduce_mean(ratios, -1)
                variance = tf.math.reduce_variance(ratios, -1)*repetitions/(repetitions-1)
                return ratios, average, tf.sqrt(variance/repetitions), ordered[:, lower], ordered[:, upper]

            ones = uncertainty(tf.fill([6, repetitions], oracle["logZ"][-1]))
            flag("bootstrap_all_ones_control",
                all(bool(tf.reduce_all(value == 1).numpy()) for value in (ones[0], ones[1], ones[3], ones[4]))
                and bool(tf.reduce_all(ones[2] == 0).numpy()))
            ratios, average, mcse, low, high = uncertainty(logz)
            flag("finite_normalizer_ratios", bool(tf.reduce_all(tf.math.is_finite(ratios)).numpy()))
            statistics = []
            for i, (name, _) in enumerate(arms):
                lo, hi, mu = scalar(low[i]), scalar(high[i]), scalar(average[i])
                classification = ("DISCREPANCY_REQUIRES_DIAGNOSIS" if not lo <= 1 <= hi
                    else "RESOLVED_WITHIN_TEN_PERCENT" if lo >= .9 and hi <= 1.1
                    else "COMPATIBLE_UNDERRESOLVED")
                statistics.append(dict(arm=name, mean_normalizer_ratio=mu, mcse=scalar(mcse[i]),
                    simultaneous99_bootstrap_interval=[lo, hi], classification=classification,
                    min_ratio=scalar(tf.reduce_min(ratios[i])),
                    max_ratio=scalar(tf.reduce_max(ratios[i])),
                    logZ_mean=scalar(tf.reduce_mean(logz[i])),
                    logZ_sd=scalar(tf.math.reduce_std(logz[i])),
                    mean_ess_fraction=scalar(tf.reduce_mean(ess_values[i]))))
            @tf.function(input_signature=[tf.TensorSpec([6, repetitions, h], D)],
                jit_compile=True, autograph=False)
            def conditional_errors(filtering_means):
                masks = tf.stack([y < 5, (y >= 5) & (y < 15), y >= 15])
                count = tf.reduce_sum(tf.cast(masks, D), -1)
                squared = (filtering_means-oracle["mean"][None, None, :])**2
                mse = tf.einsum("arh,ch->ac", squared, tf.cast(masks, D))/(repetitions*count[None, :])
                zero = tf.einsum("h,ch->c", oracle["mean"]**2, tf.cast(masks, D))/count
                return count, mse, zero
            counts, mse, zero = conditional_errors(fm)
            flag("all_observation_situations_present", bool(tf.reduce_all(counts > 0).numpy()))
            heuristic, losses = [], []
            for j, situation in enumerate(("y_lt5", "y_5_to15", "y_ge15")):
                baseline = {"bpf512": scalar(mse[4, j]), "bpf6553": scalar(mse[5, j]),
                    "zero_mean": scalar(zero[j]), "quadrature_oracle": 0.}
                for i, (name, _) in enumerate(arms):
                    loss = scalar(mse[i, j])
                    losing = [key for key in ("bpf512", "bpf6553", "zero_mean")
                        if i < 4 and loss > baseline[key]]
                    row = dict(arm=name, situation=situation, times=int(counts[j].numpy()),
                        filtering_mean_mse=loss, heuristics=baseline,
                        observed_underperformance=losing, ranking="DESCRIPTIVE_ONLY")
                    heuristic.append(row)
                    if losing: losses.append(dict(arm=name, situation=situation, versus=losing))
            traces = {k: dict(wrapper=v.run.experimental_get_tracing_count(),
                consumer=v.filter.forward.experimental_get_tracing_count(),
                noise=v.noise.experimental_get_tracing_count()) for k, v in runners.items()}
            flag("bounded_tracing", all(v == 1 for lane in traces.values() for v in lane.values()))
            summary.update(status="PASS", oracle_terminal_logZ=scalar(oracle["logZ"][-1]),
                normalizer_comparisons=statistics, conditional_heuristics=heuristic,
                heuristic_dominance_verdict="OBSERVED_UNDERPERFORMANCE" if losses else "NO_OBSERVED_LOSS",
                observed_heuristic_losses=losses, ranking="NO_STATISTICALLY_SUPPORTED_RANKING",
                traces=traces, progress=progress, repetitions=repetitions,
                inference_scope="CONDITIONAL_ON_FOUR_FROZEN_POLICIES",
                call_chain="ForwardDiagnostic._run -> snapshotted RBFControlledSMC.forward -> _forward",
                seeds=dict(common512=args.seed_base, bpf6553=args.seed_base+1,
                    identity_control=args.seed_base+2, bootstrap=args.seed_base+9),
                nonclaims=["No paired fitting equivalence", "No unbiasedness proof from Monte Carlo",
                    "No independent fitting repetitions", "No Table2 replication",
                    "No other noise regimes", "No default readiness or superiority"])
    except Exception as exc:
        summary.update(status="FAIL", failure=type(exc).__name__+": "+str(exc), progress=progress)
    finally:
        summary.update(wall_seconds=time.monotonic()-started, tensorflow=tf.__version__,
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save_json(args.output/"summary.json", summary)
        print(json.dumps({k: summary.get(k) for k in ("status", "failure", "wall_seconds")}), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
