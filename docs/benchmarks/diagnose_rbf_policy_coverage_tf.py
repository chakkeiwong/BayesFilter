"""Deterministic diagnostic of fixed RBF proposal coverage; no fitting."""
import argparse
import json
import math
from pathlib import Path
import time
from diagnose_rbf_fixed_policy_tf import Quadrature, core, MEMORY, D
import tensorflow as tf


class Coverage:
    def __init__(self, reference, components=16):
        self.reference = reference
        self.h, self.n = reference.h, reference.n
        policy = tf.TensorSpec([self.h, 4, components], D)
        field = tf.TensorSpec([self.h, 4], D)
        grid = tf.TensorSpec([self.h, self.n], D)
        self.run = tf.function(self._run, input_signature=[
            tf.TensorSpec([self.h], D), grid, grid, field, policy, policy, field],
            jit_compile=True, autograph=False)

    def _run(self, y, prediction, filtering_probability, beta, weights, knots, constant):
        x = self.reference.grid
        logw = tf.math.log(self.reference.integration_weight)
        def logg(t):
            return -.5*(tf.math.log(tf.constant(2*math.pi, D))+(y[t]-x*x/20)**2)
        terminal = logg(self.h-1)
        array = tf.TensorArray(D, self.h).write(self.h-1, terminal)
        def backward(t, likelihood, output):
            mean = .5*x+25*x/(1+x*x)+8*tf.cos(1.2*tf.cast(t+1, D))
            logf = -.5*(x[:, None]-mean[None, :])**2/10-.5*tf.math.log(tf.constant(20*math.pi, D))
            likelihood = logg(t)+tf.reduce_logsumexp(logf+likelihood[:, None]+logw[:, None], 0)
            return t-1, likelihood, output.write(t, likelihood)
        _, first, array = tf.while_loop(lambda t, *_: t >= 0, backward,
            (tf.constant(self.h-2), terminal, array), parallel_iterations=1)
        likelihood = array.stack()
        prior_log = -.5*x*x/5-.5*tf.math.log(tf.constant(10*math.pi, D))
        backward_logz = tf.reduce_logsumexp(prior_log+first+logw)
        thresholds = tf.constant([1e-4, 1e-6, 1e-8, 1e-12, 1/(4096*512)], D)
        arrays = tuple(tf.TensorArray(D, self.h) for _ in range(7))
        def body(t, output):
            log_predict = tf.math.log(prediction[t])
            target_unnormalized = log_predict+likelihood[t]+logw
            lognorm_target = tf.reduce_logsumexp(target_unnormalized)
            log_target = target_unnormalized-lognorm_target
            logpsi = constant[t, :, None]+tf.reduce_logsumexp(
                tf.math.log(weights[t])[:, None, :]-beta[t, :, None, None]
                *(x[None, :, None]-knots[t, :, None, :])**2, -1)
            proposal_unnormalized = log_predict[None, :]+logpsi+logw[None, :]
            lognorm_proposal = tf.reduce_logsumexp(proposal_unnormalized, -1)
            log_proposal = proposal_unnormalized-lognorm_proposal[:, None]
            # Cancel the shared predictive density before taking the ratio;
            # this remains defined even where both grid masses underflow.
            logratio = logpsi-likelihood[t][None, :]+lognorm_target-lognorm_proposal[:, None]
            region = logratio[:, None, :] < tf.math.log(thresholds)[None, :, None]
            posterior_mass = tf.reduce_sum(tf.where(region, tf.exp(log_target)[None, None, :], 0.), -1)
            proposal_mass = tf.reduce_sum(tf.where(region, tf.exp(log_proposal)[:, None, :], 0.), -1)
            filtering_mass = tf.reduce_sum(tf.where(region, filtering_probability[t][None, None, :], 0.), -1)
            log_proposal_mass = tf.reduce_logsumexp(tf.where(region, log_proposal[:, None, :],
                tf.constant(float("-inf"), D)), -1)
            mean = tf.reduce_sum(tf.exp(log_target)*x)
            output = tuple(a.write(t, v) for a, v in zip(output,
                (posterior_mass, proposal_mass, filtering_mass, log_proposal_mass, mean, lognorm_proposal, logratio)))
            return t+1, output
        _, arrays = tf.while_loop(lambda t, _: t < self.h, body,
            (tf.constant(0), arrays), parallel_iterations=1)
        out = {k: a.stack() for k, a in zip(("posterior_mass", "proposal_mass",
            "filtering_mass", "log_proposal_mass", "smoothing_mean", "proposal_log_normalizer", "log_q_to_smoothing"), arrays)}
        analytic = core.rbf_gaussian_mixture(tf.zeros([4, 1], D), tf.constant(5., D),
            beta[0], weights[0], knots[0], constant[0])
        out.update(backward_logZ=backward_logz,
            initial_analytic_error=tf.reduce_max(tf.abs(out["proposal_log_normalizer"][0]
                -analytic["log_integral"][:, 0])), analytic_valid=analytic["valid"])
        return out


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--precision", type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    summary = dict(status="RUNNING", scope="IDEAL_RBF_PROPOSAL_COVERAGE_DIAGNOSTIC",
        replication="INCOMPLETE", checks=[], cpu_only=False, numpy_runtime=False,
        pfor=False, jit_compile=True, tf32=False, dtype="float64",
        gpu_memory_policy=MEMORY, seeds="N/A: deterministic saved-policy quadrature")
    def flag(name, passed, **details):
        summary["checks"].append(dict(name=name, passed=bool(passed), **details))
        if not passed: raise RuntimeError(name)
    try:
        fixture = json.loads(args.fixture.read_text())
        precision = json.loads(args.precision.read_text())
        flag("scope", int(fixture["observations"]) == 101 and float(fixture["observation_variance"]) == 1.)
        with tf.device("/GPU:0"):
            y = tf.constant(fixture["y"], D)
            raw = {k: tf.io.parse_tensor(tf.io.read_file(str(args.candidate/("fit_"+k+".tensor"))), D)
                for k in ("beta", "weights", "knots", "constant")}
            values, names = [], []
            for i, case in enumerate(fixture["cases"]):
                values.append([raw[k][:, i] for k in raw])
                names.append("qr_seed%d" % int(case["seed"]))
                values.append([tf.constant(case["rounds"][1][k], D) for k in raw])
                names.append("svd_seed%d" % int(case["seed"]))
            policies = [tf.stack([v[i] for v in values], 1) for i in range(4)]
            all_results = []
            for points in (2049, 4097):
                reference = Quadrature(101, points, 80., 1.)
                quadrature = reference.run(y)
                diagnosis = Coverage(reference)
                result = diagnosis.run(y, quadrature["prediction"], quadrature["probability"], *policies)
                number = lambda v: float(v.numpy())
                ze = number(tf.abs(result["backward_logZ"]-precision["oracle_terminal_logZ"]))
                fbe = number(tf.abs(result["backward_logZ"]-quadrature["logZ"][-1]))
                mass = number(tf.reduce_max(tf.abs(quadrature["predictive_mass"]-1)))
                boundary = number(tf.reduce_max(quadrature["boundary_mass"]))
                analytic = number(result["initial_analytic_error"])
                flag("quadrature_valid_%d" % points, ze <= 1e-7 and fbe <= 1e-7
                    and mass <= 1e-10 and boundary <= 1e-12 and analytic <= 1e-8
                    and bool(result["analytic_valid"].numpy()),
                    backward_vs_oracle=ze, forward_backward_error=fbe,
                    predictive_mass_error=mass, boundary_mass=boundary,
                    initial_analytic_log_integral_error=analytic)
                flag("finite_mass_%d" % points, all(bool(tf.reduce_all(tf.math.is_finite(result[k])).numpy())
                    for k in ("posterior_mass", "proposal_mass", "filtering_mass", "smoothing_mean")))
                flag("one_trace_%d" % points, diagnosis.run.experimental_get_tracing_count() == 1
                    and reference.run.experimental_get_tracing_count() == 1)
                data = {k: v.numpy().tolist() for k, v in result.items()}
                (args.output/("grid%d.json" % points)).write_text(json.dumps(clean(data), allow_nan=False)+"\n")
                all_results.append(result)
                print(json.dumps(dict(points=points, backward_logZ=number(result["backward_logZ"]),
                    max_suppressed_smoothing_mass=number(tf.reduce_max(result["posterior_mass"])))), flush=True)
            mean_error = float(tf.reduce_max(tf.abs(all_results[0]["smoothing_mean"]-all_results[1]["smoothing_mean"])).numpy())
            flag("smoothing_mean_resolution", mean_error <= 1e-6, max_absolute_error=mean_error)
            worst = []
            for i, arm in enumerate(names):
                for j, threshold in enumerate((1e-4, 1e-6, 1e-8, 1e-12, 1/(4096*512))):
                    t = int(tf.argmax(all_results[1]["posterior_mass"][:, i, j]).numpy())
                    row = dict(arm=arm, ratio_threshold=threshold, time=t,
                        observation=float(y[t].numpy()), grids=[])
                    for points, result in zip((2049, 4097), all_results):
                        values = {k: float(result[k][t, i, j].numpy()) for k in
                            ("posterior_mass", "proposal_mass", "filtering_mass", "log_proposal_mass")}
                        values.update(points=points, ideal_expected_hits_4096x512=values["proposal_mass"]*4096*512)
                        row["grids"].append(values)
                    worst.append(row)
            summary.update(status="PASS", worst_regions=worst, arms=names, device=result["posterior_mass"].device,
                criterion="Deterministic identity/resolution checks only; regional mass is explanatory",
                nonclaims=["No exact finite-particle proposal law", "No finite-N hitting-probability proof",
                    "No unbiasedness or paper-validity verdict", "No fitting/default promotion"])
    except Exception as exc:
        summary.update(status="FAIL", failure=type(exc).__name__+": "+str(exc))
    finally:
        summary.update(wall_seconds=time.monotonic()-start, tensorflow=tf.__version__,
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        (args.output/"summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: summary.get(k) for k in ("status", "failure", "wall_seconds")}), flush=True)
    if summary["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
