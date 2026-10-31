"""Diagnostic-only full-retention backward refit on two frozen training clouds."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
# This import configures GPU growth before any device initialization.
import diagnose_rbf_fixed_policy_tf as reference
import tensorflow as tf
import rbf_nnls_tf as fitting

D = tf.float64

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    fixture = json.loads(args.fixture.read_text())
    checks, extra = [], {}
    def flag(name, passed, **diagnostics):
        checks.append(dict(name=name, passed=bool(passed), **diagnostics))
    def save(name, value):
        tf.io.write_file(str(args.output/("fit_"+name+".tensor")), tf.io.serialize_tensor(value))
    try:
        h, n, b = int(fixture["observations"]), int(fixture["particles"]), len(fixture["cases"])
        flag("frozen_scope", (h, n, b, fixture["observation_variance"], fixture["bandwidth_factor"]) == (101, 512, 2, 1., .3))
        # Imported helpers must equal their preserved source snapshots.
        for function in (fitting.rbf_gaussian_mixture, fitting.multimodal_transition):
            actual = Path(inspect.getfile(function))
            snapshot = Path(__file__).with_name(actual.name)
            flag("source_snapshot_"+function.__name__, actual.read_bytes() == snapshot.read_bytes())
        if not all(c["passed"] for c in checks):
            raise RuntimeError("Scope or dependency snapshot mismatch")
        with tf.device("/GPU:0"):
            cloud = tf.stack([tf.constant(c["rounds"][0]["x"], D) for c in fixture["cases"]], 1)
            y = tf.repeat(tf.constant(fixture["y"], D)[:, None], b, axis=1)
            fitter = fitting.RBFBackwardFit(observations=h, particles=n, replicas=b,
                components=n, observation_variance=1., bandwidth_factor=.3,
                capacity=128, stopping_factor=.1, jit_compile=True)
            before = time.monotonic()
            fitted = fitter.fit(cloud, y)
            fitted["valid"].numpy()
            extra["fit_first_call_seconds"] = time.monotonic()-before
            for name, value in fitted.items():
                save(name, value)
            flag("backward_fit_valid", bool(fitted["valid"].numpy()))
            extra["fit_diagnostics"] = {k: fitted[k].numpy().tolist() for k in
                ("iterations", "peak_active", "removed", "dual_error", "complementarity", "minimum_qr_margin", "rss")}
            extra["peak_active"] = int(tf.reduce_max(fitted["peak_active"]).numpy())
            extra["max_iterations"] = int(tf.reduce_max(fitted["iterations"]).numpy())
            extra["max_dual_error"] = float(tf.reduce_max(fitted["dual_error"]).numpy())
            extra["max_complementarity"] = float(tf.reduce_max(fitted["complementarity"]).numpy())
            extra["minimum_qr_margin"] = float(tf.reduce_min(fitted["minimum_qr_margin"]).numpy())
            if not checks[-1]["passed"]:
                raise RuntimeError("Full backward refit invalid; downstream evaluation prohibited")

            scalar = tf.TensorSpec([h, b], D)
            matrix = tf.TensorSpec([h, b, n], D)
            @tf.function(input_signature=[matrix, scalar, scalar, matrix, matrix, matrix, scalar, matrix],
                jit_compile=True, autograph=False)
            def audit(x, observations, beta, coefficients, weights, knots, constant, target):
                # Independently recompute the source Gaussian product and the
                # recursion from unsorted coefficients and original particle knots.
                def body(t, target_error, constant_error, policy_error, all_retained, finite):
                    logg = -.5*(tf.math.log(tf.constant(2*math.pi, D))
                        +(observations[t, :, None]-x[t]**2/20)**2)
                    def lookahead():
                        mu = .5*x[t]+25*x[t]/(1+x[t]**2)+8*tf.cos(1.2*tf.cast(t+1, D))
                        denom = 1+20*beta[t+1]
                        return constant[t+1, :, None]-.5*tf.math.log(denom)[:, None]+tf.reduce_logsumexp(
                            tf.math.log(coefficients[t+1])[:, None, :]
                            -beta[t+1, :, None, None]/denom[:, None, None]
                            *(mu[:, :, None]-x[t+1, :, None, :])**2, -1)
                    logtarget = logg+tf.cond(t+1<h, lookahead, lambda: tf.zeros([b, n], D))
                    expected_constant = tf.reduce_max(logtarget, -1)
                    expected_target = tf.exp(logtarget-expected_constant[:, None])
                    full_log = tf.reduce_logsumexp(tf.math.log(coefficients[t])[:, None, :]
                        -beta[t, :, None, None]*(x[t, :, :, None]-x[t, :, None, :])**2, -1)
                    represented_log = tf.reduce_logsumexp(tf.math.log(weights[t])[:, None, :]
                        -beta[t, :, None, None]*(x[t, :, :, None]-knots[t, :, None, :])**2, -1)
                    return (t+1,
                        tf.maximum(target_error, tf.reduce_max(tf.abs(expected_target-target[t]))),
                        tf.maximum(constant_error, tf.reduce_max(tf.abs(expected_constant-constant[t]))),
                        tf.maximum(policy_error, tf.reduce_max(tf.abs(full_log-represented_log))),
                        all_retained & tf.reduce_all(tf.sort(coefficients[t], direction="DESCENDING") == weights[t]),
                        finite & tf.reduce_all(tf.math.is_finite(coefficients[t]) & (coefficients[t]>=0))
                        & tf.reduce_all(tf.reduce_sum(coefficients[t], -1)>0))
                result = tf.while_loop(lambda t,*_:t<h, body,
                    (tf.constant(0),tf.constant(0.,D),tf.constant(0.,D),tf.constant(0.,D),
                     tf.constant(True),tf.constant(True)), parallel_iterations=1)
                return result[1:]
            errors = audit(cloud, y, fitted["beta"], fitted["coefficients"],
                fitted["weights"], fitted["knots"], fitted["constant"], fitted["target"])
            for name, value in zip(("backward_target_identity", "backward_constant_identity", "full_policy_identity"), errors[:3]):
                error = float(value.numpy())
                flag(name, math.isfinite(error) and error<=1e-10, max_absolute_error=error, tolerance=1e-10)
            flag("every_coefficient_retained", bool(errors[3].numpy()))
            flag("finite_nonnegative_positive_mass", bool(errors[4].numpy()))
            traces = dict(fit=fitter.fit.experimental_get_tracing_count(), audit=audit.experimental_get_tracing_count())
            flag("bounded_tracing", all(v==1 for v in traces.values()))
            extra.update(traces=traces,
                full_backward_refit=True, retained_components=n,
                forward_positive_component_slots=int(tf.reduce_max(tf.reduce_sum(tf.cast(fitted["weights"]>0,tf.int32),-1)).numpy()),
                call_chain=["RBFBackwardFit.fit", "RBFBackwardFit._fit", "NonnegativeLeastSquares._solve"],
                fitting_source_sha256=hashlib.sha256(Path(inspect.getfile(fitting.RBFBackwardFit)).read_bytes()).hexdigest())
    except Exception as exc:
        extra["failure"] = repr(exc)
    summary = dict(status="PASS" if checks and all(c["passed"] for c in checks) and "failure" not in extra else "FAIL",
        scope="RBF_FULL_RETENTION_BACKWARD_REFIT_DIAGNOSTIC",replication="INCOMPLETE",promotion="NOT_REQUESTED",
        checks=checks,**extra,seeds=[int(c["seed"]) for c in fixture["cases"]],
        gpu_memory_policy=reference.MEMORY,tensorflow=tf.__version__,device="/device:GPU:0",
        dtype="float64",jit_compile=True,tf32=False,numpy_runtime=False,pfor=False,python_time_loop=False,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),wall_seconds=time.monotonic()-started,
        nonclaims=["No source-M16 or Table2 replication","Same two training clouds, no independent training replication",
                   "No downstream normalizer claim until the separate evaluation","No filtering/default/HMC promotion"])
    reference.save_json(args.output/"summary.json",summary)
    print(json.dumps({k:summary.get(k) for k in ("status","failure","wall_seconds","peak_active","max_iterations","max_dual_error","max_complementarity")}),flush=True)
    if summary["status"]!="PASS":
        raise SystemExit(1)

if __name__=="__main__":
    main()
