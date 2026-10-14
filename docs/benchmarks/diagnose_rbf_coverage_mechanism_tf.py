"""Deterministic diagnostic of saved RBF coverage before/after top16."""
import argparse
import json
import math
from pathlib import Path
import time
from diagnose_rbf_policy_coverage_tf import Coverage, clean
from diagnose_rbf_fixed_policy_tf import Quadrature, MEMORY, D
import tensorflow as tf


class KilledModel:
    def __init__(self, reference):
        self.ref = reference
        self.run = tf.function(self._run, input_signature=[
            tf.TensorSpec([reference.h], D),
            tf.TensorSpec([reference.h, 4, reference.n], tf.bool)],
            jit_compile=True, autograph=False)

    def _run(self, y, kill):
        ref = self.ref
        x, weight = ref.grid, ref.integration_weight
        def body(t, probability, logz, valid):
            def transition():
                mean = .5*x+25*x/(1+x*x)+8*tf.cos(1.2*tf.cast(t, D))
                density = tf.exp(-.5*(x[:, None]-mean[None, :])**2/10)
                density /= tf.sqrt(tf.constant(20*math.pi, D))
                return tf.linalg.matmul(probability, density, transpose_b=True)
            prediction = tf.cond(t == 0,
                lambda: tf.broadcast_to(tf.exp(-.5*x*x/5)/tf.sqrt(tf.constant(10*math.pi, D)), [4, ref.n]),
                transition)
            logg = -.5*(tf.math.log(tf.constant(2*math.pi, D))+(y[t]-x*x/20)**2)
            shift = tf.reduce_max(logg)
            unnormalized = prediction*weight*tf.exp(logg-shift)*tf.cast(~kill[t], D)
            mass = tf.reduce_sum(unnormalized, -1)
            ok = tf.reduce_all(tf.math.is_finite(mass) & (mass > 0))
            safe_mass = tf.where(mass > 0, mass, tf.ones_like(mass))
            return t+1, unnormalized/safe_mass[:, None], logz+tf.math.log(safe_mass)+shift, valid & ok
        final = tf.while_loop(lambda t, *_: t < ref.h, body,
            (tf.constant(0), tf.zeros([4, ref.n], D), tf.zeros([4], D), tf.constant(True)),
            parallel_iterations=1)
        return dict(logZ=final[2], valid=final[3])


def count_signature(h, g):
    return [tf.TensorSpec([h, 4, g], D), tf.TensorSpec([h, 4, 512], D)]


def counter(reference):
    # Interpolation is explanatory; compare counts at both grid resolutions.
    def count(logratio, cloud):
        position = (cloud+reference.bound)*(reference.n-1)/(2*reference.bound)
        low = tf.cast(tf.floor(position), tf.int32)
        valid = tf.reduce_all((low >= 0) & (low < reference.n-1))
        low = tf.clip_by_value(low, 0, reference.n-2)
        fraction = position-tf.cast(low, D)
        a = tf.gather(logratio, low, axis=2, batch_dims=2)
        b = tf.gather(logratio, low+1, axis=2, batch_dims=2)
        interp = a+(b-a)*fraction
        counts = tf.reduce_sum(tf.cast(interp < tf.math.log(tf.constant(1e-8, D)), tf.int32), -1)
        return dict(count=counts, valid=valid)
    return tf.function(count, input_signature=count_signature(reference.h, reference.n),
        jit_compile=True, autograph=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "fixture", "candidate", "coverage"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    summary = dict(status="RUNNING", scope="SAVED_RBF_COVERAGE_MECHANISM_DIAGNOSTIC",
        replication="INCOMPLETE", gpu_memory_policy=MEMORY, dtype="float64",
        tf32=False, jit_compile=True, numpy_runtime=False, pfor=False, cpu_only=False,
        seeds="N/A: deterministic saved-policy analysis", checks=[])
    def flag(name, passed, **details):
        summary["checks"].append(dict(name=name, passed=bool(passed), **details))
        if not passed: raise RuntimeError(name)
    try:
        fixture = json.loads(args.fixture.read_text())
        previous = json.loads((args.coverage/"summary.json").read_text())
        flag("fixed_scope", previous["status"] == "PASS" and fixture["observations"] == 101
            and fixture["particles"] == 512 and fixture["observation_variance"] == 1.)
        with tf.device("/GPU:0"):
            raw = {k: tf.io.parse_tensor(tf.io.read_file(str(args.candidate/("fit_"+k+".tensor"))), D)
                for k in ("beta", "weights", "knots", "constant", "coefficients")}
            y = tf.constant(fixture["y"], D)
            retained, full, clouds, names = [], [], [], []
            for i, case in enumerate(fixture["cases"]):
                cloud = tf.constant(case["rounds"][0]["x"], D)
                reference = {k: tf.constant(case["rounds"][1][k], D)
                    for k in ("beta", "weights", "knots", "constant")}
                for label, fields, coefficients in (
                    ("qr", {k: raw[k][:, i] for k in reference}, raw["coefficients"][:, i]),
                    ("svd", reference, tf.constant(case["full_coefficients"], D))):
                    names.append(label+"_seed"+str(int(case["seed"])))
                    retained.append([fields[k] for k in reference])
                    full.append([fields["beta"], coefficients, cloud, fields["constant"]])
                    clouds.append(cloud)
                    selected = tf.math.top_k(coefficients, k=16, sorted=True)
                    weight_error = float(tf.reduce_max(tf.abs(selected.values-fields["weights"])).numpy())
                    selected_knots = tf.gather(cloud, selected.indices, axis=1, batch_dims=1)
                    # Zero-weight ties do not identify a kernel; compare only positive entries.
                    knot_error = float(tf.reduce_max(tf.where(selected.values > 0,
                        tf.abs(selected_knots-fields["knots"]), tf.zeros_like(selected_knots))).numpy())
                    flag("top16_reconstruction_"+names[-1], weight_error <= 1e-10 and knot_error <= 1e-10,
                        weight_error=weight_error, positive_knot_error=knot_error)
            policies = {key: [tf.stack([row[i] for row in values], 1) for i in range(4)]
                for key, values in (("retained", retained), ("full", full))}
            cloud = tf.stack(clouds, 1)
            grid_results = []
            for points in (2049, 4097):
                reference = Quadrature(101, points, 80., 1.)
                oracle = reference.run(y)
                killed = KilledModel(reference)
                control = killed.run(y, tf.zeros([101, 4, points], tf.bool))
                control_error = float(tf.reduce_max(tf.abs(control["logZ"]-oracle["logZ"][-1])).numpy())
                flag("no_kill_oracle_"+str(points), bool(control["valid"].numpy()) and control_error <= 1e-7,
                    max_logZ_error=control_error)
                count = counter(reference)
                grid = dict(points=points, arms={})
                for kind, components in (("retained", 16), ("full", 512)):
                    diagnosis = Coverage(reference, components=components)
                    result = diagnosis.run(y, oracle["prediction"], oracle["probability"], *policies[kind])
                    finite = all(bool(tf.reduce_all(tf.math.is_finite(result[k])).numpy())
                        for k in ("posterior_mass", "proposal_mass", "filtering_mass", "log_q_to_smoothing"))
                    back_error = float(tf.abs(result["backward_logZ"]-oracle["logZ"][-1]).numpy())
                    analytic_error = float(result["initial_analytic_error"].numpy())
                    flag(kind+"_identities_"+str(points), finite and back_error <= 1e-7
                        and analytic_error <= 1e-8 and bool(result["analytic_valid"].numpy()),
                        forward_backward_error=back_error, initial_analytic_error=analytic_error)
                    if kind == "retained":
                        old = json.loads((args.coverage/("grid"+str(points)+".json")).read_text())
                        error = max(float(tf.reduce_max(tf.abs(result[k]-tf.constant(old[k], D))).numpy())
                            for k in ("posterior_mass", "proposal_mass", "filtering_mass"))
                        flag("replay62_"+str(points), error <= 1e-10, max_mass_error=error)
                    mask = result["log_q_to_smoothing"] < tf.math.log(tf.constant(1e-8, D))
                    survival = killed.run(y, mask)
                    probability = 1-tf.exp(survival["logZ"]-oracle["logZ"][-1])
                    counts = count(result["log_q_to_smoothing"], cloud)
                    flag(kind+"_valid_paths_"+str(points), bool(survival["valid"].numpy())
                        and bool(counts["valid"].numpy()) and bool(tf.reduce_all(tf.math.is_finite(probability)
                        & (probability >= -1e-10) & (probability <= 1+1e-10)).numpy()))
                    flag(kind+"_one_trace_"+str(points), diagnosis.run.experimental_get_tracing_count() == 1)
                    rows = []
                    for i, name in enumerate(names):
                        t = int(tf.argmax(result["posterior_mass"][:, i, 2]).numpy())
                        rows.append(dict(arm=name, worst_time=t,
                            posterior_path_mass=float(probability[i].numpy()),
                            posterior_mass=float(result["posterior_mass"][t, i, 2].numpy()),
                            proposal_mass=float(result["proposal_mass"][t, i, 2].numpy()),
                            filtering_mass=float(result["filtering_mass"][t, i, 2].numpy()),
                            fitting_particles_in_region=int(counts["count"][t, i].numpy())))
                    grid["arms"][kind] = rows
                    # Preserve complete time diagnostics without emitting them in tool output.
                    (args.output/(kind+"_grid"+str(points)+".json")).write_text(json.dumps(clean(dict(
                        rows=rows, posterior_mass=result["posterior_mass"][:, :, 2].numpy().tolist(),
                        proposal_mass=result["proposal_mass"][:, :, 2].numpy().tolist(),
                        filtering_mass=result["filtering_mass"][:, :, 2].numpy().tolist(),
                        fitting_particle_counts=counts["count"].numpy().tolist())), allow_nan=False)+"\n")
                flag("shared_graphs_once_"+str(points), reference.run.experimental_get_tracing_count() == 1
                    and killed.run.experimental_get_tracing_count() == 1 and count.experimental_get_tracing_count() == 1)
                grid_results.append(grid)
                print(json.dumps(dict(points=points, rows=grid["arms"])), flush=True)
            summary.update(status="PASS", grids=grid_results, device=oracle["logZ"].device,
                criterion="Saved-policy replay and numerical identities; mechanism quantities explanatory only",
                nonclaims=["No full-basis backward refit", "No finite-particle hitting law",
                    "No QR/SVD quality ranking", "No fitting or default promotion"])
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
