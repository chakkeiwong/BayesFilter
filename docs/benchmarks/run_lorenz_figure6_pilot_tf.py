"""Bounded GPU/XLA Lorenz Figure6 fresh-seed validity and cost pilot."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, "/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import lorenz_refinement_tf as controller
D = tf.float64


def random_design(particles, stages):
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def draw(seed):
        normal = tf.random.stateless_normal([stages, 101, 1, particles, 8], seed, dtype=D)
        uniform = tf.random.stateless_uniform([stages, 101, 1],
                                              seed + tf.constant([0, 100000]), dtype=D)
        return normal, uniform
    return draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks, rows = [], []
    summary = dict(status="FAIL", scope="LORENZ_FIGURE6_COST_PILOT",
        replication="INCOMPLETE", device="GPU", jit_compile=True, dtype="float64",
        tf32=False, gpu_memory_policy=memory, tensorflow=tf.__version__,
        seed_base=202609370, checks=checks, rows=rows,
        heuristic_dominance_verdict="NOT_ASSESSED_FEASIBILITY_ONLY",
        nonclaims=["variance reduction", "GPU cost-matched comparison",
                   "Figure6 reproduction", "default readiness"])
    def flag(name, good, **detail):
        checks.append(dict(name=name, passed=bool(good), **detail))
        if not good:
            raise ValueError(name)
    def write_progress():
        (args.output / "progress.json").write_text(json.dumps(dict(
            completed_rows=len(rows), context=dict(completed_pairs=len(rows)), rows=rows),
            indent=2, allow_nan=False) + "\n")
    def checked_snapshot(cls):
        source = Path(inspect.getfile(cls))
        saved = Path(__file__).parent / source.name
        flag("snapshot_" + cls.__name__,
             hashlib.sha256(source.read_bytes()).hexdigest()
             == hashlib.sha256(saved.read_bytes()).hexdigest())
    try:
        checked_snapshot(controller.MultivariateGaussianControlledSMC)
        from bayesfilter.replication.controlled_smc_tf import systematic_ancestors
        checked_snapshot(systematic_ancestors)
        fixture = json.loads((args.reference / "fixture.json").read_text())
        flag("source_shape", fixture["metadata"]["horizon"] == 101
             and fixture["metadata"]["dimension"] == 8)
        y = tf.constant(fixture["metadata"]["y"], D)[:, None, :]
        controlled_random = random_design(512, 5)
        apf_random = {}
        for grid_index in (0, 15, 39):
            forcing = 2.5 + 6.0 * grid_index / 39
            controlled = controller.LorenzRefinement(forcing=forcing)
            apf_engines = {}
            for repeat in range(4):
                seed = [202609370, grid_index * 1000 + repeat * 10 + 1]
                before = time.monotonic()
                normal, uniform = controlled_random(tf.constant(seed, tf.int32))
                output = controlled.run(y, normal, uniform, tf.constant(4))
                count = int(output["iterations"].numpy())  # Synchronize all requested work.
                elapsed = time.monotonic() - before
                flag("controlled_valid_%d_%d" % (grid_index, repeat),
                     output["valid"].numpy() and 1 <= count <= 4)
                final_ess = output["history"]["ess"][count]
                margin = float(tf.reduce_min(output["history"]["precision_margin"][:count+1]).numpy())
                gram = float(tf.reduce_min(output["minimum_fit_gram_ratio"][1:count+1]).numpy())
                controlled_logz = float(output["normalized_logZ"][count, -1, 0].numpy())
                initial_logz = float(output["normalized_logZ"][0, -1, 0].numpy())
                flag("controlled_finite_%d_%d" % (grid_index, repeat),
                     math.isfinite(controlled_logz) and math.isfinite(initial_logz)
                     and math.isfinite(margin) and math.isfinite(gram))
                n = controller.apf_particles_for_refinements(count)
                cold_apf = n not in apf_engines
                if cold_apf:
                    apf_engines[n] = controller.LorenzRefinement(forcing=forcing, particles=n)
                if n not in apf_random:
                    apf_random[n] = random_design(n, 1)
                apf = apf_engines[n]
                policy = apf.initial_policy(y)
                apf_seed = [202609370, grid_index * 1000 + repeat * 10 + 2]
                before = time.monotonic()
                an, au = apf_random[n](tf.constant(apf_seed, tf.int32))
                forward = apf.engine.forward(*policy, y, an[0], au[0])
                flag("apf_valid_%d_%d" % (grid_index, repeat), forward["valid"].numpy())
                apf_seconds = time.monotonic() - before
                apf_logz = float(forward["logZ"][-1, 0].numpy()) - .5*6*101*math.log(2*math.pi*1e-4)
                flag("apf_finite_%d_%d" % (grid_index, repeat), math.isfinite(apf_logz))
                row = dict(grid_index=grid_index, forcing=forcing, repeat=repeat,
                    controlled_seed=seed, apf_seed=apf_seed, refinements=count,
                    apf_particles=n, controlled_cold=(repeat == 0), apf_cold=cold_apf,
                    controlled_seconds=elapsed, apf_seconds=apf_seconds,
                    controlled_logZ=controlled_logz, initial_apf512_logZ=initial_logz,
                    apf_logZ=apf_logz,
                    minimum_final_ess_fraction=float(tf.reduce_min(final_ess).numpy())/512,
                    minimum_precision_margin=margin, minimum_gram_ratio=gram)
                rows.append(row)
                write_progress()
            trace_counts = dict(controller=controlled.run.experimental_get_tracing_count(),
                forward=controlled.engine.forward.experimental_get_tracing_count(),
                fit=controlled.engine.fit.experimental_get_tracing_count())
            flag("controlled_single_traces_%d" % grid_index,
                 all(v == 1 for v in trace_counts.values()), counts=trace_counts)
            for n, engine in apf_engines.items():
                flag("apf_single_trace_%d_%d" % (grid_index, n),
                     engine.engine.forward.experimental_get_tracing_count() == 1)
            # Graphs are scoped to this forcing; collect before the next one.
            del controlled, apf_engines, apf, output, forward
            gc.collect()
        flag("controlled_random_single_trace", controlled_random.experimental_get_tracing_count() == 1)
        for n, draw in apf_random.items():
            flag("apf_random_single_trace_%d" % n, draw.experimental_get_tracing_count() == 1)
        flag("all_twelve_pairs", len(rows) == 12)
        warm_c = [r["controlled_seconds"] for r in rows if not r["controlled_cold"]]
        warm_a = [r["apf_seconds"] for r in rows if not r["apf_cold"]]
        flag("warm_timings_present", bool(warm_c) and bool(warm_a))
        max_cold_c = max(r["controlled_seconds"] for r in rows if r["controlled_cold"])
        max_cold_a = max(r["apf_seconds"] for r in rows if r["apf_cold"])
        forecast = 2 * (4000 * (max(warm_c) + max(warm_a))
                        + 40 * max_cold_c + 160 * max_cold_a)
        summary.update(status="PASS", forecast_gpu_seconds=forecast,
            timing_components=dict(max_warm_controlled=max(warm_c), max_warm_apf=max(warm_a),
                                   max_cold_controlled=max_cold_c, max_cold_apf=max_cold_a),
            interpretation="Feasibility and fresh-seed validity only; no empirical ranking.")
    except Exception as error:
        summary["error"] = repr(error)
        write_progress()
    summary["wall_seconds"] = time.monotonic() - started
    summary["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=summary["status"], pairs=len(rows),
                         forecast_gpu_seconds=summary.get("forecast_gpu_seconds"),
                         error=summary.get("error"), wall_seconds=summary["wall_seconds"])))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
