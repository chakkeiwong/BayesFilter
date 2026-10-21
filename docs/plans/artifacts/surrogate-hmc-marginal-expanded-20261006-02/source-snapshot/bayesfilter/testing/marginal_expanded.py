"""Expanded diagnostic comparison support; not a tuning/admission authority.

TensorFlow executes filters and references. SciPy/stdlib statistics here are
post-run diagnostic analysis only. No reference derivative is fed to LEDH.
"""
from __future__ import annotations
import importlib.util
import math
import statistics
from pathlib import Path

import tensorflow as tf
from bayesfilter.testing import marginal_weight_campaign as base

POLICIES = ("ancestor", "marginal_mixture")
HORIZONS = (10, 20, 50, 120, 240)
DIMENSIONS = (1, 3, 10, 20)


def scope_names():
    return tuple(f"lg{d}-t{t}" for t in HORIZONS for d in DIMENSIONS) + tuple(
        f"{model}-t{t}" for t in HORIZONS for model in ("ksc", "m13"))


def compile_batch(spec, n, controls, policy, rows=1, dtype=tf.float32):
    """Exercise the real public multi-row consumer with all safety outputs."""
    d, p, t, m = (spec[k] for k in ("d", "p", "t", "m"))
    covariance = tf.cast(spec["fixture"].P0, dtype) if spec["kind"] == "m13" else tf.eye(d, dtype=dtype)
    covariance = tf.broadcast_to(covariance, [n, d, d])
    signature = [tf.TensorSpec([rows, p], dtype), tf.TensorSpec([n, d], dtype),
                 tf.TensorSpec([t, n, d], dtype), tf.TensorSpec([n, d], dtype),
                 tf.TensorSpec([t, m], dtype)]
    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate(theta, initial, innovations, reset, observations):
        value, score, status = base.canonical_batch_fused_value_score(
            spec["model"], theta, tf.broadcast_to(tf.eye(p, dtype=dtype), [rows, p, p]),
            initial, covariance, innovations, observations,
            importance_weight_policy=policy, reset_design=reset,
            return_filter_moments=True, return_all_correction_diagnostics=True, **controls)
        # Summaries retain computed safety diagnostics without serializing all
        # filtering covariance matrices at every design. Failures retain codes.
        diagnostics = {}
        for key, tensor in status.items():
            if key in ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance"):
                continue
            numeric = tf.cast(tensor, tf.float64)
            diagnostics[key] = dict(minimum=tf.reduce_min(numeric), maximum=tf.reduce_max(numeric),
                                    finite=tf.reduce_all(tf.math.is_finite(numeric)))
        valid = status["program_valid"] & tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score), axis=1)
        return dict(value=value, score=score, valid=valid, diagnostics=diagnostics)
    return evaluate


def comparator_module():
    path = Path(__file__).resolve().parents[2]/"scripts/diagnose_marginal_weight_campaign.py"
    module_spec = importlib.util.spec_from_file_location("expanded_heuristics", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class Kernels:
    """Stable signatures permit reuse on fresh observations without retracing."""
    def __init__(self, spec):
        self.spec = spec
        self.spec64 = base.specification(spec["name"], tf.float64)
        self.cache = {}

    def get(self, kind, n=0, extent=24., replicates=16, dtype=tf.float64):
        """Construct model metadata outside tracing; keep observations dynamic."""
        dtype = tf.as_dtype(dtype)
        key = (kind, n, extent, replicates, dtype.name)
        if key in self.cache:
            return self.cache[key]
        if kind in ("ksc", "fisher") and dtype != tf.float64:
            raise ValueError("independent integration/Fisher authorities require FP64")
        spec = self.spec64 if dtype == tf.float64 else self.spec
        options = dict(dynamic_observations=True)
        helper = comparator_module() if kind in ("prior", "gaussian") else None
        if kind == "kalman":
            raw = base.kalman_reference(spec, None, dtype=dtype, **options)
        elif kind == "ksc":
            raw = base.ksc_grid_reference(spec, None, n, extent, **options)
        elif kind == "fisher":
            raw = base.m13_fisher_reference(spec, None, n, replicates, **options)
        elif kind == "bootstrap":
            raw = base.bootstrap_reference(spec, None, n, dtype=dtype, **options)
        elif kind == "prior":
            raw = helper.prior_trajectory_reference(tf, base, spec, None, n, dtype=dtype, **options)
        elif kind == "gaussian":
            raw = helper.gaussian_reference(tf, base, spec, None, dtype=dtype, **options)
        else:
            raise ValueError(kind)
        signature = [tf.TensorSpec([spec["p"]], dtype),
                     tf.TensorSpec([spec["t"], spec["m"]], dtype), tf.TensorSpec([2], tf.int32)]
        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def function(theta, observed, seed):
            if kind in ("kalman", "ksc", "gaussian"):
                return raw(theta, observed)
            if kind == "fisher":
                return raw(theta, observed, seed)
            return dict(value=raw(theta, observed, seed))
        self.cache[key] = function
        return function


def finite(result):
    return all(math.isfinite(float(x)) for x in [result["value"]]+result.get("score", []))


def one(result):
    """Convert a one-row consumer output without dropping validity diagnostics."""
    return dict(value=result["value"][0], score=result["score"][0],
                valid=result["valid"][0], diagnostics=result["diagnostics"],
                seconds=result["seconds"])


def references(kernels, observed, dataset_id, partition, evaluate, claim):
    spec = kernels.spec
    kinds = ("calibration", "validation", "claim", "engineering")
    seed_base = 20261006 + 7919*(dataset_id+1) + 101*kinds.index(partition)
    points = [tf.constant(p, tf.float64) for p in spec["points"]]
    def run(kind, n=0, extent=24., reps=16):
        function = kernels.get(kind, n, extent, reps)
        return [evaluate(function, p, observed, tf.constant([seed_base, n+1009*j], tf.int32))
                for j, p in enumerate(points)]
    if spec["kind"].startswith("lg"):
        result = run("kalman")
        return dict(rows=result, valid=all(finite(r) for r in result), kind="exact_Kalman_reference_AD_only",
                    uncertainty=[dict(value=0., score=0.) for _ in result])
    if spec["kind"] == "ksc":
        levels = [(512, 18.), (1024, 24.)] if claim else [(512, 18.)]
        history = []
        for n, extent in levels:
            history.append(dict(cells=n, extent=extent, rows=run("ksc", n, extent)))
        if claim:
            deltas = discrepancies(history[-2]["rows"], history[-1]["rows"])
            if any(max(x.values()) > .001 for x in deltas):
                history.append(dict(cells=2048, extent=32., rows=run("ksc", 2048, 32.)))
        rows = history[-1]["rows"]
        uncertainty = discrepancies(history[-2]["rows"], rows) if len(history)>1 else [dict(value=0., score=0.) for _ in rows]
        return dict(rows=rows, valid=all(finite(r) for r in rows) and all(max(x.values())<.001 for x in uncertainty),
                    kind="refined_1D_grid_analytical_score", refinements=history, uncertainty=uncertainty)
    levels = (65536, 131072) if claim else (8192,)
    history = [dict(particles=n, rows=run("fisher", n, reps=16 if claim else 8)) for n in levels]
    def sensitivity():
        rows = history[-1]["rows"]
        shift = discrepancies(history[-2]["rows"], rows) if len(history)>1 else [dict(value=0., score=0.) for _ in rows]
        return [dict(value=s["value"]+3*r["mcse"][0],
                     score=s["score"]+.3*math.sqrt(sum(x*x for x in r["mcse"][1:])/spec["p"]))
                for s, r in zip(shift, rows)]
    uncertainty = sensitivity()
    if claim and any(max(r.values())>.025 for r in uncertainty):
        history.append(dict(particles=262144, rows=run("fisher", 262144)))
        uncertainty = sensitivity()
    rows = history[-1]["rows"]
    return dict(rows=rows, valid=all(finite(r) for r in rows), exact=False,
                kind="replicated_PF_Fisher", refinements=history, uncertainty=uncertainty,
                qualification="refinement plus 3 MCSE is sensitivity, not a bound on remaining bias")


def discrepancies(left, right):
    return [dict(value=abs(a["value"]-b["value"]),
                 score=.1*math.sqrt(sum((x-y)**2 for x,y in zip(a["score"],b["score"]))/len(b["score"])))
            for a,b in zip(left,right)]


def summary(datasets, point_count, family_cells=255):
    """Cluster at dataset, with a predeclared family/metric/look/tail correction."""
    from scipy.stats import t as student_t
    alpha = .05/(family_cells*2*3*2)
    output = []
    for point in range(point_count):
        result = dict(point=point, datasets=len(datasets), metrics={}, heuristic={})
        records = [d["points"][point] for d in datasets]
        valid = (len(datasets) >= 4 and all(d["reference"]["valid"] for d in datasets)
                 and all(len(record["rows"]) == 32 and all(row[p]["valid"] and finite(row[p])
                     for row in record["rows"] for p in POLICIES) for record in records))
        result["valid"] = valid
        if not valid:
            result["verdict"] = "INVALID_CANDIDATE"
            output.append(result)
            continue
        for metric in ("value", "score"):
            cluster = []
            old_means, new_means, sensitivities = [], [], []
            for dataset, record in zip(datasets, records):
                ref = dataset["reference"]["rows"][point]
                old = [base.errors(r["ancestor"],ref)[metric] for r in record["rows"]]
                new = [base.errors(r["marginal_mixture"],ref)[metric] for r in record["rows"]]
                cluster.append(statistics.fmean(a-b for a,b in zip(new,old)))
                old_means.append(statistics.fmean(old)); new_means.append(statistics.fmean(new))
                sensitivities.append(2*dataset["reference"]["uncertainty"][point][metric])
            mean = statistics.fmean(cluster)
            se = statistics.stdev(cluster)/math.sqrt(len(cluster))
            critical = float(student_t.ppf(1-alpha,len(cluster)-1))
            lo, hi = mean-critical*se, mean+critical*se
            sensitivity = statistics.fmean(sensitivities)
            tensor = tf.constant(cluster,tf.float64)
            indices = tf.random.stateless_uniform([50000,len(cluster)],[20261006,point+(metric=="score")],
                                                  maxval=len(cluster),dtype=tf.int32)
            boot = tf.sort(tf.reduce_mean(tf.gather(tensor,indices),axis=1))
            boot95 = [float(boot[1249]),float(boot[48749])]
            loo = [(sum(cluster)-x)/(len(cluster)-1) for x in cluster]
            robust_upper, robust_lower = hi+sensitivity, lo-sensitivity
            noninferior = robust_upper < .05 and boot95[1] < .05 and max(loo) < .05
            harm = robust_lower > .05 and boot95[0] > .05 and min(loo) > .05
            result["metrics"][metric] = dict(baseline_mean=statistics.fmean(old_means),
                candidate_mean=statistics.fmean(new_means), difference_mean=mean, dataset_differences=cluster,
                standard_error=se, simultaneous_interval=[lo,hi], bootstrap_95_sensitivity=boot95,
                leave_one_dataset_out_range=[min(loo),max(loo)],
                reference_sensitivity=sensitivity, reference_adjusted_interval=[robust_lower,robust_upper],
                verdict="NONINFERIOR" if noninferior else ("MATERIAL_DETERIORATION" if harm else "INCONCLUSIVE"),
                supported_lower_error=robust_upper < 0 and boot95[1] < 0 and max(loo)<0)
        for comparator in ("bootstrap", "prior", "gaussian", "bootstrap_equal_runtime"):
            candidate, cheap = [], []
            for dataset,record in zip(datasets,records):
                ref = dataset["reference"]["rows"][point]["value"]
                for row in record["rows"]:
                    if comparator in row and row[comparator].get("matched_budget",True):
                        candidate.append(abs(row["marginal_mixture"]["value"]-ref))
                        cheap.append(abs(row[comparator]["value"]-ref))
            if cheap:
                result["heuristic"][comparator] = dict(candidate_mean_error=statistics.fmean(candidate),
                    comparator_mean_error=statistics.fmean(cheap),
                    veto=statistics.fmean(candidate)>statistics.fmean(cheap), statistically_ranked=False)
        required = {"prior", "bootstrap", "gaussian", "bootstrap_equal_runtime"}
        result["heuristic_dominance_verdict"] = ("PROMOTION_VETO" if any(
            r["veto"] for r in result["heuristic"].values()) else
            "INCOMPLETE" if set(result["heuristic"]) != required else "NO_OBSERVED_LOSS")
        result["verdict"] = ("NONINFERIOR" if all(r["verdict"]=="NONINFERIOR" for r in result["metrics"].values())
                             else "NOT_ESTABLISHED")
        output.append(result)
    return output

