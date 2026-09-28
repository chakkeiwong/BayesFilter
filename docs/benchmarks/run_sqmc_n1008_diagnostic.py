"""Fixed-data N=1008 accuracy diagnostic; frozen controls are UNTUNED.

FP64 GPU/XLA reference comparison, with graph parity. This reporting harness
does not issue tuning artifacts, change defaults, or confer admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/sqmc-n1008-diagnostic-20260926.md"
RESULT_NOTE = "docs/benchmarks/sqmc-n1008-results-20260926.md"
SAVED = ROOT / "docs/plans/artifacts/sqmc-repair-20260925/06-tuning-workflow"
N = 1008
DATA_SEEDS = (93001, 93002)
SCRAMBLE_SEEDS = tuple(range(94001, 94009))
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = list((ROOT / "bayesfilter/highdim").glob("*.py"))
    paths += list((ROOT / "bayesfilter/runtime").glob("*.py"))
    paths += [Path(__file__), ROOT / PLAN]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def close_vectors(actual, expected, tolerance):
    return len(actual) == len(expected) and all(
        abs(a - b) <= tolerance * (1.0 + abs(b))
        for a, b in zip(actual, expected)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    start = time.perf_counter()
    report = {
        "program": "P44 D3 T2 analytical recursive SQMC accuracy diagnostic",
        "particle_count": N,
        "dtype": "float64",
        "jit_compile": True,
        "tuning_status": "UNTUNED at N=1008; frozen corrected N=12 controls",
        "evidence_role": "diagnostic_only",
        "claim_eligible": False,
        "scientific_admission": False,
        "data_seeds": list(DATA_SEEDS),
        "additional_filter_seeds": list(SCRAMBLE_SEEDS),
        "parity": [],
        "cases": [],
    }
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "command": [sys.executable, *sys.argv],
        "environment": sys.executable,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cpu_gpu_status": "CPU-frozen data and inputs; GPU FP64 XLA diagnostic",
        "precision_exception": "FP64 reference comparison; not FP32 TF32 default evidence",
        "plan": PLAN,
        "result_note": RESULT_NOTE,
        "result": str(out / "results.json"),
        "data_version": "saved corrected P44 datasets 93001/93002, exact oracle replay required",
        "data_seeds": list(DATA_SEEDS),
        "filter_seeds": [*DATA_SEEDS, *SCRAMBLE_SEEDS],
        "source_sha256": source_hashes(),
        "status": "running",
        "input_sha256": {},
    }
    (out / "working-tree-status.txt").write_text(subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ))
    dump(out / "manifest.json", manifest)
    dump(out / "results.json", report)
    tf = None
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        manifest["gpu_memory_policy"] = configure_tensorflow_gpu_memory_growth(tf)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest["tensorflow"] = tf.__version__
        manifest["tf32"] = bool(tf.config.experimental.tensor_float_32_execution_enabled())
        manifest["physical_gpu_inventory"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"],
            text=True,
        ).strip()
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.ones([2, 2], tf.float64))
        manifest["framework_gpu_probe"] = {"device": probe.device, "value": float(probe.numpy())}
        if "GPU:0" not in probe.device or float(probe.numpy()) != 4.0:
            raise RuntimeError("Trusted TensorFlow GPU probe failed")

        from bayesfilter.highdim import sqmc_campaign_tf as campaign
        from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size

        spec = LGSSMSpec("p44", 3)
        routes = tuple(campaign.ROUTES)
        manifest["transport_chunk_size"] = select_transport_chunk_size(N)
        saved = {r: json.loads((SAVED / r / "result.json").read_text()) for r in routes}
        controls = {r: s["selected_controls"] for r, s in saved.items()}
        report["controls_by_route"] = controls
        report["route_settings"] = {r: campaign.route_settings(r) for r in routes}
        manifest["baseline_sha256"] = {r: digest(SAVED / r / "result.json") for r in routes}
        observations = {}
        oracles = {}
        with tf.device("/CPU:0"):
            theta_cpu = spec.default_theta(tf.float64)
            report["theta"] = theta_cpu.numpy().tolist()
            report["data"] = {}
            for seed in DATA_SEEDS:
                obs = spec.simulate(theta_cpu, 2, seed, jit_compile=False)
                value, score = spec.reference_value_and_score(theta_cpu, obs)
                oracle = {"value": float(value.numpy()), "score": score.numpy().tolist()}
                baseline = next(x for x in saved[routes[0]]["untouched_results"] if x["seed"] == seed)
                if not close_vectors([oracle["value"], *oracle["score"]],
                                     [baseline["oracle_value"], *baseline["oracle_score"]], 1e-10):
                    raise RuntimeError(f"Saved exact oracle changed for data seed {seed}")
                observations[seed] = obs
                oracles[seed] = oracle
                report["data"][str(seed)] = {
                    "observations": obs.numpy().tolist(), "oracle": oracle,
                    "n12_baseline": {
                        r: {k: next(x for x in saved[r]["untouched_results"] if x["seed"] == seed)[k]
                            for k in ("value", "score", "value_error", "score_l2_error")}
                        for r in routes
                    },
                }
        with tf.device("/GPU:0"):
            theta = tf.identity(theta_cpu)
            observations = {s: tf.identity(o) for s, o in observations.items()}
        input_cache = {}

        def inputs_for(route, seed):
            family = "iid_dual_cap" if route == "iid_dual_cap" else "repaired_permutation"
            key = (family, seed)
            if key not in input_cache:
                with tf.device("/CPU:0"):
                    tensors = campaign.random_inputs(family, seed, N, 3, 2, tf.float64)
                path = out / "inputs" / f"{family}-{seed}.json"
                dump(path, {k: v.numpy().tolist() for k, v in zip(
                    ("initial_normals", "process_normals", "ancestor_uniforms"), tensors)})
                manifest["input_sha256"][str(path.relative_to(out))] = digest(path)
                with tf.device("/GPU:0"):
                    input_cache[key] = tuple(tf.identity(x) for x in tensors)
            return input_cache[key]

        def call(route, data_seed, input_seed, jit):
            if time.perf_counter() - start > 1350:
                raise TimeoutError("Internal numerical budget exhausted; no new cell started")
            inputs = inputs_for(route, input_seed)
            begun = time.perf_counter()
            with tf.device("/GPU:0"):
                value, score, valid = campaign.value_and_score(
                    spec, route, controls[route], theta, observations[data_seed],
                    input_seed, N, jit_compile=jit, inputs=inputs,
                )
            actual_value, actual_score = float(value.numpy()), score.numpy().tolist()
            if not bool(valid.numpy()) or not all(math.isfinite(v) for v in [actual_value, *actual_score]):
                raise RuntimeError(f"Invalid value/score: {route}, data={data_seed}, input={input_seed}")
            if "GPU:0" not in value.device:
                raise RuntimeError(f"Expected GPU result, got {value.device}")
            errors = [a - b for a, b in zip(actual_score, oracles[data_seed]["score"])]
            row = {
                "route": route, "data_seed": data_seed, "input_seed": input_seed,
                "particle_count": N, "jit_compile": jit, "valid": True,
                "compute_device": value.device, "value": actual_value, "score": actual_score,
                "score_error": errors, "component_absolute_errors": [abs(e) for e in errors],
                "score_l2_error": math.sqrt(sum(e * e for e in errors)),
                "value_error": actual_value - oracles[data_seed]["value"],
                "wall_seconds": time.perf_counter() - begun,
            }
            print(json.dumps({k: row[k] for k in (
                "route", "data_seed", "input_seed", "jit_compile", "score", "score_l2_error", "wall_seconds"
            )}), flush=True)
            return row

        dump(out / "results.json", report)
        dump(out / "manifest.json", manifest)
        print(json.dumps({"stage": "oracle_and_gpu_probe_passed", "particle_count": N}), flush=True)
        # Gate the backend change before interpreting any accuracy results.
        for route in routes:
            graph = call(route, 93001, 93001, False)
            compiled = call(route, 93001, 93001, True)
            reference_vector = [graph["value"], *graph["score"]]
            compiled_vector = [compiled["value"], *compiled["score"]]
            parity = {
                "route": route, "graph_reference": graph,
                "max_absolute_difference": max(abs(a-b) for a,b in zip(compiled_vector, reference_vector)),
                "tolerance": "1e-7*(1+abs(graph_reference)) per value/score component",
                "passed": close_vectors(compiled_vector, reference_vector, 1e-7),
            }
            report["parity"].append(parity)
            if not parity["passed"]:
                dump(out / "results.json", report)
                raise RuntimeError(f"GPU XLA/graph parity failed for {route}")
            report["cases"].append(dict(compiled, role="original_filter_seed"))
            dump(out / "results.json", report)
        for route in routes:
            report["cases"].append(dict(call(route, 93002, 93002, True), role="original_filter_seed"))
            dump(out / "results.json", report)
        for input_seed in SCRAMBLE_SEEDS:
            for data_seed in DATA_SEEDS:
                for route in routes:
                    report["cases"].append(dict(call(route, data_seed, input_seed, True), role="independent_scramble"))
                    dump(out / "results.json", report)
            dump(out / "manifest.json", manifest)

        report["independent_scramble_summary"] = []
        for seed in DATA_SEEDS:
            for route in routes:
                rows = [r for r in report["cases"] if r["role"] == "independent_scramble"
                        and r["data_seed"] == seed and r["route"] == route]
                columns = list(zip(*(r["score"] for r in rows)))
                means = [statistics.mean(x) for x in columns]
                sds = [statistics.stdev(x) for x in columns]
                report["independent_scramble_summary"].append({
                    "data_seed": seed, "route": route, "replications": len(rows),
                    "score_mean": means, "score_sample_sd": sds,
                    "score_mean_mcse": [s / math.sqrt(len(rows)) for s in sds],
                    "score_observed_min": [min(x) for x in columns],
                    "score_observed_max": [max(x) for x in columns],
                    "mean_score_error": [a-b for a,b in zip(means, oracles[seed]["score"])],
                    "mean_score_l2_error": statistics.mean(r["score_l2_error"] for r in rows),
                    "root_mean_squared_vector_error": math.sqrt(statistics.mean(r["score_l2_error"]**2 for r in rows)),
                    "descriptive_only": True,
                })
        report["heuristic_diagnostics"] = {
            str(seed): {
                "exact_kalman_absolute_vector_error": 0.0,
                "zero_score_absolute_vector_error": math.sqrt(sum(s*s for s in oracles[seed]["score"])),
                "iid_baseline": next(r for r in report["independent_scramble_summary"]
                                     if r["data_seed"] == seed and r["route"] == "iid_dual_cap"),
                "heuristic_dominance_verdict": "no_promotion_attempted; exact oracle comparison remains primary",
            } for seed in DATA_SEEDS
        }
        if len(report["cases"]) != 72:
            raise RuntimeError("Incomplete requested comparison")
        if source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("Source changed during execution")
        report["decision"] = {
            "primary_criterion": "complete_72_finite_cells",
            "validity_vetoes": "all_clear_in_checked_cells",
            "statistically_supported_ranking": False,
            "default_ready": False,
            "uncertainty": "eight filter scrambles, two fixed short datasets, untuned controls",
            "next_action": "interpret absolute accuracy; investigate persistent bias before any promotion",
        }
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        (out / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if tf is not None:
            try:
                manifest["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
            except (ValueError, RuntimeError) as exc:
                manifest["gpu_allocator_query_error"] = str(exc)
        manifest["source_unchanged"] = source_hashes() == manifest["source_sha256"]
        report["wall_seconds"] = manifest["wall_seconds"]
        dump(out / "manifest.json", manifest)
        dump(out / "results.json", report)
        print(json.dumps({"status": manifest["status"], "wall_seconds": manifest["wall_seconds"],
                          "completed_xla_cells": len(report["cases"])}), flush=True)


if __name__ == "__main__":
    main()
