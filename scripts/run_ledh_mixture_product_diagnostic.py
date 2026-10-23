"""Bounded fresh-seed diagnostic of optional product-moment global proposals."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_ledh_ksc_fitted_guides import (
    check_evaluation, digest, expired, materialize, source_hashes, tensor_fit, write_json)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ledh-mixture-product-moment-repair-20261018.md"
PRIOR = ROOT / "docs/plans/artifacts/ledh-ksc-fitted-guides-20261017"
STARTED = time.monotonic()
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def integer_leaves(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in integer_leaves(child)]
    if isinstance(value, list):
        return [item for child in value for item in integer_leaves(child)]
    return [value] if isinstance(value, int) else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=600.)
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 900:
        parser.error("budget must be in (0,900]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    frozen_path = PRIOR / "development-01/frozen-selection.json"
    config = json.loads(frozen_path.read_text())["selection"]["kernel"]
    expected = {"id": "kernel_n256_b0.5", "family": "kernel", "count": 256,
                "transition_replicates": 16, "bandwidth_factor": .5}
    if config != expected:
        raise ValueError("frozen development choice differs from the plan")
    seeds = {"fits": [[821800 + 100 * condition + cluster for cluster in range(8)]
                       for condition in range(3)],
             "clouds": [[[831800 + 1000 * condition + 100 * cluster + replicate
                           for replicate in range(8)] for cluster in range(8)] for condition in range(3)]}
    leaves = integer_leaves(seeds)
    old_seeds = set(integer_leaves(json.loads((PRIOR / "seed-design.json").read_text())))
    if len(set(leaves)) != len(leaves) or set(leaves) & old_seeds:
        raise ValueError("new seeds collide with each other or previous campaign")
    manifest = {"schema": "mixture_product_diagnostic_v1", "plan": PLAN,
                "result": str(output / "result.json"), "command": [sys.executable, *sys.argv],
                "cwd": str(ROOT), "python": sys.executable, "environment": "tftwogpu",
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "status": "started", "budget_seconds": args.budget_seconds,
                "seed_design": seeds, "seeds_disjoint_from_prior": True,
                "frozen_selection": config, "frozen_selection_path": str(frozen_path),
                "frozen_selection_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                "plan_sha256": hashlib.sha256((ROOT / PLAN).read_bytes()).hexdigest(),
                "nonclaims": ["optional consumer, not canonical LEDH", "no default or HMC promotion",
                              "fixed datasets, not population ranking", "not original KDM training"]}
    result = {"rows_completed": 0, "fits_completed": 0, "references": {}, "finite_differences": []}
    write_json(output / "manifest.json", manifest)
    failure = None
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.highdim import covariance_proposal_tf as core
        from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec, checked_reference, enumeration_reference
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        from bayesfilter.score_study.ksc_guidance_diagnostic_tf import ARMS, CONDITIONS, future_guides, make_clouds
        from bayesfilter.score_study.ksc_fitted_guides_diagnostic_tf import make_ksc_kernel_fit
        controls = core.ProposalControls(reset_steps=5, correction_steps=2)
        repaired = replace(controls, global_guide_conditioning="mixture_product_moments")
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1., 2.], tf.float64))
        if "GPU" not in probe.device:
            raise RuntimeError("GPU device probe failed")
        sources = source_hashes()
        sources[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest.update({"tensorflow": tf.__version__, "gpu_memory_policy": memory_policy,
                         "gpu_probe_device": probe.device, "gpu_trust": "escalated_local_gpu_access",
                         "jit_compile": True, "dtype": "float64", "tf32_enabled": False,
                         "execution_role": "FP64 GPU/XLA diagnostic reference exception",
                         "cpu_reference_exception": "independent enumeration/grid checks explicitly on CPU",
                         "controls": vars(controls), "repaired_controls": vars(repaired),
                         "particle_count": 32, "chunk_extent": 32, "horizon": 2,
                         "data_version": "sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1",
                         "observations": CONDITIONS, "theta": [1.5, 0.], "source_hashes": sources,
                         "score_route": "analytical_recursive_frozen_guide_total_derivative",
                         "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                                                 "--format=csv,noheader"], text=True).strip()})
        write_json(output / "manifest.json", manifest)
        cpu_started = time.monotonic()
        with tf.device("/CPU:0"):
            theta = tf.constant([1.5, 0.], tf.float64)
            enumerator = enumeration_reference(2)
            for condition, observed in CONDITIONS.items():
                refined = checked_reference((1.5, 0.), tuple(row[0] for row in observed))
                value, score = enumerator(theta, tf.constant(observed, tf.float64))
                reference = {"value": float(value), "score": materialize(score), "grid": materialize(refined)}
                reference["value_difference"] = abs(reference["value"] - refined["value"])
                reference["score_difference"] = max(abs(left - right) for left, right in
                                                     zip(reference["score"], refined["score"]))
                if reference["value_difference"] > 1e-6 or reference["score_difference"] > 1e-5:
                    raise RuntimeError("independent reference mismatch")
                result["references"][condition] = reference
        result["cpu_reference_seconds"] = time.monotonic() - cpu_started
        if result["cpu_reference_seconds"] > 60:
            raise TimeoutError("CPU reference sub-budget exhausted")
        spec = KSCSpec()
        programs = {
            "quadratic": make_diagnostic_programs(32, 2, controls, spec=spec)[0],
            "mixture": make_diagnostic_programs(32, 2, controls, spec=spec,
                                                guide_family="mixture", mixture_components=7)[0],
            "kernel": make_diagnostic_programs(32, 2, controls, spec=spec,
                                               guide_family="mixture", mixture_components=256)[0],
            "product": make_diagnostic_programs(32, 2, repaired, spec=spec,
                                                guide_family="mixture", mixture_components=256)[0]}
        builder = make_ksc_kernel_fit(theta, 256, 16, .5)
        arms = list(ARMS) + [(f"{prefix}_{name}", family, beta)
                            for prefix, family in (("fitted_kernel", "kernel"), ("product_kernel", "product"))
                            for name, beta in (("global", (.1, .9, 0.)), ("equal", (1. / 3.,) * 3))]
        manifest["arms"] = arms
        write_json(output / "manifest.json", manifest)
        with tf.device("/GPU:0"), (output / "rows.jsonl").open("x") as stream:
            for condition_index, (condition, observed) in enumerate(CONDITIONS.items()):
                observations = tf.constant(observed, tf.float64)
                exact_guides = future_guides(theta, observations)
                for cluster in range(8):
                    fit_seed = seeds["fits"][condition_index][cluster]
                    fitted = tensor_fit(tf, builder, config, observations[:, 0], fit_seed)
                    write_json(output / f"fit-{condition}-{cluster}.json", {"seed": fit_seed, "fit": fitted})
                    result["fits_completed"] += 1
                    if not bool(fitted["guide_eligible"]):
                        raise RuntimeError("fresh fitted guide is ineligible; no substitution")
                    fitted_parameters = tuple(fitted[key] for key in ("logits", "centers", "covariances", "active"))
                    guide_hash = digest(fitted_parameters)
                    for replicate, seed in enumerate(seeds["clouds"][condition_index][cluster]):
                        clouds = make_clouds(32, seed)
                        cloud_hash = digest(clouds)
                        for arm, family, beta in arms:
                            row_started = time.monotonic()
                            auxiliary = fitted_parameters if family in ("kernel", "product") else exact_guides[family]
                            program = programs[family if family in ("kernel", "product", "mixture") else "quadratic"]
                            arguments = (observations, *clouds, tf.constant(beta, tf.float64), *auxiliary)
                            evaluated = program(theta, *arguments)
                            check_evaluation(tf, evaluated)
                            row = {"condition": condition, "cluster": cluster, "replicate": replicate,
                                   "seed": seed, "cloud_sha256": cloud_hash, "arm": arm, "beta": beta,
                                   "status": "valid", "fit_seed": fit_seed,
                                   "guide_sha256": guide_hash if family in ("kernel", "product") else digest(auxiliary),
                                   "global_guide_conditioning": "mixture_product_moments" if family == "product" else "positive_information",
                                   "value_device": evaluated["value"].device, **materialize(evaluated)}
                            if family == "product" and cluster == replicate == 0:
                                for coordinate in range(2):
                                    for step in (1e-4, 5e-5):
                                        displacement = tf.one_hot(coordinate, 2, dtype=tf.float64) * step
                                        plus = program(theta + displacement, *arguments)
                                        minus = program(theta - displacement, *arguments)
                                        check_evaluation(tf, plus)
                                        check_evaluation(tf, minus)
                                        difference = float((plus["value"] - minus["value"]) / (2 * step))
                                        error = abs(difference - float(evaluated["score"][coordinate]))
                                        tolerance = 2e-7 + 2e-5 * abs(difference)
                                        check = {"condition": condition, "arm": arm, "coordinate": coordinate,
                                                 "step": step, "difference": difference, "error": error,
                                                 "tolerance": tolerance, "passed": error <= tolerance}
                                        result["finite_differences"].append(check)
                                        if not check["passed"]:
                                            raise RuntimeError("finite-program derivative mismatch")
                            row["wall_seconds"] = time.monotonic() - row_started
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            result["rows_completed"] += 1
                    print(f"{condition}, cluster {cluster}: {result['rows_completed']} rows", flush=True)
        if result["rows_completed"] != 2304 or result["fits_completed"] != 24:
            raise RuntimeError("incomplete predeclared grid")
        if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected_hash
               for path, expected_hash in sources.items()):
            raise RuntimeError("source changed during comparison")
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["status"] = "complete"
    except Exception as error:
        failure = error
        manifest["status"] = "budget_incomplete" if isinstance(error, TimeoutError) else "invalid"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["status"] = manifest["status"]
        result["wall_seconds"] = time.monotonic() - STARTED
        manifest["wall_seconds"] = result["wall_seconds"]
        write_json(output / "result.json", result)
        write_json(output / "manifest.json", manifest)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
