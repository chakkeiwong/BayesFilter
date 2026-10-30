"""Conditional diagnostic: saved future messages, paired fresh particle clouds."""

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
    check_evaluation, digest, expired, materialize, source_hashes, write_json)

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ledh-message-consumer-continuation-20261018.md"
SAVED = ROOT / "docs/plans/artifacts/ledh-mixture-product-20261018/attempt-01"
DEVELOPMENT = ROOT / "docs/plans/artifacts/ledh-ksc-fitted-guides-20261017"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def integer_leaves(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in integer_leaves(child)]
    if isinstance(value, list):
        return [item for child in value for item in integer_leaves(child)]
    return [value] if isinstance(value, int) and not isinstance(value, bool) else []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=700.)
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 700:
        parser.error("attempt budget must be in (0,700]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((SAVED / "manifest.json").read_text())
    previous_result = json.loads((SAVED / "result.json").read_text())
    if previous["status"] != "complete" or previous_result["status"] != "complete":
        raise ValueError("saved fit campaign is incomplete")
    frozen_path = DEVELOPMENT / "development-01/frozen-selection.json"
    config = json.loads(frozen_path.read_text())["selection"]["kernel"]
    expected = {"id": "kernel_n256_b0.5", "family": "kernel", "count": 256,
                "transition_replicates": 16, "bandwidth_factor": .5}
    if config != expected:
        raise ValueError("saved fit choice differs from the plan")
    seeds = {"fits": [[821800 + 100 * condition + cluster for cluster in range(8)]
                       for condition in range(3)],
             "clouds": [[[841800 + 1000 * condition + 100 * cluster + replicate
                           for replicate in range(8)] for cluster in range(8)]
                        for condition in range(3)]}
    cloud_seeds = integer_leaves(seeds["clouds"])
    old_seeds = set(integer_leaves(previous["seed_design"]))
    old_seeds.update(integer_leaves(json.loads((DEVELOPMENT / "seed-design.json").read_text())))
    if len(set(cloud_seeds)) != 192 or set(cloud_seeds) & old_seeds:
        raise ValueError("fresh cloud seeds collide")
    inputs = {str(path): sha256(path) for path in
              (SAVED / "manifest.json", SAVED / "result.json", frozen_path,
               DEVELOPMENT / "seed-design.json")}
    manifest = {"schema": "message_consumer_diagnostic_v1", "plan": PLAN,
                "plan_sha256": sha256(ROOT / PLAN), "result": str(output / "result.json"),
                "command": [sys.executable, *sys.argv], "cwd": str(ROOT),
                "python": sys.executable, "environment": "tftwogpu",
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "status": "started", "budget_seconds": args.budget_seconds,
                "seed_design": seeds, "cloud_seeds_disjoint_from_prior": True,
                "fit_reuse": "conditional inference on 24 saved fits; no fresh-fit generalization",
                "frozen_selection": config, "input_sha256": inputs, "fit_inputs": {},
                "nonclaims": ["optional consumer, not canonical LEDH", "no default or HMC promotion",
                              "fixed datasets and fits, not population ranking",
                              "exact message does not make the particle filter exact",
                              "frozen guide is not differentiated through fitting"]}
    result = {"rows_completed": 0, "fits_loaded": 0, "references": {}, "finite_differences": []}
    write_json(output / "manifest.json", manifest)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.budget_seconds - (time.monotonic() - started)))
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
        controls = core.ProposalControls(reset_steps=5, correction_steps=2)
        product_controls = replace(controls, global_guide_conditioning="mixture_product_moments")
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1., 2.], tf.float64))
        if "GPU" not in probe.device:
            raise RuntimeError("GPU device probe failed")
        sources = source_hashes()
        for name in (Path(__file__).name, "analyze_ledh_message_consumer_diagnostic.py"):
            path = ROOT / "scripts" / name
            sources[str(path.relative_to(ROOT))] = sha256(path)
        manifest.update({"tensorflow": tf.__version__, "gpu_memory_policy": memory_policy,
                         "gpu_probe_device": probe.device, "gpu_trust": "escalated_local_gpu_access",
                         "jit_compile": True, "dtype": "float64", "tf32_enabled": False,
                         "execution_role": "FP64 GPU/XLA diagnostic reference exception",
                         "cpu_reference_exception": "independent enumeration/grid checks explicitly on CPU",
                         "controls": vars(controls), "product_controls": vars(product_controls),
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
        programs = {"quadratic": make_diagnostic_programs(32, 2, controls, spec=spec)[0]}
        for family, count, selected in (("mixture", 7, controls), ("kernel", 256, controls),
                                        ("product", 256, product_controls),
                                        ("exact_product", 7, product_controls)):
            programs[family] = make_diagnostic_programs(
                32, 2, selected, spec=spec, guide_family="mixture", mixture_components=count)[0]
        arms = list(ARMS) + [(f"{prefix}_{name}", family, beta)
                            for prefix, family in (("fitted_kernel", "kernel"),
                                                   ("product_kernel", "product"),
                                                   ("exact_product", "exact_product"))
                            for name, beta in (("global", (.1, .9, 0.)), ("equal", (1. / 3.,) * 3))]
        manifest["arms"] = arms
        write_json(output / "manifest.json", manifest)
        with tf.device("/GPU:0"), (output / "rows.jsonl").open("x") as stream:
            for condition_index, (condition, observed) in enumerate(CONDITIONS.items()):
                observations = tf.constant(observed, tf.float64)
                exact_guides = future_guides(theta, observations)
                for cluster in range(8):
                    fit_seed = seeds["fits"][condition_index][cluster]
                    original = SAVED / f"fit-{condition}-{cluster}.json"
                    copied = output / original.name
                    fit_bytes = original.read_bytes()
                    copied.write_bytes(fit_bytes)
                    inputs[str(original)] = sha256(original)
                    manifest["fit_inputs"][copied.name] = {
                        "source": str(original), "sha256": inputs[str(original)], "seed": fit_seed}
                    fit_data = json.loads(fit_bytes)
                    fitted = fit_data["fit"]
                    if fit_data["seed"] != fit_seed or not fitted["guide_eligible"]:
                        raise RuntimeError("saved fit identity or eligibility mismatch")
                    fitted_parameters = tuple(tf.constant(fitted[key], tf.bool if key == "active" else tf.float64)
                                              for key in ("logits", "centers", "covariances", "active"))
                    result["fits_loaded"] += 1
                    for replicate, seed in enumerate(seeds["clouds"][condition_index][cluster]):
                        clouds = make_clouds(32, seed)
                        cloud_hash = digest(clouds)
                        for arm, family, beta in arms:
                            row_started = time.monotonic()
                            if family in ("kernel", "product"):
                                auxiliary = fitted_parameters
                            elif family == "exact_product":
                                auxiliary = exact_guides["mixture"]
                            else:
                                auxiliary = exact_guides[family]
                            program = programs.get(family, programs["quadratic"])
                            arguments = (observations, *clouds, tf.constant(beta, tf.float64), *auxiliary)
                            evaluated = program(theta, *arguments)
                            check_evaluation(tf, evaluated)
                            row = {"condition": condition, "cluster": cluster, "replicate": replicate,
                                   "seed": seed, "cloud_sha256": cloud_hash, "arm": arm, "beta": beta,
                                   "status": "valid", "fit_seed": fit_seed, "guide_sha256": digest(auxiliary),
                                   "global_guide_conditioning": "mixture_product_moments"
                                   if family in ("product", "exact_product") else "positive_information",
                                   "value_device": evaluated["value"].device, **materialize(evaluated)}
                            if family in ("product", "exact_product") and cluster == replicate == 0:
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
                    write_json(output / "manifest.json", manifest)
                    print(f"{condition}, saved-fit cluster {cluster}: {result['rows_completed']} rows", flush=True)
        if result["rows_completed"] != 2688 or result["fits_loaded"] != 24:
            raise RuntimeError("incomplete predeclared grid")
        if any(sha256(ROOT / path) != expected_hash for path, expected_hash in sources.items()):
            raise RuntimeError("source changed during comparison")
        if any(sha256(Path(path)) != expected_hash for path, expected_hash in inputs.items()):
            raise RuntimeError("input changed during comparison")
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
        result["wall_seconds"] = time.monotonic() - started
        manifest["wall_seconds"] = result["wall_seconds"]
        write_json(output / "result.json", result)
        write_json(output / "manifest.json", manifest)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
