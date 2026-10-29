"""Conditional FP64 diagnostic: paired particle counts on the KSC positive tail."""

import argparse
from dataclasses import replace
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
from run_ledh_message_consumer_diagnostic import integer_leaves, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ledh-particle-count-stress-repair-20261018.md"
PARENT = ROOT / "docs/plans/artifacts/ledh-message-consumer-20261018/attempt-01"
SAVED = ROOT / "docs/plans/artifacts/ledh-mixture-product-20261018/attempt-01"
DEVELOPMENT = ROOT / "docs/plans/artifacts/ledh-ksc-fitted-guides-20261017"
PREVIOUS = ROOT / "docs/plans/artifacts/ledh-particle-count-stress-20261018/attempt-01"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=600.)
    parser.add_argument("--followup", action="store_true")
    args = parser.parse_args()
    counts = (64, 128) if args.followup else (32, 64)
    replicates = 12 if args.followup else 6
    seed_base = 862800 if args.followup else 861800
    plan = "docs/plans/ledh-particle-count-followup-20261018.md" if args.followup else PLAN
    budget_cap = 500 if args.followup else 600
    design_id = "positive_tail_n64_n128_r12_v1" if args.followup else "positive_tail_n32_n64_r6_v1"
    if not 0 < args.budget_seconds <= budget_cap:
        parser.error(f"attempt budget must be in (0,{budget_cap}]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent_manifest = json.loads((PARENT / "manifest.json").read_text())
    parent_result = json.loads((PARENT / "result.json").read_text())
    parent_analysis = json.loads((PARENT / "analysis.json").read_text())
    if (parent_manifest["status"] != "complete" or parent_result["status"] != "complete"
            or parent_analysis["terminal_audit"] != "PASS"):
        raise ValueError("parent comparison is not complete and audited")
    frozen_path = DEVELOPMENT / "development-01/frozen-selection.json"
    config = json.loads(frozen_path.read_text())["selection"]["kernel"]
    expected = {"id": "kernel_n256_b0.5", "family": "kernel", "count": 256,
                "transition_replicates": 16, "bandwidth_factor": .5}
    if config != expected:
        raise ValueError("saved fit choice differs from the plan")
    seeds = {"fits": [822000 + cluster for cluster in range(8)],
             "clouds": [[seed_base + 100 * cluster + replicate for replicate in range(replicates)]
                        for cluster in range(8)]}
    cloud_seeds = integer_leaves(seeds["clouds"])
    old_seeds = set(integer_leaves(parent_manifest["seed_design"]))
    old_seeds.update(integer_leaves(json.loads((SAVED / "manifest.json").read_text())["seed_design"]))
    old_seeds.update(integer_leaves(json.loads((DEVELOPMENT / "seed-design.json").read_text())))
    if args.followup:
        previous_manifest = json.loads((PREVIOUS / "manifest.json").read_text())
        previous_result = json.loads((PREVIOUS / "result.json").read_text())
        previous_analysis = json.loads((PREVIOUS / "analysis.json").read_text())
        if (previous_manifest["status"] != previous_result["status"]
                or previous_result["status"] != "complete"
                or previous_analysis["terminal_audit"] != "PASS"):
            raise ValueError("previous particle-count diagnostic is not complete and audited")
        old_seeds.update(integer_leaves(previous_manifest["seed_design"]))
    if len(set(cloud_seeds)) != 8 * replicates or set(cloud_seeds) & old_seeds:
        raise ValueError("fresh cloud seeds collide")
    input_paths = [PARENT / name for name in ("manifest.json", "result.json", "analysis.json")]
    input_paths.extend((SAVED / "manifest.json", SAVED / "result.json", frozen_path,
                        DEVELOPMENT / "seed-design.json"))
    if args.followup:
        input_paths.extend(PREVIOUS / name for name in ("manifest.json", "result.json", "analysis.json"))
    inputs = {str(path): sha256(path) for path in input_paths}
    manifest = {"schema": "particle_count_stress_diagnostic_v1", "plan": plan,
                "plan_sha256": sha256(ROOT / plan), "result": str(output / "result.json"),
                "design_id": design_id, "cloud_replicates": replicates, "cloud_seed_base": seed_base,
                "command": [sys.executable, *sys.argv], "cwd": str(ROOT),
                "python": sys.executable, "environment": "tftwogpu",
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "status": "running", "budget_seconds": args.budget_seconds,
                "seed_design": seeds, "input_sha256": inputs, "fit_inputs": {},
                "scope": "conditional fixed-fit positive-tail particle-count diagnostic",
                "canonical_ledh": False, "default_ready": False}
    result = {"rows_completed": 0, "fits_loaded": 0, "finite_differences": [], "references": {}}
    write_json(output / "manifest.json", manifest)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.budget_seconds - (time.monotonic() - started)))
    failure = None
    try:
        sys.path.insert(0, str(ROOT))
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.highdim import covariance_proposal_tf as core
        from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec, checked_reference, enumeration_reference
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        from bayesfilter.score_study.ksc_guidance_diagnostic_tf import ARMS, future_guides, make_clouds
        controls = core.ProposalControls(reset_steps=5, correction_steps=2)
        product_controls = replace(controls, global_guide_conditioning="mixture_product_moments")
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1., 2.], tf.float64))
        if "GPU" not in probe.device:
            raise RuntimeError("GPU device probe failed")
        sources = source_hashes()
        for name in (Path(__file__).name, "analyze_ledh_particle_count_stress_diagnostic.py",
                     "run_ledh_message_consumer_diagnostic.py"):
            path = ROOT / "scripts" / name
            sources[str(path.relative_to(ROOT))] = sha256(path)
        observed = [[.2], [5.]]
        arms = list(ARMS[:6]) + [("product_kernel_global", "product", (.1, .9, 0.)),
                                ("exact_product_global", "exact_product", (.1, .9, 0.))]
        manifest.update({"tensorflow": tf.__version__, "gpu_memory_policy": memory_policy,
                         "gpu_probe_device": probe.device, "gpu_trust": "escalated_local_gpu_access",
                         "jit_compile": True, "dtype": "float64", "tf32_enabled": False,
                         "execution_role": "FP64 GPU/XLA diagnostic reference exception",
                         "cpu_reference_exception": "independent enumeration/grid checks explicitly on CPU",
                         "controls": vars(controls), "product_controls": vars(product_controls),
                         "particle_counts": list(counts), "chunk_extents": list(counts), "horizon": 2,
                         "data_version": "sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1",
                         "observations": observed, "theta": [1.5, 0.], "source_hashes": sources,
                         "arms": arms, "score_route": "analytical_recursive_frozen_guide_total_derivative",
                         "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                                                 "--format=csv,noheader"], text=True).strip()})
        write_json(output / "manifest.json", manifest)
        cpu_started = time.monotonic()
        with tf.device("/CPU:0"):
            theta = tf.constant([1.5, 0.], tf.float64)
            enumerator = enumeration_reference(2)
            refined = checked_reference((1.5, 0.), (.2, 5.))
            value, score = enumerator(theta, tf.constant(observed, tf.float64))
            reference = {"value": float(value), "score": materialize(score), "grid": materialize(refined)}
            reference["value_difference"] = abs(reference["value"] - refined["value"])
            reference["score_difference"] = max(abs(left - right) for left, right in
                                                 zip(reference["score"], refined["score"]))
            if reference["value_difference"] > 1e-6 or reference["score_difference"] > 1e-5:
                raise RuntimeError("independent reference mismatch")
            result["references"]["positive_future_tail"] = reference
        result["cpu_reference_seconds"] = time.monotonic() - cpu_started
        if result["cpu_reference_seconds"] > 60:
            raise TimeoutError("CPU reference sub-budget exhausted")
        programs = {}
        for count in counts:
            programs[count] = {"quadratic": make_diagnostic_programs(count, 2, controls, spec=KSCSpec())[0]}
            for family, components in (("product", 256), ("exact_product", 7)):
                programs[count][family] = make_diagnostic_programs(
                    count, 2, product_controls, spec=KSCSpec(), guide_family="mixture",
                    mixture_components=components)[0]
        with tf.device("/GPU:0"), (output / "rows.jsonl").open("x") as stream:
            observations = tf.constant(observed, tf.float64)
            exact_guides = future_guides(theta, observations)
            for cluster in range(8):
                fit_seed = seeds["fits"][cluster]
                original = SAVED / f"fit-positive_future_tail-{cluster}.json"
                fit_bytes = original.read_bytes()
                (output / original.name).write_bytes(fit_bytes)
                inputs[str(original)] = sha256(original)
                manifest["fit_inputs"][original.name] = {
                    "source": str(original), "sha256": inputs[str(original)], "seed": fit_seed}
                fit_data = json.loads(fit_bytes)
                fitted = fit_data["fit"]
                if fit_data["seed"] != fit_seed or not fitted["guide_eligible"]:
                    raise RuntimeError("saved fit identity or eligibility mismatch")
                fitted_parameters = tuple(tf.constant(fitted[key], tf.bool if key == "active" else tf.float64)
                                          for key in ("logits", "centers", "covariances", "active"))
                result["fits_loaded"] += 1
                for replicate, seed in enumerate(seeds["clouds"][cluster]):
                    parent_cloud = make_clouds(counts[-1], seed)
                    parent_hash = digest(parent_cloud)
                    for count in counts:
                        clouds = (parent_cloud[0][:count], parent_cloud[1][:, :count],
                                  parent_cloud[2][:, :count])
                        cloud_hash = digest(clouds)
                        for arm, family, beta in arms:
                            row_started = time.monotonic()
                            auxiliary = (fitted_parameters if family == "product" else
                                         exact_guides["mixture"] if family == "exact_product" else
                                         exact_guides[family])
                            program = programs[count].get(family, programs[count]["quadratic"])
                            arguments = (observations, *clouds, tf.constant(beta, tf.float64), *auxiliary)
                            evaluated = program(theta, *arguments)
                            check_evaluation(tf, evaluated)
                            row = {"condition": "positive_future_tail", "cluster": cluster,
                                   "replicate": replicate, "seed": seed, "cloud_sha256": cloud_hash,
                                   "parent_cloud_sha256": parent_hash, "particle_count": count,
                                   "chunk_extent": count, "arm": arm, "beta": beta, "status": "valid",
                                   "fit_seed": fit_seed, "guide_sha256": digest(auxiliary),
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
                                        check = {"particle_count": count, "arm": arm, "coordinate": coordinate,
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
                print(f"saved-fit cluster {cluster}: {result['rows_completed']} rows", flush=True)
        if (result["rows_completed"] != 8 * replicates * len(counts) * len(arms) or result["fits_loaded"] != 8
                or len(result["finite_differences"]) != 16):
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
