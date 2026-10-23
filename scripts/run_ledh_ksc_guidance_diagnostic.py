"""Bounded GPU/XLA KSC guidance diagnostic; never canonical or fitted KDM evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback


STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
PLAN = "docs/plans/ledh-ksc-guidance-audit-20261017.md"
CPU_SECONDS_AVAILABLE = 110.76
NONCLAIMS = (
    "No fitted controlled-SMC or KDM result; no canonical LEDH admission, "
    "default readiness, HMC readiness, or statistically supported ranking. "
    "Scores differentiate the same finite program with guide coefficients frozen."
)


def materialize(value):
    if hasattr(value, "numpy"):
        value = value.numpy().tolist()
    if isinstance(value, dict):
        return {str(key): materialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [materialize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(materialize(value), indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(materialize(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = {Path(__file__).resolve()}
    paths.update(Path(module.__file__).resolve()
                 for name, module in list(sys.modules.items())
                 if name.startswith("bayesfilter") and getattr(module, "__file__", None))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_relative_to(ROOT)}


def expired(signum, frame):
    raise TimeoutError("KSC diagnostic wall-time allowance exhausted")


def render_note(result):
    decision = result["decision"]
    lines = ["# KSC frozen future-guidance diagnostic", "", NONCLAIMS, "",
             "| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action |",
             "| --- | --- | --- | --- | --- |",
             f"| {decision['status']} | {decision['primary_criterion']} | "
             f"{decision['veto_status']} | Eight clouds per constructed situation | "
             f"{decision['next_action']} |", "",
             "| Inference status | Finding |", "| --- | --- |",
             f"| Hard veto screen | {decision['veto_status']} |",
             "| Statistically supported ranking | None; no confirmatory uncertainty analysis |",
             "| Descriptive-only differences | Conditional coordinate MSE and likelihood error |",
             "| Default readiness | Not evaluated; untuned diagnostic controls |",
             "| Next evidence needed | Target-specific fitting and untouched downstream score validation |", ""]
    if result.get("summary"):
        losses = result["summary"]["heuristic_dominance"]["loss_count"]
        lines += [f"Guided arms lose {losses} of 120 comparisons against constructed heuristics.",
                  "Every loss vetoes promotion of that arm in that situation and coordinate.", "",
                  "| Situation | Arm | Score MSE: persistence chart | Score MSE: log beta |",
                  "| --- | --- | ---: | ---: |"]
        for row in result["summary"]["conditional_metrics"]:
            lines.append(f"| {row['condition']} | {row['arm']} | "
                         f"{row['score_mse'][0]:.8g} | {row['score_mse'][1]:.8g} |")
    lines += ["", "The Gaussian observation approximation is a cheap heuristic with a different",
              "likelihood. The enumerated seven-mixture likelihood remains the reference.", "",
              "The claimed derivative is the analytical total derivative of the frozen-guide",
              "finite estimator. Agreement with finite differences does not establish small",
              "error against the exact likelihood score.", "",
              "Strongest alternative explanation: proposal geometry, reset approximation, or",
              "finite-cloud score error can dominate the quality of the future message.",
              "A fitted guide passing untouched conditional downstream comparisons would",
              "overturn candidate rejection. Eight clouds are the weakest statistical evidence.", "",
              f"Measured wall time: {result['wall_seconds']:.3f} seconds; CPU references: "
              f"{result.get('cpu_reference_seconds', 0.0):.3f} seconds.",
              "Full traces, reference checks, finite differences and source provenance are in",
              "`result.json`, `evaluations.jsonl`, and `manifest.json` beside this note.", ""]
    return "\n".join(lines)


def summarize(rows, references, gaussian_scores, arms, conditions):
    metrics = []
    indexed = {}
    for condition in conditions:
        target = references[condition]["score"]
        for arm, family, beta in arms:
            selected = [row for row in rows
                        if row["condition"] == condition and row["arm"] == arm]
            mse = [statistics.fmean((row["score"][coordinate] - target[coordinate]) ** 2
                                    for row in selected) for coordinate in range(2)]
            metric = {"condition": condition, "arm": arm, "guide_family": family,
                      "beta": beta, "clouds": len(selected), "score_mse": mse,
                      "mean_log_likelihood_error": statistics.fmean(
                          row["value"] - references[condition]["value"] for row in selected)}
            metrics.append(metric)
            indexed[condition, arm] = mse
        approximate_mse = [(gaussian_scores[condition][coordinate] - target[coordinate]) ** 2
                           for coordinate in range(2)]
        metrics.append({"condition": condition, "arm": "gaussian_observation_approximation",
                        "score_mse": approximate_mse, "clouds": 0})
        indexed[condition, "gaussian_observation_approximation"] = approximate_mse
    heuristics = [arm for arm, family, beta in arms if family == "none"]
    heuristics.append("gaussian_observation_approximation")
    comparisons = [{"condition": condition, "arm": arm, "coordinate": coordinate,
                    "heuristic": heuristic,
                    "promotion_veto": indexed[condition, arm][coordinate] >
                    indexed[condition, heuristic][coordinate]}
                   for condition in conditions for arm, family, beta in arms if family != "none"
                   for coordinate in range(2) for heuristic in heuristics]
    return {"conditional_metrics": metrics, "heuristic_dominance": {
        "comparisons": comparisons,
        "loss_count": sum(row["promotion_veto"] for row in comparisons),
        "comparison_count": len(comparisons),
        "verdict": "promotion_veto" if any(row["promotion_veto"] for row in comparisons)
        else "descriptive_screen_pass_only"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=450.0)
    args = parser.parse_args()
    if not 0.0 < args.budget_seconds <= 450.0:
        parser.error("budget must be positive and no greater than 450 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, args.budget_seconds - (time.monotonic() - STARTED)))
    manifest = {"schema": "ksc_frozen_guidance_diagnostic_v1", "plan": PLAN,
                "result": str(output / "result.json"), "command": [sys.executable, *sys.argv],
                "cwd": str(ROOT), "python": sys.executable, "budget_seconds": args.budget_seconds,
                "cpu_reference_budget_seconds": CPU_SECONDS_AVAILABLE,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                                  text=True).strip(),
                "nonclaims": NONCLAIMS, "status": "started"}
    result = {"nonclaims": NONCLAIMS, "references": {}, "finite_differences": [],
              "rows_completed": 0, "cpu_reference_seconds": 0.0}
    write_json(output / "manifest.json", manifest)
    failure = None
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.highdim import covariance_proposal_tf as core
        from bayesfilter.highdim.sqmc_ksc_tf import (
            KSCSpec, checked_reference, enumeration_reference, gaussian_approximation_reference)
        from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
        from bayesfilter.score_study.ksc_guidance_diagnostic_tf import (
            ARMS, CONDITIONS, cloud_seed, future_guides, make_clouds)

        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1.0, 2.0], tf.float64))
        if "GPU" not in probe.device:
            raise RuntimeError("trusted GPU probe did not execute on GPU")
        controls = core.ProposalControls(flow_steps=16, reset_steps=5, reset_epsilon=1.0,
                                        correction_steps=2, correction_basis=4,
                                        correction_rate=0.1, correction_radius=0.25)
        before_hashes = source_hashes()
        manifest.update({"tensorflow": tf.__version__, "gpu_memory_policy": memory_policy,
                         "gpu_probe_device": probe.device, "gpu_probe_value": float(probe),
                         "gpu_trust": "escalated_local_gpu_access", "jit_compile": True,
                         "dtype": "float64", "tf32_enabled": False,
                         "execution_class": "noncanonical_reference_diagnostic_exception",
                         "target_id": KSCSpec().target_id, "particles": 32, "horizon": 2,
                         "transport_chunk": 32, "controls": vars(controls), "arms": ARMS,
                         "conditions": CONDITIONS, "data_sha256": digest(CONDITIONS),
                         "seeds": {condition: [cloud_seed(index, replicate) for replicate in range(8)]
                                   for index, condition in enumerate(CONDITIONS)},
                         "score_method": "analytical_recursive_frozen_guide_total_derivative",
                         "reference_method": "CPU_exact_Gaussian_branch_enumeration_and_refined_grid",
                         "source_hashes_before": before_hashes, "trace_fields": core.TRACE_FIELDS})
        write_json(output / "manifest.json", manifest)
        references = {}
        gaussian_scores = {}
        cpu_started = time.monotonic()
        try:
            with tf.device("/CPU:0"):
                theta_cpu = tf.constant([1.5, 0.0], tf.float64)
                for condition, observations in CONDITIONS.items():
                    refined = checked_reference((1.5, 0.0), tuple(row[0] for row in observations))
                    exact_value, exact_score = enumeration_reference(2)(
                        theta_cpu, tf.constant(observations, tf.float64))
                    reference = {"value": float(exact_value), "score": materialize(exact_score),
                                 "refined_grid": materialize(refined)}
                    reference["value_difference"] = abs(reference["value"] - refined["value"])
                    reference["score_difference"] = max(abs(left - right) for left, right in
                                                       zip(reference["score"], refined["score"]))
                    references[condition] = reference
                    result["references"] = references
                    if reference["value_difference"] > 1e-6 or reference["score_difference"] > 1e-5:
                        raise RuntimeError(f"reference mismatch for {condition}")
                    gaussian_value, gaussian_score = gaussian_approximation_reference(
                        theta_cpu, tf.constant(observations, tf.float64))
                    gaussian_scores[condition] = materialize(gaussian_score)
                    reference["gaussian_approximation"] = {
                        "value": float(gaussian_value), "score": gaussian_scores[condition]}
                single_grid = checked_reference((1.5, 0.0), (0.2,))
                single_value, single_score = enumeration_reference(1)(
                    theta_cpu, tf.constant([[0.2]], tf.float64))
                single_check = {"value_difference": abs(float(single_value) - single_grid["value"]),
                                "score_difference": max(abs(left - right) for left, right in
                                                        zip(materialize(single_score), single_grid["score"]))}
                result["single_observation_reference_check"] = single_check
                if single_check["value_difference"] > 1e-6 or single_check["score_difference"] > 1e-5:
                    raise RuntimeError("single-observation reference mismatch")
        finally:
            result["cpu_reference_seconds"] = time.monotonic() - cpu_started
        if result["cpu_reference_seconds"] > CPU_SECONDS_AVAILABLE:
            raise TimeoutError("remaining CPU reference allowance exhausted")
        write_json(output / "reference-checks.json", result)
        print(f"References valid in {result['cpu_reference_seconds']:.3f} CPU wall seconds", flush=True)
        programs = {"quadratic": make_diagnostic_programs(32, 2, controls, spec=KSCSpec(),
                                                        guide_family="quadratic", jit_compile=True)[0],
                    "mixture": make_diagnostic_programs(32, 2, controls, spec=KSCSpec(),
                                                      guide_family="mixture", mixture_components=7,
                                                      jit_compile=True)[0]}
        rows = []
        with (output / "evaluations.jsonl").open("w") as stream, tf.device("/GPU:0"):
            theta = tf.constant([1.5, 0.0], tf.float64)
            for condition_index, (condition, observed) in enumerate(CONDITIONS.items()):
                observations = tf.constant(observed, tf.float64)
                guides = future_guides(theta, observations)
                manifest.setdefault("frozen_guides", {})[condition] = materialize(guides)
                write_json(output / "manifest.json", manifest)
                for replicate in range(8):
                    seed = cloud_seed(condition_index, replicate)
                    clouds = make_clouds(32, seed)
                    cloud_hash = digest(clouds)
                    for arm, family, beta_values in ARMS:
                        started = time.monotonic()
                        program = programs["mixture" if family == "mixture" else "quadratic"]
                        beta = tf.constant(beta_values, tf.float64)
                        arguments = (observations, *clouds, beta, *guides[family])
                        evaluated = program(theta, *arguments)
                        row = {"condition": condition, "replicate": replicate, "seed": seed,
                               "cloud_sha256": cloud_hash, "arm": arm, "guide_family": family,
                               "beta": beta_values, **materialize(evaluated),
                               "value_device": evaluated["value"].device,
                               "wall_seconds": time.monotonic() - started}
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        result["rows_completed"] = len(rows)
                        finite = all(bool(tf.reduce_all(tf.math.is_finite(evaluated[key])))
                                     for key in ("value", "score", "trace", "direction_trace"))
                        if not finite or not row["valid"] or not all(row["direction_valid"]):
                            raise RuntimeError(f"invalid numerical output: {condition}/{replicate}/{arm}")
                        if "GPU" not in row["value_device"]:
                            raise RuntimeError("claim-bearing diagnostic kernel left GPU")
                        if replicate == 0 and arm in ("gaussian_future_global", "mixture_future_global"):
                            for step in (1e-4, 5e-5):
                                for coordinate in range(2):
                                    displacement = tf.one_hot(coordinate, 2, dtype=tf.float64) * step
                                    plus = program(theta + displacement, *arguments)
                                    minus = program(theta - displacement, *arguments)
                                    finite_difference = float((plus["value"] - minus["value"]) / (2.0 * step))
                                    difference = abs(row["score"][coordinate] - finite_difference)
                                    tolerance = 2e-7 + 2e-5 * abs(finite_difference)
                                    valid = bool(plus["valid"]) and bool(minus["valid"])
                                    check = {"condition": condition, "arm": arm, "coordinate": coordinate,
                                             "step": step, "analytical": row["score"][coordinate],
                                             "finite_difference": finite_difference, "difference": difference,
                                             "tolerance": tolerance, "pass": valid and math.isfinite(difference)
                                             and difference <= tolerance}
                                    result["finite_differences"].append(check)
                                    write_json(output / "finite-differences.json", result["finite_differences"])
                                    if not check["pass"]:
                                        raise RuntimeError(f"analytical derivative mismatch: {check}")
                    print(f"{condition}: cloud {replicate + 1}/8 valid; {len(rows)}/192 base calls", flush=True)
        result["summary"] = summarize(rows, references, gaussian_scores, ARMS, CONDITIONS)
        result["decision"] = {"status": "engineering_screen_passed",
                              "primary_criterion": "references, likelihood wiring, validity and two-step derivatives pass",
                              "veto_status": "no continuation veto; per-cell heuristic promotion vetoes recorded",
                              "next_action": "specify target-specific fitting protocol before training"}
        result["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        result["source_hashes_after"] = source_hashes()
        if any(result["source_hashes_after"].get(path) != checksum
               for path, checksum in before_hashes.items()):
            raise RuntimeError("imported source changed during execution")
    except BaseException as error:
        failure = error
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        result["decision"] = {"status": "incomplete_or_failed_screen",
                              "primary_criterion": "not established",
                              "veto_status": "unresolved failure; classify before retry",
                              "next_action": "inspect failure and repair within remaining campaign allowance"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        result["wall_seconds"] = time.monotonic() - STARTED
        manifest.update({"status": result["decision"]["status"],
                         "wall_seconds": result["wall_seconds"],
                         "cpu_reference_seconds": result["cpu_reference_seconds"]})
        write_json(output / "manifest.json", manifest)
        write_json(output / "result.json", result)
        (output / "result.md").write_text(render_note(result))
    print(json.dumps({"status": result["decision"]["status"], "wall_seconds": result["wall_seconds"],
                      "rows_completed": result["rows_completed"], "output": str(output)}), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
