"""Bounded diagnostic KSC fits and frozen analytical-score comparisons."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
PLAN = "docs/plans/ledh-ksc-fitting-consumer-protocol-20261017.md"
CONDITIONS = ("ordinary", "negative_future_tail", "positive_future_tail")
TAIL_STATE_CUTOFF = 2.0 * math.sqrt(1.0 + (0.5 * (1.0 + math.erf(1.5 / math.sqrt(2.0)))) ** 2)
NONCLAIMS = ["diagnostic consumer, not canonical LEDH admission",
             "not original KDM or full twisted controlled-SMC recursion",
             "not HMC, default-readiness, native SV, or cross-model evidence"]
CONFIGURATIONS = ([{"id": f"quadratic_n{count}_p{passes}", "family": "quadratic",
                    "count": count, "passes": passes}
                   for count in (64, 256) for passes in (1, 2)]
                  + [{"id": f"kernel_n{count}_b{factor:g}", "family": "kernel",
                      "count": count, "transition_replicates": 16,
                      "bandwidth_factor": factor}
                     for count in (64, 256) for factor in (0.5, 1.0, 2.0)])


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
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("bayesfilter") and path and Path(path).suffix == ".py":
            paths.add(Path(path).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_relative_to(ROOT)}


def seed_design():
    return {
        "calibration": {condition: {config["id"]: [
            651000 + condition_index * 1000 + config_index * 10 + replicate
            for replicate in range(4)]
            for config_index, config in enumerate(CONFIGURATIONS)}
            for condition_index, condition in enumerate(CONDITIONS)},
        "validation": {condition: [661000 + condition_index * 100 + replicate
                                   for replicate in range(4)]
                       for condition_index, condition in enumerate(CONDITIONS)},
        "final_fits": {condition: {family: [
            671000 + condition_index * 1000 + family_index * 100 + replicate
            for replicate in range(8)] for family_index, family in enumerate(
                ("quadratic", "kernel"))}
                       for condition_index, condition in enumerate(CONDITIONS)},
        "final_clouds": {condition: [[
            681000 + condition_index * 1000 + cluster * 10 + replicate
            for replicate in range(8)] for cluster in range(8)]
                         for condition_index, condition in enumerate(CONDITIONS)},
        "stateless_streams": {"quadratic": {"normal": 1, "uniform": 2},
                              "kernel": {"component": 1, "design": 2, "transition": 3},
                              "cloud": {"initial": 1, "innovations": 2, "uniform": 3}}}


def expired(*unused):
    raise TimeoutError("predeclared launch wall-time budget exhausted")


def tensor_fit(tf, builder, config, observations, seed):
    count = config["count"]
    if config["family"] == "quadratic":
        passes = config["passes"]
        return builder(observations,
                       tf.random.stateless_normal([passes, 2, count], [seed, 1], dtype=tf.float64),
                       tf.random.stateless_uniform([passes, 2], [seed, 2], dtype=tf.float64))
    return builder(observations,
                   tf.random.stateless_uniform([count], [seed, 1], dtype=tf.float64),
                   tf.random.stateless_normal([count], [seed, 2], dtype=tf.float64),
                   tf.random.stateless_normal([count, config["transition_replicates"]],
                                              [seed, 3], dtype=tf.float64))


def make_loss_program(tf, family, count):
    signature = ([tf.TensorSpec([256], tf.float64)] * 4
                 + ([tf.TensorSpec([], tf.float64)] * 2 if family == "quadratic" else
                    [tf.TensorSpec([count], tf.float64), tf.TensorSpec([count], tf.float64),
                     tf.TensorSpec([], tf.float64)]))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate(nodes, weights, exact_log_density, exact_gradient, *parameters):
        if family == "quadratic":
            precision, linear = parameters
            mean = linear / precision
            log_density = (0.5 * tf.math.log(precision / tf.constant(2.0 * math.pi, tf.float64))
                           - 0.5 * precision * tf.square(nodes - mean))
            gradient = linear - precision * nodes
        else:
            logits, centers, variance = parameters
            terms = (tf.nn.log_softmax(logits)[None, :]
                     - 0.5 * tf.square(nodes[:, None] - centers[None, :]) / variance
                     - 0.5 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64) * variance))
            log_density = tf.reduce_logsumexp(terms, axis=1)
            gradient = tf.reduce_sum(tf.nn.softmax(terms, axis=1)
                                     * (centers[None, :] - nodes[:, None]) / variance, axis=1)
        tail = tf.cast(tf.abs(nodes) > tf.constant(TAIL_STATE_CUTOFF, tf.float64), tf.float64)
        tail_mass = tf.reduce_sum(weights * tail)
        return {"cross_entropy": -tf.reduce_sum(weights * log_density),
                "excess_cross_entropy": tf.reduce_sum(weights * (exact_log_density - log_density)),
                "gradient_mse": tf.reduce_sum(weights * tf.square(gradient - exact_gradient)),
                "tail_weighted_log_error": tf.reduce_sum(weights * tail
                                                         * tf.abs(log_density - exact_log_density)),
                "tail_mass": tail_mass}

    return evaluate


def exact_density(tf, nodes, auxiliary):
    logits, centers, covariances, _ = auxiliary
    means = centers[0, :, 0]
    variances = covariances[0, :, 0, 0]
    terms = (tf.nn.log_softmax(logits[0])[None, :]
             - 0.5 * tf.square(nodes[:, None] - means) / variances
             - 0.5 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64) * variances))
    return (tf.reduce_logsumexp(terms, axis=1),
            tf.reduce_sum(tf.nn.softmax(terms, axis=1) * (means - nodes[:, None]) / variances, axis=1))


def fit_parameters(family, fitted):
    if family == "quadratic":
        return fitted["precision"][0, 0, 0], fitted["linear"][0, 0]
    return fitted["logits"][0], fitted["nodes"], fitted["covariances"][0, 0, 0, 0]


def run_development(tf, output, result, seeds, theta, conditions, future_guides,
                    quadratic_builder, kernel_builder):
    builders = {}
    losses = {}
    for config in CONFIGURATIONS:
        builders[config["id"]] = (quadratic_builder(theta, config["count"], config["passes"])
                                  if config["family"] == "quadratic" else kernel_builder(
                                      theta, config["count"], 16, config["bandwidth_factor"]))
        losses[config["id"]] = make_loss_program(tf, config["family"], config["count"])
    validation_config = {"family": "kernel", "count": 256, "transition_replicates": 64}
    validation_builder = kernel_builder(theta, 256, 64, 1.0)
    summaries = {config["id"]: {"eligible": True, "losses": []} for config in CONFIGURATIONS}
    with (output / "fits.jsonl").open("x") as stream:
        for condition, observed in conditions.items():
            observations = tf.constant([row[0] for row in observed], tf.float64)
            exact = future_guides(theta, tf.constant(observed, tf.float64))["mixture"]
            validations = []
            exact_validations = []
            for seed in seeds["validation"][condition]:
                fitted = tensor_fit(tf, validation_builder, validation_config, observations, seed)
                if not bool(fitted["density_eligible"]):
                    raise RuntimeError("invalid independent validation design")
                exact_log_density, exact_gradient = exact_density(tf, fitted["nodes"], exact)
                validations.append((fitted["nodes"], tf.exp(fitted["log_weights"]),
                                    exact_log_density, exact_gradient))
                exact_weights = tf.nn.softmax(exact_log_density - fitted["log_design_density"])
                exact_validations.append((fitted["nodes"], exact_weights, exact_log_density, exact_gradient))
                write_json(output / f"validation-{condition}-{seed}.json",
                           {"seed": seed, "fit": fitted, "exact_log_density": exact_log_density})
            for config in CONFIGURATIONS:
                for replicate, seed in enumerate(seeds["calibration"][condition][config["id"]]):
                    fitted = tensor_fit(tf, builders[config["id"]], config, observations, seed)
                    eligible = bool(fitted["density_eligible"])
                    row = {"condition": condition, "config": config, "replicate": replicate,
                           "seed": seed, "fit": fitted, "eligible": eligible}
                    if eligible:
                        validation = validations[replicate]
                        metrics = losses[config["id"]](*validation, *fit_parameters(config["family"], fitted))
                        if not all(math.isfinite(float(value)) for value in metrics.values()):
                            raise RuntimeError("non-finite validation metric")
                        row["validation"] = metrics
                        row["exact_h_validation_reference"] = losses[config["id"]](
                            *exact_validations[replicate], *fit_parameters(config["family"], fitted))
                        summaries[config["id"]]["losses"].append(metrics["excess_cross_entropy"])
                        if config["family"] == "kernel":
                            exact_log, _ = exact_density(tf, fitted["nodes"], exact)
                            oracle_logits = tf.nn.log_softmax(exact_log - fitted["log_design_density"])
                            weights = tf.exp(oracle_logits)
                            mean = tf.reduce_sum(weights * fitted["nodes"])
                            variance = tf.reduce_sum(weights * tf.square(fitted["nodes"] - mean))
                            ess = 1.0 / tf.reduce_sum(tf.square(weights))
                            bandwidth = (tf.constant(config["bandwidth_factor"], tf.float64)
                                         * tf.sqrt(variance) * tf.pow(ess, tf.constant(-0.2, tf.float64)))
                            row["exact_h_representation_control"] = losses[config["id"]](
                                *validation, oracle_logits, fitted["nodes"], tf.square(bandwidth))
                    else:
                        summaries[config["id"]]["eligible"] = False
                    stream.write(json.dumps(materialize(row), allow_nan=False) + "\n")
                    stream.flush()
                    result["fits_completed"] += 1
            print(f"Development {condition}: {result['fits_completed']} fits preserved", flush=True)
    selected = {}
    for family in ("quadratic", "kernel"):
        eligible = [config for config in CONFIGURATIONS
                    if config["family"] == family and summaries[config["id"]]["eligible"]]
        if eligible:
            means = tf.stack([tf.reduce_mean(tf.stack(summaries[config["id"]]["losses"]))
                              for config in eligible])
            selected[family] = eligible[int(tf.argmin(means))]
        else:
            selected[family] = None
    result.update({"selection": selected, "selection_metrics": summaries})
    write_json(output / "frozen-selection.json", {"selection": selected, "seed_design": seeds,
               "selection_metric": "mean independent-validation excess message cross entropy",
               "source_hashes": source_hashes(), "references": result["references"],
               "fit_count": result["fits_completed"], "nonclaims": NONCLAIMS})


def check_evaluation(tf, evaluated):
    if not bool(evaluated["valid"]) or not bool(tf.reduce_all(evaluated["direction_valid"])):
        raise RuntimeError("consumer validity veto")
    if not all(bool(tf.reduce_all(tf.math.is_finite(evaluated[key])))
               for key in ("value", "score", "trace", "direction_trace")):
        raise RuntimeError("non-finite accepted consumer output")
    if "GPU" not in evaluated["value"].device:
        raise RuntimeError("consumer did not execute on GPU")


def run_final(tf, output, result, seeds, theta, conditions, future_guides, make_clouds,
              quadratic_builder, kernel_builder, make_programs, controls, spec, frozen):
    selection = frozen["selection"]
    programs = {"quadratic": make_programs(32, 2, controls, spec=spec, jit_compile=True)[0],
                "mixture": make_programs(32, 2, controls, spec=spec, guide_family="mixture",
                                          mixture_components=7, jit_compile=True)[0]}
    builders = {}
    for family, config in selection.items():
        if config is not None:
            builders[family] = (quadratic_builder(theta, config["count"], config["passes"])
                                if family == "quadratic" else kernel_builder(
                                    theta, config["count"], 16, config["bandwidth_factor"]))
            if family == "kernel":
                programs["kernel"] = make_programs(32, 2, controls, spec=spec,
                    guide_family="mixture", mixture_components=config["count"], jit_compile=True)[0]
    from bayesfilter.score_study.ksc_guidance_diagnostic_tf import ARMS

    arms = list(ARMS) + [(f"fitted_{family}_{beta_name}", family, beta)
                        for family in ("quadratic", "kernel")
                        for beta_name, beta in (("global", (0.1, 0.9, 0.0)),
                                                ("equal", (1.0 / 3.0,) * 3))]
    with (output / "rows.jsonl").open("x") as stream:
        for condition, observed in conditions.items():
            observations = tf.constant(observed, tf.float64)
            exact_guides = future_guides(theta, observations)
            for cluster in range(8):
                fitted_guides = {}
                for family, config in selection.items():
                    if config is None:
                        fitted_guides[family] = None
                        continue
                    seed = seeds["final_fits"][condition][family][cluster]
                    fitted = tensor_fit(tf, builders[family], config, observations[:, 0], seed)
                    write_json(output / f"fit-{condition}-{family}-{cluster}.json",
                               {"seed": seed, "config": config, "fit": fitted})
                    fitted_guides[family] = (fitted if bool(fitted["guide_eligible"]) else None)
                for replicate, seed in enumerate(seeds["final_clouds"][condition][cluster]):
                    clouds = make_clouds(32, seed)
                    cloud_hash = digest(clouds)
                    for arm, family, beta in arms:
                        row_started = time.monotonic()
                        row = {"condition": condition, "cluster": cluster, "replicate": replicate,
                               "seed": seed, "cloud_sha256": cloud_hash, "arm": arm,
                               "guide_family": family, "beta": beta, "status": "valid"}
                        if arm.startswith("fitted_"):
                            fitted = fitted_guides[family]
                            row["fit_seed"] = seeds["final_fits"][condition][family][cluster]
                            row["config"] = selection[family]
                            if fitted is None:
                                row["status"] = "candidate_ineligible"
                                auxiliary = None
                            elif family == "quadratic":
                                auxiliary = (fitted["precision"], fitted["linear"])
                            else:
                                auxiliary = tuple(fitted[key] for key in
                                                  ("logits", "centers", "covariances", "active"))
                            program = programs.get(family)
                        else:
                            auxiliary = exact_guides[family]
                            program = programs["mixture" if family == "mixture" else "quadratic"]
                        if auxiliary is not None:
                            arguments = (observations, *clouds, tf.constant(beta, tf.float64), *auxiliary)
                            evaluated = program(theta, *arguments)
                            check_evaluation(tf, evaluated)
                            row.update(materialize(evaluated))
                            row["value_device"] = evaluated["value"].device
                            if cluster == replicate == 0 and arm in (
                                    "fitted_quadratic_global", "fitted_kernel_global"):
                                for coordinate in range(2):
                                    for step in (1e-4, 5e-5):
                                        displacement = tf.one_hot(coordinate, 2, dtype=tf.float64) * step
                                        plus = program(theta + displacement, *arguments)
                                        minus = program(theta - displacement, *arguments)
                                        check_evaluation(tf, plus)
                                        check_evaluation(tf, minus)
                                        difference = float((plus["value"] - minus["value"]) / (2.0 * step))
                                        error = abs(difference - float(evaluated["score"][coordinate]))
                                        tolerance = 2e-7 + 2e-5 * abs(difference)
                                        check = {"condition": condition, "arm": arm,
                                                 "coordinate": coordinate, "step": step,
                                                 "difference": difference, "error": error,
                                                 "tolerance": tolerance, "passed": error <= tolerance}
                                        result["finite_differences"].append(check)
                                        if not check["passed"]:
                                            raise RuntimeError("frozen-coefficient derivative mismatch")
                        row["wall_seconds"] = time.monotonic() - row_started
                        stream.write(json.dumps(materialize(row), allow_nan=False) + "\n")
                        stream.flush()
                        result["rows_completed"] += 1
                print(f"Final {condition}, fit cluster {cluster}: {result['rows_completed']} rows", flush=True)
    if result["rows_completed"] != 2304:
        raise RuntimeError("incomplete predeclared arm/condition/seed grid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "final", "seeds"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=450.0)
    args = parser.parse_args()
    if not 0.0 < args.budget_seconds <= 450.0:
        parser.error("launch budget must be in (0, 450]")
    seeds = seed_design()
    if args.stage == "seeds":
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, seeds)
        return 0
    if args.stage == "final" and args.frozen is None:
        parser.error("final stage requires frozen development selection")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, args.budget_seconds - (time.monotonic() - STARTED)))
    manifest = {"schema": "ksc_fitted_guide_campaign_v1", "stage": args.stage, "plan": PLAN,
                "result": str(output / "result.json"), "command": [sys.executable, *sys.argv],
                "cwd": str(ROOT), "python": sys.executable, "budget_seconds": args.budget_seconds,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "seed_design": seeds, "seed_design_sha256": digest(seeds),
                "nonclaims": NONCLAIMS, "status": "started"}
    result = {"nonclaims": NONCLAIMS, "references": {}, "finite_differences": [],
              "rows_completed": 0, "fits_completed": 0, "cpu_reference_seconds": 0.0}
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
            CONDITIONS as OBSERVATIONS, future_guides, make_clouds)
        from bayesfilter.score_study.ksc_fitted_guides_diagnostic_tf import (
            make_ksc_quadratic_fit, make_ksc_kernel_fit)
        with tf.device("/GPU:0"):
            probe = tf.reduce_sum(tf.constant([1.0, 2.0], tf.float64))
        if "GPU" not in probe.device:
            raise RuntimeError("GPU probe did not execute on GPU")
        controls = core.ProposalControls(flow_steps=16, reset_steps=5, reset_epsilon=1.0,
                                        correction_steps=2, correction_basis=4,
                                        correction_rate=0.1, correction_radius=0.25)
        before_hashes = source_hashes()
        manifest.update({"tensorflow": tf.__version__, "gpu_memory_policy": memory_policy,
                         "gpu_probe_device": probe.device, "gpu_probe_value": float(probe),
                         "gpu_trust": "escalated_local_gpu_access", "jit_compile": True,
                         "dtype": "float64", "tf32_enabled": False,
                         "execution_role": "FP64 diagnostic/reference exception, not production promotion",
                         "particle_count": 32, "horizon": 2, "chunk_extent": 32,
                         "controls": vars(controls), "source_hashes": before_hashes,
                         "data_version": "sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1",
                         "observations": OBSERVATIONS, "score_route": "analytical_recursive_frozen_guide_total_derivative",
                         "tail_state_cutoff": TAIL_STATE_CUTOFF,
                         "eager_exceptions": ["host diagnostics, exact-density reference evaluation and selection reduction"],
                         "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                                                 "--format=csv,noheader"], text=True).strip()})
        write_json(output / "manifest.json", manifest)
        if args.stage == "development":
            cpu_started = time.monotonic()
            with tf.device("/CPU:0"):
                theta = tf.constant([1.5, 0.0], tf.float64)
                for condition, observed in OBSERVATIONS.items():
                    refined = checked_reference((1.5, 0.0), tuple(row[0] for row in observed))
                    exact_value, exact_score = enumeration_reference(2)(theta, tf.constant(observed, tf.float64))
                    reference = {"value": float(exact_value), "score": materialize(exact_score),
                                 "refined_grid": materialize(refined)}
                    reference["value_difference"] = abs(reference["value"] - refined["value"])
                    reference["score_difference"] = max(abs(left - right) for left, right in
                                                         zip(reference["score"], refined["score"]))
                    if reference["value_difference"] > 1e-6 or reference["score_difference"] > 1e-5:
                        raise RuntimeError("reference mismatch")
                    gaussian_value, gaussian_score = gaussian_approximation_reference(theta, tf.constant(observed, tf.float64))
                    reference["gaussian_approximation"] = {"value": float(gaussian_value),
                                                            "score": materialize(gaussian_score)}
                    result["references"][condition] = reference
            result["cpu_reference_seconds"] = time.monotonic() - cpu_started
            if result["cpu_reference_seconds"] > 60.0:
                raise TimeoutError("60-second CPU reference sub-budget exhausted")
            with tf.device("/GPU:0"):
                run_development(tf, output, result, seeds, tf.constant([1.5, 0.0], tf.float64),
                                OBSERVATIONS, future_guides, make_ksc_quadratic_fit, make_ksc_kernel_fit)
            if result["fits_completed"] != 120:
                raise RuntimeError("incomplete development grid")
        else:
            frozen = json.loads(args.frozen.read_text())
            if frozen["seed_design"] != seeds or frozen["fit_count"] != 120:
                raise RuntimeError("frozen design mismatch")
            if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != checksum
                   for path, checksum in frozen["source_hashes"].items()):
                raise RuntimeError("source changed after selection")
            manifest["frozen_selection_path"] = str(args.frozen.resolve())
            manifest["frozen_selection_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
            result["references"] = frozen["references"]
            result["selection"] = frozen["selection"]
            with tf.device("/GPU:0"):
                run_final(tf, output, result, seeds, tf.constant([1.5, 0.0], tf.float64),
                          OBSERVATIONS, future_guides, make_clouds, make_ksc_quadratic_fit,
                          make_ksc_kernel_fit, make_diagnostic_programs, controls, KSCSpec(), frozen)
        after_hashes = source_hashes()
        if any(after_hashes.get(path) != checksum for path, checksum in before_hashes.items()):
            raise RuntimeError("source mutated during launch")
        result["source_hashes_after"] = after_hashes
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["status"] = "complete"
    except Exception as error:
        failure = error
        manifest["status"] = "budget_incomplete" if isinstance(error, TimeoutError) else "invalid"
        result["error"] = {"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        result["status"] = manifest["status"]
        result["wall_seconds"] = time.monotonic() - STARTED
        manifest["wall_seconds"] = result["wall_seconds"]
        write_json(output / "result.json", result)
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"],
                      "rows": result["rows_completed"], "fits": result["fits_completed"],
                      "output": str(output), "error": str(failure) if failure else None}), flush=True)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
