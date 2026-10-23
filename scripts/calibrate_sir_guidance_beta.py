"""Diagnostic independent-pilot beta fitting; no reference-score access."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/benchmarks"))
from run_ledh_covariance_proposal import clean, design, prepare, spec_for


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.guides_from / "manifest.json").read_text())
    if source["status"] != "complete" or source["completed_rows"] != 96:
        raise ValueError("First-stage design is not complete")
    if source["particles"] != 256 or source["horizon"] != 50 or source["reset_epsilon"] != 1. or source["reset_steps"] != 5:
        raise ValueError("Unexpected first-stage scope")
    for path, checksum in source["source_sha256"].items():
        if not path.startswith("scripts/") and hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Numerical source changed: {path}")
    manifest = dict(status="running", command=sys.argv, environment="tftwogpu",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    particles=256, horizon=50, reset_epsilon=1., reset_steps=5,
                    dtype="float64", tf32=False, jit_compile=True, cpu_only=False,
                    beta_floor=.1, gap_relative_tolerance=1e-7, solver_max_steps=20000,
                    solver='cyclic_pairwise_simplex_log_domain_v1',
                    flow_steps=16, correction_steps=0,
                    budget_seconds=args.seconds, data=source["data"], fits=source["fits"],
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), xla_flags=os.environ.get("XLA_FLAGS"),
                    plan="docs/plans/ledh-sir-beta-repair-20261019.md",
                    reference_score_used=False, scope="conditional_diagnostic_no_canonical_admission",
                    result=str(args.output / "calibration.json"))
    paths = [Path(__file__), ROOT / "bayesfilter/score_study/sir_beta_calibration_diagnostic_tf.py",
             ROOT / "bayesfilter/highdim/covariance_proposal_beta_tf.py",
             ROOT / "bayesfilter/highdim/covariance_proposal_tf.py",
             ROOT / "bayesfilter/highdim/ledh_backward_guide_tf.py",
             ROOT / "docs/benchmarks/run_ledh_covariance_proposal.py"]
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest["source_sha256"].update({path: checksum for path, checksum in source["source_sha256"].items() if not path.startswith("scripts/")})
    write(args.output / "manifest.json", manifest)
    def timeout(_signum, _frame):
        raise TimeoutError("Beta calibration allocation exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        args.device, args.dtype, args.no_tf32 = "gpu", "float64", True
        tensorflow, memory = prepare(args)
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
        from bayesfilter.highdim.covariance_proposal_beta_tf import make_pairwise_beta_solver, pilot_log_terms, pilot_log_objective
        from bayesfilter.highdim.ledh_backward_guide_tf import FrozenMixtureBackwardGuide, FrozenQuadraticBackwardGuide
        from bayesfilter.score_study.sir_beta_calibration_diagnostic_tf import make_context_pilot
        dtype = tensorflow.float64
        spec = spec_for("sir_d18")
        theta = spec.default_theta(dtype)
        controls = ProposalControls(reset_epsilon=1., reset_steps=5, flow_steps=16, correction_steps=0)
        program = make_filter(spec, 256, 50, controls, dtype, jit_compile=True)
        solver = make_pairwise_beta_solver(50, 256, dtype, jit_compile=True, max_steps=20000)
        equal = tensorflow.constant([1 / 3] * 3, dtype)
        manifest.update(memory_policy=memory, tensorflow=tensorflow.__version__)
        records = []
        for data_seed in (26100831, 26100832):
            data_path = ROOT / source["data"][str(data_seed)]["path"]
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != source["data"][str(data_seed)]["sha256"]:
                raise ValueError("Calibration data checksum changed")
            observed = tensorflow.constant(json.loads(data_path.read_text())["observations"], dtype)
            fit_path = args.guides_from / f"guides-{data_seed}.json"
            if hashlib.sha256(fit_path.read_bytes()).hexdigest() != source["fits"][str(data_seed)]["sha256"]:
                raise ValueError("Calibration guide checksum changed")
            fit = json.loads(fit_path.read_text())
            guides = dict(plain=None,
                csmc_quadratic=FrozenQuadraticBackwardGuide(tensorflow.constant(fit["precision"], dtype), tensorflow.constant(fit["linear"], dtype)),
                kernel=FrozenMixtureBackwardGuide(*(tensorflow.constant(fit[key], dtype) for key in ("logits", "centers", "covariances")),
                                                  tensorflow.constant(fit["active"], tensorflow.bool)))
            first_seed = 261019501 if data_seed == 26100831 else 261019601
            histories = []
            seeds = []
            for phase in range(2):
                seed = first_seed + phase
                fixed = design(tensorflow, 256, 50, 18, seed, dtype)
                history = program(theta, tensorflow.zeros_like(theta), observed, *fixed, tensorflow.constant([1., 0., 0.], dtype))
                if not bool(history["valid"]) or int(history["steps_completed"]) != 50:
                    write(args.output / f"failed-history-{data_seed}-{seed}.json", history)
                    raise RuntimeError("Invalid prior calibration context")
                noise_seed = seed + 10000
                pilot = design(tensorflow, 256, 50, 18, noise_seed, dtype)
                histories.append((history["ancestor_history"], pilot[1], pilot[2]))
                seeds.append(dict(history=seed, mixture=noise_seed))
                manifest["actual_device"] = history["value"].device
            for arm, guide in guides.items():
                print(json.dumps(dict(data_seed=data_seed, arm=arm, phase="pilot")), flush=True)
                pilot = make_context_pilot(spec, 256, 50, controls, guide, dtype, jit_compile=True)
                contexts = [pilot(theta, observed, *history) for history in histories]
                if not all(bool(tensorflow.reduce_all(context["valid"])) for context in contexts):
                    raise RuntimeError(f"Invalid proposal pilot: {data_seed}/{arm}")
                write(args.output / f"pilot-{data_seed}-{arm}.json", contexts)
                print(json.dumps(dict(data_seed=data_seed, arm=arm, phase="optimization")), flush=True)
                fitted = solver(contexts[0]["log_components"], contexts[0]["log_target"],
                                tensorflow.constant(.1, dtype), tensorflow.constant(1e-7, dtype))
                training_terms = pilot_log_terms(contexts[0]["log_components"], contexts[0]["log_target"])
                validation_terms = pilot_log_terms(contexts[1]["log_components"], contexts[1]["log_target"])
                objective = pilot_log_objective(fitted["beta"], *validation_terms)[0]
                equal_objective = pilot_log_objective(equal, *validation_terms)[0]
                def context_objectives(terms, probabilities):
                    log_denominator = tensorflow.reduce_logsumexp(terms[0] + tensorflow.math.log(probabilities), axis=-1)
                    return tensorflow.exp(tensorflow.reduce_logsumexp(terms[1] - log_denominator, axis=1))
                record = dict(data_seed=data_seed, arm=arm, seeds=seeds, beta=fitted["beta"],
                              objective=fitted["objective"], gap=fitted["gap"], iterations=fitted["iterations"],
                              converged=bool(fitted["converged"]), validation_objective=objective,
                              validation_equal_objective=equal_objective,
                              training_context_objectives=context_objectives(training_terms, fitted["beta"]),
                              validation_context_objectives=context_objectives(validation_terms, fitted["beta"]),
                              validation_equal_context_objectives=context_objectives(validation_terms, equal),
                              context_diagnostics=[context["diagnostics"] for context in contexts],
                              valid=bool(fitted["converged"]) and math.isfinite(float(objective))
                              and math.isfinite(float(equal_objective))
                              and bool(tensorflow.reduce_all(tensorflow.math.is_finite(fitted["beta"]))))
                records.append(record)
                write(args.output / "calibration.json", records)
                print(json.dumps(clean({key:record[key] for key in ("data_seed", "arm", "beta", "gap", "iterations", "valid", "validation_objective", "validation_equal_objective")})), flush=True)
                if not record["valid"]:
                    raise RuntimeError("Invalid or unconverged beta fit")
        manifest["status"] = "complete"
        manifest["calibration_sha256"] = hashlib.sha256((args.output / "calibration.json").read_bytes()).hexdigest()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        manifest["wall_seconds"] = time.monotonic() - started
        write(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--guides-from", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=1800)
    run(parser.parse_args())
