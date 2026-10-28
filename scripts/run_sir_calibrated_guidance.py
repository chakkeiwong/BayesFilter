"""Diagnostic SIR score comparison with independently frozen proposal mixtures."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def calibrated_arms(records, data_seed):
    selected = {record["arm"]: record for record in records if record["data_seed"] == data_seed}
    if len(selected) != 3 or sum(record["data_seed"] == data_seed for record in records) != 3:
        raise ValueError("Missing or duplicate calibration records")
    for record in selected.values():
        beta = record["beta"]
        if not record["valid"] or not record["converged"] or len(beta) != 3:
            raise ValueError("Invalid calibration")
        if not all(math.isfinite(value) and value >= 0 for value in beta):
            raise ValueError("Invalid mixture probabilities")
        if beta[0] < .1 - 1e-12 or abs(sum(beta) - 1.) > 1e-12:
            raise ValueError("Mixture violates defensive simplex")
        if not all(math.isfinite(record[field]) for field in ("gap", "objective")):
            raise ValueError("Nonfinite mixture certificate")
        if record["gap"] > 1e-7 * max(1., record["objective"]):
            raise ValueError("Unconverged mixture objective")
    return dict(plain=selected["plain"]["beta"],
                csmc_quadratic=selected["csmc_quadratic"]["beta"],
                kernel=selected["kernel"]["beta"],
                plain_quadratic_beta=selected["csmc_quadratic"]["beta"],
                plain_kernel_beta=selected["kernel"]["beta"],
                prior=[1., 0., 0.], local=[.1, 0., .9])


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    calibrated = json.loads((args.calibration / "manifest.json").read_text())
    source = json.loads((args.guides_from / "manifest.json").read_text())
    if calibrated["status"] != "complete" or digest(args.calibration / "calibration.json") != calibrated["calibration_sha256"]:
        raise ValueError("Incomplete or changed calibration")
    if calibrated["data"] != source["data"] or calibrated["fits"] != source["fits"]:
        raise ValueError("Calibration scope differs from frozen guides/data")
    scope = dict(particles=256, horizon=50, reset_epsilon=1., reset_steps=5, dtype="float64", tf32=False, jit_compile=True, flow_steps=16, correction_steps=0, beta_floor=.1, gap_relative_tolerance=1e-7)
    if any(calibrated.get(key) != value for key, value in scope.items()) or calibrated.get("reference_score_used") is not False:
        raise ValueError("Calibration numerical scope or independence changed")
    for path, checksum in calibrated["source_sha256"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Calibration source changed: {path}")
    records = json.loads((args.calibration / "calibration.json").read_text())
    arms = {str(data_seed): calibrated_arms(records, data_seed) for data_seed in (26100831, 26100832)}
    manifest = dict(status="running", command=sys.argv, environment="tftwogpu",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        particles=256, horizon=50, reset_epsilon=1., reset_steps=5, flow_steps=16, correction_steps=0,
        jit_compile=True, dtype="float64", tf32=False, cpu_only=False,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), xla_flags=os.environ.get("XLA_FLAGS"),
        budget_seconds=args.seconds, completed_rows=0, planned_rows=112,
        data=source["data"], fits=source["fits"], reference=source["reference"],
        arms=list(arms["26100831"]), arms_by_dataset=arms, design="independent_beta_matched_guidance_v1",
        calibration=dict(path=str(args.calibration), sha256=digest(args.calibration / "calibration.json"), manifest_sha256=digest(args.calibration / "manifest.json")),
        seeds={"26100831": list(range(261019701, 261019709)), "26100832": list(range(261019801, 261019809))},
        plan="docs/plans/ledh-sir-beta-repair-20261019.md", result=str(args.output / "result.md"),
        evidence_role="conditional_diagnostic_not_canonical_ledh_not_original_kdm")
    numerical_paths = [path for path in source["source_sha256"] if not path.startswith("scripts/")]
    for path in numerical_paths:
        if digest(ROOT / path) != source["source_sha256"][path]:
            raise ValueError(f"Numerical source changed: {path}")
    manifest["source_sha256"] = {path: digest(ROOT / path) for path in numerical_paths}
    manifest["source_sha256"][str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    write(args.output / "manifest.json", manifest)
    def timeout(_signum, _frame):
        raise TimeoutError("Calibrated evaluation allocation exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        args.device, args.dtype, args.no_tf32 = "gpu", "float64", True
        tensorflow, memory = prepare(args)
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, make_filter
        from bayesfilter.highdim.ledh_backward_guide_tf import FrozenQuadraticBackwardGuide, FrozenMixtureBackwardGuide
        dtype = tensorflow.float64
        spec = spec_for("sir_d18")
        theta = spec.default_theta(dtype)
        controls = ProposalControls(proposal_policy="original_three_branch", reset_epsilon=1., reset_steps=5, flow_steps=16, correction_steps=0)
        manifest.update(memory_policy=memory, tensorflow=tensorflow.__version__, trace_fields=TRACE_FIELDS,
                        theta=theta, parameter_names=spec.parameter_names)
        observations, programs, smoke = {}, {}, []
        plain = make_filter(spec, 256, 50, controls, dtype, jit_compile=True)
        short_plain = make_filter(spec, 32, 2, controls, dtype, jit_compile=True)

        def evaluate(program, observed, fixed, beta):
            results = [program(theta, tensorflow.one_hot(coordinate, 3, dtype=dtype), observed, *fixed, beta)
                       for coordinate in range(3)]
            values = [float(result["value"]) for result in results]
            scores = [float(result["score"]) for result in results]
            valid = all(bool(result["valid"]) and int(result["steps_completed"]) == observed.shape[0] for result in results)
            valid &= all(math.isfinite(value) for value in values + scores)
            if max(values) - min(values) > 1e-9 * max(1., abs(values[0])):
                raise RuntimeError("Analytical direction changed primal value")
            return dict(value=values[0], score=scores, valid=valid,
                        steps_completed=[int(result["steps_completed"]) for result in results],
                        traces=[result["trace"] for result in results], device=results[0]["value"].device)

        for data_seed in (26100831, 26100832):
            identity = str(data_seed)
            data_path = ROOT / source["data"][identity]["path"]
            fit_path = args.guides_from / f"guides-{data_seed}.json"
            if digest(data_path) != source["data"][identity]["sha256"] or digest(fit_path) != source["fits"][identity]["sha256"]:
                raise ValueError("Frozen data or guide changed")
            observed = tensorflow.constant(json.loads(data_path.read_text())["observations"], dtype)
            observations[identity] = observed
            fit = json.loads(fit_path.read_text())
            (args.output / fit_path.name).write_bytes(fit_path.read_bytes())
            guides = dict(csmc_quadratic=FrozenQuadraticBackwardGuide(tensorflow.constant(fit["precision"], dtype), tensorflow.constant(fit["linear"], dtype)),
                          kernel=FrozenMixtureBackwardGuide(*(tensorflow.constant(fit[key], dtype) for key in ("logits", "centers", "covariances")), tensorflow.constant(fit["active"], tensorflow.bool)))
            programs[identity] = {arm: plain for arm in arms[identity]}
            shorts = {arm: short_plain for arm in arms[identity]}
            for arm, guide in guides.items():
                programs[identity][arm] = make_filter(spec, 256, 50, controls, dtype, jit_compile=True, backward_guide=guide)
                fields = ("precision", "linear") if arm == "csmc_quadratic" else ("logits", "centers", "covariances", "active")
                short_guide = type(guide)(*(tensorflow.gather(getattr(guide, field), [0, 49]) for field in fields))
                shorts[arm] = make_filter(spec, 32, 2, controls, dtype, jit_compile=True, backward_guide=short_guide)
            fixed = design(tensorflow, 32, 2, 18, 26101902, dtype)
            for arm, probabilities in arms[identity].items():
                beta = tensorflow.constant(probabilities, dtype)
                result = evaluate(shorts[arm], observed[:2], fixed, beta)
                checks, extrapolations = [], []
                for coordinate in range(3):
                    direction = tensorflow.one_hot(coordinate, 3, dtype=dtype)
                    estimates = []
                    for step in (2.5e-5, 1.25e-5):
                        plus = shorts[arm](theta + step * direction, tensorflow.zeros_like(theta), observed[:2], *fixed, beta)
                        minus = shorts[arm](theta - step * direction, tensorflow.zeros_like(theta), observed[:2], *fixed, beta)
                        numerical = (float(plus["value"]) - float(minus["value"])) / (2 * step)
                        relative = abs(numerical - result["score"][coordinate]) / max(1., abs(numerical))
                        checks.append(dict(coordinate=coordinate, step=step, relative_error=relative,
                                           passed=bool(plus["valid"]) and bool(minus["valid"]) and relative < 2e-5))
                        estimates.append(numerical)
                    extrapolated = (4 * estimates[1] - estimates[0]) / 3
                    relative = abs(extrapolated - result["score"][coordinate]) / max(1., abs(extrapolated))
                    extrapolations.append(dict(coordinate=coordinate, relative_error=relative, passed=relative < 2e-6))
                smoke.append(dict(data_seed=data_seed, arm=arm, beta=probabilities, **result, derivative_checks=checks, richardson=extrapolations))
                write(args.output / "smoke.json", smoke)
                if not result["valid"] or not all(check["passed"] for check in checks + extrapolations):
                    raise RuntimeError(f"Derivative smoke failed: {data_seed}/{arm}")
        manifest["smoke_seconds"] = time.monotonic() - started
        write(args.output / "manifest.json", manifest)
        for replicate in range(8):
            for identity, observed in observations.items():
                seed = manifest["seeds"][identity][replicate]
                fixed = design(tensorflow, 256, 50, 18, seed, dtype)
                for arm, probabilities in arms[identity].items():
                    row_started = time.monotonic()
                    result = evaluate(programs[identity][arm], observed, fixed, tensorflow.constant(probabilities, dtype))
                    row = dict(data_seed=int(identity), seed=seed, arm=arm, beta=probabilities,
                               seconds=time.monotonic() - row_started, reset_epsilon=1., reset_steps=5, **result)
                    with (args.output / "rows.jsonl").open("a") as stream:
                        stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
                    manifest["completed_rows"] += 1
                    manifest["wall_seconds"] = time.monotonic() - started
                    write(args.output / "manifest.json", manifest)
                    print(json.dumps(clean({key: row[key] for key in ("data_seed", "seed", "arm", "seconds", "valid", "score")})), flush=True)
                    if not row["valid"]:
                        raise RuntimeError(f"Invalid evaluated row: {identity}/{seed}/{arm}")
        manifest["status"] = "complete"
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
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=7200)
    run(parser.parse_args())
