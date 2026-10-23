"""Fixed-OT SIR guidance diagnostic with fresh paired evaluation randomness."""

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
PLAN = ROOT / "docs/plans/ledh-sir-guidance-execution-20261019.md"
SAVED = ROOT / "docs/plans/artifacts/ledh-independent-validation-20261008-01"
REFERENCE = SAVED / "report-20261009-011148/comparison.json"
ARMS = {
    "plain": [1 / 3] * 3,
    "csmc_quadratic": [1 / 3] * 3,
    "kernel": [1 / 3] * 3,
    "prior": [1., 0., 0.],
    "global": [.1, .9, 0.],
    "local": [.1, 0., .9],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    from run_ledh_covariance_proposal import clean
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def previous_rows(directory, current, diagnostic_path=None):
    previous = json.loads((directory / "manifest.json").read_text())
    current = json.loads(json.dumps(current))
    if previous["status"] not in ("budget_exhausted", "complete"):
        raise ValueError("Continuation requires a cleanly stopped attempt")
    fields = ("git_commit", "jit_compile", "dtype", "tf32", "cpu_only",
              "particles", "horizon", "reset_epsilon", "reset_steps", "arms",
              "evidence_role", "guide_fitting", "data", "seeds", "fit_seeds",
              "theta", "parameter_names", "trace_fields", "reference", "tensorflow")
    for field in fields:
        if previous[field] != current[field]:
            raise ValueError(f"Continuation scope mismatch: {field}")
    runner = str(Path(__file__).relative_to(ROOT))
    for path, checksum in previous["source_sha256"].items():
        if path != runner and current["source_sha256"].get(path) != checksum:
            raise ValueError(f"Continuation implementation mismatch: {path}")
    for data_seed, saved_fit in previous["fits"].items():
        fit_path = directory / f"guides-{data_seed}.json"
        if digest(fit_path) != saved_fit["sha256"] or saved_fit["sha256"] != current["fits"][data_seed]["sha256"]:
            raise ValueError("Continuation guide mismatch")
    path = directory / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != previous["completed_rows"]:
        raise ValueError("Continuation row count mismatch")
    expected = {(int(data_seed), seed, arm) for data_seed, seeds in current["seeds"].items()
                for seed in seeds for arm in current["arms"]}
    if len(expected) != 96:
        raise ValueError("Continuation requires the unchanged 96-row design")
    indices = [current["trace_fields"].index(name)
               for name in ("valid", "moment_valid", "correction_cap_valid")]
    found = set()
    quarantined = set()
    diagnostic = None
    if diagnostic_path is not None:
        diagnostic = json.loads(diagnostic_path.read_text())
        if (not diagnostic["reproduced"] or diagnostic["valid"]
                or diagnostic["source_manifest_sha256"] != digest(directory / "manifest.json")
                or diagnostic["reset_epsilon"] != 1. or diagnostic["reset_steps"] != 5):
            raise ValueError("Invalid failure-isolation diagnostic")
        quarantined.add((diagnostic["data_seed"], diagnostic["seed"], diagnostic["arm"]))
    for row in rows:
        key = (row["data_seed"], row["seed"], row["arm"])
        if key not in expected or key in found:
            raise ValueError("Unexpected or duplicate continuation key")
        found.add(key)
        if row["reset_epsilon"] != 1. or row["reset_steps"] != 5 or row["beta"] != current["arms"][row["arm"]]:
            raise ValueError("Continuation numerical controls changed")
        valid = row["valid"] and row["steps_completed"] == [50] * 3
        valid &= len(row["score"]) == 3 and all(math.isfinite(value) for value in [row["value"], *row["score"]])
        valid &= len(row["traces"]) == 3 and all(len(trace) == 50 for trace in row["traces"])
        valid &= all(len(step) == len(current["trace_fields"]) and all(isinstance(value, (int, float)) and math.isfinite(value) for value in step)
                     and all(step[index] == 1. for index in indices)
                     for trace in row["traces"] for step in trace)
        if key in quarantined:
            if valid or row["valid"] or row["steps_completed"] != [diagnostic["steps_completed"]] * 3:
                raise ValueError("Failure-isolation diagnostic does not match row")
        elif not valid:
            raise ValueError("Invalid continuation row")
    return rows, dict(directory=str(directory.resolve()), rows_sha256=digest(path),
                      manifest_sha256=digest(directory / "manifest.json"), count=len(rows),
                      original_wall_seconds=previous["wall_seconds"],
                      original_runner_sha256=previous["source_sha256"][runner],
                      quarantined_keys=sorted(quarantined),
                      diagnostic_sha256=digest(diagnostic_path) if diagnostic_path else None)


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running", command=sys.argv, python=sys.executable, environment="tftwogpu",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        jit_compile=True, dtype="float64", tf32=False, cpu_only=False,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        plan=str(PLAN.relative_to(ROOT)), result=str(output.parent / "result.md"),
        budget_seconds=args.seconds, output=str(output), completed_rows=0, planned_rows=96,
        particles=256, horizon=50, reset_epsilon=1., reset_steps=5, arms=ARMS,
        evidence_role="noncanonical_frozen_guidance_diagnostic_not_original_kdm",
        guide_fitting=dict(particles=512, validation_count=256, components=32,
                           bandwidths=[.5, 1., 2.], nnls_iterations=200,
                           csmc_passes=1, precision_policy="none", solver="centered_qr"),
        reference=dict(path=str(REFERENCE.relative_to(ROOT)), sha256=digest(REFERENCE)),
        data={}, seeds={}, fit_seeds={},
    )
    source_paths = [
        Path(__file__).relative_to(ROOT),
        Path("bayesfilter/score_study/sir_backward_guidance_diagnostic_tf.py"),
        Path("bayesfilter/replication/multivariate_controlled_smc_tf.py"),
        Path("bayesfilter/highdim/covariance_proposal_tf.py"),
        Path("bayesfilter/highdim/ledh_backward_guide_tf.py"),
        Path("bayesfilter/highdim/ledh_canonical_models_tf.py"),
        Path("bayesfilter/highdim/sqmc_nonlinear_tf.py"),
        Path("docs/benchmarks/run_ledh_covariance_proposal.py"),
    ]
    manifest["source_sha256"] = {str(path): digest(ROOT / path) for path in source_paths}
    write(output / "manifest.json", manifest)
    def timed_out(_signum, _frame):
        raise TimeoutError("Campaign worker reached its process budget")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(math.ceil(args.seconds))
    try:
        from run_ledh_covariance_proposal import design, prepare, spec_for
        args.device, args.dtype, args.no_tf32 = "gpu", "float64", True
        tensorflow, memory = prepare(args)
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, make_filter
        from bayesfilter.highdim.ledh_backward_guide_tf import FrozenMixtureBackwardGuide, FrozenQuadraticBackwardGuide
        from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import SIRGuideFitter
        manifest.update(tensorflow=tensorflow.__version__, memory_policy=memory, trace_fields=TRACE_FIELDS)
        dtype = tensorflow.float64
        spec = spec_for("sir_d18")
        theta = spec.default_theta(dtype)
        manifest["theta"] = theta
        manifest["parameter_names"] = spec.parameter_names
        observations = {}
        references = {}
        for dataset in json.loads(REFERENCE.read_text())["datasets"]:
            if dataset["model"] == "sir_d18":
                references[dataset["data_seed"]] = dataset["reference"]["zhao_rank20"]
        manifest["reference"]["datasets"] = references
        for data_seed in (26100831, 26100832):
            path = SAVED / "inputs" / f"sir_d18-d{data_seed}" / "dataset.json"
            data = json.loads(path.read_text())
            if data["target_id"] != "canonical_sir_d18_x0_then_transition_observe_v1":
                raise ValueError("Unexpected SIR target identity")
            observed = tensorflow.constant(data["observations"], dtype)
            if observed.shape != (50, 9):
                raise ValueError("Unexpected SIR data shape")
            observations[data_seed] = observed
            manifest["data"][data_seed] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                observation_sha256=data["observation_sha256"], target_id=data["target_id"])
            first_seed = 261019301 if data_seed == 26100831 else 261019401
            manifest["seeds"][data_seed] = list(range(first_seed, first_seed + 8))
            manifest["fit_seeds"][data_seed] = 261019101 if data_seed == 26100831 else 261019102
        write(output / "manifest.json", manifest)
        fitter = SIRGuideFitter(50) if args.guides_from is None else None
        guides = {}
        for data_seed, observed in observations.items():
            fit_started = time.monotonic()
            def record(stage, diagnostic):
                write(output / f"fit-{data_seed}-{stage}.json", diagnostic)
                print(json.dumps(dict(fit_stage=stage, data_seed=data_seed,
                                      seconds=time.monotonic() - fit_started)), flush=True)
            if args.guides_from is None:
                quadratic, kernel, diagnostic = fitter.fit(observed, manifest["fit_seeds"][data_seed], record)
            else:
                saved_manifest = json.loads((args.guides_from / "manifest.json").read_text())
                saved_path = args.guides_from / f"guides-{data_seed}.json"
                saved_fit = saved_manifest["fits"][str(data_seed)]
                if digest(saved_path) != saved_fit["sha256"]:
                    raise ValueError("Frozen guide digest mismatch")
                if saved_manifest["data"][str(data_seed)] != manifest["data"][data_seed]:
                    raise ValueError("Frozen guide data mismatch")
                diagnostic = json.loads(saved_path.read_text())
                if not diagnostic["valid"] or diagnostic["fit_seed"] != manifest["fit_seeds"][data_seed]:
                    raise ValueError("Frozen guide validity/seed mismatch")
                quadratic = FrozenQuadraticBackwardGuide(
                    tensorflow.constant(diagnostic["precision"], dtype),
                    tensorflow.constant(diagnostic["linear"], dtype))
                kernel = FrozenMixtureBackwardGuide(
                    *(tensorflow.constant(diagnostic[key], dtype) for key in
                      ("logits", "centers", "covariances")),
                    tensorflow.constant(diagnostic["active"], tensorflow.bool))
                quadratic.validate(50, 18, dtype)
                kernel.validate(50, 18, dtype)
                manifest.setdefault("reused_fits", {})[data_seed] = dict(
                    path=str(saved_path), sha256=digest(saved_path), original_fit_seconds=saved_fit["seconds"])
            fit_path = output / f"guides-{data_seed}.json"
            write(fit_path, diagnostic)
            guides[data_seed] = {"csmc_quadratic": quadratic, "kernel": kernel}
            manifest.setdefault("fits", {})[data_seed] = dict(
                artifact=str(fit_path), sha256=digest(fit_path), seconds=time.monotonic() - fit_started,
                valid=bool(diagnostic["valid"]), projected_times=int(tensorflow.reduce_sum(
                    tensorflow.cast(diagnostic["projected"], tensorflow.int32))),
                kernel_active_times=int(tensorflow.reduce_sum(tensorflow.cast(kernel.active, tensorflow.int32))),
            )
            write(output / "manifest.json", manifest)
            print(json.dumps({"fitted": data_seed, **manifest["fits"][data_seed]}), flush=True)

        controls = ProposalControls(reset_epsilon=1., reset_steps=5, flow_steps=16, correction_steps=0)
        def evaluate(program, observed, fixed, beta):
            results = []
            for coordinate in range(3):
                result = program(theta, tensorflow.one_hot(coordinate, 3, dtype=dtype),
                                 observed, *fixed, tensorflow.constant(beta, dtype))
                result["value"].numpy()
                results.append(result)
            values = [float(result["value"]) for result in results]
            scores = [float(result["score"]) for result in results]
            valid = all(bool(result["valid"]) and int(result["steps_completed"]) == observed.shape[0]
                        for result in results)
            valid &= all(math.isfinite(value) for value in values + scores)
            if max(values) - min(values) > 1e-9 * max(1., abs(values[0])):
                raise RuntimeError("Analytical direction changed the primal value")
            return dict(value=values[0], score=scores, valid=valid,
                        steps_completed=[int(result["steps_completed"]) for result in results],
                        traces=[result["trace"] for result in results], device=results[0]["value"].device)

        smoke_results = []
        fixed = design(tensorflow, 32, 2, 18, 26101901, dtype)
        observed = observations[26100831][:2]
        short_guides = {"plain": None}
        quadratic = guides[26100831]["csmc_quadratic"]
        kernel = guides[26100831]["kernel"]
        short_guides["csmc_quadratic"] = FrozenQuadraticBackwardGuide(
            tensorflow.gather(quadratic.precision, [0, 49]), tensorflow.gather(quadratic.linear, [0, 49]))
        short_guides["kernel"] = FrozenMixtureBackwardGuide(
            *(tensorflow.gather(getattr(kernel, key), [0, 49])
              for key in ("logits", "centers", "covariances", "active")))
        for name, guide in short_guides.items():
            program = make_filter(spec, 32, 2, controls, dtype, jit_compile=True, backward_guide=guide)
            result = evaluate(program, observed, fixed, ARMS[name])
            if not result["valid"]:
                raise RuntimeError(f"Invalid {name} smoke")
            checks = []
            for coordinate in range(3):
                direction = tensorflow.one_hot(coordinate, 3, dtype=dtype)
                for step in (1e-4, 5e-5, 2.5e-5, 1.25e-5):
                    plus = program(theta + step * direction, tensorflow.zeros_like(theta),
                                   observed, *fixed, tensorflow.constant(ARMS[name], dtype))
                    minus = program(theta - step * direction, tensorflow.zeros_like(theta),
                                    observed, *fixed, tensorflow.constant(ARMS[name], dtype))
                    numerical = (float(plus["value"]) - float(minus["value"])) / (2 * step)
                    relative = abs(numerical - result["score"][coordinate]) / max(1., abs(numerical))
                    passed = bool(plus["valid"]) and bool(minus["valid"]) and relative < 2e-5
                    checks.append(dict(coordinate=coordinate, step=step, numerical=numerical,
                                       analytical=result["score"][coordinate], relative_error=relative, passed=passed))
            smoke_results.append(dict(arm=name, **result, derivative_checks=checks))
            write(output / "smoke.json", smoke_results)
            fine_checks = [check for check in checks if check["step"] <= 2.5e-5]
            richardson = []
            for coordinate in range(3):
                values = [check["numerical"] for check in fine_checks if check["coordinate"] == coordinate]
                extrapolated = (4 * values[1] - values[0]) / 3
                relative = abs(extrapolated - result["score"][coordinate]) / max(1., abs(extrapolated))
                richardson.append(dict(coordinate=coordinate, extrapolated=extrapolated,
                                       relative_error=relative, passed=relative < 2e-6))
            smoke_results[-1]["richardson"] = richardson
            write(output / "smoke.json", smoke_results)
            if not all(check["passed"] for check in fine_checks + richardson):
                raise RuntimeError(f"Analytical derivative check failed for {name}")
        print("SIR GPU/XLA smoke, 18 fine-step differences and 9 Richardson checks passed", flush=True)
        retained = set()
        if args.resume_from is not None:
            rows, provenance = previous_rows(args.resume_from, json.loads((output / "manifest.json").read_text()), args.quarantine_diagnostic)
            with (output / "rows.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    retained.add((row["data_seed"], row["seed"], row["arm"]))
            manifest.update(completed_rows=len(rows), resumed_rows=provenance)
            write(output / "manifest.json", manifest)
            print(json.dumps({"retained_rows": len(rows), "remaining_rows": 96 - len(rows)}), flush=True)

        plain = make_filter(spec, 256, 50, controls, dtype, jit_compile=True)
        programs = {}
        for data_seed in observations:
            programs[data_seed] = {name: plain for name in ("plain", "prior", "global", "local")}
            programs[data_seed].update({name: make_filter(spec, 256, 50, controls, dtype,
                jit_compile=True, backward_guide=guide) for name, guide in guides[data_seed].items()})
        last_seconds = 5.
        stop = False
        for replicate in range(8):
            for data_seed, observed in observations.items():
                seed = manifest["seeds"][data_seed][replicate]
                fixed = design(tensorflow, 256, 50, 18, seed, dtype)
                for arm, beta in ARMS.items():
                    if (data_seed, seed, arm) in retained:
                        continue
                    elapsed = time.monotonic() - started
                    if elapsed + max(15., 1.5 * last_seconds) >= args.seconds:
                        stop = True
                        break
                    row_started = time.monotonic()
                    result = evaluate(programs[data_seed][arm], observed, fixed, beta)
                    last_seconds = time.monotonic() - row_started
                    row = dict(data_seed=data_seed, seed=seed, arm=arm, seconds=last_seconds,
                               reset_epsilon=1., reset_steps=5, beta=beta, **result)
                    from run_ledh_covariance_proposal import clean
                    with (output / "rows.jsonl").open("a") as stream:
                        stream.write(json.dumps(clean(row), allow_nan=False) + "\n")
                    manifest["completed_rows"] += 1
                    manifest["wall_seconds"] = time.monotonic() - started
                    write(output / "manifest.json", manifest)
                    print(json.dumps({key: row[key] for key in ("data_seed", "seed", "arm", "seconds", "valid", "score")}), flush=True)
                    if not row["valid"]:
                        raise RuntimeError(f"Invalid evaluated row: {data_seed}/{seed}/{arm}")
                if stop:
                    break
            if stop:
                break
        manifest["status"] = "complete" if manifest["completed_rows"] == 96 else "budget_exhausted"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        manifest["wall_seconds"] = time.monotonic() - started
        write(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--guides-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--quarantine-diagnostic", type=Path)
    run(parser.parse_args())
