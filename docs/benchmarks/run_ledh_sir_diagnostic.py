"""Bounded SIR diagnostic of the shared finite covariance-mixture consumer."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
PLAN = ROOT / "docs/plans/ledh-sir-diagnostic-20261018.md"
SAVED = ROOT / "docs/plans/artifacts/ledh-independent-validation-20261008-01"
REFERENCE = SAVED / "report-20261009-011148/comparison.json"
FROZEN_BETA = [0.1, 0.04913405758312983, 0.8508659424168701]
ARMS = {
    "requested5": (FROZEN_BETA, 5),
    "baseline40": (FROZEN_BETA, 40),
    "prior-map": ([1.0, 0.0, 0.0], 5),
    "defensive-global": ([0.1, 0.9, 0.0], 5),
    "defensive-local": ([0.1, 0.0, 0.9], 5),
    "equal-mixture": ([1 / 3, 1 / 3, 1 / 3], 5),
}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references():
    saved = json.loads(REFERENCE.read_text())
    selected = {}
    for dataset in saved["datasets"]:
        if dataset["model"] == "sir_d18":
            reference = dataset["reference"]["zhao_rank20"]
            selected[dataset["data_seed"]] = {
                "theta": dataset["theta"],
                "parameter_names": dataset["parameter_names"],
                "reference": {key: reference[key] for key in
                              ("value", "score", "value_se", "score_se",
                               "replications", "complete", "kind")},
            }
    if sorted(selected) != [26100831, 26100832]:
        raise RuntimeError("unexpected reference data sets")
    return selected


def run(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "command": sys.argv, "python": sys.executable,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "environment": "tftwogpu", "device": "gpu", "cpu_only": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "tf_xla_flags": os.environ.get("TF_XLA_FLAGS"),
        "jit_compile": True, "dtype": "float64", "tf32": False,
        "evidence_role": "noncanonical_consumer_diagnostic_not_admission",
        "csmc_kdm_sir_guidance_fitted": False,
        "plan": str(PLAN.relative_to(ROOT)), "result": str(output / "result.md"),
        "output": str(output), "gpu_worker_budget_seconds": args.seconds,
        "planned_rows": 96, "completed_rows": 0, "arms": ARMS,
        "particles": 1008, "horizon": 50, "chunk_extent": 1008,
        "source_sha256": {str(PLAN.relative_to(ROOT)): digest(PLAN),
                          str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        "data": {}, "seeds": {}, "reference": references(),
        "reference_sha256": digest(REFERENCE),
    }
    write_json(output / "manifest.json", manifest)
    try:
        from run_ledh_covariance_proposal import design, prepare, spec_for
        settings = argparse.Namespace(device="gpu", dtype="float64", no_tf32=True)
        tensorflow, memory = prepare(settings)
        manifest.update(tensorflow=tensorflow.__version__, memory_policy=memory)
        from bayesfilter.highdim.covariance_proposal_tf import (
            ALGORITHM_ID, ProposalControls, TRACE_FIELDS, make_filter)
        spec = spec_for("sir_d18")
        dtype = tensorflow.float64
        theta = tensorflow.zeros([3], dtype)
        programs = {
            count: make_filter(spec, 1008, 50,
                               ProposalControls(reset_steps=count), dtype,
                               jit_compile=True)
            for count in (5, 40)
        }
        observations = {}
        for data_seed in manifest["reference"]:
            path = SAVED / f"inputs/sir_d18-d{data_seed}/dataset.json"
            data = json.loads(path.read_text())
            if (data["target_id"] != "canonical_sir_d18_x0_then_transition_observe_v1"
                    or len(data["observations"]) != 50
                    or any(len(row) != 9 for row in data["observations"])):
                raise RuntimeError("unexpected prepared SIR target/data shape")
            observations[data_seed] = tensorflow.constant(data["observations"], dtype)
            manifest["data"][data_seed] = {
                "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                "observation_sha256": data["observation_sha256"],
                "target_id": data["target_id"], "generator": data["generator"],
            }
            first_seed = 261018301 if data_seed == 26100831 else 261018401
            manifest["seeds"][data_seed] = list(range(first_seed, first_seed + 8))

        def evaluate(program, observed, fixed, beta):
            scores = []
            directions = []
            first = None
            for coordinate in range(3):
                result = program(theta, tensorflow.one_hot(coordinate, 3, dtype=dtype),
                                 observed, *fixed, beta)
                value = float(result["value"])
                score = float(result["score"])
                valid = bool(result["valid"])
                completed = int(result["steps_completed"])
                directions.append({"coordinate": coordinate, "valid": valid,
                                   "steps_completed": completed})
                if not valid or completed != int(observed.shape[0]):
                    raise RuntimeError(f"invalid/incomplete SIR direction {directions}")
                if not math.isfinite(value) or not math.isfinite(score):
                    raise RuntimeError("nonfinite SIR value/score")
                if first is None:
                    first = result
                elif abs(value - float(first["value"])) > 1e-9 * max(1.0, abs(value)):
                    raise RuntimeError("direction-dependent primal value")
                scores.append(score)
            return {"value": float(first["value"]), "score": scores,
                    "valid": True, "directions": directions,
                    "trace": first["trace"].numpy().tolist(),
                    "device": first["value"].device}

        smoke_started = time.monotonic()
        smoke_program = make_filter(spec, 32, 2, ProposalControls(reset_steps=5),
                                    dtype, jit_compile=True)
        smoke_observed = observations[26100831][:2]
        smoke_fixed = design(tensorflow, 32, 2, 18, 26101801, dtype)
        smoke_beta = tensorflow.constant(FROZEN_BETA, dtype)
        smoke = evaluate(smoke_program, smoke_observed, smoke_fixed, smoke_beta)
        parity = []
        for coordinate in range(3):
            direction = tensorflow.one_hot(coordinate, 3, dtype=dtype)
            for step in (1e-4, 5e-5):
                plus = smoke_program(theta + step * direction,
                                     tensorflow.zeros_like(theta), smoke_observed,
                                     *smoke_fixed, smoke_beta)
                minus = smoke_program(theta - step * direction,
                                      tensorflow.zeros_like(theta), smoke_observed,
                                      *smoke_fixed, smoke_beta)
                difference = (float(plus["value"]) - float(minus["value"])) / (2 * step)
                error = difference - smoke["score"][coordinate]
                valid = bool(plus["valid"]) and bool(minus["valid"])
                parity.append({"coordinate": coordinate, "step": step,
                               "finite_difference": difference, "error": error,
                               "valid": valid})
                if not valid or not math.isfinite(error):
                    raise RuntimeError("invalid SIR finite-difference smoke")
                if step == 5e-5 and abs(error) > 2e-4 * max(1, abs(smoke["score"][coordinate])):
                    raise RuntimeError("SIR analytical/FD smoke parity failed")
        write_json(output / "smoke.json", dict(smoke, parity=parity,
                   trace_fields=TRACE_FIELDS, wall_seconds=time.monotonic() - smoke_started))
        manifest.update(algorithm_id=ALGORITHM_ID, trace_fields=TRACE_FIELDS,
                        logical_devices=[str(device) for device in
                                         tensorflow.config.list_logical_devices()],
                        smoke_passed=True)
        write_json(output / "manifest.json", manifest)
        durations = []
        stopped = False
        with (output / "rows.jsonl").open("x") as rows_file:
            for seed_offset in range(8):
                for data_seed in sorted(observations):
                    seed = manifest["seeds"][data_seed][seed_offset]
                    fixed = design(tensorflow, 1008, 50, 18, seed, dtype)
                    for arm, (beta, count) in ARMS.items():
                        reserve = max(35.0, 2 * max(durations[-6:], default=0.0))
                        if time.monotonic() - started + reserve >= args.seconds:
                            stopped = True
                            break
                        row_started = time.monotonic()
                        print(json.dumps({"stage": "row_started", "data_seed": data_seed,
                                          "seed": seed, "arm": arm}), flush=True)
                        result = evaluate(programs[count], observations[data_seed], fixed,
                                          tensorflow.constant(beta, dtype))
                        duration = time.monotonic() - row_started
                        durations.append(duration)
                        row = dict(result, data_seed=data_seed, seed=seed, arm=arm,
                                   beta=beta, reset_steps=count, wall_seconds=duration)
                        rows_file.write(json.dumps(row, allow_nan=False) + "\n")
                        rows_file.flush()
                        manifest["completed_rows"] += 1
                        manifest["wall_seconds"] = time.monotonic() - started
                        write_json(output / "manifest.json", manifest)
                        print(json.dumps({key: row[key] for key in
                                          ("data_seed", "seed", "arm", "value", "score",
                                           "valid", "wall_seconds")}), flush=True)
                    if stopped:
                        break
                if stopped:
                    break
        manifest["status"] = "budget_partial" if stopped else "complete"
    except Exception as error:
        manifest.update(status="failed", failure_type=type(error).__name__,
                        failure=str(error))
        raise
    finally:
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.is_file() and path.is_relative_to(ROOT) and path.suffix == ".py":
                    manifest["source_sha256"][str(path.relative_to(ROOT))] = digest(path)
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)


def average(values):
    return statistics.mean(values)


def mcse(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def analyze(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    rows_path = output / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    generator = random.Random(26101899)
    summary = {"status": manifest["status"], "rows": len(rows), "datasets": [],
               "ranking_conditioning": "fixed approximate reference, not true score",
               "heuristic_dominance_verdict": "not_checked", "heuristic_losses": [],
               "hard_veto": manifest["status"] == "failed", "default_ready": False}
    for data_seed, source in sorted(manifest["reference"].items()):
        reference = source["reference"]
        grouped = {arm: [row for row in rows if str(row["data_seed"]) == data_seed
                         and row["arm"] == arm] for arm in ARMS}
        stats = {}
        errors = {}
        for arm, selected in grouped.items():
            if not selected:
                continue
            vectors = [row["score"] for row in selected]
            differences = [[score - target for score, target in zip(vector, reference["score"])]
                           for vector in vectors]
            errors[arm] = {row["seed"]: math.sqrt(sum(value * value for value in difference))
                           for row, difference in zip(selected, differences)}
            stats[arm] = {
                "replications": len(selected),
                "score_mean": [average(vector[index] for vector in vectors) for index in range(3)],
                "score_mcse": [mcse([vector[index] for vector in vectors]) for index in range(3)],
                "coordinate_rmse": [math.sqrt(average(vector[index] ** 2 for vector in differences))
                                    for index in range(3)],
                "mean_l2_error": average(list(errors[arm].values())),
                "value_mean": average(row["value"] for row in selected),
                "value_mcse": mcse([row["value"] for row in selected]),
                "value_bias_vs_reference": average(row["value"] for row in selected) - reference["value"],
            }
        contrasts = {}
        for comparator in tuple(ARMS)[1:]:
            paired = sorted(set(errors.get("requested5", {})) & set(errors.get(comparator, {})))
            if not paired:
                continue
            differences = [errors["requested5"][seed] - errors[comparator][seed] for seed in paired]
            interval = None
            if len(paired) == 8:
                samples = sorted(average(generator.choices(differences, k=len(differences)))
                                 for _ in range(20000))
                interval = [samples[49], samples[19949]]
            contrasts[comparator] = {
                "paired_replications": len(paired), "mean_l2_difference": average(differences),
                "bonferroni_99_5_percentile_interval": interval,
                "requested5_lower_error_supported": interval is not None and interval[1] < 0,
                "requested5_higher_error_supported": interval is not None and interval[0] > 0,
            }
            if comparator != "baseline40" and len(paired) == 8:
                losses = [index for index in range(3)
                          if stats["requested5"]["coordinate_rmse"][index]
                          > stats[comparator]["coordinate_rmse"][index]]
                if losses or contrasts[comparator]["mean_l2_difference"] > 0:
                    summary["heuristic_losses"].append({"data_seed": int(data_seed),
                        "comparator": comparator, "coordinate_rmse_losses": losses,
                        "mean_l2_loss": contrasts[comparator]["mean_l2_difference"] > 0})
        summary["datasets"].append(dict(source, data_seed=int(data_seed), arms=stats, contrasts=contrasts))
    complete = manifest["status"] == "complete" and len(rows) == 96
    if complete:
        summary["heuristic_dominance_verdict"] = "veto_descriptive_loss" if summary["heuristic_losses"] else "screen_pass"
    summary["primary_criterion_passed"] = complete and not summary["heuristic_losses"] and all(
        dataset["contrasts"].get("baseline40", {}).get("requested5_lower_error_supported", False)
        for dataset in summary["datasets"])
    summary["complete_design"] = complete
    paired_counts = [dataset["contrasts"].get("baseline40", {}).get("paired_replications", 0)
                     for dataset in summary["datasets"]]
    uncertainty = f"{paired_counts} completed primary pairs by dataset; fixed approximate reference"
    ranking_status = (
        "Only contrasts whose simultaneous interval excludes zero, conditional on fixed approximate reference"
        if complete else "None: incomplete eight-pair design; no confidence intervals or ranking issued")
    summary["diagnostics"] = {}
    for field in ("ess", "max_weight", "sinkhorn_row_error", "sinkhorn_column_error",
                  "reset_mean_error", "reset_covariance_relative_error", "moment_roundoff_warning"):
        index = manifest.get("trace_fields", []).index(field) if field in manifest.get("trace_fields", []) else None
        values = [trace[index] for row in rows for trace in row["trace"]] if index is not None else []
        if values:
            summary["diagnostics"][field] = {"minimum": min(values), "maximum": max(values)}
    summary["analysis_wall_seconds"] = time.monotonic() - started
    write_json(output / "analysis.json", summary)
    report = ["# Fresh SIR diagnostic results", "", f"Status: {manifest['status']}; {len(rows)}/96 rows.",
              "This is the current optional finite covariance-mixture consumer, not an admitted canonical route.",
              "No SIR cSMC/KDM fitting or guidance is tested. Reference errors are conditional on a saved approximate Zhao–Cui solution.", ""]
    for dataset in summary["datasets"]:
        report += [f"## Data seed {dataset['data_seed']}", "",
                   f"Reference score: {dataset['reference']['score']}; between-fit SE: {dataset['reference']['score_se']}.",
                   "Rank/support/approximation bias is unbounded; these SEs are not total oracle uncertainty.", "",
                   "| Arm | n | Mean score | Score MCSE | Coordinate RMSE | Mean L2 error |",
                   "|---|---:|---|---|---|---:|"]
        for arm, metrics in dataset["arms"].items():
            display = lambda values: ", ".join("N/A" if value is None else f"{value:.4f}" for value in values)
            report.append(f"| {arm} | {metrics['replications']} | {display(metrics['score_mean'])} | {display(metrics['score_mcse'])} | {display(metrics['coordinate_rmse'])} | {metrics['mean_l2_error']:.4f} |")
        report += ["", "Paired requested5 minus comparator L2 errors (negative favors requested5):", ""]
        for comparator, contrast in dataset["contrasts"].items():
            report.append(f"- {comparator}: {contrast['mean_l2_difference']:.4f}; simultaneous interval {contrast['bonferroni_99_5_percentile_interval']}.")
        report.append("")
    report += ["## Decision", "", "| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |",
               "|---|---|---|---|---|---|",
               f"| Diagnostic only; no promotion | {summary['primary_criterion_passed']} | hard veto={summary['hard_veto']}; heuristic={summary['heuristic_dominance_verdict']} | {uncertainty} | Inspect conditional score errors before any SIR guide-fitting campaign | Exact-oracle, canonical, default or cSMC/KDM improvement |", "",
               "## Inference status", "", "| Question | Status |", "|---|---|",
               f"| Hard veto screen | {manifest['status']}; {manifest.get('failure', 'all completed rows valid')} |",
               f"| Statistically supported ranking | {ranking_status} |",
               "| Descriptive-only differences | Score means/MCSE, coordinate RMSE, likelihood and trace diagnostics |",
               "| Default-readiness | Not evaluated; controls are untuned and route is noncanonical |",
               "| Next evidence needed | SIR-specific fitted-guidance design, untouched validation and stronger reference/replication |", "",
               "## Post-run red team", "",
               "The strongest alternative explanation is approximation error in the fixed reference combined with seed noise. More replications and an independent, more accurate reference are needed to establish a conditional ranking. The small replication count limits interpretation, especially for extreme score errors. A failed current setting is not evidence against the future-guidance research direction.", "",
               "Provenance and commands: `manifest.json`; fresh complete rows and traces: `rows.jsonl`; analytical/FD smoke: `smoke.json`; structured analysis: `analysis.json`."]
    (output / "result.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": summary["status"], "rows": len(rows),
                      "primary_criterion_passed": summary["primary_criterion_passed"],
                      "heuristic_dominance_verdict": summary["heuristic_dominance_verdict"],
                      "analysis_wall_seconds": summary["analysis_wall_seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200:
        parser.error("worker budget must be positive and at most 1200 seconds")
    if args.command == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
