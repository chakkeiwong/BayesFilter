"""Bounded diagnostic studies through the real coordinator; no numerical fork."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from bayesfilter.score_study.coordinator import execute, fingerprint, write_json
from bayesfilter.score_study.registry import default_registry

bundle_path, stage, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if stage not in ("smoke", "screen"):
    raise ValueError("stage must be smoke or screen")
bundle = json.loads(bundle_path.read_text())
output.mkdir(exist_ok=False)
started = time.monotonic()
registry = default_registry()
manifest = {"schema": "control_safety_run_v1", "plan": bundle["plan"], "stage": stage,
    "command": sys.argv, "interpreter": sys.executable, "source_root": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    "environment": "tftwogpu", "hardware": "RTX 4080 SUPER; trusted GPU; per-row runtime metadata authoritative",
    "wall_seconds": 0, "studies": [], "status": "running"}
write_json(output / "run-manifest.json", manifest)
checks = []
failure_classes = []


def read_results(state, destination):
    return {row_id: json.loads((destination / entry["result_path"]).read_text())
            for row_id, entry in state["rows"].items() if entry["execution_status"] == "complete"}


try:
    for item in bundle[stage]:
        if time.monotonic() - started >= (900 if stage == "smoke" else 1800):
            raise RuntimeError("phase wall budget exhausted")
        study = json.loads(Path(item["study"]).read_text())
        destination = output / item["name"]
        state = execute(study, registry, destination)
        if state["fingerprint"] != fingerprint(study, registry):
            raise RuntimeError("source drift during execution")
        results = read_results(state, destination)
        manifest["studies"].append({"name": item["name"], "study": item["study"],
            "output": str(destination), "completed": len(results), "rows": len(study["rows"]),
            "wall_seconds": state["wall_seconds"], "status": state["execution_status"]})
        by_dataset = {}
        for row in study["rows"]:
            if row["id"] not in results:
                reasons = state["rows"][row["id"]].get("reasons", [])
                numerical = any("control diagnostics invalid" in text or
                    "nonlinear score : Tensor had" in text or "nonlinear value : Tensor had" in text
                    for text in reasons)
                failure_classes.append({"condition": item["name"], "row": row["id"],
                    "classification": "numerical_candidate_failure" if numerical else "unclassified_failure",
                    "reasons": reasons})
                if stage == "smoke" or not numerical:
                    raise RuntimeError("required wiring or unclassified failure: " + row["id"])
                continue
            result = results[row["id"]]
            key = row["dataset"]
            signature = (result["diagnostics"]["data_version"], result["oracle_value"], result["oracle_score"])
            if key in by_dataset and signature != by_dataset[key]:
                raise RuntimeError("data/reference mismatch within a paired condition")
            by_dataset[key] = signature
            if row.get("collect_control_diagnostics"):
                diag = result["diagnostics"]["control_diagnostics"]
                required = {"higher_moment_minimum_pearson_feasibility_margin",
                    "higher_moment_minimum_finite_particle_upper_margin",
                    "higher_moment_maximum_diagonal_scaled_system_condition",
                    "higher_moment_skew_residual", "higher_moment_kurtosis_residual",
                    "higher_moment_minimum_coordinatewise_cap_derivative", "particle_ess_per_time",
                    "transport_row_sum_error_per_time", "transport_column_weight_error_per_time"}
                if not required <= diag.keys():
                    raise RuntimeError("missing diagnostic fields")
                if stage == "smoke":
                    baseline = results[row["id"].replace("_diagnostics_", "_ordinary_")]
                    pairs = [(result["value"], baseline["value"]), *zip(result["score"], baseline["score"])]
                    scaled = [abs(a-b) / (64 * 2**-23 * (1+abs(b))) for a, b in pairs]
                    entry = {"condition": item["name"], "proposal": row["proposal"],
                        "maximum_absolute_difference": max(abs(a-b) for a,b in pairs),
                        "maximum_scaled_difference": max(scaled), "pass": max(scaled) <= 1}
                    checks.append(entry)
                    if not entry["pass"]:
                        raise RuntimeError("diagnostics changed finite values/scores")
        manifest.update(wall_seconds=time.monotonic()-started)
        write_json(output / "run-manifest.json", manifest)
        print(json.dumps(manifest["studies"][-1]), flush=True)
    manifest["status"] = "complete" if not failure_classes else "complete_with_candidate_failures"
except BaseException as error:
    manifest.update(status="failed", error=type(error).__name__ + ": " + str(error))
    raise
finally:
    manifest.update(wall_seconds=time.monotonic()-started, parity_checks=checks,
                    failure_classes=failure_classes)
    write_json(output / "run-manifest.json", manifest)
