"""Offline diagnosis of the preserved M12 GPU transition reconstruction."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def read(path):
    return json.loads(path.read_text())


def main():
    baseline = ROOT.parent / "m11-r1/regression-bounded-boundary-fresh-2-gpu-r1/preparation/preparation_progress.json"
    replay = ROOT / "fresh-2-trajectory-replay-gpu-r2"
    def windows(payload):
        return [e["details"]["window"] for e in payload["events"] if e["phase"].endswith("metric_decision")]
    old, new = windows(read(baseline)), windows(read(replay / "preparation_progress.json"))
    differences = [[key for key in a if key != "runtime_s" and a[key] != b.get(key)] for a, b in zip(old, new)]
    if len(old) != len(new) or len(old) != 3 or any(differences):
        raise ValueError("completed numerical windows did not reproduce")
    rows = []
    for path in sorted(replay.glob("trajectory_diagnostics/window-03/transition-*/trajectory.json")):
        report = read(path)
        if not all(value["equal_including_nan"] for value in report["endpoint_comparison"].values()):
            raise ValueError("independent endpoint reconstruction mismatch")
        first = report["first_nonfinite"]
        row = {"transition": report["transition_index"], "epsilon": report["epsilon"], "first_nonfinite": first,
               "endpoint_equal_including_nan": True, "path": str(path.relative_to(REPO)),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if first:
            index = first["leapfrog_index"]
            transform = report["affine_layers"][0]["transform"]
            z = report["trajectory"]["position"][index]
            # Post-run diagnostic arithmetic; numerical execution is already archived.
            theta = [c + sum(a*b for a, b in zip(f, z)) for c, f in zip(transform["center"], transform["factor"])]
            exponent = -2*theta[-1]
            row.update(theta_at_first_invalid_value=theta, likelihood_scale_exponent=exponent,
                log_fp64_max=math.log(float.fromhex("0x1.fffffffffffffp+1023")),
                exponential_overflow=exponent > math.log(float.fromhex("0x1.fffffffffffffp+1023")),
                accepted=report["accepted"], previous_target=report["trajectory"]["target"][index-1])
            assert row["exponential_overflow"] and not row["accepted"] and first["field"] == "target"
        rows.append(row)
    trace_windows = read(replay / "trajectory_diagnostics/windows.json")
    assert trace_windows[-1]["failed_indices"] == [31, 153]
    assert all(row[field] for row in trace_windows for field in (
        "retained_state_finite", "retained_score_finite", "retained_target_finite"))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
        "baseline": str(baseline.relative_to(REPO)), "completed_windows_equal_except_runtime": True,
        "retained_health_finite": True, "transitions": rows,
        "diagnosis": "rejected proposal trajectories overflow exp(-2 log_sigma); no retained invalidity observed",
        "tfp_source": "installed hmc.py one_step and SimpleLeapfrogIntegrator._one_step; hashes in instrumentation.json",
        "next_hypothesis": "longer fixed-step probes before a metric handoff; no invalid window recovery",
        "default_promoted": False, "statistical_ranking_supported": False}
    with (ROOT / "trajectory-assessment.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
