"""Offline fresh-failure localization and inactive-option trajectory parity."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = ROOT.parent / "m11-r1/fresh-0-bounded-replay-gpu-r1"
    new = ROOT / "inactive-option-parity-gpu-r1"
    a, b = read(old / "assessment.json"), read(new / "assessment.json")
    tensor_pairs = []
    for index in range(len(a["windows"])):
        left, right = (root / f"window-{index:02d}-latent.tensor" for root in (old, new))
        tensor_pairs.append({"window": index, "baseline": str(left.relative_to(REPO)),
            "current": str(right.relative_to(REPO)), "sha256": sha(left), "equal": sha(left) == sha(right)})
    parity = {"all_latent_tensors_equal": len(tensor_pairs) == 5 and all(p["equal"] for p in tensor_pairs),
        "final_transform_equal": a["final_transform"] == b["final_transform"],
        "final_factor_equal": a["final_transform"]["factor"] == b["final_transform"]["factor"],
        "final_epsilon_equal": a["final_epsilon"] == b["final_epsilon"], "tensor_pairs": tensor_pairs,
        "probe_num_results": b["config"]["metric_probe_num_results"], "candidate_authority": False}
    assert parity["all_latent_tensors_equal"] and parity["final_factor_equal"] and parity["final_epsilon_equal"]

    original = ROOT / "regression-sequential-probe-fresh-0-gpu-r1/preparation/preparation_progress.json"
    replay = ROOT / "fresh-0-trajectory-replay-gpu-r1"
    def windows(payload):
        return [e["details"]["window"] for e in payload["events"] if e["phase"].endswith("metric_decision")]
    before, after = windows(read(original)), windows(read(replay / "preparation_progress.json"))
    equal = len(before) == len(after) and all(
        {k:v for k,v in a.items() if k != "runtime_s"} == {k:v for k,v in b.items() if k != "runtime_s"}
        for a, b in zip(before, after))
    assert equal
    trace_windows = read(replay / "trajectory_diagnostics/windows.json")
    assert trace_windows[-1]["failed_indices"] == [19]
    retained_ok = all(w[key] for w in trace_windows for key in (
        "retained_state_finite", "retained_score_finite", "retained_target_finite"))
    records = []
    for path in sorted(replay.glob("trajectory_diagnostics/window-*/transition-*/trajectory.json")):
        row = read(path)
        exact = all(v["equal_including_nan"] for v in row["endpoint_comparison"].values())
        record = {"path": str(path.relative_to(REPO)), "sha256": sha(path),
                  "transition": row["transition_index"], "epsilon": row["epsilon"],
                  "first_nonfinite": row["first_nonfinite"], "endpoint_equal": exact}
        assert exact
        if row["first_nonfinite"]:
            k = row["first_nonfinite"]["leapfrog_index"]
            transform = row["affine_layers"][0]["transform"]
            position = row["trajectory"]["position"][k]
            theta = [c + sum(x*y for x,y in zip(f, position)) for c,f in zip(transform["center"],transform["factor"])]
            record.update(theta=theta, exp_negative_twice_log_sigma_overflows=(
                -2*theta[-1] > math.log(float.fromhex("0x1.fffffffffffffp+1023"))), accepted=row["accepted"])
            record["exp_positive_twice_log_sigma_overflows"] = (
                2*theta[-1] > math.log(float.fromhex("0x1.fffffffffffffp+1023")))
        records.append(record)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "inactive_option_parity": parity,
        "fresh_failure": {"completed_windows_equal_except_runtime": equal, "retained_health_finite": retained_ok,
                          "failed_indices": trace_windows[-1]["failed_indices"], "trajectories": records},
        "diagnostic_only": True, "ranking_supported": False, "default_promoted": False}
    with (ROOT / "final-checks-assessment.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
