"""Post-run artifact audit and descriptive tables; no training or numerical kernel."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26"


def read(path):
    return json.loads(path.read_text())


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit(directory):
    m, r = read(directory / "manifest.json"), read(directory / "result.json")
    initial = read(directory / "initial-checkpoint.json")
    checkpoint = read(directory / "checkpoint.json")
    frozen = read(directory / "frozen-map.json")
    p, coverage = read(directory / "post-training-1000.json"), read(directory / "coverage.json")
    initial_probe = read(directory / "initial-post-training-1000.json")
    history = read(directory / "progress.json")
    checks = {
        "checkpoint_hash": content_hash({k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}) == checkpoint["checkpoint_hash"],
        "frozen_hash": content_hash({k: v for k, v in frozen.items() if k != "transport_hash"}) == frozen["transport_hash"],
        "frozen_checkpoint_binding": frozen["training_state_hash"] == checkpoint["checkpoint_hash"],
        "frozen_parameter_equality": frozen["parameters"] == checkpoint["parameters"],
        "target_binding": m["target_signature"] == frozen["target_signature"] == checkpoint["target_signature"] == initial["target_signature"],
        "canonical_width": m["transport_config"]["hidden_layers"] == [16, 16],
        "memory_growth": m["memory_policy"]["all_physical_devices_memory_growth"] is True,
        "tf32_off_xla_on": m["tf32"] is False and m["jit_compile"] is True,
        "probe_complete_finite": p["complete"] and p["finite"] and p["valid_rows"] == p["rows"] == 1000,
        "initial_probe_complete_finite": initial_probe["complete"] and initial_probe["finite"] and initial_probe["valid_rows"] == 1000,
        "update_counter": int(checkpoint["optimizer"][0]["value"]) == r["optimizer_updates"],
        "declared_artifact_hashes": all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                                       for name, digest in r["artifact_sha256"].items()),
        "progress_all_valid": all(row["valid"] for row in history),
        "coverage_measure": coverage["reference_log_weight_definition"] == "log_p_minus_log_g_not_log_p_minus_log_q",
    }
    mismatches = []
    for name, expected in m["source_sha256"].items():
        committed = subprocess.check_output(["git", "show", f"{m['git_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected:
            mismatches.append(name)
    checks["source_hashes_match_recorded_commit"] = not mismatches
    if "continuation" in m:
        parent = Path(m["continuation"]["run"])
        checks["continuation_hash"] = read(parent / "checkpoint.json")["checkpoint_hash"] == m["continuation"]["checkpoint_hash"]
        checks["continuation_initial_state"] = read(parent / "initial-checkpoint.json") == initial
    clips = [x for row in history for x in row.get("correction_clipped_fractions", [])]
    return {"directory": str(directory.relative_to(ROOT)), "arm": m["arm"], "target": m["target"], "seed": m["seed"],
        "status": r["status"], "updates": r["optimizer_updates"], "wall_seconds": r["wall_seconds"],
        "git_commit": m["git_commit"], "target_signature": m["target_signature"],
        "initial_state_hash": hashlib.sha256((directory / "initial-checkpoint.json").read_bytes()).hexdigest(),
        "checks": checks, "source_commit_mismatches": mismatches,
        "score_residual_rms": p["score_residual_rms"], "centered_log_density_rms": p["centered_log_density_rms"],
        "score_residual_norm": p["score_residual_norm"], "scale_diagnostics": p["scale_diagnostics"],
        "initial_score_residual_rms": initial_probe["score_residual_rms"],
        "initial_centered_log_density_rms": initial_probe["centered_log_density_rms"],
        "coverage": coverage,
        "last_ten_ais_ess_median": statistics.median(row["ais_ess"] for row in history[-10:]) if m["arm"] == "fab" else None,
        "replay_correction_clipped_fraction": statistics.mean(clips) if clips else None}


def main():
    runs = [audit(p.parent) for phase in ("three-mode", "q20-r1", "q20-continuation-r1")
            for p in sorted((BASE / phase).glob("*/result.json"))]
    pairs = []
    for seed in (0, 1, 2):
        selected = {}
        for arm in ("fab", "reverse_kl"):
            selected[arm] = max((r for r in runs if r["target"] == "q20" and r["seed"] == seed and r["arm"] == arm),
                                key=lambda r: r["updates"])
        fab, rkl = selected["fab"], selected["reverse_kl"]
        matched = fab["updates"] == rkl["updates"] == 240
        pairs.append({"seed": seed, "arms": {arm: r["directory"] for arm, r in selected.items()},
            "initial_state_identical": fab["initial_state_hash"] == rkl["initial_state_hash"],
            "target_identical": fab["target_signature"] == rkl["target_signature"],
            "complete_matched_240_updates": matched,
            "fab_passes_descriptive_geometry_screen": (fab["score_residual_rms"] < rkl["score_residual_rms"] and
                fab["centered_log_density_rms"] < rkl["centered_log_density_rms"]) if matched else None})
    queue = read(BASE / "q20-continuation-r1/queue.json")
    charged = queue["previous_worker_seconds"] + sum(r["process_wall_seconds"] for r in queue["completed"])
    smoke = sum(read(p)["wall_seconds"] for phase in ("three-mode", "preflight")
                for p in (BASE / phase).glob("*/result.json"))
    result = {"schema": "bayesfilter.fab_iaf_campaign.summary.v1", "runs": runs, "pairs": pairs,
        "all_recorded_checks_pass": all(all(r["checks"].values()) for r in runs),
        "budget": {"q20_and_pricing_charged_seconds": charged, "q20_cap_seconds": 16200.,
                   "remaining_seconds": 16200. - charged, "smoke_control_charged_seconds": smoke, "smoke_control_cap_seconds": 600.},
        "interpretation": {"training_recipe_promoted": False, "posterior_ready": False, "all_modes_verified": False,
            "statistically_supported_ranking": False, "complete_q20_pairs": sum(p["complete_matched_240_updates"] for p in pairs),
            "decision": "bounded_candidate_does_not_jointly_pass_whitening_and_coverage; third_pair_incomplete"}}
    (BASE / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks": result["all_recorded_checks_pass"], "pairs": pairs, "budget": result["budget"],
                      "failures": [{"run": r["directory"], "checks": [k for k,v in r["checks"].items() if not v],
                                    "source_mismatches":r["source_commit_mismatches"]} for r in runs if not all(r["checks"].values())]}, indent=2))


if __name__ == "__main__":
    main()
