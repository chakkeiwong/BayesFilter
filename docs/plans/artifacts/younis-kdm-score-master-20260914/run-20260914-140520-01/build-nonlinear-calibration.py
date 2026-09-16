"""Build the predeclared nonlinear pilot configuration; no numerical backend."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "nonlinear-calibration-01"
PLAN = str(ROOT.parents[2] / "younis-score-nonlinear-calibration-2026-09-15.md")
DEST.mkdir(exist_ok=False)
partitions = {"calibration": list(range(300, 304)), "validation": [310, 311], "claim": list(range(400, 424))}
providers = ["ledh", "sgqf", "kdm_covariance"]
heuristics = ["ekf", "ukf", "bootstrap", "local_linear"]
campaign = {"plan": PLAN, "max_numerical_rows": 1440, "wall_seconds": 5400,
            "calibration": [], "evaluation": [], "progress_output": str(DEST/"progress.json"),
            "comparison_output": str(DEST/"comparison.json"),
            "comparison_contract": {"conditions": ["weak", "curved", "concentrated"],
                "methods": providers+heuristics, "candidates": providers, "heuristics": heuristics,
                "datasets": partitions["claim"], "replicates": [0, 1],
                "coordinates": ["A", "log_sigma_Q", "log_sigma_R", "H", "m0", "log_sigma_P0"],
                "bootstrap_resamples": 20000, "bootstrap_seed": 95231}}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def row(condition, proposal, dataset, replicate, role, label, **extra):
    return {"id": f"{label}_{dataset}_{replicate}", "model": "nonlinear_scalar", "proposal": proposal,
            "estimator": "nonlinear_analytical", "comparison_target": "model_score",
            "comparison": "approximation_error", "condition": condition, "dataset": dataset,
            "replicate": replicate, "role": role, "coupling_group": "baseline", **extra}


for condition in campaign["comparison_contract"]["conditions"]:
    prior = json.loads((ROOT/f"nonlinear-{condition}-cpu-01/state.json").read_text())["study"]
    controls = copy.deepcopy(next(r["controls"] for r in prior["rows"] if r["proposal"] == "ledh"))
    settings = copy.deepcopy(prior["settings"])
    settings.update(device="GPU", dtype="float32", tf32=True, particles=128,
                    prepared_data_regime="scalar_sine_transition_quadratic_observation_six_parameter_v1")
    base = {"schema": "younis_score_study_v1", "phase": "2-nonlinear-scoped-pilot", "version": 1,
            "plan": PLAN, "seed": 92167, "evidence_class": "research_pilot", "settings": settings,
            "partitions": partitions, "heuristic_adversaries": heuristics}
    for proposal in providers:
        family = []
        if proposal == "ledh":
            for flow in (2, 8):
                for transport in (3, 30):
                    family.append({**controls, "flow_substeps": flow,
                                   "reset_sinkhorn_steps": transport, "reset_balance_steps": transport})
        else:
            for flow, transport in ((2, 3), (8, 30)):
                for provider_value in ((2, 3) if proposal == "sgqf" else (.25, .75)):
                    family.append({"controls": {**controls, "flow_substeps": flow,
                        "reset_sinkhorn_steps": transport, "reset_balance_steps": transport},
                        "sgqf_level" if proposal == "sgqf" else "within_fraction": provider_value})
        tune_rows = []
        for index, candidate in enumerate(family):
            extra = {"controls": candidate} if proposal == "ledh" else candidate
            for role in ("calibration", "validation"):
                for dataset in partitions[role]:
                    for replicate in (0, 1):
                        tune_rows.append(row(condition, proposal, dataset, replicate, role, f"{proposal}_c{index}", **extra))
        stem = f"{condition}-{proposal}"
        selection = str(DEST/f"{stem}-selection.json")
        tune = {**base, "required_proposals": [proposal], "tuning_candidate_family": family,
                "rows": tune_rows, "budget": {"wall_seconds": 600, "max_attempts": len(tune_rows), "max_attempts_per_row": 1}}
        tune_path = DEST/f"{stem}-calibration-study.json"
        write(tune_path, tune)
        campaign["calibration"].append({"study": str(tune_path), "output": str(DEST/f"{stem}-calibration"), "selection": selection})
        claim_rows = [row(condition, proposal, d, r, "claim", proposal, tuning_selection=selection)
                      for d in partitions["claim"] for r in (0, 1)]
        claim = {**base, "required_proposals": [proposal], "tuning_candidate_family": family,
                 "rows": claim_rows, "budget": {"wall_seconds": 600, "max_attempts": len(claim_rows), "max_attempts_per_row": 1}}
        claim_path = DEST/f"{stem}-claim-study.json"
        write(claim_path, claim)
        campaign["evaluation"].append({"study": str(claim_path), "output": str(DEST/f"{stem}-claim")})
    baseline_rows = [row(condition, method, d, r, "claim", method)
                     for d in partitions["claim"] for r in (0, 1) for method in heuristics]
    baseline = {**base, "required_proposals": heuristics, "rows": baseline_rows,
                "budget": {"wall_seconds": 600, "max_attempts": len(baseline_rows), "max_attempts_per_row": 1}}
    baseline_path = DEST/f"{condition}-heuristics-study.json"
    write(baseline_path, baseline)
    campaign["evaluation"].append({"study": str(baseline_path), "output": str(DEST/f"{condition}-heuristics")})
write(DEST/"campaign.json", campaign)
print(json.dumps({"campaign": str(DEST/"campaign.json"), "studies": len(campaign["calibration"])+len(campaign["evaluation"])}))
