"""Materialize the predeclared control-safety studies, without numerical work."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "control-safety-config-01"
DEST.mkdir(exist_ok=False)
PLAN = str(ROOT.parents[2] / "younis-score-control-safety-calibration-2026-09-16.md")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


arms = {
    "baseline": {}, "coordinate_cap_4": {"coordinate_cap": 4.},
    "coordinate_cap_8": {"coordinate_cap": 8.},
    "damping_0": {"correction_lm_damping": 0.},
    "damping_001": {"correction_lm_damping": .001},
    "damping_1": {"correction_lm_damping": .1},
    "trust_1": {"correction_trust_radius": .1},
    "trust_2": {"correction_trust_radius": 2.},
    "ridge_1e7": {"reset_ridge": 1e-7}, "ridge_1e3": {"reset_ridge": 1e-3},
    "epsilon_05": {"reset_epsilon": .5}, "epsilon_4": {"reset_epsilon": 4.},
    "flow_16": {"flow_substeps": 16},
    "transport_60": {"reset_sinkhorn_steps": 60, "reset_balance_steps": 60}}
heuristics = ["bootstrap", "local_linear", "ekf", "ukf"]
bundle = {"schema": "control_safety_bundle_v1", "plan": PLAN,
          "max_numerical_rows": 120, "gpu_wall_seconds": 2700,
          "smoke": [], "screen": [], "arms": arms, "heuristics": heuristics}


def base(condition):
    prior = json.loads((ROOT / "nonlinear-calibration-01" /
                       f"{condition}-ledh-calibration-study.json").read_text())
    settings = copy.deepcopy(prior["settings"])
    controls = copy.deepcopy(prior["rows"][0]["controls"])
    controls.update(flow_substeps=8, reset_sinkhorn_steps=30, reset_balance_steps=30)
    return {"schema": "younis_score_study_v1", "phase": "2-control-safety-screen", "version": 1,
        "plan": PLAN, "seed": 92167, "evidence_class": "exploratory_control_sensitivity",
        "settings": settings, "partitions": {"calibration": [700, 701], "validation": [], "claim": []},
        "required_proposals": ["ledh"], "heuristic_adversaries": heuristics}, controls


def row(condition, proposal, arm, dataset, role="calibration", **extra):
    return {"id": f"{arm}_{dataset}", "model": "nonlinear_scalar", "proposal": proposal,
        "estimator": "nonlinear_analytical", "comparison_target": "model_score",
        "comparison": "approximation_error", "condition": condition, "dataset": dataset,
        "replicate": 0, "role": role, "coupling_group": "baseline", "control_arm": arm, **extra}


for condition in ("weak", "curved", "concentrated"):
    study, controls = base(condition)
    study["rows"] = [row(condition, "ledh", arm, dataset, controls={**controls, **change},
                         collect_control_diagnostics=True)
                     for arm, change in arms.items() for dataset in (700, 701)]
    study["rows"] += [row(condition, proposal, proposal, dataset)
                      for proposal in heuristics for dataset in (700, 701)]
    study["required_proposals"] = ["ledh", *heuristics]
    study["budget"] = {"wall_seconds": 800, "max_attempts": len(study["rows"]), "max_attempts_per_row": 1}
    path = DEST / f"{condition}-screen-study.json"
    write(path, study)
    bundle["screen"].append({"study": str(path), "name": condition})

smoke, controls = base("curved")
smoke.update(phase="0-control-diagnostics-wiring", evidence_class="mechanics")
smoke["rows"] = [row("curved", proposal, proposal + ("_diagnostics" if collect else "_ordinary"),
                      700, role="mechanics", controls=controls, collect_control_diagnostics=collect,
                      sgqf_level=2, within_fraction=.5)
                 for proposal in ("ledh", "sgqf", "kdm_covariance") for collect in (False, True)]
smoke["budget"] = {"wall_seconds": 450, "max_attempts": 6, "max_attempts_per_row": 1}
write(DEST / "nonlinear-smoke-study.json", smoke)
bundle["smoke"].append({"study": str(DEST / "nonlinear-smoke-study.json"), "name": "nonlinear"})

gaussian = copy.deepcopy(smoke)
gaussian["settings"].update(dimension=2, horizon=2, particles=16)
for name in ("transition_curve", "observation_curve", "reference"):
    gaussian["settings"].pop(name, None)
gaussian["settings"]["prepared_data_regime"] = "gaussian_all_parameters_six_parameter_v1"
gaussian["rows"] = [dict(row("gaussian_d2", "ledh", "ledh_" + label, 700, role="mechanics",
                             controls=controls, collect_control_diagnostics=collect),
                         model="gaussian_all_parameters", estimator="analytical_filter")
                    for label, collect in (("ordinary", False), ("diagnostics", True))]
gaussian["budget"] = {"wall_seconds": 450, "max_attempts": 2, "max_attempts_per_row": 1}
write(DEST / "gaussian-smoke-study.json", gaussian)
bundle["smoke"].append({"study": str(DEST / "gaussian-smoke-study.json"), "name": "gaussian"})
write(DEST / "bundle.json", bundle)
print(json.dumps({"bundle": str(DEST / "bundle.json"), "smoke_rows": 8, "screen_rows": 108}))
