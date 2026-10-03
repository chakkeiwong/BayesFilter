"""Freeze M32 parity and M34 paired activation designs before execution."""
import argparse
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
from tests.inference_validation.test_reuse_multimodel import parity_design
from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec

rows = []
for target in ("rotated_gaussian", "dirichlet", "residual"):
    d = parity_design(target, "gpu")
    path = args.root / "m32-r1" / (target + "-design.json")
    with path.open("x") as out:
        json.dump(d.payload(), out, indent=2)
    rows.append({"phase": "M32", "path": str(path), "identity": d.identity})
rng = random.Random(2026092341)
theta = rng.gauss(0, 2)
data = [rng.gauss(theta, 1) for _ in range(6)]
for arm, control, severity in (("baseline", "baseline", None), ("noop", "noop", None),
                               ("quarter", "location_shift", .25), ("half", "location_shift", .5)):
    params = {"tau": 2., "sigma": 1.}
    if severity is not None:
        params["location_shift_posterior_sd"] = severity
    d = ValidationDesign(design_id="m34-normal-activation", engine="accuracy",
        scenario=ScenarioSpec("normal_conjugate", "ordinary", control, parameters=params),
        replications=1, draws=500, seed=2026092341, budget_seconds=700, device="gpu",
        purpose="paired normal-target activation; no power/coverage inference",
        numerical_provenance="M31/M34 development counts from M27; fixed shared data/streams test no-op parity",
        posterior_cap=10000, mcse_tolerance=.05,
        options={"data": data, "native_search": True, "posterior_members": "selected",
            "member_rule": "first_verified", "bootstrap_initialization_rounds": 20,
            "metric_evidence_policy": "finite_window", "metric_probe_num_results": 16,
            "preparation_bound_expansion_steps": 1, "preparation_max_restarts": 3,
            "preparation_preset": "standard"})
    path = args.root / "m34-r1" / (arm + "-design.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(d.payload(), out, indent=2)
    rows.append({"phase": "M34", "arm": arm, "path": str(path), "identity": d.identity})
with (args.root / "m31-r1/frozen-fit-inventory.json").open("x") as out:
    json.dump({"rows": rows, "data_seed": 2026092341,
        "m34_pairing": "shared simulated data and streams; four controls are NOT independent experiments",
        "m34_output_rule": "first_verified_id_last_retained_draw_chain0.v1"}, out, indent=2)
print(json.dumps(rows))
