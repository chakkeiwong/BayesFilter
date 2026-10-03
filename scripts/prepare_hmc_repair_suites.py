"""Resolve the master repair's diagnostic development suites; launch no workers."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import write_json

PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"
COUNTS = {"warmup_chunk_results":500, "warmup_min_results":2000,
          "warmup_check_window_results":1000,"warmup_max_results":10000,
          "retained_chunk_results":500,"retained_min_results":1000,"retained_max_results":10000}


def suite(name, rows, groups=()):
    value = {"schema":"bayesfilter.inference_validation_suite.v1", "suite_id":name,
        "profile":"master", "profiles":{"master":sorted({r.engine for r in rows})},
        "designs":[r.payload() for r in rows], "aggregate_groups":list(groups)}
    plan_suite(value)
    return value


def build():
    base = ValidationDesign(design_id="master", engine="sbc",
        scenario=ScenarioSpec("normal_conjugate","ordinary"), replications=2, rank_draws=2,
        draws=500, seed=2026091620, budget_seconds=900.,
        purpose="fresh ordinary expansion development and cost; no default promotion",
        numerical_provenance=PLAN, device="cpu_reference", measurement_draws=128,
        posterior_cap=10000, options={"plan_file":str((REPO/PLAN).resolve()), "native_search":True,
            "preparation_bound_expansion_steps":1,"member_rule":"first_verified",
            "posterior_members":"selected","posterior_settings":COUNTS})
    output = {}
    for device, tag, seconds in (("cpu_reference","cpu",900.),("gpu","gpu",1500.)):
        output[f"pilot-{tag}"] = suite(f"master-pilot-{tag}",
            [replace(base,design_id=f"master-pilot-{tag}",device=device,budget_seconds=seconds)])
        rows = [replace(base,design_id=f"master-fresh-{tag}-{i}",device=device,
                        seed=2026091621,budget_seconds=800. if tag=="cpu" else 1500.) for i in range(3)]
        output[f"fresh-{tag}"] = suite(f"master-fresh-{tag}",rows,
            [{"group_id":f"master-fresh-{tag}","design_ids":[r.design_id for r in rows],"replications":6}])
    calibration = []
    for mean in (.54,.64,.7,.76,.86):
        for persistence in (0.,.9):
            calibration.append(replace(base,design_id=f"acceptance-{str(mean).replace('.','p')}-{str(persistence).replace('.','p')}",
                engine="acceptance",scenario=ScenarioSpec("gaussian","controller"),replications=64,
                budget_seconds=100.,measurement_draws=64,
                purpose="actual screen operating characteristics under stationary known-mean marks",
                options={"plan_file":str((REPO/PLAN).resolve()),"acceptance_mean":mean,
                    "acceptance_concentration":20.,"acceptance_persistence":persistence,"evidence_rungs":[1,2,4]}))
    output["acceptance-cpu"] = suite("master-acceptance-cpu",calibration)
    modes = []
    for start in ("single_mode","mode_dispersed"):
        modes.append(replace(base,design_id="master-mode-"+start.replace("_","-"),engine="stopping",
            scenario=ScenarioSpec("mixture","prepared",start=start),device="gpu",replications=2,
            budget_seconds=1800.,seed=2026091622,step_size=.6,
            purpose="mode-aware posterior assessment under two predeclared start regimes",
            options={"plan_file":str((REPO/PLAN).resolve()),"global_quantities":["left_mode_probability"],
                     "member_rule":"first_verified","posterior_members":"selected","posterior_settings":COUNTS}))
    output["modes-gpu"] = suite("master-modes-gpu",modes)
    return output


def build_m7():
    """M7 cost/power designs, with independent reference-generated fixture data.

    Only observed data reach a fit. Generating parameters and reference draws
    never initialize or tune it. This is development validation, not SBC.
    """
    from bayesfilter.testing.inference_validation.references.analytic import simulate
    root_seed = 2026091730
    common = {"plan_file": str((REPO/PLAN).resolve())}
    powers, mechanics = [], []
    for epsilon in (.3, .6):
        tag = str(epsilon).replace(".", "p")
        for power in (1, 8, 32):
            child = ValidationDesign(design_id=f"m7-kernel-{tag}-{power}", engine="invariance",
                scenario=ScenarioSpec("gaussian", "frozen", parameters={"scale": 1.}),
                replications=128, draws=64, rank_draws=7, step_size=epsilon, leapfrog_steps=5,
                seed=seed_for(root_seed, "power", epsilon, power)[0], budget_seconds=760.,
                purpose="fixed-look K^s sensitivity at the previously weak reversed-MH settings",
                numerical_provenance=PLAN, options={**common, "kernel_power": power})
            powers.append(replace(child, design_id=f"m7-power-{tag}-{power}", engine="power",
                replications=8, options={**common, "calibration_design": child.payload(),
                    "power_controls": ["baseline", "noop", "wrong_energy"]}))
        for control in ("baseline", "noop", "wrong_energy"):
            mechanics.append(replace(child, design_id=f"m7-energy-{tag}-{control}",
                engine="mechanics", scenario=replace(child.scenario, control=control),
                replications=16, seed=seed_for(root_seed, "energy", epsilon)[0], budget_seconds=40.,
                purpose="independent endpoint Hamiltonian check of the actual TFP MH ratio",
                options=common))
    pilots = []
    parameters = {
        "beta_binomial": {"alpha": 2., "beta": 3., "n": 12},
        "lgssm_location": {"tau": 2., "rho": .6, "state_variance": 1., "sigma": .5, "n": 6},
        "funnel": {"scale": 3.},
        "rotated_gaussian": {"angle": .6, "condition": 9.},
    }
    for target, values in parameters.items():
        options = {**common, "native_search": True, "preparation_preset": "standard",
            "preparation_bound_expansion_steps": 1, "member_rule": "first_verified",
            "posterior_members": "selected", "posterior_settings": COUNTS,
            "profile_execution": True}
        if target in {"beta_binomial", "lgssm_location"}:
            data_seed = seed_for(root_seed, "pilot", target, "data")
            _, data = simulate(target, data_seed, values)
            options.update(data=data, data_provenance={"seed": list(data_seed),
                "generator": "independent diagnostic references.analytic.simulate",
                "truth_passed_to_fit": False})
        pilots.append(ValidationDesign(design_id=f"m7-pilot-{target}", engine="accuracy",
            scenario=ScenarioSpec(target, "ordinary", parameters=values), replications=1, draws=500,
            seed=seed_for(root_seed, "pilot", target, "fit")[0], budget_seconds=1200.,
            purpose="complete automatic pipeline cost/availability pilot; no calibration or default promotion",
            numerical_provenance=PLAN, posterior_cap=10000, measurement_draws=128, options=options))
    return {"m7-energy-gpu": suite("m7-energy-gpu", mechanics),
            "m7-power-gpu": suite("m7-power-gpu", powers),
            "m7-pilots-gpu": suite("m7-pilots-gpu", pilots)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage", choices=("initial", "m7"), default="initial")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name, value in (build_m7() if args.stage == "m7" else build()).items():
        write_json(args.output/(name+".json"),value)
        print(name,plan_suite(value)["budget_by_device"])


if __name__ == "__main__":
    main()
