"""Resolve the September 18 funded validation designs; never launch workers."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prepare_hmc_repair_suites import COUNTS, PLAN, suite
from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.references.analytic import simulate
from bayesfilter.testing.inference_validation.storage import write_json


def build():
    common = {"plan_file": str(REPO/PLAN)}
    out = {}
    for phase, replications, seconds in (("pilot", 4, 1200.), ("confirmation", 32, 7200.)):
        powers = []
        for epsilon in (.3, .6):
            tag = str(epsilon).replace(".", "p")
            child = ValidationDesign(design_id="m8-invariance-"+tag, engine="invariance",
                scenario=ScenarioSpec("gaussian", "frozen", parameters={"scale":1.}),
                replications=8192, draws=64, rank_draws=7, step_size=epsilon, leapfrog_steps=5,
                seed=seed_for(2026091840, phase, "power", epsilon)[0], budget_seconds=seconds,
                purpose="fixed-look sensitivity to the same reversed-MH defect with independent Gaussian energy",
                numerical_provenance=PLAN, phase="confirmation" if phase=="confirmation" else "development",
                options={**common, "kernel_power":32, "invariance_quantities":["bounded_radius"],
                         "gaussian_energy_test":True})
            powers.append(replace(child, design_id=f"m8-power-{phase}-{tag}", engine="power",
                replications=replications, options={**common, "calibration_design":child.payload(),
                    "power_controls":["baseline", "noop", "wrong_energy"]}))
        out[f"m8-power-{phase}-gpu"] = suite(f"m8-power-{phase}-gpu", powers)
    for phase, count, seconds in (("pilot",4,1200.),("confirmation",32,4800.)):
        original = out[f"m8-power-{phase}-gpu"]["designs"][0]
        payload = dict(original)
        payload["design_id"] = f"m8-power-16k-{phase}-0p3"
        payload["replications"] = count
        payload["budget_seconds"] = seconds
        payload["seed"] = seed_for(2026091840, "power-anchor-expansion",phase)[0]
        payload["options"] = dict(payload["options"])
        child = dict(payload["options"]["calibration_design"])
        child["replications"] = 16384
        child["budget_seconds"] = seconds
        payload["options"]["calibration_design"] = child
        out[f"m8-power-16k-{phase}-gpu"] = suite(f"m8-power-16k-{phase}-gpu",
            [ValidationDesign.from_payload(payload)])
    for phase, count in (("pilot", 1), ("fresh", 3)):
        rows = []
        for target in ("funnel", "funnel_noncentered"):
            for index in range(count):
                rows.append(ValidationDesign(design_id=f"m8-{phase}-{target}-{index}", engine="accuracy",
                    scenario=ScenarioSpec(target, "ordinary", parameters={"scale":3.}),
                    replications=1, draws=500, posterior_cap=10000, seed=seed_for(2026091840, phase, "funnel", index)[0],
                    budget_seconds=1200., purpose="same-model centered/noncentered full ordinary fit; preserve failures",
                    numerical_provenance=PLAN, options={**common, "native_search":True,
                        "preparation_bound_expansion_steps":1, "member_rule":"first_verified",
                        "posterior_members":"selected", "posterior_settings":COUNTS}))
        out[f"m8-funnel-{phase}-gpu"] = suite(f"m8-funnel-{phase}-gpu", rows)
    for phase, count in (("pilot", 4), ("fresh", 32)):
        rows = []
        for epsilon in (.15, .3, .6):
            rows.append(ValidationDesign(
                design_id=f"m8-acceptance-{phase}-{str(epsilon).replace('.', 'p')}",
                engine="acceptance", scenario=ScenarioSpec("gaussian", "frozen", start="reference"),
                replications=count, draws=256, seed=seed_for(2026091840, phase, "acceptance", epsilon)[0],
                step_size=epsilon, leapfrog_steps=5, budget_seconds=1200.,
                purpose="actual stationary HMC acceptance calibration; screen is not posterior convergence",
                numerical_provenance=PLAN, device="cpu_reference",
                options={**common, "evidence_rungs":[1,2,4]}))
        out[f"m8-acceptance-{phase}-cpu"] = suite(f"m8-acceptance-{phase}-cpu", rows)
    for phase, count, seconds in (("pilot", 2, 240.), ("fresh", 32, 3000.)):
        rows = []
        for epsilon in (.9, 1.4, 1.5, 1.6, 2.):
            rows.append(ValidationDesign(
                design_id=f"m8-screen-{phase}-{str(epsilon).replace('.', 'p')}",
                engine="acceptance", scenario=ScenarioSpec("gaussian", "prepared"),
                replications=count, draws=256, seed=seed_for(2026091840, phase, "public-screen", epsilon)[0],
                step_size=epsilon, leapfrog_steps=5, budget_seconds=seconds,
                purpose="public fixed-start HMC repeated-look decisions with independent stationary acceptance reference",
                numerical_provenance=PLAN, options={**common, "evidence_rungs":[1,2,4], "reference_anchors":8192}))
        out[f"m8-screen-{phase}-gpu"] = suite(f"m8-screen-{phase}-gpu", rows)
    for phase, count in (("pilot", 1), ("fresh", 32)):
        rows = []
        for target in ("normal_conjugate", "beta_binomial"):
            values = {"tau":2., "sigma":1., "n":6} if target == "normal_conjugate" else {"alpha":2.,"beta":3.,"n":12}
            # A single predeclared dataset per target: conditional repeated-fit
            # stopping calibration, not prior-predictive SBC or varied-data coverage.
            data_seed = seed_for(2026091840, "stopping", target, "data")
            _, data = simulate(target, data_seed, values)
            for index in range(count):
                rows.append(ValidationDesign(design_id=f"m8-stop-{phase}-{target}-{index}", engine="stopping",
                    scenario=ScenarioSpec(target, "ordinary", parameters=values), replications=1,
                    draws=500, posterior_cap=10000, seed=seed_for(2026091840, phase, "stopping", target, index)[0],
                    budget_seconds=500. if phase == "pilot" else 1050., purpose="conditional complete-fit stopped/fixed error and interval calibration",
                    numerical_provenance=PLAN, options={**common, "native_search":True,
                        "preparation_bound_expansion_steps":1, "member_rule":"first_verified",
                        "posterior_members":"selected", "posterior_settings":COUNTS,
                        "data":data, "data_provenance":{"seed":list(data_seed),"truth_passed_to_fit":False},
                        "fixed_comparator":{"warmup_results":2000,"retained_results":10000}}))
        # The current aggregation helper is intentionally SBC-only. Keep
        # stopping rows as independent assessments and summarize them from the
        # per-design result records rather than mislabelling them SBC shards.
        out[f"m8-stopping-{phase}-gpu"] = suite(f"m8-stopping-{phase}-gpu", rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in build().items():
        write_json(args.output/(name+".json"), value)
        print(name, len(value["designs"]))


if __name__ == "__main__":
    main()
