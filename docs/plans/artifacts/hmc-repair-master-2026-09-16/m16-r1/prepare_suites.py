"""Resolve the reviewed M16 design without launching experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source-r1"))
from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import write_json

PLAN = "docs/plans/bayesfilter-hmc-repair-m16-design-2026-09-21.md"
COMMON = {"plan_file": PLAN}
MEANS = (.64, .65, .66, .74, .75, .76)


def design(name, engine, route, device, count, seconds, **kwargs):
    target = "normal_conjugate" if route == "reference" else "gaussian"
    return ValidationDesign(design_id=name, engine=engine, scenario=ScenarioSpec(target, route),
        device=device, replications=count, draws=128,
        budget_seconds=seconds, seed=seed_for(2026092196, name)[0],
        purpose="M16 bounded operating characteristics; all failures and uncertainty retained",
        numerical_provenance=PLAN, **kwargs)


def save(name, designs):
    name += "-r2"
    suite = {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": name,
             "profile": "master", "profiles": {"master": sorted({d.engine for d in designs})},
             "designs": [d.payload() for d in designs]}
    plan = plan_suite(suite)
    write_json(ROOT / (name + ".json"), suite)
    write_json(ROOT / (name + "-plan.json"), plan)
    print(name, plan["budget_by_device"])


def main():
    for phase, device, count, seconds in (("pilot", "gpu", 2, 100),
            ("fresh", "gpu", 32, 650), ("fresh", "cpu_reference", 64, 900)):
        rows = []
        for mean in MEANS:
            name = f"m16-boundary-{phase}-{device}-{int(mean*100)}"
            epsilon = (64 * (1-mean)**2 / (mean*(2-mean)))**(1/6)
            rows.append(design(name, "acceptance", "prepared", device, count, seconds,
                step_size=epsilon, leapfrog_steps=1,
                options={**COMMON, "evidence_rungs": [1, 2, 4], "reference_anchors": 8192,
                         "analytic_stationary_acceptance": mean}))
        save(f"boundary-{phase}-{device}", rows)
    rows = []
    for mean in MEANS:
        for persistence in (0., .9):
            name = f"m16-synthetic-{int(mean*100)}-{int(persistence*10)}"
            rows.append(design(name, "acceptance", "controller", "cpu_reference", 256, 400,
                options={**COMMON, "acceptance_mean": mean, "acceptance_concentration": 20.,
                         "acceptance_persistence": persistence, "evidence_rungs": [1, 2, 4]}))
    save("boundary-synthetic-cpu", rows)
    for phase, device, seconds, replications in (
            ("pilot", "cpu_reference", 200, 2), ("pilot", "gpu", 200, 2),
            ("fresh", "cpu_reference", 3000, 128), ("fresh", "gpu", 4000, 128)):
        name = f"m16-sequential-{device}" if phase == "fresh" else f"m16-sequential-pilot-{device}"
        child = design(name + "-kernel", "invariance", "frozen", device, 16384, seconds,
            rank_draws=7, step_size=.3, leapfrog_steps=5,
            options={**COMMON, "kernel_power": 32, "invariance_quantities": ["bounded_radius"],
                     "gaussian_energy_test": True, "sequential": {"max_looks": 3, "sample_multiplier": 2}})
        outer = design(name, "power", "frozen", device, replications, seconds,
            options={**COMMON, "calibration_design": child.payload(),
                     "power_controls": ["baseline", "noop", "wrong_energy"]})
        save(name, [outer])
    rows = []
    from dataclasses import replace
    for datasets in (32, 128):
        for ranks in (3, 15):
            for severity in (.25, .5):
                name = f"m16-sbc-power-{datasets}-{ranks}-{int(severity*100)}"
                row = design(name, "power", "reference", "cpu_reference", 256, 600,
                    rank_draws=ranks, options={**COMMON, "normal_data_count": 6,
                        "location_severity": severity, "include_radius": True, "include_ks": False})
                rows.append(replace(row, draws=datasets))
    save("sbc-power-cpu", rows)


if __name__ == "__main__":
    main()
