"""Resolve M15's reviewed datasets, allocations and independent shard groups."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
SOURCE = ROOT.parent / "m14-r1/source-schedule-r1"
sys.path.insert(0, str(SOURCE))
from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.references.analytic import simulate
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import write_json

PLAN = "docs/plans/bayesfilter-hmc-repair-m15-design-2026-09-21.md"
PARAMETERS = {"normal_conjugate": {"tau": 2., "sigma": 1., "n": 6},
              "beta_binomial": {"alpha": 2., "beta": 3., "n": 12},
              "lgssm_location": {"tau": 2., "sigma": .5, "state_variance": 1., "rho": .6, "n": 6}}
COUNTS = {"warmup_min_results": 2000, "warmup_check_window_results": 1000,
          "warmup_chunk_results": 500, "warmup_max_results": 10000,
          "retained_min_results": 1000, "retained_chunk_results": 500, "retained_max_results": 10000}


def common():
    return {"plan_file": PLAN, "native_search": True, "preparation_preset": "standard",
            "metric_evidence_policy": "finite_window", "bootstrap_initialization_rounds": 20,
            "metric_probe_num_results": 16, "preparation_max_restarts": 3,
            "preparation_bound_expansion_steps": 1, "member_rule": "first_verified",
            "posterior_members": "selected", "posterior_settings": COUNTS}


def save(name, rows, groups=()):
    suite = {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": name,
             "profile": "master", "profiles": {"master": sorted(set(r.engine for r in rows))},
             "designs": [r.payload() for r in rows], "aggregate_groups": list(groups)}
    plan = plan_suite(suite)
    write_json(ROOT / (name + ".json"), suite)
    write_json(ROOT / (name + "-plan.json"), plan)
    print(name, len(rows), plan["budget_by_device"])


def main():
    for phase, device, counts, seconds in (
        ("pilot", "cpu_reference", dict.fromkeys(PARAMETERS, 1), 900),
        ("fresh", "cpu_reference", {"normal_conjugate": 32, "beta_binomial": 32, "lgssm_location": 8}, 900),
        ("fresh", "gpu", {"normal_conjugate": 4, "beta_binomial": 6}, 1200),
    ):
        rows, groups = [], []
        for model, n in counts.items():
            ids = []
            for shard in range(n):
                name = f"m15-sbc-{phase}-{device}-{model}-{shard:03}"
                ids.append(name)
                rows.append(ValidationDesign(design_id=name, engine="sbc",
                    scenario=ScenarioSpec(model, "ordinary", parameters=PARAMETERS[model]),
                    replications=1, rank_draws=3, draws=500, posterior_cap=10000,
                    seed=2026092192, budget_seconds=seconds, device=device,
                    purpose="whole-fit SBC on fresh independent datasets; all missingness retained",
                    numerical_provenance=PLAN, options=common()))
            groups.append({"group_id": f"m15-sbc-{phase}-{device}-{model}",
                           "design_ids": ids, "replications": n})
        save(f"sbc-{phase}-{device}", rows, groups)
    for phase, device, starts, count, seconds in (
        ("pilot", "gpu", ("dispersed",), 1, 900),
        ("fresh", "cpu_reference", ("dispersed", "remote"), 4, 600),
    ):
        rows = []
        for i in range(count):
            for start in starts:
                for model, params in PARAMETERS.items():
                    name = f"m15-stopping-{phase}-{device}-{model}-{start}-{i:03}"
                    data_seed = seed_for(2026092192, name, "data")
                    _, data = simulate(model, data_seed, params)
                    options = {**common(), "data": data,
                        "data_provenance": {"seed": list(data_seed), "truth_passed_to_fit": False},
                        "fixed_comparator": {"warmup_results": 2000, "retained_results": 10000}}
                    rows.append(ValidationDesign(design_id=name, engine="stopping",
                        scenario=ScenarioSpec(model, "ordinary", start=start, parameters=params),
                        replications=1, draws=500, posterior_cap=10000, seed=2026092192,
                        mcse_tolerance={"normal_conjugate": .005, "beta_binomial": .001, "lgssm_location": .01}[model],
                        budget_seconds=seconds, device=device,
                        purpose="later precision looks and caps on new data, independent fixed comparator",
                        numerical_provenance=PLAN, options=options))
        save(f"stopping-{phase}-{device}", rows)


if __name__ == "__main__":
    main()
