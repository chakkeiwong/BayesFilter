"""Gandy--Scott (2021) Algorithm 3 for independent diagnostic experiments.

Theorem 3.1 is conditional on independent looks and superuniform component
p-values. A callback's compliance is not proved by this decision routine.
Source cross-check: CRAN mcunit R/expect_mc.R, commit 9e3fd08e41c1d4cd19664e95d2baf646286a2b88.
"""
from __future__ import annotations

import math
from numbers import Real


def sequential_test(experiment, *, alpha, max_looks, initial_samples, sample_multiplier, dimension):
    """Call experiment(sample_count, look_index) with fresh randomness each time."""
    for name, value in (("max_looks", max_looks), ("initial_samples", initial_samples), ("dimension", dimension)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly inside (0,1)")
    if (isinstance(sample_multiplier, bool) or not isinstance(sample_multiplier, Real)
            or not math.isfinite(sample_multiplier) or sample_multiplier < 1):
        raise ValueError("sample_multiplier must be finite and at least one")
    enlarged = initial_samples * sample_multiplier
    if not math.isfinite(enlarged):
        raise ValueError("increased sample count is not representable")
    later_samples = math.ceil(enlarged)
    beta = alpha / max_looks
    if beta <= 0:
        raise ValueError("sequential rejection threshold is not representable")
    gamma = beta ** (1. / max_looks)
    looks = []
    decision = "look_cap_without_rejection"
    for index in range(max_looks):
        count = initial_samples if index == 0 else later_samples
        values = tuple(experiment(count, index))
        if (len(values) != dimension or any(isinstance(p, bool) or not isinstance(p, Real)
                or not math.isfinite(p) or not 0 <= p <= 1 for p in values)):
            raise ValueError("one finite p-value in [0,1] per declared component required")
        q = dimension * min(values)
        row = {"look_index": index, "sample_count": count, "p_values": list(values),
               "beta": beta, "gamma": gamma, "bonferroni_min_p": q}
        if q <= beta:
            decision = "reject"
        elif q > gamma + beta:
            decision = "early_nonrejection"
        else:
            decision = "continue" if index + 1 < max_looks else "look_cap_without_rejection"
        row["decision"] = decision
        looks.append(row)
        if decision != "continue":
            break
        beta /= gamma
    return {"finding": "discrepancy_detected" if decision == "reject" else "no_discrepancy_detected",
            "decision": decision, "looks": looks, "alpha": alpha, "dimension": dimension,
            "max_looks": max_looks, "sample_multiplier": sample_multiplier,
            "type_i_bound": alpha,
            "bound_assumptions": "independent look vectors; each component superuniform under the null; fixed dimension",
            "arbitrary_callback_validity_established": False, "accuracy_established": False,
            "mixing_established": False, "method": "Gandy--Scott Algorithm 3 / Theorem 3.1"}


def run_invariance(design, root, deadline, single_look):
    """Use disjoint complete frozen-kernel experiments, never running p-values."""
    from dataclasses import replace
    from pathlib import Path
    import time
    from ..designs import seed_for
    from ..storage import write_json

    root = Path(root)
    settings = design.options["sequential"]
    dimension = max(design.multiplicity, 2 * len(design.options["invariance_quantities"])
                    + int(design.options.get("gaussian_energy_test", False)))
    observations = []
    def experiment(count, index):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("sequential invariance deadline exhausted")
        options = {key: value for key, value in design.options.items() if key != "sequential"}
        child = replace(design, design_id=f"{design.design_id}-look-{index}", replications=count,
            seed=seed_for(design.seed, design.design_id, "independent-look", index)[0], options=options)
        directory = root / f"look-{index:03}"
        directory.mkdir(parents=True, exist_ok=True)
        result = single_look(child, directory, deadline)
        tests = {f"{family}/{name}": value for family in ("rank_tests", "two_sample_tests", "analytic_tests")
                 for name, value in result.get(family, {}).items()}
        expected = 2 * len(design.options["invariance_quantities"]) + int(design.options.get("gaussian_energy_test", False))
        if len(tests) != expected:
            raise ValueError("sequential invariance test family changed")
        # A declared larger family reserves unused components at p=1. No
        # fixed-look threshold/finding is reused as a sequential decision.
        values = [test["p_value"] for test in tests.values()] + [1.] * (dimension - expected)
        observations.append({"look_index": index, "design": child.payload(),
            "raw_tests": tests, "reserved_family_components": dimension - expected,
            "result_path": str(directory / "invariance.json")})
        write_json(root / "sequential-progress.json", {"observations": observations})
        return values

    result = sequential_test(experiment, alpha=design.alpha, max_looks=settings["max_looks"],
        initial_samples=design.replications, sample_multiplier=settings["sample_multiplier"], dimension=dimension)
    result.update(observations=observations, kernel_control=design.scenario.control,
        independent_unit="fresh independent frozen-kernel experiment per look",
        reused_cumulative_samples=False, ranking_supported=False)
    write_json(root / "invariance.json", result)
    return result
