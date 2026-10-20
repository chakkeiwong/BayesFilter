#!/usr/bin/env python3
"""Check the paired-comparison statistics used by run_sqmc_tuned_vs_untuned.py.

The Phase 3 verdict (favours_tuned / favours_untuned / indistinguishable) is
produced by a paired bootstrap CI and an exact sign test.  If either is wrong the
GPU run produces a confidently mislabelled conclusion, so both are checked here
against cases with known answers before the run.

CPU-only, no model code, runs in seconds.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Import the statistics helpers without pulling in TensorFlow-heavy modules.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_phase3_src",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_sqmc_tuned_vs_untuned.py"),
)


def _load_helpers():
    """Extract the two pure-python helpers by exec'ing only their definitions.

    The module imports TensorFlow at top level, which we do not need and do not
    want to pay for here, so the source is parsed and only the two target
    function definitions are executed.
    """
    import ast

    source = open(_spec.origin).read()
    tree = ast.parse(source)
    wanted = {"_paired_bootstrap", "_sign_test"}
    namespace: dict = {"np": np, "Sequence": list, "Dict": dict, "Any": object}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, _spec.origin, "exec"), namespace)
    missing = wanted - set(namespace)
    if missing:
        raise SystemExit(f"could not load helpers: {missing}")
    return namespace["_paired_bootstrap"], namespace["_sign_test"]


paired_bootstrap, sign_test = _load_helpers()

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("=" * 78)
    print("PAIRED BOOTSTRAP")
    print("=" * 78)

    # A clear, consistent improvement: every pair negative, well away from zero.
    clear = [-0.10, -0.12, -0.09, -0.11, -0.13, -0.08, -0.10, -0.11]
    result = paired_bootstrap(clear, resamples=4000, seed=1)
    print(f"  clear improvement: mean={result['mean_difference']:+.5f} "
          f"CI=[{result['ci_low']:+.5f}, {result['ci_high']:+.5f}]")
    check(
        "clear improvement: CI entirely below zero",
        result["ci_high"] < 0.0,
        "so a lower-is-better metric is labelled favours_tuned",
    )
    check(
        "clear improvement: mean matches numpy",
        abs(result["mean_difference"] - float(np.mean(clear))) < 1e-12,
    )
    check(
        "CI brackets the mean",
        result["ci_low"] <= result["mean_difference"] <= result["ci_high"],
    )

    # Pure noise centred on zero: must NOT be called a difference.
    generator = np.random.default_rng(7)
    noise = list(generator.normal(0.0, 0.15, size=16))
    result = paired_bootstrap(noise, resamples=4000, seed=2)
    print(f"  centred noise:     mean={result['mean_difference']:+.5f} "
          f"CI=[{result['ci_low']:+.5f}, {result['ci_high']:+.5f}]")
    check(
        "centred noise: CI contains zero",
        result["ci_low"] < 0.0 < result["ci_high"],
        "so the verdict is indistinguishable",
    )

    # Degradation: every pair positive.
    worse = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08]
    result = paired_bootstrap(worse, resamples=4000, seed=3)
    print(f"  degradation:       mean={result['mean_difference']:+.5f} "
          f"CI=[{result['ci_low']:+.5f}, {result['ci_high']:+.5f}]")
    check(
        "degradation: CI entirely above zero",
        result["ci_low"] > 0.0,
        "so a lower-is-better metric is labelled favours_untuned",
    )

    # Determinism: the seed must make the interval reproducible, otherwise the
    # recorded artifact cannot be re-derived.
    a = paired_bootstrap(clear, resamples=2000, seed=99)
    b = paired_bootstrap(clear, resamples=2000, seed=99)
    check(
        "bootstrap is deterministic for a fixed seed",
        a == b,
        "artifact intervals are reproducible",
    )

    # A constant difference has zero variance; the CI must collapse onto it
    # rather than producing a degenerate or NaN interval.
    constant = [-0.05] * 10
    result = paired_bootstrap(constant, resamples=1000, seed=4)
    check(
        "constant difference: CI collapses to the value",
        abs(result["ci_low"] + 0.05) < 1e-12 and abs(result["ci_high"] + 0.05) < 1e-12,
    )

    print()
    print("=" * 78)
    print("EXACT SIGN TEST")
    print("=" * 78)

    # All eight pairs in the same direction: p = 2 * (1/2)^8 = 0.0078125.
    result = sign_test([-1.0] * 8)
    print(f"  8/8 negative: n={result['n_effective']} p={result['p_value']:.7f}")
    check(
        "8/8 one-sided gives p = 2*(1/2)^8",
        abs(result["p_value"] - 2 * (0.5**8)) < 1e-12,
        f"expected {2 * (0.5 ** 8):.7f}",
    )
    check("8/8: counts correct", result["n_negative"] == 8 and result["n_positive"] == 0)

    # An even split is maximally uninformative: p must be 1.
    result = sign_test([-1.0, -1.0, 1.0, 1.0])
    print(f"  2/2 split:    n={result['n_effective']} p={result['p_value']:.7f}")
    check("even split gives p = 1", abs(result["p_value"] - 1.0) < 1e-12)

    # Ties must be dropped, not counted as evidence either way.
    result = sign_test([-1.0, -1.0, -1.0, 0.0, 0.0])
    print(f"  3 neg + 2 ties: n={result['n_effective']} p={result['p_value']:.7f}")
    check("ties are excluded from n_effective", result["n_effective"] == 3)
    check(
        "3/3 after dropping ties gives p = 2*(1/2)^3",
        abs(result["p_value"] - 2 * (0.5**3)) < 1e-12,
    )

    # All ties: no evidence at all, and must not divide by zero.
    result = sign_test([0.0, 0.0, 0.0])
    print(f"  all ties:     n={result['n_effective']} p={result['p_value']:.7f}")
    check("all ties: n=0 and p=1, no crash", result["n_effective"] == 0 and result["p_value"] == 1.0)

    # p must never exceed 1 even in the near-symmetric case where doubling the
    # tail would otherwise overshoot.
    result = sign_test([-1.0, 1.0, 1.0])
    print(f"  1 neg + 2 pos: n={result['n_effective']} p={result['p_value']:.7f}")
    check("p is capped at 1", result["p_value"] <= 1.0)

    # Symmetry: swapping the sign of every difference must not change p.
    left = sign_test([-1.0, -1.0, -1.0, 1.0])
    right = sign_test([1.0, 1.0, 1.0, -1.0])
    check(
        "sign test is symmetric under negation",
        abs(left["p_value"] - right["p_value"]) < 1e-12,
    )

    print()
    print("=" * 78)
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        print("Phase 3 verdicts would be unreliable. Fix before running.")
        return 1
    print("All checks passed. The Phase 3 verdict logic is sound on known cases.")
    print()
    print("Not established: that the metrics themselves are the right ones, or")
    print("that a bootstrap over ~16 paired seeds has adequate power. These checks")
    print("only confirm the estimators do what they claim on known inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
