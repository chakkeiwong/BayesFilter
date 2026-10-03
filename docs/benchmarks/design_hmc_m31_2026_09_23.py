"""Independent diagnostic arithmetic for M31; never an HMC decision engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scipy import stats


def tail(n, k, probability):
    """Independent log-binomial sum used to check SciPy at decision boundaries."""
    return math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
        - math.lgamma(n - i + 1) + i * math.log(probability)
        + (n - i) * math.log1p(-probability)) for i in range(k, n + 1))


def lower_screen(n, floor, true_probability):
    k = int(stats.binom.ppf(.975, n, floor)) + 1
    return k, float(stats.binom.sf(k - 1, n, true_probability))


def upper_screen(n, ceiling, true_probability):
    k = int(stats.binom.ppf(.025, n, ceiling))
    if stats.binom.cdf(k, n, ceiling) > .025:
        k -= 1
    return k, float(stats.binom.cdf(k, n, true_probability))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.repo / 'docs/plans/artifacts/hmc-repair-master-2026-09-16'
    prices = {}
    source = {}
    for target in ('gaussian', 'beta-binomial'):
        path = base / 'm30-r1' / f'{target}-full-dynamic-gpu-r1/execution.json'
        row = json.loads(path.read_text())
        assert row['exit_code'] == 0 and row['launch_failure'] is None
        prices[target] = row['elapsed_seconds']
        source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = []
    for n in (64, 128, 256, 384):
        k, chance = lower_screen(n, .90, .95)
        assert abs(tail(n, k, .90) - stats.binom.sf(k - 1, n, .90)) < 1e-11
        assert tail(n, k, .90) <= .025 < tail(n, k - 1, .90)
        coverage.append({'fits': n, 'minimum_successes': k, 'assurance_at_095': chance,
                         'gpu_hours_by_target': {t: n * p / 3600 for t, p in prices.items()}})
    # Two predeclared tests (mean and squared residual) use Bonferroni alpha/2.
    # The mean-test power is a lower bound on rejection of that family under
    # the stated iid normal translation comparator, not measured HMC power.
    critical = float(stats.norm.ppf(1 - .05 / 2 / 2))
    shift_designs = []
    for delta in (.25, .5):
        n = next(n for n in range(1, 10001)
            if stats.norm.cdf(-critical - math.sqrt(n) * delta)
            + stats.norm.sf(critical - math.sqrt(n) * delta) >= .90)
        power = float(stats.norm.cdf(-critical - math.sqrt(n) * delta)
                      + stats.norm.sf(critical - math.sqrt(n) * delta))
        repeats = next(r for r in range(1, 5001) if lower_screen(r, .80, .90)[1] >= .95)
        successes, assurance = lower_screen(repeats, .80, .90)
        shift_designs.append({'delta_posterior_sd': delta, 'fits_per_experiment': n,
            'analytic_mean_test_power_lower_bound': power,
            'power_confirmation_experiments': repeats, 'minimum_detected': successes,
            'assurance_if_power_09': assurance, 'complete_fits_per_arm': n * repeats,
            'illustrative_gpu_hours_m30_gaussian_price': n * repeats * prices['gaussian'] / 3600})
    null_repeats = next(r for r in range(1, 5001) if upper_screen(r, .10, .05)[1] >= .95)
    paper_root = args.repo / '.localresources/papers/hmc_validation_standards_20260915'
    for name in ('talts-sbc.txt', 'sbc-test-quantities-2211.02383.txt',
                 'gandy-scott-mcmc-unit-2001.06465.txt', 'mcunit-expect-invariant.R',
                 'sbc-diagnostics.R'):
        path = paper_root / name
        source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'schema': 'bayesfilter.hmc_m31_design_arithmetic.v1',
        'observed_full_fit_gpu_seconds': prices, 'coverage': coverage,
        'normal_endpoint_family': {'alpha': .05, 'multiplicity': 2,
            'statistics': ['sqrt(n)*mean(z)', 'sum(z**2)'],
            'nulls': ['N(0,1)', 'chi_square(n)'],
            'analytic_shift_designs': shift_designs,
            'null_confirmation_experiments_at_095_assurance': null_repeats,
            'maximum_null_rejections': upper_screen(null_repeats, .10, .05)[0]},
        'nested_baseline': {'datasets': 128, 'fits_per_dataset': 3,
            'fits_per_experiment': 384,
            'illustrative_one_arm_gpu_hours': 384 * prices['gaussian'] / 3600},
        'source_sha256': source,
        'funding_decision': 'complete-fit coverage and repeated full-fit power are underfunded at measured costs; fund engineering and bounded activation only',
        'provenance': {'coverage_floor': 'M26 inherited .90, two-sided .95 CP',
            'power_and_null_screens': 'M22 inherited .80 lower / .10 upper',
            'planning_assurance': '.95 inherited coverage planning target, explicit hypothesis for power',
            'true_probabilities': '.95 coverage and .90 power are hypothetical, not estimated',
            'shift_sizes': 'M22 .25/.5 posterior SD',
            'fit_price_transfer': 'illustrative only; normal-conjugate requires target-specific pricing'},
        'nonclaims': ['no observed HMC power or stopping coverage', 'no runtime guarantee',
            'no default promotion', 'oracle independence does not prove stopped output law']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'prices': prices, 'shift_designs': shift_designs,
                      'null_repeats': null_repeats, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
