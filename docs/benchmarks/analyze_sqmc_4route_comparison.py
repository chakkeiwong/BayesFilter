#!/usr/bin/env python3
"""Statistical analysis of SQMC 4-route comparison at N=1008."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_route_results(base_dir: Path, route_name: str, attempt_num: int) -> Dict:
    """Load results for one route."""
    result_file = base_dir / f"claim_attempt{attempt_num:02d}" / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing: {result_file}")

    data = json.load(open(result_file))

    # Extract N=1008 rows only
    rows = [r for r in data['rows'] if r['particle_count'] == 1008]

    if len(rows) != 16:
        raise ValueError(f"{route_name}: expected 16 seeds, got {len(rows)}")

    # Sort by seed for paired analysis
    rows.sort(key=lambda r: r['seed'])

    return {
        'route': route_name,
        'seeds': [r['seed'] for r in rows],
        'values': [r['value'] for r in rows],
        'finite': all(r['finite'] for r in rows),
        'valid': all(r.get('program_valid', True) for r in rows),
    }

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 5000, ci_level: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap mean and confidence interval."""
    n = len(data)
    bootstrap_means = np.zeros(n_bootstrap)

    rng = np.random.RandomState(42)  # Reproducible

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    mean = np.mean(data)
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean, ci_lower, ci_upper

def statistical_verdict(ci_lower: float, ci_upper: float) -> str:
    """Determine statistical verdict from CI."""
    if ci_lower > 0:
        return "SUPERIOR"
    elif ci_upper < 0:
        return "INFERIOR"
    else:
        return "INDISTINGUISHABLE"

def main():
    base_dir = Path("docs/benchmarks/artifacts/sqmc-rerun-corrected-filter-20260906")

    # Load all 4 routes
    routes_config = [
        ('repaired_permutation', 1),  # Already complete
        ('iid_dual_cap', 3),
        ('previous_inverse_cdf', 6),  # Restarted run
        ('repaired_fixed_previous_controls', 7),  # Final run
    ]

    print("="*80)
    print("SQMC 4-Route Statistical Comparison")
    print("="*80)
    print(f"Model: Austria SIR T=20")
    print(f"Particle count: N=1008")
    print(f"Seeds: 16 (97701-97716)")
    print(f"Bootstrap samples: 5000")
    print(f"CI level: 95%")
    print()

    # Load all routes
    results = {}
    for route_name, attempt in routes_config:
        try:
            result = load_route_results(base_dir, route_name, attempt)
            results[route_name] = result
            print(f"✓ Loaded {route_name}: {len(result['values'])} seeds")
        except FileNotFoundError as e:
            print(f"✗ {route_name}: {e}")
            return
        except ValueError as e:
            print(f"✗ {route_name}: {e}")
            return

    # Verify all routes have same seeds
    seed_sets = [set(r['seeds']) for r in results.values()]
    if len(set(frozenset(s) for s in seed_sets)) != 1:
        print("\n✗ ERROR: Routes have different seed sets")
        return

    common_seeds = results['repaired_permutation']['seeds']

    print(f"\n✓ All routes use same {len(common_seeds)} seeds")
    print()

    # Check validity
    print("="*80)
    print("VALIDITY CHECKS")
    print("="*80)
    all_valid = True
    for route_name, result in results.items():
        finite_status = "✓" if result['finite'] else "✗ NON-FINITE"
        valid_status = "✓" if result['valid'] else "✗ INVALID"
        print(f"{route_name:40s}: finite={finite_status}, valid={valid_status}")
        if not (result['finite'] and result['valid']):
            all_valid = False

    if not all_valid:
        print("\n✗ HARD VETO: Some routes have invalid results")
        return

    print("\n✓ All routes pass validity checks")
    print()

    # Descriptive statistics
    print("="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    print(f"{'Route':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-"*80)

    stats = {}
    for route_name, result in results.items():
        values = np.array(result['values'])
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val

        stats[route_name] = {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val,
        }

        print(f"{route_name:<40} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {range_val:>10.4f}")

    print()

    # Bootstrap analysis (no MC baseline - just absolute comparison)
    print("="*80)
    print("BOOTSTRAP ANALYSIS (5000 samples, 95% CI)")
    print("="*80)
    print(f"{'Route':<40} {'Mean':>10} {'CI Lower':>10} {'CI Upper':>10} {'Verdict':<20}")
    print("-"*80)

    bootstrap_results = {}
    for route_name, result in results.items():
        values = np.array(result['values'])
        mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=5000, ci_level=0.95)

        bootstrap_results[route_name] = {
            'mean': float(mean),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
        }

        print(f"{route_name:<40} {mean:>10.4f} {ci_lower:>10.4f} {ci_upper:>10.4f}")

    print()

    # Pairwise comparisons
    print("="*80)
    print("PAIRWISE COMPARISONS (Route A - Route B)")
    print("="*80)

    route_names = list(results.keys())
    print(f"{'Comparison':<50} {'Mean Diff':>12} {'CI Lower':>12} {'CI Upper':>12} {'Verdict':<20}")
    print("-"*80)

    pairwise_results = []
    for i, route_a in enumerate(route_names):
        for route_b in route_names[i+1:]:
            values_a = np.array(results[route_a]['values'])
            values_b = np.array(results[route_b]['values'])

            # Paired differences
            diffs = values_a - values_b

            mean_diff, ci_lower, ci_upper = bootstrap_ci(diffs, n_bootstrap=5000, ci_level=0.95)
            verdict = statistical_verdict(ci_lower, ci_upper)

            comparison_name = f"{route_a} - {route_b}"
            pairwise_results.append({
                'comparison': comparison_name,
                'mean_diff': float(mean_diff),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'verdict': verdict,
            })

            verdict_symbol = "✓" if verdict == "SUPERIOR" else ("✗" if verdict == "INFERIOR" else "≈")
            print(f"{comparison_name:<50} {mean_diff:>12.4f} {ci_lower:>12.4f} {ci_upper:>12.4f} {verdict_symbol} {verdict}")

    print()

    # Save JSON output
    output = {
        'schema': 'sqmc_4route_statistical_analysis_v1',
        'model': 'austria_sir_T20',
        'particle_count': 1008,
        'n_seeds': 16,
        'seeds': common_seeds,
        'routes': route_names,
        'descriptive_statistics': stats,
        'bootstrap_results': bootstrap_results,
        'pairwise_comparisons': pairwise_results,
        'bootstrap_config': {
            'n_bootstrap': 5000,
            'ci_level': 0.95,
            'random_seed': 42,
        },
    }

    output_file = base_dir / "sqmc_4route_statistical_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    # Find best route by mean
    best_route = max(stats.items(), key=lambda x: x[1]['mean'])
    worst_route = min(stats.items(), key=lambda x: x[1]['mean'])

    print(f"Best route (highest mean): {best_route[0]} ({best_route[1]['mean']:.4f})")
    print(f"Worst route (lowest mean): {worst_route[0]} ({worst_route[1]['mean']:.4f})")
    print(f"Range across routes: {best_route[1]['mean'] - worst_route[1]['mean']:.4f}")

    # Count statistically significant pairwise differences
    n_superior = sum(1 for p in pairwise_results if p['verdict'] == 'SUPERIOR')
    n_inferior = sum(1 for p in pairwise_results if p['verdict'] == 'INFERIOR')
    n_indist = sum(1 for p in pairwise_results if p['verdict'] == 'INDISTINGUISHABLE')

    print(f"\nPairwise comparisons:")
    print(f"  Superior: {n_superior}")
    print(f"  Inferior: {n_inferior}")
    print(f"  Indistinguishable: {n_indist}")
    print(f"  Total: {len(pairwise_results)}")

if __name__ == '__main__':
    main()
