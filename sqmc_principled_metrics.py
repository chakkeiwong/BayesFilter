#!/usr/bin/env python3
"""Display principled score quality metrics for SQMC oracle comparison."""

import json
import math

with open('docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/diagnostic_attempt02/result.json', 'r') as f:
    data = json.load(f)

print('SQMC vs Kalman Oracle: Principled Score Quality Metrics')
print('=' * 80)
print()
print('Model: 3D LGSSM, T=20, N=1008')
print('Oracle: Exact Kalman filter (innovation likelihood + GradientTape score)')
print()

# Map ancestry_policy to readable names
route_map = {
    'existing_one_to_one': 'iid_dual_cap',
    'hilbert_inverse_cdf': 'previous_inverse_cdf',
    'hilbert_permutation_one_to_one': 'repaired_permutation'
}

# Group by seed
for seed_num in [97701, 97702]:
    cells = [c for c in data['cells'] if c['seed'] == seed_num]
    print(f'Seed {seed_num}:')
    print()

    for cell in cells:
        policy = cell['ancestry_policy']
        role = cell['configuration_role']
        route_name = route_map.get(policy, policy)

        if role == 'control_family_ablation':
            route_name += ' (ablation)'

        print(f'  {route_name}:')

        # Value comparison
        value_error = cell['absolute_value_error']
        print(f'    Value error: {value_error:.4f}')

        # Score vector comparison
        oracle_score = cell['oracle_score']
        sqmc_score = cell['score']
        score_l2 = cell['score_l2_error']

        # Compute principled metrics
        oracle_norm = math.sqrt(sum(x**2 for x in oracle_score))
        sqmc_norm = math.sqrt(sum(x**2 for x in sqmc_score))
        dot_product = sum(o*s for o,s in zip(oracle_score, sqmc_score))
        cosine_sim = dot_product / (oracle_norm * sqmc_norm) if oracle_norm > 0 and sqmc_norm > 0 else 0

        rel_norm_error = abs(sqmc_norm - oracle_norm) / oracle_norm if oracle_norm > 0 else 0

        # Per-direction errors
        err_per_direction = [abs(s - o) for s, o in zip(sqmc_score, oracle_score)]

        print(f'    Score L2 error: {score_l2:.4f}')
        print(f'    Gradient direction (cosine similarity): {cosine_sim:.6f}')
        print(f'    Relative gradient norm error: {rel_norm_error:.4f}')
        print(f'      (|{sqmc_norm:.2f} - {oracle_norm:.2f}| / {oracle_norm:.2f})')

        print(f'    Per-parameter error vs Fisher scale:')
        for i, (err, oracle_val) in enumerate(zip(err_per_direction, oracle_score)):
            # Err / sqrt(|oracle_score|) as proxy for Err / sqrt(Fisher_info)
            err_over_sqrt = err / math.sqrt(abs(oracle_val)) if abs(oracle_val) > 0 else float('nan')
            print(f'      θ_{i}: Err={err:.4f}, |Oracle|={abs(oracle_val):.2f}, Err/√|Oracle|={err_over_sqrt:.4f}')

        print()

    print('-' * 80)
    print()

print()
print('Interpretation:')
print('  - Cosine similarity ≈ 1.0: gradient direction is correct')
print('  - Relative norm error: gradient magnitude accuracy')
print('  - Err/√|Oracle|: error scaled by √Fisher information (lower is better)')
print('  - For HMC with ε=0.01: induced parameter error ≈ ε × (Err / |Gradient|)')
print('    Typical: Err~1, |Gradient|~30 → parameter error ~0.0003 per step')
print()
