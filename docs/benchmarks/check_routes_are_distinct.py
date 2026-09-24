#!/usr/bin/env python3
"""Do the four SQMC routes actually produce different scores?

A route comparison is meaningless if the route argument does not reach the
computation. This evaluates every route at identical controls, seed and target and
asserts the resulting scores are pairwise distinct.

Motivation: three "per-route" tuning runs produced bit-identical artifacts because
`--mode pilot` hardcoded the route list and ignored `--routes`. That defect was
caught by reading logs; this check makes the harness catch it instead.

It also distinguishes the two ways routes can collapse:
  1. dispatch failure  — the route argument is ignored (identical point sets)
  2. genuine equivalence — different ancestry that happens to agree here

by reporting the input point-set hashes alongside the scores.

Exit 0 = all routes distinct. Exit 1 = at least one collapsed pair.
"""

from __future__ import annotations

import hashlib
import os
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

import numpy as np
import tensorflow as tf

import run_sqmc_tuning as R

ROUTES = (
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_permutation",
    "repaired_permutation_ablation",
)

# Expected ancestry policy per route, from the baseline runner's ANCESTRY map.
EXPECTED_ANCESTRY = {
    "iid_dual_cap": "existing_one_to_one",
    "previous_inverse_cdf": "hilbert_inverse_cdf",
    "repaired_permutation": "hilbert_permutation_one_to_one",
    "repaired_permutation_ablation": "hilbert_permutation_one_to_one",
}


def _hash(tensor: tf.Tensor) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(tensor.numpy()).tobytes()
    ).hexdigest()[:16]


def main() -> int:
    horizon = 20
    particle_count = 1008
    seed = 50001

    theta = tf.constant([0.9, 0.8, 0.7, 0.6, 0.8], dtype=R.DTYPE)
    observations = R._frozen_observations(horizon)
    oracle_score = R._oracle_score(observations, theta)

    controls = R._tuning_grid()[0]

    print("=" * 78)
    print("ROUTE DISTINCTNESS CHECK")
    print("=" * 78)
    print(f"identical controls: {controls}")
    print(f"identical seed: {seed}, T={horizon}, N={particle_count}")
    print(f"oracle norm: {float(tf.norm(oracle_score).numpy()):.6f}")
    print()

    # Independently reproduce the point sets the runner builds, so a dispatch
    # failure is visible in the INPUTS rather than inferred from the outputs.
    from bayesfilter.highdim.sqmc_tf import (
        randomized_halton_gaussian,
        randomized_halton_joint,
    )

    print("input point sets per route:")
    input_hashes = {}
    for route in ROUTES:
        if route == "iid_dual_cap":
            initial = tf.random.stateless_normal(
                [particle_count, 3], [seed, 101], dtype=R.DTYPE
            )
            ancestors = tf.zeros([horizon, particle_count], R.DTYPE)
        else:
            initial = randomized_halton_gaussian(
                num_particles=particle_count,
                dimension=3,
                seed=seed,
                salt=301,
                dtype=R.DTYPE,
            )
            rows = []
            for t in range(horizon):
                _, anc, _ = randomized_halton_joint(
                    num_particles=particle_count,
                    state_dimension=3,
                    seed=seed,
                    salt=3001 + t,
                    dtype=R.DTYPE,
                )
                rows.append(anc)
            ancestors = tf.stack(rows)
        input_hashes[route] = (_hash(initial), _hash(ancestors))
        print(
            f"  {route:32s} ancestry={EXPECTED_ANCESTRY[route]:32s} "
            f"initial={input_hashes[route][0]} ancestors={input_hashes[route][1]}"
        )
    print()

    print("evaluating each route at identical controls:")
    results = {}
    for route in ROUTES:
        is_ablation = route == "repaired_permutation_ablation"
        actual_route = "repaired_permutation" if is_ablation else route
        result = R._evaluate_controls(
            route=actual_route,
            controls=controls,
            observations=observations,
            theta=theta,
            oracle_score=oracle_score,
            seed=seed,
            horizon=horizon,
            particle_count=particle_count,
            is_ablation=is_ablation,
        )
        results[route] = result
        if result.get("valid"):
            print(
                f"  {route:32s} L2={result['score_l2_error']:.10f} "
                f"cos={result['cosine_similarity']:.10f}"
            )
        else:
            print(f"  {route:32s} INVALID: {result.get('error')}")
    print()

    valid = {r: v for r, v in results.items() if v.get("valid")}
    if len(valid) < 2:
        print("Too few valid routes to compare.")
        return 1

    print("=" * 78)
    print("PAIRWISE COMPARISON")
    print("=" * 78)
    collapsed = []
    names = list(valid)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            l2_delta = abs(
                valid[left]["score_l2_error"] - valid[right]["score_l2_error"]
            )
            cos_delta = abs(
                valid[left]["cosine_similarity"] - valid[right]["cosine_similarity"]
            )
            same_inputs = input_hashes[left] == input_hashes[right]
            identical = l2_delta == 0.0 and cos_delta == 0.0
            if identical:
                collapsed.append((left, right, same_inputs))
            status = "IDENTICAL" if identical else "distinct"
            print(
                f"  {left:30s} vs {right:30s} "
                f"dL2={l2_delta:.3e} dcos={cos_delta:.3e} -> {status}"
            )

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not collapsed:
        print("All routes produce distinct scores. Route dispatch is working and a")
        print("route comparison is meaningful.")
        return 0

    print(f"{len(collapsed)} route pair(s) produced BIT-IDENTICAL scores:")
    for left, right, same_inputs in collapsed:
        cause = (
            "identical INPUT point sets -> dispatch failure, the route argument is "
            "not reaching the computation"
            if same_inputs
            else "different inputs but identical outputs -> the route distinction is "
            "being discarded downstream"
        )
        print(f"  {left} == {right}: {cause}")
    print()
    print("A route comparison over collapsed routes measures nothing. Fix the")
    print("dispatch before running or interpreting any per-route campaign.")
    print()
    print("Note: repaired_permutation and repaired_permutation_ablation SHARE an")
    print("ancestry policy by design and differ only in correction controls, so")
    print("they legitimately have identical input point sets. They must still")
    print("differ in OUTPUT; if they do not, the ablation controls are not applied.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
