#!/usr/bin/env python3
"""Value-vs-score discriminating test for correction_steps and pairwise_steps.

The knob search showed that correction_steps 4→0 and pairwise_steps 4→0 change
the *score* by only 9.6e-04 and 7.7e-04, well inside the measurement noise.
This test checks whether the *value* changes by a similar tiny amount (→ the
stages genuinely have little to correct in this well-conditioned fixture) or by
a much larger amount (→ the score's dependence on those stages is not fully
wired, which would matter far beyond this campaign since the acceptance ratio
rests on that seam).

Measures:
- exact value at correction_steps=4, pairwise_steps=4 (baseline)
- exact value at correction_steps=0, pairwise_steps=4
- exact value at correction_steps=4, pairwise_steps=0
- exact value at correction_steps=0, pairwise_steps=0

Reports absolute and relative deltas. Runtime ~2-3 minutes.
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import numpy as np
import tensorflow as tf

# Fail-closed memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    growth_verified = all(
        tf.config.experimental.get_memory_growth(gpu) for gpu in gpus
    )
    if not growth_verified:
        raise RuntimeError("Memory growth verification failed")
    print(f"GPU_MEMORY_POLICY: growth enabled on {len(gpus)} GPU(s)")

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)
from bayesfilter.inference.ledh_dual_parameter_target import DualParameterLEDHTarget


def measure_value(target_kwargs):
    """One exact value at the given knob settings."""
    # Use the canonical lgssm target (T=50, d=3, frozen observations)
    # but override particle count to N=24 to match the knob-search fixture
    base_target = make_canonical_neutra_target("lgssm", particle_count=24, substeps=8)

    # Wrap with the dual adapter at the given knob settings
    target = DualParameterLEDHTarget(
        model=base_target.fused_model,
        initial_states=base_target.initial_states,
        initial_covariances=base_target.initial_covariances,
        noises=base_target.noises,
        observations=base_target.observations,
        exact_params={"substeps": base_target.substeps, **target_kwargs},
        biased_params={"substeps": base_target.substeps, **target_kwargs},  # not used in value_only
    )
    theta = tf.constant([0.95, 0.35, 0.45, 0.35, 0.45], dtype=tf.float64)
    value = target.value_only(theta)
    return float(value.numpy())


def main():
    baseline_kwargs = {
        "correction_steps": 4,
        "pairwise_steps": 4,
        "reset_ridge": 1e-8,
        "correction_lm_damping": 1e-6,
        "reset_sinkhorn_steps": 8,
    }

    print("Value-vs-score discriminating test")
    print("Fixture: canonical lgssm, T=50, d=3, N=24")
    print("Theta: [0.95, 0.35, 0.45, 0.35, 0.45] (only first 3 components used)")
    print()

    v_baseline = measure_value(baseline_kwargs)
    print(f"Baseline (correction=4, pairwise=4): {v_baseline:.15e}")

    # correction_steps 4→0
    kwargs_c0 = {**baseline_kwargs, "correction_steps": 0}
    v_c0 = measure_value(kwargs_c0)
    delta_c0 = v_c0 - v_baseline
    rel_c0 = abs(delta_c0 / v_baseline) if v_baseline != 0 else float("inf")
    print(f"correction_steps=0:             {v_c0:.15e}  Δ={delta_c0:+.6e}  rel={rel_c0:.6e}")

    # pairwise_steps 4→0
    kwargs_p0 = {**baseline_kwargs, "pairwise_steps": 0}
    v_p0 = measure_value(kwargs_p0)
    delta_p0 = v_p0 - v_baseline
    rel_p0 = abs(delta_p0 / v_baseline) if v_baseline != 0 else float("inf")
    print(f"pairwise_steps=0:               {v_p0:.15e}  Δ={delta_p0:+.6e}  rel={rel_p0:.6e}")

    # both 4→0
    kwargs_both0 = {**baseline_kwargs, "correction_steps": 0, "pairwise_steps": 0}
    v_both0 = measure_value(kwargs_both0)
    delta_both0 = v_both0 - v_baseline
    rel_both0 = abs(delta_both0 / v_baseline) if v_baseline != 0 else float("inf")
    print(f"both=0:                         {v_both0:.15e}  Δ={delta_both0:+.6e}  rel={rel_both0:.6e}")

    print()
    print("Interpretation:")
    print("- If value deltas are also ~1e-3 or smaller, the fixture is well-conditioned")
    print("  and these stages have little to correct.")
    print("- If value deltas are orders of magnitude larger than the score deltas")
    print("  (~1e-4), the score's dependence on these stages is not fully wired.")
    print()

    result = {
        "baseline_value": v_baseline,
        "correction_steps_0": {"value": v_c0, "delta": delta_c0, "rel": rel_c0},
        "pairwise_steps_0": {"value": v_p0, "delta": delta_p0, "rel": rel_p0},
        "both_0": {"value": v_both0, "delta": delta_both0, "rel": rel_both0},
        "fixture": "canonical_lgssm_T50_d3_N24",
        "theta": [0.95, 0.35, 0.45, 0.35, 0.45],
    }

    out_path = "/tmp/ledh_value_knob_discrimination.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result: {out_path}")


if __name__ == "__main__":
    main()
