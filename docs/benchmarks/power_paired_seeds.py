#!/usr/bin/env python3
"""What effect size can 16 paired seeds actually detect?

Phase 3 can return "indistinguishable". That verdict has two very different
meanings — "the effect is absent" or "the design cannot see an effect this small"
— and they must not be conflated in the result note. This computes the detectable
effect as a function of the paired-difference standard deviation, so the
distinction can be made from the observed data rather than asserted.

The relevant noise is the SD of the PAIRED DIFFERENCES, not the per-seed L2 SD
(~0.17). Pairing cancels the seed-common component, so the paired SD is typically
much smaller. Its actual value is not known until Phase 3 runs, hence the sweep.

CPU-only, no model code, seconds to run.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

# Observed in the pilot: weakest-transport configs sat ~0.03 above the baseline
# mean L2 on the same four seeds. That is the order of effect Phase 3 must resolve.
TARGET_EFFECT = 0.03
N_SEEDS = 16
RESAMPLES = 2000
TRIALS = 400


def bootstrap_excludes_zero(
    differences: np.ndarray, generator: np.random.Generator
) -> bool:
    count = differences.size
    means = np.mean(
        differences[generator.integers(0, count, size=(RESAMPLES, count))], axis=1
    )
    low, high = np.percentile(means, [2.5, 97.5])
    return bool(low > 0.0 or high < 0.0)


def main() -> int:
    generator = np.random.default_rng(20260913)

    print("=" * 78)
    print("DETECTABLE EFFECT: 16 PAIRED SEEDS, PERCENTILE BOOTSTRAP 95% CI")
    print("=" * 78)
    print(f"target effect to resolve: {TARGET_EFFECT} (pilot-observed L2 shift)")
    print(f"paired seeds: {N_SEEDS}   bootstrap resamples: {RESAMPLES}   trials: {TRIALS}")
    print()
    print("Detection rate = fraction of simulated experiments whose CI excludes 0.")
    print("Read it as the probability Phase 3 calls a real effect of this size.")
    print()

    paired_sds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.17]

    print(f"{'paired SD':>10} | {'effect/SD':>9} | {'detection rate':>14} | reading")
    print("-" * 78)
    for paired_sd in paired_sds:
        detections = 0
        for _ in range(TRIALS):
            sample = generator.normal(TARGET_EFFECT, paired_sd, size=N_SEEDS)
            if bootstrap_excludes_zero(sample, generator):
                detections += 1
        rate = detections / TRIALS
        ratio = TARGET_EFFECT / paired_sd
        if rate >= 0.80:
            reading = "well powered"
        elif rate >= 0.50:
            reading = "marginal"
        else:
            reading = "UNDERPOWERED"
        print(f"{paired_sd:>10.3f} | {ratio:>9.2f} | {rate:>13.1%} | {reading}")

    print()
    print("=" * 78)
    print("FALSE POSITIVE CHECK (true effect exactly zero)")
    print("=" * 78)
    for paired_sd in (0.01, 0.05, 0.17):
        detections = 0
        for _ in range(TRIALS):
            sample = generator.normal(0.0, paired_sd, size=N_SEEDS)
            if bootstrap_excludes_zero(sample, generator):
                detections += 1
        rate = detections / TRIALS
        print(f"  paired SD {paired_sd:.3f}: CI excludes zero in {rate:.1%} of trials "
              f"(nominal 5%)")
    print()
    print("  A percentile bootstrap at n=16 is known to under-cover slightly, so a")
    print("  rate somewhat above 5% is expected and is a reason to treat a")
    print("  marginal interval as suggestive rather than decisive.")

    print()
    print("=" * 78)
    print("HOW TO USE THIS IN THE RESULT NOTE")
    print("=" * 78)
    print("Phase 3 reports the paired differences, so compute their SD and read the")
    print("row above. Then:")
    print()
    print("  - CI excludes zero            -> effect detected at this seed count.")
    print("  - CI contains zero AND the    -> genuinely indistinguishable; the design")
    print("    row says 'well powered'        could have seen an effect this size.")
    print("  - CI contains zero AND the    -> UNDERPOWERED. Report 'no detectable")
    print("    row says 'UNDERPOWERED'        difference at 16 seeds', never 'no")
    print("                                   difference'. State the seed count that")
    print("                                   would be needed.")
    print()
    print("Required seeds scale as (SD/effect)^2, so halving the resolvable effect")
    print("costs 4x the seeds. If the paired SD turns out near the per-seed SD 0.17,")
    print("resolving 0.03 needs on the order of hundreds of seeds, which is outside")
    print("this campaign's budget and must be stated as such rather than papered over.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
