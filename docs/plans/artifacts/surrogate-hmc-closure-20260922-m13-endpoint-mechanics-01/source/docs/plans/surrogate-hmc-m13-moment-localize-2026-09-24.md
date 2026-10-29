# M13 remaining correction failure localization — 2026-09-24

Existing master continuation after controls-11; surrogate-HMC remains the goal.
Question: which shared correction substep still fails after removing weighted
fourth-power overflow? Preserve already-computed target covariance, initial
standardization, diagonal-loop, pairwise-loop, affine-restoration and output
standardization validity flags through the existing batch diagnostic signature.
This is observability only: no numerical expression or validity rule changes.

Comparator: preserved controls-11 candidate, same seed1703/partition6, N256/T20,
nine nominal rows and controls. Reuse its GPU primitive parity result; do not
repeat it. Save full consumer diagnostics and each substep flag. Completion
means all flags arrive at the consumer. Source changes must be limited to the
shared higher-moment helper and its shared batched output signature; all input
and control hashes must match controls-07. Typed candidate failures trigger
repair, not a scientific rejection. Unflagged nonfinite output, unexpected
source/input drift, missing flags, wrong device or budget expiry veto execution.
No controls, precision, endpoint or HMC promotion follows from this diagnostic.

Assumptions: the weighted-moment change remains an unpromoted candidate.
Previously healthy and failed rows are separate diagnostic situations. The
preserved FP32 and FP64 results are arithmetic comparators, not sampler
baselines. The GPU/XLA/memory-growth default and analytical call chain persist.
Pre-mortem: a partial return dictionary could drop flags in an inactive or
zero-step branch. Update all shared signatures/branches and run focused tests.
Skeptical audit passes because only observability changes, data are exact,
classification is explicit, and the result identifies the next repair.

One bounded attempt: CPU review30 seconds/60 wall; GPU worker75 CPU seconds/
180 wall; total105 CPU, within111.242237 remaining controls CPU. Charge all
usage in the existing pool. Command:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-moment-localize --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-12
```

The master stores sources, command logs, accounting and full diagnostics in
controls-12. This is part of the existing repair roadmap, not a direction change.
