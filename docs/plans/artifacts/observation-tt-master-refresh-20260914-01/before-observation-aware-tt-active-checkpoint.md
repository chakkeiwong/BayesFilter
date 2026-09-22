# Observation-aware TT: active checkpoint — 2026-09-14

## Objective and authority

The authorized pair-block SGQF/TT remedy and attempt05 mathematical/results
writeup are complete. Worktree `/home/chakwong/BayesFilter`, branch
`surrogate-hmc`; preserve unrelated changes. Scientific scope:
[pair remedy](observation-tt-pair-block-remedy-20260914.md). Recovery and budget:
[sampler recovery](observation-tt-pair-sampler-recovery-20260914.md).

## Checked terminal result

[Result and decisions](observation-tt-pair-block-remedy-20260914-result.md).
Artifact root: `docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/`.
Replay-01 localized negative conditional mass to an incorrect Gram contraction:
`nald` reused a left bond index; `nbld` restores the squared-polynomial integral.
The guard and threshold are unchanged. An independent expanded-coefficient test
failed before correction; 13 focused tests pass afterward. GPU/XLA replay-02
completes all four d4 seeds with positive masses and max CDF residual 9.99e-16.
Campaign-02 completes all particle/reference arms, reusing identical saved fits;
its d4 pair results exactly match replay-02. d1 was recomputed and exactly
matches the original one-core results, which were unaffected by the index bug.
All five terminal launch source hashes match the checkout. Recorded HEAD:
`5836f0344293f1c4af85abba23689ba83d34d9af`.

All six methods pass mean/log-evidence reference screens in d1 and d4. Pair
promotion remains rejected: observed regime errors exceed cheap heuristics in
ordinary/large d1 and large d4 observations. No statistical ranking follows
from four particle seeds on one sequence. The allowed solver repair did not
establish fitting convergence. The inherited 0.05 defensive mixture mass lacks
calibration evidence. The tested configuration failed promotion; the research
direction remains open. This is an SGQF/TT extension, not source-faithful Zhao–Cui.

## Frozen scope and consumed budget

d1/d4 T=20, N=512, reference particles 32768; degree/ranks 3, rows 1024, pair
sweeps 4, proximal steps 128; fitting/reference seeds 64100/74100, particle
seeds 2101–2104. Diagnostic-02 selection and fixture unchanged; repair reused
exposed observations and made no fresh-holdout claim. tftwogpu, float64,
RTX5080 GPU/XLA with verified growth and escalated permissions, threads 2/1.
Terminal retry: 48.464704 s. Conservative numerical allowance remains
1824.932937 s, but all three full-launch slots and the one solver repair are
consumed. Do not launch another experiment under this completed plan.

## Manuscript and exact next action

[Completed 48-page PDF](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf).
The abstract and Section 15 include the correction, derivation, complete results
and uncertainty. Both protected source hashes match; no mathematics/citations
were removed, and one objective was reformatted with identical algebra.
Final pdflatex builds pass; rendered overview and detailed pair/result inspection
are recorded in
[render review](artifacts/observation-tt-pair-block-20260914/rendered-review/review.md).
Eight theorem checks remain inconclusive; nine scoped identities, one weight
identity and one Gram identity were checked. No whole-document proof, pair
gradient, HMC, default-readiness or human prose-acceptance claim is made.

No further execution remains in the authorized campaign. Hand over the result
and PDF for reading. A future numerical campaign needs fresh calibration,
sequence replication and a new bounded plan; it must not select on these
exposed observations. Human manuscript feedback remains pending and may guide
subsequent revisions. Incident investigation stays separate; do not reload
raw failed sessions.
