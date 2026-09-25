# Observation-aware TT: active checkpoint — 2026-09-14

## Objective and authority

The governing [master program](observation-aware-tt-repair-complete-program-20260913.md)
has been refreshed on the owner's 2026-09-14 instruction. It now indexes the
original stages, later remedies, evidence and complete recorded budget.
[A03 reconciliation](observation-aware-tt-master-amendment-03-20260914.md) has
[independent review: AGREE](../reviews/observation-aware-tt-master-amendment-03-review-20260914.md).
The earlier suggested extra GPU-hour was not an approved phase and remains
withdrawn. The owner subsequently authorized **five hours for any continuation
work**, with an explanation of the problems first. This is recorded as H6 in
the master: a new 18,000-second total cap for design, review, implementation,
checks, experiments, repairs and closeout, without adding the old unused time.
The continuation starts after that explanation; phase 7 will record its start,
allocation and attempt limits. No continuation experiment has been launched.
The detailed protocol still needs the owner's requested review before execution;
the budget itself does not need to be requested again.

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
At terminal verification all five launch source hashes matched the checkout.
The terminal run recorded HEAD:
`5836f0344293f1c4af85abba23689ba83d34d9af`.

All six methods pass mean/log-evidence reference screens in d1 and d4. Pair
promotion remains rejected: observed regime errors exceed cheap heuristics in
ordinary/large d1 and large d4 observations. No statistical ranking follows
from four particle seeds on one sequence. The allowed solver repair did not
establish fitting convergence. The inherited 0.05 defensive mixture mass lacks
calibration evidence. The tested configuration failed promotion; the research
direction remains open. This is an SGQF/TT extension, not source-faithful Zhao–Cui.

[SGQF/fitting clarification](observation-aware-tt-sgqf-fit-clarification-20260914.md):
the observed first-transition pair fit improves substantially over original TT.
The recursive fitting problem is concentrated in d4 (median audit 0.143162,
maximum 1.307943 at t=18); d1 median is 0.032652. SGQF is both an independent
Gaussian proposal arm and the TT guide. Pair filtering errors are descriptively
lower in near-zero/ordinary observations and slightly higher in large ones.
The fitted joint density has not been compared with an SGQF-only joint on a
common error metric. Resolve that baseline and the correction's Gaussian limit
in phase 7; this is not a new execution or promotion rule.
Owner follow-up: a correlated SGQF joint already supplies training rows. The
TT product reference does not incorporate that joint's dependence. A poor TT
fit is not evidence of poor SGQF; compare the two explicitly. Changing to a
coupled chart would require an amendment to A03, not an ad hoc implementation.

## Frozen scope and consumed budget

d1/d4 T=20, N=512, reference particles 32768; degree/ranks 3, rows 1024, pair
sweeps 4, proximal steps 128; fitting/reference seeds 64100/74100, particle
seeds 2101–2104. Diagnostic-02 selection and fixture unchanged; repair reused
exposed observations and made no fresh-holdout claim. tftwogpu, float64,
RTX5080 GPU/XLA with verified growth and escalated permissions, threads 2/1.
Terminal retry: 48.464704 s. Conservative numerical allowance remains
1824.932937 s, but all three full-launch slots and the one solver repair are
consumed. Do not launch another experiment under this completed plan.

## Manuscript and next program phase

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

Exact next action: explain the checked problems to the owner, then prepare and
review the phase-7 numerical protocol under the new five-hour authorization:
separate fitting convergence from capacity, calibrate the 5% mass assumption,
freeze fresh partitions and independent-sequence uncertainty, and cost the
complete commands. Phases 8–10 are conditional calibration, independent-sequence
comparison and closeout; numerical launches await the detailed protocol review.
No further experiment remains under the completed campaign, and its unused
time does not reset attempt limits. The new budget does not change the scientific
sequence, promotion criteria or the historical A03 review.
Human manuscript feedback is pending. Incident investigation stays separate;
do not reload raw failed sessions.
