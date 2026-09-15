# Observation-aware TT active checkpoint — 2026-09-15

## Research question and current authority

Can TT fitting preserve and improve the available SGQF approximation, and does
an empirical SGQF safeguard help actual filtering? Follow the
[master](observation-aware-tt-repair-complete-program-20260913.md), completed
amendments A04–A06 and H11's additional 24-hour authorization. Do not choose ad
hoc experiments. The owner requested the problems and results before continuation;
the [briefing](observation-aware-tt-24h-results-briefing-20260915.md) answers that.

Checkout `/home/chakwong/BayesFilter`, branch `surrogate-hmc`. Preserve unrelated
AGENTS.md, CLAUDE.md, c2_gaussian_hermite_proposal_tf.py and
ledh_canonical_score_tf.py changes. Current budget:
`artifacts/observation-tt-continuation-24h-20260915-01/budget.json`.
H11 adds 86400 seconds to the unused H6 balance of 816.529454 seconds; charge
future work only through the new ledger. It includes all active planning,
review, implementation, computation and reporting once; crash/user idle is
excluded. H6's completed ledger and pre-H11 master/checkpoint are preserved.
A06's numerical cap was met. Its separate active-work suballocation was not
timed apart from concurrent phase-10 drafting, so compliance is unverified.

Final H6 accounting at 2026-09-15 06:28:16 UTC: **17183.471 seconds charged; 816.529 seconds unused** (about 13 minutes), including a conservative 60-second final-response allowance. The ledger is paused; subsequent idle time is excluded.

H11 briefing accounting at 2026-09-15 07:13:52 UTC: 836.637 seconds charged; **86379.893 seconds remain** (about 23.99 hours), including a conservative 120-second response allowance. Ledger paused after the requested results briefing; resume the recorded amendment task without re-requesting the authorized budget.

## Checked findings and history

[A04 initialization result](observation-aware-tt-sgqf-initialization-20260915-result.md):
24 cases completed; SGQF initialization is descriptively favorable in difficult
d4 cases, but TT conversion/refinement loses to the analytical SGQF joint at
early d4 targets. Validation selection is no worse by construction; audit is
no worse in 23/24 cases. This does not guarantee population/filter performance.
The full-coefficient initializer is diagnostic and restricted to d<=4.

[A05 safety/consumer result](observation-aware-tt-defense-consumer-20260915-result.md):
complete, terminal review AGREE. The optional 1e-5 fraction passes the declared
N=512 healthy non-harm and small-mass rescue bounds. Inherited 5% fails healthy
non-harm. Exact SGQF joint conditional, retained marginal, actual mixed consumer
and correction weights pass focused tests. One 19.103-second GPU run, 96 stress
configurations and 24 exposed-target fresh-row cases. Some early TT losses to
SGQF remain; late d4 target-row ESS reaches only 11.94/8192. No filter promotion.

## Completed execution and next-action boundary

A06 full attempt-01 COMPLETE. All 24 references pass. One shared SGQF guide
fails at d4 sequence 8, time 18: all levels 2–5 give non-SPD signed covariance.
The unchanged CPU replay reproduces it. Six guide-dependent methods are
unavailable on that sequence; the two unguided methods complete. All stored
particle numbers on completed methods are finite, with zero recorded CDF
bracket failures. The paired d1 criterion fails: observed losses against the
transition proposal in ordinary observations and SGQF joint in large
observations; one interval also misses precision. No promotion. d4 descriptive
tables must use the same 11 successful sequences and cannot support population
ranking. Preserve the failed sequence and all 12 unguided outcomes.

Run: `docs/benchmarks/artifacts/observation_tt_independent_filtering_20260915/attempt-01/`.
Operational/report files: `artifacts/observation-tt-independent-filtering-20260915-01/`.
`assemble_report.py` distinguishes absent validity fields from explicit false
flags and checks all stored particle numeric values; it does not change run
results. `report.json` and `tables.md` now use matched sequence comparisons.

The [A06 result](observation-aware-tt-independent-filtering-20260915-result.md)
is complete and its bounded terminal interpretation review returned AGREE.
The [phase-10 closeout](observation-aware-tt-phase10-closeout-20260915.md)
completes the master program, result/inference tables and manuscript update.
All continuation phases 7–10 have executed; no experiment is running. Candidate
rejection does not reject the research direction. H11 now funds the next
continuation, whose numerical protocol has not yet been specified or reviewed.

Exact next action: write and review the next master amendment, using the results
briefing and H11's authorized ceiling. The amendment must distinguish
SGQF signed-quadrature robustness from TT conversion/selection generalization,
state its evidence contract and budget, and use fresh confirmation data.
No additional numerical protocol is active. Do not silently retune the A06
holdout or substitute a covariance repair after seeing its failure.

Phase-10 protected baseline and drafting record:
`artifacts/observation-tt-manuscript-20260915-01/`.
The [54-page PDF](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf)
includes the SGQF joint derivation, A04/A05 results and matched A06 comparisons.
Final build and rendered checks passed; all 187 prior math environments and
the appendix are preserved. Human reading feedback remains pending. No pair
gradients, HMC, default readiness, source-faithful TT-cross or scalable
initializer is claimed. Keep crash investigation separate.
