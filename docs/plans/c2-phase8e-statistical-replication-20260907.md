# C2 Phase 8E Statistical Replication Plan

Date: 2026-09-07  
Parent: `docs/plans/c2-phase8-recovery-20260907.md`  
Status: `READY_TO_LAUNCH_AFTER_INTEGRATION_AND_BOUNDARY_FREEZE`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Purpose and research question

Phase 8D established that the fixed Newton schedule
`quarter_long_12` repairs the observed stationarity failure on four C2 paths.
Those paths each used one proposal-randomization branch, so their ESS table is
descriptive.  Phase 8E asks:

> Across independent observation fixtures and repeated proposal branches, does
> the exact-likelihood K=1 Laplace proposal with `quarter_long_12` remain a
> valid finite program and show a statistically supported active-time ESS
> contrast against the declared cheap heuristics?

This is a replication and uncertainty exercise.  It is not a posterior-
correctness proof, a general-model theorem, or a default-readiness run.

## Research-intent ledger

| Field | Predeclared entry |
| --- | --- |
| Mechanism under test | exact observation score/curvature, fixed per-ancestor Laplace bank, complete-mixture DMIS denominator, frozen analytical score |
| Candidate control | `quarter_long_12`: temperatures `(0.25,0.5,0.75,1,1,1,1,1,1,1,1,1)`, unit steps |
| Expected failure | invalid Newton rows, active-time heuristic loss, seed-sensitive ESS, or branch-to-branch variance |
| Primary validity criterion | every accepted candidate and comparator branch has finite exact value/score, complete denominator, APF identity, score parity, and declared GPU/memory provenance |
| Promotion veto | any candidate loss at an active transition in any declared salient situation; any candidate or comparator validity failure prevents efficiency interpretation |
| Statistical nomination criterion | for every comparator, the predeclared fixture-cluster interval for the primary log-ESS contrast is strictly above zero, after all validity and heuristic screens pass |
| Continuation veto | target/measure mismatch, missing or corrupted fixture/record, invalid comparator, unresolvable integration failure, or exhausted Phase 8E budget |
| Repair trigger | candidate-only fixed-schedule failure, renderer/aggregation defect, or bounded infrastructure failure with unchanged target, data contract, and budget |
| Nonclaims | no posterior accuracy, unbiased likelihood, global Newton convergence, general-model transfer, HMC, production, default, or statistical superiority claim from this phase alone |

## Integration prerequisite

No Phase 8E run may use the dirty canonical worktree.  First complete the
file-by-file integration described in
`docs/memos/c2-phase8-recovery-integration-handoff-20260907.md` on a clean
branch.  The integrated branch must import `HermiteBasis1D` and
`c2_exact_likelihood_laplace_adapter`, use the base-mass-aware evaluator, and
pass the focused contracts before any fixture is generated.  Record the
resulting commit and all source hashes in every Phase 8E manifest.

The conditional heuristic gate uses the committed boundary artifact
`docs/benchmarks/fixtures/c2_phase8e_observation_bin_boundaries_v1.json`.
Its three calibration fixtures use state seed `20260912` and disjoint
observation seeds `424290`--`424292`.  The analyzer verifies each listed
fixture hash, seed/order, dimension, and horizon, then recomputes the declared
linear one-third and two-thirds cut points before accepting a Phase 8E claim
fixture.  These calibration fixtures are excluded from the Phase 8E effect
estimate and must remain frozen after claim execution begins.

## Candidate and comparator ladder

All families use the same C2 observations, parameter value, particle count,
time horizon, exact target, and complete conditional proposal denominator.
The active five-family ladder is:

1. exact-likelihood Laplace K=1 candidate with `quarter_long_12`;
2. per-ancestor UKF/APF K=1;
3. bootstrap conditional;
4. transformed Student proposal with `nu=8`; and
5. stationary Gaussian independence.

The stale Gaussian-hint snapshot is excluded unless a separately hashed,
complete snapshot bank is restored and reviewed.  Retained-TT and Student-TT
routes remain separate negative/reference work; they are not silently mixed
into this estimand.

## Replicate design

The primary arm uses `N=4096`, horizon `T=20`, and two proposal branches per
fixture.  Generate twelve fresh observation fixtures with the fixed state seed
`20260912` and observation seeds:

```
424300, 424301, 424302, 424303, 424304, 424305,
424306, 424307, 424308, 424309, 424310, 424311
```

These seeds are disjoint from the Phase 8B--8D fixtures.  The runner's branch
seed formula is deterministic and is recorded in each branch record; branch
zero and branch one are not treated as independent observation fixtures.

If the primary arm has no continuation veto and budget remains, run a smaller
confirmatory arm at `N=8192` on six untouched fixtures with observation seeds
`424312` through `424317`, again with two branches.  A candidate promotion
veto does not become a continuation veto; it is recorded and the optional arm
may still test whether the failure persists.  This arm is a robustness check,
not a license to pool different particle counts into one effect size.

Each fixture and run gets a new, non-overwriting output directory.  The exact
requested schedule is frozen with
`--fixed-schedule-config quarter_long_12`; calibration may check schedule
validity but may not select a schedule from claim-run ESS.

## Estimands and uncertainty calculation

For candidate `c`, comparator `h`, fixture `s`, branch `b`, and active time
`t >= 1`, retain

\[
D_{sbt}^{(h)} = \log(\operatorname{ESS}_{sbt}^{(c)}+\epsilon)
              - \log(\operatorname{ESS}_{sbt}^{(h)}+\epsilon),
\qquad \epsilon=10^{-12}.
\]

The primary summary for one fixture and branch is the median of
`D_{sbt}^{(h)}` over the nineteen active transition times.  For each fixture,
average the two branch summaries.  The uncertainty unit is therefore the
independent observation fixture, not an individual time point or particle.
The statistical-reporting helper must:

- compute the paired fixture summaries from the raw per-time tensors;
- form a paired percentile 95% bootstrap interval by resampling fixture IDs
  (at least 10,000 resamples, with a recorded seed);
- report the paired sign count and exact two-sided sign-test p-value as a
  small-sample cross-check; and
- retain the full time-by-time table so a positive aggregate cannot hide a
  salient loss.

With twelve fixtures, intervals and sign tests are still finite-sample
evidence, not a universal guarantee.  If the interval includes zero, the
contrast is descriptive and the candidate is not statistically nominated.
The branch streams are paired by the runner's deterministic seed convention,
but the proposal transformations do not produce fully common random numbers;
the analysis therefore reports the pairing limitation explicitly.  No ranking
among candidates is reported from ESS alone.

## Heuristic-dominance situations

The heuristic gate is evaluated conditionally, not only on an overall average.
The declared situations are:

- every active transition time `t=1,...,19`;
- the near-zero, ordinary, and upper-tail observation-magnitude bins defined
  by the fixture's observed absolute values (bin cut points are computed once
  from the calibration partition and frozen before claim fixtures); and
- the specifically monitored times `t=5` (the observed Phase 8D failure time)
  and `t=14` (the earlier transformed-guide collapse time).

At each situation, compare the candidate with bootstrap, transformed Student,
stationary independence, and UKF K=1.  One candidate loss is a promotion veto,
even if the fixture-level bootstrap interval is positive.  This gate is a
sanity check and is never used to tune the schedule or the observations.

## Evidence contract

| Item | Declaration |
| --- | --- |
| Scientific question | validity and replicated active-time efficiency contrast of the repaired exact-likelihood guide |
| Exact comparator | shared base-mass-aware frozen APF evaluator with complete `q` |
| Primary pass | all branches valid and all required identities/parity checks pass |
| Promotion veto | any active-time/situation heuristic loss or invalid accepted branch |
| Statistical criterion | fixture-cluster log-ESS interval above zero against every declared comparator; otherwise descriptive only |
| Explanatory diagnostics | raw ESS, max normalized weight, log-weight spread, stationarity residual, observation bins, runtime, trace counts |
| Continuation veto | target/measure mismatch, missing records, corrupted fixture, invalid comparator, GPU/memory failure, or budget exhaustion |
| Nonclaims | no posterior, likelihood-unbiasedness, global convergence, generality, HMC, production, or default claim |
| Preserving artifact | per-fixture manifests/results, aggregate JSON/Markdown, statistical-analysis JSON, and a phase-close note |

## Budget and stop conditions

The primary arm is capped at twelve fixtures, two branches, five families, and
two GPU-hours.  The optional confirmatory arm is capped at six fixtures, two
branches, five families, and one additional GPU-hour.  A localized harness or
serialization repair may consume at most two retries with fresh output roots.

Stop immediately for a target/measure mismatch, missing exact denominator,
non-finite accepted program, invalid comparator, corrupted artifact, or a
budget breach.  A candidate-only Newton failure is preserved and triggers the
predeclared schedule-repair path; it does not erase the other valid records.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| `quarter_long_12` | Phase 8D fixed-schedule diagnostic and four clean replays | may be C2- or seed-specific | independent calibration residual and per-row validity before ESS | candidate hypothesis, not default |
| `N=4096` primary arm | bounded-cost replication design | particle-count effects may differ from `N=8192` | optional confirmatory arm and per-`N` report | convenience design |
| twelve observation fixtures | disjoint seed set chosen for cluster uncertainty | still too few for tail/general claims | bootstrap width and sign-test result | minimum evidence target |
| two branches per fixture | runner's deterministic branch seeds | branch dependence may be underestimated | report within-fixture branch spread | minimum replication |
| median active-time contrast | protects against one extreme time point | can hide a single loss | unconditional full-time heuristic table | primary summary only |
| Transformed Student `nu=8` comparator | existing reviewed C2 comparator | fixed degrees of freedom may be unfavorable | retain it as a baseline, do not tune on claim data | comparator, not truth |

## Pre-mortem and red-team checks

The run could appear successful while being misleading if all fixtures are
unusually mild, if the median hides time-14 losses, or if branch randomness is
correlated across families.  The observation-bin table, full-time veto, and
recorded stream offsets address these cases.  It could fail for an integration
or ABI reason rather than for the proposal; clean-branch imports, custom-op
identity, and a one-fixture GPU smoke are required before the ladder.  A
positive ESS interval could still coexist with biased likelihood estimates or
poor posterior state accuracy; those quantities are outside this phase and
must not be inferred.

The strongest result that would overturn a favorable interpretation is a fresh
model or observation regime in which the same fixed schedule fails validity or
loses conditionally to a cheap heuristic.  The weakest evidence remains any
single-branch ESS table.

## Required implementation before launch

1. **Completed:** the standard-library statistical aggregator performs the
   fixture-cluster summaries, bootstrap, sign test, conditional heuristic
   table, and strict artifact validation.  Its focused contract suite is
   `tests/highdim/test_c2_phase8e_statistical_aggregate.py` (4 tests).
2. **Completed:** the Phase 8E runner records the replicate-set identifier,
   analysis seed, observation/model seeds, and deterministic branch-seed
   formula in both result and manifest payloads.  The token contract is
   covered by `tests/highdim/test_c2_phase8b_comparator_contract.py`.
3. **Completed:** the frozen observation-bin boundary artifact and its
   calibration partition are committed and verified by the analyzer's
   recomputation/tamper test.
4. **Completed:** the clean integration smoke and focused contracts pass on
   `codex/c2-root-integration-20260907` at `665d085e` (`35 passed, 2
   warnings`), and the bounded launcher is available.  **Remaining launch
   work:** invoke
   `docs/benchmarks/run_c2_phase8e_statistical_replication_20260907.py` from
   the clean integrated branch.  The launcher creates a non-overwriting
   campaign root, generates the twelve claim fixtures, executes the runner for
   each fixture, and invokes the analyzer with the frozen boundary artifact.
   Assemble the result and inference-status tables; the close note must state
   separately which hard vetoes passed, whether any ranking is statistically
   supported, which differences remain descriptive, and what evidence is still
   needed.

## Skeptical plan audit

`PASS_FOR_PHASE8E_LAUNCH_READINESS_AFTER_BOUNDARY_AUDIT`.

The plan does not treat the current one-branch ESS values as uncertainty
evidence, uses independent fixture IDs as the resampling unit, preserves the
active-time heuristic veto, separates validity from efficiency, freezes the
repaired schedule, and declares compute and continuation limits.  The primary
arm cannot launch until the canonical integration boundary, frozen boundary
   artifact, statistical aggregator/runner contracts, and bounded launcher are
   present.  Those code and artifact prerequisites are now satisfied in the
   recovery and clean integration branches; no twelve-fixture claim run has
   been launched yet.

## Exact commands

Run the primary campaign only from a clean integrated commit with one visible
GPU and the `tftwogpu` environment:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n tftwogpu python \
  docs/benchmarks/run_c2_phase8e_statistical_replication_20260907.py \
  --output-root \
  docs/benchmarks/artifacts/c2_phase8e_statistical_replication_20260907/attempt01
```

The launcher freezes `N=4096`, two branches, state seed `20260912`, observation
seeds `424300`--`424311`, `quarter_long_12`, the five-family comparator ladder,
and the boundary artifact.  Fixture generation is explicitly CPU-only; the
five-family runner uses the requested visible GPU and verifies memory growth.
The launcher refuses a dirty worktree, a non-reviewed seed, or an existing
output root.  Its campaign manifest records the exact child commands, commit,
environment, source hashes, return codes, and analysis path.
