# GenUT Austria SIR Antithetic-Ensemble Experiment Plan

Date: 2026-08-03

Status: `CURRENT_SOURCE_RETUNING_REPAIR_AUTHORIZED`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | On the frozen Austria SIR `T=20`, `N=1008` target, does complete-run innovation reflection reduce the Monte Carlo variance, severe score tails, or variance per wall-second of the GenUT likelihood value and its same-scalar score? |
| Candidate | For `K` independent innovation clouds, average the `2K` complete GenUT runs at `Z_k` and `-Z_k`. Both initial and every process innovation are reflected. |
| Equal-cost baseline | Average `2K` complete GenUT runs generated from `2K` mutually independent innovation clouds, using the same evaluator, observations, controls, arithmetic, and number of filter passes. |
| Ensemble sizes | `K in {1,2,4}`. `K=4` is the primary endpoint; `K=1,2` are explanatory scaling diagnostics. |
| Expected failure mode | The nonlinear weighting and reset recursion may destroy negative pair correlation, or sign reflection may cancel value noise without cancelling the unstable score tangent. Rare score explosions may dominate both estimators. |
| Primary promotion criterion | At `K=4`, the paired-replicate bootstrap familywise 95% interval for `log(Var_antithetic / Var_independent)` has upper endpoint below zero for a coordinate. Promotion is coordinate-specific, not all-or-nothing. |
| Promotion veto | Any invalid constituent, failed GPU/XLA/TF32 or memory policy, broken `Z/-Z` identity, incomplete equal-cost ensemble, or more than 5% relative finite-difference disagreement between the antithetic score and the derivative of its own averaged scalar. |
| Continuation veto | Wrong observations/controls, corrupted prior tuning artifact, evaluator/noise coupling that does not implement tensor reflection, unrecoverable GPU failure, or the 20-minute campaign budget being exhausted. |
| Repair trigger | A localized harness, serialization, XLA compilation, or resource error that leaves the scientific target, comparison, criteria, and total budget unchanged. |
| Explanatory diagnostics | `K=1,2`; antithetic pair correlation; estimator means; SGQF differences and RMSE; severe-tail counts; raw and ensemble runtimes; variance-times-runtime ratios. |
| What must not be concluded | No exact Austria likelihood or score, unbiasedness, general antithetic superiority, HMC efficiency, posterior correctness, default change, or physical-score agreement. |

The estimator and its score are

```text
ell_anti,K(theta) = (1/(2K)) sum_k [ell_G(theta; Z_k) + ell_G(theta; -Z_k)]
score_anti,K(theta) = (1/(2K)) sum_k [score_G(theta; Z_k) + score_G(theta; -Z_k)].
```

Because every constituent score is the total derivative of its own fixed-noise
finite GenUT scalar, the average is the total derivative of the averaged
scalar. The claim run verifies this identity by central finite differences at
the executed FP32/TF32 precision. No external particle-filter score is inserted
into the GenUT score.

## Evidence Contract

| Item | Contract |
|---|---|
| Scientific question | Whether antithetic complete-run coupling improves conditional GenUT value/score computation on the one frozen Austria dataset at equal filter-pass cost |
| Exact baseline | `2K` independent complete GenUT evaluations, not one evaluation and not a median or trimmed estimator |
| Primary endpoint | `K=4` variance ratio for value and each of three physical-score coordinates across 16 independent ensemble replicates |
| Statistical evidence | Deterministic paired bootstrap of ensemble-replicate rows; Bonferroni familywise 95% intervals across the four primary coordinates |
| Hard veto evidence | Finiteness, existing OT/reset residuals, displacement gate, GPU placement, XLA/TF32 and memory policy, exact tensor-negation audit, equal constituent count, same-scalar finite difference |
| Explanatory evidence | Pointwise intervals for `K=1,2`, pair correlations, severe tails, timing, and deterministic same-target SGQF gaps |
| Accuracy reference | Existing fixed level-2 SGQF value/score on the exact observation hash, explicitly an approximation rather than truth |
| Unavailable reference | The current `O(N^2)` online SIR teacher observes before its first transition and is therefore event-order-mismatched to frozen `y1:y20`; it is excluded rather than silently relabeled |
| Result artifact | Versioned JSON plus Markdown under `docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/attempt01/`, followed by a result note |

## Frozen Scope And Replication

| Choice | Value | Provenance and status |
|---|---:|---|
| Model/observations | `austria_sir_T20`, frozen `y1:y20` | Existing leaderboard target; source observation hash must be `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07` |
| Parameter | `(0,0,0)` | Existing target point; baseline, not a universal operating point |
| Particle count | `1008` | Existing scope-specific tuned GenUT target |
| State/score dimensions | `18 / 3` | Austria model |
| Controls | `epsilon=8`, Sinkhorn/balance `16/16`, ridge `1e-5`, diagonal correction `4` steps at strength `0.2`, floor `1e-5` | Frozen prior scope-specific tuning artifact; no retuning on claim rows |
| Design | Fixed replicated cubature design | Existing positive design for dimension 18 |
| Replicates | `16` independent ensemble rows | Feasibility-scale uncertainty; not a broad default decision |
| Maximum pair count | `K=4` | Eight complete passes per estimator |
| Arithmetic/backend | TensorFlow FP32, TF32 enabled, GPU/XLA | Repository production target |
| Randomness | Stateless, disjoint seed domains for antithetic roots and independent controls | Every seed and sign is stored with its constituent output |

Within a replicate, results for smaller `K` reuse prefixes of the `K=4`
constituents. Across replicates the root clouds are independent. The
antithetic and independent arms use disjoint root clouds. The statistical unit
is the independent ensemble replicate, never an individual constituent.

The severe-tail diagnostic is predeclared as any score constituent or ensemble
with `max(abs(score)) > 1000`. This is an explanatory threshold chosen to
separate the historical order-`10^4` Austria tangent explosion from ordinary
order-`10^2` scores; it is not a correctness theorem or promotion criterion.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Prior controls | Austria leaderboard scope tuning | Holds the finite program fixed while testing only coupling | Stale source or mismatched observations | Validate controls, target hash, dimensions, and tuning artifact path before GPU evaluation | Reviewed baseline |
| `K=4` primary | Bounded cost and requested multiple antithetic copies | Tests whether averaging preserves any benefit at a useful ensemble size | Smaller `K` may look favorable by chance | Report all three `K`, but allow primary claims only at `K=4` | Hypothesis |
| 16 replicates | Prior Austria and antithetic feasibility convention | Gives a bounded screen with raw rows and bootstrap uncertainty | Heavy tails can leave wide/unstable intervals | Preserve all constituents and report intervals/tails; no ranking when inconclusive | Convenience choice |
| SGQF comparator | Existing same-target deterministic artifact | Detects gross movement relative to a second approximation | SGQF bias could make closer look better incorrectly | Label all SGQF results explanatory; never use as promotion criterion | Diagnostic approximation |
| 5% FD gate | Existing FP32 GenUT score-audit convention | Tests the claimed derivative at executed precision | Cancellation or branch proximity can cause numerical FD failure | Evaluate relative factors `0.004,0.008` with absolute floors `0.0004,0.0008` on a frozen pair and report both | Reviewed engineering veto |
| Tail threshold 1000 | Historical Austria scores, including a `-13722` seed | Separates catastrophic tangent excursions from the ordinary scale | A large but valid physical score could be mislabeled | Explanatory only; raw score and SGQF scale remain visible | Diagnostic convention |

## Skeptical Pre-Execution Audit

| Risk checked | Finding and disposition |
|---|---|
| Wrong baseline | A one-run baseline is half cost. Rejected. Both estimators execute exactly `2K` complete filter passes. |
| Proxy promoted | SGQF distance, moment residuals, pair correlation, runtime, and `K=1,2` are explanatory only. The primary statistic is the `K=4` equal-cost variance ratio. |
| Missing exact oracle | Austria has no exact `T=20` likelihood/score oracle. The result will distinguish precision from accuracy and will not claim bias reduction from SGQF proximity. |
| Mismatched score teacher | The available teacher has observation-before-transition event order. It is excluded from the claim. |
| Hidden target changes | Observations, theta, design, controls, particle count, and evaluator are identical between arms and hash/manifest checked. |
| Unfair comparison | Both arms use disjoint root clouds, the same number of passes, identical aggregation, and the same compiled evaluator. Runtime is measured per constituent and summed by estimator. |
| Branch/nonfinite masking | Every constituent is retained; an ensemble is invalid if any member is invalid. No trimming, median selection, seed rejection, or replacement is allowed. |
| Score not derivative of value | The averaged score is checked against finite differences of the exact averaged scalar at both predeclared steps. |
| Underpowered ranking | Sixteen rows may be inconclusive under heavy tails. In that case the correct verdict is descriptive viability, not improvement. |
| Artifact cannot answer question | Raw signed antithetic constituents, independent constituents, cumulative ensembles, runtime, validity, pair correlation, primary intervals, and SGQF diagnostics are all preserved. |

Audit verdict: `PASS_AFTER_EQUAL_COST_PRIMARY_ENDPOINT_AND_REFERENCE_REPAIR`.
The plan answers the bounded Austria coupling question and cannot support a
default or HMC claim.

### Current-Source Tuning Repair

The first smoke stopped before a GenUT evaluation because the historical July
tuning checkpoint contains GenUT plus comparator rows and the initial reader
incorrectly required one total row. That harness defect was repaired and
regression-tested. The second smoke selected the GenUT row and then correctly
failed the source-closure identity gate: later opt-in pairwise and projected-
cumulant extensions changed the shared callable closure after the July tuning
artifact was issued. Zero-step tests establish structural no-op behavior, but
the repository per-scope tuning policy still makes the July artifact stale.

Before reading antithetic claim rows, run a fresh current-source tuning step:

- reuse the original Austria eight-arm `SIR_CONTROLS_GRID` on the same disjoint
  calibration/validation datasets and tuning seeds `98101,98102`;
- explicitly freeze pairwise correction steps/strength to `0/0` and projected-
  cumulant correction steps/strength to `0/0`, with both floors `1e-5`;
- retain the original selection order: validation diagonal-moment objective,
  calibration objective, then validation scaled conditional variance;
- issue and validate a new repository-owned route identity from the selected
  current callable and full control set; and
- save the tuning rows, selection, source hashes, GPU/XLA/TF32/memory policy,
  and manifest in a fresh tuning artifact.

This repair does not change the target, candidate, equal-cost baseline,
primary endpoint, promotion criteria, claim seeds, or nonclaims. It refreshes
a stale prerequisite required by the active tuning policy. If no current arm
is eligible, stop. If an arm is selected, both the smoke and claim must consume
that exact tuning artifact and revalidate its current source identity.

The first complete current-source smoke then exercised all 16 constituents,
which were valid, but the finite-difference harness incorrectly treated the
relative factors `0.004,0.008` as absolute steps at the zero parameter vector.
The intended established convention realizes the absolute floors
`0.0004,0.0008` at this target. The 10-times-larger perturbations crossed
unstable branches and one endpoint became nonfinite. This is a localized audit-
harness error: repair the step construction, preserve the vetoed smoke, and
rerun in a fresh directory without changing the scientific comparison.

## Budget, Stop Conditions, And Commands

- One current-source eight-arm tuning step, one one-replicate GPU/XLA smoke,
  and one 16-replicate claim run; localized infrastructure retries remain
  within the same 20-minute total campaign budget.
- Claim budget: 256 complete GenUT constituent evaluations plus at most 24
  finite-difference constituent evaluations and one compile-only warmup.
- Stop on target/control mismatch, invalid output directory reuse, failed GPU
  memory policy, inability to establish GPU/XLA/TF32 execution, broken
  reflection/equal-cost identity, or total wall time above 20 minutes.
- Never overwrite a prior attempt directory.

Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_genut_austria_antithetic_ensemble.py

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_austria_antithetic_ensemble.py \
  --tune-only \
  --output-root \
  docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/tuning_attempt01

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_austria_antithetic_ensemble.py \
  --replicates 1 \
  --tuning-artifact \
  docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/tuning_attempt01/result.json \
  --output-root \
  docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/smoke_attempt03

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_austria_antithetic_ensemble.py \
  --replicates 16 \
  --tuning-artifact \
  docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/tuning_attempt01/result.json \
  --output-root \
  docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/attempt01
```
