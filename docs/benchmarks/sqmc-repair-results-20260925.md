# SQMC correctness repair results

Date: 2026-09-25  
Branch: sqmc-development  
Baseline: f3995a06a467f16574f96bbc8a68ccbbc4e30dad  
Plan: master repair program (docs/plans/sqmc-repair-master-program-20260925.md)

## What was repaired

The shared TensorFlow LGSSM specification now fixes the P44 variance
parameterization, raw initial law, transition-before-first-observation timing,
and analytical tangents. Generic, transfer, 10D, 3D, and tuning consumers call
that shared implementation. The four route constructors now enforce their specified
ancestry and cap settings, including the .97 ablation; every direction returns a complete
score; non-finite values and incomplete seed sets fail closed. Tuning requires
exact scope identity, complete valid calibration/validation partitions, and
disjoint untouched claim seeds. The restored multi-stage annealed executor
passes its existing analytical parity tests. The inverse-CDF boundary and
ancestry count use the checked TensorFlow operations.

The retired comparison CLI files remain runnable compatibility entry points,
but their former claims and command-line semantics are retired. They now
delegate to the scope-bound diagnostic tuner. Saved tuning JSON is evidence,
not a reloadable authority; the in-process issued artifact is required for an
untouched evaluation.

## Validation evidence

The bounded repair campaign used CPU reference exceptions with GPUs hidden and
two escalated GPU/XLA compatibility attempts. The complete final CPU suite
reported 73 passed, 1 deselected in 113.83 s. Focused groups reported 25
consumer/tuning tests and 44 corrected finite-program tests. The two early
groups reported 9/10 and 32/33 because an old autodiff/XLA fixture failed; the
replacement analytical path and final suite pass. The failed fixture is
preserved in its attempt log and is not counted as evidence for the repaired
path.
Four tests in the pre-existing historical runner test module could not be collected because
docs/benchmarks/run_ledh_pfpf_genut_sqmc.py is absent from this checkout; the
missing file was not recreated. The final suite therefore reports the repaired
SQMC/core tests that are present, and this collection limitation remains part of
the evidence boundary.

The tuning smoke evaluated all four routes on P44, D=3,T=2,N=12, with
two disjoint calibration, validation, and untouched seeds per route. All 32
evaluations were complete, finite, and full-score; controls were frozen only
for that diagnostic scope. The GPU/XLA probe ran all four routes in float64 and
float32, with active correction and pairwise steps. All eight cases passed
non-JIT versus JIT agreement; float64 score differences were at most
5.75e-16, and float32 differences were below 3.30e-4. Memory growth was
verified on the RTX 4080 SUPER. These are compatibility and finite-program
checks, not accuracy or HMC evidence.

Artifacts, exact commands, manifests, source hashes, and logs are under
docs/plans/artifacts/sqmc-repair-20260925/. The run used 341.13 CPU seconds (including the collection failure)
across the completed groups and 224.43 GPU seconds across the two GPU probes,
well within the 3600/1200 second budget. A final reporting-only check additionally passed four tests in 0.19 pytest seconds (18.97 seconds reported command wall time). Historical Austria raw files cited by
the former reports were not present and were not fabricated.

## Observed score accuracy in the tiny held-out checks

The corrected P44 checks are D=3, T=2, N=12 CPU float64 reference diagnostics
at theta=(0.25, log(.18), log(.12), .04), using the saved scope-specific
diagnostic tuning artifacts. They do not establish production accuracy.
The four score coordinates are transformed persistence, log process-variance
scale, log observation-variance scale, and initial-mean scale.

Relative error is not a suitable universal score criterion: the exact score
vanishes at a regular interior MLE. The raw scores and absolute Euclidean
errors are reported below instead. These checks are at the fixed theta above,
not at an estimated MLE.

Seed 93001:

| Method | Score 1 | Score 2 | Score 3 | Score 4 | Absolute L2 error |
|---|---:|---:|---:|---:|---:|
| Exact Kalman | -0.148674 | -0.595155 | -0.561413 | +0.183930 | 0 |
| IID | -0.114613 | -0.590909 | -0.541693 | +0.163803 | 0.044408 |
| Hilbert inverse CDF | +0.377395 | -0.807210 | -0.446072 | +0.165716 | 0.579095 |
| Hilbert permutation | +0.109937 | -0.779935 | -0.416824 | +0.167703 | 0.349560 |
| Permutation cap ablation | +0.109990 | -0.779921 | -0.416859 | +0.167703 | 0.349578 |

Seed 93002:

| Method | Score 1 | Score 2 | Score 3 | Score 4 | Absolute L2 error |
|---|---:|---:|---:|---:|---:|
| Exact Kalman | -0.284568 | -1.000254 | -0.541294 | -0.047953 | 0 |
| IID | -0.402709 | -0.989242 | -0.552918 | -0.049096 | 0.119227 |
| Hilbert inverse CDF | -0.268766 | -1.024743 | -0.540234 | -0.052076 | 0.029454 |
| Hilbert permutation | -0.288375 | -1.026583 | -0.533921 | -0.052357 | 0.027955 |
| Permutation cap ablation | -0.288311 | -1.026597 | -0.533945 | -0.052360 | 0.027953 |

The first-score sign error for seed 93001 was subsequently reproduced and
localized to the small-particle likelihood approximation. The
[September 26 diagnosis](sqmc-score-93001-diagnosis-20260926.md) reports
finite-difference parity, first/second-observation contributions, shared
Halton inputs, inverse-CDF ancestry duplication, and bounded interventions.
These original held-out data are now used for debugging and cannot become new
untouched claim data. There is no statistically supported ranking or
before/after campaign performance conclusion. The original 112-cell transfer
campaign has not been rerun.

## Finding dispositions

| Finding | Disposition |
|---|---|
| A1--A4 | Fixed and exercised at the shared consumer/core boundary. |
| A5 | Complete values, scores, oracle values/scores, and finite validity are now required. |
| A6 | False Fisher and HMC metrics removed from active selection and reporting. |
| A7 | Non-finite sentinels are rejected even when a caller sets a validity flag. |
| A8 | Multi-stage annealing restored and tested, including reset composition. |
| A9 | Theorem scope is explicitly limited; no extension theorem is claimed. |
| A10 | Historical equivalence, ranking, default, and HMC claims are withdrawn. |
| A11 | Exact scope, all-seed validity, and disjoint partitions are enforced. |
| A12 | Touched runtime paths use TensorFlow/XLA defaults; historical missing raw data remains an evidence limitation. |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep repaired route implementations as diagnostic candidates | finite values, complete analytical scores, fixed-program FD parity | no repair veto in checked fixtures | small N/T and branch dependence | fresh scope-specific target campaign | route accuracy or ranking |
| Admit any route as default | exact-scope accuracy and uncertainty evidence | NOT ELIGIBLE | no large matched holdout | plan a new campaign with Kalman and practitioner baselines | default superiority |
| Use tuning artifact for HMC | exact scope and downstream posterior evidence | NOT ELIGIBLE | no HMC run and no theorem | separate HMC plan if requested | HMC readiness |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the checked repair fixtures; old historical cells are not revalidated. |
| Statistically supported ranking | None. |
| Descriptive differences | Small-fixture values and route outputs are descriptive only. |
| Default readiness | Not established. |
| Next evidence needed | Fresh disjoint scope tuning, larger matched Kalman comparisons, heuristic set (exact Kalman, bootstrap IID PF, uncorrected PF), conditional regimes, uncertainty intervals, and any HMC claim-specific checks. |

The strongest alternative explanation remains finite-particle approximation or
branch-specific behavior outside these small fixtures. The local finite-
difference checks do not remove that uncertainty. The historical reports are
preserved under docs/plans/artifacts/sqmc-repair-20260925/historical-docs/ and
are superseded for active interpretation.

## Final review and remaining coverage

The final regression deliberately deselected
test_production_score_lane_value_is_likelihood_estimand, whose N=1008, T=10,
four-seed likelihood screen is outside this small-fixture repair validation.
The four missing-runner tests are also not counted as passes. The unavailable
runner was absent in the new worktree, main checkout, and old Claude worktree;
Git history for its path was empty. This is unrecovered historical coverage,
not evidence that those tests passed.

The terminal skeptical review checked target identity, the shared analytical
call chain, incoming annealing weights, the ablation setting, complete-seed
rejection, and separation of calibration from untouched data. No method was
ranked from smoke errors or raw log-likelihood differences. The plan's
correctness tasks are complete; scientific accuracy, production eligibility,
and HMC questions remain separate. Review was by Codex; no independent Claude
review was performed.

The wrapper migration changes the old scripts' CLI/API behavior. Use
run_sqmc_tuning.py --help for supported flags. The 10D legacy wrapper supplies
N=1000 because its 2D residual design requires divisibility by 20; it does not
silently claim to have rerun historical N=1008. The P44 replication entry point
now exposes the two different model definitions instead of subtracting scores
in incompatible parameter coordinates.

The [mathematical note](sqmc-repair-mathematical-corrections-20260925.md) gives
the local derivations and primary-source anchors. Replay scripts, GPU logs,
a snapshot of changed Python files, and final source hashes are preserved under
the artifact root. All repairs remain uncommitted in sqmc-development.
