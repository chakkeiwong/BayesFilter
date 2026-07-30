# PP-UKF preserved-sample posterior validation plan

Date: 2026-07-30

Status: `READY_FOR_REPRODUCIBLE_ATTEMPT_07`

## Research intent ledger

| Field | Binding decision |
| --- | --- |
| Main question | Do the ten current frozen-kernel PP-UKF HMC sample sets agree with an independently archived same-mathematical-target affine plain-HMC reference at distribution-sensitive summary level? |
| Candidate mechanism | Attempt-11/09/08/07 retained model-coordinate archives from the fixed-identity NeuTra HMC campaign |
| Reference | July-16 target-bound affine plain-HMC comparator, using its archived retained model-coordinate tensor; wrapper identity `036948...` is admissible only because its artifact binds mathematical target signature `d3ed745...` and scope `PP-UKF-six-probit-initial-observation-first-v1` |
| Primary criteria | Per-parameter mean and standard-deviation differences with Monte Carlo uncertainty; simultaneous 5/50/95% quantile interval overlap checks; covariance/correlation discrepancy as an explanatory distributional check |
| Promotion criterion | A candidate remains posterior-compatible only if every declared primary mean/scale/quantile check passes against the reference under the predeclared uncertainty rule and all source/shape/finiteness checks pass |
| Hard veto | Missing or mismatched target/math signature, archive hash failure, wrong coordinate shape, nonfinite draws, nonfinite uncertainty, or reference/candidate identity ambiguity |
| Explanatory only | Acceptance, HMC runtime, ESS, R-hat, pairwise candidate distance, covariance/correlation differences, and tail/energy diagnostics |
| Nonclaims | No exact posterior correctness, no proof that the affine comparator is exact, no sampler superiority, no best-kernel ranking, no production/default readiness, and no broad scientific PP-UKF validity |

## Evidence contract

The comparison is on the same declared approximate PP-UKF posterior target, in
the six source-probit/model coordinates named by the target adapter. Warmup is
excluded. Each candidate and the comparator must have four chains and finite
retained draws. The comparison artifact will preserve source paths, SHA-256
hashes, target signatures, coordinate names, draw counts, summary estimates,
uncertainty intervals, decision statuses, and the exact command/environment.

The uncertainty method is a chain-aware block bootstrap over retained draws:
resample chains with replacement, then resample contiguous within-chain blocks
of length `max(20, floor(sqrt(draws_per_chain)))` until the original chain
length is restored. Use 1,000 fixed-seed bootstrap replicates. For a candidate
versus reference, bootstrap the difference of the pooled equal-weight chain
means, standard deviations, and quantiles. A primary check passes when the
95% percentile interval for the difference lies within the predeclared
practical tolerance:

- mean: `0.10 * reference SD`;
- SD: `0.10 * reference SD`;
- quantiles: `0.15 * reference SD` for each of q05, q50, q95.

These are compatibility screens, not proofs of equality. Because retained
lengths differ, no candidate is ranked by the resulting descriptive metrics.
Standard deviations use the population convention (`sqrt(mean((x-mean)^2))`)
to match the archived comparator's `posterior_summary` field.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Archived affine plain-HMC as reference | Existing same-target comparator artifact | Reference has its own sampler or finite-MCMC error | Verify mathematical target signature, scope, shape, hashes, and finite draws | Reviewed reference, not exact oracle |
| Model-coordinate archive comparison | Current PP-UKF archive callback and comparator `model.tensor` | Comparing latent coordinates instead of target coordinates | Require six named coordinates and explicit archive role metadata | Binding |
| Equal-weight chain pooling | Four independent chains in both campaigns | One unusually long/short chain dominates | Report per-chain counts and use chain-aware bootstrap | Reviewed estimator |
| Block bootstrap | MCMC serial dependence | IID bootstrap understates uncertainty | Fixed contiguous blocks and sensitivity report | Reviewed uncertainty method |
| Tolerances | P4 historical six-mean margin adapted to distributional summaries | Too loose or too strict for full-distribution diagnostics | Record tolerances in artifact; do not rank candidates | Hypothesis/screen |
| No new HMC | Ten candidate archives and an admitted same-target comparator exist | Archive mismatch makes comparison meaningless | Fail closed on identity/hash/shape/finiteness before summaries | Binding non-action |

## Implementation sequence

1. Add a validation harness that loads and verifies all ten current cumulative
   model archives plus the comparator model archive, including hashes, target
   signatures, coordinate names, chain count, and finite values.
2. Compute repaired rank-normalized R-hat/ESS summaries, per-chain moments,
   pooled means/SDs/quantiles, covariance/correlation matrices, and pairwise
   candidate summaries without using any result to rank candidates.
3. Compute the fixed-seed chain-aware block-bootstrap uncertainty intervals for
   candidate-reference differences and apply the declared compatibility screen.
4. Emit a compact JSON result, Markdown decision note, and strict manifest with
   command, environment, git commit, seeds, source hashes, and artifact paths.
5. Run focused tests/compile checks and inspect the terminal result. No GPU or
   HMC launch is authorized by this plan.

## Skeptical plan audit

- **Wrong baseline:** the reference is not accepted by wrapper signature alone;
  mathematical target identity and scope are checked from the artifact.
- **Proxy promotion:** convergence diagnostics and pairwise distances remain
  explanatory; only predeclared uncertainty intervals drive compatibility.
- **Missing stop conditions:** identity, hash, shape, finiteness, bootstrap,
  and reference integrity failures are hard stops.
- **Unfair comparison:** all summaries use the same model-coordinate names and
  same quantile definitions; unequal retained lengths are recorded and block
  ranking.
- **Hidden assumptions:** the comparator is an approximate-filter posterior
  MCMC reference, not an exact Bayesian posterior oracle; this is explicit.
- **Environment mismatch:** the calculation is deterministic TensorFlow/CPU
  post-processing and does not claim GPU or sampler performance.
- **Plan-could-pass-while-misleading:** agreement can occur if both samplers
  target the same wrong implementation. The note therefore cannot conclude
  posterior correctness and reserves exact/reference validation for a later
  target audit.

Audit verdict: `PASS_FOR_EXECUTION_WITH_NO_HMC_LAUNCH`.

## Harness repair ledger

Attempt 1 stopped before loading the reference because the stored target scope
was nested under the batch execution surface. Attempt 2 stopped at module
import because the standalone script had not inserted the repository root into
`sys.path`. Attempt 3 reached archive verification and stopped because the
reference hash ledger uses paths relative to the comparator root. These are
recorded infrastructure failures with no scientific output and no HMC launch.
The harness was then revised to use TensorFlow for all numerical, bootstrap,
and decision calculations; NumPy is not imported or used by the admitted path.

Attempt 4 completed the frozen screen but its compact result omitted candidate
archive path/hash provenance. Attempt 5 added that provenance. Post-run red-team
review then found that a failed equivalence criterion must not be described as
material disagreement when the interval merely crosses the equivalence margin.
Attempt 6 therefore added an explicit three-way classification without changing
the samples, statistics, bootstrap seeds, tolerances, or decisions:
`equivalence_established`, `material_disagreement_supported`, or
`inconclusive`. Its result was internally consistent, but its run manifest
identified commit `74f7aa9b`, which predates the new harness. Attempt 6 is
therefore superseded for reproducibility. The frozen harness and tests must be
committed before one identical terminal attempt is executed.

## Provisional result before reproducible rerun

Attempt 6 completed in `11.9 s` of CPU-only TensorFlow post-processing. All ten
source archives passed identity, SHA-256, shape, and finiteness verification.
All ten repaired HMC convergence screens reproduced their terminal pass status.

- `L=12` and `L=17`: posterior equivalence established on all 30 declared
  mean/SD/quantile checks.
- `L=5,9,13,14,18,19,24,25`: posterior equivalence inconclusive because one or
  more uncertainty intervals cross a practical equivalence boundary.
- No candidate: material posterior disagreement supported.
- Every candidate's point estimate is within its practical tolerance.
- No ranking was performed.

Provisional verdict: `PASS_TWO_EQUIVALENT_EIGHT_INCONCLUSIVE_ZERO_DISAGREEMENT`.
No HMC rerun occurred or is required to preserve this result. A later decision
to qualify any of the eight inconclusive kernels would require additional
retained evidence under a new bounded continuation plan; this result does not
justify rejecting those kernels or ranking `L=12` against `L=17`.
