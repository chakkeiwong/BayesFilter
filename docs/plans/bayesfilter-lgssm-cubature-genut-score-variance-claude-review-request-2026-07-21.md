# Claude Review Request: LGSSM Cubature/GenUT Recursive Score

Status: `READY_FOR_REVIEW`

Date: 2026-07-21

## Role Contract

Codex is the supervisor and executor. Claude is a read-only reviewer. Claude
must not edit files, run experiments, launch agents, install packages, or
change repository state. Claude should return the complete review as Markdown
text; Codex will save that response as a separate review artifact.

## Review Question

Audit the current LGSSM Cubature/GenUT algorithm and implementation thoroughly.
Determine whether the implemented likelihood value, reset-moment claims, and
recursive physical-parameter score are mathematically and programmatically
correct for the finite computation that is actually executed. Pay particular
attention to the large seed-to-seed score variance observed at `T=50`, and
propose concrete variance-reduction remedies.

The review must distinguish:

1. correctness of the finite value program;
2. correctness of the derivative of that same finite value program;
3. agreement with the exact Kalman likelihood/score;
4. statistical precision of the 16-seed comparison; and
5. any claim that would transfer to nonlinear high-dimensional filtering.

LGSSM is a controlled diagnostic for a future high-dimensional nonlinear
filtering application. It is not an LGSSM-estimation objective and is not a
NAWM experiment. Do not recommend a local LGSSM-only optimization that would
not plausibly transfer to high-dimensional nonlinear state-space models.

## Exact Scope

Review only these files and the cited line/section anchors. Inspect additional
files only when needed to verify a symbol imported by one of these paths, and
identify every such additional file in the report.

### Runtime and comparison harness

- `docs/benchmarks/run_lgssm_cubature_genut_fp32.py`
  - lines 116-165: Cubature, Gaussian GenUT, and Contract E residual designs;
  - lines 189-331: finite Sinkhorn barycentric map and its all-parameter JVP;
  - lines 334-423: Contract E reset and reset JVP composition;
  - lines 426-555: observation stream and analytic Kalman value/score oracle;
  - lines 649-804: recursive particle value and forward sensitivity score;
  - lines 807-833: finite-difference diagnostic only;
  - lines 897-1040: route evaluation, audit fields, and score/value outputs;
  - lines 1045-1250: intervals, hard-valid screen, and artifact construction.
- `docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py`
  - lines 57-164: 16-seed intervals and paired absolute-error comparison;
  - lines 166-276: matched `T=2,10,50` campaign and validity contract.
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py`
  - lines 18-166: Contract E-Chol forward reset;
  - lines 169-344: moment, Cholesky, affine, and reset JVPs;
  - lines 347-480: reverse derivative helpers, if needed for comparison.
- `tests/test_lgssm_cubature_genut_fp32.py`
  - all tests, especially lines 104-171 and 174 onward.

### Mathematical source and claims

- `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`
  - sections around lines 1495-1680: non-fused Cubature residual, OT order,
    Cholesky restoration, covariance proposition, and total-score boundary;
  - sections around lines 1681-1830: GenUT construction, positivity,
    equal-weight representation, value/variance/score proposition;
  - lines 1832-1899: comparison table and limitations/remedies.
- `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-plan-2026-07-21.md`
  - research question, evidence contract, skeptical audit, and nonclaims.
- `docs/plans/bayesfilter-lgssm-recursive-score-matched-three-horizon-comparison-result-2026-07-21.md`
  - completed 16-seed results, score intervals, paired comparisons, and
    post-run red-team note.
- `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02/result.json`
  - raw rows, exact oracle values/scores, interval calculations, route IDs,
    replay and reset/Sinkhorn diagnostics.
- `docs/benchmarks/artifacts/lgssm_recursive_score_matched_t2_t10_t50_20260721/attempt02/run_manifest.json`
  - commit, environment, hardware, memory policy, controls, seeds, and hashes.

## Required Mathematical Audit

Derive or check the following in the repository notation, not just by verbal
inspection:

- Cubature points `sqrt(d)[+e_a,-e_a]`, replication, zero mean, and identity
  population covariance.
- Gaussian GenUT specialization (`s=0`, `k=3`) and whether the claimed alias
  is exact for the executed design.
- Sinkhorn scaling conventions, coupling marginals, barycentric map, and the
  derivative of the finite unrolled Sinkhorn iterations, including the
  derivative of the cost normalization and the `1e-7` denominator floor.
- Contract E target/transported moments, residual injection, cross-covariance
  terms, fixed ridge, Cholesky restoration, and the exact covariance identity
  actually proved by the code.
- Every recurrence in `_particle_value_score_recursive`, including stationary
  initial covariance derivatives, transition/process-noise derivatives,
  observation log-likelihood derivatives, normalized-weight derivatives,
  reset JVP, and the fact that weights are reset to uniform after each reset.
- Whether the analytic Kalman score includes the derivative of the stationary
  initial covariance and agrees with the value oracle.
- Whether any operation is nondifferentiable or branch-dependent in a way that
  invalidates the claimed total derivative: `maximum`, finite floors, Cholesky
  branches, finite Sinkhorn truncation, parameter-dependent random/design
  construction, or TensorFlow loop semantics.
- Whether the LaTeX propositions overclaim likelihood, variance, or score
  validity relative to the finite algorithm.

Classify each issue as `correct`, `wrong relative to the stated target`,
`unsupported`, `not checked`, or `heuristic only`. Give exact file and line or
equation anchors for every material finding.

## Required Variance-Reduction Analysis

Explain why the score variance is large, especially at `T=50`, and rank at
least five concrete remedies. For each remedy state:

- the estimator or algorithmic change;
- whether it preserves the derivative of the current finite likelihood scalar;
- whether it changes the probability measure, reset target, or scientific
  quantity;
- expected effect on bias, variance, memory, and high-dimensional scaling;
- whether it is compatible with float32/TF32 and the no-autodiff recursive path;
- the cheapest discriminating experiment and its pass/fail diagnostics.

At minimum assess: common random numbers/antithetic innovations, replicated
designs, Rao-Blackwellization or conditional Gaussian treatment where valid,
score control variates/baseline subtraction, per-time score decomposition,
common-path paired estimators, increasing `N`, Sinkhorn convergence controls,
and whether averaging scores across independent particle clouds is preferable
to changing the reset rule. Do not call a target-changing estimator a repair
without saying that it changes the claim.

## Required Statistical Audit

Check whether the simultaneous 95% interval gate and paired method comparison
are correctly interpreted for 16 seeds. Discuss near-zero Kalman score
denominators, multiplicity/critical-value assumptions, Monte Carlo error, and
what can and cannot be concluded from the current result. Do not rank methods
from descriptive means when intervals include zero.

## Requested Report Format

Return a self-contained Markdown report with these sections:

1. `Executive Verdict`;
2. `Findings` (severity ordered, exact anchors, and concrete consequences);
3. `Mathematical Audit`;
4. `Implementation Audit`;
5. `Score-Variance Diagnosis`;
6. `Variance-Reduction Options` (ranked table, preserving vs changing target);
7. `Statistical and Evidence Audit`;
8. `Minimal Repair/Test Plan`;
9. `Nonclaims and Transfer Limits`;
10. `VERDICT: AGREE` or `VERDICT: REVISE`.

Findings must come before any general summary. If a claim cannot be proved from
the inspected code or equations, say `unsupported` or `not checked` directly.
Do not edit files or execute commands.

## Evidence Snapshot

The completed matched campaign used `N=1008`, 16 particle seeds `82220` to
`82235`, dataset seed `81100`, `T=2,10,50`, float32/TF32 GPU arithmetic,
`epsilon=2`, eight Sinkhorn steps, ridge `1e-5`, and no XLA. It executed 96
particle rows and passed all hard-valid/replay checks. Peak allocator memory
was `474,037,248` bytes and wall time was `209.3 s`.

All score-coordinate simultaneous intervals against the analytic Kalman score
included zero, but intervals were wide at `T=50`, especially for `phi3`,
`q_scale`, and `r_scale`. No paired Cubature-versus-Contract-E score or value
interval excluded zero. Gaussian GenUT was bitwise identical to Cubature in
this Gaussian specialization.

Focused tests: `11 passed` in `6.13 s`.

Source hashes in the result artifact:

- runner: `f07a0122e5687de18f2ae9d72d7a745e6f0380b11a85d487d573dd949b0b62bd`;
- campaign: `18596442272b67794c901ad998bd48b73794588ae28c7ea20c2bdefbe9697a5e`;
- plan: `48fb0103ca04c16f26c6a8986bbc339f2d20665dc93dc9362f9d2bc2fdb1a125`.

## Nonclaims

Even a positive review must not claim exact nonlinear filtering validity,
unbiasedness, method-wide superiority, `1/N` convergence, HMC readiness, XLA
readiness, production readiness, or any NAWM result. The review is advisory;
Codex remains responsible for any code changes and experiments.

## Required Ending

End the report with exactly one of:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```
