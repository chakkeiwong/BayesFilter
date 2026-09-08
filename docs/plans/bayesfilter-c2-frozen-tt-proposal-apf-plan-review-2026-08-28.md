# C2 Frozen-TT Proposal APF Plan Review

Date: 2026-08-28

Reviewed plan:
`docs/plans/bayesfilter-c2-frozen-tt-proposal-apf-plan-2026-08-28.md`

Reviewed mathematical specification:
`docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`,
Section `sec:frozen-proposal`

Verdict: `PASS_FOR_BOUNDED_DIAGNOSTIC_EXECUTION`

This is not a promotion, posterior-correctness, pseudo-marginal, HMC, or
default-readiness verdict.

## Findings Repaired During Review

1. The first draft attributed per-step covariance to the original T=20 PF
   reference. That file contains total SE and per-step means, not covariance.
   The plan now binds the T=20 total to the attempt05 reference and the t=0..4
   covariance to the independent Stage 2 T=5 reference, without extrapolating
   covariance beyond its horizon.
2. A compiler-local recursion for reference weights would have duplicated the
   scalar call chain. The plan now requires every auxiliary law to come from a
   partial branch evaluated by the existing generic
   `FrozenProposalAPFProgram`.
3. The existing bounded Legendre transport has the wrong domain and reference
   measure. The plan requires the exact normalized probabilists' Hermite
   incomplete Gram derived in the manuscript.
4. A t=0 TT capture was unnecessary. The plan now implements the manuscript's
   frozen stationary-prior proposal at the reference parameter.
5. Proposal ESS and reference agreement were initially at risk of carrying
   implementation-correctness claims. The final plan makes density-law,
   same-scalar score, and XLA parity hard engineering gates and keeps ESS as a
   proposal-quality diagnostic.
6. A late executable-scope cross-check found that the plan had called 42 the
   config seed. Attempt05 uses 42 as the observation seed and derives fitter
   seed 98466 from `98000 + 100*n + 10*degree + rank`. The plan now pins both
   identities separately and also replaces "current tau" with exact tau
   `1e-6`.

## Equation Cross-Check

| Manuscript requirement | Plan match | Review status |
| --- | --- | --- |
| Retained quadratic `h=VEV^T`, exact `Z_H`, absolute tau, and affine Jacobian | Stage 1 constructor from production-captured cores, snapshot Gram/tau/map parity | Correct |
| Complete Hermite/Student-t mixture | Complete-mixture density and selected-component sampling with density-law test | Correct |
| Normalized Hermite basis and incomplete Gram | Degree 0-6 closed-form kernel plus independent quadrature | Correct |
| Paired left/right KR conditional CDF ending in `E` | Batched paired environments, bracket/monotonicity/inverse tests | Correct |
| Fixed APF scalar includes selected previous `W`, selected `a`, and `q` | Existing generic `_evaluate_core`; direct recomposition and wiring tests | Correct |
| Fixed branch and parameter-independent proposal | Repository-issued frozen branch; runtime receives branch plus theta only | Correct |
| Centered recursive analytical score | Existing generic centered-mark recursion; same-scalar FD and XLA parity | Correct |
| C2 `A=C+gamma I`, stationary `P`, differentiated Lyapunov solve | Dedicated TensorFlow model and residual/FD tests | Correct |
| Initial, transition, and log-beta observation scores | Manual model endpoints and pointwise FD tests | Correct |
| No exact-posterior claim for one frozen branch | Explicit target class and nonclaims | Correct |

## Skeptical Audit

- Exact baseline: checked and scope-bound.
- Proxy metrics: ESS, weight spread, and CDF residual cannot promote the
  method; hard correctness gates are separate.
- Stop conditions: engineering vetoes, candidate rejection, budget exhaustion,
  and owner-decision boundaries are explicit.
- Fairness: same N, same fixture, same horizon, frozen reference parameter,
  three constructed simple proposal adversaries, and the high-N reference are
  all declared.
- Hidden assumptions: tau, Student-t nu, bisection budget/bracket, N, branch
  count, auxiliary law, and runtime parameter domain are audited.
- Stale context: no pre-2026-08-21 LEDH evidence is used; the active C2 direct
  artifact and 2026-08-28 root-cause artifacts are named.
- Environment: CPU-only tests deliberately hide CUDA; serious runs require
  trusted GPU access, memory growth, actual placement, float64, and XLA.
- Artifact relevance: every output is tied to math correctness, call-chain
  correctness, proposal-law correctness, or the n=4 diagnostic question.

No remaining material mismatch between the plan and Section
`sec:frozen-proposal` was found. Implementation may proceed under the stated
budget and nonclaims.

## Terminal Review Amendment (2026-08-29)

The verdict above was a pre-execution review for bounded diagnostic execution.
The terminal evidence now separates the implementation gates from the
historical direct-trajectory gate.  The retained proposal equations,
generic-APF call chain, and analytical same-scalar score pass their executable
checks (16/16 serious branches).  The retained-TT candidate nevertheless
fails the declared viability screen because its minimum mean normalized ESS is
`0.0006033922 < 0.0025`, and it loses descriptively to simple proposals at
several salient times and on the full total.  Those are candidate-rejection
findings, not evidence against the importance-correction derivation.

The current seven-output direct route does not reproduce the preserved
attempt05 T=20 scalar (`31.1071889584` versus `36.9423456524`; an earlier
attempt was `27.1328858463`).  This compatibility failure is recorded as a
hard terminal boundary and is not silently folded into the proposal-law
verdict.  The plan and the LaTeX implementation-boundary subsection therefore
match: proposal/APF correctness is assessed independently, while historical
trajectory compatibility remains unresolved and blocks claim-bearing reuse.
The smoke artifact is intentionally different: its direct control checks only
the retained seven-output call chain and CPU parity, with no historical scalar
comparator. Future result schemas must keep that field distinct from
`historical_reference_match`.
No additional serious run was launched after the two fit attempts because the
plan budget and the declared compatibility stop condition were exhausted.
