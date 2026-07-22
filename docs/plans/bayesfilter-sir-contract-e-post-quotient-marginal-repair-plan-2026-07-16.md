# SIR Contract E Post-Quotient Marginal Repair Plan

Date: 2026-07-16

Status: `PASS_LOCALIZED_REPAIR_T5_SCIENTIFIC_LADDER_PENDING`

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can the finite positive transport feeding Contract E satisfy both first-order marginals after the actual row-normalization operation, so that the positive covariance-gap proposition applies and Contract E--Chol remains valid without audit-seed ridge escalation? |
| Candidate mechanism | A fixed, differentiable, GPU/XLA-compatible finite transport schedule or balancing construction whose actual output plan has row sums one and column sums `N*w`; Contract E consumes that same plan without a marginal-changing repair. |
| Exact baseline | Current two-step finite plan followed by row quotient, frozen in attempt 9. |
| Expected failure | More iterations still leave a material post-quotient column residual; a proposed balancing step changes the finite scalar without a total derivative; or a schedule valid at the center fails over the declared domain. |
| Promotion criterion | Predeclared post-quotient row and column marginal requirements pass over the declared design domain, covariance-gap positivity follows numerically within a predeclared roundoff rule, same-scalar AD/manual/FD checks pass, then the frozen `J=2,T=1,2,5` comparison ladder completes. |
| Promotion veto | Any marginal, Cholesky, finite-score, identity, CPU/GPU, or same-scalar derivative failure. |
| Continuation veto | The proposed plan cannot satisfy both marginals without changing to an unapproved mathematical route, the finite program is not differentiable through all balancing operations, artifacts are incomplete, or compute budget is exhausted. |
| Repair trigger | Marginal failure with finite tensors triggers transport-schedule or balancing analysis; it does not trigger ridge escalation. |
| Explanatory diagnostics | Raw and post-normalization marginal residuals, gap spectrum, condition proxies, runtime, peak memory, and per-seed scatter. |
| Must not conclude | Contract E accuracy, teacher accuracy, HMC readiness, leaderboard readiness, source-faithful Zhao--Cui closure, or all-model score correctness from marginal feasibility alone. |

## Entry Conditions

- Attempt 9 is the binding failed-candidate artifact:
  `docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase5_phase6_gpu_paired_attempt09/`.
- The online teacher is viable at `J=1`; at `J=2` it is an independent
  disagreement comparator, not an oracle.
- Contract E--Chol remains the only canonical reset route.
- Attempts 5--9 are exhausted. No new GPU launch is authorized by this plan.

## Mathematical Requirement

Let `P` be the finite nonnegative plan, `M_i=sum_j P_ij`, and
`P'_ij=P_ij/M_i`. Contract E consumes barycentres from `P'`. Its positive-gap
proposition requires

```text
sum_j P'_ij = 1,
sum_i P'_ij = N*w_j.
```

The first equality is automatic when `M_i>0`; the second is not. Any proposed
repair must produce and differentiate the same `P'` that satisfies both. It is
not enough for the pre-quotient `P` to satisfy the column constraint.

## Skeptical Pre-Execution Audit And Frozen Repair

The audit rejected merely replacing `steps=2` by another unexplained count.
The old finite loop mixes epsilon annealing with averaged potential updates;
two steps do not generally reach the terminal epsilon, and even 100 old-style
steps left an independent two-node residual near `2.8e-4`.  The repaired
finite program therefore separates:

- `20` fixed annealing warm-start iterations; and
- a terminal-epsilon IPFP ladder `20, 40, 60, 100`, selected on independent
  fixtures outside audit seeds `87200--87215`.

The smallest ladder point passing both independent two-node and Austria
two-time fixtures at the declared roundoff envelope was `100`.  At `60`, the
two-node post-quotient column residual remained about `9.24e-10`; at `100`, it
was about `2.89e-15`.  Austria reached roundoff earlier but did not determine
the shared schedule.  The frozen repaired route is consequently
`annealing_steps=20`, `balance_steps=100`.

The marginal gate is derived from the standard floating-point summation bound
`gamma_k = k*u/(1-k*u)` with float64 unit roundoff `u`, using an operation-depth
envelope `k=16*N` for exponential inputs, column normalization, streamed
reductions, terminal scaling, and row quotient.  This is a roundoff model, not
a tolerance fitted to the failed audit seeds.

The audit also classified the diagnostics:

- promotion criterion: both final consumed-plan marginals within the declared
  roundoff envelope and nonnegative covariance gap up to that envelope;
- promotion veto: same-scalar derivative mismatch, non-finite tensors,
  Cholesky failure, or identity/XLA failure;
- explanatory only: old pre-quotient marginal residual and convergence-ladder
  trend;
- nonclaim: marginal feasibility does not establish filtering accuracy,
  teacher agreement, HMC readiness, or all-model correctness.

## Default And Assumption Audit

The current values `steps=2`, `epsilon=0.25`, `scaling=0.9`, and
`ridge=1e-6` are convenience/inherited hypotheses, not SIR-certified defaults.
Attempt-9 magnitudes must not select replacements. Before a numerical arm, a
reviewed amendment must independently freeze:

- the parameter/input domain and design seeds distinct from audit seeds;
- candidate schedules or a balancing construction with mathematical provenance;
- absolute and relative row/column marginal requirements justified by the
  positive-gap error propagation, not chosen from observed residuals;
- a roundoff allowance for the minimum gap eigenvalue;
- a raw covariance-bias budget for any fixed ridge;
- CPU/GPU attempt and wall-time budgets and fresh output roots;
- selection, tie, failure, and audit rules.

Audit seeds `87200--87215` and attempts 6--9 are final-only evidence and cannot
be used to select steps, epsilon, scaling, ridge, or thresholds.

## Phases

1. **Complete.** Derive and test a tiny dense finite plan whose actual consumed transform
   satisfies both marginals. Compare every blockwise diagnostic to the dense
   oracle and prove no production `N x N` allocation.
2. **Complete.** Implement the total JVP/VJP through the fixed terminal IPFP
   refinement. Verify
   manual JVP/VJP against autodiff and same-scalar FD on independent tiny
   fixtures.
3. **Complete.** Freeze the independently designed `20+100` SIR transport
   protocol and domain. Review
   the mathematics, thresholds, seeds, compute budget, and nonclaims before
   observing target outputs.
4. **Complete.** Run CPU-XLA design-domain feasibility and same-scalar checks. Reject invalid
   candidates without using audit seeds.
5. **Localized repair complete; scientific comparison pending.** Trusted
   GPU/XLA canaries and the final-only `d=18,T=2,N=256` marginal audit passed.
   The frozen `J=2,T=1,2,5` LEDH--teacher comparison ladder remains a separate
   scientific step because marginal feasibility alone cannot answer agreement.
6. Write a terminal result separating engineering correctness, numerical
   validity, and scientific interpretation.

## Evidence Contract

- Primary criterion: the actual consumed positive plan satisfies both declared
  marginal requirements and yields a valid Contract E--Chol chart over the
  frozen design domain.
- Same-scalar derivatives are engineering evidence only.
- Teacher refinement is assessed before LEDH--teacher intervals.
- An interval containing zero is inconclusive, not equivalence.
- A disagreement does not identify which method is closer to truth.
- Every serious run writes a manifest, checkpoints, terminal result or failure,
  hashes, GPU cap/status, seeds, wall time, and source closure.

## Stop And Handoff Conditions

Stop before implementation if no mathematically coherent plan can make the
actual consumed transform satisfy both marginals with a complete derivative.
Stop before target execution if the pre-result amendment lacks independent
provenance or justified marginal/bias requirements. Handoff to Phase 5 only
after dense/streaming parity, no-`N^2` allocation, total-derivative checks,
identity closure, and an owner-approved GPU budget all pass.
