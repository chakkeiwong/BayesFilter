# Audit: SGU combined structural target closure plan

## Audit Role

This is a second-developer audit of
`docs/plans/bayesfilter-sgu-combined-structural-target-closure-plan-2026-05-07.md`.
The audit checks whether the plan is safe to execute and whether any point is
missing before implementation.

## Verdict

The plan is correctly scoped and executable, but only if Gate B is treated as a
real go/no-go gate rather than as an expected pass.  If the quadratic policy
does not improve both RMS and max residuals relative to the linear policy on
the same completed-state grid, the combined label must remain blocked.

## Strengths

- It requires both deterministic support completion and approximation-quality
  evidence.
- It preserves the existing SGU exact-nonlinear blocker:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

- It keeps SGU economics in `/home/chakwong/python` and BayesFilter generic.
- It explicitly blocks derivative, compiled, and HMC claims.

## Required Clarifications

1. The plan's state equations `(7,8,10,11)` correspond to zero-based canonical
   residual entries `H[7], H[8], H[10], H[11]`.
2. The SGU risk-premium equation uses current `zeta_t`, not next-period
   `zeta'`.
3. Gate A should preserve stochastic candidate coordinates `(a,zeta,mu)` and
   overwrite only `(d,k,r,riskprem)`.
4. Gate B must use the same previous-state grid and the same completed next
   states for linear and quadratic residual summaries.
5. Passing Gate A alone must not mint:

```text
sgu_quadratic_policy_residual_improvement_passed
sgu_combined_structural_approximation_target_passed
```

6. If Gate B fails, the result note should record the failed inequalities as
   evidence, not loosen the threshold after seeing the result.

## Missing Points Closed By Audit

- Add an explicit blocker label for the stricter user requirement if it fails:

```text
blocked_sgu_quadratic_policy_not_better_than_linear_residual
```

- Record whether the next phase is justified after each phase.  In particular,
  once Gate B fails, derivative, compiled, HMC, and BayesFilter promotion
  phases are not justified.
- Run all SGU tests CPU-only unless deliberately doing a GPU probe.  This
  avoids confusing the local CUDA sandbox policy with value-side residual
  evidence.

## Stop Rule

Execution should continue through Gate A implementation and Gate B diagnostic
testing.  Execution should stop before any combined SGU promotion if either of
the following is true:

```text
quadratic_rms >= linear_rms
quadratic_max > linear_max
```

## Audit Outcome

Approved for bounded execution with the above clarifications.  No BayesFilter
backend code change is justified by the plan unless the client exposes a
missing generic metadata abstraction; the expected BayesFilter work is
provenance and reset-memo documentation only.
