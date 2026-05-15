# Audit: BayesFilter V1 Master And Subplans

## Date

2026-05-14

## Scope

Audited the updated V1 master and the fresh P1-P8 subplans:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p3-benchmark-refresh-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-plan-2026-05-14.md
docs/plans/bayesfilter-v1-p8-external-integration-plan-2026-05-14.md
```

## Audit Question

Does the revised program close the planning gap identified by the user: the
absence of a dedicated phase to validate analytic score and Hessian status for
SVD cubature, SVD-UKF, and SVD-CUT4 on the nonlinear model suite?

## Findings

### A1. Derivative Validation Is Now A First-Class Gate

Pass.

The master now makes P1 the next phase and requires a derivative-validation
matrix over Models A-C and SVD cubature, SVD-UKF, and SVD-CUT4.  The P1
subplan requires value status, score status, branch label, derivative
provider, finite-difference/reference evidence, compiled/eager parity where
available, and Hessian status for every model/backend cell.

This fixes the prior ambiguity where score parity and Hessian deferral were
described in separate places without a consolidated derivative-status matrix.

### A2. Hessian Work Is Not Accidentally Promoted

Pass.

The master and P1 require Hessian status, not Hessian implementation.  P5
remains the consumer gate for production nonlinear Hessians.  This preserves
the existing policy that nonlinear Hessians are deferred unless a concrete
Newton, Laplace, Riemannian HMC, or observed-information consumer is named.

### A3. Phase Ordering Is Safe

Pass.

The new order is:

1. P1 derivative-validation matrix;
2. P2 branch diagnostics;
3. P3 benchmark refresh;
4. P4 nonlinear HMC target/smoke;
5. P5 Hessian consumer assessment;
6. P6 optional GPU/XLA scaling;
7. P7 optional exact references;
8. P8 external integration.

This ordering prevents HMC/GPU/external work from starting before derivative
claims and branch boxes are explicit.

### A4. Model C Structural Contract Is Preserved

Pass.

The master and P1/P2 subplans explicitly require default Model C score claims
to use `allow_fixed_null_support=True`.  They also preserve the old collapsed
smooth branch as a blocker rather than silently treating it as score-ready.

### A5. Lane Boundaries Are Preserved

Pass.

The fresh subplans keep MacroFinance, DSGE, Chapter 18b, structural plan files,
and the shared monograph reset memo out of scope unless the user opens those
lanes explicitly.  P8 is a planning-only external integration phase, not a
client switch-over.

### A6. Evidence Boundaries Are Clear

Pass with one watch item.

The subplans distinguish:
- local finite-difference or exact affine score certification;
- dense projection diagnostics;
- testing-only autodiff oracles;
- branch-grid diagnostics;
- optional GPU/XLA timing diagnostics;
- optional Monte Carlo references.

Watch item:
When executing P1, the result artifact must be careful with older result files
that mention "fixed-null" language before the structural fixed-support
resolution.  The matrix should use the newer Chapter 18 structural
fixed-support terminology for default Model C.

### A7. Subplan Coverage

Pass.

Fresh subplans now exist for all remaining phases P1-P8.  Each subplan cites
the master, names entry gates, primary gates, veto diagnostics, and expected
artifacts.

## Required Tightenings Before Execution

Resolved during audit:
- updated the master phase table so P1-P8 point to the existing fresh subplans
  instead of saying those subplans still need to be created;
- updated the P1/P2 action text to execute the existing P1/P2 subplans.

No further tightening is required before P1.  P1 is ready to execute after
normal lane checks.

## Execution Recommendation

Begin with P1 only.  Use the master cycle:

```text
plan -> execute -> test -> audit -> tidy -> update V1 reset memo -> result artifact -> gate decision
```

Continue automatically to P2 only if:
- the derivative-validation matrix has no unknown score-status cells;
- every certified score cell cites reference evidence;
- every Hessian cell has explicit non-promotional status;
- default Model C structural rows use the fixed-support branch and null
  diagnostics.

## Drift Risks To Monitor

- Implementing nonlinear Hessians before P5 names a consumer.
- Treating branch-grid diagnostics as HMC readiness.
- Treating dense one-step projection as exact nonlinear likelihood evidence.
- Running GPU/XLA diagnostics before P2/P3 define stable benchmark boxes.
- Editing MacroFinance, DSGE, Chapter 18b, or shared monograph reset memos from
  the V1 lane.
