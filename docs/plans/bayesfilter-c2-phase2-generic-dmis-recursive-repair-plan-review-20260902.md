# Skeptical Review: C2 Phase 2 Generic-DMIS Repair Plan

Date: 2026-09-02  
Reviewed plan: `bayesfilter-c2-phase2-generic-dmis-recursive-repair-plan-20260902.md`  
Verdict: PASS FOR BOUNDED EXECUTION, with the explicit Stage D continuation condition

## Review question

Does the plan test the missing claim directly, use a valid target and measure,
contain an operational evidence contract, and stop without turning a proxy
diagnostic into evidence of a repaired recursive filter?

## Findings and disposition

### 1. Scope and stale context

The previous generic-DMIS execution ended at a one-step fixed-parent smoke,
while the prior C2 Stage 2 result ended at an independent-integration
precision veto. The new plan names both gaps and uses the preserved t=2,3,4
snapshots with fingerprint checks. Historical values are comparators only, not
new measurements.

Disposition: resolved in Sections 1, 3, and Stage A.

### 2. Target/measure contract

The snapshot evaluator returns the branch-summed quantity `E_t` whose
standard-normal expectation is its existing `z_t`. The plan explicitly defines
the Lebesgue target as `gamma_t=eta*E_t`; this prevents omitting or doubling
the reference density. Stage A requires an independent analytic eta-weighted
fixture before GPU work.

Disposition: pass, subject to the executable convention check.

### 3. Deterministic-bank semantics

Rows from separate component banks use base mass `alpha/N_j`, not iid mixture
weights. The plan freezes rows and reuses them for plain DMIS and the control
variate. Component permutation and mass-closure checks are required.

Disposition: pass.

### 4. Precision and arm disagreement

The original draft incorrectly made arbitrary cross-arm disagreement a hard
DMIS veto. Individual proposal arms can have different finite-sample variance.
The revised plan makes complete-DMIS replication precision and independent
target agreement the promotion screen; standard-normal, Student, and
Christoffel disagreement remains explanatory unless it survives the DMIS
precision check as a target-convention problem.

Disposition: repaired before execution.

### 5. Calibration leakage

The Student-mixture weight is no longer silently fixed as a default. The
predeclared calibration choices are `alpha={0.25,0.5,0.75}`; at most one is
selected on calibration rows and then frozen for untouched claim rows. No
claim-data retuning is allowed.

Disposition: repaired before execution.

### 6. Recursive call chain

The repository currently exposes external GH9 hints in the C2 engine, not a
generic recursive retained-moment callable. The plan therefore requires an
actual TensorFlow callable and a call-chain check. If it cannot be supplied
without changing the target convention, the C2 recursive stage must be
reported as a continuation veto and the same mechanics may be run only on an
analytic linear fixture. An external GH9 result may not be relabeled as a
recursive repair.

Disposition: explicit condition; no hidden assumption.

### 7. Baselines and interpretation

The ladder includes the stored Gram value, original integration arms, plain
DMIS, DMIS with the exact Gram control, and the recursive candidate. PF is
explicitly compatibility-only. ESS, shell residuals, conditioning, and RMS
are explanatory unless a later plan promotes them. The heuristic adversary
gate is conditional and remains a veto, not a tuning objective.

Disposition: pass.

### 8. Gradient and numerical policy

Rows, maps, masses, and discrete choices are frozen for the analytical tangent
check. The plan requires float64 TensorFlow, stable XLA signatures, GPU memory
growth, and a placement/manifest record for serious runs. CPU-only tests are
clearly classified as mechanics diagnostics.

Disposition: pass.

### 9. Budget and stop conditions

The row ladder, scramble ceiling, branch ceiling, GPU/CPU budgets, fresh
output roots, and hard vetoes are stated. Precision failure stops causal
interpretation rather than being converted into a candidate rejection.

Disposition: pass.

## Execution decision

The plan is sufficiently specific for a bounded execution of Stages A and B.
Stage C is conditional on the precision gate. Stage D is conditional both on
that gate and on an executable moment-map callable; this is a scientific
continuation condition, not permission to substitute a different algorithm.
No result from the campaign may be reported as a production repair or as
evidence of exact posterior inference.

## Residual risks to record in the result

* A stable DMIS estimate can still target the wrong physical coordinate if the
  independent convention check is weak.
* A CV estimate can be numerically positive while having large residual
  variance; residual moments must remain visible.
* Four to eight scrambles are a bounded precision pilot, not a universal
  variance guarantee.
* The C2 nonlinear recursive map may remain unimplemented after the frozen
  integration stage; that outcome must be labeled unresolved rather than
  silently promoted.
