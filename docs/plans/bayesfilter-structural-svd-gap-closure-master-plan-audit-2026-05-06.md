# Audit: structural SVD remaining-gap closure master plan

## Date

2026-05-06

## Plan audited

`docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md`

## Audit stance

This audit treats the plan as if it was handed off by another developer.  The
question is whether the plan is complete, ordered correctly, and safe to
execute under the requested:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

cycle.

## Current execution context

BayesFilter is ahead of `origin/main` and has scoped uncommitted planning
changes plus unrelated dirty/untracked files.  The unrelated files must remain
outside this pass.

The DSGE client at `/home/chakwong/python` is ahead of its origin at:

```text
59c05f5 Close Rotemberg structural completion gate
```

That newer client commit changes the first substantive phase from fresh
Rotemberg implementation into validation of existing Rotemberg closure
evidence.

## Findings

### 1. Dependency order is correct

The plan correctly orders the scientific gates:

```text
model semantics
  -> residual evidence
  -> derivative/Hessian certification
  -> compiled parity
  -> HMC diagnostics
  -> provenance cleanup
```

This prevents the major failure mode from the earlier cycle: interpreting HMC
or gradient symptoms before the target definition is structurally valid.

### 2. Ownership boundaries are correct

The plan keeps DSGE economics in `/home/chakwong/python`, MacroFinance
economics in `/home/chakwong/MacroFinance`, and BayesFilter generic.  This is
essential.  Rotemberg, SGU, and EZ completion maps must not be encoded in
BayesFilter.

### 3. Rotemberg baseline needs current-state interpretation

The plan records the historical blocker:

```text
blocked_pruned_second_order_dy_identity_residual
```

That label remains valid for the raw uncompleted pruned path, but the DSGE
client now includes a committed helper and tests with the positive label:

```text
rotemberg_second_order_dy_completion_residual_passed
```

Execution should validate the committed evidence and update BayesFilter
provenance, not reimplement Rotemberg.

### 4. SGU stop rule is necessary

The plan correctly makes SGU the next structural decision.  The latest DSGE
client reset memo records that a four-state solve can close selected SGU
equations `(7,8,10,11)`, but cannot close the full nonlinear canonical
residual.  Therefore a code implementation is not justified until the target
is chosen:

- state-identity completion only;
- joint state-control nonlinear projection;
- perturbation-policy residual target.

### 5. EZ is independent but should not jump the queue automatically

EZ timing work is intellectually separable from SGU, but the requested
execution protocol says to stop when the next phase is not justified.  Since
the active DSGE reset memo explicitly stops after the SGU target-definition
decision, automatic EZ metadata promotion should not proceed in this pass.

### 6. Derivative, compiled, and HMC phases are correctly gated

The derivative/Hessian phase must wait for a promotable target or remain a
generic toy-fixture policy exercise.  Compiled parity can begin on exact LGSSM
or generic fixtures, but any DSGE production/HMC claim must wait for the same
model/backend pair to have residual and derivative evidence.  HMC remains
downstream.

## Missing points added by this audit

- Treat the master-plan blocker table as the historical BayesFilter baseline,
  not as an assertion about the newest DSGE client HEAD.
- Record the newer Rotemberg closure commit in BayesFilter reset-memo
  provenance.
- Stop after SGU validation unless the user chooses the SGU structural target.
- Do not run EZ, derivative, compiled parity, or HMC phases automatically after
  the SGU stop rule.
- Commit only BayesFilter planning/provenance files in this pass.  The DSGE
  client already has its own commit and has a dirty reset memo that must not be
  overwritten.

## Verdict

The plan is safe to execute through Phase 2 as a validation/provenance pass.
Phase 3 and later are not automatically justified unless the SGU target choice
is supplied.  If Phase 2 reproduces the current SGU blocker evidence, the
correct action is to stop, commit the BayesFilter provenance update, and ask
for direction on the SGU target.
