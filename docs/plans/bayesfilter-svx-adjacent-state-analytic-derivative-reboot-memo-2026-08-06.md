# SVX adjacent-state analytic derivative reboot memo

Date: 2026-08-06
Status: `REBOOT_RECOVERY_NOTE`

## Purpose

This memo is the handoff point for the next agent after reboot. The current task
is no longer route selection; it is implementing the **analytic adjacent-state
derivative for the active SVX-ZC finite program** and then rerunning focused
tests before any serious SVX tuning.

## Primary restart files

Read these first, in order:

1. `docs/plans/bayesfilter-svx-adjacent-state-analytic-derivative-derivation-2026-08-06.md`
2. `docs/plans/glistening-jumping-stroustrup.md`
3. `docs/plans/bayesfilter-svx-analytic-route-diagnostic-result-2026-08-06.md`

## Current diagnosis

We have established by source tracing and probe tests that:

- the active `SVX-ZC` NeuTra target still uses the autodiff score backend in
  `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`;
- the current active value program is materially **adjacent-state frozen-core**
  dependent starting at `t=1`;
- the nearby transformed-SV analytic route in `sv_mixture_cut4.py` is **not**
  a drop-in replacement because its values diverge from the active route after
  the adjacent-state branch activates;
- therefore the missing serious-route artifact is an **analytic adjacent-state
  derivative for the same active frozen-core finite program**.

## What is already done

### Plans / notes

- A derivation note has been written:
  - `docs/plans/bayesfilter-svx-adjacent-state-analytic-derivative-derivation-2026-08-06.md`
- A diagnostic result note has been written:
  - `docs/plans/bayesfilter-svx-analytic-route-diagnostic-result-2026-08-06.md`
- The working implementation plan is captured in:
  - `docs/plans/glistening-jumping-stroustrup.md`

### Code and tests already touched

- An analytic SVX adapter prototype exists in:
  - `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py`
- A focused test file exists in:
  - `tests/test_zhao_cui_actual_sv_neutra_target.py`

These are **not yet correct** as a final solution; the earlier attempts showed
that the analytic candidate route was the wrong target and that the active
adjacent-state derivative is still missing.

### Registry state

The registry was restored to the last known-good active target during the
analysis phase. The next agent should be careful not to switch the registry to
an analytic candidate until the new backend passes tests.

## What the traces showed

### Active route sensitivity

The current active value program behaves like this:

- zeroing the one-axis initial core leaves `T=1` and `T=2` values unchanged;
- zeroing the adjacent frozen cores leaves `T=1` unchanged but sends `T=2` to
  `-inf`.

This means the meaningful frozen-core dependence is the **adjacent-state UKF
cores**, not the one-axis initial core.

### Candidate analytic route mismatch

The transformed-SV analytic candidate showed the following value gaps against the
current active route on the same probe point:

- `T=1`: small gap (~1.7e-2)
- `T=2`: gap appears (~9.2e-2)
- `T=3`: persists (~1.0e-1)
- `T=10`: grows (~5.0e-1)

So the candidate analytic route is mathematically **different** from the active
finite program.

## What to do next

The next implementation step is:

1. keep the active SVX value program unchanged;
2. replace only the score backend in the active module;
3. implement the **analytic adjacent-state derivative** for the current active
   frozen-core finite program;
4. add focused finite-difference tests against the same active value program;
5. only then switch the serious adapter / registry wiring;
6. only after those tests pass, rerun `SVX-ZC` common tuning on the 4080-only
   mask.

## Files likely to change

Primary implementation target:

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`

Likely integration updates after tests pass:

- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
- `bayesfilter/testing/neutra_model_registry_tf.py`

Reused helper references:

- `bayesfilter/highdim/filtering.py`
- `bayesfilter/highdim/derivatives.py`
- `bayesfilter/highdim/zhao_cui_fixed_adjacent_tt_tf.py`
- `bayesfilter/highdim/zhao_cui_moment_teacher_als.py`

## Verification checklist for the next agent

1. Implement the analytic adjacent-state derivative in the active SVX module.
2. Run the focused SVX tests until they pass.
3. Re-run shared-procedure and end-to-end contract tests.
4. Run a tiny smoke call on a probe batch.
5. If and only if those pass, rerun `SVX-ZC` common tuning on the 4080-only mask.

## Working-tree caution

There are many unrelated local modifications and untracked files in the repo.
Do **not** mass-stage, clean, or revert unrelated work. Only touch the SVX
adjacent-state analytic derivative path and the tests/registry files needed for
that fix.
