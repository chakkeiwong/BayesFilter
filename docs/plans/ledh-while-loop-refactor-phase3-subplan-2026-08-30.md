# Phase 3: Hoisting and Constant Precomputation — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`
**Phase**: 3
**Status**: NOT STARTED (blocked on Phase 2)
**Created**: 2026-08-30

---

## Objective

Remove work that is repeated per loop iteration but does not depend on the loop
index, and remove code that does nothing. After Phase 1 the loop bodies are
traced once, so hoisting no longer reduces *graph size* much — it reduces
*runtime arithmetic*, which is now the binding cost.

This phase is bounded and low-risk by construction: every change is either
deleting dead code or moving an invariant computation earlier. None of it
changes the arithmetic performed on live values. Any parity movement beyond
float64 reassociation noise means a genuine mistake, not a tolerance question.

---

## Preconditions

1. Phase 2 complete (or Phase 2 rejected on measured evidence with the swept form retained — Phase 3 applies either way)
2. Phase 2 result note recorded, with warm eval time as the Phase 3 baseline
3. Committed

---

## Work items

### 3.1 Dead code removal

`r_inv_obs = tf.linalg.matvec(r_inv, observation)` at line 214 (Phase 1
numbering) is computed every timestep and never referenced. Verified by
grepping the identifier across the module: one assignment, zero reads.

Delete it. Before deleting, grep the whole worktree for `r_inv_obs` to confirm
nothing outside the module reaches in for it.

### 3.2 Closure hoisting

Two helpers are defined inside `canonical_batch_fused_value_score`:

- `gaussian_log_and_tangent(points, d_points, means, d_means, chol)` — captures `m`, `dtype`, `log_two_pi`
- `chol_diff(chol, d_matrix)` — captures nothing beyond its arguments

`chol_diff` is trivially hoistable to module level as-is; it is pure in its
arguments. Do that.

`gaussian_log_and_tangent` needs `m`, `dtype`, and `log_two_pi` passed
explicitly. Two sub-items inside it are worth attention:

- `log_norm` (the `k·log 2π + 2 Σ log diag(chol)` term) depends only on `chol`.
  For the transition and observation densities the Cholesky factor is
  loop-invariant (`process_chol`, `obs_chol` are computed once at setup), so
  `log_norm` is recomputed every timestep for a value that never changes.
  Precompute it per fixed Cholesky at setup and pass it in. For the *proposal*
  density the factor depends on the predicted covariance and therefore on the
  loop index — that call site must keep computing `log_norm` inside the loop.
  Distinguish these two call classes explicitly; do not hoist the loop-dependent one.
- `tf.broadcast_to(chol, [m, *chol.shape])` materializes an `[m, k, k]` tensor
  from a `[k, k]` factor purely to satisfy `triangular_solve`'s batch shape.
  Check whether `triangular_solve` broadcasts the `[k, k]` factor against an
  `[m, k, 1]` right-hand side without the explicit expansion. If it does, drop
  the `broadcast_to` — at `m = batch·N = 6·252 = 1512` and `dim = 5` this is a
  1512×5×5 float64 materialization per call site per timestep. If it does not
  broadcast, leave it and record that it was checked.

### 3.3 Setup-block audit

Walk every statement in the setup block (Phase 1 numbering, lines 84–113) and
classify each as: already correctly hoisted / hoistable further / must stay.
Record the table in the result note. Known items: `mean_w`, `cov_w`, `scale`,
`eye`, `eps`, `process_chol`, `obs_chol`, `r_inv`, `log_two_pi`, `theta_flat`,
`dtheta_flat`.

`_unscented_weights` builds `mean_w` and `cov_w` via `tf.concat` of Python lists
of length `2·dim+1`. It is called once per invocation, so this is not a hot
path, but it produces `2·dim+1`-element constants through concat nodes rather
than a single constant. Replace with a direct `tf.constant` construction if it
is clean to do so; skip if it obscures the weight derivation, which is
correctness-bearing and should stay readable.

### 3.4 Repeated-pattern extraction

The sequence "symmetrize, scale, add jitter, Cholesky" appears twice with
identical structure — once in S1 predict, once in S5 update. Extract a
module-level helper only if the two sites are genuinely identical after reading
both. The innovation Cholesky and the observation-covariance Cholesky are
*similar but not identical* (one omits the symmetrization, one applies jitter to
an identity rather than to the covariance), so they must not be folded into the
same helper. Verify by reading, not by pattern-matching the shape of the code.

This item is optional. It reduces duplication but does not reduce arithmetic.
If reading shows the sites differ in any respect, skip it and record why.

### 3.5 Loop-invariant hoisting out of the inner loop

Confirm that `predicted_covs`, `d_predicted_covs`, `anchors`, `d_anchors`,
`observation`, `r_inv`, `eps`, `eye` enter the inner `tf.while_loop` as captured
tensors and not as loop variables. A captured tensor is computed once per outer
iteration; a loop variable is threaded through every inner iteration and
prevents some optimizations. Phase 1 should have done this correctly; Phase 3
verifies it.

---

## Verification

| Step | Command | Pass condition |
|---|---|---|
| 1 | `bash scripts/run_phase_tests.sh parity-fused` | all pass, rtol 5e-4 |
| 2 | `bash scripts/run_phase_tests.sh parity` | all pass |
| 3 | `bash scripts/run_phase_tests.sh score-suite` | no new failures vs Phase 2 |
| 4 | `bash scripts/run_phase_tests.sh eval-time` | warm eval ≤ Phase 2 baseline |
| 5 | `bash scripts/run_phase_tests.sh graph-size` | node count ≤ Phase 2 |
| 6 | `bash scripts/run_phase_tests.sh coverage` | coverage ≥ Phase 2 baseline |

Step 4's pass condition is "no worse", not "better". Hoisting invariants out of
a loop cannot increase arithmetic; if warm eval regresses, something was moved
that was not actually invariant, and that is a correctness signal.

Because every item here is deletion or motion of invariant work, parity errors
should be identical or differ only by float64 reassociation. An error that
grows into the 1e-6 range or beyond means a genuinely loop-dependent value was
hoisted. Treat that as a bug to find, not a tolerance to widen.

---

## Repair scope

Repairable without asking:

- Reverting any individual hoist that moves parity
- Argument-threading mistakes in the hoisted helpers
- Skipping any optional item (3.4, the `_unscented_weights` rewrite, the
  `broadcast_to` removal) with a recorded reason
- Import and `__all__` adjustments for module-level helpers

Not repairable — stop conditions:

- Parity error above rtol 5e-4 after all hoists are individually reverted and re-applied
- Warm eval regression that persists after reverting every optional item
- A hoist that appears necessary for performance but demonstrably changes results

Phase 3 has an unusual property worth stating: **every item is individually
skippable**. If any single hoist causes trouble, drop that item, record it, and
continue. The phase cannot block the program.

---

## Deliverables

1. Kernel with dead code removed, helpers at module level, invariants hoisted
2. `ledh-while-loop-refactor-phase3-result-2026-08-30.md`: the setup-block
   classification table, which items were applied vs skipped and why, parity
   errors, warm eval and node count before/after, coverage, run manifest
3. Semantic commit
4. Phase 4 precondition check

---

## Success criteria

- [ ] `r_inv_obs` removed, absence of external readers verified
- [ ] `chol_diff` at module level
- [ ] `gaussian_log_and_tangent` at module level with explicit arguments
- [ ] `log_norm` precomputed for fixed Cholesky factors; loop-dependent site left alone
- [ ] `broadcast_to` necessity checked and the finding recorded
- [ ] Setup-block classification table recorded
- [ ] Inner-loop captured-vs-loop-variable status verified
- [ ] All parity and score tests pass
- [ ] Warm eval ≤ Phase 2, node count ≤ Phase 2
- [ ] Coverage ≥ Phase 2 baseline
- [ ] Result note with run manifest
- [ ] Committed

---

## Next

[Phase 4 subplan](ledh-while-loop-refactor-phase4-subplan-2026-08-30.md) —
integration and surrogate-force HMC readiness.
