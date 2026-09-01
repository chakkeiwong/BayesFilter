# Phase 1: Single-Direction tf.while_loop Conversion — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`
**Phase**: 1
**Status**: NOT STARTED (blocked on Phase 0 completion)
**Created**: 2026-08-30
**Corrected**: 2026-09-01 (campaign authorization alignment)

---

## Objective

Convert the two Python loops in `canonical_batch_fused_value_score` to
`tf.while_loop` while keeping the single-direction tangent contract exactly as
it is today. The mathematical program does not change. Only the graph
representation changes: one bounded loop body replaces `horizon × substeps`
replicated subgraphs.

This phase deliberately does NOT touch the direction dimension. Phase 2 owns
that. Separating them means a parity failure in Phase 1 can only come from the
loop conversion, and a parity failure in Phase 2 can only come from the
multi-direction generalization.

---

## Preconditions

1. Phase 0 complete: API-drift repairs applied, coverage measured, fresh baseline captured
2. Phase 0 result document written with exact pass/fail counts and coverage percentages
3. Campaign authorization granted (ONE upfront authorization per master program)
4. Baseline graph-size numbers available for comparison (from Phase 0 or existing diagnostics)

---

## Current Loop Structure (verified by reading the kernel, 2026-08-30)

### Outer loop — `for time_index in range(horizon)`, line 146

Loop-carried state, exactly six tensors:

| Name | Shape | Role |
|---|---|---|
| `states` | `[m, dim]` | posterior cloud, becomes `children` at line 433 |
| `d_states` | `[m, dim]` | its parameter tangent |
| `covariances` | `[m, dim, dim]` | posterior covariance from S5, line 429 |
| `d_covariances` | `[m, dim, dim]` | its parameter tangent |
| `total` | `[batch]` | accumulated log-likelihood increment, line 348 |
| `d_total` | `[batch]` | accumulated score, line 349 |

Per-iteration reads: `observations[time_index]` (line 147),
`noises[time_index]` (line 148). Both become `tf.gather(·, t)`.

### Inner loop — `for step_index in range(substeps)`, line 219

Loop-carried state, exactly six tensors:

| Name | Shape | Role |
|---|---|---|
| `actual` | `[m, dim]` | flowed particle, line 281 |
| `d_actual` | `[m, dim]` | its tangent, line 284 |
| `auxiliary` | `[m, dim]` | flow anchor path, line 289 |
| `d_auxiliary` | `[m, dim]` | its tangent, line 292 |
| `log_det` | `[m]` | accumulated flow log-determinant, line 304 |
| `d_log_det` | `[m]` | its tangent, line 312 |

Per-iteration scalar: `lam = (step_index + 1) / substeps`, line 220.

Loop-invariant inside the inner loop, hoistable to the outer body:
`predicted_covs`, `d_predicted_covs`, `anchors`, `d_anchors`, `observation`,
`r_inv`, `eps`, `eye`. These are already computed before the inner loop and
merely closed over — the conversion must pass them as captured tensors, not as
loop variables.

### Values already hoisted outside both loops (no action needed)

`mean_w`, `cov_w`, `scale`, `eye`, `eps`, `process_chol`, `obs_chol`, `r_inv`,
`log_two_pi`, `theta_flat`, `dtheta_flat`. Phase 3 audits whether anything
remains.

---

## Design

### Inner loop conversion

```python
def _flow_body(s, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
    lam = (tf.cast(s, dtype) + 1.0) / tf.cast(substeps, dtype)
    # ... lines 221-312 verbatim, with `lam` now a tensor ...
    return (s + 1, new_actual, new_d_actual, new_aux, new_d_aux,
            log_det + ldet_inc, d_log_det + dldet_inc)

_, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
    cond=lambda s, *_: s < substeps,
    body=_flow_body,
    loop_vars=(tf.constant(0, tf.int32), pre_flow, d_pre_flow,
               anchors, d_anchors,
               tf.zeros([m], dtype), tf.zeros([m], dtype)),
    maximum_iterations=substeps,
    parallel_iterations=1,
)
```

`parallel_iterations=1` is required: the recursion is sequential, and allowing
speculative parallel iterations would let TensorFlow keep multiple iterations'
intermediates live, inflating peak memory for no benefit.

### Outer loop conversion

```python
def _step_body(t, states, d_states, covariances, d_covariances, total, d_total):
    observation = tf.gather(observations, t)
    noise = tf.tile(tf.gather(noises, t), [batch, 1])
    # ... S1, S2, inner while_loop, S4, S8, S5 ...
    return (t + 1, children, d_children, new_covariances, new_d_covariances,
            total + increment, d_total + d_increment)

_, states, d_states, covariances, d_covariances, total, d_total = tf.while_loop(
    cond=lambda t, *_: t < horizon,
    body=_step_body,
    loop_vars=(tf.constant(0, tf.int32), states, d_states,
               covariances, d_covariances, total, d_total),
    maximum_iterations=horizon,
    parallel_iterations=1,
)
```

### Shape invariants

Every loop variable has a fully static shape (`m`, `dim`, `batch`, `obs_dim`
are Python ints derived from input shapes). Declare `shape_invariants`
explicitly with those exact static shapes rather than relying on inference, so
an accidental rank or extent change fails at trace time instead of silently
becoming `None`.

---

## Two semantic differences this conversion introduces

Both are expected, both are within the declared rtol 5e-4, and both must be
recorded in the Phase 1 result note rather than discovered later.

**1. `lam` stops being a trace-time constant.** Today `lam` is
`tf.constant((step_index + 1) / substeps)` — a folded Python float. Inside the
loop body it becomes a computed tensor. Grappler can no longer constant-fold
`lam * php` (line 227) or `lam * d_php` (line 236). The float64 value is
identical; the op order around it is not. Expected effect: differences at the
1e-15 level, far inside tolerance.

**2. The Python `if model.observation_log_density_fn is not None` at line 319
stays a Python branch.** It tests a model attribute, not a tensor, so it
resolves at trace time inside the body closure. No `tf.cond` is needed and none
should be introduced. Both branches remain reachable across different model
configurations; the loop body traces exactly one of them per model.

---

## Verification

Run in this order. Each command is a single allowlisted wrapper invocation.

| Step | Command | Pass condition |
|---|---|---|
| 1 | `bash scripts/run_phase_tests.sh parity-fused` | 3/3 pass, rtol 5e-4 |
| 2 | `bash scripts/run_phase_tests.sh parity` | 6/6 pass |
| 3 | `bash scripts/run_phase_tests.sh score-suite` | no new failures vs Phase 0 baseline |
| 4 | `bash scripts/run_phase_tests.sh graph-size` | node count drops by ≥ 20× vs 110,628 |
| 5 | `bash scripts/run_phase_tests.sh eval-time` | trace+first < 20 s; warm ≤ 0.30 s |
| 6 | `bash scripts/run_phase_tests.sh coverage` | coverage ≥ Phase 0 baseline |

Step 4 is the phase's reason for existing. The prediction is explicit: one loop
body of roughly the current per-timestep size (~2,200 nodes) plus a bounded
outer body, so low thousands of nodes rather than 110,628. If the count does
not drop by at least 20×, the conversion did not achieve its purpose and that
is a reportable outcome, not something to tune around.

---

## Repair scope for this phase

Repairable without asking:

- Loop-variable ordering or packing mistakes
- Missing or over-tight `shape_invariants`
- Tangent accumulation written as in-place `+=` on a captured tensor instead of
  a returned loop variable
- `lam` cast or dtype slips
- Variables that should be captured constants but were threaded as loop vars
  (or the reverse)
- Parity failures whose measured error is below rtol 5e-4
- Docstring and comment updates describing the new loop structure

Not repairable — these are stop conditions:

- Parity error above rtol 5e-4 after the loop-variable wiring is confirmed correct
- Any test in the score suite that passed in Phase 0 and now fails
- Node count that does not drop, or drops by less than 20×
- Warm eval time above 0.60 s (worse than the already-rejected naive batched form)
- An operation that cannot be expressed inside `tf.while_loop`

---

## Deliverables

1. `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` with both loops converted
2. `ledh-while-loop-refactor-phase1-result-2026-08-30.md` recording: measured
   parity errors per test, node count / GraphDef bytes / trace time / warm eval
   before and after, coverage before and after, the two semantic differences
   above with their measured effect, and the run manifest (commit, command,
   conda env, CPU-only status, wall time)
3. Semantic commit
4. Phase 2 precondition check

---

## Success criteria

- [ ] Both Python loops replaced by `tf.while_loop` with `parallel_iterations=1`
- [ ] Explicit static `shape_invariants` on both loops
- [ ] `parity-fused` 3/3, `parity` 6/6
- [ ] `score-suite` shows no regression against the Phase 0 baseline
- [ ] Node count reduced ≥ 20×
- [ ] Trace+first < 20 s, warm ≤ 0.30 s
- [ ] Coverage ≥ Phase 0 baseline
- [ ] Result note written with the run manifest
- [ ] Committed

---

## Next

[Phase 2 subplan](ledh-while-loop-refactor-phase2-subplan-2026-08-30.md) —
multi-direction tangent.
