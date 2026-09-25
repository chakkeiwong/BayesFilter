# Phase 2: Multi-Direction Tangent — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`
**Phase**: 2
**Status**: NOT STARTED (blocked on Phase 1)
**Created**: 2026-08-30

---

## Correction to the master program

The master program specifies tangent state `[m, K, dim]`. **That design is not
implementable against the current model contract**, and this subplan replaces
it. The correction is recorded here rather than silently applied.

`PerPointScoreModel` holds user-supplied callbacks whose tangent arguments are
rank-2:

```python
transition_mean_tangent_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
observation_tangent_fn: Callable[[Tensor, Tensor], Tensor]
observation_log_density_tangent_fn: Callable[[...], Tensor] | None
```

Inserting a leading K axis changes the signature every model factory and test
fixture implements. That is a breaking change to the public model contract,
which the refactor contract places out of scope.

Two candidate designs, and why each is rejected or chosen:

**(a) Rank-3 tangents `[m, K, dim]` through the callbacks.** Rejected. Breaks
the model contract for every existing caller, which the contract forbids.

**(b) Fold K into the point axis: reshape `[m, K, dim] → [m*K, dim]` and tile
the primal.** Rejected on measured evidence. This is the batched-direction form
already diagnosed in
`docs/benchmarks/diagnose_direction_cost_scaling_20260830.py`: graph size 1.00×
but arithmetic 3.51× for 5 directions, because tiling the primal defeats
Grappler CSE. Warm eval measured 0.485 s against 0.185 s for the swept form.
Rebuilding the rejected arm is not progress.

**(c) K tangent evaluations sharing one primal inside the loop body. CHOSEN.**
The loop body computes the primal once and evaluates the tangent recursion K
times against it. Because all K tangent evaluations live in the same
`tf.while_loop` body as the single shared primal, the sharing is structural —
it does not depend on Grappler recognizing a common subexpression across
separately traced calls.

Cost model for (c), stated as a prediction to be tested, not a claim:

- Graph: one body containing 1 primal + K tangents. Bounded, roughly linear in
  K, and small in absolute terms because the body is not replicated over
  `horizon × substeps`.
- Arithmetic: 1 primal + K tangents per iteration. The K-fold tangent work is
  irreducible — it is the gradient. What (c) avoids is the K-fold *primal*
  repetition of the swept form and the tiling of form (b).
- Calls per gradient: 1, replacing the current 6.

---

## Objective

Generalize the tangent recursion from one direction to K directions in a single
call, so a P-parameter gradient costs one traced call instead of P+1.

---

## Preconditions

1. Phase 1 complete: both loops converted, parity 6/6, node count reduced ≥ 20×
2. Phase 1 result note recorded, including measured warm eval time as the Phase 2 baseline
3. Phase 1 committed

---

## API change

```python
def canonical_batch_fused_value_score(
    model, theta, theta_directions,
    initial_states, initial_covariances, noises, observations,
    *, substeps: int, jitter: float = 1.0e-12,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
```

| | Before (Phase 1) | After (Phase 2) |
|---|---|---|
| `theta_directions` | `[B, P]` — one direction per row | `[B, K, P]` — K directions per row |
| returned score | `[B]` | `[B, K]` |
| `program_valid` | `[B]` | `[B]` (unchanged; a row is valid iff value and all K tangents are finite) |

Backward compatibility: accept rank-2 `theta_directions` by promoting to
`[B, 1, P]` and squeezing the returned score back to `[B]`. This keeps the three
existing fused parity tests valid without edits, which preserves them as
independent evidence that Phase 2 did not perturb the K=1 path.

The rank promotion is decided from `theta_directions.shape.rank` at trace time,
so it is a Python branch, not a `tf.cond`.

---

## Tangent state changes

Every tangent tensor gains a leading K axis at the *loop-variable* level, while
the tensors handed to model callbacks stay rank-2.

| Tangent | Phase 1 shape | Phase 2 shape |
|---|---|---|
| `d_states` | `[m, dim]` | `[K, m, dim]` |
| `d_covariances` | `[m, dim, dim]` | `[K, m, dim, dim]` |
| `d_total` | `[batch]` | `[K, batch]` |
| `d_actual`, `d_auxiliary` | `[m, dim]` | `[K, m, dim]` |
| `d_log_det` | `[m]` | `[K, m]` |
| `dtheta_flat` | `[m, P]` | `[K, m, P]` |

K leads rather than trails so that `d[k]` is a contiguous rank-2 slice matching
exactly what the Phase 1 code expected. The per-direction inner computation is
then the Phase 1 code unchanged, applied to slice `k`.

Primal tensors are untouched: `states`, `covariances`, `total`, `actual`,
`auxiliary`, `log_det` keep their Phase 1 shapes.

---

## Implementation approach

Structure each tangent stage as a Python-level loop over `k` inside the loop
body, gathering the k-th tangent slice, running the Phase 1 tangent arithmetic
verbatim, and stacking:

```python
d_out = tf.stack([
    _tangent_stage_k(primal_shared, d_in[k], dtheta_flat[k])
    for k in range(k_count)
], axis=0)
```

`k_count` is a Python int from `theta_directions.shape[1]`, so this unrolls K
times inside one bounded body. That is the intended structure: K is small
(equal to P, the parameter count), the body is traced once, and the primal is
computed once outside the k-loop.

This is not the graph-unrolling problem Phase 1 fixed. That problem was
replication over `horizon × substeps` — up to 1,200 repetitions of a
2,200-node body. Here K is the parameter count, single digits in every current
use, and it multiplies only the tangent arithmetic inside a body that is itself
no longer replicated.

Where a tangent stage is a plain linear map over the point axis (`gaussian_log_and_tangent`,
`chol_diff`, the QR trace term), the k-loop can be replaced by a batched call
over a `[K*m, ...]` reshape *without* touching the model callbacks, because
those are internal helpers, not user-supplied. Do this only where it is
provably the same arithmetic; do not do it across the model-callback boundary.
Record which stages were batched and which stayed k-looped.

---

## Sequencing within the phase

Land these in order, testing after each. Each step is independently
verifiable, so a parity failure localizes to one step.

1. Rank promotion and squeeze-back for `theta_directions`, K forced to 1. Parity
   tests must pass unchanged — this proves the plumbing is neutral before any
   tangent shape changes.
2. Tangent state reshaped to leading-K with K still 1. Parity unchanged.
3. K > 1 enabled. New test: K=P in one call must match P swept single-direction
   calls to within rtol 5e-4.
4. Selective batching of internal tangent helpers. Parity must not move.

---

## New tests

Add to `tests/highdim/test_ledh_canonical_batch_fused.py`:

**`test_fused_multi_direction_matches_swept`** — build a P=3 fixture, call once
with `theta_directions` of shape `[B, 3, P]` holding the three basis
directions, and separately make three single-direction calls. Assert
`|score[:, k] - swept_k| < 5e-4 × max(|swept_k|, 1.0)` for each k. This is the
phase's primary correctness gate.

**`test_fused_multi_direction_rank_two_backward_compatible`** — pass rank-2
`theta_directions`, assert the returned score has shape `[B]` and equals the
Phase 1 result bit-for-bit where the arithmetic is unchanged, or within 1e-12
where reshapes reorder reductions.

**`test_fused_multi_direction_graph_compilable`** — `tf.function(autograph=False)`
with K=3, assert score shape `(B, 3)` and all finite.

---

## Verification

| Step | Command | Pass condition |
|---|---|---|
| 1 | `bash scripts/run_phase_tests.sh parity-fused` | all pass including the 3 new tests |
| 2 | `bash scripts/run_phase_tests.sh parity` | all pass |
| 3 | `bash scripts/run_phase_tests.sh score-suite` | no new failures vs Phase 1 |
| 4 | `bash scripts/run_phase_tests.sh graph-size` | K=5 node count ≤ 3× the K=1 count |
| 5 | `bash scripts/run_phase_tests.sh direction-cost` | one K=5 call cheaper in warm eval than 6 swept calls |
| 6 | `bash scripts/run_phase_tests.sh coverage` | coverage ≥ Phase 1 baseline |

Step 5 is the phase's reason for existing and must be measured, not assumed.
The comparison is one K=5 call against the current 6-call pattern
(`step1_true_surrogate_force.py`: 1 value + 5 swept directions).

---

## Repair scope

Repairable without asking:

- Axis-ordering errors in the K dimension
- Stack/unstack and reshape slips
- `shape_invariants` needing the K extent added
- Rank-promotion branch mistakes
- A batched internal helper that changes results — revert that helper to k-looped
- Parity errors below rtol 5e-4

Not repairable — stop conditions:

- K=P multi-direction disagreeing with P swept calls above rtol 5e-4 after axis
  wiring is confirmed
- Node count for K=5 exceeding 3× the K=1 count
- One K=5 call slower in warm eval than 6 swept calls, i.e. the phase's premise
  measured false
- A tangent stage that cannot be expressed without changing a model callback
  signature

That third condition deserves emphasis. If measurement shows form (c) does not
beat sweeping, the honest outcome is a result note reporting that the
multi-direction premise failed, with the swept form retained on top of the
Phase 1 loop conversion. Phase 1's gains are independent of Phase 2 and survive
a Phase 2 rejection. Do not respond to that measurement by reaching for form
(b), which is already rejected on evidence.

---

## Deliverables

1. Kernel with multi-direction tangent support and rank-2 backward compatibility
2. Three new tests
3. `ledh-while-loop-refactor-phase2-result-2026-08-30.md`: measured K=P vs swept
   parity errors, node count and warm eval as functions of K, the one-call vs
   six-call comparison, which internal helpers were batched vs k-looped,
   coverage, run manifest
4. Semantic commit
5. Phase 3 precondition check

---

## Success criteria

- [ ] `theta_directions` accepts `[B, K, P]` and rank-2 `[B, P]`
- [ ] Score returns `[B, K]`, or `[B]` under rank-2 input
- [ ] K=P in one call matches P swept calls within rtol 5e-4
- [ ] All pre-existing parity tests pass without edits
- [ ] K=5 node count ≤ 3× K=1
- [ ] One K=5 call cheaper than 6 swept calls in warm eval
- [ ] Coverage ≥ Phase 1 baseline
- [ ] Result note with run manifest
- [ ] Committed

---

## Next

[Phase 3 subplan](ledh-while-loop-refactor-phase3-subplan-2026-08-30.md) —
hoisting and constant precomputation.
