# LEDH Single-Authority Engine Unification — Phase Plan

**Phase ID:** `ledh-engine-unification-2026-09-04`
**Date:** 2026-09-04
**Status:** DRAFT — awaiting owner sequencing decision
**Parent program:** `ledh-surrogate-force-hmc-master-program-2026-09-04.md`
**Position:** new Phase 2.5, between Phase 2 and Phase 3

---

## Why this phase exists

The LEDH canonical algorithm is implemented twice. `ENTRY_POINTS`
(`ledh_alg1_contract.py`) registers four lanes, but only two contain
mathematics:

| Lane | File | LOC | What it is |
|---|---|---:|---|
| `single_cloud` | `ledh_canonical_score_tf.py` | 617 | full engine |
| `batch_fused` | `ledh_canonical_batch_fused_tf.py` | 736 | full engine, independent reimplementation |
| `batch` | `ledh_canonical_batch_tf.py` | 94 | Python row loop over `single_cloud`; no arithmetic |
| `neutra_target` | `ledh_canonical_neutra_targets_tf.py` | — | thin forwarder to `batch_fused`; no arithmetic |

(`ledh_canonical_filter_tf.py`, 476 LOC, is the value-lane entry point and
carries the reset call plus the diagnostic payload.)

So ~1,350 lines express the same recursion twice. Every stage — UKF predict
tangent, anchor chaining, flow map and log-det tangent, PF-PF weight tangent,
UKF update tangent, softmax accumulation — exists in two hand-derived copies
that must be kept numerically identical by test discipline alone.

### Measured consequences of the fork

These are not hypotheticals; each was hit in the last ten days.

1. **Capability divergence.** `batch_fused` cannot run the production program.
   Its signature is `substeps` and `jitter` only — no `reset_policy`, no
   dual-cap, no trust-region. `LEDH_PRODUCTION_PROGRAM_V1` requires all three.
   The single-cloud lane exposes 14 such controls.
2. **Tuning artifacts stranded.** The 2026-09-03 trust-region campaign
   (956 s, 4 models, 108 evaluations) selected controls
   (`higher_moment_correction_steps: 4`, `higher_moment_trust_radius: 0.1`,
   `coordinatewise_standardized_cap: 0.98`, `epsilon: 2.0`, `balance_steps: 8`)
   that the fused lane has no parameter to receive.
3. **Parity gates pin the wrong route.** The 6 fused parity tests call the
   authority without `reset_policy`, i.e. against the slice the authority's own
   docstring marks *diagnostic-only, NOT a log-likelihood estimator for T > 1*
   (measured score bias +4.2 at T=50). Passing them says the fused lane
   reproduces the authority's diagnostic mode at `horizon=3`. It says nothing
   about the production estimand.
4. **Registry drift.** The August 29 audit found `ENTRY_POINTS` naming
   `canonical_value_score_and_diagnostics` while the module exports
   `canonical_value_and_analytical_score`.
5. **Model-contract drift.** Two incompatible model dataclasses, so every model
   must be written twice to be usable in both lanes.
6. **Policy divergence.** The pfor violation repaired on 2026-09-04 existed in
   the fused lane only; the single-cloud lane never had it. One engine, one
   audit surface.

### The strategic payoff

This phase **subsumes the Phase 3 blocker**. The recorded option A ("port
Contract-E reset + dual-cap + trust-region into the fused lane") is the bulk of
this work. Unifying onto one engine carrying the union of capabilities means
Phase 3 acquires a lane that is simultaneously batch-native, K-direction, and
production-conformant — instead of choosing between fast-but-wrong-estimand and
correct-but-slow.

---

## Scope

### In scope

- One engine implementing the canonical LEDH value + multi-direction analytical
  score, parameterised by batch size (B=1 is not a special case), direction
  count K, and the full reset/correction control family.
- One model contract replacing `NonlinearScoreModel` and `PerPointScoreModel`.
- `single_cloud`, `batch`, `batch_fused`, `neutra_target` retained as **thin
  adapters** over the one engine, preserving their public signatures.
- Characterisation-test suite covering the capability union, including the
  reset family, which currently has no batch-lane coverage.
- `ENTRY_POINTS` registry repaired and its discovery guard extended to fail on
  a lane that does not resolve to the unified engine.

### Explicitly out of scope

- Any change to the mathematics. This phase is behaviour-preserving by
  definition; a numeric change is a defect, not an improvement.
- New capability beyond the union of what the two engines already support.
- Retuning. If the phase succeeds, tuning artifacts remain scope-valid.
- Phase 3 execution, XLA promotion, dtype policy change, performance work.

### The hard constraint

Per the LEDH Per-Scope Tuning Rule, any changed bound field is a new tuning
scope. If unification perturbs numerics on the `contract_e` route, the
2026-09-03 tuning is invalidated and owed again. Therefore:

> **Behaviour preservation on the `contract_e` route is a promotion veto, not a
> success metric.** Refactoring and capability addition must never appear in the
> same commit.

---

## Capability union to be supported

### Model contract delta

`NonlinearScoreModel` (13 fields) vs `PerPointScoreModel` (9 fields):

| Field | single_cloud | batch_fused |
|---|---|---|
| `transition_mean_fn` | ✅ | ✅ |
| `transition_mean_tangent_fn` | ✅ 3-arg (direction closed over) | ✅ 4-arg (`d_theta` explicit) |
| `observation_fn` / `observation_jacobian_fn` / `observation_tangent_fn` | ✅ | ✅ |
| `process_covariance` / `observation_covariance` | ✅ | ✅ |
| `observation_log_density_fn` | ✅ | ✅ |
| `observation_log_density_tangent_fn` | ✅ 4-arg | ✅ 5-arg |
| `process_covariance_tangent_fn` | ✅ | ❌ |
| `observation_covariance_tangent_fn` | ✅ | ❌ |
| `transition_log_density_fn` | ✅ | ❌ |
| `transition_log_density_tangent_fn` | ✅ | ❌ |

Unified contract takes the **explicit-direction arity** (fused convention:
`d_theta` passed, not closed over), because closing the direction into the
callback is what makes K-direction batching impossible. Single-cloud adapters
therefore need their tangent callbacks rewritten from 3-arg to 4-arg.

Construction sites to migrate (small, verified):

- `NonlinearScoreModel(`: 1 library file (`ledh_canonical_models_tf.py`),
  3 test files.
- `PerPointScoreModel(`: 1 library file (`ledh_canonical_neutra_targets_tf.py`),
  2 benchmark files, 1 test file.

### Execution capability

| Capability | single_cloud | batch_fused | unified |
|---|---|---|---|
| Batch-native (no Python row loop) | ❌ | ✅ | ✅ |
| K directions sharing one primal | ❌ | ✅ | ✅ |
| `tf.while_loop` bounded horizon/substep | ❌ (Python unroll) | ✅ | ✅ |
| Contract-E reset (S6) | ✅ | ❌ | ✅ |
| Dual-cap (pairwise + coordinate) | ✅ | ❌ | ✅ |
| Trust-region LM solver (S7) | ✅ | ❌ | ✅ |
| Annealed telescope (`annealed_stages > 1`) | ✅ | ❌ | ✅ |
| Q(θ)/R(θ) tangents | ✅ | ❌ | ✅ |
| Non-Gaussian transition density | ✅ | ❌ | ✅ |
| Full diagnostic payload | ✅ (filter lane) | ❌ (`program_valid` only) | ✅ |

---

## Principal design risk

**Per-row reductions under flattening.**

The fused strategy flattens `[B, N] → [B*N]` on the claim that every
per-particle operation is pointwise, with only weight normalisation
(`logsumexp`, `softmax`) needing a per-row reshape.

The reset breaks that claim. Sinkhorn transport and Contract-E moment matching
are reductions **over the N particles of a single row**: the transport plan
couples particles within a cloud, and moment matching targets that cloud's mean
and covariance. Under flattening these become segment reductions over B
segments of length N, and their analytical tangents must respect the same
segmentation.

This is the crux. A naive port that reuses the pointwise flattening for the
reset stages will silently mix clouds across θ rows and produce plausible,
wrong numbers that batch-size-1 parity tests cannot detect — B=1 has exactly
one segment, so segmentation bugs are invisible at B=1.

**Consequences for the plan:**

- The segmentation derivation is written and reviewed **before** any reset code
  is ported (Step 3a below).
- Multi-row cloud-independence tests at B ≥ 2 with *distinct* θ rows are
  mandatory promotion criteria, not nice-to-have. A refactor that passes B=1
  parity but fails B≥2 row independence is a failed refactor.
- The dual-cap and trust-region JVPs inherit the same segmentation and are
  covered by the same tests.

Secondary risk: contract C-9 forbids autodiff in the claim-bearing path, so
each ported stage needs its hand-derived JVP checked against the autodiff
oracle (`ledh_canonical_autodiff_oracle_tf`) on small fixtures — the same
gate the original derivation used.

---

## Method: Fowler discipline

The refactor follows *Refactoring* (Fowler) and *Working Effectively with Legacy
Code* (Feathers) explicitly, because the tuning-artifact constraint makes
behaviour preservation load-bearing rather than aesthetic.

### Step 1 — Characterisation tests (golden master)

Fowler/Feathers: before changing code you do not fully understand, pin its
current behaviour with tests that assert *what it does*, not what it should do.

The existing 75 canonical tests are **not** a sufficient net:

- The 6 fused parity tests use `horizon=3`, `n=6`, `reset_policy` defaulted to
  `"none"` — they pin the diagnostic slice only.
- No batch-lane test exercises the reset family at all.
- Tolerances are rtol 5e-4, too loose to detect the small numeric drift that
  would invalidate tuning scope.

Deliverables:

1. **Golden-master fixtures.** Serialise exact numeric output (value, score,
   full diagnostic payload) of `single_cloud` with `reset_policy="contract_e"`
   and the 2026-09-03 tuned controls, across a fixture matrix:
   - horizons T ∈ {1, 2, 3, 10, 50}
   - particle counts N ∈ {6, 64, 1008}
   - dims d ∈ {2, 3}
   - directions K ∈ {1, 2, 5}
   - dtypes {float64, float32}
   - `annealed_stages` ∈ {1, 8}
   Stored under `tests/highdim/fixtures/ledh_golden_master_20260904/` with a
   generation script and a manifest recording git commit, env, and seeds.
2. **Row-independence and segmentation tests** at B ∈ {1, 2, 4} with distinct θ
   rows, asserting: identical rows give identical outputs; distinct rows give
   distinct outputs; row `i` output is unchanged by any perturbation of row `j`.
   This is the test class that catches cloud mixing.
3. **Capability-matrix tests**: one test per union capability above, each
   currently-supported-somewhere capability exercised through the lane that
   supports it, so the post-refactor engine must satisfy all of them.
4. **Tolerance policy**: golden-master comparisons at rtol 1e-12 (float64) —
   bitwise where achievable. The loose 5e-4 op-order tolerance is reserved for
   cross-lane parity, not for pre/post-refactor comparison of the same lane.

Exit criterion: the full suite passes against **unmodified** code. A
characterisation suite that does not pass before the refactor is not a baseline.

### Step 2 — Refactor contract

A written contract, reviewed before code:

- Public signatures preserved for all four entry points (adapters keep their
  current call shape).
- Numeric output on the `contract_e` route preserved to golden-master
  tolerance.
- No autodiff introduced in the claim-bearing path (C-9).
- No pfor / `tf.vectorized_map` (repository policy).
- Batch-native preserved; no Python loop over rows or directions in the traced
  graph.
- Diagnostic payload preserved or extended, never reduced.
- Chunk policy selector untouched (`dpf_transport_exact_divisor_cap3000_v1`).
- Tuning scope preserved — enumerated bound fields listed explicitly, each with
  the test that pins it.

### Step 3 — Branch by Abstraction

Fowler's pattern for replacing a load-bearing component without a long-lived
branch. The unified engine is built behind the existing entry points, and call
sites migrate one at a time with everything green throughout.

- **3a. Segmentation derivation.** Write the per-row segment reduction for
  Sinkhorn transport, Contract-E moment matching, dual-cap, and trust-region,
  with analytical tangents, in repository notation. Reviewed before code.
  *No code in this sub-step.*
- **3b. Extract the shared kernel.** Introduce the unified engine module with
  the union model contract. Initially it may delegate internally; the point is
  a single named seam.
- **3c. Port stages one at a time.** For each stage (UKF predict tangent,
  anchor chaining, flow + log-det, PF-PF weight, UKF update, softmax
  accumulation, then S6 reset, dual-cap, S7 trust-region): port, check the
  hand-derived JVP against the autodiff oracle, run the full characterisation
  suite. Commit only when green.
- **3d. Parallel run.** Run old and new engines side by side over the fixture
  matrix, comparing value, score, and diagnostics. Discrepancy beyond
  golden-master tolerance halts the port and is diagnosed before proceeding.
- **3e. Migrate adapters.** Repoint `single_cloud`, `batch`, `batch_fused`,
  `neutra_target` at the unified engine one at a time. `batch`'s Python row
  loop is deleted — it becomes a reshape.
- **3f. Retire duplicates.** Delete the superseded engine only after all
  adapters are migrated and green. Deletion is a separate commit.

### Step 4 — Registry and guard

- Repair `ENTRY_POINTS` so every registered `callable_name` resolves.
- Extend the discovery guard to fail closed on: an unregistered qualifying
  route, a lane not resolving to the unified engine, or a lane accepting a
  production control it silently ignores.
- Add the wiring test the Implementation Audit Call-Chain Rule requires: each
  claim-bearing endpoint resolves to the unified engine, proven executably.

### Step 5 — Close the inherited debt

With one engine, the deferred items become cheap and are closed here:

- Phase 1's five unmeasured diagnostics (FD-vs-JVP < 1e-6, Sinkhorn marginal
  TV < 1e-3, Contract-E moment residual, Cholesky condition number, dual-cap
  convergence counts) — run on the unified engine, recorded as measurements.
- Graph-size measurement, never taken: node count and GraphDef bytes at
  (B=6, horizon=50, N=252), against the 110,628-node unrolled baseline.
- The August audit's C3 (lane dtype/reset divergence) and C4 (missing score
  diagnostics) — structurally resolved by unification; verified, not asserted.

---

## Promotion criteria

| # | Criterion | Threshold |
|---|---|---|
| 1 | Golden-master parity, `contract_e` route | rtol ≤ 1e-12 (float64) across full fixture matrix |
| 2 | Row independence at B ≥ 2, distinct θ | exact; row `i` invariant to row `j` |
| 3 | Cross-lane parity, all four adapters | rtol ≤ 5e-4 vs unified engine |
| 4 | Hand-derived JVP vs autodiff oracle, every stage | passes existing oracle gates |
| 5 | Full canonical suite | 75 pre-existing tests + new characterisation suite green |
| 6 | Policy | no pfor, no autodiff in claim path, no Python row/direction loop in graph |
| 7 | Registry guard | fails closed on unledgered / unresolved / non-unified lane |
| 8 | Tuning scope | enumerated bound fields unchanged; 2026-09-03 artifacts remain valid |
| 9 | Diagnostic payload | preserved or extended for every lane |

## Promotion vetoes

- Any golden-master discrepancy beyond tolerance on the `contract_e` route.
- Row-independence failure at B ≥ 2 (indicates cloud mixing).
- Any JVP stage failing the autodiff oracle.
- Reduced diagnostic payload on any lane.
- Tuning scope invalidated → retune owed before any claim-bearing run.

## Continuation vetoes

- The segmentation derivation (3a) shows the flattening strategy is
  **mathematically incompatible** with per-row reset reductions. Then unification
  onto the flattened engine is wrong, and the phase stops for redesign — the
  alternative being a `[B, N, ...]`-shaped engine that never flattens.
- Behaviour preservation proves unreachable, i.e. the two engines are already
  numerically divergent on the `contract_e` route. That is a pre-existing defect
  this phase would surface rather than cause, and it needs its own
  investigation before unification.

Neither is a reason to abandon single-authority as a goal.

---

## Budget

| Step | Content | Estimate |
|---|---|---|
| 1 | Characterisation tests + golden-master fixtures | 1–2 days |
| 2 | Refactor contract (written, reviewed) | 0.5 day |
| 3a | Segmentation derivation | 0.5–1 day |
| 3b–3d | Extract, port stages, parallel run | 2–4 days |
| 3e–3f | Migrate adapters, retire duplicates | 1 day |
| 4 | Registry + guard + wiring tests | 0.5 day |
| 5 | Close inherited debt (5 diagnostics + graph size) | 0.5 day |

**Total: 6–9 days.** Attempt budget 12. Compute: negligible except Step 5
(< 1 GPU-hour) — this is CPU test work, not a campaign.

Compare: recorded option A ("port reset into fused") was multi-day on its own
and left two engines standing. The marginal cost of full unification over
option A is roughly Steps 1, 4, and 3e–3f — and it removes the drift class
permanently.

---

## Sequencing

This phase blocks Phase 3 **if** Phase 3 is to run on a production-conformant
fast lane. Two orderings are viable:

- **Serial.** Unification, then Phase 3 on the unified engine. Phase 3 then
  measures the lane you intend to ship, and its cost/ESS numbers mean something.
  Delays scientific evidence by 6–9 days.
- **Parallel.** Phase 3 runs on `single_cloud` with `contract_e` (valid estimand,
  available today) while unification proceeds. Correctness evidence arrives
  sooner; the performance question is re-answered later on the unified engine.
  Costs one extra Phase 3 run.

Owner decision. Under either ordering, the Phase 2 T3 acceptance measurement
(the `tf.custom_gradient` wrapper) is a prerequisite and is cheap — hours, not
days — because that wrapper is needed by every path.

---

## Artifacts

- This plan.
- `docs/plans/ledh-engine-unification-refactor-contract-2026-09-04.md` (Step 2)
- `docs/plans/ledh-engine-unification-segmentation-derivation-2026-09-04.md` (Step 3a)
- `tests/highdim/fixtures/ledh_golden_master_20260904/` + generator + manifest
- `tests/highdim/test_ledh_engine_characterisation.py`
- `tests/highdim/test_ledh_engine_row_independence.py`
- `tests/highdim/test_ledh_engine_capability_matrix.py`
- `tests/highdim/test_ledh_engine_wiring.py`
- `bayesfilter/highdim/ledh_canonical_engine_tf.py` (unified engine)
- `docs/plans/ledh-engine-unification-result-2026-09-04.md`

---

## Non-claims

- Unification establishes no scientific result. It changes representation, not
  mathematics.
- It does not validate surrogate-force HMC; that remains Phase 3's job.
- It does not establish the LEDH score is unbiased. The August audit's
  `BLOCKED_FOR_CLAIM` verdict on score accuracy stands and is untouched here.
- It does not retune anything. If tuning scope is invalidated, that is a veto,
  not a deliverable.
- Preserved behaviour means preserved relative to the current `contract_e`
  route, which is itself the object the August audit found underidentified.
  Faithful reproduction of the current route is the goal; endorsement of that
  route's accuracy is not.

---

**END OF PHASE PLAN**

Status: DRAFT — awaiting owner sequencing decision
Last updated: 2026-09-04
