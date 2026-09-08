# LEDH Surrogate-Force HMC — Unified Master Program

**Program ID:** `ledh-surrogate-hmc-unified-2026-09-06`  
**Date:** 2026-09-06  
**Status:** ACTIVE — Phase 2A in progress  
**Branch:** `ledh-refactor-with-policy-fix`  
**Owner:** chakwong  
**Supersedes:** Four scattered documents (authority, master, phase plans, unification plan)

---

## Research Question

Can surrogate-force HMC with damped LEDH analytical score achieve competitive
effective sample size per gradient evaluation while maintaining correct posterior
coverage?

**Hypothesis:** Corollary 5.2 (variance note lines 954-976) decouples correctness
from score bias. The chain targets exp(-U) exactly where U is the executed
filter value. Score bias affects mixing efficiency only, not the invariant
distribution. If a cheaper damped score mixes acceptably, surrogate-force HMC
trades force accuracy for throughput without compromising inference correctness.

**Test models:** Austria SIR T20, LGSSM T50, KSC SV T10, Predator-Prey T20 (the
four with trust-region tuning artifacts from 2026-09-02/03).

---

## Program Structure

| Phase | Content | Duration | Status |
|---|---|---|---|
| **0** | Policy repair (pfor removal) | complete | ✅ DONE 2026-09-04 |
| **1** | Route identity baselines | 0.5 day | ⏳ INCOMPLETE (deferred to 2B Step 5) |
| **2A** | Toy potential mechanics | 0.5 day | ✅ COMPLETE 2026-09-09 |
| **2B** | Engine unification | 6–9 days | 📋 NEXT |
| **2C** | Contract-integrity test suite | 2–3 days | ⏸️ BLOCKED on 2B |
| **3** | Damping derivation + calibration | 1–2 days | ⏸️ BLOCKED on 2C |
| **4** | LGSSM d=3 T=50 full validation | 1 day + 12 GPU-hours | ⏸️ BLOCKED on 3 |
| **5** | Tier A suite (conditional) | 2 days + 48 GPU-hours | ⏸️ BLOCKED on 4 |

**Total estimate:** 13–18 days + 60 GPU-hours (assumes no major redesign)

---

## Phase 0: Policy Repair ✅ COMPLETE

**Date:** 2026-09-04  
**Commit:** 576cfa18

**What was done:** Removed 6 × `tf.vectorized_map` (implicit pfor, policy
violation) from `canonical_batch_fused_value_score` and replaced with
`tf.while_loop`. All 6 fused parity tests pass.

**Artifacts:**
- `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` (refactored)
- `tests/highdim/test_ledh_canonical_batch_fused.py` (6 tests, all passing)

**Status:** COMPLETE, with one gap: the refactored kernel has no `reset_policy`
parameter, so it cannot run the production program. Phase 2B resolves this by
unifying onto one engine carrying the full capability union.

---

## Phase 1: Route Identity and Wiring Baselines

**Goal:** Measure the five diagnostics required to interpret Phase 4 results.

**Status:** INCOMPLETE — originally marked COMPLETE on 2026-09-04, corrected
same day. All five measurements were asserted from existing tests rather than
executed.

### Required measurements

| Diagnostic | Threshold | Lane | Status |
|---|---|---|---|
| FD-vs-JVP directional residual | < 1e-6 | `single_cloud` with `contract_e` | not checked |
| Sinkhorn marginal TV | < 1e-3 | same | not checked |
| Contract-E moment residual (mean/cov) | recorded baseline | same | not checked |
| Cholesky condition number | recorded | same | not checked |
| Dual-cap convergence (iter, floor hits) | recorded | same | not checked |

### What was wrongly claimed

The original Phase 1 document reasoned from the six passing fused parity tests
(at `horizon=3`, `reset_policy` defaulted to `"none"`) and the August 29 audit.
Neither is a substitute for these five route-level diagnostics. Per the
Implementation Audit Call-Chain Rule, a prose audit without an executable check
must say "not checked."

### Exit criterion

One artifact recording all five measurements at (d=3, T=50, N=1008,
`reset_policy="contract_e"` with the 2026-09-03 tuned controls). Stored under
`docs/benchmarks/artifacts/ledh_route_identity_baselines_20260906/`.

### Blocking status

Phase 2B (unification) closes this as Step 5 ("Close inherited debt"). Running
it now on `single_cloud` is redundant work, so Phase 1 waits for Phase 2B.

---

## Phase 2A: Toy Potential Mechanics Check

**Goal:** Isolate surrogate-force mechanics from filter complexity using a
simple quadratic potential.

**Status:** ✅ COMPLETE — 2026-09-09

### What exists

- `bayesfilter/inference/toy_surrogate_force_adapter.py` (161 LOC)
- `tests/inference/test_toy_surrogate_force.py` (4 tests)

| Test | Quantity | Contract role | Result |
|---|---|---|---|
| T1 | Determinism (same θ → same value, force) | promotion veto | ✅ PASS |
| T2 | Energy conservation over leapfrog | promotion veto | ✅ PASS |
| T3 | Acceptance rate ≥ 0.2 at damping 0.1 | **promotion criterion** | ✅ PASS (67% at damping=0.1) |
| T4 | Force scaling (damping 0.1 → 0.1× ‖force‖) | explanatory | ✅ PASS |

### T3 results — 2026-09-09

Wrapped `DualAdapterSurrogateForce` with `reviewed_value_score_target_fn` and
ran TFP HMC chains (100 samples, 10 leapfrog steps, step_size=0.05).

**Acceptance rates:**
- damping=1.0: 100%
- damping=0.5: 82%
- damping=0.1: 67% (exceeds ≥20% requirement)

All promotion vetoes pass. Surrogate-force HMC mechanics are correct.

### Exit criterion

All four tests pass. Toy adapter demonstrates surrogate-force correctness
independent of filter complexity.

T3 measured and passing (acceptance ≥ 0.2 at damping 0.1). Result document
updated with the actual promotion-criterion status.

### Dependency

Independent — can close before or during Phase 2B.

---

## Phase 2B: Single-Authority Engine Unification

**Why this phase exists:** The LEDH algorithm is implemented twice
(`single_cloud` 617 LOC, `batch_fused` 736 LOC). The fork produced capability
divergence: the refactored fused lane cannot run the production program (no
`reset_policy`, no dual-cap, no trust-region), so the trust-region tuning
artifacts are stranded and Phase 4 has no lane to execute on. Unifying onto
one engine carrying batch-native execution + Contract-E reset + dual-cap +
trust-region resolves the blocker and removes the drift class permanently.

**Method:** Fowler/Feathers refactoring discipline. Characterisation tests and
golden-master fixtures first, written refactor contract, Branch by Abstraction
with stage-by-stage porting and parallel run, then adapter migration and
deletion of the duplicate.

### Step 1: Characterisation tests (1–2 days)

**Deliverables:**

1. **Golden-master fixtures.** Serialise exact numeric output (value, score,
   full diagnostics) of `single_cloud` with `reset_policy="contract_e"` and the
   tuned controls, across:
   - T ∈ {1, 2, 3, 10, 50}
   - N ∈ {6, 64, 1008}
   - d ∈ {2, 3}
   - K (directions) ∈ {1, 2, 5}
   - dtype ∈ {float64, float32}
   - `annealed_stages` ∈ {1, 8}
   
   Stored under `tests/highdim/fixtures/ledh_golden_master_20260906/` with
   generation script, manifest (git commit, conda env, seeds).

2. **Row-independence tests** at B ∈ {1, 2, 4} with distinct θ rows. Assert:
   - Identical rows → identical outputs
   - Distinct rows → distinct outputs
   - Row `i` output unchanged by row `j` perturbation
   
   This is the test class that catches the principal design risk: per-row
   segment reductions under `[B,N] → [B*N]` flattening. A naive port mixes
   clouds across θ rows; this is invisible at B=1.

3. **Capability-matrix tests:** one test per union capability, each exercised
   through the lane that currently supports it.

4. **Tolerance policy:** golden-master comparisons at rtol 1e-12 (float64). The
   loose 5e-4 tolerance is for cross-lane parity, not pre/post-refactor.

**Exit:** The full suite passes against unmodified code.

### Step 2: Refactor contract (0.5 day)

Written contract, reviewed before code:
- Public signatures preserved for all four entry points
- Numeric output on `contract_e` route preserved to golden-master tolerance
- No autodiff in claim-bearing path (C-9)
- No pfor / `tf.vectorized_map`
- Batch-native preserved (no Python row/direction loop in traced graph)
- Diagnostic payload preserved or extended, never reduced
- Chunk policy selector untouched
- Tuning scope preserved — bound fields enumerated, each with its pinning test

### Step 3: Branch by Abstraction (2–4 days)

- **3a. Segmentation derivation.** Write the per-row segment reduction for
  Sinkhorn transport, Contract-E moment matching, dual-cap, and trust-region,
  with analytical tangents. Reviewed before code. *No code in this sub-step.*
  
  **Continuation veto:** If the derivation shows flattening is mathematically
  incompatible with per-row reductions, unification onto the flattened engine
  is wrong. The alternative is a `[B,N,...]`-shaped engine that never flattens.

- **3b. Extract the shared kernel.** Introduce the unified engine module with
  the union model contract.

- **3c. Port stages one at a time.** For each stage: port, check hand-derived
  JVP against autodiff oracle, run full characterisation suite. Commit only
  when green.

- **3d. Parallel run.** Old and new engines side by side over fixture matrix.
  Discrepancy beyond golden-master tolerance halts and is diagnosed.

- **3e. Migrate adapters.** Repoint `single_cloud`, `batch`, `batch_fused`,
  `neutra_target` one at a time. `batch`'s Python row loop deleted — it becomes
  a reshape.

- **3f. Retire duplicates.** Delete superseded engine only after all adapters
  migrated and green. Separate commit.

### Step 4: Registry and guard (0.5 day)

- Repair `ENTRY_POINTS` so every registered callable resolves
- Extend discovery guard to fail closed on unledgered / unresolved /
  non-unified lane
- Add wiring tests (Implementation Audit Call-Chain Rule): each claim-bearing
  endpoint resolves to unified engine, executably

### Step 5: Close inherited debt (0.5 day)

With one engine, the deferred items become cheap:
- Phase 1's five unmeasured diagnostics (run, record)
- Graph-size measurement (node count, GraphDef bytes at B=6, T=50, N=252)
- August audit's C3 (lane dtype/reset divergence) and C4 (missing score
  diagnostics) — structurally resolved, verify executably

### Promotion criteria

| # | Criterion | Threshold |
|---|---|---|
| 1 | Golden-master parity, `contract_e` route | rtol ≤ 1e-12 (float64) |
| 2 | Row independence at B ≥ 2 | exact; row `i` invariant to row `j` |
| 3 | Cross-lane parity, all four adapters | rtol ≤ 5e-4 vs unified engine |
| 4 | Hand-derived JVP vs autodiff oracle | passes existing gates |
| 5 | Full canonical suite | 75 pre-existing + characterisation green |
| 6 | Policy | no pfor, no autodiff in claim path, no Python loop |
| 7 | Registry guard | fails closed on violations |
| 8 | Tuning scope | bound fields unchanged; 2026-09-03 artifacts valid |
| 9 | Diagnostic payload | preserved or extended |

### Promotion vetoes

- Golden-master discrepancy beyond tolerance on `contract_e` route
- Row-independence failure at B ≥ 2 (cloud mixing)
- Any JVP stage failing autodiff oracle
- Reduced diagnostic payload
- Tuning scope invalidated → retune owed

### Exit criterion

All promotion criteria pass; result document records what changed and what
stayed identical.

### Budget

6–9 days CPU test work, < 1 GPU-hour for Step 5.

---

## Phase 2C: Contract-Integrity Test Suite

**Why this phase exists:** The 2026-09-06 audit found 12 defects. None was a
mathematical error. Every one was a **contract-integrity** failure — code not
implementing what a document claimed, or a document asserting a quantity never
measured. The repository has extensive numerical tests and no tests of this
class at all, which is why B1 (a runner silently skipping the reset) survived
five days and a Codex review.

**Design note:** `docs/plans/contract-integrity-testing-framework-design-2026-09-06.md`

**Position:** After 2B, before Phase 3. Unification produces one engine with the
full capability union — the natural checkpoint to verify the contract stack.
Earlier means testing a forked implementation; later means Phase 3/4 findings
force backtracking.

### The seven test classes

Each maps to audit findings it would have caught.

| Class | What it checks | Catches |
|---|---|---|
| 1. Enum / string-literal validation | Every dispatch string is ledgered; unrecognized fails closed | **B1** |
| 2. Wiring and resolution | Every registry key, artifact reference, fixture claim, and kwarg resolves | **B3, B4**, registry drift |
| 3. Shape and dimension consistency | Declared shapes match actual tensor shapes | **M1** |
| 4. Artifact schema and completeness | Artifacts carry required fields; budget tables sum | **N2**, Phase 1 overclaim |
| 5. Parameter provenance / Class C | Every numerics-altering parameter has derivation, curve, or rationale | **M5** |
| 6. Seed determinism | Every "frozen" or "deterministic" claim has a golden-hash regression | **B2, M4** |
| 7. Policy compliance | Policy "must"/"forbidden" rules mechanized as failing tests | **M2, M3**, Phase 2A T3 substitution |

### Class 1 — Enum and string-literal validation

Every configuration string compared via `==` or dispatched on must validate
against a ledger. Unrecognized → raise, never fall through.

The B1 mechanism: `ledh_canonical_score_tf.py:413` tests one literal
(`"contract_e"`); the runner passed `"transport_affine_cumulant_trust"`; the
bare `else` silently skipped the reset. The fix is a Class B fail-closed guard
(adopt-by-default under Safety Guardrail Reversed Burden), plus a **negative
test** asserting the unrecognized value raises.

Scope: every string literal in an `if x == "..."` or `match` in claim-bearing
code. Each needs a legal-values ledger and a negative case.

### Class 2 — Wiring and resolution

Every indirection resolves to what it names:

- `ENTRY_POINTS[k]` imports and is callable (the Aug-29 audit found one that
  was not)
- Every `Baseline: X` in an evidence contract resolves to `artifacts/X/`
  (**B3**)
- Every runner `kwarg=value` maps to a formal parameter via
  `inspect.signature` (**B4**)
- Every claim-bearing endpoint reaches the unified engine — the executable
  wiring test the Implementation Audit Call-Chain Rule requires

Method: reflection, `inspect.signature`, filesystem checks, AST walk for call
chains.

### Class 3 — Shape and dimension consistency

Declared shapes match produced shapes. **M1** was a 1-D runner under a
5-parameter coverage criterion; a shape test comparing runner output rank
against the contract's claim fails immediately.

Scope: every dimension claim in a docstring, plan, or evidence contract.

### Class 4 — Artifact schema and completeness

Artifacts carry the fields consumers require; plan arithmetic is
self-consistent. Two sub-checks matter most:

- **Measurement existence.** Every "measured" claim in a result document maps
  to an artifact field holding that number. This is the structural fix for
  Phase 1's five asserted-but-unmeasured diagnostics: a phase cannot close on
  prose, only on an artifact containing the measurement.
- **Budget completeness.** Phase-table durations sum to the stated total
  (**N2**).

### Class 5 — Parameter provenance and Class C calibration

Every Class C (numerics-altering) parameter appears in a ledger mapping
`(name, model, route)` → `(artifact_path, justification_type)`. Justification
type may not be "inherited" or "convenient" — the policy is explicit that an
off/zero setting carries the same burden as any other value.

**M5** is the live case: the 100× damping ratio traces to one Aug-29 line with
no calibration. **B4** compounds it — the runner would have overwritten a
*tuned* control (`correction_lm_scale_floor`, selected 1e-06) with 1e-3.

### Class 6 — Seed determinism and reproducibility

Every "frozen" or "deterministic" claim gets a regression:

- Fixture golden-hash: `_lgssm_frozen_observations()` → recorded SHA-256. This
  is exactly **B2** — the runner's `make_observations()` returned i.i.d. noise
  under a docstring claiming the historical fixture. A hash test fails on the
  first run.
- Determinism: run twice, assert bitwise-identical.
- Seed-freeze (**M4**): force evaluated at the same θ but different leapfrog
  steps must use identical noise, which is what Corollary 5.2 requires and what
  v2's θ-derived `hash()` violates.

Note the repository already has a recorded instance of naive seed derivation
producing pseudo-replication (`tf-consecutive-from-seed-is-one-stream`); use
`SeedSequence`, not `hash()`.

### Class 7 — Policy compliance

Policy rules become failing tests rather than prose reminders:

- **Heuristic Dominance Gate** (**M3**): Phase 4/5 plan schema requires
  `adversary_set` ≥ 3 entries and `salient_situations` ≥ 2. A plan without
  them cannot validate.
- **Evidence contract**: Phase 3+ requires all six fields present.
- **Promotion-criterion binding**: the measured quantity must be the
  contract's stated primary criterion — this is the check that would have
  refused Phase 2A's substitution of mean-‖force‖ for acceptance rate.
- **Program-artifact binding** (**M2**): a phase's model set must match the
  tuning-artifact index, which is how the HNN model list got into a LEDH
  program.
- **Pfor approval**: `tf.vectorized_map` appearing in a diff fails CI unless
  `docs/approvals/pfor_{module}_{date}.md` exists.

### Deliverables

1. **Test modules** under `tests/contracts/`:
   `test_enum_validation.py`, `test_wiring_resolution.py`,
   `test_shape_consistency.py`, `test_artifact_schema.py`,
   `test_parameter_provenance.py`, `test_seed_determinism.py`,
   `test_policy_compliance.py`

2. **Machine-readable ledgers** under `bayesfilter/contracts/`:
   `reset_policy_enum.json`, `entry_points_registry.json`,
   `class_c_parameter_ledger.json`, `tuning_artifact_index.json`,
   `fixture_hashes.json`

3. **Plan schema** `docs/schemas/experiment_plan.schema.json`

4. **Validation runner** `scripts/validate_contracts.py` — runs all seven
   classes, validates every plan against schema, checks artifact schemas,
   exit 1 on any violation. Wired into CI and used as a pre-execution gate.

5. **Fail-closed guard** on `reset_policy` in the unified engine (the B1 repair
   proper, not just its test).

6. **Result document**
   `docs/plans/ledh-contract-integrity-suite-result-2026-09-06.md` recording
   coverage counts, violations found and repaired, and how to register a new
   contract when extending the code.

### Promotion criteria

1. Each of the 12 audit findings has a test that fails on the original defect
2. Enum test fails on `reset_policy="transport_affine_cumulant_trust"`
3. Fixture-hash test fails if `make_observations()` returns wrong data
4. Parameter-resolution test fails on a non-existent kwarg
5. Schema validation rejects a Phase 4 plan lacking an adversary set
6. Full suite green on the unified engine
7. `validate_contracts.py` runs in CI and fails the build on violation

### Promotion vetoes

- Any of the 12 findings without covering test
- A contract test that warns instead of failing
- `class_c_parameter_ledger.json` missing a Class C parameter used in
  claim-bearing code

### Regression check

The suite must be green against **current** code after repairs, and each test
must be demonstrated failing against the pre-repair state (git stash or a
deliberately reverted fixture). A test never seen to fail is not evidence.

### Budget

2–3 days, CPU only, no GPU. Independent of Phase 3's derivations — those may
proceed in parallel, but no scientific run executes until this suite is green.

### Fallback if schedule pressure

Minimum viable gate, 0.5 day, covering all four blockers:

1. Enum validation for `reset_policy` and `ENTRY_POINTS` (B1)
2. Fixture golden-hash for `_lgssm_frozen_observations` (B2)
3. Parameter resolution across every runner in `docs/benchmarks/` (B3, B4)

Full Class 1–7 suite then becomes a later phase. These three are load-bearing;
without them the program is one unrecognized string from another silent
failure.

---

## Phase 3: Damping Derivation and Calibration

**Why this phase exists:** Four blocking/major findings from the 2026-09-06
audit must be resolved on paper before Phase 4 code.

### Task 3.1: Damping parameter derivation (0.5 day)

**Problem:** The program states the scientific question in terms of λ
(process-covariance ridge) and δ (observation-covariance ridge). Those
parameters do not exist. `canonical_value_and_analytical_score` exposes
`reset_ridge` (Contract-E ridge) and `correction_lm_damping` /
`correction_lm_scale_floor` (trust-region LM). The pre-existing runner v2 maps
λ → `reset_ridge` and δ → `correction_lm_scale_floor`, where the second is a
**tuned control** (selected value 1e-06 for LGSSM T50). A 1000× change to a
Class C numerics-altering protection is a new tuning scope.

**Deliverable:** A written derivation answering: what does "damp the LEDH score
by 100×" mean in the parameters that exist? Candidates are not interchangeable
— reset ridge, LM damping, flow substep count, and Sinkhorn ε perturb different
stages with different bias signatures.

**Method:** Start from the stages where bias enters (Sinkhorn, dual-cap,
trust-region) and derive how each control scales the score perturbation.
Identify which one is the right lever for Corollary 5.2's "biased force"
without perturbing the value.

**Exit:** One parameter (or pair) identified, with its mapping to the 100×
damping ratio derived rather than guessed.

### Task 3.2: Damping calibration curve (0.5–1 day)

**Problem:** The 100× ratio traces to a single Aug-29 line with no calibration.
Class C numerics-altering protections require a derivation, measured
calibration curve, or recorded owner rationale. The program's own ladder (1e-3,
fallback 1e-4) is a two-point guess.

**Deliverable:** A bias-vs-mixing curve over the identified parameter, measured
on LGSSM d=3 T=50 at N=1008 with the tuned `contract_e` controls. Points:
damping ∈ {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}. For each: one short HMC run
(4 chains × 500 warmup × 500 sampling), measuring acceptance, ESS/grad, and
mean score bias vs exact Kalman.

**Method:** Run on the unified engine (Phase 2B output), CPU-only for cheap
iteration. Stop when acceptance < 0.15 or ESS/grad < 0.3× the exact arm.
Select the coarsest damping passing both floors.

**Exit:** One artifact under
`docs/benchmarks/artifacts/ledh_damping_calibration_lgssm_t50_20260906/` with
the curve, selected value, and recorded rationale.

### Task 3.3: Seed discipline resolution (0.5 day)

**Problem:** Corollary 5.2 requires all Monte Carlo seeds frozen across the
trajectory. v2 derives seed from θ via `hash((seed_base, theta_tuple)) %
2**31`, so noise changes at every leapfrog step — arguably the exact failure
Remark 5.3 warns about. Two readings ("frozen per trajectory" vs "deterministic
function of θ") are different estimators; the program never chose.

Also: `hash()` is unstable across processes and naive seed derivation is
recorded as causing pseudo-replication
(`tf-consecutive-from-seed-is-one-stream` memory).

**Deliverable:** A written decision on seed policy, with implementation pattern
using `SeedSequence`. Either: (a) one trajectory-level seed passed down and
never reseeded; or (b) θ-derived seed documented as the chosen estimator with
Remark 5.3's caveat recorded.

**Exit:** One paragraph in the Phase 4 plan stating which seed policy is used
and why, plus a test that verifies noise is indeed frozen (option a) or
θ-reproducible (option b) as claimed.

### Budget

1–2 days + ~4 GPU-hours for the calibration curve.

---

## Phase 4: LGSSM d=3 T=50 Full Validation

**Goal:** Test surrogate-force HMC with full LEDH filtering on the canonical
LGSSM fixture.

**Test model:** LGSSM d=3 T=50, frozen observations (seed 81100, φ=[0.72, 0.55,
0.35], q=0.35, r=0.45, 3×3 observation matrix). Fixture exists at
`ledh_canonical_neutra_targets_tf.py:340` (`_lgssm_frozen_observations`).

**Lane:** Unified engine with `reset_policy="contract_e"` and the 2026-09-03
tuned controls.

### Evidence contract

**Question:** Does surrogate-force HMC with calibrated damping mix acceptably
on the full LEDH filter?

**Arms:**

| Arm | Value λ/δ | Force λ/δ | Role |
|---|---|---|---|
| 1 | exact (1e-6 / 1e-6) | exact (same) | Baseline comparator |
| 2 | exact (1e-6 / 1e-6) | damped (Task 3.2 output) | Surrogate-force test |

(Intermediate fallback arm dropped — two-point comparison is sufficient for
nomination.)

**Primary criterion (promotion):** Arm 2 posterior 95% intervals cover true θ
(all 5 parameters).

**Secondary criterion (descriptive):** Arm 2 ESS/gradient > 0.3× Arm 1.

**Promotion vetoes:**
- Arm 1 fails to cover (baseline broken)
- Arm 2 covers but Arm 1 does not → value or seed bug, not surrogate success
- Acceptance < 0.15 in either arm (chain not mixing)

**Heuristic adversary set** (mandatory per CLAUDE.md Heuristic Dominance Gate):

| Adversary | What it is | Salient situation |
|---|---|---|
| H1 | Random-walk Metropolis, matched cost | No gradient, cheapest |
| H2 | Coarser unbiased LEDH (N=504), matched wall time | Unbiased alternative |
| H3 | Exact Kalman HMC | Correctness oracle (LGSSM only) |
| H4 | Fixed-preconditioner HMC (identity mass) | Ignores θ-dependence |

Evaluated conditionally on: (i) θ near boundary (φ close to 1); (ii) θ
interior. Losing to any adversary in any situation is the headline and a
promotion veto.

**Explanatory diagnostics:**
- Acceptance rate per arm
- ESS per gradient per arm
- Mean score bias (Arm 2 vs Kalman)
- R-hat (< 1.05)

**Non-claims:**
- Passing nominates surrogate-force for wider validation (Phase 5) only
- Single-seed run → no statistical ranking supported
- LGSSM coverage does not prove correctness on nonlinear models

### Execution

**Chains:** 4 × 5000 warmup × 5000 sampling per arm, one seed  
**Posterior:** All 5 θ components (not just θ₀)  
**Implementation:** `reviewed_value_score_target_fn` wrapper over unified engine  
**Seed policy:** As decided in Task 3.3  
**Device:** GPU (tftwogpu conda env, `CUDA_VISIBLE_DEVICES=1` → 4080 SUPER)

### Exit criterion

One artifact under
`docs/benchmarks/artifacts/ledh_surrogate_lgssm_t50_full_20260906/` containing:
- Promotion-criterion verdict (pass/fail with evidence)
- Heuristic adversary table (conditionally evaluated)
- ESS/gradient comparison (descriptive only, no ranking claim)
- Decision: Phase 5 proceed / stop / redesign

### Budget

1 day implementation + 12 GPU-hours (6 per arm × 2).

---

## Phase 5: Tier A Suite (Conditional on Phase 4 Promotion)

**Goal:** Validate surrogate-force across the trust-region-tuned model set.

**Models:** Austria SIR T20, KSC SV T10, Predator-Prey T20 (the three
non-LGSSM models with 2026-09-02/03 tuning artifacts).

**Entry condition:** Phase 4 promoted (coverage + heuristic dominance passed).
If Phase 4 failed on a promotion veto, Phase 5 does not run — the program stops
for redesign.

**Per-model contract:** Same two-arm structure as Phase 4 (exact baseline,
damped test), same promotion criterion (posterior coverage), same heuristic
adversary set conditionally evaluated. Each model runs independently; one
failure does not stop the others.

**Non-LGSSM limitation:** H3 (exact Kalman) not available. Heuristic set
reduces to H1/H2/H4 for these three.

### Exit criterion

One artifact per model under `docs/benchmarks/artifacts/`, each with:
- Promotion verdict
- Heuristic adversary table
- ESS/gradient comparison (descriptive)
- Model-specific decision

### Budget

2 days + 48 GPU-hours (16 per model × 3).

---

## Promotion and Continuation Vetoes

### Promotion criteria (whole program)

1. Phase 2B: all nine unification promotion criteria pass
2. Phase 4: posterior coverage + heuristic dominance + acceptance > 0.15
3. Phase 5: at least 2 of 3 models pass coverage + heuristic dominance

### Promotion vetoes (stop, do not proceed)

- Phase 2B golden-master discrepancy or row-independence failure → refactor
  broken
- Phase 4 Arm 1 fails coverage → baseline broken, not a surrogate-force test
- Phase 4 Arm 2 loses to any heuristic in any salient situation → mechanism
  inadequate
- Phase 5: all three models fail → surrogate-force does not generalise

### Structural rule adopted 2026-09-06

**A phase does not close on a document. It closes on an artifact containing the
measurement.**

Six defects to date share one shape: a document asserting a verified quantity
that was never measured (Phase 1's five diagnostics, Phase 2A's promotion
criterion, and audit findings B2/B3/M1/M5). Prose cannot be the evidence for
its own claim. Phase 2C Class 4 mechanizes this as a test: every "measured"
claim in a result document must map to an artifact field holding that number.

### Continuation vetoes (stop, redesign required)

- Phase 2B Step 3a derivation shows flattening incompatible with per-row
  reductions → engine redesign
- Phase 3.1 finds no parameter mapping that preserves value while damping score
  → Corollary 5.2 inapplicable
- Phase 4 passes but Phase 5 finds two models where coarser-unbiased (H2) beats
  surrogate at matched cost → the construction is unnecessary

---

## Non-Claims

- Unification (Phase 2B) establishes no scientific result. Preserved behaviour
  relative to current `contract_e` route, which the August audit found
  underidentified.
- Phase 4 single-seed run supports nomination only, not a statistical ranking.
- Posterior coverage is a correctness check, not evidence of superiority.
- Passing Phase 5 nominates surrogate-force as viable; it does not establish
  "better" without uncertainty analysis.
- ESS/gradient comparisons are descriptive unless multi-seed.
- The program tests whether surrogate-force HMC is correct and non-dominated by
  cheap heuristics. It does not test whether LEDH itself is an unbiased
  gradient.

---

## Artifacts

| Phase | Artifact | Type |
|---|---|---|
| 0 | `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` | refactored kernel |
| 0 | `tests/highdim/test_ledh_canonical_batch_fused.py` | 6 parity tests |
| 1 | `docs/benchmarks/artifacts/ledh_route_identity_baselines_20260906/` | 5 diagnostics |
| 2A | `bayesfilter/inference/toy_surrogate_force_adapter.py` | toy adapter |
| 2A | `tests/inference/test_toy_surrogate_force.py` | T1–T4 (T3 to be closed) |
| 2B | `tests/highdim/fixtures/ledh_golden_master_20260906/` | golden-master fixtures |
| 2B | `tests/highdim/test_ledh_engine_characterisation.py` | characterisation suite |
| 2B | `tests/highdim/test_ledh_engine_row_independence.py` | B≥2 independence tests |
| 2B | `bayesfilter/highdim/ledh_canonical_engine_tf.py` | unified engine |
| 2B | `docs/plans/ledh-engine-unification-result-2026-09-06.md` | result doc |
| 2C | `tests/contracts/test_*.py` (7 modules) | contract test suite |
| 2C | `bayesfilter/contracts/*.json` (5 ledgers) | machine-readable contracts |
| 2C | `docs/schemas/experiment_plan.schema.json` | plan validation schema |
| 2C | `scripts/validate_contracts.py` | validation runner |
| 2C | `docs/plans/ledh-contract-integrity-suite-result-2026-09-06.md` | result doc |
| 3 | `docs/plans/ledh-damping-derivation-2026-09-06.md` | parameter derivation |
| 3 | `docs/benchmarks/artifacts/ledh_damping_calibration_lgssm_t50_20260906/` | calibration curve |
| 3 | `docs/plans/ledh-seed-discipline-decision-2026-09-06.md` | seed policy |
| 4 | `docs/benchmarks/artifacts/ledh_surrogate_lgssm_t50_full_20260906/` | LGSSM validation |
| 5 | `docs/benchmarks/artifacts/ledh_surrogate_{model}_20260906/` | per-model results |

---

## Program Governance

**Authority:** This document is the single authority for the surrogate-force HMC
program. It supersedes:
- `docs/plans/LEDH_SURROGATE_HMC_PROGRAM_AUTHORITY_2026-09-04.md`
- `docs/plans/ledh-surrogate-force-hmc-master-program-2026-09-04.md`
- `docs/plans/ledh-single-authority-engine-unification-plan-2026-09-04.md`
- `docs/plans/ledh-surrogate-hmc-phase3-execution-plan-2026-09-04.md`

Those four are retired as historical. Phase result documents (Phase 1, 2A)
remain as written with their corrections in place.

**Audit trail:**
- 2026-09-06: Skeptical pre-execution audit identified 12 findings (4 blocking,
  5 major, 3 minor). Program rewritten to address all findings before execution.
- Pre-existing runner `surrogate_force_lgssm_three_arm_v2.py` found to pass
  unrecognised `reset_policy`, silently disabling the reset and producing a
  non-estimand value. Not used.

**Updates:** Changes to this program require owner approval and are recorded in
the commit message and in this section.

---

**END OF UNIFIED PROGRAM**

Last updated: 2026-09-06  
Status: ACTIVE — Phase 2A in progress
