# Contract-Integrity Testing Framework — Design Note

**Date:** 2026-09-06  
**Context:** 12 audit findings, 4 blocking. Pattern analysis shows no single
cause, but one systematic gap: **missing contract-integrity tests** that would
catch configuration errors, wiring breaks, parameter drift, and document-code
mismatches before execution rather than after silent failure.

---

## What each finding needed

| Finding | Test class that would have caught it |
|---|---|
| B1 (silent reset disable) | **Enum validation test**: `reset_policy` accepts only ledgered values, fails closed on unrecognized |
| B2 (placeholder observations) | **Fixture identity test**: hash of `make_observations()` output matches declared seed/generator |
| B3 (baseline doesn't exist) | **Artifact existence test**: every `Baseline: X` claim resolves to a file with required fields |
| B4 (λ/δ don't exist) | **Parameter name resolution test**: every runner argument maps to an actual formal parameter |
| M1 (1-D runner, 5-D criterion) | **Dimension consistency test**: runner output shape matches evidence-contract claim |
| M2 (wrong model list) | **Program-artifact binding test**: Phase 4 model set matches tuning artifact inventory |
| M3 (no adversary set) | **Policy compliance test**: Heuristic Dominance Gate check in plan schema |
| M4 (seed ambiguity) | **Seed-freeze test**: force evaluations at same θ, different leapfrog steps → identical noise |
| M5 (uncalibrated damping) | **Calibration-artifact test**: Class C parameter has derivation or curve on file |
| N2 (budget omits phase) | **Budget completeness test**: phase table sums to stated total |
| Phase 1 overclaim | **Measurement existence test**: "measured" diagnostic resolves to artifact with that field |
| Phase 2 T3 substitution | **Promotion criterion test**: measured quantity matches contract's primary criterion |

**Pattern:** None of these are mathematical correctness (those need numerical tests). All are **contract-integrity**: "does the code implement what the document claims, and vice versa?"

---

## Proposed test taxonomy

### Class 1: Enum and String-Literal Validation

**What:** Every configuration string compared via `==` or passed to a dispatcher
must be validated against a ledger of legal values. Unrecognized → fail closed.

**Examples:**
- `reset_policy` ∈ {`"none"`, `"contract_e"`} — reject others
- `ENTRY_POINTS` keys resolve to actual callables
- Model names in runner match a tuning-artifact index
- `description` in cron/monitor/workflow matches actual purpose

**How:** One test per enum. Negative case (unrecognized value) must raise or
return an error, not fall through to a default.

**Coverage target:** Every string literal in an `if x == "..."` or match
statement in claim-bearing code.

---

### Class 2: Wiring and Resolution Tests

**What:** Every indirection (registry lookup, fixture reference, parameter name,
artifact path claim) resolves to the thing it names.

**Examples:**
- `ENTRY_POINTS["batch_fused"]` resolves and is callable
- Every `Baseline: X` in an evidence contract resolves to `artifacts/X/`
- Every runner `kwarg=value` maps to a formal parameter of the callee
- Every `Fixture: Y` references a file whose hash matches the documented seed
- `canonical_value_and_analytical_score` with `reset_policy="contract_e"` calls
  `sinkhorn_contract_e_reset_with_tangent` (call-chain test per Implementation
  Audit Call-Chain Rule)

**How:** Reflection, `inspect.signature`, filesystem checks, import tests. For
call chains: either static analysis (AST walk) or a wiring smoke (does the
claimed path execute without error on minimal input).

**Coverage target:** Every registry, every artifact reference in a plan, every
fixture claim, every "implemented via X" statement.

---

### Class 3: Shape and Dimension Consistency

**What:** Declared shapes in documents match actual tensor shapes in code.

**Examples:**
- Runner claiming "5-parameter posterior" returns `(n_chains, n_samples, 5)`
- Model with `DIM=3` produces observations shaped `(HORIZON, 3)`
- Phase plan "K directions" → `theta_directions.shape[1] == K`
- Batch runner at `B=4` produces output batch dim 4

**How:** Property tests over a shape matrix. Assertion: `output.shape ==
claimed_shape`.

**Coverage target:** Every dimension claim in a docstring, plan, or evidence
contract.

---

### Class 4: Artifact Schema and Completeness

**What:** Every artifact directory contains the fields its consumers require,
and every plan's budget/phase table is self-consistent.

**Examples:**
- Tuning artifact has `config.json` with all 16 bound fields
- HMC artifact has `samples.npz`, `diagnostics.json`, `manifest.json`
- Evidence contract "5 diagnostics" → artifact has 5 top-level keys
- Phase table: sum(duration column) == stated total budget
- Result document: every "measured" claim has a corresponding artifact field

**How:** JSON Schema or dataclass validation. Budget: parse markdown table, sum
durations, compare to header.

**Coverage target:** Every artifact type, every plan document.

---

### Class 5: Parameter Provenance and Class-C Calibration

**What:** Every numerics-altering (Class C) parameter has a derivation,
calibration curve, or recorded rationale on file. Every "inherited" or
"transferred" setting is documented with source and justification.

**Examples:**
- `correction_lm_damping=0.01` → derivation or curve under `docs/derivations/`
- `reset_epsilon=2.0` → rationale or calibration artifact
- `flow_substeps=24` → provenance note (LGSSM T50 tuning 2026-09-03)
- Any setting claimed as "default" has a justification that is not "inherited"
  or "convenient"

**How:** Ledger file mapping (parameter_name, model, route) → (artifact_path,
justification_type). Test: every Class C usage in claim-bearing code appears in
ledger.

**Coverage target:** All Class C parameters per Safety Guardrail Reversed Burden
policy.

---

### Class 6: Seed Determinism and Reproducibility

**What:** Every "frozen seed" or "deterministic" claim is verified with a
regression fixture.

**Examples:**
- `_lgssm_frozen_observations()` with no args → hash matches recorded value
- `tf.random.Generator.from_seed(81100)` produces identical sequence on re-run
- Seed-from-theta function: same θ → same seed, different θ → different seed
- HMC seed-freeze: force at (θ, step_i) and (θ, step_j) uses identical noise

**How:** Golden-hash tests for fixtures. Determinism: run twice, assert
bitwise-identical output. Seed-freeze: instrument and compare RNG state.

**Coverage target:** Every fixture, every "frozen" or "deterministic" claim.

---

### Class 7: Policy-Compliance Gates

**What:** Every policy-mandated check is mechanized so a plan violating it
cannot reach execution.

**Examples:**
- Heuristic Dominance Gate: Phase 4/5 plan schema requires `adversary_set` with
  ≥3 entries and `salient_situations` with ≥2 entries
- Evidence Contract: plan schema requires `primary_criterion`,
  `promotion_vetoes`, `continuation_vetoes` before Phase 3+ execution
- Class C parameter: usage triggers ledger lookup; missing entry → fail
- Pfor approval: `tf.vectorized_map` in git diff → CI fails unless
  `docs/approvals/pfor_{module}_{date}.md` exists

**How:** Plan schema validation (JSON Schema or dataclass). Code gates: precommit
hooks, CI lint rules. The violation is a test failure, not a reminder.

**Coverage target:** Every "must" / "required" / "forbidden" in CLAUDE.md and
AGENTS.md.

---

## Proposed Phase 2C: Contract-Integrity Test Suite

Insert between Phase 2B (unification complete) and Phase 3 (damping derivation).

**Why here:** Phase 2B produces one unified engine with full capability union.
That is the natural checkpoint to verify the whole contract stack before any
scientific runs. Doing it earlier means testing against a forked/incomplete
implementation; doing it later means Phase 3/4 findings require backtracking.

**Duration:** 2–3 days

**Deliverables:**

1. **Class 1–7 test modules**, one per class, under `tests/contracts/`:
   - `test_enum_validation.py`
   - `test_wiring_resolution.py`
   - `test_shape_consistency.py`
   - `test_artifact_schema.py`
   - `test_parameter_provenance.py`
   - `test_seed_determinism.py`
   - `test_policy_compliance.py`

2. **Contract ledgers** (machine-readable):
   - `bayesfilter/contracts/reset_policy_enum.json` — legal values
   - `bayesfilter/contracts/entry_points_registry.json` — expected callables
   - `bayesfilter/contracts/class_c_parameter_ledger.json` — every Class C param
     with its (artifact_path, justification_type)
   - `bayesfilter/contracts/tuning_artifact_index.json` — (model, horizon,
     route) → artifact directory
   - `bayesfilter/contracts/fixture_hashes.json` — every frozen fixture with its
     SHA-256

3. **Plan schema** (`docs/schemas/experiment_plan.schema.json`) enforcing:
   - Phases 3+ require `evidence_contract` with all six fields
   - Phases 4+ require `heuristic_adversary_set` (≥3 entries) and
     `salient_situations` (≥2)
   - Every "Baseline: X" in `evidence_contract` must resolve via wiring test
   - Every `measured` claim must map to an artifact field

4. **Validation runner** (`scripts/validate_contracts.py`) that:
   - Runs all seven test classes
   - Validates all experiment plans against schema
   - Checks artifact directories against their schemas
   - Returns exit code 1 on any violation
   - Used in CI and as a pre-execution gate

5. **Result document** (`docs/plans/ledh-contract-integrity-suite-result-2026-09-06.md`)
   recording:
   - Coverage: X enums validated, Y wiring paths checked, Z fixtures hashed
   - Violations found and repaired
   - Ledger and schema locations
   - How to add a new contract when extending the code

**Promotion criteria:**

1. All 12 audit findings have a corresponding test that would catch them
2. `reset_policy` enum test fails on `"transport_affine_cumulant_trust"`
3. Fixture hash test fails if `make_observations()` returns wrong data
4. Parameter resolution test fails if runner passes non-existent kwarg
5. Full suite green on unified engine
6. Schema validation rejects a Phase 4 plan with no adversary set
7. CI runs `validate_contracts.py` and fails the build on violations

**Promotion vetoes:**

- Any of the 12 audit findings not covered by a test
- Enum/wiring/schema test that only warns instead of failing
- Ledger incomplete (missing a Class C param in claim-bearing code)

**Budget:** 2–3 days CPU work, no GPU. Independent of Phase 3's mathematical
derivations — those can proceed in parallel if desired, but nothing executes
until the contract suite is green.

---

## What this fixes going forward

**Before:** Document claims "X is implemented" → discover at execution that X
uses wrong parameters / falls to unrecognized branch / references nonexistent
baseline.

**After:** Document claims "X is implemented" → wiring test confirms callable
exists and resolves, enum test confirms all branches ledgered, artifact test
confirms baseline file present with required schema. Violation is a test
failure before execution, not a silent wrong answer after.

**Before:** New phase plan omits adversary set → executes → result is
uninterpretable because no heuristic comparison.

**After:** Plan omits adversary set → schema validation fails → plan cannot
proceed to execution until corrected.

**Before:** Class C parameter transferred from another model → used without
calibration → result affected by uncalibrated numerics-altering control.

**After:** Parameter usage triggers ledger lookup → no entry or justification
type is "inherited" without source-target similarity argument → test fails →
calibration owed before usage.

---

## Alternative: lighter-weight "smoke contracts" first

If 2–3 days is too much before Phase 3, a minimal viable gate is:

**Day 0.5: Critical triad**
1. Enum validation (Class 1) for `reset_policy` and `ENTRY_POINTS`
2. Fixture hash (Class 6) for `_lgssm_frozen_observations`
3. Parameter resolution (Class 2) for every runner in `docs/benchmarks/`

That directly prevents the four blockers and costs half a day. Full Class 1–7
suite becomes its own phase later.

---

## Recommendation

**Add Phase 2C (full suite) after Phase 2B, before Phase 3.** The unification
checkpoint is the natural place to verify the whole contract stack, and 2–3
days of prevention is cheaper than one more round of "confident wrong answer"
findings after a 12-GPU-hour Phase 4 run.

If budget is tight, run the critical triad now (0.5 day), defer the full suite
to after Phase 5. But the enum + fixture + parameter-resolution tests are
load-bearing — without them we're one unrecognized string away from another
silent failure.

---

**END OF DESIGN NOTE**
