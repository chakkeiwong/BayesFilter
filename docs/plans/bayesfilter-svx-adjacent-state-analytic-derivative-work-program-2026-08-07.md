# SVX adjacent-state analytic derivative implementation program (revised)

Date: 2026-08-07
Status: `ACTIVE_GOVERNING_PROGRAM`

## Program purpose

This program governs the active `SVX-ZC` repair effort so that the work does not
wander between target changes, convenience analytic routes, test-only activity,
or premature tuning reruns.

The governing objective is narrow and fixed:

> implement the analytic adjacent-state derivative for the **current active
> SVX-ZC frozen-core finite value program**, verify it against the same active
> value authority, and only then resume serious `SVX-ZC` tuning.

This is a same-program score-backend repair. It is **not** a target redesign,
route-selection exercise, or permission to substitute a nearby transformed-SV
analytic lane.

## Research intent ledger

### Main question

Can we replace the active autodiff HMC score path in
`bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` with an analytic
adjacent-state derivative for the **same** active finite program, while
preserving the current frozen-core value path and target contract?

### Candidate / mechanism under test

A derivative-aware replay of the active one-axis and adjacent two-axis fixed-TT
program using:

- explicit target-log derivatives,
- derivative-aware LSQ / ALS solves,
- squared-TT normalizer derivatives,
- retained-marginal derivative propagation,
- and the same UKF-frozen initial / adjacent core identity already used by the
  active route.

### Expected failure mode

The most likely failure is not numerical noise alone but a **program mismatch**:
for example, accidentally validating a different transformed-SV route, omitting
adjacent-state design dependence, or reproducing only a scalar fixed-design
surrogate rather than the active batched adjacent-state program.

### Promotion criterion

Promotion from implementation to integration is allowed only if all of the
following are true:

1. the active value authority is unchanged;
2. the new score backend is finite on the reviewed probe batch;
3. centered finite differences of the **same active value program** agree with
   the analytic score within the declared test tolerances;
4. adjacent frozen-core sensitivity is preserved in the value path;
5. focused SVX tests pass;
6. shared-procedure and end-to-end contract tests pass.

### Promotion vetoes

Any one of the following blocks promotion:

- any evidence that the value path changed relative to the active authority;
- any use of the transformed-SV independent-panel analytic route as if it were
  the same target;
- nonfinite score/value on the reviewed probe batch;
- failed finite-difference agreement on the same active finite program;
- failed adjacent frozen-core sensitivity preservation;
- failed focused SVX or contract tests.

### Continuation vetoes

Stop the program and escalate only if one of these happens:

- the derivation is shown to be wrong relative to the active finite program;
- the active target contract is discovered to be inconsistent or stale;
- the required derivative path cannot be expressed with the current frozen-core
  program without changing the target;
- a required destructive or outward-facing action becomes necessary.

A failed implementation attempt, failed local test, or failed first derivative
candidate is **not** by itself a continuation veto; it triggers the next repair
phase below.

### Repair trigger

If any focused derivative check fails, repair must target the first failing
layer in this order:

1. target-log derivative,
2. one-axis fit derivative,
3. adjacent ALS design derivative,
4. normalizer derivative,
5. retained-marginal derivative propagation,
6. adapter/registry integration.

### Explanatory diagnostics

These explain failures but do not by themselves promote the route:

- local condition proxies from LSQ/ALS solves,
- probe-batch score magnitudes,
- smoke-call telemetry,
- XLA compilation status,
- runtime/memory observations.

### What must not be concluded

Even after a successful repair, do **not** conclude:

- posterior correctness beyond the finite program already claimed;
- HMC convergence or production readiness;
- superiority relative to other SV routes;
- broad SVX default readiness;
- cross-route equivalence to the transformed-SV independent-panel analytic lane.

## Current baseline facts

The program starts from these established facts.

1. The active route still uses autodiff in
   `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
2. The active value program depends materially on the adjacent frozen UKF cores
   beginning at `t=1`.
3. The nearby transformed-SV analytic lane is not the same finite program and
   is therefore not an admissible replacement baseline.
4. The active registry should remain on the last known-good target until the
   same-program analytic backend passes the gates in this program.

## Phases

### Phase 0 — Freeze the question and baseline

#### Goal

Prevent drift before more implementation.

#### Required outputs

- This governing program.
- A confirmed statement of the active target, active value authority, and banned
  substitute route.

#### Required checks

- Confirm the active score backend is the autodiff path in
  `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
- Confirm the active adapter wiring in
  `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`.
- Confirm the transformed-SV analytic candidate is treated only as a rejected
  comparison route, not as an implementation baseline.

#### Exit gate

Exit Phase 0 only when the target question is frozen as:

> same-program adjacent-state analytic derivative for the active SVX-ZC finite
> program.

---

### Phase 1 — Derivation lock

#### Goal

Lock the mathematics of the active finite program before code changes.

#### Inputs

- `docs/plans/bayesfilter-svx-adjacent-state-analytic-derivative-derivation-2026-08-06.md`
- the active source implementation and helper references.

#### Required outputs

A derivation note that explicitly covers:

- time-0 target and derivative,
- `t>=1` adjacent-state target and derivative,
- derivative-aware ALS system,
- normalizer derivative,
- retained-marginal derivative recursion,
- final accumulated score formula.

#### Required checks

- verify that every derivative term corresponds to the active finite program,
  not the nearby transformed-SV analytic route;
- identify which helper algebra is reused from `filtering.py`,
  `derivatives.py`, and `zhao_cui_moment_teacher_als.py`;
- explicitly name the missing new artifact: adjacent-state design-aware ALS
  derivative propagation in the active batched program.

#### Exit gate

No unresolved mismatch between the derivation and the active program structure.

---

### Phase 2 — Skeptical derivation audit

#### Goal

Red-team the derivation before implementation.

#### Required outputs

A short audit note or appended section recording:

- checked assumptions,
- hidden-default risks,
- math-to-code mapping,
- any remaining unproved but isolated implementation questions.

#### Required tools

Use the local MathDevMCP CLI where applicable for:

- assumptions-for,
- derive-from / debug-derivation,
- audit-math-to-code.

#### Promotion role

This phase is a **continuation gate**, not a promotion gate.

#### Exit gate

Either:

- derivation audit passes with bounded caveats and implementation may start; or
- a material derivation flaw is found and must be repaired before coding.

---

### Phase 3 — Active-module derivative implementation

#### Goal

Implement the same-program analytic score in the active module.

#### Primary file

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`

#### Scope

Allowed:

- factoring the active value path into helper functions;
- adding derivative helpers for target logs, LSQ/ALS, normalizers, and retained
  marginals;
- replacing only the score backend of the active module.

Not allowed:

- changing the target semantics;
- swapping in the independent transformed-SV analytic lane;
- moving the registry to a new backend before tests pass.

#### Required implementation subphases

##### Phase 3A — one-axis step derivative
- implement and verify `t=0` target-log derivative;
- implement and verify one-axis fit derivative;
- implement and verify one-axis normalizer derivative.

##### Phase 3B — adjacent-state target derivative
- implement `t>=1` target-log derivative using the active retained marginal,
  transition term, and observation term;
- verify value-path invariance.

##### Phase 3C — adjacent ALS derivative replay
- propagate derivatives through the active `(0, 1, 1, 0)` sweep order;
- include design-matrix tangents where the active program requires them;
- include the derivative of the active scaled ridge solve used by
  `_scaled_qr_solve(...)`, or explicitly freeze and justify the scaling branch if
  that is the reviewed implementation contract;
- verify against the locked derivation.

##### Phase 3D — retained recursion derivative
- propagate retained normalized log-density derivatives into the next time step;
- verify quotient-rule consistency and finite behavior.

#### Exit gate

The active module returns finite same-program scores on the reviewed probe batch
without changing the value path.

---

### Phase 4 — Focused verification

#### Goal

Demonstrate that the repaired backend differentiates the same active finite
program.

#### Required test files

- `tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py`
- `tests/test_zhao_cui_actual_sv_neutra_target.py`

#### Required checks

1. finite value and score on 2–3 probe rows;
2. unchanged value relative to the current active value authority;
3. centered finite-difference agreement for the same active finite program;
4. preserved adjacent frozen-core sensitivity;
5. required telemetry/status keys present;
6. tiny smoke batch succeeds.

#### Decision rule

A failed focused check blocks integration but does **not** reject the research
question. It triggers a localized repair in Phase 3.

#### Exit gate

All focused SVX checks pass.

---

### Phase 5 — Integration and contract verification

#### Goal

Wire the repaired backend into the serious route only after focused proof of
same-program correctness.

#### Allowed files

- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
- `bayesfilter/testing/neutra_model_registry_tf.py`

#### Required checks

- active adapter advertises the repaired backend accurately;
- serious-route metadata says non-autodiff only if true;
- shared procedure and end-to-end contract tests pass;
- registry still points to the intended target contract.

#### Exit gate

Integration tests pass with no target-scope drift.

---

### Phase 6 — Isolated SVX-ZC tuning rerun

#### Goal

Resume serious tuning only after the derivative repair is admitted.

#### Scope

- rerun only `SVX-ZC`;
- use the 4080-only mask;
- use a fresh output root;
- keep the broader sweep frozen.

#### Required artifact

A fresh experiment/result note recording:

- exact command,
- environment,
- repaired backend identity,
- diagnostics,
- decision and next step.

#### Exit gate

Either a viable-set artifact is produced, or a new mathematically interpretable
failure is documented.

## Drift controls

To keep the work from drifting, apply these rules throughout all phases.

### Drift control 1 — one question only

Any activity must answer one of these:

- does it help implement the same-program adjacent-state derivative?
- does it test that derivative against the same active value path?
- does it integrate the admitted derivative into the serious route?

If not, it is out of scope.

### Drift control 2 — no substitute route promotion

The transformed-SV independent-panel analytic route may be cited only as a
rejected comparator. It may not be promoted into the active route, test oracle,
or serious backend.

### Drift control 3 — no tuning before admission

Do not rerun serious `SVX-ZC` common tuning until Phases 0 through 5 pass.

### Drift control 4 — value authority is frozen

The active value path is the authority. Score work may change only the score
backend, not the value semantics.

### Drift control 5 — fail at the first unpassed gate

When a phase gate fails, stop advancing phases. Repair the failing layer first
and record the failure class.

## Minimal decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue same-program SVX repair | active adjacent-state analytic derivative not yet implemented | no continuation veto currently known | exact active ALS derivative replay still unwritten | execute Phase 1/2/3 in order | no HMC readiness or route superiority claim |
| Reject transformed-SV analytic swap as fix | active-value equality fails after adjacent branch activates | veto supported by existing diagnostics | none material for this decision | keep it as rejected comparator only | does not say that route is wrong in its own scope |
| Defer serious tuning rerun | repair-admission gates not yet passed | veto active until Phases 4-5 pass | implementation still incomplete | finish derivative repair and verification first | no statement about final SVX viability |

## Immediate next action

The immediate next action under this program is:

1. keep the active value path unchanged;
2. refine the derivation so the LSQ / normalizer chain is tool-certifiable;
3. implement the Phase 3 derivative replay in
   `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`;
4. do not resume tuning until Phase 5 is passed.
