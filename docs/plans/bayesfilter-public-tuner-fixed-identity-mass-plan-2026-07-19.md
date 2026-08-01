# Public BayesFilter Tuner: Fixed-Identity Mass Policy

Date: 2026-07-19

Status: `PLAN_READY_FOR_IMPLEMENTATION`

## 1. Research intent and evidence contract

| Field | Decision |
| --- | --- |
| Main question | Can the public `bayesfilter.inference.tune_hmc_kernel` API support NeuTra's fixed identity mass in transport coordinates without a second tuner or a second convergence controller? |
| Candidate mechanism | A public, opt-in mass policy that constructs an identity covariance artifact in the caller's coordinates and carries that immutable artifact through native step-size/trajectory tuning and sequential rank/folded-R-hat verification. |
| Comparator | Existing public tuner with its current default mass/geometry and windowed-mass adaptation behavior. |
| Primary promotion criterion | The fixed-identity call completes through the public tuner, emits a valid final-kernel handoff with unchanged identity-mass signature, and reaches the existing sequential verification policy without a fixed 1,000-draw R-hat veto. |
| Hard vetoes | API/config ambiguity, invalid or mismatched mass artifact, any mass signature change under the fixed policy, nonfinite target/trace/status/energy diagnostics, failed native acceptance policy, failed sequential R-hat/ESS at the declared cap, failed artifact lineage, or default-path regression. |
| Explanatory diagnostics | Runtime, acceptance distance from 0.70, candidate count, mass condition summaries, and per-check R-hat trajectory before the final gate. These may explain or nominate repairs but do not establish scientific correctness or superiority. |
| Nonclaims | This change does not prove NeuTra posterior correctness, sampler superiority, convergence beyond the declared diagnostics, or readiness for every model. It does not promote identity mass as the repository-wide default. |
| Result artifact | Versioned plan-attempt root under `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/`, with config, public tuning artifact, sequential verification checkpoints, manifests, tests, and a terminal result note. |

## 2. Scope and design boundary

The current public tuner already owns geometry initialization, bootstrap,
windowed-mass staging, fixed-mass step/trajectory tuning, Phase 7 repair, and
sequential rank-normalized/folded-R-hat verification. The current NeuTra
campaign instead calls `tune_fixed_transport_hmc_kernel`, which performs a
separate fixed-kernel verification and can reject a candidate after a fixed
1,000-result screen. That route is not the canonical public tuning procedure.

This program changes the public tuner and its internal phase handoff only. It
does not add another HMC kernel, tuner, mass constructor, or convergence
implementation. The existing adaptive/windowed mass policy remains the default.

The proposed public option is:

```text
HMCKernelTuningConfig.mass_policy = "windowed_adaptive"  # existing default
HMCKernelTuningConfig.mass_policy = "fixed_identity"    # explicit NeuTra arm
```

The policy is interpreted in the coordinates of `initial_position` and binds
the adapter signature, dimension, position role, covariance source, and mass
artifact signature. `fixed_identity` means covariance `I_d` and no empirical
mass replacement. It does not mean that step size, leapfrog count, or the
sequential convergence gate are disabled.

No caller-supplied identity array is required for this option. The public
function must construct and validate the artifact through the repository-owned
mass factory. Supplying `initial_covariance=I` to the existing API is not an
equivalent implementation because the current later windowed stage can replace
that mass.

## 3. Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Promotion status |
| --- | --- | --- | --- | --- | --- |
| Keep `windowed_adaptive` as default | Current `tune_hmc_kernel` behavior | Avoids a silent global behavior change | A caller may assume fixed mass when it is not requested | Existing public API regression tests and artifact diff | Existing default |
| Add `fixed_identity` as an opt-in enum | NeuTra transport coordinates and existing identity fallback | Makes the required geometry explicit and reusable | Policy could be accepted but not propagated to Phase 4/5/7 | End-to-end mass-signature lineage test | Reviewed new option |
| Identity covariance `I_d` | Existing `_identity_hint` semantics | Exact, deterministic, no hidden learned geometry | Poor conditioning or acceptance for a target that needs scaling | Native acceptance/energy/status screens | Baseline arm, not a universal recommendation |
| Keep target acceptance `0.70`, band `[0.65, 0.75]` | Repository policy and current plans | Preserves reproducible sampler tuning contract | Very high/low acceptance can be mistaken for convergence | Native acceptance telemetry and repair classification | Fixed reviewed policy |
| Use native sequential verifier for R-hat/ESS | `build_sequential_rhat_hmc_verifier` and existing public tuner | It owns adaptive checks and the 10,000-draw bound | A short check could still be misread as posterior proof | Checkpoint schedule and cap metadata | Existing canonical mechanism |
| NeuTra coordinates for mass, raw/model coordinates for final diagnostics | Transport target definition | Identity mass is meaningful in `z`, while scientific checks remain in target coordinates | Coordinate mismatch can invalidate the result | Adapter and transform signatures plus coordinate labels | Required invariant |
| GPU/XLA for serious execution | Repository execution policy | Matches BayesFilter production target | Sandbox/device mismatch | Trusted GPU probe, memory-growth record, XLA metadata | Required for serious run |

## 4. Skeptical pre-execution audit

The plan must pass this audit before implementation or a long run begins.

- **Wrong baseline:** compare the new fixed-identity policy with the current
  public default on the same tiny target; do not compare it only with the
  campaign-private tuner.
- **Proxy promotion:** acceptance and early R-hat are health/tuning evidence;
  only the native sequential verifier's declared final checks can admit the
  kernel, and posterior truth-tail checks remain downstream evidence.
- **Mass semantics:** verify whether `PrecomputedMassArtifact.covariance` is
  the object consumed as the HMC mass/preconditioning covariance. Record the
  source and matrix convention in the artifact; do not infer it from an array
  name.
- **Hidden propagation:** inspect every phase that can rebuild or reset mass.
  The fixed policy must either bypass that phase's replacement or return an
  equivalent frozen handoff with the same signature.
- **Stopping rule:** no fixed 1,000-draw R-hat hard veto may remain on the new
  public route. Sequential verification must be allowed to check in declared
  chunks through its cap.
- **Default regression:** the unchanged adaptive route must retain its current
  artifact schema/semantics and pass its focused tests.
- **Duplication:** the implementation must not copy code from
  `fixed_transport_hmc_tuning.py` into the public tuner. That module becomes
  historical/compatibility evidence only after migration and must not be the
  new NeuTra execution route.
- **Environment mismatch:** CPU-only tests are reference/smoke evidence only;
  serious NeuTra execution requires the trusted GPU/XLA policy.

Audit record required before implementation: `phase0-skeptical-audit.md` with
the inspected files, decisions, unresolved risks, and verdict.

## 5. Phase program

### Phase 0: contract and API design audit

Objective: freeze the public option name, mass convention, artifact identity,
default compatibility, and exact phase handoff behavior.

Entry conditions: this plan exists; current public tuner and specialized
NeuTra tuner have been inspected; no scientific target or acceptance policy is
being changed.

Required artifacts: `phase0-skeptical-audit.md`, API decision table, mass
lineage diagram, and refreshed Phase 1 subplan.

Required checks/reviews: inspect public exports, config payloads, geometry
factory, windowed stage, Phase 5/6/7 handoffs, verifier, and existing tests;
run focused import/compile checks; review for default-path and schema drift.

Evidence contract: a written proof that `fixed_identity` is implemented in
the public tuner rather than in a campaign wrapper, plus a list of all places
where an empirical mass update must be bypassed.

Forbidden claims/actions: no code change, no result claim, no deletion of the
specialized module, and no serious GPU run.

Handoff: proceed only when the mass convention and API behavior are explicit,
the default remains unchanged, and no missing phase owner is found.

Stop conditions: unresolved mass convention, incompatible public API contract,
or a required behavior change that needs user direction.

### Phase 1: public policy and identity implementation

Objective: add `mass_policy` to the public config and make the repository-owned
identity mass factory produce a validated artifact bound to the actual adapter.

Entry conditions: Phase 0 audit passes; API name and default are frozen.

Required artifacts: updated config/payload schema, identity-mass factory or
shared helper, public exports if needed, and Phase 1 close record.

Required checks/reviews: unit tests for enum validation, default payload
compatibility, identity covariance/dimension/SPD checks, adapter-signature
binding, deterministic artifact hash, and rejection of unsupported values;
compile/import checks; focused review of API surface and NumPy boundary.

Evidence contract: tests demonstrate that the public tuner can request a
fixed identity artifact without importing or calling the specialized tuner.

Forbidden claims/actions: do not make fixed identity the default; do not
change target acceptance, R-hat thresholds, or mass semantics; do not expose
raw arrays in public progress records beyond existing artifact rules.

Handoff: identity config and artifact pass all focused tests and the next
subplan is refreshed with exact propagation points.

Stop conditions: artifact cannot be validated, default payload changes
unexpectedly, or the implementation requires duplicated sampler logic.

### Phase 2: fixed-policy phase handoff integration

Objective: carry the frozen identity artifact through bootstrap, windowed-mass,
fixed-mass step/trajectory, repair, and Phase 7 verification without mass
replacement.

Entry conditions: Phase 1 identity policy and factory pass.

Required artifacts: phase handoff payloads with `mass_policy`, initial/final
mass signatures, update count, and explicit `mass_frozen` invariant; Phase 2
close record; refreshed Phase 3 subplan.

Required checks/reviews: tests that fixed policy has zero empirical mass
updates, unchanged signatures across all handoffs and retries, correct
coordinate/adapter binding, and unchanged adaptive behavior when policy is
`windowed_adaptive`; tests for repair after acceptance failure and for hard
veto on signature mutation.

Evidence contract: a passed fixed-policy handoff is engineering evidence only;
it must show the same public phase machinery was used and that no private
fallback route executed.

Forbidden claims/actions: do not call the specialized tuner as a fallback;
do not silently downgrade to identity after a failed supplied geometry; do not
label a frozen mass as a posterior-optimal or scientifically superior mass.

Handoff: all phase-level mass signatures agree and the adaptive default has no
regression; otherwise repair and rerun focused checks.

Stop conditions: any hidden mass mutation, route fallback, or inability to
preserve the existing Phase 7 verifier contract.

### Phase 3: canonical NeuTra migration

Objective: migrate `neutra_end_to_end.py` to call `tune_hmc_kernel` with the
explicit fixed-identity policy and remove its campaign-private tuning gate.

Entry conditions: public fixed-policy handoffs pass; preserved frozen
transport artifacts remain hash-addressable.

Required artifacts: migrated runner, updated campaign config/manifest,
focused migration tests, and Phase 3 close record.

Required checks/reviews: prove the runner imports/calls only the public tuner
for tuning; prove target acceptance is `0.70` with band `[0.65,0.75]`; prove
the public sequential verifier owns warm-up/R-hat/ESS through 10,000 per-chain
caps; verify transport and target signatures; run static duplicate-route scan.

Evidence contract: a tiny fixed-identity NeuTra smoke must reach native
sequential verification or produce a genuine target/health veto, never a
private fixed-1,000-draw R-hat veto.

Forbidden claims/actions: do not retrain preserved transports; do not claim
posterior validity from the smoke; do not delete historical artifacts or
rewrite prior results.

Handoff: migration tests pass and a smoke artifact records the canonical public
route, mass policy, coordinate system, and verifier schedule.

Stop conditions: public tuner cannot represent the fixed transport target,
identity mass changes, or the smoke enters a private/legacy tuner route.

### Phase 4: focused LGSSM validation

Objective: rerun LGSSM using the preserved frozen transport and the repaired
public tuner, separating native acceptance/health tuning from adaptive
warm-up and retained convergence.

Entry conditions: Phase 3 migration smoke passes; trusted GPU/XLA and memory
growth are available; output root is fresh and versioned.

Required artifacts: run manifest, public tuning artifact, sequential warm-up
checkpoints, retained convergence artifact, sample archive, truth-tail result,
phase result, and next-subplan refresh.

Required checks/reviews: finite/status/energy checks; acceptance in
`[0.65,0.75]`; warm-up folded/rank-normalized R-hat checks from the native
controller starting at its declared minimum and extending to 10,000; retained
R-hat/ESS checks through 10,000; truth-tail checks only after a valid retained
sample; terminal red-team review.

Evidence contract: distinguish tuning admission, warm-up sufficiency,
retained convergence, and truth recovery. Preserve all failed attempts and
record repair/retry without overwriting evidence.

Forbidden claims/actions: no claim that one LGSSM run proves universal NeuTra
validity, no ranking by runtime or acceptance alone, and no promotion if a
hard veto or missing artifact remains.

Handoff: either a complete LGSSM terminal result exists or a precise
target/health/infrastructure blocker is recorded for the next planned repair.

Stop conditions: target invalidity, nonfinite gradients/values, mass identity
mutation, missing sequential diagnostics, exhausted campaign budget, or a
scientific-contract change.

### Phase 5: broader model smoke and terminal review

Objective: apply the same public route to the remaining executable registry
cells only after LGSSM validates the repaired mechanics, then write a terminal
decision without overstating one-seed evidence.

Entry conditions: Phase 4 closes; model registry and target signatures are
unchanged or explicitly re-baselined.

Required artifacts: per-cell manifests/results, aggregate decision table,
inference-status table, terminal review, reset memo, and preserved blocked
inventory.

Required checks/reviews: same hard-veto order for every cell; no private tuner
imports; fresh output roots; one-seed policy as diagnostic evidence only;
reviewer availability is advisory and cannot override numerical vetoes.

Evidence contract: report viable screens, hard failures, descriptive runtime
differences, statistically unsupported rankings, and the next evidence needed.

Forbidden claims/actions: no default promotion, superiority claim, or broad
scientific conclusion from a short campaign; no rerun that changes target,
hardware class, or budget without a new plan.

Handoff: terminal result and reset memo state exactly which cells reached
sequential sampling and which did not.

Stop conditions: shared harness regression, target signature drift, exhausted
budget, or any material missing artifact.

## 6. Repair and continuation procedure

At the end of every phase:

1. Run the phase's local checks and record the exact command and result.
2. Write a phase close record with decision, evidence, vetoes, uncertainty,
   nonclaims, and remaining budget.
3. Draft or refresh the next phase subplan using the actual artifacts and
   unresolved risks.
4. Review that next subplan for consistency, mathematical correctness,
   feasibility, artifact coverage, and boundary safety.
5. Continue automatically after a localized implementation, harness,
   serialization, multiprocessing, or resource repair when target, method,
   criteria, vetoes, hardware class, privacy boundary, and campaign budget are
   unchanged. Use a fresh attempt directory and record the repair and focused
   regression.

Stop for direction only when the scientific contract, target, data, hardware
class, budget, privacy boundary, or promotion criteria would change, or when
a true continuation veto fires. A candidate failure is not by itself a
research-direction failure.

## 7. Planned code and test surface

Expected implementation files (subject to Phase 0 audit):

- `bayesfilter/inference/hmc_kernel_tuning.py`: public policy, identity mass
  construction, phase handoff/freeze behavior, and artifact payloads.
- `bayesfilter/inference/hmc.py` only if the existing sequential verifier needs
  a narrowly scoped public-policy parameter; no new verifier implementation.
- `bayesfilter/inference/neutra_end_to_end.py`: call the public tuner and pass
  `mass_policy="fixed_identity"`.
- Focused tests in `tests/test_hmc_kernel_tuning_*` and
  `tests/test_neutra_all_models_end_to_end_contract.py`.

Required test groups:

1. Public config/API validation and backward-compatible default payload.
2. Identity mass artifact construction, signature, coordinate binding, and
   serialization.
3. Fixed-policy zero-update/frozen-signature invariants across all tuning
   phases and repairs.
4. Adaptive-policy regression and existing native sequential verifier tests.
5. NeuTra route scan proving no import/call to
   `tune_fixed_transport_hmc_kernel` remains in the active runner.
6. Tiny deterministic end-to-end smoke with bounded CPU/reference settings;
   serious GPU/XLA evidence remains a separate campaign phase.

## 8. Review and execution records

The following records must be created under the plan's artifact root:

- `phase0-skeptical-audit.md`
- `phase1-close.md` and `phase1-next-subplan.md`
- `phase2-close.md` and `phase2-next-subplan.md`
- `phase3-close.md` and `phase3-next-subplan.md`
- `phase4-result.md` and `phase4-next-subplan.md`
- `terminal-result.md` and `reset-memo.md`

Claude, if used, is a read-only reviewer of one bounded plan or result path at
a time. It cannot authorize execution or override numerical, budget, or
scientific vetoes. Reviewer timeout is recorded and does not block routine
local implementation when the focused checks and evidence contract pass.

## 9. Decision summary

This is a public-tuner extension and migration plan, not permission to run the
old private tuner again. The old fixed-transport tuner may remain readable as
historical evidence during migration, but the active NeuTra route must have
one owner: `tune_hmc_kernel`, with its native sequential verification and the
new explicit fixed-identity mass policy.
