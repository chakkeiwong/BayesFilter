# q=20 Factor-Route Fresh Tuning and Numerical Admission Plan

Date: 2026-09-04  
Status: `A3_FACTOR_BACKEND_ADMITTED_PHASE9B_BLOCKED`
Parent result: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-result-2026-09-04.md`  
Master authority: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Planned result: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md`  
Artifact root: `docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/`

## Research intent and claim boundary

The immediate question is whether the already parity-qualified backend
`tensorflow_eigh_strict_factor_cached` can build fresh q=20 charts, tune all
six `(chart,beta)` scopes, issue identity-consistent handoffs, and execute the
Phase 9A replica-exchange mechanics check under its own identity and seeds.

This is a numerical-backend admission campaign. A complete pass, combined
with the paired value/score/status and HMC-mechanics evidence in the parent
result, permits the factor backend to become the required q=20 Phase 9B
backend. It does not change the generic public API default, certify a trained
NeuTra posterior, or promote the whole sampler scientifically. Full NeuTra
promotion still requires Phase 9B sequential warmup, retained cold sampling,
modern convergence diagnostics, downstream posterior checks, comparator
evidence, and uncertainty analysis under a separately refreshed plan.

## Evidence contract

| Item | Predeclared meaning |
|---|---|
| Candidate | `tensorflow_eigh_strict_factor_cached`, q=20, float64, TensorFlow/TFP, XLA, TF32 enabled, GPU0, memory growth verified before initialization |
| Numerical baseline | `tensorflow_eigh_strict`, already paired against the candidate in the 2026-09-04 parent campaign; it remains the explicit fallback and comparator |
| Tuning protocol | Existing `measured_joint_grid_v1`: the original `(0.25,0.55,0.85,1.20)` grid failed on scope 1; R2 uses the measured repair grid `(0.055,0.060,0.070,0.075)`, leapfrog counts `(3,8)`, two selection replications, and fresh held-out verification |
| Primary promotion criterion | The original factor canary passes, then a source-synchronized full run completes all six scopes, every declared pair is attempted and measured, every scope issues a verified handoff, and the four-draw Phase 9A transition mechanics check passes |
| Promotion vetoes | Any backend/bridge/checkpoint/adapter/handoff identity mismatch; incomplete grid; missing or reused seed; nonfinite target/state/score/log acceptance, invalid target status, no movement, or energy hard veto in the selected candidate, its held-out verification, or the transition controller; graph retracing beyond the declared stable runner; unavailable allocator telemetry; allocator peak over 4 GiB; missing XLA/GPU0/memory-growth provenance; corrupt or incomplete artifact |
| Continuation vetoes | Invalid target or bridge assumptions; changed C5 target/data identity; harness unable to distinguish strict from factor; corrupted artifacts; changed hardware/privacy/method contract; aggregate GPU budget exhausted; or three consecutive localized infrastructure failures with no discriminating repair |
| Repair triggers | Timeout before the aggregate budget is exhausted, serialization/launcher error, retracing, resource envelope failure, or one failed candidate scope whose cause can be localized without changing the target, grid, promotion rules, or hardware class |
| Explanatory diagnostics | Runtime, acceptance, selected `(epsilon,L)`, ESS/gradient, pullback residuals, allocator use, and per-call compile/steady timing |
| Artifact | Fresh immutable attempt directories plus a terminal result note, audit receipt, and master-program reset memo |

No timing, acceptance rate, ESS value, R-hat value from the four-draw mechanics
check, whitening residual, or mode-travel observation can independently promote
the backend. No run here establishes IID Gaussian whitening, exhaustive mode
discovery, posterior correctness, convergence, sampler superiority, broad
dimension scaling, or a global BayesFilter default.

## Defaults and assumptions

| Choice | Provenance and status | Why retained here | Failure mode and early diagnostic |
|---|---|---|---|
| Factor backend | Passed paired q=20 numerical and mechanics tests in the parent campaign; candidate | Removes repeated strict factor eigendecompositions while retaining the checked derivative formula | Spectral sensitivity outside the earlier bank; canary target status, chart reliability, and every tuning call fail closed |
| Six scopes, betas `(0,.5,1)`, two charts | Frozen C5/Phase 9A protocol; baseline | Tests every handoff needed by the current three-temperature/two-chart transition | A partial pass masquerades as admission; full manifest must contain exactly scope indices `0..5` |
| Eight-pair grid | Inherited target-specific Phase 9A hypothesis, not a universal HMC default | Preserves the scientific comparison while retuning the changed backend | Grid misses viable mechanics or exhausts budget; complete-grid record and per-pair outcomes expose this |
| Two chart updates per beta | Inherited Phase 9A mechanics preflight baseline | This campaign tests backend compatibility, not transport-quality optimization | Poor charts can cause tuning failure; classify as chart/tuning evidence, never posterior rejection |
| Canary scope index 3 | Existing hardest localized chart-1/beta-0 scope; measured baseline | Cheapest known stress scope before six-scope spend | It may not predict every scope; therefore it only opens, never substitutes for, the full run |
| 1,800 s canary cap | Measured prior localized wall times near 495 s and historical 1,800 s envelope; reviewed cap | Allows compile variance while remaining bounded | Timeout is resource evidence and debits aggregate budget |
| 10,000 s full cap | User-selected campaign cap; factor timing forecast about 4,000--4,800 s typical and 9,000--10,900 s conservative | Covers the measured typical path and most conservative projections | Timeout preserves partial calls but cannot yield promotion |
| 4 GiB allocator cap | Inherited Phase 9A resource gate; prior measured peak about 1.4 GiB | Preserves the existing shared-GPU envelope | Missing or excessive peak telemetry is a hard veto |
| Fresh 20260904 seed namespaces `93xxx`/`94xxx` | New source-owned campaign choice | Separates canary calibration from full evidence and all historical runs | Contract tests and terminal seed audit reject overlap |
| R2 scale grid `(0.055,0.060,0.070,0.075)` | Measured diagnostic on the failed scope: both backends finite; acceptance 0.846, 0.666, 0.508, 0.525 at the four tested values (small-chain descriptive evidence) | Brackets the observed transition between 0.05 (0.956) and 0.10 (0.125) while retaining four distinct step candidates | Tiny diagnostic chains are not tuning evidence; the full grid must still measure every pair and apply the predeclared acceptance/status/movement vetoes |
| R2 seed namespace `97xxx` | Fresh source-owned continuation after the failed `94xxx` attempt and diagnostic-only `96xxx` seeds | Prevents partial handoff or diagnostic reuse | Seed-overlap audit rejects the run |
| R2 selection/verification draws `8/2` | Budget repair derived from measured 40--105 s cost of the prior 16/4 calls; still two independent selection replications and a separate held-out check | Allows all six scopes and the complete grid to run inside the remaining GPU budget | Shorter tuning chains increase Monte Carlo noise; this run cannot establish convergence, posterior quality, or a default HMC step |
| Source-synchronized profile `phase9a_factor_tuning_full_source_sync_v1` | Required after the HMC tuning contract/export source-closure changed; fresh `98xxx` roots and measured remaining-budget cap | Rebinds every tuning artifact to the current executable source without editing old hashes | One 4,000-second all-scope run is a budget-constrained terminal screen, not an independent replication |

The strict baseline and factor candidate are not rerun side-by-side in this
campaign. Their mathematical/mechanics comparison is the parent campaign;
this run asks a different question, namely whether fresh factor-bound tuning
and all downstream Phase 9A interfaces work. Consequently, this campaign does
not support a new stochastic performance ranking.

## Phases and repair closeouts

### A0: source binding and rejection tests

Add immutable factor canary/full profiles to the existing runner. Bind the
backend into profile payloads, scopes, checkpoint scopes, bridge construction,
run-start/failure/final manifests, and result routing. Restore each checkpoint
with an exact expected context so a strict checkpoint cannot enter a factor
run. The verified handoff constructor must continue to compare the current
factor adapter signature with the tuned signature. Add a dedicated launcher
that accepts only `canary` or `full` and maps them to exact profiles/scopes,
GPU0, memory growth, XLA, unique output directories, and their caps.

Run focused CPU contract tests. R0 records changed paths, verifies that all
historical profiles remain strict, verifies the factor canary/full/R2 profiles
and disjoint seeds, and confirms caller scope widening/backend substitution is
impossible.

### A1: factor canary

Run exactly:

```bash
PYTHON_BIN=/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  bash scripts/run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh canary
```

The profile is `phase9a_factor_tuning_canary_v1`, pinned to scope index 3 only.
R1 validates the manifest, backend identity through every checkpoint/scope and
tuning handoff, complete eight-pair coverage, seed isolation, device/memory/XLA
facts, stable tracing, status/energy/movement vetoes, and remaining aggregate
budget. A pass opens A2; a localized failure is repaired and retried only with
a fresh directory and within remaining budget.

### A2: six-scope factor campaign

Run exactly:

```bash
PYTHON_BIN=/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  bash scripts/run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh full
```

The profile is `phase9a_factor_tuning_full_v1`, pinned to all six scopes. The
full run uses fresh charts and seeds; it cannot read, merge, or reuse canary or
strict tuning calls. It must run the shared four-chain sequential controller
for four discarded warmup and four retained mechanics draws only after all six
handoffs pass.

R2 checks exact scope coverage, per-scope durable tuning artifacts, complete
candidate grids, selected-candidate and held-out verification, adapter and
handoff signatures, transition binding, target status, memory/tracing facts,
hashes, and wall time. Rejected grid pairs remain visible diagnostics; their
vetoes do not invalidate a different candidate that passes the source-owned
selection and held-out rules. Partial completion is preserved as failure
evidence and cannot be promoted.

### A3: decision, repair, and next-plan refresh

Write the terminal result with separate engineering, numerical, and scientific
ledgers, a decision table, an inference-status table, attempt accounting, and a
post-run red team. On a complete pass, mark the factor backend
`PROMOTED_Q20_PHASE9_NUMERICAL_BACKEND` and require it in the refreshed Phase
9B plan; retain strict as the explicit comparator/fallback. On failure, state
whether the harness, numerical implementation, chart/tuning candidate, or only
the resource envelope failed. Refresh the master program either way.

The next Phase 9B plan may be written after a pass, but no Phase 9B posterior
run is charged to or authorized by this campaign's GPU budget.

## Budget, attempts, and artifacts

The aggregate serious-GPU budget is 11,800 material seconds. The completed
canary and failed full attempt consumed about 3,129.29 seconds; the three
diagnostic receipts consumed 267.80 seconds of measured HMC time (plus their
bounded setup). The R2 full attempt is capped at 8,400 seconds, which remains
inside the measured remainder with a small accounting reserve. Every actual
failed or successful GPU second debits this aggregate. A retry is allowed only
if enough aggregate time remains and all scientific boundaries are unchanged.
Routine CPU tests and artifact inspection have a 600-second engineering
budget. No package, environment, data, target, hardware-class, or privacy
change is authorized.

Each launch creates a new directory below the artifact root. Attempts are
never overwritten. Every serious manifest must record Git revision and dirty
state, exact command/profile, environment, GPU visibility, TensorFlow version,
TF32/XLA, memory-growth verification, allocator peak, target/backend/bridge
identities, seeds, wall time, plan/result paths, scope records, and hashes.

## Pre-mortem

The campaign could falsely pass if the factor name appears only in a top-level
manifest while strict code executes; therefore bridge and adapter signatures
and checkpoint expected contexts are checked at execution boundaries. It could
pass mechanics while the transport is poor; chart-quality diagnostics remain
explanatory and Phase 9B is still required. It could fail because the grid or
tiny two-update chart baseline is weak rather than because factor reuse is
wrong; such a failure rejects admission under this protocol but not the factor
mathematics. It could time out after useful partial work; partial calls are
diagnostic only and cannot be combined with a later run.

## Skeptical pre-run audit

The first draft failed review in two material ways. It called a four-draw
mechanics transition an "untouched downstream screen," which could be mistaken
for posterior evidence, and it required backend identity without specifying
the missing checkpoint and bridge construction bindings. This revision names
the transition correctly, narrows the possible promotion to the q=20 Phase 9
numerical backend, reserves posterior promotion for Phase 9B, and specifies all
identity boundaries and rejection behavior.

The revised plan was then checked for wrong baseline, proxy promotion,
unsupported numeric constants, incomplete grid/scope evidence, stale seed and
artifact reuse, environment mismatch, hidden API-default change, missing stop
conditions, unfair comparison, and commands unable to answer the question.
The strict baseline is the paired parent evidence rather than the rejected raw
cache; descriptive diagnostics cannot promote; caps and grids have explicit
provenance; the factor run is fresh and complete-or-fail; GPU0/XLA/memory growth
are bound before import; and the claim boundary separates backend admission
from posterior science. Audit verdict:
`PASS_AFTER_REPAIR_READY_FOR_A0_IMPLEMENTATION`.

A final schema-level audit also found that the tuner preserves vetoes from
rejected grid pairs even when a separately selected pair passes. The promotion
rule now applies hard vetoes to the selected pair, held-out verification, and
transition controller, while requiring every rejected pair to remain in the
complete evidence table. This is a clarification of the predeclared selection
rule, not a post-result relaxation; it was repaired before the first canary.

## A0/R0/A1/R1 execution receipt

The source-binding repair and focused tests passed before GPU execution. The
fresh factor canary then completed with status
`PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL` in `1511.6580389949959` seconds. Its
independent audit is
`docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/`
`canary-attempt-20260904T150551Z/a1-r1-canary-audit.json` and reports
`PASS_FACTOR_CANARY_AUDIT`. It verified scope index 3, all eight measured
pairs, a passed held-out verification, unique fresh seeds, factor-bound
checkpoint/bridge/adapter identities, one trace per reusable runner, GPU0,
memory growth, and a 1,402,668,544-byte allocator peak. The run manifest hash
is `4f5fccd59b2e9f4b0a3f4116cb9ff39f1642976590366f5b51c690c30d05b206`; the
audit hash is `73292e1e53208e5bfc1a3bbaa09ce004eb17bd0222a37eae99a0a80bfc63ecb5`.

R1 therefore opens A2. The full profile remains pinned to all six scopes and
must use a new `full-*` output directory; no canary call or checkpoint may be
reused.

## A2/R2 first full-attempt closeout (2026-09-05)

The first A2 launch used the source-owned `phase9a_factor_tuning_full_v1`
profile in
`docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/`
`full-attempt-20260904T185852Z/`. It completed chart construction and scope
index 0 (`chart-0`, beta 0), including a durable handoff, then failed at scope
index 1 (`chart-0`, beta 0.5) after 1,617.6309049129486 seconds. All eight
declared grid pairs were attempted. The finite calls had nonfinite proposed
target values and zero movement; larger pairs also produced runtime
serialization errors because the runner call receipt rejected diagnostic
NaN/Inf values after the numerical call had already returned. This is
classified as `candidate_numerical_and_harness_serialization`, not a target
identity, GPU, or research-direction failure.

The failed attempt is immutable failure evidence. Its scope-0 handoff is not
combined with a later attempt, and no scope is promoted from this partial run.
The launch consumed approximately 1,617.63 seconds in addition to the
1,511.6580389949959-second canary, leaving approximately 8,670.71 seconds of
the 11,800-second aggregate campaign budget before any further material GPU
diagnostic. The estimate is accounting information, not a new timeout
guarantee.

R2 repair is now authorized under the unchanged q=20 target, factor method,
grid, promotion rules, GPU0/XLA/memory-growth contract, and privacy boundary:

1. Repair call-level serialization so finite and nonfinite diagnostics are
   represented by explicit JSON values (the canonical marker is
   `{"__nonfinite__":"nan|inf|-inf"}`), and always emit a durable failure
   receipt when post-call artifact writing fails.
2. Run a focused strict-versus-factor diagnostic on the failed scope's frozen
   chart at identical latent points and small HMC settings. It must report
   target value/score, transport log-determinant and pullback score finiteness,
   and one short call for each backend. The diagnostic is explanatory only and
   cannot promote either route.
3. If strict and factor agree on the failure, classify the issue as a
   scope/chart-training or proposal-scale failure and do not change the
   backend claim. If strict is finite while factor is not, stop the repair
   retry and repair the factor derivative implementation first. If the
   diagnostic itself is invalid, stop for a harness repair rather than spend
   the remaining campaign budget.
4. After the diagnostic and focused regression pass, a fresh source-owned
   repair profile may attempt the six scopes once, with a cap no larger than
   the measured remaining aggregate budget. The R2 profile is
   `phase9a_factor_tuning_full_r2_v1`, uses the fresh `97xxx` namespace, the
   measured step grid `(0.055,0.060,0.070,0.075)`, two selection replications
   and 8/2 selection/verification draws, and an 8,400-second cap. It must not
   read the failed or canary handoffs. A complete pass is still
   required for backend admission; a second localized numerical failure closes
   this campaign without promotion and refreshes Phase 9B as blocked.

The first four-step diagnostic completed with finite, backend-matched target
values. An extended scale diagnostic then found finite calls at 0.10, 0.15,
and 0.20 but acceptance below the repair band; the fine diagnostic measured
the R2 grid on the same frozen chart. These diagnostics are explanatory only.
They justify the repair grid as a bounded hypothesis, not as evidence that the
factor backend or the chart has been promoted.

The authorized R2 launch command is:

```bash
PYTHON_BIN=/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  bash scripts/run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh r2
```

## R2 final skeptical pre-run audit

The repair was re-audited before launch. The baseline remains the paired strict
backend, not the failed factor artifact; the factor identity is bound in the
profile, checkpoint, bridge, adapter, handoff, and manifest; all six scopes and
all eight grid pairs remain mandatory; and the smaller 8/2 draw counts are
recorded as a budget-constrained tuning hypothesis rather than a convergence
criterion. The scale values come from the frozen-chart diagnostic but its
short-chain acceptance is treated as descriptive only. The cap is bounded by
the remaining campaign accounting, every retry uses fresh `97xxx` seeds and a
new directory, and no command changes target/data/hardware/privacy or writes a
public default. A timeout, incomplete grid, nonfinite selected/held-out call,
identity mismatch, or resource violation remains a hard failure. Audit verdict:
`PASS_AFTER_R2_SCALE_AND_BUDGET_REPAIR_READY_FOR_GPU_RUN`.

The R2 diagnostic and any retry remain below the posterior boundary. No
intermediate-temperature short-chain result, acceptance value, or finite
serialization receipt is evidence of whitening, mode discovery, convergence,
posterior correctness, or NeuTra/HMC readiness.

## R2 terminal serialization repair (2026-09-05)

The authorized R2 launch completed all six scope pairs and all eight declared
candidate pairs per scope. It then failed while constructing the terminal
manifest hash because the reliability/transition diagnostic contained a raw
`NaN`; the runner's artifact writer had been repaired, but its separate hash
normalizer still passed non-finite Python floats to `json.dumps(...,
allow_nan=False)`. The durable scope records are therefore useful diagnostic
evidence, but the attempt is not an admissible complete run: it has no
transition record or verified terminal manifest.

This is a localized artifact/harness failure, not evidence that the factor
backend or chart failed. The repair is to normalize non-finite floats in the
runner hash path to the same explicit `{"__nonfinite__": "nan|inf|-inf"}`
markers used by the artifact writer. A focused regression now exercises both
serialization paths (21 Phase 9A tests pass), and the runner compiles with the
repair. Under the unchanged target, method, promotion criteria, hardware
class, and aggregate budget, one serialization-repair replay is authorized in
a new output directory. Its result is still required to contain all six
scopes, the complete grid, and the transition mechanics record. The replay is
terminal artifact repair; it does not add an independent stochastic
replication and cannot weaken any promotion or posterior gate.

## Source-synchronized retune after provenance veto (2026-09-05)

The serialization-repair replay was stopped before scope tuning when the
repository-owned source-closure check found that the HMC tuning contract and
package export files had changed since the R2 artifacts were created. The
changes are part of the concurrent HMC interface repair and its focused tests
pass (89 HMC contract/API/documentation tests); they must not be hidden by
editing an old artifact or disabling the closure guard. The R2 scope records
remain historical diagnostics, and neither R2 attempt can issue a current
handoff.

The campaign therefore opens one new source-synchronized factor scope under
the same q=20 target, factor eigensystem route, bridge, measured `(epsilon,L)`
grid, and promotion/nonclaim rules. The new run binds the current source
closure and fresh seed namespace `98xxx` in a new profile. Because the
aggregate campaign ledger has 4,126.89 material seconds remaining after the
two R2 attempts and diagnostics, the source-synchronized run is capped at
4,000 seconds and covers all six scopes in one fail-closed launch. The prior
factor canary remains numerical-route evidence; the six-scope run itself is
the source-synchronized entry and terminal screen. No separate long canary is
authorized because it would exceed the remaining aggregate budget.

### Source-synchronized pre-run audit

This amendment was checked for stale artifact reuse, source-closure bypass,
budget overrun, hidden target or hardware changes, incomplete scope/grid
coverage, and proxy promotion. The new profile has fresh roots and a unique
output directory; the runner still requires GPU0, XLA, TF32, memory growth,
finite selected/held-out calls, all eight measured pairs, and the mechanics
transition. The 4,000-second cap is a measured-budget constraint, not a
convergence criterion. A closure mismatch, timeout, incomplete scope, or
non-finite candidate closes this campaign without factor promotion. Audit
verdict: `PASS_SOURCE_SYNC_RETUNE_READY_WITHIN_REMAINING_BUDGET`.

The source-synchronized profile is
`phase9a_factor_tuning_full_source_sync_v1` with a 4,000-second cap. Its
authorized launch is:

```bash
BAYESFILTER_FACTOR_TUNING_ATTEMPT_ID=source-sync-<UTC-stamp> \
PYTHON_BIN=/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  bash scripts/run_ssl_lstm_q20_phase9a_factor_tuning_admission.sh source-sync
```

## A3 source-synchronized terminal closeout (2026-09-06)

The source-synchronized launch completed in `3551.299326295033` seconds with
status `PASS_PHASE9A_SCOPE_PREFLIGHT`. It covered all six scopes and all eight
grid pairs per scope, issued six factor-bound handoffs, passed selected and
held-out finite/movement/status checks, and passed the four-draw transition
mechanics controller. The independent receipt
`source-sync-audit.json` returned `PASS_FACTOR_SOURCE-SYNC_AUDIT`; its manifest
and audit hashes are recorded in the terminal result note
`docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md`.

The narrow decision is
`PROMOTED_Q20_PHASE9_NUMERICAL_BACKEND`: the factor backend is admitted for a
new q=20 Phase 9B candidate lane and strict remains the explicit comparator and
fallback. This does not change the generic API default. Extreme selected
acceptance values, four-draw folded R-hat up to `3.3187` in the mechanics
receipt, large pullback residuals, and reproducible trainer retracing warnings
are repair triggers. They keep Phase 9B and all posterior, whitening, mode,
convergence, and HMC-readiness claims blocked until a separately reviewed
sequential validation plan passes.

Terminal audit verdict: `PASS_FACTOR_SOURCE-SYNC_AUDIT`.
