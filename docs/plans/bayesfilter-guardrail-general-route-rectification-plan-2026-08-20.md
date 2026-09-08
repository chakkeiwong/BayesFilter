# BayesFilter Guardrail And General-Route Rectification Master Plan

Date: 2026-08-20 (living document; phase annotations appended, never rewritten)

Authorization: owner directives of 2026-08-20 — (1) dual-cap trust-region is
to become the default GENERAL implementation with no model- or lane-specific
forks; (2) every Class C protection (ridge, trust radius, LM damping) must be
verified and calibrated with principled justification for its value,
INCLUDING the choice not to use it (zero/off is also a setting); (3) execute
with minimal stopping. Safety-guardrail reversed-burden policy (claudecodex,
installed 2026-08-20) governs classification.

## Conversation Coverage Checklist

Every issue raised in the 2026-08-18..20 sessions, mapped to a phase:

| # | Issue | Status before this plan | Phase |
|---|---|---|---|
| 1 | XLA T=20 NaN | Localized: TF32-seeded Stage D blowup; guard correct | R2 (non-harm eval), R6 (calibration), R7 (ladder) |
| 2 | Graph value/JVP program split (grappler) | Localized; meta-off restores identity | OWNER DECISION still open; recorded R8 |
| 3 | Class A: runners discard diagnostics dict | New localization runner serializes; policy adopted | R0 done; rule now in global policy |
| 4 | Class B: three unridged/unchecked Stage D Choleskys | Unprotected | R3 (check-only guards + no-fire regression) |
| 5 | Class C: trust radius / LM damping / ridge unjustified (0.5 and 0.0 alike) | trust=0, damping=0 inherited; 0.5/1e-2 convenience | R2 (non-harm evidence), R6 (calibration protocol) |
| 6 | Dual-cap trust NOT general in the claim-bearing lane (batch fork lacks pairwise + coordinate caps) | Verified 2026-08-20: `higher_moment_shape_jvp` general single-cloud; batch lane diagonal-only | R4 (batch port + parity oracle), R5 (wiring gate) |
| 7 | Codex report / Claude audit verified function existence, not call chain | Process gap | R1 (policy: call-chain audit rule), memory note |
| 8 | Class C hyperparameter justification incl. "off" | Not in policy | R1 (policy extension) |
| 9 | False/ambiguous doc claims that dual-cap-trust is generally implemented | Unswept for this specific claim | R5 (targeted sweep + banners) |
| 10 | Eager-GPU vs CPU ~2.3 log-unit TF32 value offset | Recorded explanatory, not checked | R6 scope note (calibration campaign measures TF32 arms) |
| 11 | NeuTra/HMC/tuning/cross-model/dual-cap promotion | All blocked | Unchanged nonclaims; R7 defines reopening path |
| 12 | Doc hygiene (banners, attempt numbering, migration note) | Delegated to cleanup agent (separate handoff) | Out of scope here; R5 covers only issue 9 |

## Phases

- R0 (done): Class A adopted — localization runner serializes diagnostics;
  reversed-burden policy canonical in claudecodex and installed to repo.
- R1: policy extensions in claudecodex — (a) implementation audits must
  verify the call chain from each claim-bearing endpoint, not function
  existence; executable parity/wiring checks outrank narrative verification;
  (b) every Class C hyperparameter value, including zero/off, requires a
  principled justification (derivation, measurement, or recorded owner
  rationale). Install, commit, push; refresh repo install.
- R2: Class C non-harm evaluation on the CURRENT frozen source (before any
  edit, for comparability): trust/LM candidate controls (damping 1e-2, scale
  floor 1e-4, radius 0.5 — 2026-08-15 route contract values, explicitly
  warm-start hypotheses, NOT justified values) on (a) eager `T=20,4` healthy
  arm — measure value shift and invariant health; (b) XLA TF32-on `T=20,4`
  failing arm — does the NaN vanish? Runner gains control-override flags;
  overrides recorded in manifest. Fresh dirs `trust_eval_attempt01/02`.
- R3: Class B check-only guards in the batch route: finiteness +
  positive-diagonal checks on the three unprotected Choleskys
  (`cubature_genut_batch_tf.py:1273,:1288` and per-iteration `:746` via the
  returned `corrected_chol`), feeding `valid` and a new
  `minimum_higher_moment_cholesky_diagonal` diagnostic. NO numerical change
  to accepted results. No-fire regression: focused CPU suites must pass and
  eager behavior must be value-identical (checks only extend the mask).
  This begins a NEW source state; all prior artifacts become historical for
  the edited files (already true after any repair).
- R4: batch-native port of the full general correction surface (pairwise
  co-skew/co-kurtosis with radial step cap; coordinate clamp; alongside the
  existing diagonal+LM+trust), default-OFF controls, with a PARITY ORACLE
  test: batch-size-1 output vs `higher_moment_shape_jvp` (zero tangents) on
  identical inputs, declared tolerance, plus default-off regression (new
  controls off => existing suites pass unchanged). Port follows the
  reference-port-first policy: the general single-cloud function is the
  semantic authority; any intentional deviation must be recorded.
- R5: wiring gate + doc sweep: a focused test asserting the Austria batch
  route exposes the full general capability surface (fork-regrowth guard),
  and a targeted grep/banner pass on docs claiming dual-cap-trust was
  "implemented generally" without the lane caveat.
- R6: calibration campaign PLAN (execution is its own campaign): principled
  justification protocol per Class C control — trust radius from a measured
  model-trust curve (predicted vs actual residual reduction across a radius
  ladder); LM damping from bias-vs-robustness curve with FP32 noise floor;
  ridge derived from worst-lane effective epsilon (TF32 ~5e-4) x safety
  factor, relative form `delta*(tr(C)/d)`; coordinate/radial caps attach
  owner rationale or enter the same protocol. Zero/off arms are mandatory
  comparators, not defaults.
- R7: after R4-R6: switch the wired Austria callable to the general route
  with calibrated controls, fresh tuning scope (LEDH rule), and the full
  confirmation ladder (CPU authority, eager, graph meta-off, XLA, both TF32
  states). Explicitly out of this session's scope to complete.
- R8: record the two open owner decisions (graph-mode claim-bearing status;
  TF32 default posture for this route) — inputs now include R2 evidence.

## Evidence Contract (for R2, the only research-grade runs this session)

- Question: with candidate trust/LM controls on the frozen source, (a) does
  the TF32+XLA `T=20,4` NaN vanish, and (b) how far do healthy-arm outputs
  move (there may be NO healthy regime at this scope — RMS 56 >> radius 0.5;
  finding that is a valid outcome, not a failure)?
- Comparators: existing trust-off artifacts (eager attempt01/07; XLA
  attempt04).
- Hard vetoes: identity/hash mismatch, wrong env, `status != COMPLETE`.
  NaN or value shift are RESULTS, not vetoes.
- Explanatory only: value deltas, margin changes, wall times.
- Not concluded: no promotion of 0.5/1e-2 (unjustified warm starts), no
  default change, no tuning validity, no NeuTra/HMC readiness.

## Stop Conditions

Hard veto above; process cap 100 min per GPU arm; 3 consecutive launch
failures; any need to change scientific targets or invent tolerances.

## Skeptical Audit

Passed 2026-08-20: R2 runs precede source edits (comparability preserved);
R3 is check-only (Class B) with a declared no-fire gate; R4 has a parity
oracle and default-off regression so the port cannot silently change wired
behavior; the wiring/default switch (R7) is explicitly deferred until
calibration exists, per owner directive 2 — no unjustified value gets wired;
candidate controls in R2 are labeled warm-start hypotheses. Known residual
risks: parity may not be bitwise across op orders (declared tolerance +
recorded max diff); R2 healthy-arm interpretation is bounded by the
no-healthy-regime caveat, predeclared.

## Phase Annotations

### R0/R1 — COMPLETE
Policy extensions (Class C justification incl. zero/off; call-chain audit
rule) canonical in claudecodex, committed+pushed, installed to home targets
and this repo's AGENTS.md/CLAUDE.md marked blocks.

### R2 Class C non-harm evaluation — COMPLETE (frozen pre-edit source)

Controls: damping 1e-2, scale floor 1e-4, radius 0.5 (warm-start values).

| Arm | Artifact | Result |
|---|---|---|
| Eager T=20,4 | `trust_eval_attempt01/eager_trust_controls.json` | COMPLETE, valid, exact within-mode identity; value `-682.6732` vs uncapped `-683.0019` (shift 0.329); cap binds 51.30->0.49998; LM condition 199.8; Pearson margin 1.098; T=1 control bitwise unchanged |
| XLA TF32-on T=20,4 | `trust_eval_attempt02/xla_trust_controls.json` | COMPLETE, **FINITE AND VALID** (was NaN uncapped), **exact within-mode identity under XLA**; value `-685.6978`; cap binds 44.26->0.49997; Pearson margin 0.215 |

Findings:
1. The trust cap REMOVES the XLA TF32 NaN at the frozen scope — the Class C
   candidate passes the "bounded, flagged behavior where unhealthy" half of
   the non-harm criterion in the strongest form (valid, finite, identity).
2. There is NO healthy regime at this scope: pre-cap RMS 44-51 >> 0.5 in
   every arm, so the cap binds everywhere and outputs move (eager shift
   0.329 log-units; XLA-trust vs eager-trust cross-mode drift 3.02 remains,
   explanatory, TF32-dominated). Per the predeclared caveat this is a
   RESULT: the uncapped default's outputs are artifacts of an uncontrolled
   iteration, and "identical outputs on healthy trajectories" is vacuously
   satisfied because no trajectory at this scope is healthy.
3. Within-mode value/score identity holds in BOTH arms with trust controls.
   Notably XLA, which failed validity uncapped, is exact within-mode when
   capped. (Graph-mode grappler question remains separate and open.)
4. Nonclaims preserved: 0.5/1e-2 remain UNJUSTIFIED warm starts pending the
   R6 calibration protocol; no default change; cross-mode drift untouched.

### R3 Class B check-only guards — COMPLETE (new source state)

`_higher_moment_batch_value` now tracks `minimum_higher_moment_cholesky_diagonal`
across the target, point, and per-iteration corrected Choleskys and folds
finiteness + strict positivity into `valid`. Check-only: accepted results
numerically unchanged. No-fire regression: focused CPU suites
`test_genut_batch_primal_parity.py` + `test_cubature_genut_batch.py` = 9
passed. This edit begins a new source state for `cubature_genut_batch_tf.py`;
R2 artifacts (pre-edit) remain the comparability anchors.

### R4/R5 — COMPLETE (general-route batch port + gates)

Port implemented in `cubature_genut_batch_tf.py` (new source state):

- `_pairwise_iteration_batch_primal`: batch port of the general pairwise
  co-skew/co-kurtosis step (semantic authority: primal lines of
  `higher_moment_contract_e._pairwise_shape_iteration_jvp`), including the
  radial step cap and post-step re-standardization, with its Cholesky under
  the Class B guard.
- `_higher_moment_batch_value` gains default-off controls
  `pairwise_correction_steps/strength/floor/particle_rms_cap` and
  `coordinate_cap/power` (smooth clamp + re-standardization, general-route
  stage order: diagonal -> pairwise -> coordinate cap -> affine restore),
  with new pairwise/coordinate diagnostics and guard coverage.

Gates, all passing (`tests/highdim/test_genut_batch_general_route_parity.py`):

1. Parity oracle: batch-size-1 full-capability output vs
   `higher_moment_shape_jvp` (zero tangents), 3 seeds, declared FP32
   op-order tolerance 5e-4 relative — 3 passed.
2. Default-off bitwise inertness: new controls off == old signature — pass.
3. Fork-regrowth guard: signature must retain the general capability
   surface — pass.
4. No-fire/default-off regression: focused CPU suites 9 passed post-port.

Known limitation (recorded, not hidden): the port covers the VALUE route.
`_higher_moment_batch_jvp` (score route) remains diagonal-only; wiring
pairwise/coordinate controls into the batch JVP is part of R7's switch work
and MUST be completed before any claim-bearing run enables those controls,
since value/score program identity is the campaign's core contract. Enabling
pairwise/coordinate controls on the value side only would split the programs
by construction.

Doc sweep (R5b): grep over docs/plans found no false "dual-cap is generally
implemented" claim — campaign docs consistently classify the wired route as
`batch_diagonal_candidate`. The false claim was confined to an out-of-repo
Codex report. No banners needed; the call-chain audit rule (R1) is the
structural fix.

### R6 calibration protocol — DEFINED (execution deferred to its own campaign)

Per-control justification protocol (owner directive: zero/off carries the
same burden):

| Control | Justification form | Anchor |
|---|---|---|
| trust_radius | measured model-trust curve: predicted vs actual moment-residual reduction across radius ladder (0.1..2.0) on reference clouds; choose largest radius with agreement above declared threshold | standardized units make O(1) the dimensional prior |
| lm_damping | bias-vs-robustness curve: fixed-point shift vs damping, floor at FP32 gradient-noise scale | bounded-step guarantee 1/damping |
| ridge (Contract-E, existing 1e-5) | derivation: worst-lane effective epsilon (TF32 ~5e-4) x covariance scale x safety factor, relative form delta*(tr(C)/d) | must be re-derived, current 1e-5 is absolute and inherited |
| pairwise_particle_rms_cap=2.0, coordinate_cap=0.98,p=8 | attach owner rationale from 07-xx dual-cap selection notes if found; else same ladder protocol | owner-selected constants |
| zero/off arms | mandatory comparators in every ladder | non-use is a setting |

### R8 — open owner decisions (inputs now include R2)

1. Graph-mode claim-bearing status (grappler split; meta-off restores).
2. TF32 default posture for this route (NaN was TF32-seeded; trust cap
   removes it at warm-start controls; calibrated controls pending).
3. NEW from R2: whether the uncapped diagonal route's outputs (pre-cap RMS
   44-56 at frozen scope) remain an acceptable interim reference at all, or
   whether R7 should be accelerated.
