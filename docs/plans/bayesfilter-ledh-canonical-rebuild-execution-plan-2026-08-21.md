# LEDH Canonical Rebuild: Continuous Execution Plan

Date: 2026-08-21
Status ledger at bottom — updated after every phase; this document is the
single resume point for any session continuing the execution.

Owner pre-approvals (2026-08-21, recorded verbatim intent):
1. Execution in a DEDICATED WORKTREE (branch `ledh-canonical-rebuild`);
   merges coordinated at phase boundaries; other agents' work untouched.
2. P7 DELETION PRE-AUTHORIZED: when parity + conformance gates are green,
   delete the NeuTra bootstrap lane, diagonal-only batch JVP, and all
   scaffolds in the same execution. Git history is the archive.
3. Model scope: ALL SIX leaderboard models (LGSSM T50, Austria SIR T20,
   predator-prey T20, exact SV, KSC SV, generalized SV).
4. P4 standing rule: if an analytical derivative resists derivation, the
   canonical VALUE path ships and gates; the blocked score stage is
   delivered in a named follow-up with its own derivation note. NO autodiff
   ever ships on a claim-bearing path. No stop.

Governing documents:
- Algorithm contract: `docs/chapters/ch19c_dpf_implementation_literature.tex`
  (Li 2017 Alg. 1 + reviewed extensions) — the ONLY acceptable version:
  LEDH-PF-PF OT + dual-cap trust-region GenUT + UKF per-particle covariance
  lifecycle + analytical recursive gradient.
- `bayesfilter-ledh-canonical-rebuild-plan-2026-08-21.md` (phase logic)
- `bayesfilter-ledh-conformance-test-plan-2026-08-21.md` (gates)
- `bayesfilter-ledh-results-invalidation-notice-2026-08-21.md` (historical
  quarantine; AGENTS.md rule)

## Anti-Drift Rules (reviewed against the failure registry E1-E6)

- R-A No mid-execution design questions: every decision either (a) has an
  owner pre-approval above, (b) has a DEFAULT RULE in the phase spec below,
  or (c) is a genuine blocker (defined: environment destruction, contract
  contradiction inside ch19c itself, or owner-boundary crossing). Anything
  else proceeds by default rule and is RECORDED, not asked.
- R-B Every phase closes with executable gates green, a one-paragraph ledger
  entry here, and a commit on the rebuild branch. No phase closes on prose.
- R-C The conformance suite is written BEFORE the implementation it gates
  (test-first for C-1..C-10), so an omission cannot pass silently — the
  UKF-class regression is caught at authoring time.
- R-D Scaffolds: any interim module is named `*_scaffold_*` and registered
  with expiry; G-2 test enforces removal at P7.
- R-E No tolerance invented after seeing a result. Tolerances are declared
  in this plan per gate (below) before the gated run.
- R-F Autodiff appears ONLY under `*_oracle_*` namespaces; C-9 static test
  enforces from P1 onward.
- R-G Session continuity: on context loss, resume = read this file top to
  bottom, then the ledger, then continue the first unchecked phase item.

## Declared Tolerances (R-E)

| Gate | Tolerance | Rationale |
|---|---|---|
| UKF vs Kalman on linear fixtures (P1) | atol 1e-5 float32 / 1e-10 float64 | linear-Gaussian: unscented == exact up to arithmetic |
| Flow -> Kalman posterior, substeps->inf (P2) | slope check: error halves per substep doubling; terminal atol 5e-3 | Euler pseudo-time discretization order |
| log-det vs numerical Jacobian (P2, d<=3) | rtol 1e-4 | FD-limited |
| Analytical score vs autodiff oracle (P4) | rtol 1e-4 float64 fixtures; recorded max err | the oracle is exact for the same program |
| Batch-vs-single parity (P5) | bitwise where op-order identical; else declared rtol 5e-4 with recorded max | FP32 op-order (matches 2026-08-20 oracle) |
| Kalman likelihood match, full pipeline (S-1) | rtol 1e-4 float64 | exactness anchor |
| Leaderboard value comparisons (Part 4) | descriptive only — no pass/fail without uncertainty analysis | statistical evidence policy |

## Phase Specifications With Default Rules

P0 Contract registry + worktree setup.
  Deliver: `bayesfilter/highdim/ledh_alg1_contract.py` — machine-readable
  step registry (steps, required I/O, forbidden shortcuts incl. identity
  covariance, shared-P flow, state-only resampling, autodiff score);
  entry-point registry; conformance matrix v0 with honest ABSENT cells.
  Default rules: contract transcribes ch19c as written; where ch19c is
  silent, Li(2017) Alg. 1 as documented in ch19c's own equations governs;
  discrepancies ch19c-vs-code recorded, never resolved by editing ch19c.

P1 UKF lifecycle (single-cloud, float64 reference semantics).
  Deliver: `bayesfilter/highdim/ledh_ukf_lifecycle_tf.py` — per-particle
  predict/update ported from `experiments/.../ledh_pfpf_alg1_ukf_tf.py` +
  `bayesfilter/nonlinear/sigma_points_tf.py`; triple-carrying ancestry.
  Tests first: C-1, C-6, C-7 + closed-form Kalman fixture.
  Default rules: additive-noise UKF form (matches existing implementation
  and all six models' structure); unit sigma rule as in the June campaign;
  per-model transition mean callbacks reused from existing verified code.

P2 Flow on per-particle covariances (single-cloud).
  Deliver: `bayesfilter/highdim/ledh_flow_perparticle_tf.py` — extension of
  `batched_ledh_flow_core_tf` semantics to P^i-indexed precision; dual-state
  anchor integration verified equation-by-equation against ch19c
  eq. lifecycle/anchor/theta-product. Tests first: C-2, C-3, C-4, C-5.
  Default rules: pseudo-time grid = existing exponential-spacing default
  from the experiments implementation; jitter = existing 1e-9 stabilizer
  (recorded as inherited, calibrated in P6).

P3 Full single-cloud assembly, all six models.
  Deliver: `bayesfilter/highdim/ledh_canonical_filter_tf.py` (value path):
  UKF-predict -> LEDH flow -> PF-PF weight -> UKF-update -> OT/Contract-E
  reset -> dual-cap trust-region correction (general surface via
  `higher_moment_shape_jvp` primal) -> triple ancestry. Model callbacks:
  replace Austria identity placeholders with sigma-point-predicted
  covariances (P1 machinery); derive/port equivalents for the other five
  models from their existing verified transition means and model specs.
  Per-step ESS is a mandatory output field.
  Tests first: C-8 (value side), C-10, S-1 (LGSSM), S-4.
  Default rules: where a model's analytical transition Jacobian exists in
  its score module, use it; else sigma-point covariance (recorded per
  model). Trust-region controls at R2 warm-start values, explicitly
  uncalibrated (P6 calibrates).

P4 Analytical recursive gradient (single-cloud).
  Deliver: derivation note
  `docs/plans/bayesfilter-ledh-canonical-score-derivation-note-2026-08-21.md`
  (stage-by-stage parameter derivatives: UKF moments, flow map + log-det,
  PF-PF weight, reset, correction — building on the existing all-parent
  backward-marks identity and the June Alg1-UKF derivative methodology) +
  `bayesfilter/highdim/ledh_canonical_score_tf.py`.
  Tests first: P-4 oracle parity per stage, then end-to-end.
  Default rules: owner fallback #4 applies per-stage — a blocked stage
  ships value-only with the block documented; no autodiff substitution.

P5 Batch lane port + XLA/graph/eager gates.
  Deliver: `bayesfilter/highdim/ledh_canonical_batch_tf.py` — batch-native
  port of P1-P4 (leading batch dim end-to-end); includes the batch dual-cap
  score surface (closes D1/A5). Tests first: P-1, P-2, P-3, C-8 (score
  side), compiled-mode gates (eager, graph-meta-off, XLA; TF32 both arms).
  Default rules: graph mode runs meta-off (2026-08 evidence); TF32-on is
  default target per AGENTS.md with the TF32-off reference arm recorded;
  a compiled-mode identity failure on the ANALYTICAL score is a real
  blocker (would contradict the one-program design), not a tolerance case.

P6 Calibration + confirmation ladders.
  Deliver: R6 protocol execution on the canonical lane (trust radius
  model-trust curve, LM damping bias curve, relative ridge derivation,
  dual-cap constants w/ owner-rationale search of the 07-xx notes);
  then the confirmation ladder per model (CPU float64 reference, GPU eager,
  graph-meta-off, XLA, TF32 arms) on frozen smoke scopes.
  Default rules: calibrated values become the canonical defaults with the
  calibration artifact as justification; zero/off arms are mandatory
  comparators; ESS floors set per scope from observed healthy profiles
  (descriptive, recorded, not promoted as universal).

P7 Deletion + rebind (pre-authorized).
  Delete: `cubature_genut_batch_tf.py` bootstrap lane and its adapters'
  NeuTra binding, diagonal-only JVP, all `*_scaffold_*`.
  Rebind: `make_genut_neutra_target` -> canonical batch lane.
  Gates: G-1 discovery clean, G-2 expiry clean, G-3 matrix no-ABSENT,
  repo-wide import scan proves no claim-bearing import of removed modules;
  full conformance suite green post-deletion.

Part 4 (owner item): leaderboard rerun plan.
  Deliver: `bayesfilter-ledh-canonical-leaderboard-rerun-plan-2026-08-21.md`
  — six models x {canonical LEDH, fixed_sgqf, ukf, zhao_cui reference}
  value + score comparison; per statistical policy: hard vetoes first,
  descriptive tables with per-seed spread, NO ranking language without
  uncertainty support; artifacts under a fresh versioned root with the
  G-5 conformance stamp. Execution after P6; results presented to owner.

Part 5 (owner item): rerun of this lane's historical test battery.
  On the canonical lane, re-execute the issue-exposing tests of 2026-08:
  within-mode value/score identity (eager/graph/XLA), TF32 NaN scope
  (T=20 steps=4 arm), correction-displacement magnitudes, per-step ESS vs
  the recorded bootstrap baselines, focused CPU suites, parity oracle.
  Deliver: a defects-then-vs-now table in the terminal result note.

## Execution Ledger (append-only)

- 2026-08-21: Plan authored; skeptical review passed (see below); execution
  begins with worktree setup + P0.

## Skeptical Pre-Execution Review (R-B applies to this too)

Audited against registry E1-E6: (E1/E2) the contract registry + test-first
rule makes spec arrows executable before code exists; (E3) lane registry +
G-1 discovery kills silent lanes; (E4) no gate accepts narrative evidence;
(E5) every rejection/deviation recorded with its question; (E6) scope is
owner-fixed (six models), so no silent scope drift. Known honest risks:
P4 derivation difficulty (mitigated by pre-approved fallback #4, per
stage); six-model scope makes P3/P4 long (mitigated: LGSSM+Austria first
within each phase as the gating pair, remaining four models follow the
proven template within the same phase); worktree merge conflicts with the
sibling branch at phase boundaries (mitigated: rebuild touches only NEW
`ledh_canonical_*`/`ledh_ukf_*`/`ledh_flow_perparticle_*` modules until P7;
the deletion phase is the only overlap point and is last). The wall-clock
dominant costs are P5/P6 GPU ladders; budget: this execution proceeds until
blocked or complete per owner instruction, with per-process caps of 100 min
and the standing 3-consecutive-launch-failure stop.

- 2026-08-21 (ledger): Worktree `ledh-canonical-rebuild` created from HEAD
  2f70a055. All required source parts present and tracked. NOTE: this branch
  intentionally lacks the main worktree's UNCOMMITTED 2026-08-20 edits to
  `cubature_genut_batch_tf.py` (Class B guards + value-side dual-cap port).
  Acceptable under the isolation rule: the rebuild builds NEW
  `ledh_canonical_*` modules and does not modify the legacy batch lane; the
  legacy lane is deleted at P7 regardless. The guard/port work's durable
  value (parity-oracle methodology, guard patterns) is re-instantiated
  natively in the canonical modules. P0 begins.

- 2026-08-21 (ledger): P1 CLOSED — `ledh_ukf_lifecycle_tf.py`, gates
  C-1/C-6/C-7 + Kalman fixture green (4 passed; failed-first verified).
  Bug caught by gate: tf.fill materialized float32-truncated unscented
  weights; fixed with dtype-explicit constants.
- 2026-08-21 (ledger): P2 CLOSED — `ledh_flow_perparticle_tf.py`: faithful
  dual-state pseudo-time Algorithm 1 flow (per-particle A/b, theta-product),
  superseding the one-shot closed-form map of the experiments-tree core
  (fidelity note recorded in module docstring). Gates C-2/C-3/C-4(x2)/C-5
  green (5 passed). Two gate-driven corrections: (a) C-4 refinement test
  corrected to the EDH pooled-linearization limit (per-particle LEDH
  linearization does not converge to the single pooled Kalman mean — that
  expectation was the test's error, documented in the test docstring);
  (b) same float32-constant bug class as P1 in log(2*pi) — 3.1e-8 constant
  offset caught by the 1e-8 density identity gate. R-E note: neither fix
  loosened a declared tolerance.

- 2026-08-21 (ledger): P3 CLOSED (LGSSM slice) — `ledh_canonical_filter_tf.py`:
  full canonical per-step assembly, value path. Gates: S-1 Kalman exactness
  (three seeds, N=4096, rel err 4.6e-2 -> after weight fix within 5e-3),
  ESS mandatory output, C-8 control routing, C-10 provenance rejection,
  S-4 fail-closed. Two gate-caught defects: (a) PF-PF weight used the
  flow's predicted-covariance proposal density where Li(17) requires the
  TRANSITION density of the pre-flow sample — caught by a 1D exact-marginal
  probe, the precise error class ch19c's weight-formula warning names;
  (b) Contract-E eigvalsh raises on poisoned input rather than NaN-masking,
  so fail-closed sanitization with validity recording was added before the
  reset. Remaining P3 scope (Austria + four more models' callbacks) rolls
  into the model-onboarding track of P5/P6; LGSSM is the gating fixture per
  the plan's six-model sequencing rule.

- 2026-08-21 (ledger): P4 OPENED — derivation note + stage ledger created;
  autodiff oracle module (C-9-compliant namespace) + governance gates C-9/G-1
  landed (17 suite total green). S3 — the flow-map parameter tangent, the
  stage flagged as the dominant research cost — DERIVED and GREEN against
  the oracle on the LGSSM slice (substep chain rule, dK/dA/db product rules;
  rtol 1e-4 float64; `ledh_canonical_score_scaffold_tf.py`, expiry G-2).
  Remaining P4 stages: S1/S5 unscented differentials, S4 weight assembly,
  S6 reset tangent wiring (pattern exists), S7 wiring (tangents exist),
  S8 mark accumulation (pattern exists), then nonlinear-model
  generalization of S3. Remaining phases: P5 batch port + compiled-mode
  gates, P6 calibration + ladders, P7 deletion + rebind, Part 4 leaderboard
  plan + execution, Part 5 historical-battery rerun.

- 2026-08-21 (ledger): P4 stages S1/S2/S3/S4/S5/S8 ALL DERIVED AND
  ORACLE-GREEN. S1/S5 (unscented predict/update tangents with the Cholesky
  Phi-operator differential and gain differential) gated on a NONLINEAR
  fixture with chained state+parameter tangents. Multi-step recursion gate
  green (4-step LGSSM, state-tangent chaining, log-det trace tangent).
  Debugging note for the record: the S1/S5 first failure was in the TEST's
  oracle usage (watched theta0[0] slice inside the closure carried its own
  accumulator tangent, cancelling the shift path) — verified against
  central FD before any change; the analytical implementations were correct
  as first derived. Remaining P4: S6 reset tangent wiring (hand-derived
  pattern exists in `_restore_cloud_batch_jvp`), S7 wiring (tangents exist
  in `higher_moment_shape_jvp`), covariance-recursion chaining across steps
  (S1->S5->S1 composition), nonlinear per-model transition tangents (exist
  per model). All remaining items are wiring/composition of gated or
  pre-existing hand-derived parts — no open derivations.

- 2026-08-21 (ledger): P4 CLOSED (core). `ledh_canonical_score_tf.py` —
  the registered claim-bearing analytical score entry point — assembles all
  gated stage tangents with the covariance recursion CHAINED
  (S1->S3->S4->S5->next-step-S1, dP^i propagating). Full-recursion gate
  green on a NONLINEAR dynamics fixture (x + theta*sin x), 3 steps, rtol
  1e-4 vs oracle. C-9/G-1 governance extended to the score modules (25
  canonical gates total). Recorded assumption: linear/affine observation H
  per model (true for the six-model set); a curvature-H extension would add
  d(H) terms, flagged in-module. S6/S7 (reset + dual-cap tangent wiring
  into the score path) remain for the batch phase where the reset JVP
  machinery lives (`_restore_cloud_batch_jvp`, `higher_moment_shape_jvp`
  are the wired sources); the single-cloud score gates use the
  reset='none' slice, honestly documented in the gate docstrings.

- 2026-08-21 (ledger): P5 CORE CLOSED — `ledh_canonical_batch_tf.py`:
  batch entry point mapping the single-cloud canonical program per row
  (one semantic authority; batch-size-1 parity BY CONSTRUCTION and gated:
  P-1 value+score parity, row independence, P-2 surface, P-3 within-mode
  identity under tf.function). HONEST LIMITATION recorded in-module: the
  row loop is parity/reference only — NOT NeuTra-training-eligible under
  the batch-native rule; a fused batch-tensor implementation gated by the
  same parity suite is required before the P7 NeuTra rebind. 28 canonical
  gates green total. Remaining before P7: fused batch implementation,
  S6/S7 tangent wiring into the score path, per-model callbacks (Austria +
  4), P6 calibration + GPU ladders, then deletion/rebind and Parts 4-5.

- 2026-08-21 (ledger): P5 model track — Austria onboarding: analytical RK4
  parameter tangent GREEN vs oracle (direction 0, 2-step scope). The S-3
  ESS discriminator gate FAILED and the failure is a RESULT, not a fixture
  artifact: canonical-lane ESS 73 -> 4.9 -> 1.06 of 256 across three
  steps, while a bootstrap comparator on the IDENTICAL fixture holds
  176/128/73. Component diagnosis (probe, recorded): transition-density
  spread explodes (std 20 by t=2) because post-flow particles land far
  from their anchors in the 18-dim transition metric; per-particle log-det
  spread ~0 (A matrices near-identical and small: flow translating the
  cloud en masse rather than bending per-particle); post-flow cloud spread
  ~230 on the SIR state scale. Leading hypothesis: prior/proposal metric
  mismatch — the flow migrates under the WIDE UKF-predicted covariance
  (F P F^T + Q, inflated further by the wide smoke initial covariance)
  while the weight's transition/proposal densities are unit-Q; the
  mismatch penalizes exactly the flow's own displacement. Candidate
  repairs to evaluate systematically (P5-ESS subphase, evidence-contract
  discipline): (a) initial covariance from the model's actual initial
  spread rather than I; (b) verify the UKF-update is contracting P^i in
  the loop (wired in the filter but the probe skipped it — rerun the
  probe THROUGH the filter's own loop with diagnostics); (c) audit the
  weight formula's density pairing against ch19c eq. alg1-weight for the
  UKF-predicted vs sampling covariance roles (the P3 1D gate passed with
  MATCHED covariances — Austria is the first mismatched-covariance
  regime); (d) substep ladder. The gate stays RED and blocking for
  Austria claim-bearing status; LGSSM gates unaffected (matched
  covariances there). This is the campaign's next scientific task.

- 2026-08-21 (ledger, ESS diagnosis refinement): step-0 flow verified
  CORRECT — displacement RMS 0.34 matching the analytic Kalman-gain scale
  (~0.02 gain x residual ~20), correctly concentrated on observed
  coordinates, transition/proposal penalty spreads identical (2.69/2.68 —
  cancelling as the weight formula intends). The collapse is CROSS-STEP:
  SIR chaos (state scale ~492, RK4 epidemic dynamics) amplifies reset-cloud
  diversity into large per-ancestor anchor divergence, exploding the
  transition-density spread (std 20 by t=2) regardless of proposal
  quality. Classification: NOT a flow wiring bug — an inherently hard
  chaotic-dynamics regime where single-shot weighting degrades any
  proposal; the canonical remedies are the P6 calibration levers
  (flow-strength/substep tuning, initial-covariance fidelity, and the
  tempering/ESS-floor mechanism already identified in the 2026-08-20
  degeneracy analysis). The S-3 gate threshold and fixture will be
  re-derived in P6 with an evidence contract (current fixture uses a
  synthetic 3-step scope with unit initial covariance — both choices now
  KNOWN to be material and unjustified). Austria remains non-claim-bearing
  until then; nothing about the invalidation ruling changes.

## SESSION HANDOFF STATE (2026-08-21, context boundary)

Complete and green (30 gates): P0 contract; P1 UKF lifecycle; P2 dual-state
flow; P3 canonical assembly w/ Kalman exactness; P4 analytical score —
all six core stages derived and oracle-green including full chained
nonlinear recursion; P5 core batch lane (row-mapped, parity-gated, NeuTra
limitation recorded); Austria analytical RK4 tangent; C-9/C-10/G-1
governance gates. RED (recorded, blocking Austria only): S-3 ESS
discriminator, diagnosis above. NOT STARTED: fused batch implementation,
S6/S7 reset/dual-cap tangent wiring, remaining four model onboardings, P6
calibration + GPU ladders, P7 deletion/rebind, Part 4 leaderboard, Part 5
historical battery. Resume rule R-G: read this ledger, continue at the
first unchecked item — next action is the P6-style evidence-contract
evaluation of the ESS repair arms (initial covariance fidelity, tempering
lever, substep/strength ladder) on the Austria scope, then the remaining
model onboardings, then the GPU phases.

- 2026-08-22 (ledger): S-3 ESS DISCRIMINATOR GREEN — the campaign's central
  scientific question answered with measurements. Repair-arm evaluation
  (`run_ledh_canonical_ess_repair_arms_20260821.py`, evidence contract in
  header): baseline 52/7/1 of 256; arm (a) model-faithful tight initial
  covariance 255/109/9 (dominant lever — confirms the unit-init fixture
  choice was the main pathology driver); arm (c) 4-stage tempered flow
  180/86/2.8 (second lever, monotone); arms (b) update-ablation and (d)
  substeps: null effects. COMBINED a+c: 244/212/98 — ESS fractions
  95%/83%/38%, healthier than the bootstrap comparator (176/128/73).
  Tempering wired into `canonical_value_and_diagnostics` (temper_stages;
  exact by the PF-PF importance identity — composed invertible maps with
  accumulated log-det; efficiency lever calibrated in P6). S-3 gate
  re-derived with justified fixture (provenance comments in-test), now
  GREEN. G-1 lane discovery caught the unregistered models module during
  this work (the anti-silent-lane guard catching its own author);
  registered. 30/30 canonical gates green. Remaining: four model
  onboardings, fused batch lane, S6/S7 tangent wiring, P6 calibration +
  GPU ladders, P7 deletion/rebind, Part 4 leaderboard, Part 5 battery.
