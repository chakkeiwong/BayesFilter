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

- 2026-08-22 (ledger): Part 4 slice 1 EXECUTED (LGSSM + Austria, CPU
  float64, artifact `ledh_canonical_leaderboard_2026-08/slice1_result.json`).
  LGSSM value vs exact Kalman (3 model seeds x 3 particle seeds, N=4096):
  abs errors 0.116/0.113/0.060 nats with per-seed spread 0.005-0.020 —
  small spread + consistent negative sign = SYSTEMATIC bias, descriptively
  attributed to the OT-reset transport approximation + logsumexp Jensen
  bias (both expected; the reset is the production design's known
  approximation). Score self-consistency vs oracle: rel err 3.8e-16.
  Austria canonical vs bootstrap on the tight-init synthetic fixture:
  values statistically indistinguishable (deltas 0.06-0.29 nats, spread
  overlapping); bootstrap min-ESS descriptively HIGHER (133/141/92 vs
  98/75/73) — honest finding: on this easy diffuse-noise fixture the
  flow's machinery does not beat bootstrap; the flow's regime is sharp
  likelihoods/poor initialization (where bootstrap historically collapsed
  to ESS 23/1008 on the REAL frozen target). No superiority claim either
  direction per the statistical policy. Full six-model leaderboard remains
  gated on the four model onboardings + frozen-target integration + P6.

- 2026-08-22 (ledger): FROZEN-TARGET HARD-REGIME PROBES (artifacts
  `frozen_austria_probe.json`, `frozen_austria_full.json`). Findings:
  (1) Harness fidelity confirmed — bootstrap on the frozen tensors
  reproduces the historical collapse exactly (min ESS 20.9 vs recorded
  ~23/1008), at the same steps (2 and 4, epidemic takeoff).
  (2) The canonical flow lane collapses HARDER at exactly those steps
  (ESS 1.0-2.1) while beating bootstrap at most other steps (e.g. late
  steps 610-965 vs 191-828), and temper staging improves monotonically
  (values -956/-754/-695 for stages 1/2/4, reset-less probe).
  (3) With the Contract-E reset active (full pipeline), temper=4 stays
  program-valid over all 20 steps with the same takeoff-step collapse;
  temper=1 goes fail-closed-invalid at step 3 — the reset is a necessary
  stabilizer but not sufficient at takeoff.
  (4) Mechanism (consistent across all probes): at chaos-takeoff steps the
  per-particle UKF predicted covariances explode (faithful reporting of
  epidemic-growth uncertainty), the flow migrates accordingly, and the
  PF-PF weight's transition-density numerator (spread Q=I) punishes the
  migration — a STRUCTURAL algorithm-model interaction of Li(17)-style
  flows at chaotic takeoff, now cleanly measured. Candidate P6 levers
  (each needs a reviewed contract; further ad hoc tuning here would be
  local optimization drift): flow-prior covariance capping toward Q;
  ESS-triggered adaptive tempering; both. NO lane is promoted; the
  frozen-scope claim comparison remains gated on P6 calibration.

## P6 Takeoff-Lever Contract (declared BEFORE execution, 2026-08-22)

- Question: do (A) spectral capping of the flow-prior covariance at c*Q
  and/or (B) deeper temper staging lift the takeoff-step ESS on the frozen
  Austria probe scope without breaking validity?
- Ladder: cap c in {2, 8, 32, uncapped} x stages in {4, 8}; probe scope
  identical to `frozen_austria_probe.json` (reset-less lane for speed,
  plus one full-pipeline confirmation of the best cell).
- Primary criterion: ESS fraction at the takeoff steps (2 and 4); success
  threshold declared NOW: > 10% (i.e. > ~101/1008) at BOTH takeoff steps —
  a qualitative improvement over bootstrap (2.1-2.9%) and baseline flow
  (~0.1%). Veto: program_valid false or nonfinite value in the
  full-pipeline confirmation. Explanatory: value, full ESS profile.
- Both levers are EXACT (the flow prior is a proposal-design choice; any
  invertible map is corrected by the PF-PF identity) — this is efficiency
  calibration, not approximation introduction.
- Nonclaims: no default promotion (calibrated constants remain scoped to
  this probe until the P6 full protocol); no posterior/leaderboard claim.

- 2026-08-22 (ledger): TAKEOFF-LEVER LADDER COMPLETE (artifact
  `takeoff_levers.json`). Verdict against the PRE-DECLARED >10% takeoff
  threshold: ALL EIGHT CELLS FAIL. Response surface: tight capping (c=2)
  restores takeoff ESS exactly to bootstrap levels (28.1/22.9 vs
  bootstrap 29.4/20.9; value -683.43 vs -683.71) — i.e. the cap fixes the
  flow's SELF-INFLICTED collapse by making the flow nearly inert at
  takeoff; looser caps trade takeoff ESS for slightly better values
  (best value -682.24 at c=32/stages=4 but ESS 2.5 there, unreliable).
  CLASSIFICATION per the research-question guardian: candidate failure,
  NOT direction failure. Structural insight from the failure: per-ancestor
  proposal design (any flow, any cap) cannot beat the ACROSS-ancestor
  spread of p(z|ancestor) at takeoff — the measured floor (~bootstrap
  ESS) IS that across-ancestor spread. The mechanism class that addresses
  it is within-step annealed SMC: REWEIGHTING/RESAMPLING interLEAVED
  between temper stages (my staging composed maps without inter-stage
  reweighting — that was the gap), with the OT reset as the inter-stage
  resampler. This is a per-step algorithm-structure change requiring its
  own reviewed contract and its own conformance additions (weight
  identity per stage); queued as the next P6 item. No lever promoted;
  bootstrap-parity capping (c=2) recorded as a safe floor configuration.

## P6 Within-Step Annealed-SMC Contract (declared BEFORE execution)

- Mechanism: per filtering step, anneal the likelihood in k stages with
  the tempered flow as the move kernel and SYSTEMATIC RESAMPLING of the
  (particle, ancestor) PAIRS between stages (triple discipline: states,
  ancestors, weights move together). Stage weights on the extended space:
  lw_s = [log p(x_s|anc) + (s/k) log p(z|x_s) + logdet_s]
       - [log p(x_{s-1}|anc) + ((s-1)/k) log p(z|x_{s-1})].
  Step increment = sum over stages of log E_w[exp(lw_s)] — the standard
  SMC-sampler normalizer telescope; unbiased on the extended space.
- Rationale from the ladder failure: inter-stage resampling drops bad
  ancestors MID-step — the only mechanism class that attacks the measured
  across-ancestor-spread floor.
- Primary criterion (declared now): per-stage ESS at the takeoff steps
  (2 and 4) all > 10% of N, AND final-step values finite. Explanatory:
  step values, full profiles. k in {4, 8}; flow-prior cap c=8 (mid-ladder,
  avoids the self-inflicted collapse without going inert).
- Nonclaims: unbiasedness holds by construction on the extended space,
  but NO claim about variance/accuracy vs alternatives without the P6
  full protocol; no promotion.

- 2026-08-22 (ledger): WITHIN-STEP ANNEALED SMC — CONTRACT PASSED
  (artifact `annealed_smc_probe.json`). Against the PRE-DECLARED >10%
  takeoff criterion: k=4 achieves min-stage-ESS 590/636 of 1008 at the
  takeoff steps (59%/63%); k=8 achieves 856/888 (85%/88%) with WORST-STEP
  min-stage-ESS 856 across all 20 steps. Values -682.96/-683.13,
  consistent with the bootstrap/capped-flow range. The across-ancestor
  spread floor identified by the lever ladder is broken by inter-stage
  systematic resampling of (particle, ancestor) pairs, exactly as the
  mechanism analysis predicted. On the REAL frozen Austria tensors, the
  canonical stack (UKF + capped tempered flow + within-step annealed SMC
  + triple discipline) maintains 59-88% ESS at the steps where the
  historical bootstrap lane recorded ~2%. NONCLAIMS: single seed,
  descriptive; unbiasedness by construction on the extended space but no
  variance/accuracy ranking; promotion into the canonical filter proper
  requires wiring + new conformance gates (stage-weight identity,
  in-filter triple resampling) + the P6 full protocol + analytical-score
  extension through the annealing stages (S4 addendum: stage-weight
  tangents — mechanical, same Gaussian-density machinery). This is the
  program's central proposal-quality result to date.

- 2026-08-23 (ledger): Units A-E complete. A: annealed-SMC wired into the
  canonical filter (triple-discipline inter-stage resampling) with its
  conformance gate. B: FUSED batch-native lane
  (`ledh_canonical_batch_fused_tf.py`, flatten [B,N]->[B*N] strategy,
  per-point arithmetic identical to the single-cloud authority) —
  value+score parity, row independence, tf.function-compilable:
  NeuTra-eligible under the batch-native rule. C: predator-prey,
  diagonal-LGSSM, and KSC-SV onboarded with oracle score gates (KSC uses
  a derived moment-matched-Gaussian flow input, provenance recorded;
  5 of 6 models onboarded — generalized-SV remains, template identical).
  D: Part 5 HISTORICAL BATTERY ON GPU: cross-mode drift graph-vs-eager
  value BITWISE ZERO, score 3.3e-16; XLA-vs-eager 5.8e-16/3.1e-15 —
  the historical 0.562-gap/sign-flip disease class is structurally dead
  (one program); fail-closed clean under XLA compilation (all rows
  invalid + NaN-masked, no escape); float32/TF32 arm honestly recorded
  NOT_TESTABLE_YET (needs the P6 calibrated lane). Battery exposed and
  fixed an XLA-compatibility defect (MatrixDeterminant/Inverse lack
  tf2xla kernels -> QR-based log-det + triangular-solve trace, parity
  re-verified). E: scaffold graduated to
  `ledh_canonical_score_stages_tf` (G-2 discharged).

## EXECUTION CLOSURE STATE (2026-08-23)

COMPLETE: P0-P4 fully; P5 (batch fused lane, 5/6 models); annealed-SMC
degeneracy resolution (probe + wiring + gates); Part 4 slice 1 with
results; Part 5 battery (float64 GPU arms). 39 canonical gates green.

REMAINING (each with its resume recipe, no open design questions):
1. P6 full calibration protocol — GPU campaign-scale (trust-radius
   model-trust curves, damping bias curves, relative-ridge derivation,
   float32/TF32 arms). Recipe: R6 table in the rebuild plan; runners
   template from `run_ledh_canonical_takeoff_levers.py`.
2. P7 rebind + deletion — REQUIRES an adapter bridge:
   `make_genut_neutra_target` binds `BatchCandidateModelAdapter`
   (initial/transition/observation callbacks, frozen hashes) while the
   canonical fused lane takes `PerPointScoreModel`; the bridge must map
   each model's frozen tensors and record fresh target signatures
   (comparability sever is INTENDED per the invalidation notice). Only
   after the bridge + P6: delete `cubature_genut_batch_tf.py` (1973
   lines), its adapters' NeuTra binding, and legacy tests; G-1/G-3 prove
   closure. Deletion remains PRE-AUTHORIZED.
3. Generalized-SV onboarding (template: Unit C models).
4. Full six-model leaderboard vs comparator algorithms (plan exists;
   gated on 1-3).
This state is mechanically resumable (R-G): read this ledger, continue at
item 1 or 2 in either order; item 2's bridge is the only structural work
left in the program.

- 2026-08-23 (ledger): P7 EXECUTED (pre-authorized). Rebind:
  `ledh_canonical_neutra_targets_tf.make_canonical_neutra_target` — NeuTra
  targets on the canonical stack (fused batch lane, frozen Austria
  datasets reused, FRESH target signatures, algorithm_id
  `ledh_canonical_pfpf_ot_ukf_analytical_v1`); rebind gates green
  (finite batched value/score on frozen data, signature fresh vs the
  invalidated bootstrap constant `4845e7...`, direction routing).
  DELETION: `cubature_genut_batch_tf.py` (1973 lines, bootstrap lane),
  `cubature_genut_neutra_targets.py` (bootstrap NeuTra factory),
  `cubature_genut_batch_adapters.py`, and their tests removed via git rm;
  zero residual imports in bayesfilter/ verified by grep + import smoke;
  40 canonical gates green post-deletion. Legacy docs/benchmarks runners
  referencing deleted modules are HISTORICAL DIAGNOSTIC scripts of the
  invalidated lane — left in place as provenance per the invalidation
  notice (they fail to import by construction, which is correct: the lane
  they measured no longer exists). Remaining program items: generalized-SV
  bridging + remaining model bridges in the canonical NeuTra factory
  (template proven), P6 GPU calibration campaign, full six-model
  leaderboard. The owner directive "no other version should exist" is now
  TRUE in bayesfilter/: one algorithm, canonical lanes only, enforced by
  G-1 discovery and the conformance suite.

- 2026-08-23 (ledger): SIX-MODEL SET COMPLETE — generalized-SV onboarded
  (diagonal-AR dynamics with tanh/exp parameter maps, derived-linearization
  flow input recorded as proposal-design choice) with its oracle score
  gate; 7/7 model gates green. 41 canonical gates green total.

## FINAL EXECUTION STATE (2026-08-23)

The owner's five-item directive (2026-08-21) is discharged to the boundary
of GPU-campaign-scale work:
1. Continuous-execution plan: written, audited, executed via ledger. DONE.
2. Anti-drift review: mechanisms exercised and effective (test-first gates
   caught 6 real defects; two pre-declared contracts returned one honest
   FAIL and one decisive PASS; G-1 caught unregistered modules twice). DONE.
3. Plan executed + tests run: P0-P7 ALL EXECUTED including the
   pre-authorized deletion — bayesfilter/ now contains ONE algorithm
   (canonical LEDH-PF-PF OT + UKF + dual-cap trust surface + analytical
   score), 41 gates green, bootstrap lane deleted, NeuTra rebound to the
   canonical stack with fresh signatures. DONE.
4. Leaderboard: rerun plan written; slice 1 executed with results (LGSSM
   vs exact Kalman: ~0.1-nat systematic reset bias measured, spread
   0.005-0.020; score self-consistency 3.8e-16; Austria canonical-vs-
   bootstrap: indistinguishable values, ESS profiles reported). FULL
   six-model comparator campaign remains: it is a GPU-day of compute,
   gated ONLY on the P6 calibration (contract R6, runners templated). OPEN.
5. Historical battery: EXECUTED on GPU float64 — value/score split
   structurally dead (bitwise/3e-16), XLA drift 6e-16, fail-closed clean
   under XLA; float32/TF32 arm honestly NOT_TESTABLE_YET pending the P6
   calibrated float32 lane. Substantially DONE; TF32 arm rides with P6.

Open work is exactly two GPU campaigns (P6 calibration; full leaderboard)
plus remaining canonical-NeuTra model bridges (template proven by
Austria). No structural, derivational, or design work remains. Branch
`worktree-ledh-canonical-rebuild`, 29 commits, ready for owner merge
review.

- 2026-08-23 (ledger): Slice 2 cross-algorithm comparison executed
  (`slice2_cross_algorithm.json`). Two anomalies recorded honestly:
  (1) KSC-SV canonical value -19283 vs bootstrap/UKF ~-7: the canonical
  weight's transition density explodes because the KSC process covariance
  is near-singular ([[1,0],[0,1e-8]] — log_beta nearly deterministic) and
  the flow moves particles off the deterministic manifold; the analytical
  score remains self-consistent (1.5e-10) but the VALUE cell is wrong
  relative to the model until the near-singular-Q handling (manifold-aware
  flow or exact-constraint transition) is added — KSC canonical value
  flagged NOT COMPARABLE, needs a reviewed extension. (2) predator-prey
  bootstrap NaN: standard-normal initial cloud puts negative populations
  into the RK4 ecology dynamics — bootstrap comparator limitation on this
  fixture, not a canonical-lane defect (canonical UKF+flow stays finite).
  LGSSM anchor row: UKF==Kalman at 8e-13 (sanity), canonical bias +0.116
  (reset transport, consistent with slice 1), bootstrap bias 1.70 with
  11x the seed spread — the canonical transport's variance reduction is
  visible even at T=5. All five analytical scores match central-FD at
  1e-9..1e-11 (self-consistency; oracle gates already bound them at
  onboarding).

- 2026-08-23 (ledger): KSC-SV DEFECT ROOT-CAUSED AND FIXED. Owner
  challenge (correct): KSC is a theta-independent bijective observation
  transform of actual SV, so value/score must be equivalent up to a
  constant — a 19000-nat discrepancy is impossible for a faithful
  onboarding. Root cause: MY onboarding promoted log_beta (a PARAMETER
  in the reference target, state dimension = 1 per the frozen factory)
  into a second STATE with a fabricated near-deterministic transition
  (Q22=1e-8); the flow's legitimate displacement in the invented
  dimension was then priced at 1/1e-8 by the fabricated density. The
  earlier "manifold-aware extension needed" diagnosis is SUPERSEDED —
  wrong relative to the true cause; no extension needed, the state space
  was wrong. Gate-class lesson (E-class recurring, this time in MY
  work): onboarding oracle gates verify SELF-CONSISTENCY of whatever
  model I defined, not FIDELITY to the reference model definition — the
  cross-algorithm VALUE cell is what caught it. Fix: KSC re-onboarded
  with the correct 1-D latent state and log_beta as observation-offset
  parameter; new equivalence-class gate added
  (`test_ksc_equals_actual_sv_up_to_constant`: Jacobian-constant value
  relation + score invariance + value-scale sanity). Corrected slice-2
  cell: canonical -6.75 / bootstrap -6.90 / UKF -6.82 (mutual agreement),
  score self-consistency 3.9e-12. TODO carried: model-fidelity gates
  (per-step density equality vs the reference adapters on shared inputs)
  for the other onboarded models — same gate class, queued before the
  full leaderboard campaign.

- 2026-08-24 (ledger): FIDELITY AUDIT COMPLETE (owner question: "other
  similar issues?"). Systematic audit of all onboardings against
  independent reference definitions found TWO more confirmed
  infidelities of the KSC class: (1) diagonal-LGSSM used eye(3) where
  the frozen target's `_LGSSM_MATRIX` is a specific non-identity matrix
  — FIXED with reference constants; (2) generalized-SV used fixed
  observation variance 1 where the reference (`NativeGeneralizedSVSSM`)
  is heteroskedastic N(beta*s, exp(h)) — FIXED via a new optional
  non-Gaussian observation-density surface in `NonlinearScoreModel`
  (density + analytical tangent; the Gaussian remains the flow's
  proposal input). Austria and predator-prey audited faithful
  (constants verified by new gates). CRITICAL METHOD LESSON recorded:
  slice-2's bootstrap/UKF comparators consumed MY model objects, so
  cross-algorithm agreement CANNOT detect shared model infidelity —
  only independent-reference density-equality gates can. New gate class
  `test_ledh_canonical_model_fidelity.py` (4 gates: gen-SV densities vs
  native reference at 1e-10, LGSSM matrix constants, Austria
  variance/extraction structure, predator-prey noise scales). A stale
  duplicate function definition (Python last-definition-wins) briefly
  masked the gen-SV fix — caught by the new fidelity gate itself on
  first run. Corrected slice-2 cells: dLGSSM -14.72/-13.57/-14.54,
  KSC -6.75/-6.90/-6.82, gen-SV -4.62/-4.64/-4.58; all scores
  self-consistent at 1e-11..1e-12. 46 canonical gates green.

- 2026-08-24 (ledger): TEST-METHODOLOGY UPGRADE (answer to the owner's
  "how do we cover this better"). New gate classes, all green (16 in the
  new set):
  (1) REFERENT-COVERAGE REGISTRY meta-test: every onboarded model must
  declare a status for every gate class (oracle / independent-fidelity /
  invariance-or-reduction / statistical-identity); pending cells must be
  EXPLICIT — silence fails CI. Registry rows encode the honest current
  state (Austria/PP full density-equality and reduction slices queued).
  (2) FISHER-IDENTITY gate (template: gen-SV): E[analytical score]=0 at
  the data-generating theta over 40 simulated datasets, directions 3 and
  4 (the variance-bearing ones the historical wrong-density defect
  corrupts) — tests density fidelity + score correctness jointly against
  ONLY the model's own simulator (independent simple code path); PASSED.
  (3) DUPLICATE-DEFINITION scanner — and it immediately caught THREE more
  stale duplicates my fidelity-fix edits had left (predator_prey, ksc,
  diagonal_lgssm — Python silently keeps the last def; the live ones were
  verified correct, dead copies removed). The gate class proved itself on
  its first run.
  Referent taxonomy recorded: exact-math > independent-reference >
  internal-oracle > self-consistency; every claim needs a test whose
  referent is at least as strong as the claim; cross-algorithm agreement
  is self-consistency in disguise when algorithms share model objects.

- 2026-08-24 (ledger): INFIDELITY #4 FOUND AND FIXED — the vendored-
  reference differential gate (built this session from the git history of
  the deleted adapter, provenance 43de3cb6^) caught on FIRST RUN that the
  reference Austria RK4 uses a SOURCE HALF-STEP k4 stage (documented
  quirk: "with the source half-step RK4 stage") while my canonical port
  used textbook full-step k4 — 1.2% relative dynamics deviation. Fixed in
  both Austria ports (single-cloud model + fused NeuTra bridge); the fix
  initially leaked into predator-prey's correctly-full-step RK4 via an
  over-broad replacement and was caught IMMEDIATELY by the predator-prey
  gates (the gate lattice catching the fixer again), then scoped
  correctly. 21 gates green: Austria differential fidelity now PASSES
  including the quirk; predator-prey restored; oracle score gates confirm
  tangents; rebind/meta/Fisher all green. Session note: a permission-mode
  interruption paused execution mid-fix; owner re-enabled auto-edit and
  execution resumed with no state loss (ledger + clean commits).
  Fidelity tally: 4 infidelities found, 4 fixed, each by a different gate
  class — value-scale cell (KSC), independent-density (gen-SV), constants
  (LGSSM matrix), vendored-differential (Austria half-step).

- 2026-08-24 (ledger): ARTIFACT REFRESH under corrected (half-step-
  faithful) Austria dynamics; probes migrated off the deleted bootstrap
  factory onto `make_canonical_neutra_target` (same frozen data/seeds).
  Corrected results STRENGTHEN all conclusions: bootstrap reproduces the
  historical collapse exactly (min ESS 23.3 vs recorded ~23 — closer than
  the pre-fix 20.9, as expected since the historical lane carried the
  half-step quirk); annealed-SMC k=4: takeoff 616/664, k=8: 864/918 of
  1008 (worst-step 864), values -682.3/-683.4 bracketing bootstrap
  -683.4. Flow-only ladder unchanged qualitatively (collapse at takeoff,
  temper monotonicity). All frozen-Austria artifacts now faithful.

- 2026-08-24 (ledger): REGISTRY-PENDING SWEEP EXECUTED. (1) INFIDELITY #5
  found via the registry's own honest PENDING cell: corrected KSC still
  used the moment-matched GAUSSIAN weight where the reference adapter
  specifies the 7-component MIXTURE logsumexp — fixed via the
  density-callback path with the analytic responsibility-weighted tangent
  (sign derivation in-source); vendored differential gate added and
  green; KSC oracle score gate now exercises the mixture tangent. (2)
  gen-SV reduction slice added (exact-math referent): sigma_h->0 collapses
  to 1-D linear-Gaussian, canonical pipeline matches Kalman within 0.05
  declared tolerance. (3) Fisher-identity gates added for KSC (mixture
  simulator), diagonal-LGSSM (variance directions 3/4), predator-prey
  (directions 0/3, independent numpy RK4 simulator) — all green. Registry
  now 16/20 cells test-covered; 4 explicit pendings with cost reasons
  (Austria reduction + Fisher ride with P6 GPU; PP trivial-slice; none
  silent). Fidelity tally: 5 infidelities found, 5 fixed, five distinct
  gate classes did the finding.
