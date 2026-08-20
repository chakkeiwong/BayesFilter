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
