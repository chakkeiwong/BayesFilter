# Reset Memo: Generic Squared-TT Filtering Program — Fresh-Agent Relaunch (2026-08-17)

Status: `HANDOFF_FOR_FRESH_AGENT`
Supersedes as entry point: `bayesfilter-squared-tt-resume-checkpoint-2026-08-17.md`
(that checkpoint remains valid; this memo adds full background).

## 0. Why this relaunch + environment note

The prior session hit an infrastructure blocker: the harness's Bash
permission classifier (an upstream claude-opus-5 call that vets
non-allowlisted commands) began timing out intermittently. Trivial
commands still ran; the pytest/benchmark commands did not. This is a
transient upstream outage, not a repo problem. Mitigation for the fresh
session: pre-approve the recurring commands (pytest under
`/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, and
`docs/benchmarks/run_*.py` invocations) or run in a permission mode that
does not classify each command. FIRST ACTION of the fresh agent is the
verification run in Section 6 — it was implemented but never executed.

## 1. Mission and governing documents (read in this order)

Goal: ONE generic Zhao-Cui-family squared-TT filtering algorithm,
model-independent engine + per-model adapters + declared tuning
procedure, with the exact analytical gradient of the declared finite
program (manual, no autodiff — Method A), scaling to n~100 states,
m~100 observations, T~120, p~300 (HMC/MLE workhorse). SV/LGSSM/etc. are
test models; other filters (SVD-UKF, SGQF, LEDH, mixture-Kalman) are
independent comparator algorithms on the leaderboard.

- Master plan (rev 3 + in-session amendments):
  `bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
- Audit chain (all closed): request -> Codex audit (7 findings) -> reply +
  erratum -> re-audit -> response2 -> `P1A_CONTENT_GATE_UNBLOCKED`
  (`...codex-reply-to-fable-response2-2026-08-16.md`). No further review
  cycles required per owner.
- UB-1 score derivation + Addendum A (manual adjoint):
  `...ub1-score-derivation-note-2026-08-15.md`
- UB-2 source ledger (sole binding provenance):
  `...source-route-ledger-2026-08-15.md`
- UB-3 structural substitution derivation (for P2S only, awaiting its
  focused review at the P2S boundary):
  `...ub3-structural-substitution-derivation-2026-08-17.md`
- Branch-axis design: `bayesfilter-squared-tt-engine-branch-axis-design-2026-08-16.md`
- Execution log with ALL defect diagnoses (Addenda 1-4):
  `bayesfilter-squared-tt-engine-p0-p1a-p1b-smoke-result-2026-08-16.md`
- P2A decision: `bayesfilter-p2a-cost-prototype-result-2026-08-17.md`
- P1B ladder plan: `bayesfilter-p1b-lgssm-value-ladder-plan-2026-08-17.md`
- Baseline leaderboard (existing algorithms, standing comparison surface):
  `bayesfilter-baseline-leaderboard-result-2026-08-15.md`

Owner decisions (binding): D1 tau is per-scope TUNED (never silent 0);
D2 structural case in-program via exact Dirac substitution (invertible
subclass, V13 fence). Owner execution style: minimal stopping; review at
phase ends; patch repairable issues; stop only for true blockers.

## 2. Code state (all uncommitted in the working tree, ~49 dirty/untracked)

New library modules (bayesfilter/highdim/):
- `retained_quadratic_form_tf.py` — P1A RetainedQuadraticForm: exact
  suffix-Gram marginal (Prop. 2 structure), dual measure evaluators
  (reference-measure primary; physical via density-of-measures), tangents
  (dot_prefix, dot_E, dot_Zh), scale-relative symmetry assert.
- `squared_tt_engine_v0_tf.py` — value engine, PROGRAM v0.3:
  branch-axis target assembly (branch index = discrete TT axis with
  counting-measure mass; smooth signed branch targets u_g*sqrt(G); NO |h|
  kinks; boundary rank capped by fit rank); RELATIVE defensive mass
  tau_abs = tau*Z_h_prev (v0.2); SMOOTH SHIFT s = logsumexp(log_f) - log N
  (v0.3, replaced argmax max-shift in BOTH engines — no tie machinery);
  Cholesky branch factor with declared relative Gram floor
  (branch_gram_floor=1e-12, same factor in value and score = V5);
  `_fixed_als_fit_traced` (per-update checkpoints for the adjoint);
  optional tensor-GL `quadrature_order` rows (diagnostic, n<=2) vs
  scattered frozen rows (ladder default, V2).
- `squared_tt_adjoint_tf.py` — manual adjoint node primitives (UB-1
  Addendum A): solve_node_adjoint, design_assembly_adjoint,
  sqrt_target_adjoint (pre-v0.3 argmax form; engines now use softmax
  cotangents inline), retained_evaluator_adjoint, prefix_rows_adjoint,
  gram_chain_adjoint, cholesky_vjp; PLUS (UNVERIFIED, Section 6):
  `scaled_normal_solve` (derivative solves through the value path's
  scaled augmented QR) and `forward_jvp_replay_scaled` (ordered forward
  JVP over traced updates, scaled solves).
- `squared_tt_adjoint_engine_tf.py` — full-path adjoint score engine:
  forward trace + reverse sweep; v0.2/v0.3 semantics; lambda solve routed
  through scaled_normal_solve (UNVERIFIED).

Tests (tests/highdim/):
- `test_p1a_retained_quadratic_form.py` — 5 tests GREEN (U-MARG-TYPE-1,
  U-MARG-DERIV-1, U-MEASURE-1, U-TAU-1, symmetry).
- `test_p2_adjoint_nodes.py` — 7 pairing/FD node tests GREEN before the
  scaled-solve edit (re-verify).
- `test_p2_adjoint_engine_fd.py` — n=1 GREEN (4.7e-10); n=2 runs at
  rank 2 (well-conditioned regime; rank-3 Gram is rank-degenerate at n=2
  and its rotating null Cholesky column makes FD invalid — documented in
  the test docstring). n=2 status pending the repair verification.
- `test_p2_adjoint_vs_forward_jvp.py` — I-P2-4 instrument, now using
  `forward_jvp_replay_scaled`, xfail REMOVED, asserts 1e-9 (UNVERIFIED).
- `test_p1b_engine_v0_lgssm_smoke.py` — branch-axis gates n=1/n=2 were
  GREEN pre-v0.3 (n=1 re-verified under v0.3; n=2 needs rerun, ~40 min);
  naive route strict-xfail historical record.

Benchmarks: `run_baseline_leaderboard_20260815.py` (done, artifact
attempt01), `run_p2a_cost_prototype_20260817.py` (done),
`run_p1b_lgssm_value_ladder_20260817.py` (attempt01 = program-defect
discovery artifact under old v0.1; NOT rank evidence; rerun needed).

## 3. Key results so far

- P1A: PASSED (exact-marginal machinery proven).
- P2A: forward tangent replay ~326x value at p=300 -> ADJOINT selected
  (gate <=6x carried to the adjoint implementation); solver-reuse checks
  passed (8e-17 / 4.7e-11).
- Adjoint n=1: exact (FD rel 4.7e-10; value equality with value engine
  bit-exact).
- Defect ledger (all diagnosed via FD-quality-first, all repaired or
  fenced): |h_prev| kinks (v0.1 -> branch axis); absolute-tau value
  discontinuity (-> v0.2 relative mass); argmax shift jumps at fit
  resolution (-> v0.3 smooth shift); rank-degenerate Gram wiggle at n=2
  rank 3 (-> tuning gate: rank must keep Gram conditioned); UNSCALED
  derivative solves losing digits at ill-conditioned fits (-> scaled
  solves, THE UNVERIFIED REPAIR).

## 4. Known open items / honest caveats

- The scaled-solve repair is implemented but never executed (Section 6).
- `sqrt_target_adjoint` in the node module still carries the pre-v0.3
  argmax form; the engines use inline softmax cotangents. Harmonize or
  document when touching that module (node test covers the old form).
- v0.3 changed the declared program: U-TAU/smoke assertions were written
  for absolute tau; U-TAU-1 in the P1A file tests the RetainedQuadraticForm
  API (still valid); the ENGINE-level tau semantics are v0.2 relative —
  keep this distinction straight when extending tests.
- P1B ladder attempt01 r_star=null is NOT rank evidence (old program).
- UB-3 needs its focused Codex review only at the P2S boundary.
- The plan file has NOT yet been updated with v0.2/v0.3 program semantics
  (they live in the Addenda); fold them into the plan at the next phase
  review.

## 5. Environment facts

- Python: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python` (TF 2.19.1,
  TFP; ALWAYS `CUDA_VISIBLE_DEVICES=-1` for these CPU-deterministic runs).
- Repo: `/home/chakwong/BayesFilter`, branch main @ 18cfe609, all
  program work uncommitted (owner has not asked to commit).
- Long runs: n=2 smoke cell ~40 min; ladder cells minutes..86 min;
  background + monitor pattern works well. tf.function/XLA not yet used
  (P3).

## 6. CONTINUATION POINT — first actions in order

1. Verification run (the blocked command):
   `CUDA_VISIBLE_DEVICES=-1 <python> -m pytest
    tests/highdim/test_p2_adjoint_nodes.py
    tests/highdim/test_p2_adjoint_vs_forward_jvp.py
    tests/highdim/test_p2_adjoint_engine_fd.py -q`
   - All pass -> repair confirmed; record in Addendum 5; extend FD-gate
     comment (FD resolution-limited at n>=2; I-P2-4 decisive).
   - I-P2-4 >1e-9 but improved -> bisect per-update via pairing identity
     (first diverging update localizes it); check solve_node_adjoint
     residual precision and any remaining raw matvec digits.
   - U-ADJ-SOLVE-1 fails -> scaled_normal_solve bug; compare against
     `_solve_scaled_augmented_ridge` solution on the same system (~1e-15
     agreement expected on well-conditioned fixtures).
2. v0.3 smoke reruns: n=1 gate (quick re-check), n=2 gate (~40 min,
   background).
3. P1B ladder rerun under v0.3: reuse
   `run_p1b_lgssm_value_ladder_20260817.py` -> attempt02 (fresh dir);
   declared tolerance unchanged (2.5e-3/step); expect the shift-jump
   noise floor GONE; produce r_star(n) for n in {2,4,8} + E-conditioning
   telemetry. If cells are slow, trim ranks to {2,4,6} first.
4. T=120 adjoint-state stress (P2A full-horizon obligation): n=2, T=120,
   measure peak memory/wall of forward-trace+reverse vs value-only;
   store-vs-recompute trade if checkpoint memory binds.
5. Phase review: fold v0.2/v0.3 semantics + rank-conditioning tuning gate
   into the master plan; then P3 (XLA port, 1e-12 parity), P4 (adapters +
   near-bit SV reproduction + leaderboard integration), P5 (tuning v1.1
   incl. T-tau step), P2S (after UB-3 review), P6 (full leaderboard +
   HMC campaign + NAWM-representative gate).

## 7. Discipline reminders for the fresh agent

FD-quality-first before blaming analytic code; single-run gaps are
descriptive; declared tolerances BEFORE execution; fresh versioned
artifact dirs (attemptNN); never overwrite prior evidence; plain-language
verdicts (correct / wrong-relative-to-target / unsupported / not
checked); V1-V13 vetoes in the plan bind all new code; no wall-clock
claims without measured artifacts (V12); repair claims only after
verifying artifact text (ledger E17); "called != active" (E16).
