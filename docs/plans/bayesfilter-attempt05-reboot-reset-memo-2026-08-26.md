# Reset memo: attempt05 SV rank-ladder — reboot handoff — 2026-08-26

Date: 2026-08-26
Status: `CAMPAIGN_COMPLETE` — the attempt05 ladder RAN TO COMPLETION and
wrote `verdict.json` at 15:47 before the reboot. n=2 arm: r*(2)=6
(clean verdict). n=4 arm: r*(4)=None (NO rank ≤ 6 passes at any degree;
candidate failure, not yet root-caused). The orchestrator has already
exited — there is NOTHING to resume. Next work is DIAGNOSIS + result
note, not re-running the ladder.

This memo is the single document a fresh session needs to pick up the
attempt05 campaign after a machine reboot. It is written as an
operational runbook (modular form is intended). Governing plan:
`docs/plans/bayesfilter-attempt05-sv-rank-ladder-plan-2026-08-26.md`
(status `AWAITING_OWNER_GO` in its header but the ladder was launched;
see "Authorization" below). Program checkpoint memo it builds on:
`docs/plans/bayesfilter-squared-tt-program-reset-memo-2026-08-24.md`.

## Context

The squared tensor-train (squared-TT) program approximates a filtering
density as a squared TT with a Gaussian reference and normalized-Hermite
product bases (route C2). The central scientific question is r*(n): the
smallest TT rank whose per-step log-likelihood gap versus a trusted
reference stays under the accuracy bar for state dimension n.

attempt04 (bounded-box route C1) closed with r*(2)=6 and n=4 declared
unmeasurable in that route. The program then pivoted the reference
domain: C1 eliminated, C2 (Gaussian reference) selected, C3 kept as
fallback. Decision D2 made LGSSM the machine-precision parity oracle
(Gate A3: n=4 oracle-exact 5.5e-10 at degree6/rank3/sweeps32 on LGSSM)
and moved the r*(n) rank claim onto the stochastic-volatility (SV) arm.
attempt05 is the SV rank ladder under C2: ranks {1,2,3,4,6} × degrees
{2,4,6} × seeds {42,142,242} at n ∈ {2,4}, horizon T=20, model seed 52.

- Accuracy bar: 2.5e-3 nats per step (inherited; τ_max = bar/25).
- Cell pass rule: per_step_gap = |engine corrected_total − reference
  total| / T ≤ 2.5e-3, with a VALID reference and all engine vetoes
  clean. r*(n) = smallest rank passing at ALL three seeds at the working
  degree.
- Vetoes (`_evaluate_cell`, run_attempt05...py:167): `reference_invalid`,
  `non_finite`/`crash`, `row_ess_floor`, `tau_at_cap` (tau_max_seen ≥
  1e-4), `alpha_exceeds_declared` (alpha_max_seen > ALPHA_MAX=0.8).
- References: n=2 exact 2-D tensor grid (zero MC error, Gate-C certified);
  n=4 bootstrap particle filter with ESS-degeneracy screen + R-replicate
  SE-of-mean (PF_NS=(400k,800k), PF_R=10), validity = doubling_ok ∧
  se_ok ∧ screen_ok.

### Authorization

attempt05 is an authorized serious research campaign (owner go-ahead was
given to launch the ladder; the plan header string still reads
`AWAITING_OWNER_GO` and should be updated to `LAUNCHED`/`IN_PROGRESS` on
resume). Under the campaign-repair rule, localized infra/serialization/
numerical repair and retry proceed without renewed approval while target,
data, method, promotion criteria, vetoes, hardware class, and budget are
unchanged. External/irreversible actions still need approval at the
boundary. Do NOT commit unless explicitly asked.

## Decision / policy — do not re-litigate

These are Gate-certified scope pins from the plan §2. Changing any one
re-opens a gate and is out of scope for a resume:

- Fixture: `sv_fixture_c2_20260826.sv_model(n, seed)` — ZC24 Example 1
  synthetic values (γ=0.6, σ=1, β=0.4), coupled-A vector extension
  (A = γ·I + 0.1·randn/(n−1); diagonal observations
  Y_{t,i}=eps·β·exp(X_{t,i}/2)). The coupling is deliberate: independent
  components would factorize the target and trivialize the rank question.
- Hints: `sv_gh_hint_factory(model, gh_points=9)` (GH 9-point), frozen
  per step. Measured n=1 hint quality 1.3e-2/2.8e-2; α measured 0.67.
- Defensive floor: Student-t, ν = 27.617 = student_t_nu_criterion(
  ALPHA_MAX=0.8, cap=12). alpha_max_seen is re-checked per cell; a cell
  with alpha > 0.8 is flagged (`alpha_exceeds_declared`).
- τ policy: clamp(ε̂², 1e-6, 1e-4).
- Row law: β(d)=0.5 (d≤4)/0.10 (d>4), N=ROWS=8192 both n. Degrees ≤ 8.
- Fitter budget: SWEEPS=32; one declared repair — a cell whose fit
  residual caps may retry once at 2× sweeps (=64), recorded.
- Backend: TensorFlow/TFP, float64, XLA engine
  (`run_value_filter_branch_axis_gaussian_xla`). NumPy only in the
  diagnostic reference/hint fixtures (permitted exception).
- Lane: XLA on the 4080 SUPER (CUDA_DEVICE_ORDER=PCI_BUS_ID,
  CUDA_VISIBLE_DEVICES=1); memory growth verified per cell
  (memory_growth_verified=true in every cell manifest). Per-cell
  subprocess isolation with timeout (A3 LLVM-retrace lesson).
- Skip-established-rank: once a rank passes all seeds, higher ranks are
  not run. Resume is artifact-gated (see below).

## Current status — detailed

Accumulator: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/
rows.jsonl` (67 rows, final) + per-cell `cell_n{n}_d{d}_r{r}_s{seed}_
w{sweeps}.json`, `reference_n{n}_s{seed}.json`, and the terminal
`verdict.json`. The ladder completed; the orchestrator process has
exited (verified: no `run_attempt05` process running).

**Machine-readable verdict (`verdict.json`):**
- n=2: working_degree=6; degree_screen d2 gap 6.79e-3 (fail), d4 gap
  2.70e-4 (fail), d6 gap 3.71e-5 (pass); **r_star=6**.
- n=4: working_degree=6; degree_screen d2 gap Infinity (fail), d4 gap
  Infinity (fail), d6 gap 6.16 (fail); **r_star=null**.

The n=4 degree-screen Infinity at d2/d4 corresponds to the non-finite
crashes (Symptom B); the d6 gap 6.16 is the tau_at_cap / wrong-sign
regime (Symptom A). r_star=null means no rank 1–6 passed veto-clean at
any screened degree — the plan's declared "no rank ≤ 6 passes" boundary
outcome, NOT a silent escalation.

### n=2 arm — COMPLETE, clean verdict

**A5 VERDICT n=2: r*(2) = 6 at degree 6.** All three seeds pass
veto-clean at d=6, r=6, sweeps=32:
- per_step_gap ≈ 3.7e-5 (s=42) to ~4.7e-5 — about 50× under the
  2.5e-3 bar; corrected_total ≈ −40.86 (s=42), reference exact grid.
- Monotone rank curve. Sub-6 ranks (1,2,3,4) all fail with `tau_at_cap`
  at both sweeps=32 and the sweeps=64 retry — i.e. the sub-6 failures
  are capacity-limited (retry-invariant), not fit-budget-limited.
- Degree screen: d=2 and d=4 at r=6 fail `tau_at_cap`; d=6 passes →
  working degree = 6 for n=2.
- Dual reading recorded in prior notes: an accuracy-only crossing
  appears near low rank, but the veto-clean crossing (the promotion
  rule) is rank 6. Report the veto-clean crossing as r*(2).

No open items on the n=2 arm. This verdict is the campaign's primary
positive result.

### n=4 arm — IN A FAILING REGIME (candidate failure)

No veto-clean passing rank found at any degree. Two distinct failure
signatures, plus a reference-validity problem on one seed:

1. **`tau_at_cap` on every d=6 cell, all ranks 1–4 and 6, both sweeps.**
   The most diagnostic cell is n=4 d=6 r=6 s=42 w=32:
   - corrected_total = **+36.94** while reference_total = **−66.70**
     → per_step_gap = 5.18 nats/step (~2000× over bar). The engine total
     has the **wrong sign** relative to the reference.
   - Yet the explanatory diagnostics look healthy: rms_max = 8.05e-3
     (fit fine), row_ess_min = 2584 > floor 1260 (rows fine),
     alpha_max_seen = 0.561 < 0.8 (defensive floor not saturated),
     cond_max = 5.5e4 (Gram conditioning unremarkable),
     tau_max_seen = 1e-4 exactly (hit the clamp cap — the veto).
   - Interpretation: the τ clamp is saturating (ε̂² driven to the 1e-4
     ceiling) while the accumulated log-evidence diverges from the
     reference by ~100 nats over T=20. This is NOT a fit-resolution
     failure (rms is small) and NOT row starvation and NOT defensive-
     floor saturation. The wrong-sign total with small rms points to a
     structural mismatch in the n=4 d=6 evidence accumulation, not a
     tuning knob. **Root cause NOT yet established.**
   - Retry at sweeps=64 is `tau_at_cap`-invariant → not fit-budget.

2. **Isolated `crash` cells (fail-closed non-finite guard).** n=4 d=2
   r=6 s=42, n=4 d=4 r=6 s=42 (degree screen), and several n=4 d=6 r=1
   cells (s=42/142/242) recorded `vetoes=['crash']`. The reproduced
   crash is `ValueError: non-finite step increment (fail-closed)` at
   `bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py:307`
   (the guard after `log_increment = shift + log(zc_new) − log(zc)`).
   This is the Class B fail-closed guard doing its job; the upstream
   non-finite source (candidate: `log(zc_new)`/`log(zc)` when a z-value
   is ≤0 or non-finite, or `exp` overflow in the SV obs density at low
   degree with the branch/mixed basis) is **NOT yet diagnosed**. The
   guard must NOT be weakened to make the crash disappear (reversed-
   burden policy); diagnose the upstream non-finite instead.

3. **`reference_invalid` on n=4 s=242** (r=2 and r=3 cells carry it
   alongside tau_at_cap). The n=4 PF reference for seed 242 failed its
   validity gate (doubling_ok ∧ se_ok ∧ screen_ok). `reference_n4_s242`
   needs inspection — this makes those cells uninterpretable as rank
   evidence regardless of the engine result.

Per the Research Question Guardian: this is **candidate failure**, not
research-direction rejection. The n=4 tau_at_cap pattern is the failure
the plan's repair phases are designed to probe. Do NOT record "C2 fails
at n=4" as a program conclusion from this state — the wrong-sign total
at small rms is unexplained and could be an implementation defect in the
n=4 d=6 path rather than a capacity result.

## Bugs / blockers open (NOT resolved — carried into next session)

- Symptom A: n=4 d=6 all-rank `tau_at_cap` with wrong-sign corrected_total
  (+36.9 vs ref −66.7) at small rms (8e-3).
  - Root cause: NOT established. Leading hypothesis: structural mismatch
    in n=4 d=6 evidence accumulation (τ saturating while log-evidence
    diverges), not fit budget / rows / defensive floor.
  - Next diagnostic: dump per-step {shift, log(zc_new), log(zc), tau_t,
    z_h} for one n=4 d=6 cell and compare the running log-evidence to the
    PF reference step-by-step to localize where the sign/magnitude
    diverges. Cross-check against the LGSSM oracle at n=4 (Gate A3 was
    exact there) to isolate whether the defect is SV-specific.
- Symptom B: isolated `non-finite step increment` crash at
  squared_tt_engine_gaussian_xla_tf.py:307 (n=4 low-degree r=1/r=6).
  - Root cause: NOT established. Do not weaken the guard.
  - Next diagnostic: reproduce n=4 d=2 r=6 s=42 in FOREGROUND (the
    orchestrator discards subprocess stderr) and print zc_new/zc/z_h at
    the failing step.
- Symptom C: n=4 s=242 reference_invalid.
  - Next diagnostic: read `reference_n4_s242.json`, check which of
    doubling_ok/se_ok/screen_ok failed; a PF re-run at larger N or the
    plan's declared "unmeasurable-under-budget" outcome may apply.

## Reboot / resume behavior (READ BEFORE RESTARTING)

The ladder is COMPLETE and the orchestrator has already exited, so the
reboot interrupts nothing. Do NOT re-launch the orchestrator to "finish"
the campaign — it will find every cell artifact present and exit having
done nothing. All artifacts (rows.jsonl 67 rows, per-cell JSON,
verdict.json) are on disk and intact.

- If you ever DO need to recompute a specific cell (e.g. after a fix),
  delete that cell's JSON first — the runner is artifact-gated and skips
  any cell whose JSON exists (run_attempt05...py:209), and skips any
  reference whose `reference_n{n}_s{seed}.json` exists (:252, :255).
- Processes were confirmed exited before the reboot; nothing to clean up.

**Do NOT re-arm a per-heartbeat Monitor.** The prior session's prompt
bloat came from replying to every per-cell heartbeat, and the ladder is
done anyway. Diagnosis work below is interactive/foreground — no monitor
needed.

## Verification already run

```bash
# n=2 verdict cells (all three seeds), n=4 fail cells, references live in:
#   docs/benchmarks/artifacts/c2_completion_20260824/attempt05/
# Compact status dump:
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('docs/benchmarks/artifacts/c2_completion_20260824/attempt05/rows.jsonl')]
for r in rows:
    print(r['n'], r['degree'], r['rank'], r.get('obs_seed'), r.get('sweeps'),
          r.get('per_step_gap'), r['passed'], r['vetoes'])
PY

# Reproduce the non-finite crash in FOREGROUND (stderr visible):
conda run -n tftwogpu env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python docs/benchmarks/run_attempt05_sv_ladder_20260826.py --cell 4 2 6 42 32
```

Observed at reboot:
- n=2 d=6 r=6 pass=True all seeds (gaps 3.7e-5–4.7e-5). r*(2)=6 firm.
- n=4 d=6 all ranks pass=False tau_at_cap; r=6 s=42 corrected_total
  +36.94 vs ref −66.70 (gap 5.18/step); rms 8e-3, ess 2584, alpha 0.56.
- n=4 s=242 references reference_invalid.
- n=4 low-degree crash reproduced earlier: ValueError non-finite at
  squared_tt_engine_gaussian_xla_tf.py:307.

## Environment / exact resume commands

- Conda env: `tftwogpu`. GPU: RTX 4080 SUPER via
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. TF float64, XLA.
  Memory growth enabled+verified before device init (TF GPU Memory Rule);
  every cell manifest records `memory_growth_verified: true`.
- GPU-touching commands require escalated/trusted permissions; treat any
  non-escalated GPU failure as sandbox evidence only.
- Bare orchestrator command (artifact-gated — with all cells present it
  will exit having recomputed NOTHING; listed only for reference / for
  recompute-after-fix once you delete the target cell's JSON):

```bash
conda run -n tftwogpu env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python docs/benchmarks/run_attempt05_sv_ladder_20260826.py
```

- Run one reference or one cell directly (this is the diagnosis-phase
  workhorse — foreground, stderr visible):

```bash
# reference:  --reference n obs_seed
# cell:       --cell n degree rank obs_seed [sweeps]
conda run -n tftwogpu env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python docs/benchmarks/run_attempt05_sv_ladder_20260826.py --reference 4 242
```

## Known limitations / cautions

- Single fixture family (coupled-A SV extension) is a declared scope
  limit. r*(n) claims are for this family only.
- Statistical discipline: n=4 continuous metrics (gaps, rms, ess, cond)
  are descriptive, not a ranking. Do not say a rank "beats" another;
  say "passed the screen"/"viable"/"failed veto". The only firm claims
  are hard-veto claims (tau_at_cap, crash, reference_invalid) and the
  n=2 veto-clean pass.
- Must NOT be concluded from current state: HMC/posterior readiness;
  transfer of tuning to other models; statistical superiority over any
  method; "C2 is infeasible at n=4" (the wrong-sign-at-small-rms defect
  is unexplained and may be an implementation bug, not a capacity limit);
  any n not run.
- The fail-closed non-finite guard is Class B (adopt-by-default). Do not
  weaken it to clear the crash.
- Plan header says `AWAITING_OWNER_GO` but the ladder was launched under
  owner go-ahead; update the header to reflect the launched state and
  avoid a future session mistaking it for un-started.

## Suggested next steps (the plan, in priority order)

The ladder is done; the campaign is now in its DIAGNOSIS + WRITE-UP
phase. Do NOT re-run the orchestrator.

1. **Diagnose Symptom A (n=4 d=6 wrong-sign total).** This is the gating
   scientific question for the n=4 verdict. Instrument one n=4 d=6 cell
   to dump per-step {shift, log(zc_new), log(zc), tau_t, z_h_new} and
   compare the running log-evidence to the PF reference step-by-step;
   localize the step where sign/magnitude diverges. Cross-check the same
   config on the LGSSM oracle (Gate A3 exact at n=4) to decide
   SV-specific vs general defect. This determines whether r*(4)=null is a
   capacity finding or an implementation bug — and that distinction
   governs what the result note may claim.
2. **Diagnose Symptom B (non-finite crash / degree-screen Infinity)** by
   foreground repro of n=4 d=2 r=6 s=42, printing zc/zc_new/z_h at the
   failing step. Fix the upstream non-finite; keep the guard.
3. **Resolve Symptom C (n=4 s=242 reference_invalid):** read
   `reference_n4_s242.json`, identify the failed validity component, and
   either re-run the PF at larger N or record r*(4) at s=242 as
   unmeasurable-under-budget per the plan's declared stop condition.
4. **Write the attempt05 result note** (verdict.json is ready to cite):
   decision table + inference-status table (hard-veto screen /
   statistically supported ranking / descriptive-only / default-readiness
   / next evidence), the n=2 verdict (r*(2)=6), and the n=4 outcome
   (r*(4)=null = "no rank ≤ 6 passes" boundary, qualified by whether
   Symptom A is a bug or a capacity limit). Separate candidate rejection
   from research-direction rejection.
5. Update the program checkpoint memo
   (`bayesfilter-squared-tt-program-reset-memo-2026-08-24.md`) and the
   monograph ch38 arc per the plan §6 deliverables.

## Supersession checklist (see docs/plans/CONVENTIONS.md)

- [ ] Documents this memo materially supersedes: none. This memo is a
      reboot handoff that ADDS to (does not supersede) the plan
      `bayesfilter-attempt05-sv-rank-ladder-plan-2026-08-26.md` and the
      program memo `bayesfilter-squared-tt-program-reset-memo-2026-08-24.md`.
- [ ] No supersession banners required (nothing superseded).
- [ ] Run `python docs/plans/generate_plans_index.py` after this memo is
      committed (only if/when the user asks to commit).
