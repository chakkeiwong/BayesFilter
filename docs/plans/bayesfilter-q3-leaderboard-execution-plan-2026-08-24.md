# Q3 Leaderboard Execution Plan (2026-08-24)

Governing program: `bayesfilter-ledh-canonical-completion-program-2026-08-24.md`
(Q3), inheriting the Part-4 rerun plan's contract
(`bayesfilter-ledh-canonical-leaderboard-rerun-plan-2026-08-22.md`).
Budget: <= 1 GPU-day; per-process cap 100 min; 3-consecutive-failure stop.
Output root: `docs/benchmarks/artifacts/ledh_canonical_leaderboard_2026-08/q3_board_<ts>/`.

## Grid

Six rows: linear-LGSSM exact anchor, diagonal LGSSM, predator-prey,
KSC SV, generalized SV, Austria SIR (frozen observations). Claim scale:
N=1008 particles, horizon T=20 (Austria: frozen T=20 tensors; others:
observations SIMULATED from each model's own law at theta0 — an upgrade
over slice-2's noise-only observations, recorded per row). 16 paired
seeds for value cells (SQMC precedent); 8 seeds for score cells with a
single-seed central-FD self-consistency reference (FD is explanatory).

Arms per row (value): exact reference (linear rows), canonical LEDH,
bootstrap PF (systematic resampling — the slice-2 comparator's
no-resampling defect is fixed and gated, see below), UKF Gaussian
filter. SGQF/zhao-cui comparator hookup rows are slice B (deferred,
recorded ABSENT with reason — the repo comparators exist for SIR and
gen-SV but need their own data-convention bridge).

Per-scope configuration discipline (LEDH tuning rule): the Q2
calibration (annealed k=4, c=8, f32/TF32+cap lane) is FROZEN-AUSTRIA
scope. Only the Austria row runs it (GPU f32/TF32 arm + f64 anchor arm).
Other rows run their onboarding-gate configurations (plain canonical,
f64) — Q2 values are warm starts elsewhere and are NOT transferred.
Score cells run the f64 score lane; the Austria score cell uses annealed
k=4 (ESS-regime caution from Q2 Curve 7: plain-mode score bias is
O(1/ESS); k=4 holds stage-ESS ~45% at N=1008), others plain with their
per-step ESS recorded.

## Evidence contract

- Hard vetoes first: non-finite value/score, program_valid false,
  conformance suite not green at the run commit (no stamp -> no row).
- Absolute correctness claims ONLY on exact-reference rows: |canonical
  mean - Kalman| on linear rows (value), |analytical - FD-of-exact-
  Kalman| (score). Everything else: descriptive tables with per-seed
  spread; NO ranking language without uncertainty support; bootstrap
  vs canonical differences are descriptive proposal-quality context.
- G-5 stamp: artifacts carry `alg1_conformance` = suite version +
  commit; the runner refuses to stamp unless the canonical gate battery
  passes at launch.
- Not concluded regardless of outcome: posterior correctness, HMC
  readiness, statistical superiority of any algorithm, transfer of any
  per-scope configuration.

## Comparator repair (pre-run, gated)

The slice-2 bootstrap comparator propagates children without resampling
while treating incoming weights as uniform in each step's increment —
wrong relative to the bootstrap-PF likelihood decomposition for T > 1
(fidelity tally item #6). Repair: systematic resampling each step.
Gate (the class that should have caught it): bootstrap value on the
linear-LGSSM anchor must converge to the exact Kalman value (declared:
|mean over seeds - Kalman| < 3*seed-SE + 0.5 nats at N=4096, T=8).

## Pre-mortem

(a) Simulated-data upgrade changes row values vs slice-1/2 — intended,
comparability sever recorded; (b) a GPU f32 arm could NaN on non-Austria
models if misapplied — prevented by the per-scope rule (f32 arm is
Austria-only); (c) score FD self-consistency could fail from FD step
size on stiff rows — FD is explanatory only and h is recorded; (d) the
16-seed spread could be large enough to make descriptive tables useless
— that is itself the honest finding if it occurs.
