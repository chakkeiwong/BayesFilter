# Active checkpoint — completed guidance/beta separation

- Question: explain model-score error before KDM capacity/nonlinear transfer.
- Worktree: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`;
  branch `ledh-refactor-with-policy-fix`, HEAD `ea1018de1`. Preserve dirty work.
- Stage COMPLETE: 616 valid FP64 GPU/XLA evaluations, 11 arms, 4 regimes, fresh
  seeds 217000 onward; beta crossed with current-only/exact-future guidance.
  Epsilon 1/Sinkhorn 5 unchanged; optional mixture, not canonical admission.
- Plan: `docs/plans/ledh-guidance-beta-factorial-20261016.md`.
  Evidence: this directory's `result.md`, `decision.json`, `attempt02/manifest.json`.
- 23 focused tests pass. Attempt01 calibration cap 2000 failed; unchanged solver
  converges at 4022 with cap 20000, confirmed on CPU/GPU. All 154 accepted attempt01
  score vectors/likelihood errors replay bitwise. All source hashes match.
- No remaining numerical continuation blocker. One retained roundoff warning;
  max covariance residual 9.411910e-15. No precision-repair campaign is justified.
- All 20 fixed-beta guidance contrasts have log-process-SD 95% intervals including
  zero. Every future arm fails at least one descriptive heuristic comparison.
  No score-improvement/default/canonical or overall-inferiority claim follows.
- Persistent impulse: current-global MSE 9.076, future-global 21.607; future with
  current-calibrated beta 33.100, with future-calibrated beta 5.768; simple local
  heuristic 5.650. Beta interaction is exploratory, not a supported global ranking.
- Seed 220205 drives much of the stress loss; future ESS minimum 1.148 and score
  squared error 229.268. Leave-one-dataset-out mean difference changes sign.
  Saved diagnostics motivate repeated particle clouds, not more KDM capacity.
- Budget used: 729/900 GPU seconds (conservative), 54/180 CPU seconds. Both GPU
  attempts finished; no live processes. The bounded stage is complete.
- Exact next action: write/audit a bounded conditional repeated-cloud and
  particle-count localization plan using fresh fixed datasets, paired current/
  exact-future controls, independent scope-specific calibration and exact Kalman
  scores. Separate dataset variability from particle variability; do not tune
  on the preserved final rows. This is the next master stage, not a new approval
  token. Keep nonlinear/KDM expansion pending that evidence.
