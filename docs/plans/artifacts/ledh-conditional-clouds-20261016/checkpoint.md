# Active checkpoint — nonlinear protocol after conditional Gaussian localization

- Branch `ledh-refactor-with-policy-fix`, commit `ea1018de1`, worktree
  `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
  Preserve prior dirty work. Replication and the 616-row factorial are complete.
- Active question: can backward information improve the model likelihood score
  on a nonlinear target in the same declared consumer?
- Plan: `docs/plans/ledh-conditional-cloud-localization-20261016.md`.
  Skeptical/default audit complete. Eight arms, N=32/64, two fixed datasets per
  regime, twelve clouds each; independent calibration/validation. OT=(1,5).
- Scope: optional shared covariance mixture, FP64 GPU/XLA diagnostic;
  no canonical/default/KDM or population-ranking claim.
- COMPLETE, 17 October: N32/N64 each produce 416 valid rows. All eight
  calibrations converge. No roundoff warnings; 79 runtime source hashes match.
  All 27 distinct focused tests pass, including shared-consumer wiring,
  independent variance algebra and data/cloud/calibration integrity.
- Evidence: `result.md`, `decision.json`, and both `n*-attempt01/manifest.json`
  under this directory. Both future arms fail the descriptive heuristic screen
  (35/64 contrasts). No primary guidance improvement is supported; one
  exploratory conditional interval indicates harm. Doubling N lowers observed
  global-arm MSE in 7/8 cells, with five pointwise intervals below zero.
  Stress mean error and variance both decrease descriptively. No population
  ranking, asymptotic-bias conclusion or default promotion follows.
- Budget charged: 815.378/1800 GPU seconds, two of four launches; 45/180 CPU
  seconds conservatively. Unused: 984.622 GPU and 135 CPU seconds. No retry.
  GPU UUID ends 305b (4080), memory growth/FP64 placement/XLA confirmed.
- No numerical continuation veto remains. Gaussian localization is complete;
  no extra count ladder or precision-repair campaign is required to continue.
- Exact next action: audit one nonlinear model's oracle and actual shared
  consumer call chain, then write the target-specific fitting/comparison plan
  for master step 4. Include independent tuning, heuristic adversaries,
  conditional score uncertainty and a bounded budget before training/runs.
  KDM capacity must answer a checked nonlinear representation need. Do not
  reuse prior final data for tuning or treat this result as canonical admission.
