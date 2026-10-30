# Controlled-SMC / KDM active checkpoint

Updated 23 October 2026 after the owner-requested commit/main synchronization.
Worktree: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Branch: `ledh-refactor-with-policy-fix`.
Source snapshot: `9bc3e7ec7`; tested main integration: `90515228c`, brought back
to LEDH. The refreshed planning state belongs to the subsequent documentation
commit. See the [sync record](controlled-smc-kdm-main-sync-and-refresh-20261023.md).

Active question: do freshly fitted controlled-SMC-derived and SIR neural-KDM
proposals improve moments produced by the actual corrected LEDH consumer at
N>=1008, using simulated labels rather than unknown model scores?
Stage: synchronization complete; fresh research preparation reviewed.
Authoritative next plan: [N1008 restart](controlled-smc-kdm-n1008-restart-20261023.md).

Checked findings:

- All prior N64 timing, training, selection, moment and score results are
  disqualified. Do not reuse their weights, calibration or simulated datasets.
- The N1008 guide-free runtime and dependency investigations are complete:
  [runtime](artifacts/ledh-sir-n1008-runtime-20261023/result.md) and
  [trace](artifacts/ledh-sir-n1008-slowdown-trace-20261023/result.md).
  They do not establish learned-guide training feasibility or efficacy.
- Determinant and innovation validity remain checked. Keep OT(1,5), the shared
  Contract-E reset, full correction and analytical model-score recursion.
- The campaign callers still hard-code N64. The neural trainer uses a traced
  per-row gradient loop; it is not a batch-native target implementation.
- CPU-only integration checks passed: 97 focused tests, 56 merged batch-path
  tests and the oracle commit hook. No GPU performance claim follows.

Exact next action: execute restart step 1's scope/configuration repair, followed
by actual-consumer wiring and gradient checks. Then measure a bounded fresh
N1008 update before sizing the training/holdout campaign. Keep final-cloud
moment loss distinct from local proposal diagnostics and model-score accuracy.

Execution budget: no new GPU time spent or assigned by this sync/refresh.
Historical campaign caps are not reset. Record a feasibility cap and command
before its launch, and a measured adequate total budget before new training.
Environment for focused checks: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`;
CUDA was deliberately hidden. GPU work requires escalated access and verified
memory growth. No task experiment is running.

Preserve the root `surrogate-hmc` edits, main's concurrent LaTeX work and local
bulk experiment files. The current request authorizes local commits and branch
synchronization; no remote push was requested.
