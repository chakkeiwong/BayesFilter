# NeuTra precision/training evaluation: completed handoff

The user's authorized implementation, Git synchronization and complete bounded
training study are finished. No campaign worker remains active.

- Remote main verified at `f3995a06a467f16574f96bbc8a68ccbbc4e30dad`.
  Result commit `c130c1027`; remote merge had no conflicts or NeuTra changes.
- Clean execution checkout: `/tmp/BayesFilter-neutra-precision-20260924`, branch
  `codex/neutra-precision-20260924`. Shared checkout has unrelated concurrent
  work; its HEAD was not moved. Owned plan/result notes were safely synchronized.
- Final report: `docs/plans/bayesfilter-neutra-training-evaluation-results-2026-09-24.md`.
- ART = `docs/plans/artifacts/neutra-precision-training-2026-09-24`.
  `master-progress.json` says `training_and_bounded_downstream_evaluation_complete`.
  Final evidence/accounting: `terminal-summary-01/result.json`,
  `manifest-index.json`, and local `git-integration.json`.
- Remaining campaign balance: 23,037.158754 worker seconds (6.40 hours).
  Deadline Sep25 18:00 Asia/Shanghai. No additional budget grant was used.

All planned fits and checks completed: three IAF maps at4,096 updates, three
NAF maps at8,192 after the log-weight underflow-guard repair, and legacy control
at4,096 additional updates. Every final map passed the1,000-point finite/status
and inverse stress checks. Failed attempts are preserved and charged.

The common2,048-point paired bootstrap comparison supports local improvement
in loss AND squared residual for all3 IAF seeds. Baseline vector residual RMS
4.5135 fell to0.5448/0.7912/0.9984. NAF seed1 passed, but seeds0/2 had RMS
21.8480/32.5173 despite lower losses, so NAF did not replicate improvement.
The legacy control also passed (RMS1.0703), making added optimization a plausible
explanation. There is no architecture ranking or universal TF32 conclusion.

Coverage remains unresolved: final IAF banks have2,048/2,048,2,048/2,048 and
2,047/2,048 positive observation weights; legacy0/2,048; NAF0/2 reach both signs.
These are transport draws, not posterior probabilities. The standard1,000-point
NAF0/2 residual RMS36.4544/19.2400 exposes sharp tails. A bounded16-point GPU
finite-difference check verified physical/full transformed scores with maximum
scaled error1.14e-7; local derivative error does not explain those tails.

HMC nominated IAF0 with four sign-dispersed chains and identity latent mass.
L3/5/9 at epsilon0.0168302 and L3 at0.0252454 were numerically healthy, but all
requested higher epsilon (acceptance98.1--99.3%). Tuning ended `partial_budget`
after2,616.959s, exceeding its2,500s sublimit by one synchronous chunk (116.959s).
Total worker2,765.557s remained within the8,000s envelope. No verified kernel,
posterior sampling or independent reference agreement was obtained. This is
incomplete tuning evidence, not a mixing failure.

If research resumes, use the final report and preserved search checkpoint.
Reprice full HMC continuation and posterior work before allocating remaining
budget. Investigate NAF sharp transition/optimizer sensitivity while preserving
coverage. Do not claim convergence from local residuals or acceptance, do not
re-run completed training, and preserve all previous evidence.
