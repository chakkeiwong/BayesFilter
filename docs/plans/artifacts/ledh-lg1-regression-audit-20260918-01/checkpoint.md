# LG1 regression-audit checkpoint

Active question answered: did the graceful-failure changes cause the new LG1
accuracy failure? On identical N1024/T10/phi=-0.9/replicate2/cap16/flow3 inputs,
baseline 6fbcf314 and execution 8a4cffa2 both have 16.0006% covariance error.
Core and actual consumer parity pass; no material regression in this case.
Source comparison of the specific September 3 SUCCESS report found different
Cubature/GenUT evaluator wiring and a validity-only criterion; it does not
establish canonical LEDH/Kalman accuracy. Other earlier successes not audited.

Worktree: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917,
branch ledh-graceful-failure-execution-20260917. No production code changed.
Result: ../../ledh-lg1-regression-audit-result-2026-09-18.md.
Evidence: comparison.json, earlier-success-call-chain.json, run-manifest.json.
All jobs ended. Three filter evaluations; one pre-evaluation preparation retry.
GPU charge 243.3921/600 seconds; CPU charge 37.7667/180 seconds; allowance closed.
The original campaign's three calibration attempts remain exhausted. This
diagnostic does not constitute another candidate or restart calibration.

Next scientific action: bind the intended prior successful post-rebuild case
to its exact evaluator, settings, data, and gate, then design a bounded repair
of the existing LG1 approximation error. No blanket regression-free claim,
posterior qualification, resumed campaign, or merge follows from this audit.
