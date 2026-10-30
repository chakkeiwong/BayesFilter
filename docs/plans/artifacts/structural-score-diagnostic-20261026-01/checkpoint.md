# Recovery checkpoint — 2026-10-26

- Task: trace Model B score discrepancy, review and execute diagnostics. Numerical stages complete; no process remains.
- Checkout: /tmp/BayesFilter-structural-score-diagnostic-20261026, branch structural-score-diagnostic-20261026, base a54add9e78678c5bd5a36ba89164586bd97dab34. Main and the other agent's surrogate-HMC branch untouched.
- Plan: ../../structural-score-diagnostic-2026-10-26.md. Results: result.md, source-audit.md, summary.json, t2-decomposition.json, execution-record.json. Successful raw attempts: 10,11,13,14,15. Failed evidence preserved.
- Findings: no demonstrated derivative error at inspected truth; FP64 T1/T2 agreement about 4e-11. Tiny FP32 finite differences suffer cancellation. T2 reset contributes +.014979 log prediction and +.030208 persistence score; 64-node quadrature refinement confirms it. T50 persistence mean-reference gap -.225850 remains unresolved.
- Diagnostic-only repairs: full ladder and baseline trace retention, truthful dtype metadata, fresh output directories, future failure records. Production FP32/N=4032, controls, target and HMC unchanged.
- Tests: 19 focused structural tests passed. Successful process wall: 368.56 GPU-enabled seconds + 529.94 CPU-only seconds. Conservative totals including failed-run intervals: 1,927.3 and 802.6 seconds; not GPU utilization. Limits: 2 GPU hours + 30 CPU minutes.
- Final checks: Python compilation and CLI help pass; all 25 JSON files parse without non-finite JSON constants; final runner SHA-256 matches summary.
- Disposition: investigation complete and closed at the owner's request. No new runs are scheduled. Future reset repair/reference refinement would be a separate accuracy study.
- Final report: ../../structural-score-final-report-2026-10-26.md. The owner authorized final reporting, commit, and push of this diagnostic work on its current isolated branch.
- Close-out checks: archived four original data/design files in reproduction-inputs.tar.xz and verified member checksums. All 26 JSON files parse; 11 local links resolve; runner hash and Python syntax pass. No temporary source checkout is required to recover the archived inputs.
