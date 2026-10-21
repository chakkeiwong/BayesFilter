# Active campaign checkpoint

Question: test marginal-mixture likelihood/analytical-score accuracy as a component
for surrogate HMC. No new posterior samples in this experiment.
Branch ledh-graceful-failure-execution-20260917, HEAD f5e69d716.
Plan: docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md.
24 GPU-worker-hour total cap; dimension-extension/budget questions remain pending.

Master 3258619 is SIGSTOP-held; its child 3290853 completed LG3 T10 and is a zombie.
Do NOT resume or duplicate this master: initialization source has changed.
The three low-dimensional LG scopes support local non-inferiority in the old
version; preserve them as version-specific evidence. High-dimensional rejections
were caused by a TF32 stationary-initialization guard failure before filtering.

Repair changes only small stationary products to FP64, keeping solves and full
filter FP32/TF32. Protected old source and all checks are in
../surrogate-hmc-marginal-initialization-repair-20261006-01/.
Isolated GPU repair passes all 24 cases; maximum FP32 covariance/tangent errors
5.89e-7 / 1.36e-6. Seventeen focused CPU tests pass. Full-consumer GPU regression
is running (PID 3596455); inspect its result before migration.

Exact next action: finish bounded full-consumer check (10 minute GPU repair cap).
If pass, reconcile completed old worker using its manifest plus startup allowance,
terminate only the held scheduler, and migrate to fresh attempt 03. Charge all
old worker time and repair probes, but not idle scheduler hold, to the same24h.
Fresh calibration dataset ID1; rerun affected LG scopes and the pilot. Preserve
unchanged M13 precision evidence with explicit provenance. Engineering recheck.
Then restart the same master and reporter, monitor nonlinear references and
high-dimensional validity; no new direction or approval gate.
Unrelated staged HMC tuning files must remain untouched.
