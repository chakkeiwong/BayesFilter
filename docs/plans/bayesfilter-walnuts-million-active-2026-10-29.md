# Active replication and supplied-mass reset

## Analytic restart supersedes the first attempt

On October 29 the owner explicitly requested stopping the literal-SciPy run
and using the faster equivalent evaluator for every method. The first campaign
is stopped; `shutdown-verification.json` confirms no live campaign processes.
Partial draws/checkpoints and the completed CPU result are preserved under
`walnuts-million-20261029-01/` and will not be pooled with new draws. The GPU
arm had already failed with CUDA_ERROR_LAUNCH_FAILED (driver Xid 13), so its
result is execution-failure evidence. Root cause is not established.

Both Python arms now use the analytic evaluator for tuning and sampling. TF
CPU/GPU already use its analytic equivalent inside XLA. Fresh analytic Appendix
C warmup gives h=0.3228754012018425, delta=0.15442326531457057, shared across all
four restarted methods. Nine focused tests and the four-process 64-draw smoke
pass. See the active amendment and skeptical audit in the replication plan.

New output: `docs/benchmarks/artifacts/walnuts-million-analytic-20261029-01/`.
The approved budget is charged cumulatively: prior maximum 2,222.77 seconds
plus 56.43-second preflight, then a common 54,000-second (15-hour) per-arm cap.
This is the second and final long launch within the original two-attempt cap.
Use physical GPU 1 (same model, idle at launch), verified memory growth, FP64,
TF32 off and XLA; all four methods start in separate processes together.
An equivalent evaluator changes neither target nor scientific screens, but
the run no longer claims literal SciPy execution or bitwise script replication.

Launched at 2026-10-29 03:10:54 UTC (11:10:54 Asia/Shanghai). Supervisor PID
781684; original-author PID 781688; corrected-author PID 781689; TF CPU PID
781690; TF GPU PID 781691. Each method owns its process group, with four
internal CPU workers for each author method. The adjacent `.launch.json`
records exact commands and cumulative budget charges. The supervisor writes
progress/checkpoints and automatically analyzes completed draws. At the first
inspection all four were running; both Python arms and TF CPU were advancing,
and GPU 1 had verified memory growth before XLA compilation. Read the live
manifest for current status; these PIDs/statuses are historical observations.
Do not duplicate or restart this second attempt.

## Historical first-attempt launch notes

The user approved the 16-hour per-method cap after literal-author SciPy
preflight projected longer runtime than the earlier analytic-reference
estimate. All four methods launched together on October 29 in separate
processes, supervised by PID 771967. Four chains run per method, one million
iterations per chain; Python references have four internal chain processes.

| Method | Process |
|---|---:|
| Original author Python | 771971 |
| Corrected author Python | 771972 |
| TF CPU/XLA | 771973 |
| TF GPU/XLA | 771974 |

Live manifest and results root:
docs/benchmarks/artifacts/walnuts-million-20261029-01/
Supervisor log: sibling walnuts-million-20261029-01.supervisor.log.
Source snapshots protect the campaign against subsequent workspace edits.
Progress JSON and state/RNG checkpoints update every 1024 transitions. The
supervisor enforces the cap, collects statuses, then runs preserved-draw
analysis automatically into comparison.json, REPORT.md and v-density.svg.
Do not restart, duplicate or overwrite the running campaign.

Protocol: literal 11D author target and Appendix C warmup. Newly measured
h=.3666358310669994, delta=.19520158705471408; shared across methods.
The author-original and author-corrected outputs remain separate. GPU 2 has
growth verified; the installed readiness helper supports only GPUs 0/1, which
were busy, so trusted NVIDIA inventory and the worker's growth checks selected
the third same-class GPU. Concurrent timings are not isolated speed evidence.

Previous estimate correction: a five-repeat, single-state diagnostic measured
153.48 microseconds per original SciPy density/gradient call versus 7.07 for
the earlier analytic reference (21.71x). This explains evaluator overhead,
not the exact end-to-end ratio: tuning and trajectories also changed.
Artifact: docs/benchmarks/artifacts/walnuts-million-evaluator-overhead-20261029-01.json.
The first 1024-draw projected reference runtime is startup-sensitive.

Preflight 01 failed only at GPU checkpoint serialization after sampling;
repaired with explicit CPU placement outside XLA. Preflight 02 completed all
four methods at 1024 draws. Short-chain posterior screens fail, as expected;
that is not a continuation veto. Long results are not yet available.

Supplied-mass implementation is completed; see
bayesfilter-walnutpie-supplied-mass-result-2026-10-29.md. Dense input supports
fixed matrices and diagonal adaptation in a fixed dense basis, not unrestricted
dense covariance learning. No git commit requested in this turn.

Last inspected at approximately 145 seconds supervisor wall time: all four
methods running with no tracebacks; TF CPU 86016 draws/chain, TF GPU 7168,
original author chains 5120--7168, corrected author 6144--8192. These are
historical progress observations, not completion claims. Early projected
completion fell to roughly 7.2 hours for the slowest original-author chain,
5.7 hours corrected author and 4.2 hours TF GPU as startup became less dominant.
The 16-hour cap remains a ceiling; estimates may change with funnel geometry.
Final dense-diagonal parity passed for both kernels (2 tests, 28.37 seconds).
