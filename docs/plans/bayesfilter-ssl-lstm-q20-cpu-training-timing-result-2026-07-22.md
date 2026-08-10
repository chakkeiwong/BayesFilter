# q=20 SSL-LSTM CPU NeuTra Training Timing Result

This note initially summarized relaxed run `r2`. That run is historical timing
provenance only because its process tree created 213 native OS threads. The
strict completed result is
`docs/plans/artifacts/ssl-lstm-q20-cpu-training-timing-2026-07-22/r3/result.json`.

Strict configuration: q=20, `(32,32)` dense IAF, three stages, batch size 100,
eight persistent scalar CPU target workers, parent TensorFlow 4 intra-op plus 1
inter-op thread, one intra/inter-op thread per worker, and non-XLA CPU graph
execution. The configured TensorFlow compute-pool total is 21, affinity is
limited to CPUs `0-49`, and the observed process-tree native thread count is
exactly 50: 10 parent threads and 5 threads in each of eight workers.

Non-XLA execution is deliberate. A calibration probe found that one tiny CPU
XLA function creates about 106 native threads in this TensorFlow build, so CPU
XLA cannot satisfy the literal 50-thread limit. This is a diagnostic exception,
not a change to the repository XLA default.

Measured values:

- Eight-worker pool startup and warm value call: `45.15 s`.
- First full batch-100 update: `4.18 s`.
- Five steady full updates: mean `2.625 s`, min `2.615 s`, max `2.632 s`, sample SD `0.008 s`.
- Validation plus support probe at the 250-step cadence: `1.186 s`.
- Terminal 256-point audit: `1.950 s`.
- Total strict diagnostic wall time: `71.21 s`.

Descriptive extrapolation, including one initial validation/support cycle per
stream, periodic checkpoint cycles, cold first update, and terminal audit:

| Streams | Steps per stream | Estimated wall time | Observed steady-step sensitivity |
| ---: | ---: | ---: | ---: |
| 1 | 250 | `709 s` (`0.197 h`) | `707-711 s` |
| 1 | 1,250 | `3,339 s` (`0.927 h`) | `3,327-3,347 s` |
| 1 | 2,000 | `5,311 s` (`1.475 h`) | `5,292-5,325 s` |
| 2 | 250 | `1,371 s` (`0.381 h`) | `1,367-1,375 s` |
| 2 | 1,250 | `6,631 s` (`1.842 h`) | `6,607-6,648 s` |
| 2 | 2,000 | `10,575 s` (`2.937 h`) | `10,537-10,602 s` |

These are descriptive extrapolations from five steady steps, not confidence
intervals. They exclude HMC acquisition and do not support transport-quality,
posterior-correctness, convergence, GPU/CPU ranking, or scientific-validity
claims. The diagnostic says that CPU-only development/training is feasible in
wall-clock terms, but the repository's GPU NeuTra execution policy remains in
force for claim-bearing training.

## Decision Tables

| Decision | Criterion | Status | Next justified action | Not concluded |
| --- | --- | --- | --- | --- |
| Strict CPU timing accepted | Complete finite run with process-tree native threads `<=50` | Passed with exactly 50 | Use `r3` estimates for strict CPU fallback scheduling | CPU training quality or HMC readiness |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Passed for strict `r3`; historical `r2` failed the literal native-thread cap |
| Statistically supported ranking | None |
| Descriptive-only differences | All timing and extrapolation variation |
| Default readiness | Not assessed |
| Next evidence needed | GPU claim-bearing NeuTra training, then sequential q=20 HMC |

## Post-Run Red Team

The strongest alternative explanation is that the 16 scalar workers and target
implementation dominate this small four-parameter training workload; a
different CPU worker topology or a batch-native target could change the
estimate. The result that would overturn the scheduling conclusion is a
repeat under the same contract with sustained target time materially above the
observed `~1.48 s` per update. The weakest evidence is the five-step steady
sample, so the ranges are deliberately descriptive rather than inferential.
