# KDM batch-64 replay — 7 October 2026

The one-update development pilot passes all 896 fixed-source replay checks,
covering MDPF and A-MDPF in float32 and float64. All source losses, gradients,
clipped gradients, parameter updates, Adam states, frozen parameters, placement
and single-trace checks pass the existing bounds. The default batch-2 export is
numerically unchanged. Source, data and checkpoint hashes agree with the
previous fixture. Both replications remain incomplete.

Evidence: [summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-batch64-20261006-summary.json)
and the comparison/manifest under its linked tf-replay directory. The source
checkpoint is a fitted MDPF checkpoint; the A-MDPF case is the source-converted
MDPF warm start, not an independently trained A-MDPF result.

## Execution and interpretation

Attempt 01 failed on an unused torchviz import in the author dependency graph
before numerical work. A local import shim leaves the source files unchanged;
absence of torchviz call sites in the executed source was checked. Attempt 02
first hit the driver's fresh-directory guard; execution then used its fresh
tf-replay child. GPU/XLA completed with memory growth, TF32 off and one trace
per case. Four first-update times are approximately 11.25, 10.98, 11.80 and
12.21 seconds; they include compilation. Peak TensorFlow allocation was
274,041,856 bytes. No steady-state or full-training estimate is established.

The source exports report 3.833 seconds of source-body time in total; complete
process wall timing and logs were not preserved by the direct launcher.
Charge a conservative 120 CPU seconds, and the measured 55.778 GPU seconds.
The companion summary corrects the generic driver's stale plan, seed and
data-window fields. Future stage runs must capture complete logs and process
timing from their launch wrapper.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Batch-64 fixed replay passes | 896/896 | None in this fixture | Development batch and warm start | Audit and implement stage/controller plus native sampling call chain | Full training equivalence |
| Cost estimate incomplete | First update measured | No runtime failure | No steady-state/native full epoch | Measure after controller wiring | Full paper scope fits the budget |

| Inference status | Finding |
|---|---|
| Hard veto screen | None in this fixed replay; prior float32 diagnostic failures remain open |
| Statistically supported ranking | None |
| Descriptive-only differences | First-update timing and allocation |
| Default-readiness | Not assessed |
| Next evidence needed | Native training, source stages, source-only calibration, untouched paired qualification |

Post-run red-team: this fixture shares source draws and uses a historical
checkpoint. It can pass despite a wrong native sampling or stage controller.
The sampler has independent diagnostic evidence but is not yet wired into this
consumer. Existing float32 discrepancies are not superseded by success on a
different batch. Heuristic superiority is outside replication acceptance;
source failures or weaknesses remain accepted comparator outcomes.
