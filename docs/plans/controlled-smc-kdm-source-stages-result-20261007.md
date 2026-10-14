# KDM source stages and complete-data epoch result — 7 October 2026

Source-stage verification passed 2204/2204 checks in float32 and float64. A GPU/XLA
pilot then completed one epoch each of weight pretraining, MDPF and A-MDPF,
passing all 36 epoch checks. Both full replications remain incomplete.

The source is the pinned Younis release b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
The plan is [source-stage mechanics](controlled-smc-kdm-source-stages-20261007.md).
The machine-readable [summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-source-stages-20261007-summary.json)
links to the exact verification and epoch results. Every attempt retains its
command, source/runtime hashes, environment, seeds, wall time and logs.

## What is checked

The actual source initial-cloud operation, pretraining cloud construction,
selected-time ordering, loss, coupled Adam decay, clipping, gradients and optimizer
states agree on the replay fixtures within the unchanged dtype bounds.
Native sampling feeds the same numerical consumers. Both source partial-module
loads and MDPF-to-A-MDPF checkpoint conversion agree in both precisions.
Atomic restore rejects incomplete state. Stage transitions reset Adam.

Each pilot stage processed all 5000 training and 1000 validation sequences once.
Training retained its final batch of8, validation its final batch of232.
Pretraining uses five labelled times per sequence. Epoch means use the effective
batch weights. Validation left all parameters and optimizer state unchanged.
Each stage made79 optimizer updates; all four shape specializations traced once
and shared the same variables. The stages loaded their predecessor checkpoints.

| Stage | First complete epoch, seconds | Repeated train-batch median, seconds | Repeated validation-batch median, seconds |
|---|---:|---:|---:|
| weights | 20.001 | 0.0110 | 0.0186 |
| mdpf | 81.278 | 0.2746 | 0.1717 |
| am_dpf | 103.813 | 0.2591 | 0.1796 |

The first-epoch times include compilation and checkpoint/report work. Repeated
call times are descriptive and affected by GPU sharing. They do not predict
training stopping times or rank methods. The initial weights came from a fitted
development model, so the resulting losses are not evidence of learning from
scratch. This is not a completed 400/800-epoch source training procedure.

TensorFlow2.20.0-dev0+selfbuilt used the RTX5080, TF32off, stable signatures and
XLA. Memory growth was verified before device initialization. Epoch allocator
peak was144743168bytes; verification peak282652928bytes. Host epoch/batch loops
orchestrate compiled numerical functions. CPU TensorFlow handles row ordering
because StatelessShuffle has no GPU kernel. No NumPy runtime or pfor was added.

## Repairs and preserved failures

Attempt01 exported source fixtures. Attempt02 stopped on a diagnostic conversion
of an unused None bandwidth. Attempt03 stopped on an int32/int64 diagnostic seed
signature mismatch after784 passing checks. Directly replaying the captured
initial cloud repaired that adapter without changing runtime draws. Attempt04
exported source stage transfers; attempt05 passed2204 checks. Attempt06 stopped
before training because CPU-only StatelessShuffle inherited GPU placement.
Explicit CPU row ordering repaired the harness; attempt07 completed the pilot.
The initial six-attempt allocation was explicitly extended by two local repair
attempts under the original CPU1800s/GPU3600s total phase caps.

These were harness failures, not evidence that CUDA is broken or that the source
algorithm fails. The earlier float32 angle and circular-gradient failures remain
failed on their original fixtures. Passing the new cases cannot erase them.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept stage mechanics for these fixtures | 2204 replay checks and36 epoch checks pass | No remaining stage/epoch veto | Development fixtures, independent RNG implementation | Preserve results and continue numerical localization | Completed training |
| Continue replication | Native full-data epoch executes correctly | Older float32 discrepancies still block broader qualification | Conditioning bounds, fresh training, cSMC fit compatibility | Independently justify bounds; fresh initialization and source-only cost/power calibration | Full numerical equivalence or rejection of either method |

| Inference status | Finding |
|---|---|
| Hard veto screen | No additional nonfinite/domain/update/coverage failures in completed pilot; older failures retained |
| Statistically supported ranking | None |
| Descriptive-only differences | Losses and runtimes from one development epoch |
| Default-readiness | Not established |
| Next evidence needed | Fresh source/TF full training, independent training replications and untouched NLL/RMSE qualification |

Post-run audit: the strongest alternative explanation for apparently successful
training is the fitted initialization. A fresh-initialization failure would
overturn the implication that these mechanics are sufficient for full training,
but not the reported replay/row-coverage results. No method-quality conclusion
is made. The weakest cost evidence is a single shared-GPU epoch.

Measured attempt charges: CPU14.311843928s, GPU473.881616277s.
An additional conservative CPU60s/GPU10s allowance covers unmetered routine
checks and the standalone GPU probe; it is an upper-bound charge, not a measured
execution time. Charge each resource once in reference-diagnostics.json. No new
compute grant is created.
