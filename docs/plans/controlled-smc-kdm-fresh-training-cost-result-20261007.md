# Fresh source/TF training cost result — 7 October 2026

The actual author CPU trainer completed one epoch of weights, MDPF and A-MDPF in
76.30 seconds total. The matching GPU/XLA run completed in 205.24 seconds including
compilation and passed all 36 checks. Each stage used all 5000 training and 1000
validation rows, including the 8/232 partial batches. Every TF batch shape traced
once. Source skipped-gradient counts were zero; frozen parameters were unchanged.

Both runs start from fresh source constructor values and transfer their own
preceding stage's best checkpoint. Source train/validation seconds were 1.53/0.15,
23.42/3.20 and 27.90/2.95 for the three stages. TF regular training batch means
were 0.0124, 0.3333 and 0.3192 seconds. TF peak allocator use was 144497920 bytes.
The first epoch includes compilation: total elapsed time is not steady-state cost.
At all source caps the measured work extrapolates to about 13 CPU and 12 GPU hours,
with uncertain overhead and early-stopping reductions. This supports one complete
development pair within an 18 CPU/16 GPU hour reservation; it does not establish
how many independent trained units qualification requires.

A report-only defect was found: each TF epoch's initial_checkpoint_role repeated
historical-fixture wording. The top-level initialization, constructor hashes and
executed constructor/transfer call chain establish fresh initialization. Preserve
the original result and record this correction in the phase summary. The helper
now takes the actual initialization description. No numerical result changed.

| Decision | Primary screen | Veto status | Main uncertainty | Next action | Conclusion limit |
|---|---|---|---|---|---|
| Proceed to complete development training | Author three stages and TF36/36 pass | No new failure; old float32 failures remain | Later epochs and stopping time unmeasured | Two-epoch harness check, then full source caps | Neither replication is accepted |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new nonfinite or coverage failure; earlier float32 vetoes preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Losses, runtime and extrapolated cost |
| Default readiness | Not established |
| Next evidence | Full trained units, numerical compatibility, source-only calibration and untouched qualification |

Strongest alternative explanation: first-epoch timing and healthy trajectories may
not represent later training. The full schedule can overturn the feasibility
estimate or expose additional numerical failures. No quality claim is made here.
Commands, environments, data/constructor hashes and complete logs are in the two
versioned run manifests. The summary is kdm-fresh-training-cost-20261007-summary.json
under the campaign artifact root. Charge 76.29585614893585 CPU seconds plus a
60-second routine-check allowance and 205.23906311788596 GPU seconds exactly once.
