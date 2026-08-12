# SSL-LSTM q=20 NeuTra continuation interim result (2026-08-06)

Status: `RUNNING_AFTER_UPDATE_500`

## Interim verdict

The repaired protocol fixes the specific premature-stopping failure mechanically and gives early descriptive evidence that continued optimization is useful. Both independent seeds completed 500 additional optimizer updates without a validation-controlled stop. Seed A's same fixed 500-row monitor decreased from `41.2811667814` after continuation update 2 to `41.2130072927` at update 500.

This is not a terminal training, transport, convergence, or HMC result. The campaign remains active through its fixed 4,000-update budget. Final checkpoint selection will compare checkpoint 0 and all eight continuation checkpoints on one disjoint 500-row selection bank, followed by one untouched 500-row audit.

## Current run

| Field | Value |
|---|---|
| Service | `bayesfilter-q20-neutra-continuation-20260806-r1.service` |
| Launch time | `2026-08-06 02:22:58 +08:00` |
| Campaign root | `docs/plans/artifacts/ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/r1/` |
| Cap | `43,200 s` |
| Seed A | CPUs 0--24, physical GPU 1, restored optimizer step 1500 |
| Seed B | CPUs 25--49, physical GPU 1, restored optimizer step 2250 |
| Memory policy | TensorFlow memory growth verified before logical-device initialization in both parents |
| Target/update topology | 25 CPU/XLA workers x 4 rows; 100-row batch; GPU/XLA FP64 transport and Adam update |

## Update-500 diagnostics

| Diagnostic | Seed A | Seed B | Evidence role |
|---|---:|---:|---|
| Continuation update | 500 | 500 | Fixed-budget progress criterion |
| Optimizer step | 2000 | 2750 | State provenance |
| Elapsed seconds | 4720.36 | 5177.40 | Descriptive runtime |
| 500-row monitor mean loss | 41.2130073 | 40.8448042 | Descriptive only |
| Monitor loss SE | 0.0694001 | 0.0711490 | Descriptive Monte Carlo precision |
| Last training-batch loss | 41.2989139 | 40.9265508 | Explanatory only |
| Last gradient norm | 19.1574848 | 12.9294326 | Explanatory only |
| Clipping at update 500 | Yes | No | Explanatory only |
| Scale saturation fraction | 0.0830 | 0.08417 | Explanatory/repair trigger only |
| Combined unique-process RSS | 17.596 GiB | 17.606 GiB | Hard resource gate passed |
| Checkpoint SHA-256 | `8db1883f0a7c29fc93e5d26169458974457aadf379b11d226523603bfa1b0232` | `4bb348c84fe0009314382600d8a14c178233a452be813336f7479d79b318f213` | Artifact integrity |

Seed A's prior two-update canary used the same 500-row monitoring bank. The paired update-500 minus update-2 loss difference is `-0.0681595`, with descriptive paired SE `0.0129400`; 334 of 500 rows have lower loss. This supports continued optimization and contradicts the old controller's implication that training should stop at the earlier state. It does not prove convergence or rank the transport statistically.

Seed B has no earlier evaluation on this new monitoring bank. Its update-500 result is finite and operationally healthy, but cross-bank comparison with the historical 64-row validation would be invalid. Seed B improvement will be assessed only by the planned common 500-row checkpoint-selection bank after training.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue through 4,000 updates | Both seeds reached update 500; seed A same-bank monitor improved | No numerical, device, worker, resource, or artifact veto | Later schedule phases and downstream HMC geometry | Continue unchanged; select checkpoints only after training | Training convergence, NeuTra validity, HMC readiness, posterior correctness |

## Inference-status table

| Evidence class | Interim status |
|---|---|
| Hard veto screen | Clear through update 500 for both seeds |
| Statistically supported ranking | None |
| Descriptive-only differences | All loss, gradient, saturation, runtime, and cross-seed values |
| Default-readiness | Not evaluated |
| Next evidence needed | Full 4,000-update trace, disjoint checkpoint selection, untouched audit, support probe, then fresh fixed-HMC retuning |

## Research-direction classification

No result so far invalidates the harness, target, data, mathematics, or NeuTra family. The old 64-row early-stop controller was the invalid decision mechanism. The current result supports the planned repair: continue fixed-budget optimization and defer checkpoint choice to the post-training 500-row selection bank.

## Post-run red team

The strongest alternative explanation for seed A is that the observed monitor decrease is local and will reverse later. The full checkpoint ladder and disjoint selection bank address that. The weakest evidence is seed B improvement because no same-bank initial monitor was taken. A later selected checkpoint that fails the untouched audit or support probe would overturn candidate nomination but would not retroactively make the validation-controlled early stop correct.
