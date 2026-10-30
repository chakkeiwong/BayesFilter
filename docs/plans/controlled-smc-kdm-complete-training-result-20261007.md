# Complete KDM development training — execution note, 7 October 2026

Status: both complete development training runs and their independent terminal
audits PASS. The source audit passed 77,938 checks; the TensorFlow audit passed
192,156. All workers have exited successfully. Neither replication is accepted.

One material outcome needs downstream investigation: the TF best validation
losses are descriptively higher than the source losses by 0.3274 nat (MDPF)
and 0.3301 nat (A-MDPF). These selected validation losses come from one
independently randomized training pair; they are neither untouched evaluation
nor evidence of a statistically supported ranking. Training-mechanics success
does not establish the accepted 0.01-nat NLL criterion.

The CPU smoke completed in 115.72 seconds with all three stages, complete coverage,
zero skipped gradients, frozen-parameter invariance and exact fresh constructor/data
hashes. The GPU smoke completed in 185.83 seconds and passed 87/87 checks.
Every stage performed 158 updates, every batch shape traced once, and all 498 batch
keys were distinct. The second-epoch costs were 1.35, 13.00 and 12.60 seconds for
weights, MDPF and A-MDPF. They are descriptive cost evidence only.

Source and TF pretraining selected epoch 0 over the worse epoch 1 validation loss.
The TF MDPF starting checkpoint therefore exercises best-checkpoint transfer rather
than final-checkpoint transfer. Later stages consumed their own preceding weights.

Code review checked the actual source trainer ordering at trainer.py:613-700:
training, validation, best-loss comparison, plateau scheduling, early-stop update,
then checkpoint save unless stopped. TF uses the already verified controller and
follows that order. The epoch driver preserves source caps, folds epoch into native
and shuffle keys, checks frozen parameters and coverage each epoch, writes latest
state, and transfers the saved best state. The supervisor bounds total phase time
and kills the process group after a 30-minute heartbeat stall. Its full-run guard
requires a passing same-code, same-reference two-epoch smoke.

Review verdict: proceed to the one complete development pair. Existing float32
compatibility failures remain promotion vetoes. Full training is expected to reveal
cost and failure modes; it cannot establish no-worse quality from one trained unit.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Launch complete development pair | Two-epoch mechanics pass | No new failures; earlier float32 discrepancies open | Full stopping time and quality | Source 400/800/800 caps with native early stopping | Replication acceptance |

| Inference status | Finding |
|---|---|
| Hard veto screen | Two-epoch mechanics passed; old numerical vetoes preserved |
| Statistically supported ranking | None |
| Descriptive differences | Losses and runtime |
| Default readiness | Not established |
| Next evidence | Complete training, numerical compatibility, independent trained units and untouched qualification |

The manifests under kdm-complete-training-20261007-01/02 preserve the exact smoke
commands, source/runtime hashes, devices, seeds and logs. Full runs use 03/04.
The live dispatcher accounts uncharged phase runs while they execute; terminal
ledger entries replace that provisional charge exactly once. Reserve at most
64800 CPU and 57600 GPU seconds including all attempts; no new compute grant.

Initial full-run inspection: source pretraining completed 107 epochs and TF
pretraining 130, both ending by the source early-stopping rule. Each then loaded
its own best weights into MDPF. Different stopping times with independent native
random streams are descriptive, not a parity failure or performance ranking.
At the inspection, CPU MDPF had reached about epoch 100 and TF about epoch 230.
TF trace counts remained one per shape, finite-update checks had not fired, and
peak live allocator use was about 147 MB. Exact current progress is in each run's
progress.json and the dispatcher checkpoint; these prose epoch counts are a snapshot.

The status reader initially failed on old reference-ledger entries without a path
field. Restricting deduplication to entries with a path repaired reporting without
changing a numerical worker. Live wall time and completed but unledgered runs now
reduce the displayed budget; a later terminal charge will be deduplicated by path.
The supervisor remains responsible for timeout, stalled progress, exit status and
process cleanup. Numerical failures require inspecting the preserved result before
a repair or retry. No unattended scientific acceptance is performed.

## TensorFlow terminal review (before author completion)

Run 04 completed all three stages in 11,892.01 supervisor seconds (3.303 hours).
Pretraining stopped after 130 epochs, MDPF after 353 and A-MDPF after 253,
each by the actual source early-stopping rule before its original cap. Best
checkpoint epochs were 90, 307 and 227 (zero based). The run passed 10,307
recorded checks. GPU/XLA ran with TF32 off and verified memory growth; the
maximum recorded TensorFlow live allocator peak was 146,656,768 bytes.
Every training/validation batch shape traced once.

A separate CPU-only audit extracted the pinned author's scheduler and
EarlyStopping classes, replayed every saved validation loss, reconstructed all
training/validation losses and row coverage, and inspected actual checkpoint
contents. It passed 192,156 checks with no failures: best-stage transfers,
frozen parameters, fresh Adam state per stage, finite updates, distinct native
keys, early stopping and final manifests all agree. This checks the mechanics
of the realized TF trajectory; it does not assert the same stochastic trajectory
as Torch or qualify the learned models.

The audit needed two harness-only repairs: selecting the conda C++ runtime
already used by the source worker, and the same is_better/_is_better API
adaptation already checked in that worker. Neither changed training code,
controller mathematics, saved losses or model weights. Failed audit attempts
01/02 are preserved; attempt 03 passed. Audit and routine work is conservatively
charged 210 CPU seconds, separately from training.

Evidence: artifacts/controlled-smc-kdm-replication-20260928/
kdm-complete-training-tf-reviewed-20261007.json and
kdm-complete-training-tf-audit-20261007-03.json. The summary binds the audit,
terminal manifests, costs, controller source and helper hashes. The combined terminal review below supersedes this intermediate status.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept complete TF development-training mechanics | All stages and independent controller/checkpoint audit pass | No new training failure; existing float32 comparisons remain failed | Author terminal outcome and trained-model accuracy | Continue author training; resolve numerical compatibility and specify independent-unit qualification | Either replication accepted |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new TF finite/update/coverage/trace failure; prior numerical vetoes preserved |
| Statistically supported ranking | None: one development training unit |
| Descriptive differences | Stopping times, validation losses and runtime only |
| Default readiness | Not established |
| Next evidence needed | Author completion, independent numerical bounds/consumer checks, source-only power calibration and untouched trained-unit comparison |

Post-run skeptical review: a perfectly reproduced training schedule can still
train a numerically different or inferior filter. Existing float32 mismatches
and the absence of independent trained units are the strongest alternative
explanations. The present audit only closes training-mechanics work. A failed
replay of source controllers or checkpoint contents would overturn that narrow
conclusion; no such failure was found.

### Author terminal audit: localized harness correction

The source run completed all stages (107/454/230 epochs), return code zero.
Audit 01 passed all controller, coverage, update and checkpoint checks except
six A-MDPF frozen initializer tensors. Inspection of the actual stage transfer
shows an audit-baseline error: A-MDPF freezes the MDPF best checkpoint's
initializer, not its own fresh constructor. The source worker already checks
against the post-transfer values. The independent auditor was corrected to compare
with the recorded MDPF best checkpoint. Audit 01 is preserved; audit 02 passed
all 77,938 checks. No training code or result is changed. This is a localized audit repair
within the authorized complete-training phase, not acceptance of changed
frozen parameters.


## Combined terminal review

The author run 03 completed in 23,864.6748 CPU seconds (6.629 hours);
the TF run 04 completed in 11,892.0081 GPU seconds (3.303 hours).
These are descriptive costs on different hardware. Both runs completed the
source protocol with its original caps, objectives, stage dependencies and
early-stopping controllers. No training retry was required.

| Stage | Source epochs / best epoch | TF epochs / best epoch | Source best validation loss | TF best validation loss |
|---|---|---|---|---|
| Weight pretraining | 107 / 70 | 130 / 90 | 4.69869863 | 4.70566443 |
| MDPF | 454 / 408 | 353 / 307 | 6.12386863 | 6.45127364 |
| A-MDPF | 230 / 135 | 253 / 227 | 5.74585259 | 6.07597390 |

Best epochs are zero based. Each stage stopped by SOURCE_EARLY_STOP.
The source audit independently reconstructed every epoch's weighted losses,
coverage, scheduler/stop decisions, checkpoint transfers, frozen parameters and
Adam steps/moments from the pinned source controllers and saved tensors. The
77,938 passing checks concern the realized source trajectory. The 192,156
passing TF audit checks concern its realized trajectory. They do not require
identical stochastic paths or numbers of epochs.

The combined machine-readable review is
[the preserved summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-complete-training-reviewed-20261007.json).
It binds both terminal manifests, results, audits and their checksums.
The terminal CPU ledger charges source run 03 exactly once, plus a conservative
120 CPU seconds for source audits and routine assembly. Source/TF smokes and TF
full training were already charged and are not charged again. The status reader
now accepts both nested and older flat diagnostic decisions and recognizes
completed training only when the reviewed run manifests/audits still match.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete development-training mechanics accepted | Both full protocols and independent audits pass | No new training/coverage/controller/checkpoint failure; float32 numerical comparisons remain failed | Higher TF validation losses on one selected development pair and unevaluated downstream accuracy | Evaluate actual best checkpoints using source metrics; justify numerical bounds; source-only precision calibration before untouched trained-unit qualification | Numerical no-worse performance or either replication accepted |

| Inference status | Finding |
|---|---|
| Hard veto screen | Training mechanics pass; prior float32 and controlled-SMC fitting vetoes remain |
| Statistically supported ranking | None; one development trained pair with independent native randomness |
| Descriptive-only differences | Validation losses, stopping epochs and CPU/GPU runtimes |
| Default-readiness | Not established |
| Next evidence needed | Checked source evaluation consumer, independently justified numerical checks, source-only power/cost calibration and untouched independent trained units |

Post-run red-team finding: independently correct controller mechanics could
coexist with systematically worse learned filters. The observed validation gap
makes downstream evaluation a concrete next diagnostic, not an optional
improvement. Using the source evaluator on both converted checkpoints can
separate model-quality differences from TF evaluation differences. The present
work neither rejects the replication direction nor accepts the no-worse claim.
