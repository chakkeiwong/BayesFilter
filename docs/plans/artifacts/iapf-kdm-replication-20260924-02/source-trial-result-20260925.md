# First complete MDPF NLL source trial and evaluation
Updated 2026-09-25T13:42:32+08:00. Both scientific replications remain incomplete.

The author-source training completed at 2026-09-25T10:52:02.189991+08:00.
Pretraining ended after 137/400 epochs and full training after 621/800; both used
the original source early-stopping rule. All 326,823 optimizer gradient checks
passed, no source gradients were skipped, and 11 best checkpoints were reloaded.
Training phase wall time was 8.589 hours.

Untouched evaluation completed on 5,000 T=150 trajectories with 25 particles.
Source descriptive mean NLL=5.9257231 and mean RMSE=3.1751099. These are one
NLL-trained checkpoint's measurements, not an MSE-refined result, comparison,
confidence interval or evidence of superiority. Source tuple order is mean,
trajectory standard deviation, and get_iqr(); source anchors:
evaluations/full_sequence_evaluations/full_sequence_evaluation_base.py:112-149,724.
Raw outputs, data hash, checkpoint hash and executed consumer counts are saved.
Evaluation took 24.831s inside the child and 33.486s supervised.

## Infrastructure recovery and accounting

The training supervisor and follow-up watcher disappeared while the independent
worker continued. At 09:21 HKT it had completed 511 epochs. Their cause of exit is
not recorded. A first recovery launch at 13:35 HKT correctly refused to attach
because training had already ended; it launched no training or evaluation.
The repaired collector verified complete matching terminal phase/epoch records,
finite losses and checkpoint hashes, then recorded RECOVERED_ARTIFACTS.
The original worker exit code is unknown and remains null; it is not recorded as
a successful process exit. The source emitted complete PASS records by 10:52 HKT,
within the original 14:07 HKT deadline. Detached recovery then ran evaluation as
attempt 17, which exited 0 and passed.

The recovery code retains the original deadline, avoids a duplicate training
launch, refuses live supervisors or incorrect process identities, preserves
old watcher evidence, and offers --detach for future long supervisors.
32 focused CPU-only tests passed, including real process-group timeout and
detached-session checks. This change is orchestration only; source-repair11
and the completed scientific program remain unchanged.

The ledger conservatively charges 41466.411s from
original launch until recovery. This includes idle time after the terminal
training artifact; it is an upper bound, not measured GPU activity.
Total GPU charge=42057.492s of 173700s; remaining=36.567h.
CPU charge=26.154s of 1800s. Attempts 17/60. No GPU job is active.
No refunded time or increased budget is assumed.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept one complete source trial's artifacts | Full source stages, complete epochs/early stop, checkpoint and consumer checks pass | No supported numerical veto; original process exit unknown | One seed; supervisor exit unrecorded | Preserve recovered artifacts and source provenance | Scientific replication or learning quality |
| Accept untouched evaluation execution | 5,000 complete T=150 trajectories; finite metrics; checkpoint hash match | No execution/numerical veto | Heuristic comparison not evaluated | Construct planned conditional heuristic checks | Method superiority |
| Keep both replication programs open | iAPF and remaining MDPF phases incomplete | Quality promotion blocked until required checks | Source/fit and statistical gaps | Resolve Appendix B, iAPF fit/floor and comparator/repeat contract | Default readiness or TensorFlow admission |

## Inference status

| Evidence class | Finding |
|---|---|
| Hard veto screen | Source training/evaluation finite and complete; supervisor loss recorded separately; heuristic check still missing |
| Statistically supported ranking | None; no comparative ranking attempted |
| Descriptive-only differences | Only one MDPF trial's loss/error and runtime; no between-method inference |
| Default readiness | Not established; this is the independent author PyTorch/NumPy reference, eager without XLA |
| Next evidence needed | Conditional heuristic baseline table, complete source comparators/repeats and uncertainty, Appendix B, iAPF repair, eligible TF/TFP actual-consumer parity |

Post-run red team: a complete training run and finite held-out metrics can still
describe a method worse than simple geometric or classical filters. The strongest
unresolved explanation is inadequate learned filtering despite correct execution.
Required conditional heuristic comparisons can overturn any favorable quality
interpretation. No such interpretation is currently made. Engineering completion,
numerical validity and scientific interpretation remain distinct.

Exact commands, source/environment snapshots, seeds, data/checkpoint hashes and
wall accounting are in ledger.json, manifest-repair11/12.json and stage0/attempt16
and17. The machine-readable companion is source-trial-result-20260925.json.
