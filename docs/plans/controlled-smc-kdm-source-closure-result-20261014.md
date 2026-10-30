# KDM released-source replication closure, 14 October 2026

Decision: ACCEPTED_BOUNDED_RELEASED_SOURCE_REPLICATION for the current
25-particle bearings MDPF/A-MDPF implementation, on the tested source operations
and consumers. Controlled-SMC bounded replication remains accepted. The
source-replication closure phase is complete; no further training is scheduled.

The corrected CPU-only audit passed 283 checks across seven operation groups.
These are evidence, provenance and wiring checks over existing numerical
experiments, not 283 new numerical trials. All eight numerical runtime files
match the relied-on current verification. The audit introduced no algorithm,
optimizer, tolerance or numerical-policy changes.

[Master](controlled-smc-kdm-replication-master-program-20260928.md) ·
[Plan](controlled-smc-kdm-source-closure-plan-20261014.md) ·
[Coverage and all checked dependencies](artifacts/controlled-smc-kdm-replication-20260928/source-closure-20261014/attempt03/coverage.json) ·
[Manifest](artifacts/controlled-smc-kdm-replication-20260928/source-closure-20261014/attempt03/manifest.json)

## Target, comparator and evidence

The target is the released author's computation, including its intrinsic
numerical sensitivities. The comparator is Younis/Sudderth author commit
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, with the locally stored NeurIPS 2023
paper. Paper equations 13–15 anchor IWSG; Appendix D.2–D.3 anchors the bearings
pretraining and full training. Current source files are checked against the
pinned Git blobs, and saved inputs/results against their recorded hashes.

The implementation computes the same specified filtering/training operations
in TensorFlow. Tested common-input values, gradients and optimizer updates
agree within their previously declared quantity-specific bounds. Native random
draws reproduce the declared sampling mechanism; identical Torch/TF seed labels
are not claimed to produce identical draws. Numerical equivalence for every
possible input is not established.

| Operation coverage | Executable evidence retained |
|---|---|
| Data/masks, initial particles, encoders, proposal/weight/bandwidth heads, IWSG, loss, TBPTT, Adam, stage transfer | 2,204 passing stage assertions, both precisions and variants, source gradients/updates/moments; pinned data and configs |
| Native categorical, normal and circular draws | 28 sampler assertions; common-random-input arithmetic, validity and diagnostic law checks |
| Native composed pretraining clouds | 138 assertions, 26 moments per backend and 26 between-backend screens, one million particles per backend through actual consumers |
| Late pretraining state, partial batches, validation and controller | 271 boundary assertions over 24 cases and 863 controller replay assertions |
| Actual training/filter consumer wiring | Executable AST checks pair the public compiled endpoints with the kernels used by the saved numerical checks; explicit stable signatures and XLA settings retained |
| Recent native MDPF/A-MDPF endpoints | 1,385 assertions over 18 evaluations of nine fixed models; NLL and RMSE replayed through author code on nine saved clouds |
| Recovered unit02 MDPF endpoint | 472 assertions; NLL and RMSE replayed on three saved clouds; exact baseline replay, no retraining |
| Prior flagged cases | Restored-state pretraining, angular-input interventions and source-only ancestry perturbations retained as case-specific explanations |

Native endpoint scopes are N=25, T=150, 256 sequences, batch 64 and two filter
seeds. Metric replay uses the saved subset clouds, not every particle/time point
from every evaluation. The same-output metric audit uses its recorded
2e-4*(1+abs(source value)) bound; this is an arithmetic check, not an allowed
difference between independently trained models. Stage replays retain their
original FP64/FP32 quantity-specific bounds. No new tolerance was selected here.

Current call-chain anchors:
[neural filtering and updates](../../bayesfilter/replication/kdm_neural_replay_tf.py),
[initialization/pretraining/full training](../../bayesfilter/replication/kdm_source_stages_tf.py),
[normal/von Mises density and IWSG](../../bayesfilter/replication/kdm_normal_von_mises_tf.py),
[native sampler](../../bayesfilter/replication/kdm_sampling_tf.py),
[epoch consumer](../benchmarks/diagnose_kdm_source_epoch_tf.py), and
[native evaluation consumer](../benchmarks/diagnose_kdm_native_evaluation_tf.py).
The neural consumer reaches NormalVonMisesKDM._resampling; the mere presence of
the separate generic IWSG module was not used as proof of that call chain.

## What explained the apparent failures

The Adam scalar-precision translation error was repaired before the evidence
accepted here. Three later discrepancies have more specific explanations:

* Restoring the author's parameters and optimizer state clears the selected
  evolving-pretraining flags. At index67 the gradient difference falls from
  2.60975e-5 to 1.11759e-8. This is a common-state result, not a proof that
  independently trained states must remain close.
* A one-ULP CDF change can select another ancestor. In the saved case, matching
  that first choice reduces gradient error from 0.03084 to 2.38e-7; a source-only
  perturbation reproduces the alternative branch to 2.98e-7. This supports
  shared resampling sensitivity for that case.
* Matching angular inputs clears the selected angular-gradient flags while
  preserving ancestry. The source controls and reverse intervention pass.
  This localizes that sensitivity; it is not a universal perturbation allowance.

Using the author's pretrained measurement weights reduced unit01's observed
validation/T150 NLL gaps by 95.26%/92.68%, and unit02's by 80.82%/79.30%.
Unit02 still has gaps of 0.098476/0.157799 nat. These finite residuals and all
unfavorable earlier checkpoints remain evidence. We have not proved that
training randomness explains every residual, nor that a new translation defect
causes one. The coverage audit found no remaining named, untested operation
within the selected released-source scope.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept bounded released-source replication | All seven coverage groups pass with current runtime and pinned author evidence | No outstanding checked implementation, validity, hash or wiring failure | Untested states and independent-training outcome variation | Archive; reopen only for a named mismatch or newly identified missing operation | Universal equivalence or population no-worse |
| Keep original-paper reproduction separate | Exact Appendix B driver/settings and original experiment coverage not established | Missing scope evidence prevents a paper-wide completion claim | Mapping released generic toy modules to the original Appendix B experiment | Reconstruct only as a separately specified paper-reproduction task | All figures, datasets and original checkpoints reproduced |

| Inference status | Finding |
|---|---|
| Hard veto screen | The selected current source-operation checks pass; invalid audit attempts are excluded |
| Statistically supported ranking | None claimed |
| Descriptive-only differences | Independently trained NLL/RMSE gaps and fixed-unit intervention reductions |
| Default-readiness | Not evaluated by this source-replication closure |
| Next evidence needed | A named common-input discrepancy for another repair; a separately justified untouched-unit design for a population-quality claim |

Exact Appendix B temporal-prediction reproduction is NOT_ESTABLISHED.
The pinned repository contains generic toy dataset/problem/evaluation modules,
so “there is no toy code” would be wrong. The mapping to the paper's exact
experiment, settings and driver has not been reconstructed. Original iAPF,
Task3, algorithm improvements, production/HMC readiness and all-paper
reproduction are not completed by this verdict.

## Execution and terminal review

The executable command and CPU-only environment are in the manifest. No
numerical framework was imported; CUDA_VISIBLE_DEVICES=-1. The terminal audit
used 1.515 wall seconds and 1.442 process CPU seconds, zero GPU seconds.
Fourteen auditor failure-case tests passed before execution and are recorded
in coverage.json. The phase cap remains 600 CPU seconds and three audit
attempts. Earlier development checks were not individually timed; no new
numerical campaign ran. Conservatively reserving the entire phase cap leaves
at least 204,863.582169 CPU seconds and unchanged 71,779.382403 GPU seconds
from the allocation recorded before closure.

Attempts01/02 are INVALID for acceptance because their reporter did not enforce
the evidence contract. Their provisional acceptance was withdrawn; preserved
INVALIDATED.md notes explain why. Attempt03 computes its verdict from nonempty
assertions, source/result hashes, operation coverage and actual call chains.
No saved numerical experiment was altered.

Terminal self-review: the strongest alternative explanation is an unexercised
interaction that affects training while the checked local operations pass.
A reproducible common-input mismatch, extra invalid port outcome, changed
numerical source or missing consumer would reopen this verdict. The weakest
evidence is population trained-quality equivalence, which is not claimed.
Moment/frequency screens are diagnostics, not exhaustive distribution proofs.

Three older harness differences are explicitly hash-bound and reviewed in
coverage.json: an added cloud-law observation branch, stricter checkpoint-scope
validation, and optional unit02 intervention export. Unchanged numerical
runtime evidence is not invalidated merely by these surrounding changes.
The old sampler's default arithmetic matches the current implementation by
AST comparison after removing only its optional pretraining concentration
argument. Current native-cloud evidence covers the added concentration use.
This is a local terminal review, not an independent external review.
