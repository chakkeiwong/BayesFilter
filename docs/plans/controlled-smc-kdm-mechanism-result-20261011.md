# Replication mechanism investigation and execution readiness

11 October 2026. This continues the replication-only campaign. The author
methods and KDM acceptance criteria are unchanged. Controlled-SMC's withdrawn
1% likelihood / 5% variance rule remains wrong planning, not a current gate.

## Checked causes and implications

The confirmed KDM translation defect was Adam's scalar arithmetic. The author
computes beta powers and the scalar step rate in Python binary64 before casting
to parameter precision. The port previously computed them in binary32. The
repair in `bayesfilter/replication/kdm_neural_replay_tf.py:261-303` reproduces
those boundaries; v2 checkpoints reject incompatible old optimizer state.
Coupled four-update failures fell from 49 to 8 of 3,616. All 1,808 binary64
checks pass. The remaining eight checks are four binary32 proposal gradients
and their clipped copies; losses, parameters, moments, particles and gradient
norms pass. Exact gradient identity is diagnostic, not a replication criterion.

The subsequent common-input investigation passed 3,840 local forward/adjoint
checks. Flattening dense layers did not resolve the residual differences, so
that candidate was not adopted. Small-radius angle normalization is sensitive,
but it has not been established as the cause of all residual network errors.
No further numerical change is justified by this diagnostic. The previous completed TF training unit predates the Adam repair. The newly repaired complete run is reported below.

Controlled-SMC's author fit uses normal equations and a conditional Octave
solver. Ill-conditioned designs amplify operation-order differences relative
to the port's centered QR fit. All 101 saved Gram/RHS source replays are exact.
Q2 retained 13,120 paired trials in 41 cells, 14 intrinsic author failures,
five TF failures on author-failed inputs, and zero additional failures on
author-valid inputs. Those intrinsic failures are acceptable baseline behavior.
The source-only feasibility analysis found no independently justified,
resolvable continuous-outcome degradation bound. The source variance spans
approximately 4.18e-5 to 4057, with effective counts as low as 1.45. These facts
explain why the withdrawn percentage rule was inappropriate. They support the
bounded implementation/failure-fidelity finding, not a population equivalence
claim or a need to repair the author's heavy tails.

Source anchors: Heng et al., published paper Section 6.2 and Appendix,
`.localresources/papers/heng2020-controlled-smc-published.txt:1188-1227`;
`.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz/cSMC_Resample.m:185-193,239-264`.
KDM: Younis/Sudderth IWSG equations 14-15,
`.localresources/papers/younis-sudderth-2023-long-range-tracking.txt:447-480`,
and the pinned author particle transform, lines 144-167, source commit
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`.

## Executed checks and harness repairs

All paths below are relative to
`artifacts/controlled-smc-kdm-replication-20260928/`.

| Execution | Result | CPU / GPU wall seconds |
|---|---|---|
| `mechanism-20261011/` attempts 01-05 | Terminal common-input check passes; rejected dense-layout intervention; preserved assertion, library-path and integer-key JSON failures, then repaired diagnostic harness | 19.322078 / 130.110410 |
| `native-training-20261011/native-smoke-01` | Both variants complete 64 native updates in each backend; finite committed updates, frozen parameters unchanged, one trace per graph | 37.833183 / 164.255309 |
| `stage-refresh-20261011/stage-refresh-01/verification` | Current runtime passes 2,204/2,204 source-stage checks | Included below |
| `stage-refresh-20261011/stage-refresh-01/smoke` | Two full-data epochs in all three stages; 87/87 checks pass, including all partial batches and stage transfers | Combined stage refresh 0 / 899.991196 |
| `repaired-training-20261011/training-pilot-01` | COMPLETED: capped two-epoch cost-readiness test with restored TensorFlow thread limits | decision.json: READY_FOR_FULL; 87/87 checks pass; 160.460316 GPU seconds |
| `repaired-training-20261011/training-pilot-02` | COMPLETED: full repaired training, independent terminal audit and common-author evaluation pass | 56.442887 / 8,845.828356 |

The fresh-training driver now requires an explicit current verification file;
the complete-training launcher passes it through. This removes a stale-proof
dependency. The repeated numerical path uses stable TensorFlow graphs and
native tensor loops. Python orchestrates batches/stages and statically traces
network layers. Runtime NumPy and a per-update retracing explosion were not
found. The recorded graphs trace once for each fixed full/partial batch shape.
An inter-instance TensorFlow warning is not evidence of repeated retracing
within a training graph. GPU/XLA and verified memory growth passed.

The stage-refresh smoke was much slower than the old full run: second epochs
were about 194 seconds for MDPF and 98 seconds for A-MDPF, versus historical
medians around 15 and 19 seconds. Its supervisor had omitted both TensorFlow
thread limits used by the old run. Oversubscription remains a hypothesis;
contention and other runtime differences remain alternatives. The reviewed
thread-limited repeat completed in 160.460316 GPU seconds: all 87/87 checks
passed, the projected full run is 8,123.672899 GPU seconds, and the reviewed
80% threshold is 22,911.631747 seconds. This clears cost readiness only. The
full pilot has an 8 GPU-hour / 30 CPU-minute allocation from the existing
campaign, at most two versioned attempts. Source training is reused only after
exact input and constructor hashes match; the author method and stopping law
are unchanged.

## Pre-pilot decision and inference status (historical)

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept Adam translation repair | Source scalar arithmetic and coupled controls pass | No new invalid outcome in the checked fixtures | Effect on complete trained models | Monitor the repaired complete-training pilot | KDM replication complete |
| Do not adopt dense-layout change | Residual failures not reduced | No need for a runtime intervention | Full-network binary32 accumulation | Preserve diagnostics; assess downstream consumer | Exact gradients or sole error cause |
| Retain cSMC bounded fidelity evidence | Source operations/controls checked; no added failures observed | Intrinsic author failures accepted | Generalization and continuous-outcome precision | Report tested scope; no retired-margin rerun | Population stochastic noninferiority |
| Repair readiness environment | Same source protocol passes smoke | Measured cost screen passed; full run active | Threads versus workload contention | Full pilot under the measured cost cap | Performance improvement or numerical acceptance |

| Inference status | Finding |
|---|---|
| Hard veto screen | Validity and mechanics pass in the completed bounded checks. Earlier harness failures were repaired and preserved. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Old single-pair checkpoint gaps, native-smoke losses and runtime comparisons. |
| Default-readiness | Outside the replication-only task. |
| Next evidence needed | Repaired complete trained checkpoints, common-author evaluation, native evaluation parity, then a separately frozen multi-unit comparison if feasible. |

Post-run skeptical review: source-stage and local derivative checks cannot
certify long-training quality. The strongest alternative explanation for the
old quality gap is ordinary independent-training variability, possibly
compounded by optimizer error; the current evidence does not separate them.
A repaired complete run that retains the gap would refute the claim that Adam
precision alone explains it. One favorable pair would still not establish
population no-worse behavior. Full execution and numerical qualification have
separate statuses.

The exact commands, code/data hashes, device policy, seeds, wall times and
failed attempts are preserved in each manifest. Before the readiness launch,
the remaining campaign balance was 220,803.018539 CPU seconds and
116,595.560631 GPU-tranche seconds. Readiness and later pilot work are charged
against those balances, not a new grant. The historical numbered-attempt
counter remains 160.


## Completed repair and endpoint execution

The repaired training pilot completed all three source-controlled stages in training-pilot-02. Best epochs were weights 90, MDPF 490 and A-MDPF 173. The independent terminal audit and common-author evaluation passed. Terminal charge: 8,845.828356 GPU seconds and 56.442887 CPU seconds. All exact commands, seeds, code/data hashes, growth/TF32/XLA settings and terminal checkpoints are in its manifest and diagnostic result.

Four descriptive common-author comparisons, TF-trained minus source-trained:

| Variant / filter seed | NLL difference (nat) | RMSE ratio |
|---|---:|---:|
| MDPF / 202610082 | -0.262569 | 0.960171 |
| MDPF / 202610083 | -0.212852 | 0.985658 |
| A-MDPF / 202610082 | -0.147241 | 0.959064 |
| A-MDPF / 202610083 | -0.061796 | 1.002056 |

The A-MDPF angle-seam slice at seed 202610083 has NLL difference +0.088531 nat. This remains a descriptive limitation; it was not a separately predeclared admission criterion. One training pair does not distinguish a repair effect from training variability and cannot establish population noninferiority.

The separate native-evaluation-20261011/native-evaluation-01 check completed the native GPU/XLA TF filter for four checkpoints and two seeds on all 256 evaluation trajectories and 150 time steps at 25 particles. All 602 validity, coverage, stable-trace and checkpoint-immutability checks passed. The pinned author metric classes replayed all eight native-cloud records successfully. Maximum scaled pointwise discrepancy was 1.7138383e-6, maximum absolute NLL discrepancy 9.536743e-7, and maximum absolute squared-error discrepancy 2.670288e-5. The run charged 55.838925 GPU seconds and 7.149969 CPU seconds.

A bookkeeping defect omitted next_action from two native-evaluation ledger entries, causing the status reporter to raise KeyError. Added the missing field to those two entries and the supervisor template. Scientific results and resource charges are unchanged; completed source snapshots remain preserved. Status now reads both completed results.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain repaired complete training | All source stages/controller, finite updates and terminal audit pass | No failed terminal mechanics check | Independent training-unit variation | Audit unit generation and freeze affordable qualification design | KDM population noninferiority |
| Accept native endpoint/metric check | Native call chain and pinned author metric replay pass | No invalid filter result or state mutation | Coverage beyond the four checked checkpoints | Retain in future unit validation | Paper-wide or production readiness |
| Preserve cSMC tested-input verdict | No added invalid outcomes in Q2 author-valid inputs | Intrinsic author failures accepted | Unspecified/resolution-limited continuous-outcome bound | Report bounded scope; do not restore retired margins | Population no-worse |

| Inference status | Finding after completed runs |
|---|---|
| Hard veto screen | Completed development training and native evaluation pass their stated validity/mechanics criteria. |
| Statistically supported ranking | None: one independent trained pair. |
| Descriptive-only differences | Four overall paired metrics, conditional slices and native filter-seed metrics. |
| Default-readiness | Not evaluated; outside replication-only scope. |
| Next evidence needed | Independently trained source/TF units, with data/constructor seed audit and a frozen uncertainty/budget design. |

Post-run red-team: common-author evaluation isolates training checkpoints, while native metric replay separately checks the TF consumer. Agreement here does not supply independent training replicates. The strongest alternative explanation for favorable repaired results is ordinary seed variability. Repeated failures on fresh author-valid units or worse downstream uncertainty bounds would overturn the corresponding positive bounded inference.

Remaining ledger balances after these completed runs: CPU 220,739.425683 seconds; GPU tranche 107,533.433034 seconds. No active resource reservation remains. Source training in the previous development unit cost 23,864.674819 CPU seconds; it is a cost anchor, not a population variance estimate.

Evidence roots (relative to artifacts/controlled-smc-kdm-replication-20260928/):
- repaired-training-20261011/training-pilot-02/: manifest.json, diagnostic/result.json, terminal-audit.json, evaluation/diagnostic/result.json, decision.json.
- native-evaluation-20261011/native-evaluation-01/: manifest.json, diagnostic/result.json, metric-audit.json, decision.json.
