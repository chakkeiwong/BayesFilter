# KDM native pretraining cloud distribution: completed

13 October 2026, 19:53:54 Hong Kong. Both actual native measurement-pretraining
consumers passed all 26 declared moment screens and all 26 cross-backend
difference screens. This closes the composed cloud-law diagnostic on the tested
inputs. It found no new translation defect.

[Reviewed plan](controlled-smc-kdm-cloud-law-plan-20261013.md);
[structured result](artifacts/controlled-smc-kdm-replication-20260928/pretraining-cloud-law-20261013/attempt01/diagnostic/result.json);
[manifest](artifacts/controlled-smc-kdm-replication-20260928/pretraining-cloud-law-20261013/attempt01/manifest.json);
[terminal review](artifacts/controlled-smc-kdm-replication-20260928/pretraining-cloud-law-20261013/attempt01/terminal-review.json).

## Checked target and execution

The compared quantity is the law of clouds consumed by native pretraining.
The source runs unchanged WeightModelTrainer.do_forward_pass and
DistributionCloudGenerator.create_cloud from pinned b0e2fd5. Instrumentation
retains already-computed centers and particles. A same-seed uninstrumented
invocation produces the identical source loss and effective batch size.

TensorFlow runs SourceWeightPretraining.native_evaluate, including actual
center and conditional samplers and downstream loss. Each backend executes four
complete passes over the same 1,000 saved unit01 validation sequences: 20,000
cloud centers and 1,000,000 particles. All passes include the final partial batch.
The measurement model is the shared fresh constructor; streams are independently
native. The source runs on CPU with GPU intentionally hidden. TF uses the
RTX 5080, GPU/XLA FP32, TF32 off and verified memory growth. Each of the two
batch signatures traced once. Peak live TF allocator use was 39,727,616 bytes.
Reporting uses diagnostic FP64 TensorFlow reductions on CPU.
No algorithm runtime module changed.

The driver passed 138 validity, coverage, immutability and instrumentation
checks. Terminal review passed 39 checks, including source/snapshot hashes,
27 pinned author files, full coverage, caps and
exactly-once charging. One pre-launch wrapper construction syntax error was
repaired before a numerical launch; it consumed no experiment attempt.

## Distribution evidence

Each uncertainty below is one standard error. Each whole cloud is an independent
sampling cluster; its 50 particles share a center and are not counted as 50
independent center draws.

| Quantity | Analytic target | Author mean +/- SE | TF mean +/- SE |
|---|---:|---:|---:|
| Center x variance | 4.000000 | 3.985000 +/- 0.039682 | 3.918316 +/- 0.039176 |
| Conditional x variance | 9.000000 | 9.000347 +/- 0.012716 | 8.988018 +/- 0.012707 |
| Marginal x variance | 13.000000 | 12.995183 +/- 0.043567 | 12.903960 +/- 0.043120 |
| Within-cloud x product | 4.000000 | 3.997177 +/- 0.041794 | 3.920454 +/- 0.041141 |
| Center angle cosine | 0.948600 | 0.949010 +/- 0.000514 | 0.949587 +/- 0.000511 |
| Marginal angle cosine | 0.899842 | 0.900521 +/- 0.000503 | 0.900829 +/- 0.000500 |

Maximum absolute standardized deviations from theory: 3.201 for the author and
3.158 for TF. Maximum cross-backend difference: 2.084 standard errors. All pass
the declared six-SE diagnostic screen. Cartesian y, cross-coordinate products,
center/conditional products, sine and pairwise circular moments are retained.

The source implies center variance 4, conditional variance 9, marginal variance
13 and distinct-particle covariance 4. Its first angular moment is I1(k)/I0(k)
at each stage and the square after composition. Both consumers match these
mathematical implications within diagnostic uncertainty. Moment agreement does
not prove equality of every aspect of the law; earlier primitive sampler and
common-cloud replay evidence answer separate questions.

## Consolidated source-consumer inventory

The current epoch driver, diagnose_kdm_source_epoch_tf.py lines 47-81 and
106-181, calls native_update/native_evaluate for pretraining, shared
SourceFullTraining.update/evaluate for MDPF/A-MDPF, and the source controller.
The selected best state feeds the next stage. kdm_stage_transfer_tf.py copies
the three measurement modules for MDPF, initializes missing A-MDPF resampling
modules from their MDPF counterparts, and resets Adam. This fresh static trace
is not a new execution proof; the executable evidence is listed separately.

| Operation | Executable evidence | Remaining limit |
|---|---|---|
| Native pretraining cloud/loss | This run, source-stage native/replay checks, saved-boundary common-cloud checks | Finite inputs and moment power |
| Checkpoint selection and Adam | 863 controller and 271 saved-boundary checks, 24 cases | Independently random histories legitimately diverge |
| Transfer, MDPF/A-MDPF training | Three repaired full pairs and terminal audits; stage-isolation replay | Different final learned models; independent paths need not be bitwise equal |
| IWSG/resampling arithmetic | Source/common-input tests; ancestor and angle-adjoint interventions | Selected FP32 sensitivity explained; not an exhaustive bound |
| Native filter and metrics | Four older checkpoints, two seeds, 256 trajectories of length150: 602 checks, eight author metric replays | Does not cover every recent trained checkpoint |
| Trained quality | Common-author evaluations of three development pairs, full native validation for units01/02 | Two MDPF checkpoints remain descriptively worse; no population ranking |
| Original paper scope | Bearings configuration and source pin recovered | Appendix B exact driver/settings and original evaluation coverage incomplete |

Unit01's controlled initial-weight intervention reduced its saved native-validation
NLL gap from 0.736745 to 0.034902 nat (95.26%). Together with the passing controller,
optimizer and cloud tests, this supports amplification of independently learned
measurement-model differences on that trajectory. It does not prove stochastic
variation is the only cause, or establish the same explanation for unit02/A-MDPF.
No further translation operation is currently identified for repair. Intrinsic
author failures remain acceptable; the unfavorable KDM checkpoints were finite
valid outcomes and remain evidence.

The next small source question is native TF endpoint coverage for recent fixed
checkpoints. A separately bounded unit02 initial-weight intervention can test
whether the unit01 explanation generalizes, using source transfer and the original
TF schedule with exact pre-intervention replay. No hyperparameter tuning, author
algorithm repair or automatic population-proof campaign follows from this phase.
Missing Appendix B information is a source-availability gap, not evidence that
the implemented bearings filter is wrong.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close composed cloud-law check | All moment screens pass | No invalidity, pin, mutation or coverage veto | Higher moments/broader inputs | Retain alongside primitive/replay evidence | Equality of every distribution feature |
| Whole KDM remains open | Checked source consumers agree | No new defect; adverse trained outcomes retained | Training variation and original-scope coverage | Bounded recent-checkpoint endpoint coverage, then specific initialization intervention | Whole-paper replication or population noninferiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Harness validity checks pass |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Native losses and observed moment differences |
| Default-readiness | Not assessed |
| Next evidence needed | Specific remaining consumer/outcome question, not another generic sampler test |

Post-run red-team: matching low-order moments can miss higher-order or
seed-correlation defects. Prior primitive/common-input tests reduce but do not
eliminate that concern. A repeatable same-state native-operation mismatch would
overturn the corresponding finding. Independent training variation remains a
plausible explanation of final gaps, not a proved universal explanation.

Exact command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_pretraining_cloud_law.py

One attempt used 27.096454 CPU /
27.096385 GPU seconds, charged once.
Active remaining allocation: 205580.430427 CPU /
77151.070263 GPU seconds. No run remains active.
