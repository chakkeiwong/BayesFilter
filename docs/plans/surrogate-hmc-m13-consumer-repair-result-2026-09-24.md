# M13 repaired-default replay result — 2026-09-24

Current interpretation, after the owner's materiality decision: the
saved-run difference is an explanatory warning, not a continuation blocker.
The saved replay passes the mandatory engineering checks under the
[implemented policy](surrogate-hmc-fp32-warning-policy-2026-09-24.md).
This is a decision-policy change over immutable saved numerical evidence;
the original execution report and comparison values below are preserved.
The one-replay allowance does not make optional repeatability a prerequisite.
M13 scoped controls and posterior testing remain pending.

The approved GPU replay completed, but restoration of the diagnostic output
contract is not numerically verified in FP32. All nine saved parameter points
remain finite. The maximum scaled value difference is 0.00497577665 and the
maximum scaled score difference is 0.0138047338, both above the unchanged
0.00003 regression limit. The 76 required CPU review tests passed.

The campaign's goal remains testing surrogate HMC: canonical analytical DCTR
forces propose trajectories, and the declared endpoint posterior determines
Metropolis acceptance. This replay checks an implementation prerequisite.
It evaluated a finite canonical likelihood and its analytical derivative for
M13, a four-state, two-observation range/bearing model, at T20/N256 and nine
fixed noise-scale settings. It ran no HMC chains and introduced no prior.

## Checked findings

| Check | Result |
| --- | --- |
| Required CPU regression | 76 passed; GPUs intentionally hidden |
| GPU/XLA and memory policy | GPU:0, RTX 4080 SUPER; growth verified on both physical GPUs |
| FP64 full finite-program derivative, h=1e-5 | Maximum scaled error 2.70491231e-9; pass |
| FP64 derivative refinement, h=1e-4 | Maximum scaled error 3.26045022e-7; pass |
| FP64 trace/value/score agreement | Value and score differences exactly zero |
| FP64 repeat of the same compiled function | Exact equality |
| Correction condition and feasibility diagnostics | Present |
| FP32 finite-program validity | All nine points valid |
| FP32 saved-baseline regression | Fails for value and score at all nine points |

For the frozen finite scalar, the FP64 analytical score agrees numerically
with central differences at the stated points. This does not prove equality
to the integrated model likelihood. The known untuned-control likelihood
errors remain a separate problem.

## Saved-baseline comparison

Scaled errors divide the absolute difference by one plus the absolute saved
value or score component. They are regression diagnostics, not percentages
of a posterior error or a convergence measure.

| lambda_Q | lambda_R | Saved log likelihood | Replay log likelihood | Absolute difference | Scaled value difference | Maximum scaled score difference |
| --- | --- | --- | --- | --- | --- | --- |
| -0.693147 | -0.693147 | -213.083908 | -213.129761 | 0.0458526611 | 0.000214180793 | 0.0029181745 |
| -0.693147 | 0 | -23.1718216 | -23.051548 | 0.12027359 | 0.00497577684 | 0.0034242278 |
| -0.693147 | 0.693147 | 14.264761 | 14.2442608 | 0.0205001831 | 0.00134297439 | 0.00406811025 |
| 0 | -0.693147 | -34.4175415 | -34.3477325 | 0.06980896 | 0.00197102783 | 0.00195040263 |
| 0 | 0 | 22.8902588 | 22.9000168 | 0.00975799561 | 0.000408450812 | 0.00181819462 |
| 0 | 0.693147 | 25.236517 | 25.2350731 | 0.00144386292 | 5.50325684e-05 | 0.000375464343 |
| 0.693147 | -0.693147 | 13.9250793 | 13.9267874 | 0.0017080307 | 0.00011444031 | 0.000377072436 |
| 0.693147 | 0 | 28.9458828 | 28.953186 | 0.00730323792 | 0.000243881203 | 0.013804734 |
| 0.693147 | 0.693147 | 24.0359421 | 24.0349522 | 0.00098991394 | 3.9539712e-05 | 0.000301665525 |

## Source review and interpretation

A read-only comparison of the archived baseline with this replay confirms
that the manifest fields for the fixture, observations, particle design,
parameter grid, controls, precision, TF32, and XLA match. The only differing
target-manifest field is source_sha256, with changes in three dependencies:
ledh_unified_correction_tf.py, ledh_canonical_batch_fused_tf.py and
ledh_range_bearing_target_tf.py. The first adds a diagnostic-shape helper;
the latter two add the opt-in diagnostic API and output construction.

The repaired default returns the original five scalar diagnostics, but it
still constructs tf.fill fallback tensors and uses dictionary lookup with
those fallbacks, where the archived implementation directly stacked the
existing diagnostics. Those unused graph operations are a candidate
explanation, not an established cause. Cross-process compilation or GPU
numerical variation has not been isolated. It is therefore unsupported to
attribute the replay difference definitively to XLA ordering or to conclude
that the numerical repair works. An archived-source repeat and a matched
current-source comparison are needed to distinguish the explanations.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Repair remains unresolved | FP32 regression fails | No invalid points or FP64 derivative failure; no-harm criterion fails | Output-graph effect versus fresh compilation variability | Localize source/graph differences and prepare a bounded repeatability comparison | Surrogate-HMC failure, posterior accuracy, or all-model completion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | FP32 restoration cannot be promoted |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-point replay differences and runtime |
| Default readiness | Repair not GPU-qualified |
| Next evidence needed | Matched archived/current execution, then scoped controls, reference and posterior tests |

The numerical replay allowance was one attempt, now used. Unused compute
within its reservation does not authorize another replay. Read-only
localization and preparation remain within the existing master. Any proposed
additional GPU comparison needs an explicit bounded plan before execution;
this is implementation repair within the same research direction.

## Execution and accounting

Exact approved command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-consumer-repair --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repair-01

Git commit: 8a4cffa2fd085f43ab5710c24d6e7b48770aea0d. Environment:
tftwogpu, TensorFlow 2.20.0-dev0+selfbuilt. Run-time source snapshots,
actual commands, data/design seeds, GPU policy and full logs are
preserved in the run directory. The driver records
COMPLETED_BOUNDED_CLOSURE_PASS_CAMPAIGN_INCOMPLETE; the worker records
M13_FULL_CONSUMER_PARITY_REQUIRES_LOCALIZATION.

Charged this pass: 143.692720 CPU-worker seconds and 96.870465483 GPU-wall
seconds. Cumulative charged totals: 42,299.956468 CPU and 33,604.547754150 GPU
seconds. Remaining global balances: 11,700.043532 CPU and 193,195.452245850 GPU
seconds against ceilings of 54,000 and 226,800. The earlier conservative
6,340-CPU-second recovery charge remains included; it is a cost bound.

Evidence:
- artifacts/surrogate-hmc-closure-20260922-m13-repair-01/accounting.json
- artifacts/surrogate-hmc-closure-20260922-m13-repair-01/run-manifest.json
- artifacts/surrogate-hmc-closure-20260922-m13-repair-01/review/command.log
- artifacts/surrogate-hmc-closure-20260922-m13-repair-01/m13-consumer-repair/report.json
- surrogate-hmc-m13-consumer-repair-2026-09-24.md

Post-run review: the strongest alternative explanation is variability in
fresh XLA compilation rather than changed mathematical operations. The
present evidence does not include an archived-source repeat to test it.
A stable archived/current paired comparison meeting the original criterion
would overturn the unresolved repair verdict. Neither outcome alone repairs
the independent likelihood-approximation gap.
