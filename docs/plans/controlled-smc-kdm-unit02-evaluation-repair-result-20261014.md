# Unit02 MDPF initialization result

Finished 14 October 2026 at 01:00 Hong Kong time. Status: COMPLETED / PASS for this diagnostic. Whole KDM replication remains open.

All declared stages completed; interpretation remains unit-conditional. Validation NLL gap: 0.513438 -> 0.098476 nat. T150 NLL gap: 0.762433 -> 0.157799 nat.

Training completed in [attempt01](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt01/manifest.json):
270 MDPF epochs, best epoch 174, 21,330 updates and source early stopping.
The two-epoch old-start control passed 1,863 checks with zero batch-loss
difference. Training passed 3,782 checks and the terminal source-controller
audit passed 70,512. The selected MDPF stage completed its full protocol;
this intervention did not repeat the entire three-stage training procedure.

The subsequent endpoint checker incorrectly required four unit02 models;
this intervention requires three MDPF checkpoints. It failed before TensorFlow
import. The repair adds an explicit evaluation scope and checks exact model
identities; nine regression cases passed. This was a harness-scope mismatch.

[Attempt02](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02/manifest.json) reused the completed training,
validation and source audit. Regenerated evaluation inputs equal the prior
inputs except for the new scope metadata. No numerical kernel, data, seed or
checkpoint changed; no training was repeated.

All six GPU/XLA evaluations and 472 runtime/provenance checks passed, as did
six author metric replays. The original source/TF baselines reproduced exactly.
Each exercised signature traced once; FP32, TF32 off and verified GPU memory
growth are recorded in the manifest. The
[terminal self-review](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02/terminal-review.json) verified 21
source hashes, nine reused-evidence hashes and unique resource charges.

The intervention changed only initial measurement weights. Source data, source stopping, optimizer and TF random schedule stayed fixed.

| Evaluation NLL gap (TF minus author) | Original TF | Author-start TF |
|---|---:|---:|
| Native validation, mean of eight draws | 0.51343836 | 0.09847584 |
| Native T150, mean of two seeds | 0.76243305 | 0.15779901 |

The observed gap reductions are 80.82% for validation and 79.30% for T150.
The initialization explanation is supported on both fixed-unit evaluations.
Residual gaps remain; this does not establish their sole cause.

| Native T150 metric, mean of two filter seeds | Author trained | Original TF trained | TF from author measurement start |
|---|---:|---:|---:|
| NLL | 6.42448497 | 7.18691802 | 6.58228397 |
| Position RMSE | 3.29368925 | 4.45788240 | 3.28573918 |

These are descriptive comparisons of fixed checkpoints. In particular, the
small favorable RMSE difference is not a statistically supported ranking.
Control max loss difference: 0.0; training checks: 3782; actual author-controller/coverage audit: 70512 checks.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limits |
|---|---|---|---|---|---|
| COMPLETED diagnostic | Three-model evaluation, source metric replay and unchanged baseline replay pass | No failed validity, coverage, trace or provenance checks | One fixed training unit; residual gaps | Consolidate source coverage; inspect saved histories for specific unchecked operations before another test | No whole-KDM completion or population ranking |

| Inference item | Finding |
|---|---|
| Hard veto screen | All declared validity checks passed |
| Statistically supported ranking | None; draws are conditional on fixed trained checkpoints |
| Descriptive differences | Validation and endpoint means, per-draw values, RMSE and histories are preserved |
| Default readiness | No algorithm/default changed or promoted |
| Next evidence needed | Root-cause interpretation of remaining gaps; population studies are separate from replication |

Post-run red-team: a favorable result can still be specific to this initialization and TF random schedule. A mismatch in any non-measurement parameter or a failed replay would invalidate attribution. The weakest evidence is generalization beyond these fixed development units.

Recovery resources: 40.595898 GPU seconds and 4.990683 CPU-reference seconds.
Both attempts together used 5155.749725 GPU and 95.731619
CPU-reference seconds, within the six-hour / one-hour phase cap. Remaining
allocation: 71779.382403 GPU and 205463.582169 CPU seconds.
Each attempt/resource was charged once; no new allocation was used.

Actual command, environments, Git commit, data version, seeds, source snapshots
and CPU/GPU status are in the
[manifest](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02/manifest.json). The command was
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_unit02_evaluation_repair.py.
[Machine decision](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02/decision.json) and
[author metric audit](artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02/endpoint/metric-audit.json) preserve
the result. The original failed endpoint and completed training remain intact.

No specific remaining translation defect is established by this experiment.
The remaining work is source-coverage consolidation and residual-difference
assessment; Appendix B driver/settings and original experiment coverage are
separate limitations. A population noninferiority campaign is not a newly
imposed replication prerequisite. Controlled-SMC bounded fidelity remains
accepted.
