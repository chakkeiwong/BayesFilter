# Recent KDM native endpoints: completed

13 October 2026, 20:38:04 Hong Kong. All nine recent fixed checkpoints completed
both native filter seeds. The run passed 1385 runtime/provenance checks,
18 pinned-author metric replays and 89 terminal checks.
No new translation defect, invalid distribution, retracing, or XLA failure
was found in these executions.

[Plan](controlled-smc-kdm-recent-endpoint-plan-20261013.md);
[decision](artifacts/controlled-smc-kdm-replication-20260928/recent-endpoint-20261013/attempt01/decision.json);
[manifest](artifacts/controlled-smc-kdm-replication-20260928/recent-endpoint-20261013/attempt01/manifest.json);
[terminal review](artifacts/controlled-smc-kdm-replication-20260928/recent-endpoint-20261013/attempt01/terminal-review.json).

## Checked target and execution

The actual SourceInitialCloud.sample and BearingsNeuralReplay.native_evaluate
consumers evaluated source-trained and TF-trained MDPF/A-MDPF checkpoints from
units01/02, plus the completed unit01 author-initialized TF MDPF checkpoint.
Each model used its unit's same 256 evaluation trajectories, all 150 time points
including the initial cloud, N=25, batch64 and two recorded native TF seeds.
No model was retrained, selected or tuned in this phase. Algorithm runtime
modules are unchanged; the diagnostic exporter and coverage counts were extended.

GPU execution used RTX5080, TensorFlow/XLA FP32, TF32 off and verified memory
growth. Every exercised initializer/filter/metric signature traced once.
Per-state and per-trajectory metrics, conditional slices, full logs, commands,
data/checkpoint hashes and source snapshots are retained. Author preparation
and metric replay ran in isolated tf-gpu2 on CPU, with CUDA_VISIBLE_DEVICES=-1
before import. All replay-recorded source hashes match pinned
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.

On each model's first two trajectories for the first seed, unchanged author
metric classes consumed saved native TF clouds, weights and bandwidths.
Maximum absolute NLL discrepancy was 1.90735e-6 nat; maximum absolute
squared-error discrepancy was 9.15527e-5. Largest pointwise scaled discrepancy
was 1.77112e-6, below the predeclared 2e-4 arithmetic screen. This checks metric
arithmetic on common inputs, not equality of independent random trajectories.

## Fixed-checkpoint outcomes

These are descriptive means over two filter seeds on reused development data.
They are not independent training replicates or a supported population ranking.

| Unit | Method | Saved checkpoint | NLL | RMSE |
|---|---|---|---:|---:|
| 1 | am_dpf | source_trained | 5.804663 | 3.195859 |
| 1 | am_dpf | tf_trained | 5.901089 | 3.298903 |
| 1 | mdpf | author_start_tf_trained | 6.198162 | 3.314408 |
| 1 | mdpf | source_trained | 6.141218 | 3.254392 |
| 1 | mdpf | tf_trained | 6.919188 | 4.007646 |
| 2 | am_dpf | source_trained | 5.839495 | 3.230127 |
| 2 | am_dpf | tf_trained | 6.069054 | 3.405186 |
| 2 | mdpf | source_trained | 6.424485 | 3.293689 |
| 2 | mdpf | tf_trained | 7.186918 | 4.457882 |

The unit01 controlled intervention changed only initial measurement weights
before the already-completed training trajectory. Its observed MDPF NLL gap
versus the source-trained model falls from 0.777971 to 0.056944 nat,
a 92.68% reduction. This second evaluation context supports the
earlier initialization explanation: full source-native validation showed
95.26% reduction. Those two evaluations use different settings and datasets;
their percentages should not be pooled as independent replications.

Unit02's observed MDPF NLL gap remains
0.762433 nat.
Its worse saved outcome is visible through the native TF evaluator and the
earlier common-author evaluator. No unit02 initial-weight intervention has yet
tested whether it shares unit01's explanation. A-MDPF differences also remain
descriptive. Finite unfavorable outcomes cannot be dismissed as intrinsic
author failures.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close recent native endpoint coverage | Nine models, 18 complete executions; author metric replays pass | No input, output, state-mutation, source or budget veto | Untested inputs and independent paths | Retain; do not repeat generic endpoint checks | Entire filter-law equality from metric replay |
| Retain initialization explanation for unit01 | Controlled model has smaller observed gap in both contexts | Training control and fixed model provenance intact | One controlled training trajectory | Test same mechanism on unit02 | Universal explanation or population superiority |
| Keep whole KDM open | Source checks pass within bounded scope | No new translation defect | Other learned outcomes and original-paper scope | Bounded unit02 intervention with exact control replay | Whole-paper replication or population noninferiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | All declared runtime, provenance and metric checks pass |
| Statistically supported ranking | None; no training-population comparison performed |
| Descriptive-only differences | Native checkpoint NLL/RMSE means and observed gap reductions |
| Default-readiness | Outside this replication phase |
| Next evidence needed | Unit02 controlled starting-weight intervention; original-source coverage separate |

The next experiment should preserve unit02's original stage index, RNG schedule,
source optimizer/controller, data and checkpoints. First replay its saved initial
epochs through the unchanged epoch driver; then replace only initial measurement
weights with the author's and evaluate fixed outputs under the same source
metrics. It needs a bounded phase plan under the existing campaign balance.

Post-run red-team: metric replay covers only two trajectories per model and
cannot certify the native filter's entire stochastic law. Earlier primitive,
common-input and training-call-chain checks remain necessary evidence. The
strongest alternative explanation for the controlled outcome is sensitivity
specific to one training trajectory. An analogous unit02 intervention that
does not reduce its gap would weaken generalization, not invalidate unit01.

Exact command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_recent_endpoint.py

One attempt completed: 215.938134 GPU /
21.116640 CPU seconds charged exactly once.
Remaining active allocation: 76935.132129 GPU /
205559.313788 CPU seconds. No campaign run is active.
