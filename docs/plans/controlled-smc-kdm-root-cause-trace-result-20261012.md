# KDM root-cause trace: checked findings and remaining attribution

12 October 2026. The code trace rules out an incorrect stage transfer in the two
new training units and locates the ineffective perturbation at the author's
first FP32 linear layer. It confirms shared numerical sensitivity, but it does
not identify the cause of the final trained-quality gap.

The claimed target remains replication of the released author algorithm.
The new computation is a deterministic replay of the author's proposal and an
inspection of actual starting checkpoints. Those are exact local checks, not
a comparison of complete learned filters. The unchanged KDM qualification
criteria remain unresolved. No source algorithm or production implementation
was modified.

## What was actually executed

Plan: controlled-smc-kdm-root-cause-trace-20261012.md.
Driver: docs/benchmarks/diagnose_kdm_root_cause_trace.py.
Evidence root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-cause-trace-20261012/.

Attempt01 failed in the diagnostic reader because weight pretraining saves
three individual module checkpoints, not a full-model best checkpoint.
The reader was corrected to use those existing source files. Attempt02 passed.
The failed attempt and its driver are preserved. This was a harness error,
not a failure of either filtering implementation.

Attempt02 regenerated the exact previous GPU/XLA FP32 Philox proposal noise,
then called the pinned author's actual proposal on the saved initial cloud and
the actual unit01 starting/best weights. The unperturbed raw proposal matches
the saved author tensor bitwise in both cases. Identical repeats also match.
The CPU author worker used PyTorch 2.11.0+cu130 with GPU intentionally hidden.
GPU noise generation used verified memory growth, TF32 disabled and one trace
per stable signature. The terminal artifact review passed 81/81 checks.
See attempt02/result.json, terminal-review.json, environment.json and manifest.json.

## 1. The independent full-training runs start from different learned weights

The two new units use the same fresh constructor parameters before weight
pretraining. At the beginning of full MDPF training, all 569 parameters in the
three pretrained modules differ between source and TensorFlow. The remaining
627 parameters are identical.

| Module | Parameters | Unit01 maximum absolute difference | Unit02 maximum absolute difference |
|---|---:|---:|---:|
| Weight model | 217 | 0.0646419 | 0.550046 |
| Observation encoder | 168 | 0.0159385 | 0.0493168 |
| Particle encoder for weights | 184 | 0.132947 | 0.0672164 |
| All nontransferred parameters | 627 | 0 | 0 |

Both engines load exactly their own selected pretrained weights. The source's
saved untrained MDPF modules equal its three pretrained best module files;
the TF starting checkpoint equals its own weights-best checkpoint. The
nontransferred parameters match the common constructor. These checks passed
for both units.

Call chain checked: reference_kdm_fresh_training_cost.py:257 constructs the
source model and then loads its own predecessor checkpoints; the author's
trainer.py:613 saves the actual pre-optimization state. The TF epoch runner
calls transfer_stage_parameters, whose weights_to_mdpf branch selects the
same three modules and resets Adam. See kdm_stage_transfer_tf.py:11 and the
author's kde_particle_filter.py:280-355. The saved-state comparisons test this
consumer wiring, rather than merely confirming that a transfer helper exists.

This is a confirmed confounder, not evidence that different pretrained weights
are inherently wrong. Parameter distance is not functional distance and cannot
rank the learned filters. Independent stochastic training can legitimately
produce different weights. The important limit is that the previous coupled
trajectory loaded the TF-produced starting/best checkpoint into both engines;
it did not test the independently source-trained starting point or establish
why weight pretraining diverged.

The full runs use independent stochastic streams. The author shuffles through
its PyTorch DataLoader (trainer.py:144); the TF driver uses stateless shuffle
and separate per-stage/per-epoch/per-batch keys
(diagnose_kdm_source_epoch_tf.py:82,105). Validation also draws fresh clouds,
and its loss controls checkpoint and learning-rate decisions at line 132.
Thus training randomness and checkpoint selection are plausible contributors.
Their contribution to final quality has not been measured by this trace.

## 2. The 1e-8 perturbation is absorbed in the first linear layer

The previous diagnostic changed proposal-noise coordinate 607 from
0.00026871569571085274 to 0.00026872570742852986, a representable difference of
1.0011717677116394e-8. This trace uses exactly that noise and coordinate.

In both starting and trained states, the concatenated neural-network input
changes, but every output of the first Linear layer is bitwise identical.
All subsequent activations, raw particles, normalized particles and final
particles are identical. This directly locates where the perturbation is lost;
it is not merely an observation that the final loss did not change.

The first layer computes Wx+b in FP32. The perturbation changes one term of
that sum; the implemented floating-point operation returns the same rounded
output. This local insensitivity says nothing about larger perturbations,
other coordinates, or proximity to a resampling boundary.

The predeclared ladder gives the following results for this one coordinate:

| Requested change | Starting-state first operation with identical output | Trained-state first operation with identical output |
|---|---|---|
| 1e-8 | First Linear | First Linear |
| 1e-7 | Sigmoid | Addition of residual to previous particle |
| 1e-6 | Sigmoid | None; final particle changes |
| 1e-5 | None; final particle changes | None; final particle changes |

At 1e-6 the trained-state final particle maximum change is 1.669e-6.
At 1e-5 the starting/trained final changes are 2.384e-7 and 9.239e-6.
These are local descriptive measurements. They are not a new error allowance
or evidence that the final likelihood/RMSE discrepancy is acceptable.

Source anchor: LearnedProposalNoTimestampResidual.forward,
proposal_models.py:173-209. The actual Sequential modules were traced with
forward hooks; sigmoid, residual scaling and addition were reconstructed and
checked bitwise against the actual source proposal output before interpretation.

## 3. Small-radius angle coordinates amplify local arithmetic differences

Both implementations learn residuals in the internal coordinates
(x,y,u,v), then normalize (u,v) and return theta=atan2(u,v).
For r=sqrt(u²+v²)>0,

    d theta = (v du - u dv) / (u²+v²),    ||gradient theta|| = 1/r.

This follows by differentiating atan2; the intermediate positive normalization
does not change the angle in exact arithmetic. The shared source operation has
no epsilon in that normalization. In the prior trained FP32 trajectory the
smallest observed raw radius was 0.00119962, giving local angular sensitivity
about 834 per unit of raw-coordinate perturbation.

Saved raw-coordinate differences quantitatively predict selected observed
angle differences in the completed coupled-trajectory result. That supports
the mechanism for those local differences. It does not prove that this
mechanism causes the final validation gap.

Source anchors: particle_transformer.py:112-156;
kde_particle_filter.py:813-848; TF kdm_neural_replay_tf.py:152-180.
The author and port share this operation. Adding an angle epsilon would change
the baseline and is not a replication repair under the current scope.

## 4. Resampling has a discontinuous boundary, with a limited demonstrated effect

Under the diagnostic's common inverse-CDF mapping, an ancestor index changes
when a common uniform crosses a cumulative-weight boundary. A tiny weight or
summation change can therefore select a different particle.

The previous coupled trained-FP32 run has one ancestor discrepancy at update
index 10, time 5. Its maximum angle difference is 1.40541 radians at that time
but returns to 1.1444e-5 at time 6; the scalar loss at that update is identical
in the two FP32 engines. The large local jump is not evidence of persistent
trajectory divergence. Gradient flags already appear at update 1, before the
first changed ancestor, so this one switch cannot explain all gradient flags.

The common inverse-CDF mapping is a diagnostic coupling. The production TF
sampler calls tf.random.stateless_categorical (kdm_sampling_tf.py:71), and the
author uses its native categorical sampler. The earlier event therefore
establishes sensitivity in the coupled diagnostic, not the location of a
specific failure in a native full-training run. Actual uniform/CDF margins at
the first discrepancy were not saved and remain to be captured.

## 5. The checked execution path does not support a current XLA/retracing cause

The active call chain is the epoch runner, SourceWeightPretraining or
SourceFullTraining, BearingsNeuralReplay, and the Normal/von-Mises KDE and
source sampler. Numerical kernels use TensorFlow with stable tf.function
signatures. Time evolution and rejection sampling use tf.while_loop.
The Python loops are host epoch/batch orchestration, optimizer-variable
bookkeeping and fixed layer construction, not a per-particle numerical loop.
The checked kernel modules import TensorFlow and standard-library math; their
host tensor materialization is for state transfer, assertions or artifacts.
No NumPy numerical runtime path was found in this chain. The independent
author reference retains its original backend.

The earlier first-step XLA problem is documented in
kdm_neural_replay_tf.py:157 and :236: differentiating a runtime conditional
around the Bessel path produced unsupported variant FakeParam nodes. The
current implementation peels that fixed first transition and uses a traced
loop for the remaining transitions. Completed coupled runs have one trace per
signature. This supports closure of that checked compilation problem, not a
claim that all future shapes or devices are validated.

The source IWSG expression is also wired to the TF consumer:
source kde_particle_filter.py:738-749 uses detached samples and
exp(log q - stop_gradient(log q))+1e-8; TF
kdm_normal_von_mises_tf.py:94-103 implements the same expression, including the
source's weight normalization convention. Previous coupled FP64 agreement
supports the tested forward/backward computation. No new missing-score or
stage-reset error was demonstrated here. The older Adam scalar-precision
defect was already repaired and cannot, without new evidence, explain the
post-repair comparisons.

## Decision and next discriminating work

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not established |
|---|---|---|---|---|---|
| Accept local trace | Exact saved raw proposals, repeat replay and checkpoint wiring pass | Initial reader failure repaired; terminal 81/81 | Two states and one noise coordinate only | Preserve trace and expand only for a stated question | General robustness |
| Keep KDM trained-quality replication open | Final NLL/RMSE qualification unresolved | Two unfavorable finite full pairs remain visible | Pretraining, independent random streams and engine arithmetic are confounded | Couple pretraining updates and cross starting weights | Population inferiority or acceptance |
| Preserve author baseline | Shared angle/resampling mechanics identified | No demonstrated new source deviation in this trace | Their contribution to final quality is unmeasured | Instrument existing operations | Need to change the author method |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Invalid checkpoint assumption rejected attempt01; exact replay and finite local comparisons pass in attempt02 |
| Statistically supported ranking | None |
| Descriptive-only differences | Module parameter distances, perturbation effects and earlier local gradient/ancestor differences |
| Default-readiness | No promotion; source reference and diagnostic noise generation only |
| Next evidence needed | Common-input pretraining updates, crossed learned starts, actual CDF margins, controlled precision inputs, then affordable untouched quality evidence |

The next bounded design should first run weight-pretraining updates from the
same constructor with identical batches and cloud draws, comparing loss,
gradient, Adam moments and parameters. Then load each actual source-trained
and TF-trained MDPF starting state into both engines, with the same moments,
rates, rows and quantized base inputs. Compare engines within each starting
state before interpreting differences between starting states. Use fresh
versioned outputs and frozen limits. Preserve ineffective perturbations and
record actual uniform/CDF boundaries.

A short crossed experiment can localize arithmetic or starting-state effects;
it cannot establish the cause of full trained quality on its own. If those
local checks pass, the remaining question is stochastic learning and
qualification, rather than an identified translation defect. Do not start
another independent full-training ladder or relax the 0.01 nat/1% criteria
from these findings.

Post-run red team: exact proposal replay is strong for the observed loss of
the perturbation. The weakest evidence is attribution of downstream quality:
the crossed-start comparison is still missing, and parameter distances do not
measure predictive behavior. A reproducible same-input pretraining or
full-stage mismatch would overturn the current absence of an identified
remaining translation defect. Source stochastic variability could also explain
some observed final differences; that requires a justified uncertainty design.

## Budget and status

Attempt01: 10.277816 seconds. Attempt02: 9.917140 seconds.
Both are charged conservatively to each resource: 20.194956 seconds each.
Remaining campaign CPU: 219816.273921 seconds (61.06h).
Remaining GPU tranche: 90868.876477 seconds (25.24h).
Both launch slots are consumed; this phase is terminal. No job is active.
The ledger counter remains160. Original iAPF and E0/Task3 remain deferred.
Controlled-SMC's bounded tested-input verdict and the withdrawal of the former
1%/5% rules are unchanged; this diagnostic made no new cSMC claim.
