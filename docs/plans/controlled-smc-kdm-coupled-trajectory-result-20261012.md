# KDM coupled-trajectory result, 12 October 2026

The bounded diagnostic completed. With identical starting parameters, Adam
state, row order and base random variates, the initial FP32 trajectory agrees
closely with the author implementation. The trained FP32 trajectory has
localized gradient and first-moment differences and one different resampled
ancestor. Parameter differences remain small. FP64 source/TF comparisons pass
throughout. These results do not establish the cause of the final trained
quality gap or complete KDM replication.

The [plan](controlled-smc-kdm-coupled-trajectory-20261012.md) is terminal:
all three permitted attempts have been used. Attempt03 completed 16 updates
from each of two saved states, with 70/70 sampler/control checks and 166/166
terminal artifact checks. No experiment is active and no production algorithm
was changed.

## Observations

All numbers below are maxima across the 16 updates and recorded time steps.
The two starting states are the repaired unit01 MDPF starting and saved-best
checkpoints; they are loaded identically into each implementation.

| Starting state and comparison | Absolute loss difference | Absolute parameter difference | Angle difference, radians | Changed ancestors |
|---|---:|---:|---:|---:|
| Initial, TF32/source32 | 3.815e-6 | 5.960e-8 | 6.437e-6 | 0 |
| Initial, TF64/source64 | 3.553e-15 | 7.841e-16 | 4.134e-13 | 0 |
| Trained, TF32/source32 | 6.723e-5 | 1.192e-7 | 1.40541 | 1 |
| Trained, TF64/source64 | 7.105e-15 | 1.110e-16 | 1.732e-11 | 0 |

The initial FP32 comparisons have no gradient or moment flags. The trained
FP32 comparisons flag gradients on 10 updates and first moments on 13 under
the declared local diagnostic tolerance. Maximum absolute differences are
0.00280005 for gradients and 0.000279669 for first moments. All parameter and
second-moment comparisons pass. These tolerances locate discrepancies; they
are not trained-quality acceptance thresholds.

The one ancestor difference occurs at update index 10, time 5. The source
and TF losses for that update are both 6.789874076843262. The maximum angle
difference is 1.40541 radians at time 5 but only 1.1444e-5 at time 6. This is
a localized branch difference, not evidence of monotonically growing error.
The exact distance of the shared uniform from the two CDF boundaries was not
retained, so the arithmetic operation responsible for crossing the boundary
has not been isolated.

The requested 1e-8 source-only perturbation changed one base-noise value by
1.0011717677116394e-8, from 0.00026871569571085274 to
0.00026872570742852986. Nevertheless, all measured proposal, particle,
loss, gradient, moment and parameter differences were zero for both states.
The input change was representable but had no observable downstream effect.
It therefore supplies no useful numerical-error allowance and does not
demonstrate robustness. This negative control result is retained.

## Source and mathematical trace

The author angle route is in
.localresources/code/younis-mdpf-neurips-2023/src/models/particle_transformer.py,
lines 112–168 (atan2 at 123; normalization at 142–156), called by the proposal
augmentation in kde_particle_filter.py, lines 813–848. The TF consumer calls
[the same transformation](../../bayesfilter/replication/kdm_neural_replay_tf.py:153)
at lines 168–173. Both normalize the two predicted coordinates and recover
the angle without an added epsilon.

Writing the raw pair as (u,v), theta = atan2(u,v) gives

    d theta = (v du - u dv) / (u² + v²),    ||gradient theta|| = 1/r,
    r = sqrt(u² + v²).

Positive normalization leaves the angle unchanged. Small r makes the shared
source operation sensitive to upstream arithmetic differences. The minimum
recorded r is 0.84325 in the initial FP32 trajectory and 0.00119962 in the
trained one. In a saved trained-state example at update 0, time 12,
row 25, particle 15, the measured raw-coordinate change predicts an angle
change of 0.000140481827 radians; the observed change is 0.000140503049.
At update 15, time 15, row 29, particle 21, prediction is -0.001588607698
versus observed -0.001589715481. This checks the local amplification mechanism;
it does not attribute every upstream difference or the final RMSE gap to it.
Adding an epsilon would change the source algorithm and is not authorized by
this diagnostic.

The source mixture sampler is in
.localresources/code/younis-mdpf-neurips-2023/packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation/kernel_density_estimator.py,
lines 146–184. The diagnostic couples categorical draws by inverse CDF:
A = min{j: U < F_j}. An arbitrarily small change in F can select another
component when U lies between two boundaries. Normal and von Mises draws
then depend on each implementation's own component parameters. This explains
how a local finite jump can coexist with small parameter differences. The
observed event is consistent with this mechanism; its exact boundary margin
still needs to be saved before assigning a more specific cause.

The sampler adapters passed 70 controls, including same-input, source-repeat,
own-state response and exact proposal-base-array checks. No completed
resampled particles were shared. The random mapping for the categorical draw
differs from the author's native multinomial implementation while preserving
the categorical distribution. Within each dtype the base variates agree;
FP32 and FP64 use dtype-specific random variates. Thus FP64 agreement supports
equation compatibility but is not a controlled demonstration that changing
precision repairs a particular FP32 trajectory.

## Execution and reproducibility

Exact commands:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/run_kdm_coupled_trajectory.py
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/review_kdm_coupled_trajectory.py

The [manifest](artifacts/controlled-smc-kdm-replication-20260928/coupled-trajectory-20261012/attempt03/manifest.json)
records commit 89065bc6354801cb368d5e163ba49fd9c4372d10, commands, environment,
seed 202610122, input and source hashes, source snapshots and output paths.
The diagnostic uses batch 64, 25 particles, 17 time steps and rows 384:1408
from the saved 5,000-row author training data. Data SHA-256 is
3d45b37797f881e9da2b21973bfa2b6a440235b606a20492dc25f4f6c619a176.
The pinned author commit is b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
Both starting checkpoint hashes are preserved in the
[decision](artifacts/controlled-smc-kdm-replication-20260928/coupled-trajectory-20261012/attempt03/decision.json).

TF 2.20.0-dev0+selfbuilt runs on GPU under escalated access, XLA enabled,
TF32 disabled, soft placement disabled and verified memory growth on both
visible devices. GPU0 is RTX 4080 SUPER; GPU1 is RTX 5080. Each of the four
training signatures traces once; peak TF allocator use is 274,865,152 bytes.
Generic startup retracing warnings concern separately constructed random
wrappers; per-wrapper trace counts were not collected. They do not indicate
repeated tracing of the measured training step. Source Torch 2.11.0+cu130
runs in tf-gpu2 as an independent CPU reference with GPUs intentionally hidden.
CPU reference loops and host serialization are diagnostic boundaries, not
fallback implementations of the TF training kernel.

Attempt01 failed at the author C++ library lookup; attempt02 failed at a
diagnostic folded-seed dtype mismatch. Both are harness failures, repaired
without changing the filtering algorithm. Their charges were 62.302774 and
7.900130 seconds per resource. Attempt03 charged 618.269346 seconds per
resource and completed at 16:10:38 HKT on 12 October. Total phase charge is
688.472249 CPU seconds and the same GPU seconds, within the 3,600-second caps.
The ledger now has 219,836.468876 CPU seconds (61.07h) and
90,889.071432 GPU-tranche seconds (25.25h). The larger aggregate GPU figure
is not additional authorization.

The [terminal review](artifacts/controlled-smc-kdm-replication-20260928/coupled-trajectory-20261012/attempt03/terminal-review/review.json)
passes 166/166 checks, covering provenance, data/checkpoint identity, completed
updates, finite outputs, sampler controls, placement, trace counts and budget.
This was an executable terminal check and Codex self-review, not an independent
reviewer certification.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not established |
|---|---|---|---|---|---|
| Complete the diagnostic | Own-state coupled trajectories and controls recorded | No invalid artifacts or nonfinite outcomes | Perturbation did not affect measured computation | Design an effective source perturbation with observed first effect | A numerical acceptance envelope |
| Preserve the present runtime | Local source/TF equations and updates agree closely; all FP64 checks pass | FP32 flags retained as localization evidence | Discrete sampling and trained angle sensitivity | Capture actual CDF margins and common inputs across precision arms | Final-quality noninferiority |
| Keep KDM replication open | Existing full-training quality criterion unresolved | Unfavorable finite training pairs remain evidence | Independently pretrained starts and independent full-run randomness | Cross pretrained starts through both engines before another long ladder | Population ranking or paper-wide replication |

| Inference question | Status |
|---|---|
| Hard veto screen | No new nonfinite or invalid outcome in the diagnostic; two harness failures repaired and charged |
| Statistically supported ranking | None; one training unit and short deterministic comparisons cannot rank learned quality |
| Descriptive-only differences | The table, gradient/moment flags and one ancestor event |
| Default-readiness | Unchanged; no algorithm/default change or new production claim |
| Next evidence needed | Effective perturbations, saved CDF boundaries, crossed pretrained starts, then an affordable untouched quality design |

The next phase needs a fresh bounded diagnostic plan. Predeclare a
representable perturbation scale ladder and retain ineffective cases; measure
the first changed proposal rather than selecting a perturbation that excuses
the TF discrepancy. Preserve the original 1e-8 result. To interpret precision
causally, supply the same quantized base inputs to FP32 and FP64. Separately,
cross source-trained and TF-trained pretrained weights through both engines
with identical rows, moments, rates and random inputs. A gap that follows the
starting weights directs investigation to pretraining; a gap that follows the
engine directs it to an implementation operation. These short comparisons
still cannot qualify final learned quality. No angle epsilon, objective
change, new independent full-training ladder or relaxed acceptance rule follows.

Post-run red-team: independently learned pretraining and ordinary training
variability remain alternative explanations of the earlier full-run quality
differences. A reproducible common-start engine-specific discrepancy in FP64
would weaken the numerical-sensitivity explanation. The weakest evidence here
is the ineffective perturbation and short, single-unit scope. Controlled-SMC
tested-input fidelity is unchanged; the retired 1%/5% rule remains withdrawn.
