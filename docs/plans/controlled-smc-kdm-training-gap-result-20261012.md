# Current MDPF training-gap code trace and diagnostic result

12 October 2026. The investigation is complete; the cause of the large learned-quality gap is not yet established. The trace identifies a poorly conditioned angle transformation shared with the author, confirms small FP32 gradient differences at current trained states, and finds no failed local parameter-update comparison. It does not establish that rounding explains the unfavorable trained checkpoints. No algorithm or numerical default was changed in this investigation.

The governing [diagnostic plan](controlled-smc-kdm-training-gap-investigation-20261012.md) keeps the released author implementation as comparator. The question concerns MDPF training in the two new calibration units. Controlled SMC, original iAPF and Task 3 are separate. The [completed calibration](controlled-smc-kdm-paired-calibration-result-20261011.md) remains unfavorable evidence: unit01/unit02 have TF-minus-author NLL differences of +0.85829/+0.76678 nat and RMSE changes of +26.59%/+36.03%. These are finite outcomes from a common author evaluator, not matched author failures. Three training pairs do not support a population ranking.

## What the call chain establishes

The training consumer is `docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py` → `diagnose_kdm_source_epoch_tf.py::run_epochs` → `bayesfilter/replication/kdm_source_stages_tf.py::SourceFullTraining` → `kdm_neural_replay_tf.py::_native_sequence/_transition/_apply_update`. The following findings combine inspected wiring with executed comparisons; function existence alone is not the evidence.

| Suspected cause | Checked code and evidence | Verdict and limit |
|---|---|---|
| Wrong pretrained weights, bandwidth or optimizer reset | `kdm_stage_transfer_tf.py:12`; actual saved-state checks in `reference_kdm_training_terminal_audit.py:124`; source `kde_particle_filter.py:280–355` | Both units load their own preceding-stage best state and reset moments/steps correctly. Passing this does not make independently trained source/TF pretraining weights identical. |
| Wrong sequence or gradient estimator | `kdm_source_stages_tf.py:80`, `kdm_neural_replay_tf.py:152–243`, `kdm_normal_von_mises_tf.py:95`; source `kde_particle_filter.py:740–748`; paper equations (14)–(15) | Checked first-transition behavior, detached resampling with IWSG weight injection, four-step truncated differentiation and selected loss times. Common-input FP64 forward/gradient/update comparisons pass at the new states. This is not a full native-trajectory proof. |
| Wrong late Adam state, rates or update equation | `kdm_neural_replay_tf.py:261`; saved-state experiment below | Actual moments, steps 21646/20777 and rates 5e-6/5e-7 were restored in both libraries. All tested parameter, moment and step comparisons pass. Two updates do not bound long-run accumulation. |
| Scheduler, early stopping or checkpoint selection mismatch | Actual controller replay and best-checkpoint hashes in both independent terminal audits; epoch runner | Source scheduling/stopping rules pass. Different validation trajectories can legitimately choose different stopping epochs. |
| Wrong final TF evaluator | Both trained models were evaluated using the same author evaluator | This cannot explain the reported common-author comparison. Prior native TF endpoint checks concern another, separately bounded question. |
| XLA failure, retracing, Python/NumPy numerical fallback | Stable input signatures in the stage/replay models, TensorFlow time loops and pure TF numerical kernels; one trace per case in every new executable replay | No such failure occurred in these tests. Python handles epochs/batches and fixed layer/parameter construction; it does not unroll the filter horizon into a Python execution loop. Host `.numpy()` conversions serve assertions/state/reporting. No NumPy computation or Python callback was found in the traced TF numerical path. This is a scoped finding, not a repository-wide clean bill. |

The author pin is `b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18` under `.localresources/code/younis-mdpf-neurips-2023`. The local paper text is `.localresources/papers/younis-sudderth-2023-long-range-tracking.txt`, equations (14)–(15) at lines 447–474. The source computes an IWSG factor from `exp(log_prob - stop_gradient(log_prob))`, with its specified epsilon; the TF route preserves that finite-program convention. This investigation provides no evidence that the paper's estimator is fundamentally wrong.

One genuine translation bug was already repaired before these training runs: the author performs Adam scalar bias correction in Python binary64, while the former TF implementation used FP32 scalar arithmetic. The current code preserves the source's scalar precision and cast boundaries. That historical error cannot, by itself, explain failures observed after the repair.

## Confirmed numerical sensitivity

The author proposal forms a residual in the internal representation `[x,y,sin(theta),cos(theta)]`. Its `BearingsOnlyTransformer.apply_norm` normalizes the angle pair without an epsilon, then `forward_tranform` uses `atan2`. See author `src/models/internal_models/proposal_models.py:173–209` and `src/models/particle_transformer.py:112–158`; the corresponding TF operation is in `kdm_neural_replay_tf.py:152–190`.

Writing the raw angle pair as `(u,v)`, with `r = sqrt(u^2+v^2)`, gives

\[
\theta=\operatorname{atan2}(u,v),\qquad
d\theta=\frac{v\,du-u\,dv}{r^2},\qquad
\|\nabla\theta\|=\frac{1}{r}.
\]

Thus a nearly zero angle pair magnifies small upstream rounding differences. At the tested starting states, the minimum radius is about 0.84–0.89. At the trained states, it falls as low as 0.002944. For a measured unit01 example, the local sensitivity is about 339.64: the first-order prediction is -0.0001267503 radians and the observed difference is -0.0001267791 radians. This quantitatively explains that local angle discrepancy.

It does **not** show that the same mechanism causes the final 26–36% RMSE gap, or even that it accounts for every failed gradient comparison. The inspected formula is shared with the author, but radii at independently source-trained best checkpoints were not measured here. Adding an epsilon, clipping the angle pair or changing the proposal would alter the baseline and is not justified as a replication repair by this evidence.

## Experiments actually executed

Artifacts are under `artifacts/controlled-smc-kdm-replication-20260928/training-gap-20261012/`. Each attempt has a manifest, commands, source/data/checkpoint hashes and complete logs. Rows 384:512 form two batches of 64, with diagnostic seed 202610121 and two consecutive updates per case. Source-native realized initial particles, proposal noise and resampled values are replayed into TF. The FP64 control promotes the same FP32 inputs; it does not draw a different test problem.

| Attempt | State tested | Source-native/source-replay equality | FP32 screen | FP64 screen |
|---|---|---:|---:|---:|
| 01 | No numerical result | Configuration loader failed | N/A | N/A |
| 02 | Actual starting/best parameters for both units, fresh diagnostic Adam | 64/64 | 1650/1656 | 1656/1656 |
| 03 | Actual best parameters **and saved moments, step and rates**, both units | 32/32 | 821/836 | 836/836 |

Attempt01 failed because JSON serialization converted integer architecture dimension keys into strings. The reference loader now reconstructs the pinned YAML and verifies its JSON representation against the saved configuration. This was a diagnostic-harness failure before any algorithm comparison.

The predeclared coordinate screens are `128*2^-23*(1+abs(source))` for FP32 and `1e-9*(1+abs(source))` for FP64. Angles retain the existing geometric conditioning check. These are localization screens, not learned-quality acceptance margins. Counts include correlated arrays and repeated cases; they are not independent statistical trials. A supervisor `status: PASS` means execution completed: both result objects explicitly retain `numerical_screen_passed: false` and `replication: INCOMPLETE`.

Attempt02's six failures are three proposal-gradient arrays and their clipped copies. Attempt03 repeats those six and adds four gradient arrays, four clipped copies and one gradient norm at unit02's second update. Across attempt03:

| Quantity | Passed / checked | Largest absolute difference |
|---|---:|---:|
| Parameter arrays after the update | 248/248 | 3.73e-9 |
| Adam first moments | 248/248 | 8.49e-6 |
| Adam second moments | 248/248 | 2.38e-7 |
| Loss | 8/8 | 4.77e-7 |
| Adam step counts | 248/248 | 0 |
| Raw gradients | 241/248 | 8.49e-5 |
| Clipped gradients | 241/248 | 8.49e-5 |
| Gradient norm | 7/8 | 1.92e-4 |

All 16 explicit saved-state restoration checks passed. The largest failed gradient discrepancy is 2.414 times its local coordinate bound; all double-precision checks pass. These findings are consistent with arithmetic sensitivity at these states. They do not establish a uniform error bound or prove that downstream training distributions match.

The GPU comparison used the RTX 5080, TensorFlow `2.20.0-dev0+selfbuilt`, XLA enabled, TF32 disabled and verified memory growth. Attempt03 peak TF allocator use was 281,488,128 bytes. The pinned author ran with Torch `2.11.0+cu130` on CPU with GPU deliberately hidden. Source and TF optimizer restoration use their actual state APIs. The TF extension is diagnostic-only; cases without the optional saved-state field retain the existing comparison path.

Commands used the supervisor `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/run_kdm_current_training_gap.py`; attempt03 added `--saved-optimizer`. The supervisor records the exact isolated CPU and escalated GPU child commands. Base Git commit: `89065bc6354801cb368d5e163ba49fd9c4372d10`, branch `surrogate-hmc`; manifests also bind dirty/untracked source by SHA-256. Attempt03's terminal integrity review passed 37/37 checks, including loaded state, data/checkpoint/source hashes, syntax, GPU policy and the preservation of failed numerical checks. This integrity pass does not convert numerical failures into passes.

| Attempt | CPU seconds charged | GPU seconds charged |
|---|---:|---:|
| 01 | 6.533137 | 0.065614 |
| 02 | 14.449558 | 109.326674 |
| 03 | 9.636109 | 56.571450 |
| Total | 30.618804 | 165.963738 |

The three-attempt diagnostic is terminal, within its 1800-second cap per resource. Remaining campaign CPU is 220524.941126 seconds (61.26h), and the GPU tranche is 91577.543682 seconds (25.44h). No run is active. The larger aggregate GPU balance is not an additional allocation.

## What remains unresolved and the next discriminating test

The training curves locate the gap before final evaluation. Unit01's TF validation loss is actually lower at epoch199 (6.81169 versus source 7.01829), before its eventual selected checkpoint loses. Unit02 is initially similar but has a clear gap by epoch199 (6.82919 versus 6.27148). The complete selected curves and conditioning observations are preserved in `attempt02/trace-analysis.json`; saved-optimizer checks and budget totals are in `attempt03/trace-analysis.json`.

The full runs share initial constructor parameters and saved datasets, but pretraining is independent. Each MDPF stage correctly receives its **own** learned weight model. Its row order, normal/von-Mises noise and categorical sampling also follow independent native random streams. Close pretraining losses do not establish identical learned parameters. A full-run difference therefore mixes pretraining variation, stochastic trajectory variation, numerical sensitivity and any still-unidentified port error.

The two-update probes hold **realized samples** fixed. They can expose a local equation, derivative or optimizer error, but suppress feedback through future native resampling. Consequently neither the fresh-state nor saved-state pass in FP64 resolves cumulative training divergence.

The next justified experiment is a bounded controlled trajectory, not another full qualification ladder:

1. Start both implementations from one identical saved pretrained parameter state, with identical moments/rates and row order. Preserve the other independently pretrained states as separate diagnostic starts; do not select a favorable one for qualification.
2. Supply common **base random variates** to each source-equivalent sampler. Recompute samples from each implementation's evolving weights, bandwidths and particles. Replaying fixed completed resampled particles throughout is not an adequate native-trajectory comparison. Verify the diagnostic sampler adapters against the source before interpreting the trajectory.
3. Record the first divergence in raw proposals, angle radii, weights, resampling decisions, loss, gradients, moments and parameter updates. Add a source-versus-source tiny-perturbation control and an FP64 control to distinguish shared sensitivity from a TF-only operation mismatch. Perturbation magnitude is an explanatory design choice, not a new acceptance margin.
4. Repair only a demonstrated source deviation. If both source and TF show comparable sensitivity, retain stochastic uncertainty and reconsider the feasibility of qualification without relabeling this as a pass. If the first divergence is sampler-specific, run a focused sampler-law test before training further.

A fresh concise plan must specify the number of updates, input inventory, coupling semantics, perturbation scale, resource/attempt caps and stopping rules before that experiment. Existing campaign authorization permits this bounded investigation; no new full training was launched in the present task.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Keep KDM replication open | Local compatibility is strong but FP32 screens are not all passed; learned-quality criterion remains unresolved | No nonfinite update, wrong restored state or invalid replay; unfavorable trained outcomes retained | Cumulative stochastic/numerical effects versus an unlocalized implementation deviation | Controlled native-trajectory diagnostic as specified above | Replication complete, paper failure, or rounding proven to explain final RMSE |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new artifact/finite/state validity veto. FP32 comparison failures are repair/localization signals. |
| Statistically supported ranking | None; these are deterministic conditional comparisons and three descriptive training pairs. |
| Descriptive-only differences | Training losses, final quality gaps, angle radii and cross-backend error magnitudes. |
| Default-readiness | Unchanged and not established by this diagnostic. |
| Next evidence needed | An identified first native-trajectory divergence with source controls, followed by an informative untouched learned-quality comparison if feasible. |

Post-run red-team: the strongest alternative explanation is that correct but independently randomized optimization reaches different solutions; another is a port error outside these two replayed batches or one that emerges through cumulative resampling feedback. The weakest evidence is the very short trajectory coverage. A reproducible common-input FP64 discrepancy or a TF-specific native-sampler deviation would overturn the local arithmetic-sensitivity interpretation. No evidence here justifies changing the author's model or silently relaxing replication criteria.
