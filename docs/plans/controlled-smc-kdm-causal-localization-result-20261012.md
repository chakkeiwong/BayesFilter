# KDM causal localization: completed bounded diagnostics

Updated 13 October 2026 after terminal review. No experiment is running.
The reviewed initial six-launch phase and conditional two-launch saved-case
phase have finished. The broader replication program remains incomplete.

The saved crossed-start failure is caused by a resampling boundary: a one-ULP
difference in a cumulative weight changes one ancestor, which changes the
subsequent gradient and update. A signed perturbation in the author engine
reproduces the TF result. This closes the cause of this particular diagnostic
case; it neither explains every earlier gradient flag nor establishes final
trained-quality equivalence.

## Source, computation and evidence

The comparator is the released Younis author implementation, pinned at
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, with saved unit01/unit02 data,
constructors and checkpoints. The TF implementation uses GPU/XLA with TF32
disabled for these arithmetic-reference comparisons. Author PyTorch runs
are CPU-only independent references. All visible TF devices have verified
memory growth. The original native categorical random mapping is replaced
only in the common-input diagnostic by an explicit inverse-CDF mapping.

The actual claim-bearing diagnostic call chain is
SourceFullTraining.update -> BearingsNeuralReplay._sequence ->
sampler._sample -> SourceKDMResampler._assemble. The repaired crossed-start
harness and saved-case regression call the same
make_observed_coupled_sampler factory. The sequence records the ancestors,
uniforms and CDFs consumed inside that compiled update. The optional capture
flag defaults to false; the ordinary runtime's numerical settings are unchanged.

For the diagnostic coupling, the selected component is

    a = min(sum_i 1[u >= C_i], N - 1),  C_i = sum_{j<=i} normalized_weight_j.

This function is discontinuous at C_i. A small change in C_i need not imply a
small change in the selected particle. The equation describes the inspected
diagnostic code, not a claim of equality between native TF and PyTorch RNGs.

Core code:

- [actual sequence capture](../../bayesfilter/replication/kdm_neural_replay_tf.py:192);
- [shared diagnostic sampler](../benchmarks/diagnose_kdm_crossed_starts_tf.py:25);
- [saved-case counterfactual](../benchmarks/diagnose_kdm_saved_case_intervention.py);
- [CPU author operation](../benchmarks/reference_kdm_crossed_starts.py);
- [terminal audit](../benchmarks/diagnose_kdm_causal_terminal_audit.py).

## Completed comparisons

| Test | Coverage | Checked result | Remaining limit |
|---|---|---|---|
| Fresh weight pretraining | 324 update/validation comparisons, two constructors, both precisions | Both FP64 trajectories and unit02 FP32 agree; unit01 FP32 has 19 flagged steps, including gradient index 67 | First feature flag index 59 clears on identical author pre-state; later gradient at index 67 was not separately replayed; unit01 validation was skipped after flags |
| Crossed starts | Four author/TF starts,16 updates each, both precisions:128 comparisons | All FP64 comparisons agree; both author starts agree in FP32; unit01 TF start diverges at update2 | Short trajectories do not qualify final trained quality |
| Saved-history controller | 12 actual histories plus two fixtures;3,278 checks | Identical losses give identical source/TF LR, best-save and stopping decisions | Different noisy histories may still change checkpoint selection; historical live optimizer wiring is not proved by history replay |
| Common checkpoint evaluation | Eight checkpoint roles,eight draws,both precisions:128 records | The checkpoint quality difference persists under the same author evaluator | First64 validation rows and selected times are a descriptive development panel |
| Direct saved-case intervention | Two launches,2,358 checks each | Capture-off exactly reproduces the original full result; capture-on leaves numerical outputs unchanged; forcing first source ancestor clears gradient/update discrepancy | One selected boundary case, not final training or native RNG equivalence |
| Terminal artifact audit |155 checks | Source snapshots, pin, resource accounting, numerical intervention and trace counts pass | No statistical qualification or wider paper-scope claim |

Fresh pretraining compares actual quantized FP32 constructor/data/cloud inputs
promoted exactly to FP64. Source/TF optimizer groups, moments, update counters,
weight decay and reduction ordering are preserved. All scalar losses and
updated parameters pass the declared diagnostic tolerance. Unit01 features
first flag at index 59; a separate gradient flag occurs at index 67. These flags
are localization triggers, not learned-quality thresholds.

The four crossed-start roles each pass the source/adapter controls; all 140
adapter checks pass. Unit01 TF-start FP32 has 14 flagged updates beginning at
index 2. FP64 agreement on the same realized base arrays rules out a large
algebraic mismatch on this tested scope, but does not qualify native training.
The old unit02 TF-start ancestry-only flag at update 15 was reconstructed
outside the kernel and remains provisional.

## Causal saved-case result

All indices below are zero based, except the sequence time label retained
from the filter. At update index 2 from the unit01 TF-pretrained start,
both engines are restored to the same author pre-update state.

| Consumed quantity at time 4, row 52, slot 1 | Author | TF |
|---|---:|---:|
| Uniform u |0.7998569011688232|0.7998569011688232|
| CDF at component 19 |0.7998569011688232|0.799856960773468|
| Selected component |20|19|

The uniform equals the author CDF boundary, so the >= convention selects 20.
The TF CDF is one FP32 ULP higher, so TF selects 19. No comparator convention
was changed between engines.

| Intervention relative to unperturbed author | Absolute loss difference | Maximum gradient difference | Maximum updated-parameter difference | Subsequent ancestor disagreements |
|---|---:|---:|---:|---:|
| Natural TF diagnostic |0.00731658935546875|0.03083752096|0.0001256465912|142|
| Force only the first differing source ancestor |0|2.384185791e-7|5.960464478e-8|0|
| Force every source ancestor |0|2.384185791e-7|5.960464478e-8|0|

Forcing only the first ancestor is sufficient. This is a counterfactual
diagnostic, not an eligible replacement filter.

The source-self panel retained both signs and zero-effect results:

| Requested perturbation of the one source uniform | Realized FP32 change | Effect |
|---|---:|---|
|0|0|Exact repeat|
|+/-1e-8|0|No representable change, no effect|
|-1e-7|-1.192092896e-7|Same loss jump and 142 ancestor changes as natural TF|
|+1e-7|+1.192092896e-7|No branch change|
|-1e-6|-1.013278961e-6|Same loss jump and 142 ancestor changes|
|+1e-6|+1.013278961e-6|No branch change|
|-1e-5|-1.001358032e-5|Same loss jump and 142 ancestor changes|
|+1e-5|+1.001358032e-5|No branch change|

The terminal audit also compared the full numerical arrays, not just their
maxima. Natural TF versus author perturbed by -1e-7 agrees exactly in loss,
within 2.980232239e-7 in gradients, 5.960464478e-8 in updated parameters and
2.235174179e-8 in first/second moments. The observation is therefore consistent
with a shared discontinuity of resampling. It is not evidence that the paper
or translated IWSG backward rule is wrong.

A boundary selected because it already disagreed cannot estimate the frequency
of such events or define an acceptable final-error envelope. The earlier
small-radius angle case has a different mechanism and remains separately open.

## Common evaluation and selection

The same author engine evaluated both saved final-best checkpoints on the
fixed 64-row development panel, times 4/8/12/16, with eight common draw sets.

| Unit | Author checkpoint NLL | TF checkpoint NLL | NLL increase | Author XY RMSE | TF XY RMSE | Relative RMSE increase |
|---|---:|---:|---:|---:|---:|---:|
|01|5.96516|6.70654|0.74138 nat|2.93289|3.86663|31.84%|
|02|6.13914|6.79465|0.65550 nat|2.86071|4.16243|45.50%|

These are descriptive results for selected checkpoints. They establish that
the observed gap is present in the learned checkpoints under one evaluator;
a TF-only metric implementation error cannot explain this panel. They do
not identify whether training paths, initialization or noisy validation
selection produced the gap, and they do not supply independent training units.

Across these eight draws, source-engine NLL standard deviations range from
about 0.075 to 0.171 across the four final-best roles. The panel uses 64 rows,
while native validation uses the full split. Comparing this small-panel
spread directly with historical full-validation checkpoint margins would
be wrong. Intermediate near-best checkpoint states were not preserved by
this execution. Selection sensitivity therefore remains unmeasured.

## Repairs and review

Attempts 01–04 failed before numerical updates: isolated Python import path,
wrong plan filename, JSON conversion of integer cloud dimensions, and an
assertion that incorrectly included a data-loader worker override. All were
preserved and charged. The Adam diagnostic collector was repaired to retain
each parameter's own optimizer state rather than overwrite other groups with
zeros. Pretraining now exposes already-computed features/logits/NLL.

Terminal review found that attempt06 recomputed ancestors/CDFs outside the
consuming XLA update. The recomputed CDF could be inconsistent with the
consumed ancestor at one-ULP boundaries. The repaired harness now captures
actual values; saved-case attempt02 exercised the shared sampler factory and
reproduced attempt01. Preserve all attempt06 results; treat reconstructed
ancestry/boundary fields as provisional. Its measured loss, gradient, sampled
particle and parameter outputs remain valid.

Each exercised update/helper signature traced once at most. GPU/XLA compiled
and executed successfully. Python orchestration loops select independent
diagnostic runs; the repeated filter kernel uses tf.while_loop. No pfor or
NumPy numerical path was introduced into the TF execution. This result does
not audit every repository route for retracing or compilation behavior.

## Decision and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this saved boundary's local cause | First-ancestor intervention and source-self reproduction pass | No additional invalid output or failed adapter control | Frequency and long-run consequence unknown | Preserve intrinsic source sensitivity; do not change the algorithm | All gradient gaps are resolved |
| Keep KDM learned-quality replication open | Existing0.01nat/1% criteria not established | No final noninferiority evidence | Few independent units; unfavorable checkpoint panel | Localize remaining gradient cases, then resolve selection and plan independent native confirmation | KDM is no worse, or the paper is faulty |
| Retain controlled-SMC bounded verdict | Existing tested-input/failure-fidelity evidence unchanged | No new finding against that result | Wider population/paper coverage separate | Do not reopen retired 1%/5% planning rule | Whole-paper or population certification |

| Inference status | Finding |
|---|---|
| Hard veto screen | Completed runs have finite valid outputs and passing controls; early harness failures are preserved |
| Statistically supported ranking | None; the evaluation panel is diagnostic |
| Descriptive-only differences | Checkpoint NLL/RMSE gaps and draw-to-draw spread |
| Default readiness | Unchanged; no scientific numerical default promoted |
| Next evidence needed | Same-state replay of pretraining gradient at index 67 and earlier trained small-radius case; native-scale validation selection experiment; independent training-unit precision design |

Remaining ordered work:

1. Restore unit01 pretraining index 67 to the identical author pre-state and
   replay its captured input. If the gradient flag disappears, record accumulated
   drift; if it persists, capture the first differing activation/backward rule.
   The index 59 feature replay cannot answer this later gradient question.
2. Replay the earlier trained-state first-gradient case independently of this
   resampling case. Use ordinary and small-radius cases, signed perturbations,
   actual representable changes and saved adjoints. Preserve IWSG stop-gradient
   semantics; an ordinary finite difference that recomputes detached terms is
   not a valid test of its intended gradient estimator.
3. Measure validation-selection sensitivity at the native validation size.
   Preserve intermediate near-best checkpoints in a bounded training window,
   then compare independent and shared validation draws with training fixed.
   Identical-history controller agreement is already established; do not repeat
   it as a substitute for this question.
4. If a translated operation is wrong, apply a source-grounded repair and then
   check a fresh native unit. If only shared stochastic/numerical variation
   remains, design the independent training-unit study around the unchanged
   acceptance criteria and remaining budget. Repeated evaluation draws do not
   count as independent training replications. Report under-budgeted inference
   rather than silently loosening criteria.
5. Preserve selected paper-experiment coverage as a separate checklist; these
   bearings diagnostics do not complete untested Appendix B/A-MDPF or other
   declared experiment coverage.

Post-run red team: the strongest remaining alternative is that ordinary
stochastic training and validation selection explain the checkpoint gap despite
local arithmetic agreement. The opposite possibility, an untested translation
defect elsewhere in training, is not excluded. A reproduced same-state
non-boundary gradient error would overturn a blanket shared-sensitivity account.
The weakest evidence is still the number of independent training units and the
absence of a downstream causal intervention. Only the saved boundary case is
causally closed.

## Reproducibility and resource accounting

Exact commands:
- /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B /home/chakwong/BayesFilter/docs/benchmarks/run_kdm_causal_localization.py
  (stage arguments and command arrays are preserved in each manifest).
- /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B /home/chakwong/BayesFilter/docs/benchmarks/run_kdm_saved_case_intervention.py
- /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B /home/chakwong/BayesFilter/docs/benchmarks/diagnose_kdm_causal_terminal_audit.py

Base commit89065bc6354801cb368d5e163ba49fd9c4372d10 on surrogate-hmc.
The worktree is dirty; each numerical run preserves exact source snapshots and
SHA-256 hashes. Unrelated edits are untouched. GPU reference environment:
tftwogpu, TensorFlow 2.20.0-dev0+selfbuilt; source reference: tf-gpu2 with GPU
hidden. Actual data/checkpoint hashes, command, seed, dtype, device policy and
timing are in the run manifests and diagnostic provenance. Saved-case seed
is 202610123; source/TF arrays are hash-checked after exact promotion.

Initial attempts 01–06 charged 1,529.189673 seconds per resource. Saved-case
attempts 01/02 charged 115.526243/184.609773 seconds. Total 1,829.325689 seconds
(30.49 minutes) charged conservatively to EACH CPU/GPU resource, within the
original 4-hour resource cap. Routine editing/report inspection is separate.
Campaign balance: CPU 217,986.948231 seconds (60.55h);
active GPU tranche 89,039.550787 seconds (24.73h).

Artifact root:
[causal-localization-20261012](artifacts/controlled-smc-kdm-replication-20260928/causal-localization-20261012).
Decisive files:
[final saved-case result](artifacts/controlled-smc-kdm-replication-20260928/causal-localization-20261012/saved-case/attempt02/diagnostic/result.json),
[terminal audit](artifacts/controlled-smc-kdm-replication-20260928/causal-localization-20261012/terminal-audit.json),
attempt05/diagnostic/result.json, attempt06/diagnostic/result.json and
attempt06/diagnostic/terminal-evaluation-summary.json.
Neither exhausted launch cap silently authorizes a new launch. The next phase
needs a bounded implementation/specification within the existing campaign
authorization; no new human approval is implied for routine local continuation.
