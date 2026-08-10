# SSL-LSTM NeuTra Training-Hierarchy Repair Result

Date: 2026-07-21  
Plan: `docs/plans/bayesfilter-ssl-lstm-neutra-training-hierarchy-plan-2026-07-21.md`  
Artifact: `docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-02/`  
Decision: `LOSS_FIRST_HIERARCHY_SUPPORTED; COMPLETE_ONE_SEED_DIAGNOSTIC`

## Result

The completed q=20 `(32,32)` seed-a run resumed from the verified step-1250
checkpoint, continued through the scale-saturation threshold, applied the
planned learning-rate repairs, and stopped at the configured maximum step.
Saturation did not terminate the run or label an improving checkpoint invalid.

| Validation step | Mean loss | Aggregate saturation | Controller action | Repair trigger | Learning rate | Best step | Stop reason |
| ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| 0 | 79.759855 | 0.000000 | `initialize_best` | none | 0.000400 | 0 | none |
| 250 | 43.929458 | 0.002604 | `improved` | none | 0.000400 | 250 | none |
| 500 | 42.517028 | 0.041667 | `improved` | none | 0.000400 | 500 | none |
| 750 | 42.271513 | 0.075521 | `improved_and_reduce_learning_rate_for_saturation` | `scale_saturation_above_cap` | 0.000200 | 750 | none |
| 1000 | 42.095882 | 0.076823 | `improved` | `scale_saturation_above_cap` | 0.000200 | 1000 | none |
| 1250 | 41.920834 | 0.076823 | `improved_and_reduce_learning_rate_for_saturation` | `scale_saturation_above_cap` | 0.000100 | 1250 | none |
| 1500 | 41.835890 | 0.078125 | `improved` | `scale_saturation_above_cap` | 0.000100 | 1500 | none |
| 1750 | 41.734811 | 0.080729 | `improved_and_reduce_learning_rate_for_saturation` | `scale_saturation_above_cap` | 0.000050 | 1750 | none |
| 2000 | 41.724647 | 0.080729 | `stop` | `scale_saturation_above_cap` | 0.000050 | 1750 | `maximum_steps_reached` |

The previous telemetry-only run stopped at step 750 with
`scale_saturation_above_cap`. Under the repaired hierarchy, the completed
run-02 continuation crossed the same threshold three times, retained each
paired-loss improvement as the best state, and reached the maximum step.

This establishes the first engineering question only: the policy can continue
past saturation and execute the repair deterministically. It does not establish
that the resulting transport is converged, useful for HMC, or scientifically
correct.

## Loss-First Follow-Up

After inspecting the prior run-01, the controller was corrected so saturation does not
make a row ineligible for loss-based best-state selection. A saturated row may
both improve the paired validation objective and trigger the learning-rate
repair. A fresh GPU/XLA follow-up was launched at
`docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-02/`.
The first process ended after writing step 1250 but before its terminal export;
that pre-recovery state was retained as an orphan diagnostic and was not
interpreted as a completed result.

| Validation step | Mean loss | Aggregate saturation | Paired mean delta vs prior best | One-sided upper bound | Action | Best step | Learning rate |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 500 | 42.517028 | 0.041667 | -1.412430 | -0.947473 | `improved` | 500 | 0.000400 |
| 750 | 42.271513 | 0.075521 | -0.245515 | -0.134182 | `improved_and_reduce_learning_rate_for_saturation` | 750 | 0.000200 |
| 1000 | 42.095882 | 0.076823 | -0.175631 | -0.123404 | `improved` | 1000 | 0.000200 |
| 1250 | 41.920834 | 0.076823 | -0.175047 | -0.129814 | `improved_and_reduce_learning_rate_for_saturation` | 1250 | 0.000100 |

Every displayed one-sided upper bound is below zero. Under the declared paired
loss rule, each row is therefore an objective improvement relative to the
previous best despite saturation. The controller did not veto or discard any
of these rows. This is the direct evidence that saturation must not be the
sole training criterion.

The run-02 continuation was subsequently resumed from the verified step-1250
checkpoint and completed at step 2000. It wrote a terminal result, best-state,
and frozen-transport export. The resumed result is complete as a one-seed
mechanism diagnostic, but it remains outside the two-seed Phase 3 admission
contract.

## Completed Loss-First Follow-Up

The orphaned run-02 progress file was recovered only after verifying the
progress-bound checkpoint file hash, checkpoint payload hash, trainer step,
controller observation step, stream, and history step. A trusted GPU preflight
reproduced the parent target signatures (`302d50...` and `941145...`). The
CPU-hidden workers produced different signatures (`e920ec...` and `06ada5...`)
because the signature includes device-dependent synthetic-observation bytes;
fixed-point values and analytic scores agreed to displayed precision. This is
not evidence of post-run source drift.

| Validation step | Mean loss | Aggregate saturation | Action | Best step | Learning rate |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1250 (resume source) | 41.920834 | 0.076823 | `improved_and_reduce_learning_rate_for_saturation` | 1250 | 0.000100 |
| 1500 | 41.835890 | 0.078125 | `improved` | 1500 | 0.000100 |
| 1750 | 41.734811 | 0.080729 | `improved_and_reduce_learning_rate_for_saturation` | 1750 | 0.000050 |
| 2000 | 41.724647 | 0.080729 | `stop` | 1750 | 0.000050 |

The terminal reason was `maximum_steps_reached`, not saturation. The best
checkpoint was step 1750. The paired best-minus-initial mean difference was
`-38.025044` with one-sided upper bound `-26.278818`; this is descriptive
within-run objective evidence, not a replicated method comparison.

## Policy Evidence

| Evidence class | Status |
| --- | --- |
| Saturation validity | Saturation was not a mathematical-validity failure. It was recorded at every affected checkpoint and used as a repair trigger. |
| Repair execution | Passed in completed run-02: repairs occurred at steps 750, 1250, and 1750 while retaining the improving best states. |
| Continuation | Passed: run-02 resumed from step 1250 and reached step 2000. |
| Ordinary stop | Passed. Terminated at `maximum_steps_reached`, not a saturation stop. |
| Numerical validity | Passed for the exported best artifact: finite support probe, round-trip maximum `2.6645352591e-15`, moderate-shell inverse-radius maximum `4.000000000000003`, and finite transformed scores. |
| Serialization/reload | Passed; the frozen best payload was loaded and probed. |
| Resource/GPU policy | Passed. Trusted GPU/XLA execution completed without resource stop; memory growth was enabled before TensorFlow device initialization. |
| Statistical ranking | None. The paired bounds are within-run loss evidence, not a method ranking; this remains one seed and a policy test. |
| Admission/HMC | Not evaluated. `phase3_admission_status` remains `NOT_EVALUATED_ONE_SEED`; no HMC was launched. |

## Interpretation

The completed run-02 supports the following narrow statement:

> The previous saturation threshold was a conservative training-policy stop,
> not a transport-validity condition. Treating it as a repair trigger allows
> the optimizer to continue while retaining the last hard-valid best state.

The result does not support the stronger statements that saturation is always
harmless, that the half-rate repair is optimal, or that the transport is ready
for posterior sampling. The one-seed run shows continuing objective
improvement while saturation remains above the diagnostic threshold, but this
is evidence about the loss-first policy only, not evidence of convergence.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept hierarchy implementation | Saturation crossing produced repair and continuation | No implementation, finite-value, support, round-trip, reload, GPU, or resource veto in completed run-02 | One seed; no downstream predictive validation | Test seed-b under the same policy, then run downstream moment checks | No default-policy or scientific promotion |
| Retain saturation as diagnostic | Stage telemetry remains recorded | Saturation is no longer a hard veto | Threshold `0.05` is inherited and uncalibrated | Use stage-local telemetry to guide scale-head repair | No claim threshold is statistically grounded |
| Reject immediate saturation stop | New run continued to step 2000 | Historical stop was superseded for this active path | Later training may still fail for another reason | Keep ordinary plateau/max/resource/numerical boundaries | No claim all future candidates will survive |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | No hard veto in the new run; exported best artifact passed finite/support/round-trip/reload checks. |
| Statistically supported ranking | None; one seed and no replicated uncertainty analysis. |
| Descriptive-only differences | Run-02 reached step 2000, with loss decreasing from 79.759855 to 41.724647 and three repairs. These are descriptive policy-run differences. |
| Default readiness | Not established. The hierarchy is an active implementation behavior, not a certified production default. |
| HMC readiness | Not evaluated and correctly withheld. |
| Next evidence needed | Repeat with seed-b, then test downstream predictive moments and HMC only after the reviewed multi-seed training gate. |

## Provenance And Checks

- Command used:
  `TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 32,32 --authorize-material-run --gpu-cap-seconds 7200 --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json --output-root docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-01`
- Charged wall time: `1876.7747034` seconds; no resource stop.
- Environment: `tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, TF32 enabled,
  XLA enabled, float64 transport tensors, physical GPU `1`, 16 CPU-hidden
  workers, host RAM cap 64 GiB.
- GPU trust basis:
  `owner_designated_managed_session_visible_gpu_trusted`.
- The artifact run manifest records git commit
  `41f2aa4f263d96e5575a6448d89bdd93bb262035` and `git_dirty=true`, because
  the hierarchy implementation was intentionally under active local review.
- The artifact's exact runtime source binding records the older single-seed
  diagnostic plan path; the hierarchy plan above is the governing plan for the
  policy edit and result interpretation. This distinction is preserved rather
  than rewritten after the run.
- CPU-hidden focused suite after the implementation change: `74 passed`.
- q=20 contract smoke: `PASSED`.
- `py_compile` and `git diff --check`: passed.
- No HMC process was launched.
- Run-02 continuation command:
  `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 32,32 --authorize-material-run --gpu-cap-seconds 3600 --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json --output-root docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-02 --resume`
- Run-02 charged wall time: `1160.844024` seconds; selected physical GPU `1`,
  parent XLA enabled, workers CPU-hidden, TF32 enabled, and host RAM cap 64
  GiB. The run manifest records `trust_basis=
  owner_designated_managed_session_visible_gpu_trusted` and `git_dirty=true`.
- Run-02 terminal artifacts: `seed-a/result.json`, `seed-a/best-state.json`,
  `seed-a/best-frozen-payload.json`, and `seed-a/checkpoint-2000.json`.
- Run-02 result SHA-256: `49465278fcf2266450c6ce0d6c6307172f777072edad93989e6259779a6d1ff1`.
- Target-signature parity preflight passed on GPU; fixed-point GPU/CPU worker
  values and scores matched to displayed precision. The exact CPU/GPU hashes
  are not interchangeable because the signature currently hashes device-
  dependent observation bytes.

## Post-Run Red Team

The strongest alternative explanation is that the lower loss in run-02 is a
validation-cloud or target-geometry effect rather than a generally useful
training improvement. The one-seed result also cannot establish robustness;
seed-b could cross the threshold earlier, plateau differently, or become
numerically invalid. The device-dependent signature construction remains a
reproducibility weakness to repair, although fixed-point target parity was
observed here. None of these uncertainties restores saturation's authority as
a mathematical validity veto.

## Close Record

The active q=20 complexity runner now treats saturation as a repair trigger,
not a validity veto, and evaluates paired loss before repair decisions. The
orphan-resume path was exercised successfully and produced a complete one-seed
diagnostic. The next smallest discriminating action is the same policy on
seed-b, followed by downstream predictive-moment validation. HMC remains
withheld until the multi-seed training and downstream validation gates are
separately met.
