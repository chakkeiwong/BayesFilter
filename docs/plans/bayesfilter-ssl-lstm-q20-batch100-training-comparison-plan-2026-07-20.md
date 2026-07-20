# SSL-LSTM q=20 Batch-100 Training Comparison Plan

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `COMPLETED_TRAINING_DIAGNOSTIC`

## Research Question And Evidence Contract

Question: after training the same q=20 model with different batch sizes, do the
resulting admitted transports produce statistically equivalent out-of-sample
posterior-predictive laws? Equality of training loss, optimizer steps, or
training-draw exposure is not the target.

Comparator: the valid q=20 batch-480 seed-a artifact at
`docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/run-03/`.
It is a descriptive engineering baseline, not a posterior or predictive
oracle. The batch-100 transport must first pass the same transport and sampler
validity gates before entering a predictive comparison.

Candidate: one fresh q=20 seed-a stream with `batch_size=100`, the same fixed
parameters (`learning_rate=4e-4`, `initialization_scale=0.01`,
`gradient_clip_norm=10`), same validation seed/batch of 64, three-stage 32x32
dense IAF, 250-step validation cadence, 2,000-step maximum, and existing
saturation/support/controller rules.

Resource stop: 7,200 charged seconds. The measured batch-480 warm rate was
about 12.06 seconds per optimizer step; linear draw-count scaling gives about
2.51 seconds at batch 100 and 5,025 seconds for 2,000 steps. The cap is a
derived 43% envelope for compilation, validation, probes, and imperfect
scaling. It is not an expected runtime claim.

Primary predictive criterion (not executed by this training diagnostic): after
both batch-size arms pass transport and retained-HMC validity, use disjoint
out-of-sample forecast innovations/observations at horizons 1--10 and the
repository's dependence-aware proper-score/equivalence procedure. The primary
estimand is the batch-100 minus batch-480 vector of predictive means and
log-variances (or the predeclared proper-score loss), with simultaneous
confidence regions inside a prospectively calibrated practical-equivalence
region. The existing Phase 8 calibration was underpowered and did not freeze a
powered margin/sample design, so no formal equivalence claim is currently
admissible.

Training diagnostics: report optimizer steps, generated training draws, and
equivalent batch-480 draw blocks only as cost/optimization diagnostics. They
are not promotion criteria.

Hard vetoes: GPU memory-growth failure, nonfinite values, worker GPU visibility,
invalid checkpoint/support/roundtrip, host-memory breach, corrupt artifact, or
source/config mismatch. Saturation above the existing 0.05 cap is a candidate
stream veto and stops the controller before learning-rate repair, as in the
comparator; it is not evidence against q=20 or NeuTra. A training-loss miss or
single-seed result is not a predictive-equivalence result.

Explanatory diagnostics: heldout loss trajectory, saturation, support radius,
roundtrip residual, learning-rate actions, wall time, CPU-worker receipts, RAM,
GPU allocator telemetry, and training-draw exposure.

Nonclaims: this one-seed training run does not establish an optimal batch size,
epoch count, convergence, seed robustness, posterior correctness, HMC
readiness, predictive equivalence, statistical superiority, or
production/default readiness.

## Default And Assumption Audit

| Choice | Provenance | Risk and early check |
| --- | --- | --- |
| `batch_size=100` | User-requested hypothesis | More noisy gradients may saturate or require more steps; compare the complete trajectory and vetoes. |
| Batch-480 result as comparator | Existing valid artifact | It is one seed and an observed best, not a global optimum; no superiority claim. |
| Same seed and validation batch | Controlled comparison design | Any changed seed would confound batch size; manifest and stream binding check this. |
| 2,000 steps / 250 cadence | Existing controller protocol | May stop before an eventual match; report right-censored if no match. |
| One GPU, 16 CPU workers, XLA, float64 | Existing q=20 route | Resource and allocator receipts are mandatory; no CPU-only scientific interpretation. |

## Skeptical Pre-Execution Audit

- Wrong baseline: passed with qualification; the comparator is explicitly a
  single-run descriptive reference.
- Proxy promotion: passed; heldout loss and saturation nominate/veto training
  mechanics only. They cannot establish out-of-sample predictive equivalence.
- Missing stop: passed; hard veto, saturation, plateau, maximum-step, and
  resource stops are inherited and artifacts are resumable where supported.
- Unfair comparison: passed; target, initialization seed, validation batch,
  parameters, architecture, and cadence are held fixed.
- Hidden assumption: “epoch” is not a native concept for this online sampler;
  results will use optimizer steps and draw exposure instead.
- Artifact adequacy: fresh output root, manifest, progress, checkpoints, and
  result summary preserve the trajectory and exact command.

Audit decision: `PASS_FOR_ONE_BOUNDED_BATCH100_TRAINING_DIAGNOSTIC`; predictive
equivalence remains a later, separately powered phase.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
  --mode single-diagnostic --q 20 --batch-size 100 \
  --authorize-material-run --gpu-cap-seconds 7200 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-q20-batch100-training-comparison-2026-07-20
```

## Result And Close Record

The close record must state the training trajectory, draw exposure, stop reason,
hard veto status, and that no predictive equivalence was assessed. A later
predictive result must preserve separate training, sampler-validity, forecast,
and statistical-inference ledgers.
