# SSL-LSTM NeuTra Scale-vs-ELU Telemetry Result

Date: 2026-07-21  
Plan: `docs/plans/bayesfilter-ssl-lstm-neutra-scale-vs-elu-telemetry-plan-2026-07-21.md`  
Artifact: `docs/plans/artifacts/ssl-lstm-q20-scale-vs-elu-telemetry-2026-07-21/run-01/`  
Decision: `HISTORICAL_SCALE_HEAD_DIAGNOSTIC; SUPERSEDED_SATURATION_STOP_POLICY`

## Result

This historical bounded q=20 seed-a diagnostic completed on the existing `(32,32)`
three-stage dense IAF baseline. It was stopped by the unchanged aggregate
scale-saturation veto at validation step 750. The new telemetry separates the
two mechanisms under question:

- bounded scale saturation and raw scale-logit tails occurred in the same
  stages and at the same fractions;
- every hidden preactivation tail fraction was zero for all completed rows;
- hidden preactivation extrema remained far below the explanatory `|a| >= 5`
  threshold.

Within this one seed and fixed training policy, the evidence supports a
scale-head/optimization explanation for the observed veto. It does not prove
that ELU is harmless in every run, and it does not establish a remedy.

| Validation step | Mean loss | Stage 1 scale / logit tail | Stage 2 scale / logit tail | Stage 3 scale / logit tail | Hidden `|a|>=5` by stage | Aggregate scale saturation |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 79.759855 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 / 0 | 0.000000 |
| 250 | 43.929458 | 0 / 0 | 0.003906 / 0.003906 | 0.003906 / 0.003906 | 0 / 0 / 0 | 0.002604 |
| 500 | 42.517028 | 0 / 0 | 0.125000 / 0.125000 | 0 / 0 | 0 / 0 / 0 | 0.041667 |
| 750 | 42.271513 | 0 / 0 | 0.226563 / 0.226563 | 0 / 0 | 0 / 0 / 0 | 0.075521 |

The two fractions in each stage pair are equal because the implementation
uses the monotone relation

\[
s=s_{\max}\tanh(r/s_{\max}), \qquad
|s|\ge .95s_{\max}
\Longleftrightarrow
|r|\ge s_{\max}\operatorname{atanh}(.95).
\]

Here `s_max=1`, so the raw-logit threshold is `1.8317808231`. At step 750,
the global raw-logit range was `[-1.49296, 3.50451]`; the stage-2 positive
tail therefore drives the veto. Hidden preactivation ranges at the same rows
were:

| Validation step | Stage 1 min/max | Stage 2 min/max | Stage 3 min/max |
| ---: | ---: | ---: | ---: |
| 0 | `[-0.0894, 0.0696]` | `[-0.0613, 0.0525]` | `[-0.0855, 0.0728]` |
| 250 | `[-0.7038, 0.7494]` | `[-0.7623, 0.8037]` | `[-0.9193, 0.9638]` |
| 500 | `[-1.0250, 1.1584]` | `[-0.8099, 0.9437]` | `[-0.5745, 0.6259]` |
| 750 | `[-1.1792, 1.2745]` | `[-0.9809, 1.2432]` | `[-0.5993, 0.6384]` |

The diagnostic threshold was `|a|=5`; no hidden unit reached it. This is
explanatory evidence, not a statistical comparison across seeds.

## Decision And Inference Status

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain the telemetry implementation | Shapes, finite values, stage order, and unchanged transport path | No telemetry or serialization failure | One material run | Keep telemetry for the next controlled scale-head repair | No claim of production readiness |
| Reject this training candidate for admission | Existing saturation cap `0.05` | Aggregate saturation `0.075521` at step 750; result status `DIAGNOSTIC_VETOED` | One seed and fixed-smoke hyperparameters | Test a predeclared scale-head/optimizer repair | No HMC or posterior claim |
| Attribute this run's veto | Stage-level raw-logit and bounded-scale telemetry | Stage 2 tails coincide; hidden tails remain zero | Correlation does not prove causation | Compare a scale-head repair with the same target and seeds | No claim that ELU is universally correct or faulty |

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Candidate vetoed solely by `dense_scale_saturation_above_cap`; support, round-trip, finite-value, target-signature, reload, GPU/XLA, and resource checks passed. |
| Statistically supported ranking | None. One seed is descriptive evidence only. |
| Descriptive differences | Stage 2 scale logits grow from no tail at initialization to `22.6563%` at step 750; hidden tail fractions stay zero. |
| Default readiness | Not established; no default, activation, optimizer, or scale bound changed. |
| HMC readiness | Not evaluated and correctly withheld. |
| Scientific validity | Not evaluated; no posterior or predictive claim follows. |
| Evidence needed next | A controlled scale-head/optimizer repair with the same target, data, seed policy, and telemetry, followed by multi-seed downstream validation if the candidate passes training gates. |

This result rejected only the candidate under the training policy active on
2026-07-21 before the hierarchy repair. It
does not invalidate the target, the trainer harness, NeuTra as a direction, or
the use of ELU in other architectures. The observation is consistent with,
but does not by itself prove, a stage-specific scale-head optimization failure.

## Checks And Provenance

- Focused CPU-hidden tests passed: `50 passed`.
- The GPU/XLA run completed normally; the inner seed result correctly records
  `DIAGNOSTIC_VETOED`, while the outer single-diagnostic summary records normal
  process completion.
- Support checks were finite: round-trip maximum `2.6645352591e-15`, moderate
  shell inverse-radius maximum `4.000000000000003`, and finite transformed
  scores.
- No HMC process was launched.
- Charged wall time was `1203.9412701` seconds; no resource stop occurred.
- Run manifest records environment `tfgpu`, TensorFlow `2.20.0`, Python
  `3.13.13`, physical GPU `1`, XLA enabled, TF32 enabled, float64 tensors,
  16 CPU-hidden workers, and `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- The run manifest records git commit
  `41f2aa4f263d96e5575a6448d89bdd93bb262035` and
  `owner_designated_managed_session_visible_gpu_trusted`.
- The implementation diff was limited to read-only telemetry, runner
  summaries, focused tests, and this plan/result documentation. Forward map,
  log determinant, gradients, optimizer, scale bound, stopping rule, and HMC
  policy were not changed.

## Comparison Boundary

Earlier `(32,32,32)` and `(64,64)` q=20 runs also failed the same saturation
screen and showed repeated stage-2 concentration, but they did not record the
new raw-logit and hidden-preactivation telemetry. Historical Rotemberg records
showed seed-dependent scale saturation; they likewise are not activation
telemetry experiments. Those artifacts motivate the controlled repair but do
not provide a cross-model statistical ranking or prove that the present
scale-head mechanism is universal.

## Post-Run Red Team

The strongest alternative explanation is that the chosen fixed-smoke learning
rate or target score geometry causes stage-2 scale growth without requiring
hidden ELU saturation. A second alternative is that the finite validation
cloud under-represents a hidden tail; this is why raw min/max values and future
multi-seed/downstream checks remain required. A result with substantial hidden
tails in a matched repair would overturn the narrow attribution made here.

## Close Record

The telemetry work is complete and the artifact is replayable. Its immediate
saturation-stop policy has since been superseded by
`bayesfilter-ssl-lstm-neutra-training-hierarchy-plan-2026-07-21.md`, which
treats saturation as a repair trigger. Do not launch HMC from this historical
result, and do not promote an architecture or activation based on this
one-seed diagnostic.
