# P6 R3 Subplan: SIR-SGQF Target-Specific NeuTra Training

Date: 2026-07-16

Status: `COMPLETE_TRAINING_ADMITTED`

## Objective And Entry Conditions

Screen a target-specific plain dense-IAF family against the admitted
affine-only chart, select a viable recipe using common heldout evidence, and
run one fresh 5,000-step batched GPU/XLA training for typed `SIR-SGQF`.

Entry requires:

- typed target signature
  `0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`;
- comparator result SHA-256
  `621c3d6e748eed38433efaa02ff097a971132de89f323f12702533723e3ce9b2`;
- geometry result SHA-256
  `cbe82fe175991c549ed1c7c309a03a719be372e040ea755e358589deeb2c6d67`;
- verified remaining capacity in the 15-GPU-hour plain dense-IAF arm.

## Research Intent And Evidence Contract

| Field | Frozen contract |
| --- | --- |
| Question | Is a learned residual dense IAF engineering-valid and not meaningfully worse than the already strong affine-only SIR-SGQF chart on common heldout reverse-KL draws? |
| Baseline | target-specific Laplace affine transport with no learned residual |
| Mechanism | three residual dense autoregressive IAF stages composed before the fixed affine map |
| Expected failure | learned residual overfits/noisily degrades the near-Gaussian affine chart, or target/gradient/artifact health fails |
| Screen pass | 500 batched GPU/XLA steps complete; finite objective/gradient/logdet; all target status valid; frozen reload/parity passes; heldout values finite |
| Affine veto | learned paired mean reverse-KL minus affine control must be `<= 2` paired MCSE on eight identical heldout batches |
| Learned selection | among affine-nonworse recipes, choose any within two paired MCSE of the lowest learned mean, then lower parameter count, lower learning rate, declared order |
| Final pass | fresh 5,000-step run; exact identity; finite/status-valid records; GPU/XLA/batch-native telemetry; frozen reload and value/score parity; common-heldout finite and affine-nonworse |
| Hard vetoes | target/geometry/comparator/hash drift; CPU or non-XLA serious training; scalar/row-mapped fallback; Python sample/training-step loop; NumPy algorithmic path; nonfinite/status failure; parity/artifact failure; seed overlap |
| Explanatory only | loss trajectory, heldout objective/force, runtime, affine-control difference unless its predeclared veto fires |
| Not concluded | transported HMC convergence, posterior agreement, learned superiority, SGQF exactness, calibration, forecasting, robustness, or readiness |

Only R4 may move the cell from `TRAINING_ADMITTED` to `NEUTRA_CONFIRMED`.

## Frozen Recipe And Seed Ledger

All recipes use three IAF stages, ELU, `s_max=1`, initialization scale `0.02`,
batch size `128`, constant learning rate, clip norm `10`, manual Adam
`(0.9,0.999,1e-8)`, and one compiled `tf.while_loop` invocation.

| Recipe | Hidden layers | Learning rate | Screen seed |
| --- | --- | --- | --- |
| `dim3_lr1e3` | `(9,9)` | `1e-3` | `(20260716,31001)` |
| `dim3_lr5e3` | `(9,9)` | `5e-3` | `(20260716,31002)` |
| `wide_lr1e3` | `(18,18)` | `1e-3` | `(20260716,31003)` |
| `wide_lr5e3` | `(18,18)` | `5e-3` | `(20260716,31004)` |

Heldout root is `(20260716,31100)`, producing one identical `[8,128,3]`
stateless base tensor for the affine control and every recipe. Fresh final
training uses `(20260716,31201)`. None overlaps target, comparator, tuning,
warm-up, retained, or later R4 seed families.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| widths `9/18` | P4 widths scaled from six to three dimensions | preserves a 3x-dimension baseline and a 2x capacity arm | both are redundant or too weak | paired common-heldout screen and R4 | target-specific hypotheses |
| rates `1e-3/5e-3` | LGSSM/P4 successful bracket | tests slow versus aggressive fitting without post-hoc search | instability or under-training | 500-step finite/status traces | inherited hypotheses, not defaults |
| three stages | repository plain dense-IAF family | matches the only completed reusable family | needless capacity for near-Gaussian target | affine-control veto | baseline hypothesis |
| 500 screen steps | P4 target-specific protocol | bounded nomination rung | misranks long training | proxy-only status and fresh final run | budgeted screen |
| 5,000 final steps | master program common rung | required common serious candidate | excessive training degrades affine chart | final heldout affine veto and R4 | required campaign rung |
| batch 128 | completed batched GPU protocol | stable throughput and common comparison | noisy gradient/heldout | eight common heldout batches | reviewed baseline |
| constant learning rate | P4 implementation | isolates rate and capacity | late-step oscillation | full finite records and final heldout | hypothesis |
| affine-only control | admitted SIR Laplace geometry and comparator | detects needless learned degradation | control metric itself is noisy | paired identical base draws | required baseline |

## Required Artifacts And Checks

1. Reconstruct and require the repository-issued typed identity before every
   job; verify comparator, geometry, and recursive hashes.
2. Enforce TensorFlow memory growth before logical initialization, physical
   GPU placement, XLA, float64, TF32 telemetry, and fail-closed device checks.
3. Use `train_campaign_neutra`; require batch-native target binding,
   `sample_axis_python_loop_used=false`, `row_mapped_scalar_target_used=false`,
   `scalar_fallback_used=false`, and `compiled_training_control_flow=tf_while_loop`.
4. Freeze each screen artifact and verify trainable/frozen forward, logdet,
   pullback-score, and logdet-score parity at `<=1e-10`.
5. Evaluate the affine control and all frozen candidates on identical compiled
   heldout draws; write paired values and MCSE, not just means.
6. Finalize only after all four screen artifacts and hashes exist. Screen
   weights must never initialize or resume the final job.
7. Hash final state, payload, progress, result, manifest, and all recursive
   artifacts. Run focused tests, syntax, policy scan, and diff check.

No external replay dataset is applicable: reverse-KL training draws standard
normal base noise inside the single compiled GPU program. This is in-graph
training noise, not external model-sample generation.

## Repair, Handoff, And Stop Conditions

Localized serialization, manifest, import-order, or XLA resource defects may
be repaired in a fresh root under unchanged target, recipes, seeds, criteria,
hardware class, and arm budget. A failed recipe does not stop remaining frozen
recipes. Zero affine-nonworse learned recipes yields `RECIPE_REJECTED` for the
plain family and hands off to P7 with no R4 learned confirmation. Final
engineering pass yields `TRAINING_ADMITTED` and requires a fresh R4 subplan.

Stop for identity/hash corruption, unavailable trusted GPU, shared trainer
contamination, scalar/unbatched fallback, all-recipe target invalidity, or
15-GPU-hour arm exhaustion. Do not change topology/rate after seeing outputs;
that requires a refreshed enhanced-family subplan.

Attempt routing after the pre-step telemetry-interface failure is
`dim3_lr1e3/attempt-02`; the other screen recipes and the eventual final recipe
use `attempt-01`. The trace-only first attempt is preserved and excluded from
screen finalization.

## Skeptical Audit

Decision: `PASS`.

The plan uses the admitted same-target comparator and target-specific affine
baseline, does not treat P4/LGSSM recipes as scientific evidence, checks the
near-Gaussian failure mode before the expensive final rung, keeps heldout loss
as a nomination/veto diagnostic, requires downstream R4, separates all seeds,
and preserves the bounded arm budget. Commands and artifacts answer the stated
engineering question without implying sampler or scientific validity.

Claude review is unavailable for this private-workspace plan because the
platform previously blocked the same one-path disclosure in P5/P6. Repeating
the blocked external disclosure is not a scientific gate under the active
review-proportionality policy. This limitation is recorded; no Claude verdict
is claimed.
