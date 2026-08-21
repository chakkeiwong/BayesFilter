# SIR Classifier-Ratio Score Result And Reset Memo

Date: 2026-08-13  
Status: `EXACT_ORACLE_FAILED__SIR_NOT_LAUNCHED`

## Direct Verdict

The requested filter-independent method was implemented and executed as
balanced observation-only classifier likelihood-ratio estimation. It did not
pass its required exact Gaussian calibration gate. Therefore no SIR score at
`T=20`, `T=40`, or `T=50` was computed or admitted.

This is not a substitution with the earlier Fisher/path-importance method. The
campaign process loaded only the lightweight classifier, Gaussian observation
simulator, SIR observation simulator, TensorFlow, and GPU-memory policy. It did
not load a state-estimation implementation. The only score expression in the
implementation is the calibrated classifier logit at the fixed observation
divided by `2*epsilon`.

## Claimed Target Versus Quantity Actually Computed

| Item | Classification |
|---|---|
| Claimed final target | `d/dtheta log p_theta(y_obs)` for the fixed SIR observation prefixes at `T=20/40/50` |
| Quantity actually computed | Classifier-ratio score estimates for an independent exact Gaussian full-path location/log-scale family |
| Relationship | The Gaussian result tests the same balanced-classifier identity and complete calibration/extrapolation procedure, but it is not an SIR score |
| Verdict | The generic procedure is not admitted because 8 of 9 Gaussian horizon/coordinate cells failed |
| SIR result | `NOT_RUN`; reporting an SIR reference after this veto would be unsupported |

## Execution Record

Environment and hardware for both full attempts:

- interpreter: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`;
- GPU: visible-device slot `1`, realized as NVIDIA GeForce RTX 5080;
- TensorFlow memory growth: configured and verified before logical-device
  initialization;
- XLA: enabled; the run log confirmed an XLA-compiled cluster;
- TF32: disabled for the calibration campaign;
- observation paths only, full paired prefixes of shapes `[T,9]`;
- per-class counts per head: train `2048`, validation `512`, calibration `512`,
  untouched test `1024`;
- independent final fits per epsilon: `3`;
- epsilons: `0.01, 0.02, 0.04, 0.08`;
- coordinates: two Gaussian mean directions and one log-scale direction.

Commands:

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_classifier_ratio_score_tf.py

/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu env \
  TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
  XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  MPLCONFIGDIR=/tmp/bayesfilter-matplotlib \
  bash scripts/run_sir_classifier_ratio_score_gpu.sh exact-full \
  docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/exact_full_attempt01

/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu env \
  TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
  XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  MPLCONFIGDIR=/tmp/bayesfilter-matplotlib \
  bash scripts/run_sir_classifier_ratio_score_gpu.sh exact-full \
  docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/exact_full_attempt02_centered_quadratics
```

Focused verification after the repair: `12 passed`. Attempt 01 wall time was
`221.60 s`; attempt 02 wall time was `245.44 s`. Both preserve structured
manifests, per-head rows, architecture-selection evidence, and terminal results.

## Repair Between Attempts

Attempt 01 exposed a plan/implementation mismatch. The reviewed architecture
specified centered squared standardized features, but the code used uncentered
squares. That offset can saturate tanh layers for a path-dimensional variance
ratio. The only repair was `z**2 -> z**2 - 1`, with a focused regression test.
Data, splits, architectures, optimizer, regularization, epsilon ladder,
calibration, score identity, and gates were unchanged. The failed first attempt
was not overwritten.

The repair materially improved the log-scale task, but it did not clear the
campaign gate.

## Exact-Oracle Results After Repair

An epsilon is counted only when all three independent heads at that epsilon
pass every frozen diagnostic. At least three admitted epsilons are required
before extrapolation.

| Horizon | Coordinate | Admitted epsilons | Exact score | Extrapolated score (SE) | Verdict |
|---:|---|---:|---:|---:|---|
| 20 | mean direction 0 | 1 | 18.8999 | N/A | fail: fewer than 3 epsilons |
| 20 | mean direction 1 | 2 | 0.4061 | N/A | fail: fewer than 3 epsilons |
| 20 | log scale | 3 | -14.1977 | -8.8916 (3.2710) | cell passes tolerance; weak/high-uncertainty evidence |
| 40 | mean direction 0 | 2 | 13.7273 | N/A | fail: fewer than 3 epsilons |
| 40 | mean direction 1 | 1 | -1.6084 | N/A | fail: fewer than 3 epsilons |
| 40 | log scale | 1 | -14.4768 | N/A | fail: fewer than 3 epsilons |
| 50 | mean direction 0 | 2 | 5.4009 | N/A | fail: fewer than 3 epsilons |
| 50 | mean direction 1 | 1 | -2.2795 | N/A | fail: fewer than 3 epsilons |
| 50 | log scale | 2 | -9.6477 | N/A | fail: fewer than 3 epsilons |

The single passing cell has absolute error `5.3061` and tolerance `9.8130`
(`3*SE`), so it is a pass under the frozen rule but not a precise estimate.

Across 108 final heads, 58 passed all head-level gates. Failure counts were:
Platt-slope `38`, held-out signal `24`, AUC range `14`, and ECE `8`. These
counts overlap because one head may fail more than one gate. All outputs were
finite. The runtime dependency audit passed with no forbidden module loaded.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not launch SIR | failed: 8/9 exact-oracle cells not admitted | exact-oracle continuation veto fired | pointwise ratios remain noisy and small-epsilon signal is weak | revise and independently review the training/tuning protocol before another oracle | classifier-ratio identity is false |
| Preserve one passing cell as diagnostic evidence | `T=20` log-scale met the declared error tolerance | no cell-level veto there | SE is large and error is 5.31 | use only to diagnose capacity/calibration | procedure is generally calibrated |
| Reject attempt 02 as a generic score reference | campaign-wide oracle gate failed | fewer than 3 admitted epsilons in 8 cells | all-or-nothing replicate admission is demanding but predeclared | do not weaken gates post hoc | SIR score has any particular value |

## Inference-Status Table

| Inference item | Status |
|---|---|
| Hard veto screen | failed at exact Gaussian oracle; SIR launch vetoed |
| Statistically supported ranking | none; no state-estimation algorithms were evaluated |
| Descriptive-only differences | centered quadratics improved scale classification; 58/108 repaired heads passed versus 37/108 before repair |
| Default readiness | no |
| Next evidence needed | a newly reviewed exact-oracle plan with coordinate-specific classifier/tuning scope, then a fresh untouched oracle campaign |

## Post-Run Red Team

Strongest alternative explanation: the classifier-ratio identity is correct,
but the frozen training protocol is not adequate for all location and scale
ratios. One horizon-wide architecture is selected by averaging validation loss
across three materially different coordinates. Mean ratios are exactly linear,
whereas log-scale ratios are quadratic, so the shared selection scope is not
well justified. Small epsilons also have weak separability, while the largest
scale epsilon becomes nearly perfectly separable at longer horizons.

What would overturn this decision: a fresh, predeclared coordinate-specific
training/tuning protocol that passes all nine untouched exact-oracle cells with
stable epsilon extrapolation. It must not tune on the fixed evaluated path,
weaken gates after seeing results, import state-estimation code, or use an
analytical score as a training target.

Weakest part of current evidence: only three classifier replicates support each
epsilon, and the sole passing cell has wide uncertainty. The result supports a
hard non-admission decision, not a broad claim that neural ratio estimation
cannot work.

## Clean Restart State

Resume from these files only:

- plan: `docs/plans/bayesfilter-sir-classifier-ratio-score-plan-2026-08-13.md`;
- independent review and repair audit:
  `docs/plans/bayesfilter-sir-classifier-ratio-score-plan-review-2026-08-13.md`;
- this result/reset memo;
- repaired exact-oracle result:
  `docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813/exact_full_attempt02_centered_quadratics/result.json`;
- repaired exact-oracle manifest in the same directory;
- implementation under `bayesfilter/independent_score/`;
- runner `docs/benchmarks/run_sir_classifier_ratio_score_20260813.py`;
- GPU wrapper `scripts/run_sir_classifier_ratio_score_gpu.sh`;
- focused tests `tests/highdim/test_classifier_ratio_score_tf.py`.

Do not resume from `simulation_score_tf.py`,
`run_sir_simulation_score_20260813.py`, or the prior
`sir_simulation_score_20260813` artifacts. They implement the rejected
Fisher/path-importance question and are historical negative evidence only.

The next plan must remain observation-only balanced classification with
`calibrated_logit/(2*epsilon)`. The smallest defensible revision is
coordinate-specific classifier and hyperparameter selection on simulated
validation data, followed by a fresh untouched exact Gaussian oracle. This is
a material experimental-protocol revision and must be reviewed before another
claim-bearing launch. SIR remains blocked until that oracle passes.
