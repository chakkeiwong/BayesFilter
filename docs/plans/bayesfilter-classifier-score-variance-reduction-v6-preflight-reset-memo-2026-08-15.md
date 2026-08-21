# V6 Variance-Reduction Preflight Reset Memo

Date: 2026-08-15  
Status: `BLOCKED_BEFORE_GPU_EXECUTION`

## Completed

- Wrote and reviewed the four-arm paired campaign plan.
- Implemented independent/CRN training banks at `n=2048` and `n=8192`.
- Implemented exact `n2048` prefixes of `n8192` banks.
- Added whole-pair minibatch shuffling for every arm.
- Validation, calibration, test, fixed, and audit paths are shared within each
  paired bundle replicate.
- Implemented ten-bundle crossed bundle/path variance aggregation with a
  5,000-replicate paired cluster bootstrap.
- Kept natural path variance separate from training-bundle variance.
- Added Gaussian audit- and fixed-path exact-MSE guards.
- Added source/runtime dependency audits, pairing hashes, serious manifests,
  allocator reporting, and incomplete-campaign refusal.
- Focused test result: `14 passed` for the variance and existing anchored
  estimator suites.
- Tiny CPU pair-minibatch integration smoke completed with finite output.

## Blocker

The required maximum-count GPU capacity smoke did not start. Two escalated
launch attempts were rejected before process creation because the automatic
approval reviewer returned `502 Bad Gateway`. This is an external execution-
permission failure, not a GPU, TensorFlow, XLA, model, memory, or scientific
result.

No maximum-count GPU artifact, Gaussian campaign, or SIR campaign exists yet.
Do not interpret the CPU smoke as scientific or capacity evidence.

## First Authorized Command

After explicit GPU approval, run exactly:

```bash
bash scripts/run_tmux_with_status.sh \
  bf_v6_capacity_sir \
  docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/gpu_capacity_sir_crn_n8192_status \
  bash scripts/run_classifier_score_variance_bundle_gpu.sh \
  --kind sir --bundle 0 --profile capacity --arm crn_n8192 --cell T50_j1 \
  --output docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/gpu_capacity_sir_crn_n8192_attempt03
```

Inspect `exit_code`, `worker.log`, the result manifest, memory growth, allocator
peak, XLA evidence, finiteness, and elapsed time. Use that elapsed time to
confirm the six-hour campaign budget before launching full campaigns.

If capacity passes within budget, launch persistent campaigns in order:

```bash
bash scripts/run_tmux_with_status.sh \
  bf_v6_gaussian \
  docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/gaussian_full_status \
  bash scripts/run_classifier_score_variance_campaign.sh \
  gaussian docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/gaussian_full

bash scripts/run_tmux_with_status.sh \
  bf_v6_sir \
  docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/sir_full_status \
  bash scripts/run_classifier_score_variance_campaign.sh \
  sir docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815/sir_full
```

The campaign scripts resume only from completed bundle artifacts and never
overwrite completed bundles.

## Interpretation Guard

The campaign tests whether CRN and/or more paths reduce variance across
independently trained bundles at identical observation paths. It does not test
whether natural score variation across simulated paths becomes smaller. A
lower-variance Gaussian arm is vetoed if its exact audit- or fixed-path MSE is
statistically worse. SIR can support only a training-variance conclusion, not
score correctness.
