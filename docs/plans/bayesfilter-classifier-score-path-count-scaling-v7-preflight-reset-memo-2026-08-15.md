# V7 Path-Count Scaling Preflight Reset Memo

Date: 2026-08-15  
Status: `BLOCKED_BEFORE_GPU_EXECUTION`

## Objective

Test the independent-noise classifier-score training ladder at 8,192, 16,384,
and conditionally 32,768 paths per class and perturbation. The completed V6
`independent_n8192` bundles are the paired baseline. Stage 2 at 32,768 runs only
if the frozen Stage-1 continuation rule passes.

## Completed Preflight

- Wrote and skeptically audited the V7 evidence contract and sequential rule.
- Kept CRN out of the ladder because completed SIR evidence showed a
  heterogeneous CRN effect that would confound the path-count question.
- Verified TensorFlow stateless SIR noise at 8,192 is bitwise identical to the
  first 8,192 rows of a 32,768 draw for initial, transition, and observation
  noise.
- Verified the current estimator and V6 runner hashes match all completed V6
  baseline manifests.
- Implemented the 8,192-row block runner for 16,384 and 32,768, preserving the
  V6 seed function, split roles, controls, batch size, epoch policy, and
  evaluation paths.
- Implemented fail-closed audits for source closure, result checksums, shared
  validation/calibration/test hashes, audit/fixed hashes, independent
  plus/minus noise, and exact nested prefixes.
- Implemented paired 5,000-replicate bootstrap ratios, normalized `1/N`
  efficiency, adjacent scaling exponents, and a three-level global exponent.
- Implemented the frozen Stage-1 continuation decision.
- Focused verification: `26 passed`.
- Shell syntax, Python compilation, and `git diff --check` passed.

## Execution Blocker

Two identical escalated attempts to launch the 16,384 SIR capacity diagnostic
were rejected before process creation because the automatic permission review
did not finish before its deadline. This is an external permission-boundary
failure, not a GPU, CUDA, TensorFlow, XLA, memory, code, mathematical, or
scientific result.

No capacity status directory, stage bundle, or result artifact was created.
No GPU campaign budget was consumed.

## First Authorized Command

Run exactly:

```bash
bash scripts/run_tmux_with_status.sh \
  bf_v7_capacity_sir_16384 \
  docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/capacity_sir_16384_status \
  bash scripts/run_classifier_score_path_count_bundle_gpu.sh \
  --kind sir --bundle 0 --path-count 16384 --profile full_cell \
  --cell T50_j1 \
  --output docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/capacity_sir_16384_attempt01
```

Inspect the result checksum, exact 8,192 prefix hashes, evaluation hashes,
finite output, optimizer completion, epochs/updates, XLA, memory growth,
allocator peak, and wall time. Use the measured time to confirm the remaining
7.5-hour total budget before the ten-bundle stage.

## Stage Commands After Capacity Passes

```bash
bash scripts/run_tmux_with_status.sh \
  bf_v7_gaussian_16384 \
  docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/gaussian_16384_status \
  bash scripts/run_classifier_score_path_count_campaign.sh \
  gaussian 16384 \
  docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/gaussian_16384

bash scripts/run_tmux_with_status.sh \
  bf_v7_sir_16384 \
  docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_16384_status \
  bash scripts/run_classifier_score_path_count_campaign.sh \
  sir 16384 \
  docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_16384
```

Aggregate Stage 1 CPU-only, apply the frozen continuation rule, and launch
32,768 only if it passes. Do not weaken the rule after seeing 16,384 results.

## Interpretation Guard

The experiment measures variance of the fully fitted estimator under the
inherited early-stopped epoch policy. Larger N receives more minibatch updates
per epoch, so this is not a fixed-update pure Monte Carlo experiment. A ratio
interval containing 0.5 is compatible with `1/N`; it does not prove exact
`1/N` scaling. No result supplies an exact SIR score or filter validation.

