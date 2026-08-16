# Defensive weighted NeuTra validation reset memo (2026-08-11)

## Current state

The resumed sequential GPU campaign completed without a stream disconnect. Rung 0
passed. The first `(0.8, 0.2)` r1 target remains open because every tested generic
IAF capacity has failed the replicated component-weight criterion. Do not start the
four-mode, NeuTra paper, DSGE, reduced SSL-LSTM, or q=20 rungs.

Read first:

- `docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md`
- `docs/plans/bayesfilter-defensive-weighted-neutra-validation-result-2026-08-11.md`

## Terminal evidence

| Candidate | Weighted minority-mass mean | 95% interval | Truth | Status |
|---|---:|---:|---:|---|
| Three-stage `(32,32)` | 0.17249 | [0.16278, 0.18219] | 0.20000 | Reject |
| Six-stage `(32,32)` | 0.18998 | [0.18602, 0.19394] | 0.20000 | Reject |
| Six-stage `(64,64)` | 0.19478 | [0.19045, 0.19910] | 0.20000 | Reject |

All terminal runs were finite and observed both modes. These are candidate
rejections, not hard numerical vetoes or evidence against weighted forward KL.

Canonical combined-capacity summary:

`docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width64-replication-summary-v1/result.json`

Its SHA-256 is
`f6e01bd17fe24551d090a1c16da7aafad07e0640d7a3b202d5fccbcf47039a3a`.

## Stream-safe workflow

The successful recovery used these constraints:

- one GPU process at a time;
- one visible physical GPU (`CUDA_VISIBLE_DEVICES=1`);
- one attached terminal session per run;
- polling at no more than 30-second intervals;
- no next run until terminal JSON, manifest, state, and hashes all verified;
- fresh output root for each attempt;
- `TF_FORCE_GPU_ALLOW_GROWTH=true` plus repository helper verification before
  TensorFlow initialization.

Replications 4--7 completed sequentially in `98.79`, `99.27`, `99.16`, and
`102.16` seconds. The previous empty `capacity-depth6-width64-replication-4-v1`
directory is ineligible and remains preserved. The valid replacement is
`capacity-depth6-width64-replication-4-v2`.

## Next scientific action

The width-128, 10,000-update candidate passed its fresh eight-seed confirmation:
mean minority mass `0.20000077`, 95% interval `[0.19782562, 0.20217592]`, containing
truth `0.2`. All weighted runs were finite and represented both modes. The immediate
next action is the remaining r1 analytic target variants: equal-weight and
unequal-covariance two-mode cases, the `(0.95,0.05)` rare-mode stress test, and the
four-mode asymmetric target. Do not start paper, DSGE, reduced SSL-LSTM, or q=20
work until those analytic gates pass.

If a remaining r1 target fails after target-specific tuning, write a componentwise
or augmented-state repair subplan that has an explicit discrete mode variable or
normalized mixture density. That repair should:

1. retain the same normalized `(0.8, 0.2)` Gaussian-mixture authority and balanced
   defensive proposal;
2. learn or fit component-conditional invertible maps without consuming analytic
   labels as an undeclared production oracle;
3. preserve exact mixture normalization and Jacobians;
4. compare against the six-stage `(64,64)` weighted IAF and matched reverse KL;
5. require the same eight-independent-run component-weight interval before r1
   promotion;
6. test inference with unknown/soft mode assignments before claiming transfer to
   posterior targets where modes are not labeled.

Freeze the successful width-128 protocol as a target-specific baseline hypothesis
for the remaining analytic targets. It is not a repository default and must not be
transferred to scientific targets without their own reviewed training protocol.

Canonical confirmation artifacts:

- `docs/plans/bayesfilter-defensive-weighted-neutra-width128-updates10000-confirmation-result-2026-08-12.md`
- `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/width128-updates10000-confirmation-summary-v1/result.json`

## Verification

Last terminal focused suite:

```text
24 passed in 10.24 s
```

The suite covers analytic mixture value/score, inverse and Jacobian mechanics,
weighted gradients and XLA updates, checkpoint replay, runner identity, and summary
interval/path validation.
