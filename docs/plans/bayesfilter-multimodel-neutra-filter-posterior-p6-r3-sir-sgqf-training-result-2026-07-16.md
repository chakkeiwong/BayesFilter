# P6 R3 Result: SIR-SGQF Target-Specific NeuTra Training

Date: 2026-07-16

Status: `TRAINING_ADMITTED`

## Result

The target-specific plain dense-IAF screen selected `dim3_lr1e3`, followed by
a fresh 5,000-step GPU/XLA training run with seed `(20260716,31201)` and no
screen-weight reuse.

- final result:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/training/final/dim3_lr1e3/attempt-01/result.json`;
- result SHA-256:
  `c69b4e4e02b68d13be74f7a87ffc0ec9b1d6a47bc8438d56c048577a78531854`;
- frozen payload SHA-256:
  `2ddff1ed2521ec674e64665bb8882a84ebc767e0850d677db72fa05a7e5ccdf4`;
- transport semantic hash:
  `dbd29efe786ec23c7b1098ba95ec6cad3a439b4889e04c67eeb2127965949c89`;
- artifact signature:
  `d722d3a3ec5a765ca3949ca894a6c4ea06dfe541e8e95a464dc1d2d81f6f2e09`;
- selection SHA-256:
  `068b7537eb09d5ba3218236dcfce23624266c4a93ba8a5c8e74f228ce85d2afc`.

## Evidence

| Gate | Result |
| --- | --- |
| Screen | all four 500-step recipes completed and were affine-nonworse on identical eight-by-128 heldout draws |
| Selection | all learned candidates within two paired MCSE of nominal; selected lower 558-parameter capacity and lower `1e-3` rate; no statistically supported ranking |
| Final training | 5,000 steps in one compiled `tf.while_loop`; 3,197.24 seconds graph runtime, 3,241.27 seconds total |
| Target health | all 501 recorded rows finite, status available/valid, zero nonvalid rows, zero floors |
| Execution | RTX 4080 SUPER, memory growth, GPU variables/moments/outputs, XLA, batch 128, no scalar/row-mapped/sample-loop fallback |
| Frozen parity | forward, logdet, pullback score, and logdet score maximum gaps all zero |
| Final affine screen | learned-minus-affine mean reverse-KL `-0.008854`; paired MCSE `0.002434`; affine-nonworse passed |
| Hashes | all eight final recursive hashes verified |

`innovation_condition_estimate` is unavailable on the unchanged typed
SIR-SGQF target and is recorded as unavailable/null. Validity-bearing status,
minimum innovation eigenvalue, values, and reviewed scores remain present.

## Decision And Inference Status

| Decision | Primary status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit frozen training artifact | engineering, identity, health, parity, affine-control, and artifact gates passed | no final-training veto | loss/heldout evidence does not establish downstream sampling | fresh transported-HMC tuning and same-target agreement | HMC convergence, posterior agreement, learned superiority, SGQF exactness, calibration, robustness, or readiness |

No stochastic recipe ranking is supported. Screen and final loss differences
are descriptive or veto evidence only. The strongest alternative explanation
is that the target-specific affine chart already captures nearly all useful
geometry and the residual IAF is scientifically unnecessary. R4 directly
tests that explanation on the downstream sampler and same-target posterior
means.

Attempt 1 of the first screen failed before step 1 because the generic trainer
incorrectly required optional condition-number telemetry. The identity-
preserving repair made that field explicitly optional and did not alter the
SIR target source or typed signature. The trace-only attempt was excluded.
