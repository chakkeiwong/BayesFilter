# P3 Result: KSC Principal-Square-Root UKF

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `P3_COMPLETE_KSC_UKF_TARGET_BLOCKED_CONTINUE_P4`

## Outcome

`KSC-UKF` remains `TARGET_BLOCKED`. The new graph-native batched recurrence is
correct for the declared component principal-square-root-UKF plus Gaussian
moment-collapse program, but that filter failed the predeclared independent
dense-reference admission gate. No typed posterior identity, HMC, NeuTra
training, or posterior sample was produced.

Terminal artifact:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p3/KSC-UKF/target-admission/attempt-01-20260715T110415Z/`.

## Evidence

| Ledger | Result |
| --- | --- |
| Engineering correctness | 28 CPU-hidden tests passed; T=1/T=2 principal-square-root wrapper parity, analytic score FD, batching, typed binding, CPU XLA, GPU XLA, and memory growth passed. |
| Numerical/status validity | All six T=1000 audit rows were valid; minimum innovation variance `>=1.177`; minimum state variance `>=0.1877`; maximum mixture-weight sum error `<=1.67e-15`. |
| Dense-reference validity | Passed. Order-401/order-601 value gap per observation `1.03e-14`; FD step gap `1.78e-9`; FD order gap `2.37e-10`. |
| Filter value gate | Failed: observed `0.00378699` per observation versus `0.001` maximum. |
| Filter score gate | Failed: observed maximum coordinate gap `0.07103196` versus `0.01` maximum. |
| Posterior identity | Not issued because the filter gate failed. |

The claimed target was the KSC seven-component transformed-SV posterior using
the component principal-square-root UKF and per-step Gaussian moment collapse.
The new implementation computed that finite approximation and its total source-
coordinate derivative. T=1/T=2 wrapper parity and FD checks support that
classification. The independent dense latent-state KSC recurrence computed a
different filter approximation; the stable observed discrepancies show that
the UKF candidate is wrong relative to the P3 admission threshold, not that the
implementation failed to compute its claimed UKF quantity.

## Integrity And Budget

Recorded and independently verified SHA-256 values:

- `result.json`: `9398d1536c9629d3dcc6fa98e24ca3d1b214422c59d866279168158eda40a187`;
- `run_manifest.json`: `f53e67da539f2e8172bb1f3484d0eaafef3a6096b0ac37e0e853986c0d6bbb96`.

The terminal attempt consumed 32.3 wall-seconds. No comparator or training
bucket was consumed.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep `KSC-UKF` blocked and continue P4 | Frozen filter-admission gate | Dense reference valid; UKF value and score vetoes fired | A looser or different UKF target might be useful but was not tested | Enter independent predator-prey target repair | No KSC HMC/NeuTra, exact-SV, calibration, superiority, or readiness claim |

## Inference Status

| Class | Status |
| --- | --- |
| Hard veto screen | Filter-admission veto supported. |
| Statistically supported ranking | None; no stochastic method ranking was run. |
| Descriptive-only differences | Audit-point value/score gaps describe this frozen target and data. |
| Default readiness | Not ready for identity, HMC, training, or default use. |
| Next evidence needed | A separately planned filter target or predeclared approximation redesign; not required for P4. |

## Post-Run Red Team

The strongest alternative explanation is that the P3 margins reject a useful
approximate posterior. That cannot change the result under the frozen gate.
The dense reference is the strongest part of the evidence because value order,
FD step, and score order convergence all passed by wide margins. The weakest
part is the six-point parameter-region coverage; therefore this result rejects
admission under the declared screen, not every possible KSC-UKF use.
