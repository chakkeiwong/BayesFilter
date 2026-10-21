# M13 canonical consumer — active result, 2026-09-23

The M13 canonical likelihood consumer is mechanically wired and its analytical
parameter score is correct for the finite program, but the M13 consumer is not
qualified for surrogate-HMC inference. The complete FP64 N256/T20 parity stage
passed finite-difference score checks, trace/non-trace value and score parity,
determinism, and correction-condition/feasibility diagnostic presence. Its
FP32 replay failed the predeclared no-harm check after the full diagnostic
payload was exposed: maximum scaled value difference was 0.012223470956087112
and maximum scaled score difference was 0.007399038877338171. The failure is
an observability/XLA-operation-order failure for that candidate output
contract, not evidence that the canonical correction equation or surrogate HMC
is invalid.

The repair restores the established five scalar cap diagnostics on the default
batch-consumer path and makes the complete correction signature explicit through
return_all_correction_diagnostics=True. The authority regression requests that
opt-in and compares every field with the single-cloud correction record. A
bounded CPU regression then passed 38 focused tests, including this wiring check.
The repair does not erase the prior GPU no-harm failure and did not consume a
new numerical attempt; a GPU no-harm retest belongs to a new scoped plan.

## Evidence and numerical findings

The qualified independent M13 reference is a fixed-grid, finite-canonical
likelihood reference for the state-4/observation-2 range/bearing fixture at
nine Q/R scale pairs and horizon T=20. All 41 reference tests passed and the
largest log interval halfwidth was 0.005539, below 0.02. It is not a posterior
reference: no parameter prior, full-support normalization, or HMC chain has
been qualified.

The consumer adapter uses the same fixture, transition-before-observation law,
wrapped correlated Gaussian observation density, Q/R parameterization
Q=Q0*exp(2 lambda_Q), R=R0*exp(2 lambda_R), Contract-E reset, dual-cap
correction, and complete analytical parameter dependencies. Controls remain a
diagnostic warm start and are not a tuning artifact.

The first consumer run had two of nine FP32 N64 points fail at the time-0
correction. The localization run made all nine points finite for N64 FP64, N256
FP32, and N1024 FP32, but severe low-Q/low-R likelihood errors and ESS collapse
remained:

| Arm | Valid points | Nominal log-likelihood error | Low-Q/low-R error | Minimum observed ESS |
|---|---:|---:|---:|---:|
| N64 FP64 reference exception | 9/9 | -7.71129 | -222.95392 | 1.004 |
| N256 FP32 | 9/9 | -7.63687 | -192.59495 | 1.000 |
| N1024 FP32 | 9/9 | -4.58305 | -174.77777 | 1.022 |

These differences are explanatory finite-cloud diagnostics. They are neither a
posterior rejection nor evidence of a statistically supported ranking.

The stored GPU parity report is
docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-consumer-04/m13-consumer-parity/report.json.
The stored CPU repair report is
docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-consumer-repair-01/result.json;
its pytest.log records 38 passed tests. The numerical attempt receipt reports
35810.338525 cumulative CPU-worker seconds and 33507.6772886668 GPU-wall
seconds through attempt04. The M13 numerical allocation has used three
numerical attempts; no fourth attempt is launched in this pass.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the finite-program consumer implementation | Full analytical score parity and callback-law checks pass | Default full-observability candidate failed FP32 no-harm | Whether the opt-in/default split is numerically unchanged on the GPU | A new scoped GPU no-harm retest after plan review | M13 posterior correctness |
| Do not qualify M13 yet | No accepted endpoint/reference/tuning package | Low-noise errors and prior no-harm failure remain | Scope-specific controls and a full-support parameter reference | Create a separate M13 tuning/reference plan | Failure of surrogate HMC as a method |

| Inference status | Finding |
|---|---|
| Hard veto screen | Prior full-diagnostic output candidate fails FP32 no-harm; default route has not been requalified on GPU after the repair |
| Statistically supported ranking | None |
| Descriptive-only differences | Likelihood offsets, ESS values, and particle-size differences are descriptive diagnostics |
| Default readiness | Not established |
| Next evidence needed | New scoped control calibration, a GPU no-harm retest of the repaired default route, a full-support parameter reference with an explicit prior, then posterior endpoint-chain checks |

This result is part of the existing closure master. The current numerical M13
allocation is complete, so the next M13 GPU run is a direction change in
scientific contract from untuned consumer mechanics to scope-specific tuning
and reference construction; it needs a new plan and approval before execution.
The master remains incomplete. AR2 uncertainty calibration and the remaining
model-specific references are also outstanding.

Post-run red team: finite-difference and trace parity can pass while the finite
particle likelihood is badly biased. The low-noise ESS collapse is the strongest
alternative explanation for the large likelihood errors. A repaired GPU replay
that is no-harm but still has low-noise error would close only the observability
gap, not the approximation or posterior gap.
