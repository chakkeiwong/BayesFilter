# Phase 0E first provider integration, 2026-09-15

The actual SGQF covariance consumer and a fully normalized fixed-power Gaussian psi-APF execute on CPU/XLA and GPU/XLA. This completes the first provider slice, not all of Phase 0E. KDM covariance lifecycle and fitted IAPF remain implementation tasks.

## Evidence

- `phase-0e-provider-tests-03.log`: six tests pass (24.29 seconds). Tests cover nonlinear batched SGQF agreement with the independent standalone filter, signed-weight invalid covariance rejection, reset covariance carry, all six parameter derivatives through LEDH, finite-state twist telescoping, zero-twist recovery, and Gaussian initial normalization. Two canonical regressions also passed in the previous focused run, `phase-0e-sgqf-tests-02.log`.
- `providers-cpu-02`: eight complete rows, 26.112 seconds, deliberate CPU FP64/XLA reference.
- `providers-gpu-01`: eight complete rows, 42.208 seconds, GPU FP32/TF32/XLA, verified memory growth. Manifest and per-row runtime fields preserve device/source/seed/settings provenance.
- Baselines: canonical UKF-LEDH, bootstrap SIR and adapted SIR. Candidates: SGQF levels 2 and 3; fixed twist powers 0, 0.6 and 1. N=8, state dimension2, horizon3. These are mechanics hypotheses, not scope-tuned defaults.

## Repairs

The first SGQF test used an incorrect private executor name; it was corrected to the actual shared executor. The first twist tangent used a solve API that did not broadcast a covariance over parameter directions; Cholesky solve repairs the broadcast and passes all-six-direction finite differences. The first CLI attempt rejected decimal points in row identifiers before any numerical execution; valid identifiers were used in a new output directory. Prior logs remain preserved.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain SGQF and fixed psi-APF as executable candidates | Mathematical mechanics and real CPU/GPU consumer pass | No invalid accepted row; the deliberately invalid signed covariance is rejected | Tiny affine fixture cannot establish nonlinear score quality | Build remaining covariance/fitted-twist consumers and stochastic FD independently | Superiority, default readiness, full IAPF, full KDM filtering replacement |

| Inference status | Result |
|---|---|
| Hard veto screen | Tested rows finite; invalid-covariance fixture rejected |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-row values/scores/costs only; one dataset/replicate |
| Default readiness | Not established |
| Next evidence | Scope tuning, nonlinear regimes, paired replicated untouched comparisons |

Strongest alternative explanation: affine dynamics conceal covariance approximation errors. The nonlinear standalone parity and signed-rule counterexample test mechanics but do not overturn that limitation. A nonlinear consumer mismatch or missing twist factor would invalidate the implementation; poor held-out MSE would reject a candidate and trigger the planned repair rather than reject the research direction.

## Refresh

Phase0F can proceed using the verified value endpoints without waiting for the remaining0E methods. Preserve this source snapshot before new source changes. Next implement paired stochastic stencils, full-score reconstruction, fixed positive h selection on calibration/validation and independent final comparison. Five total GPU framework launches used; one remains in the current120-row integration slice, seven in the overall12-launch engineering allocation. This slice has consumed70 rows (54 in0D plus16 here); numerical compute remains far below12CPU/8GPU hours.
