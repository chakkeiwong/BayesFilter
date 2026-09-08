# Corrected Parameter-Authority Phase 27 Result

Date: 2026-08-25  
Status: `PASS_CORRECTED_PARAMETER_MEASURE_CONTRACT`

## Executed evidence

The first launch was stopped before evaluation because the audit directory had
already been reserved by the launcher. A second launch reached the target and
ETPF calculation but failed while serializing a TensorFlow `TensorShape`. The
serializer repair was focused and a fresh third root completed:

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase27-measure-contract/phase27-attempt2/`

The MathDevMCP raw outputs are retained beside it at the phase root. The
Jacobian check returned `inconclusive` because the generic parser could not
encode the matrix/domain expression. The rank/code check returned
`scope_limited_match`; neither result is treated as a proof certificate.

## Hard-gate receipt

| Gate | Result |
|---|---:|
| particle shape | `[16,4]` |
| target parameter dimension | `4` |
| internal UKF state / innovation dimensions | `60 / 20` |
| source target finite/status-valid rows | `16 / 16` |
| affine round-trip residual | `1.11e-16` |
| theta/chart ratio residual | `7.11e-15` |
| ETPF output shape | `[16,4]` |
| transformed target finite/status-valid rows | `16 / 16` |

All twelve hard gates passed. The ETPF covariance residual (`5.31e-5`) and
negative correction fraction (`0.5273`) are descriptive diagnostics; the
receipt makes no density or IID claim for the empirical transform.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Admit the corrected theta/state boundary to the next phase | all shape, measure, finite/status, and chart-ratio gates | no hard veto | finite cloud; no authority or mode evidence | run a fresh theta-measure C0/M0 pilot with explicit proposal terms | no SMC-U authority, posterior, IID whitening, mode theorem, LEDH, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed after two localized harness repairs |
| Statistically supported ranking | none; one bounded diagnostic cloud |
| Descriptive-only differences | ETPF covariance residual, negative correction, and mode fractions |
| Default-readiness | not ready |
| Next evidence needed | fresh proposal/mass receipt in theta measure, then paired ETPF role check |

## Red-team note

This phase could pass while the proposal misses a posterior mode or while the
target itself is an inadequate synthetic model. The strongest alternative
explanation is therefore support/mode bias, not a successful authority. An
independent fresh proposal receipt and paired seeds are required before any
downstream training interpretation.

