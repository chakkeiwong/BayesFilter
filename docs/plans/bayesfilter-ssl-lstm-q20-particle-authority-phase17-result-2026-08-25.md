# Phase 17 Result: Source/Method Identity Audit

Status: `PASS_METHOD_IDENTITY_AUDIT_REPAIR_REQUIRED`

The CPU-hidden TensorFlow audit passed immutable-bank, protocol-hash, target
signature, source-hash, and finite-value gates. It found that the bounded q=20
Phase 3 runner does not implement the named source methods. This is an
implementation identity finding, not evidence against the methods as research
directions.

| Arm | Claimed target | Quantity actually computed | Verdict relative to claim | Required missing operation |
|---|---|---|---|---|
| M1 | Acevedo second-order ETPF | affine Cholesky finite-cloud mean/covariance match | wrong relative to named ETPF source identity | LETF `D`, OT/Sinkhorn plan, Riccati correction |
| M2 | Ebeigbe GenUT | symmetric `2d+1` mean/covariance rule | wrong relative to named GenUT source identity | skewness/kurtosis constraints and asymmetric offsets |
| M3 | Li-Coates invertible LEDH-PFPF | one fixed affine map with tautological determinant comparison | wrong relative to named LEDH-PFPF source identity | pseudo-time step product, proposal/target terms, covariance lifecycle |
| M4 | full second-order ET-PF | alias of M1 scaffold | wrong relative to named ET-PF identity | independent filter transition and reference comparison |

The measured M2 discrepancy makes the GenUT gap concrete. On the audited bank,
the source marginal third central moments were approximately
`[-32.1976, 21.7551, -0.2459, -6.8205]`, while the symmetric sigma rule gave
approximately zero; the maximum third-moment residual was `32.1976` and the
maximum fourth-moment residual was `119.5494`. These are finite-cloud
diagnostics, not density claims.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Relabel M1--M4 as scaffolds and implement one source-faithful arm at a time | source/code identity audit completed; dynamic M2 mismatch observed | no input or numerical veto | source audit is bounded to this runner; no q20 ETPF implementation yet | Phase 18: exact second-order LETF/ETPF fixture with Sinkhorn and Riccati receipts | no rejection of ETPF/GenUT/LEDH/ET-PF ideas, posterior correctness, IID law, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the audit artifact; named-arm identity gates fail as expected |
| Statistically supported ranking | None; no performance comparison was run |
| Descriptive-only differences | M2 symmetric rule misses measured marginal skewness and kurtosis |
| Default-readiness | Not ready; current M1--M4 cannot be promoted under their names |
| Next evidence needed | source-faithful M1 fixture, then q20 integration only after fixture gates |

The old Phase 3 result remains preserved as historical scaffold evidence. The
runner labels were repaired to avoid calling these computations ETPF, GenUT, or
full ET-PF. No HMC was launched.
