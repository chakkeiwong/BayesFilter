# Phase 27 Repair and Refresh Note

This note is updated after each Phase 27 attempt. It is intentionally separate
from the parent terminal record.

| Attempt | Failure class | Repair | Focused check | Decision |
|---|---|---|---|---|
| 1 | harness: output root pre-created for audit files | use a fresh child root; preserve audit files | runner refused overwrite before target execution | repaired without scientific execution |
| 2 | harness: TensorShape was not JSON serializable after target/ETPF evaluation | encode `tf.TensorShape` and `tf.DType` at artifact boundary; launch a fresh child | `py_compile` passed; attempt 3 below | repaired |
| 3 (`phase27-attempt2`) | none | none | all 12 hard gates passed | admit corrected boundary to Phase 28 |

Passing receipt: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase27-measure-contract/phase27-attempt2/`.
The command completed in `30.75 s`; all 16 source and transformed rows had
finite target values and valid status. The theta/chart round-trip residual was
`1.11e-16` and the ratio residual was `7.11e-15`. ETPF converged for this
small diagnostic (`59` Riccati iterations in the serialized diagnostics), with
covariance residual `5.31e-5` and negative correction fraction `0.5273`.
Those last two values are explanatory only. The refreshed Phase 28 entry gate
is the hard shape/measure/status receipt, not an ETPF moment threshold.

Do not reinterpret a failed moment or whitening diagnostic as a measure
failure unless the exact measure contract itself fails.
