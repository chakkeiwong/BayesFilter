# Phase 8 C2 backend-parity prerequisite

Date: 2026-08-30  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-subplan-2026-08-30.md`  
Status: `PASS_B8_SEMANTIC_PARITY`

## Question and contract

The C2 strict-backend screen must use `tensorflow_eigh_strict`. Before it
starts, the same q=20 batch must produce matching values, analytic scores, and
validity/status rows under the historical `compiled_custom_op` route and the
strict route. The comparison is a semantic prerequisite, not a performance or
XLA-equivalence claim. Values use absolute tolerance `1e-8`; scores use
absolute tolerance `1e-7` or scaled relative tolerance `1e-7`.

## Attempts

The first harness attempt evaluated both bridge calls inside an XLA function.
It failed before producing a comparison because the custom route emits
`SymmetricPrincipalSqrt`, for which this TensorFlow build has no
`XLA_GPU_JIT` kernel. This is an execution-backend incompatibility of the
custom comparator, not a target-status or strict-backend numerical failure.

The repaired harness evaluates the custom comparator in its supported graph
mode (`jit_compile=false`) and evaluates the strict route with XLA
(`jit_compile=true`) on the identical B=8 physical input bank. The C2 training
route remains strict XLA throughout.

Receipt:
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c2-strict-calibration/backend-parity/attempt-02-b8-graph-custom/run_manifest.json`

## Result

| Quantity | Observed | Limit | Status |
|---|---:|---:|---|
| Value maximum absolute difference | `1.3848300284280413e-9` | `1e-8` | pass |
| Score maximum absolute difference | `4.795940846281238e-9` | `1e-7` | pass |
| Scaled score maximum relative difference | `4.795940846281238e-9` | `1e-7` | pass |
| Valid/status rows | equal, all valid | exact | pass |
| Target signature | equal | exact | pass |
| GPU/XLA strict route | one RTX 4080 SUPER, XLA | required | pass |
| Memory growth | verified before logical initialization | required | pass |

The process wall time was `66.48853411595337` seconds. The failed first
attempt is preserved at
`.../c2-strict-calibration/backend-parity/attempt-01-b8/failure.json` and is
not treated as scientific evidence.

## Decision

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Admit the B=8 prerequisite | Same-input value/score/status parity within frozen tolerances | pass after graph-mode repair | Launch the bounded strict-backend C2 screen | No backend-wide equivalence, whitening, mode discovery, posterior, HMC, superiority, or scaling claim |

The asymmetric `jit_compile` settings are part of the receipt and must remain
explicit in every downstream manifest. No C2 result may describe the custom
route as an XLA comparator.

