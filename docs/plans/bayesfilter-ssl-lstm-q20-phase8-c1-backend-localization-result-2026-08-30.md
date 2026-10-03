# Phase 8 C1 principal-square-root backend localization result

Date: 2026-08-30  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-backend-localization-subplan-2026-08-30.md`  
Status: `PASS_TARGET_LOCALIZATION_AND_BACKEND_PARITY_STRICT_BACKEND_COST_FEASIBLE`

## Question and evidence contract

The diagnostic asked whether `tensorflow_eigh_strict` can remove the q=20
target graph bottleneck observed with `compiled_custom_op`. The target, bridge,
data, analytic score, dtype, XLA setting, GPU selection, and stateless inputs
were held fixed. The diagnostic could nominate a new cost experiment only; it
could not select a transport, establish whitening, or reopen C2--C5.

The hard screen required finite values and analytic scores, valid target rows,
the frozen target and bridge identities, verified GPU memory growth before
logical-device creation, XLA execution, and completion within 300 seconds.

## Attempts

Attempt 1 (`attempt-01-eigh-strict`) reached the target calls but stopped at a
runner serialization defect when a vector diagnostic was converted as a
scalar. The defect was repaired by serializing the vector as a list. It was a
harness defect, not a backend or target failure; its partial receipts remain
under the attempt directory.

Attempt 2 (`attempt-02-eigh-strict`) completed in `69.98432411497924` seconds
and produced a complete manifest:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-backend-localization/attempt-02-eigh-strict/run_manifest.json`

| Operation | Elapsed seconds | Result |
|---|---:|---|
| Target, B=8, beta=0 | 11.217353565967642 | finite, 8/8 valid |
| Target, B=8, beta=0.5 | 1.145947159966454 | finite, 8/8 valid |
| Target, B=256, beta=0 | 17.985415667993948 | finite, 256/256 valid |
| Target, B=256, beta=0.5 | 11.355721828062087 | finite, 256/256 valid |
| Transport beta=0 preflight | 1.9104936859803274 | valid |
| Transport beta=0.5 preflight | 1.2652144680032507 | valid |
| Pullback diagnostic, beta=0.5 | 11.95937363198027 | finite |
| First batched optimizer update | 8.844319522031583 | finite, valid |

The previous `compiled_custom_op` localization recorded B=8 target times of
54.843 seconds (`beta=0`) and 45.801 seconds (`beta=0.5`). These timing values
are descriptive historical comparators, not a claim of a fixed speedup.

## Execution receipts

The manifest records `principal_sqrt_backend=tensorflow_eigh_strict`, the
frozen q=20 target signature, the properness receipt, one logical GPU 0, XLA,
and `TF_FORCE_GPU_ALLOW_GROWTH=true`. Memory growth was verified on every
visible physical GPU before logical-device creation. The static route scan
found no `tf.map_fn`, `tf.vectorized_map`, Jacobian pfor, or explicit pfor.
The pullback diagnostic and update were finite; their residuals are diagnostic
only and are not whitening evidence.

## Decision table

| Decision | Primary criterion | Hard veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Nominate strict backend for a fresh cost arm | Complete target-localization receipt | Pass on attempt 2; attempt 1 was a repaired serialization defect | Numerical equivalence to the compiled route and full two-batch cost remain untested | Run the bounded parity check, then the new strict-backend cost subplan | No backend default, whitening, mode discovery, posterior, HMC, superiority, or scaling claim |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for the localization scope; all tested rows were finite and valid and the GPU/XLA policy passed |
| Statistically supported ranking | None; this is one diagnostic process with no stochastic comparison |
| Descriptive-only differences | Strict target timings, diagnostic residuals, loss, and gradient norm |
| Default readiness | Not established; strict backend remains a candidate execution route |
| Next evidence needed | Same-input backend value/score/status parity, then a complete strict-backend two-batch cost receipt |

## Classification and red-team

The result repairs and localizes the graph-cost hypothesis. It does not
invalidate the target, bridge mathematics, transport implementation, data, or
the ensemble direction. The strongest alternative explanation is that the
strict implementation is faster but differs numerically from the compiled
implementation on rows not sampled here, or that reliability/cross-density
work dominates the full pilot. The parity harness and complete cost receipt
are the cheapest discriminators. A passing localization must not be used to
infer a Gaussianized chart or a mode-discovery guarantee.

The follow-up parity receipt passed after the import-order repair. Its complete
result is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-strict-backend/backend-parity/attempt-02-native-eigh/result.json`.
The value and score residuals are within the frozen q=20 parity tolerances, so
the strict route is eligible for the separately reviewed cost-pilot arm. This
does not make it the repository-wide numerical default.

The subsequent full-256 cost pilot also passed. Its result note is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-result-2026-08-30.md`.
