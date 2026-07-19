# SSL-LSTM NeuTra Phase 5 Exact Preflight Native Review

Date: 2026-07-16

Verdict: `AGREE_R2_NO_BLOCKING_FINDING`

## Scope

Focused review of the Phase 5 transformed-density implementation and the two
source-bound receipts. No HMC transition, tuning result, sampler claim, or
posterior claim was reviewed or authorized.

Reviewed paths:

- `docs/benchmarks/run_ssl_lstm_neutra_phase5_exact_preflight_2026_07_16.py`;
- `tests/test_ssl_lstm_neutra_phase5_exact_preflight.py`;
- `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/cpu-reference-r2.json`;
- `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-5-trial0-gh/gpu-xla-r2.json`;
- `bayesfilter/inference/batched_value_score.py` fixed-transport adapter; and
- `bayesfilter/inference/neutra_artifacts.py` dense-IAF loader.

## Findings

No blocking mathematical, implementation, artifact-binding, or boundary
finding remains.

The change-of-variables direction is `theta=T(z)`. The adapter value is
`log_pi_theta(T(z)) + log|det dT/dz|`; its score is the explicit pullback plus
the log-Jacobian score. Both are independently checked against direct values
and central finite differences. Wrong-sign and missing/wrong log-Jacobian-score
controls are observed to fail by margins far above the pass tolerances.

Both payload byte hashes, target signature, topology/tensor/transport hashes,
and reconstructed best-state hashes are bound. Regenerating each frozen
payload from its preserved best trainer state reproduces all three transport
hashes and gives exact forward/logdet parity. The fixed binding exposes no
reachable `tf.Variable`, optimizer, trainer, or trainable-variable surface.

The trusted receipt compiles the whole `[21,4]` transformed value/score program
with CUDA XLA. G and H have nonempty, candidate-specific HLO hashes and GPU
output placement. CPU/GPU residual differences are roundoff scale; the largest
observed cross-device difference in the finite-difference maximum is
`7.11e-10` for H.

Final contract audit found and repaired one material executable-gate mismatch:
score parity had reused the `1e-6` finite-difference tolerance instead of the
prospective `1e-10` parity threshold. The first receipts are superseded. The
`r2` receipts bind corrected source with `PARITY_ATOL=1e-10`; GPU score-parity
residuals are zero for G and H, and CPU residuals are below `8e-15`.
Original-start-only roundtrip residuals are also reported explicitly.

The CPU and GPU `r2` runs record different Git commits because another lane
advanced `HEAD` between them. Their complete Phase 5 source-binding mappings
are identical, including runner, plan, target, adapter, and artifact-loader
hashes. Payload and best-state hashes are also identical, so the concurrent
commit did not change the reviewed computation.

## Residual Risks

- The 21-point finite bank cannot establish global support or mode coverage.
- G and H were trained with reverse KL and may share mode-seeking failure.
- The 1,800-second cap is checked sequentially between candidates, not by an
  asynchronous mid-call terminator. The completed trusted run used only
  `417.33` seconds, so this did not affect the result.
- The existing fixed-transport HMC tuner is identity-mass only and must not be
  represented as already implementing the master program's optional
  diagonal/dense mass-repair rung.

These are Phase 6 or posterior-validation risks. They do not invalidate the
exact frozen change-of-variables evidence.

## Boundary Review

Static inspection and focused tests confirm that the runner imports no HMC
kernel or TensorFlow Probability sampling surface. The receipts state
`no_hmc=true`. No HMC transition, tuning, retained samples, candidate search,
training, or payload mutation occurred.
