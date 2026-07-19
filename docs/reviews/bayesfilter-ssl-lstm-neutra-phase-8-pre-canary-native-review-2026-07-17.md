# SSL-LSTM NeuTra Phase 8 Pre-Canary Native Review

Date: 2026-07-17

Verdict: `AGREE_ENGINEERING_CANARY_ONLY`

Reviewed paths:

- `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-plan-2026-07-17.md`
- `docs/benchmarks/run_ssl_lstm_neutra_phase8_predictive_design_canary_2026_07_17.py`
- `bayesfilter/inference/predictive_equivalence.py`
- focused Phase 8 and predictive-equivalence tests

## Findings And Repairs

1. The initial receipt recorded only terminal/forecast trace counts and did not
   prove terminal/summary GPU placement. Repaired by recording all four XLA
   trace counters, requiring each to equal one, recording representative
   terminal and all summary numerical tensor devices, and failing closed on any
   CPU placement.
2. The covariance implementation now has a deterministic ridge ladder and
   fail-closed exhaustion, with manual-formula, singular, invalid-policy,
   authentication, and XLA/eager tests. The canary's block length of one is a
   singular mechanics fixture only and cannot calibrate the final block policy.

No unresolved material finding remains for this engineering canary.

Post-launch repair addendum: the first invocation completed the numerical
GPU/XLA path but failed strict receipt serialization on a Python `bytes` status
and wrote no partial artifact. The focused repair decodes UTF-8 bytes, tests
both raw bytes and scalar `tf.string`, preserves every numerical input and
gate, and writes to a distinct repair-01 path. Verdict for that bounded rerun:
`AGREE_ENGINEERING_CANARY_SERIALIZATION_REPAIR_01_ONLY`.

## Skeptical Audit

| Risk | Disposition |
| --- | --- |
| Wrong baseline | A0 start-derived points test mechanics; G/H are not opened and neither is treated as truth |
| Proxy promotion | Timing, traces, four-point summaries, and canary covariance cannot freeze margins or promote Phase 8 |
| Missing stop | Exact internal/outer wall caps plus source, finite, covariance, placement, trace, seed-separation, and no-overwrite vetoes |
| Unfair comparison | Shared and independent bank mechanics use the same four points and forecast equation; no method ranking is attempted |
| Hidden assumption | The fixed horizon and forecast equation remain bound to the tested A2 implementation; A3 constants are prohibited |
| Environment mismatch | Serious path is TensorFlow/TFP `float64`, trusted GPU 1, XLA JIT, with device and TF32 provenance |
| Artifact insufficiency | Receipt binds source hashes, exact command, seeds, devices, timings, outputs, and explicit nonclaims |

## Checks

- Python compilation passed for the runner and focused test.
- Phase 8 canary tests: `6 passed`.
- The broader predictive/forecast suite completed all visible tests without a
  failure; the final authoritative aggregate is rerun after this review edit.
- `git diff --check` passed for the Phase 8 implementation paths.
- The output path was unopened before launch.

Only the frozen engineering canary is authorized. Material null/power
calibration remains closed until the canary receipt is audited and an exact
prospective ladder is added to the live plan.
