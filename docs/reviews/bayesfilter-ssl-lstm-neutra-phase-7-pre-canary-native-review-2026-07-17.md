# SSL-LSTM NeuTra Phase 7 Pre-Canary Native Review

Date: 2026-07-17

Verdict: `AGREE_STAGE_A_CANARY_ONLY`

Reviewed paths:

- `docs/plans/bayesfilter-ssl-lstm-neutra-phase-7-retained-admission-plan-2026-07-16.md`
- `docs/benchmarks/run_ssl_lstm_neutra_phase7_retained_admission_2026_07_17.py`
- `tests/test_ssl_lstm_neutra_phase7_retained_admission.py`
- existing retained archive API and focused tests in
  `bayesfilter/inference/hmc.py` and
  `tests/test_hmc_retained_sample_archive_runner.py`

## Findings And Repairs

1. The first post-archive value/score audit attempted to pass a Python adapter
   through a TensorFlow input signature. This would not create a valid XLA
   program. Repaired to build one fixed-shape per-chart XLA closure and reuse it
   for all equal-size segments; trace counts are checked.
2. The first partial-admission test enforced the G/H boundary only in test-side
   control flow. Repaired by adding the executable
   `admitted_cross_replication_stability` fail-closed gate.
3. The gate insertion initially split the underlying stability function,
   causing it to return `None`. The focused test caught this placement error;
   the gate was moved after the complete function and all checks were rerun.

No unresolved material finding remains for Stage A.

## Audit Disposition

| Question | Disposition |
| --- | --- |
| Exact baseline and target | Bound to the passing Phase 5 receipt, Phase 6 final receipt, immutable G/H payloads, transport hashes, current source hashes, and four reconstructed A0 starts |
| Promotion metric discipline | Canary acceptance/convergence cannot promote or veto; retained admission gates are prospective and acquisition remains code-closed |
| Stop conditions | Evidence-invalidity, nonfinite value/score/core telemetry, lineage failure, positive native divergence, unmoved chain, GPU placement failure, and wall-cap exhaustion are explicit |
| Fair comparison | G/H use the same canary mechanics; the later retained ladder must be common and timing-derived before acquisition opens |
| Artifact sufficiency | Immutable sample, final-state, final-target, and manifest sidecars are hash-verified; continuation lineage is explicit; post-archive values and scores are recomputed |
| Privacy | Public canary receipt contains hashes and diagnostics but no raw samples or private paths; private manifests retain required readback authority |
| Statistical interpretation | No oracle or ranking; G/H stability is mapped-`theta` only and inaccessible until both independently admit |

## Checks

- `17 passed` across the new Phase 7 tests and existing retained-archive tests.
- Python compile checks passed.
- `git diff --check` passed for the three Phase 7 implementation paths.
- Exact upstream receipt and current source hashes revalidated.

The authorized next action is the trusted GPU/XLA Stage A timing/mechanics
canary only. Its samples are excluded permanently from retained evidence. The
retained acquisition entry remains closed until the measured ladder and
resource cap are reviewed and frozen.
