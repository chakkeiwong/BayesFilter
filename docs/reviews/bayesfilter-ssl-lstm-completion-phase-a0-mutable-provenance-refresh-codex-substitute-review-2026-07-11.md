# Phase A0 Mutable-Provenance Refresh CODEX_SUBSTITUTE_REVIEW

Date: 2026-07-11

Review type: `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude review.
Claude remained policy-unavailable; no Claude process ran and no repository
content was sent.

Reviewed paths:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md`
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`

Reviewer scope: bounded, read-only audit of the final-verifier conflict between
mutable governance provenance and immutable target evidence. The reviewer had
no mutation, runtime, A1 implementation, HMC, NeuTra, product, default,
scientific-claim, commit, or push authority.

## Finding

The strict verifier's byte-equality check for the old governance descriptors
conflicts with the A0 contract. Governance inputs are explicitly mutable,
excluded from the immutable fingerprint and all component signatures, and
allowed to change between verifier calls. The one stale A0-subplan descriptor
is therefore a fixable provenance-staleness veto, not evidence that Attempt
02's target or immutable evidence failed.

## Accepted Repair Lifecycle

1. Record the failed verifier, old/new A0-subplan hashes, old lock-file hash,
   immutable aggregate, and all five signature hashes.
2. Guardedly refresh only the stale A0-subplan descriptor under
   `source_provenance.governance_inputs`; abort if any other semantic field
   changes.
3. Preserve timestamps, run manifest, immutable fingerprint, target/probe,
   implementation, geometry, forecast, and signatures exactly.
4. Run the unchanged strict verifier and prove immutable/signature identity.
5. Record the new lock-file SHA-256 and refresh every downstream exact-file
   binding and its bounded review.
6. Run the unchanged verifier again after downstream-only updates.

Editing the harness would change an immutable member and force a fresh attempt.
Regenerating target evidence is unnecessary and is not authorized by this
review.

VERDICT: AGREE
