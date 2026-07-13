# SSL-LSTM Completion Phase A0 Harness Codex Substitute Review

Date: 2026-07-11

Reviewer type: `CODEX_SUBSTITUTE_REVIEW`

Review strength: weaker than requested Claude Opus review; Claude external
review was policy-unavailable and Claude liveness was not tested.

Exact reviewed path:
`docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`

## Review Shape

The first whole-file attempt and several broad slice attempts stalled without
verdicts and were interrupted. They are not substantive review rounds. Fresh
bounded reviewers then inspected complementary lifecycle and verifier slices.

## Material Findings And Repairs

1. The initial verifier self-hashed target constants and probe outputs without
   binding enough fields to canonical target data. It also permitted CLI path
   aliasing. The harness now binds the exact chart, fixture, target/filter/prior
   constants, implementation provenance, fixed A0 paths, exact tensor names and
   JSON numeric types, historical probe anchors, and complete probe algebra.
2. Observation and likelihood evidence could still be fabricated around the
   stored historical totals. The strict verifier now performs a fresh CPU-hidden
   historical target construction and compares observations and complete probe
   payloads byte-for-byte, then checks the runtime dependency closure.
3. Generation originally used one opening warm-up and recorded closing before
   later payload construction. It now requires two consecutive manifest-
   matching warm-up cycles, opens only after that fixed point, constructs all
   evidence/signatures, and performs the closing module-set and immutable rehash
   immediately before excluded lock/log writes.
4. Scalar tensor canonicalization initially risked rank promotion. Rank-zero
   arrays are now preserved as `shape=[]` throughout generation and verification.

## Final Review

The final one-path micro-review found no material issue in generator/verifier
compatibility, replay equality, immutable fingerprint lifecycle, or fixed
CPU-hidden/write boundaries.

Residual risk: runtime may still expose an environment or historical-artifact
mismatch. Such a mismatch is a fail-closed A0 result, not permission to relax
the contract or substitute an environment.

VERDICT: AGREE
