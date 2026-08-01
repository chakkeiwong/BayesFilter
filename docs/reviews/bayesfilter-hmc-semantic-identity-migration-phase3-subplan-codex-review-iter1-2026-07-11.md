# HMC Semantic Identity Migration Phase 3 Subplan Review, Iteration 1

Date: 2026-07-11

Review type: fresh findings-first Codex substitute plan review.

## Findings

1. The original bundle boundary could become circular by embedding the exact
   replay file hash in the file being hashed. A separate immutable-replay
   sidecar/envelope and external exact-file manifest are required.
2. Candidate checks cannot execute only after the known binding legacy check,
   because the legacy mismatch raises first. Candidate and legacy diagnostics
   must be independently evaluated, persisted, and then resolved with the
   legacy veto still controlling the final decision.
3. Temporary evidence is insufficient for Phase 4. Phase 3 needs a persistent
   protected sidecar, machine-readable public validation record, and pre/post
   input-integrity manifest.
4. Mutation tests need an explicit ownership oracle stating which identities
   change and which remain stable.
5. Public redaction needs an exact allowlisted schema and explicit authority for
   every disclosed file hash/byte count; recursive scans are defense in depth.

These are fixable plan defects. They do not authorize artifact rewriting,
legacy-veto removal, baseline adoption, or runtime.

VERDICT: REVISE
