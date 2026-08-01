# Phase 7 Attempt-2 Proposal Exact Codex Review

Date: 2026-07-13

Role: fresh bounded read-only Codex substitute reviewer. Claude was not invoked
because the managed external-disclosure rejection for this lane remains binding.

Scope:
`docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json`
only.

## Round 1 Finding

The reviewer initially returned `VERDICT: REVISE` because ordinary sorted,
compact JSON without the top-level `artifact_hash` produced
`sha256:f77a63047671abaa0a1845390a0da73260901e721423ea9e2a9b54edc4a326dd`
rather than the embedded proposal hash.

That calculation used the wrong canonicalization rule. BayesFilter artifact
hashing first applies the type-tagged `_strict_json_value` normalization in
`bayesfilter/inference/hmc_identity.py`, then serializes the normalized value.
The authoritative project function reproduced the embedded hash exactly:

`sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4`.

The strict parser accepted the proposal. Rebuilding from the pinned
`SeriousInheritedEvidenceSession` produced identical serialized bytes, file
SHA-256
`cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7`,
and byte count `39904`. A duplicate-key rejecting JSON parse also passed.

## Final Findings

No remaining findings. The reviewer withdrew the ordinary-JSON hash finding
after applying the declared project canonicalization.

Within the exact file, the proposal:

- preserves the reviewed transition and serious execution identities;
- preserves the fixed two-worker, four-chain, CPU-hidden, float64 Host-XLA/JIT
  runtime contract;
- binds the complete attempt-1 terminal graph and required absences;
- uses disjoint attempt-2 paths and exclusive creation for all active outputs;
- proposes exactly one launch;
- cannot reuse the consumed attempt-1 authority or claim; and
- grants no Phase 8 or NeuTra authority.

Review agreement grants no authority and does not establish runtime,
convergence, recovery, GPU, production, or scientific evidence.

`VERDICT: AGREE`
