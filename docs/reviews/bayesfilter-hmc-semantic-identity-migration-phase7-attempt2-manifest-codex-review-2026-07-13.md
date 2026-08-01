# Phase 7 Attempt-2 Terminal Manifest Exact Codex Review

Date: 2026-07-13

Role: fresh bounded read-only Codex substitute reviewer. Claude was not invoked
because the managed external-disclosure rejection for this lane remains binding.

Scope:
`docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json`
only.

## Round 1 Finding

The reviewer initially returned `VERDICT: REVISE` because ordinary sorted,
compact JSON without the top-level `artifact_hash` produced
`sha256:0d247b59ea543fe7663b1e26922f3757341760dc5f5f3954f4656ed2a97921f4`
rather than the embedded manifest hash.

That calculation used the wrong canonicalization rule. BayesFilter artifact
hashing uses the type-tagged `_strict_json_value` normalization before compact
serialization. The authoritative project function reproduced the embedded
manifest hash exactly:

`sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e`.

The strict parser accepted the manifest. Rebuilding it from the exact proposal
reference produced identical serialized bytes, file SHA-256
`e7aa19fb234dd3eff960e97c0c50a643c98663a6e87c98170a9c0f09c9a991b6`,
and byte count `869`. A duplicate-key rejecting JSON parse also passed.

The proposal reference independently matched:

- embedded artifact hash
  `sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4`;
- file SHA-256
  `cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7`;
- byte count `39904`; and
- attempt-2 proposal schema.

## Final Findings

No remaining findings. The reviewer withdrew the ordinary-JSON hash finding.
The manifest has the exact closed four-field shape, sets
`terminal_manifest=true`, binds only the exact attempt-2 proposal, and grants
no authority, claim, output reservation, runtime, Phase 8, or NeuTra action.

`VERDICT: AGREE`
