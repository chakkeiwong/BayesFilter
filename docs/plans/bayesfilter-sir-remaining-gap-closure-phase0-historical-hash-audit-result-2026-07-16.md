# SIR Gap Closure Phase 0 Result: Historical Contract E Hash Audit

Date: 2026-07-16

Status: `PASS_HISTORICAL_ARTIFACT_SOURCE_DRIFT_CLASSIFIED`

Plan:
`docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

## Verdict

The persisted Phase 3 aggregate tensor hash is not corrupted and TensorFlow
serialization is not the cause. The regression compared a historical artifact
against a later implementation while ignoring the source closure recorded in
the Phase 3 run manifest.

The persisted artifact contains output hash
`cf6d98f759d60439a02debda4b8e7207c44d7c7d420d09217befcfaf9345c5c4`.
The current implementation recomputes
`8b9f7895d33b98321d4bad947001d3f2a464da237410c69765d79541c969669a`.

## Provenance

| Item | July 13 manifest | Current source |
| --- | --- | --- |
| `ledh_contract_e_reset_tf.py` | `8b9829bf759a11f2f07f8e1de26d9a26fd184918bfc239ed9eaf7a3e81547e23` | `5a226b53f4a881a1b66cee00902dcd007c82de3c01e3440101c111c5095ee023` |
| Phase 3 test module before this compatibility repair | `ad7ad2ab41e0f24897b66798ac9531c0206841e3ea436ee15450d44b593a3f79` | matched during diagnosis; this phase then changed the test to enforce source closure |
| parity artifact | `0dd07fd9b7dd15c3227e2977d563ec1b6c35db6fd6bd2183b40ee7515baa4c56` | same |

Later Phase 8 artifacts independently record the current reset source hash
`5a226b53...`, confirming that the implementation changed after the Phase 3
certificate was written.

## Repair

The persisted-artifact test now performs exact recomputation only when the
current reset source hash equals the manifest-bound source hash. When the source
has drifted, it asserts that recomputation differs and preserves the artifact as
historical evidence. The old JSON certificate is not rewritten or silently
upgraded.

## Nonclaims

- This audit does not certify the current reset implementation from the old
  Phase 3 artifact.
- It does not establish current general-chart numerical adequacy.
- It does not change canonical, HMC, GPU, or leaderboard status.
