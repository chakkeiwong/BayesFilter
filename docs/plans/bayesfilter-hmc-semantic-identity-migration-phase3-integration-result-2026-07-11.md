# Phase 3 Result: Serialization, Replay, Redaction, And Validator Integration

Date: 2026-07-11

Status: `PASSED_TO_PHASE4_CERTIFICATE_DRAFTING`

## Direct Verdict

Phase 3 implementation and local evidence gates passed. The live deterministic
LGSSM replay reconstructs one typed transition identity, serious and smoke
execution contracts, and selection provenance. The protected sidecar, protected
input-integrity manifest, public redacted validation record, and terminal
output-integrity manifest all round-trip and cross-link.

The result remains deliberately blocked at the unchanged legacy Phase 7 gate.
The final observable exception is exactly
`DeterministicLGSSMPhase7Error: public final kernel hash mismatch`. No baseline
was adopted, no expected pin changed, and no HMC transition or sampler ran.

## Implemented Scope

- Added strict Phase 3 schemas and artifact-DAG validation in
  `bayesfilter/inference/hmc_identity_integration.py`.
- Added an opt-in evidence path in
  `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py`.
- Preserved `validate_phase7_inputs` unchanged and fail-closed.
- Added focused ownership, schema, redaction, round-trip, cross-link, exact-byte,
  terminal-manifest, live-reconstruction, and legacy-veto tests in
  `tests/test_hmc_identity_integration.py`.
- Did not modify `build_private_tuning_replay_payload` or the legacy replay.

## Persistent Evidence

Protected artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/hmc_semantic_identity_phase3_sidecar.json`
  - embedded artifact hash:
    `sha256:1de14f4bff9947463f696264a71c2f7effdc6582bf55293f98f0cf643f2f081a`
  - exact file SHA-256:
    `5505243af4bb982bf74a7fd53629f641ed4bc3c14181a274e4f8d36fff815b3c`
  - byte count: `18790`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/hmc_semantic_identity_phase3_input_integrity_manifest.json`
  - embedded artifact hash:
    `sha256:28805a32c81ea9a0a653f713ad74ded2c3b809a22df8eeffe3528967002b2a24`
  - exact file SHA-256:
    `48d3a2cb8d76e4cc6020f5e23cdef91a925d47285100d36bbe02add76fbbe57a`
  - byte count: `5018`

Public artifacts:

- `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/candidate_semantic_validation.json`
  - embedded artifact hash:
    `sha256:4b81a68e0e4204f14dfedd5d4c2668234e243752cf34d2da259e0859efc3bfbf`
  - exact file SHA-256:
    `209089f48a98b6e7e5b0ffe90f9bfec633235f2f6e62d9fdc9d32cd66a4e68e7`
  - byte count: `2856`
- `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/output_integrity_manifest.json`
  - embedded artifact hash:
    `sha256:37cd5c661eea58dab21f00dbbddc4b8ba5a170df64f88144a6e154a62b9de8b8`
  - exact file SHA-256:
    `c3fde4481bcee0fc4488662fd79e68036a3536d7cdd20450f2062db14040f0b3`
  - byte count: `1243`

The output manifest is terminal: it hashes the sidecar, input manifest, and
public record, and none of those artifacts references the output manifest.

## Candidate Identities

| Identity | Hash |
| --- | --- |
| Transition | `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` |
| Serious execution | `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4` |
| Smoke execution | `sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604` |
| Selection provenance | `sha256:8d154c31b403ae2415a83f05e4848dc9a37df59b8fa51d5c12d88f6bfe46d32f` |
| Complete tuning payload | `sha256:735e0178a76861382d49521aa726dcb3f73a8b0bd15a130775b6d74d978fc70d` |
| Legacy replay canonical payload | `sha256:14f252578e6d8e6f09194a0e1bef6af04f24fa18183dfae3620628c74f473c45` |

Named selection lineage binds bootstrap, geometry, windowed mass, fixed-mass
step, frozen-step trajectory, fresh verification/private final kernel, and the
complete tune-verify-repair loop. The selected repair attempt is index `2`.

## Candidate Checks

All exact public candidate checks are `true`:

| Check | Status |
| --- | --- |
| `transition_reconstructed` | Passed |
| `serious_execution_reconstructed` | Passed |
| `smoke_execution_reconstructed` | Passed |
| `selection_provenance_reconstructed` | Passed |
| `private_sidecar_round_trip` | Passed |
| `public_private_hashes_match` | Passed |
| `governed_inputs_unchanged` | Passed |
| `public_redaction_passed` | Passed |

The public record remains:

- `status = blocked_legacy_gate`;
- `decision = CANDIDATE_IDENTITIES_RECORDED_LEGACY_GATE_REMAINS_BINDING`;
- `legacy_gate.passed = false`;
- `legacy_gate.veto_code = LEGACY_WHOLE_PAYLOAD_HASH_MISMATCH`; and
- `legacy_gate.remains_binding = true`.

## Governed-Input Integrity

All nine governed inputs matched both the Phase 3 entry snapshot and the
post-generation snapshot:

| Input | Exact file SHA-256 |
| --- | --- |
| Fixture | `49f308e445bd621e347ab4b2e364066327ec10d491fc248955867eea634f6913` |
| XLA compile gate | `8c54c60d7d51cf5ee3d04dfa32df036fc9616c0647e399813ca846e3812e0343` |
| Geometry | `df6cde890f334fc08a445bfec55ef79b1ecda367562f954fe9b3c5ccb3c7aaf6` |
| Mass | `82574153e53bd2fbfe59e8bd3da6ab7a8261368e2408d0b9c2ee7a9186ed4b13` |
| Public kernel | `e95f1197862192ce8436ffe21ec9926519de3810d6bff4d4397a8b9caa590f43` |
| Private replay | `40c6a8d2cea8c55b5e923dace2bc3500a274b29a4abdd3ceed5b83c773f9ac0c` |
| Source tuning config | `79c7d80404c8406a118b897d1d68dd18dddd087510d0e9ac07bdf6b2b3cc7afa` |
| Phase 7 config | `746b001d3facb771b3b57b032a212683743187deb944e6fae8eb577af073c0b8` |
| Source LGSSM contract | `7cc2553e412e5d8af35c6908c2d13a4bfc818afe2fc2f333fec1d8ff09ddc0d5` |

## Checks Actually Run

Environment: deliberate CPU-hidden engineering checks with
`CUDA_VISIBLE_DEVICES=-1`; no HMC transition or experiment ran.

| Check | Result |
| --- | --- |
| Python compilation | Passed for integration module, controller, and focused tests. |
| Phase 2/3 identity, Phase 7, and serializer regression gate | `84 passed, 11 deselected`, two existing TFP `distutils` warnings. |
| Live replay reconstruction | Passed; both affine transforms and all reconstruction signatures matched. |
| Persistent evidence generation | Passed, then re-raised the exact legacy veto. |
| Protected input re-open/re-hash | Passed for all nine governed inputs. |
| Exact public allowlist and recursive secret scan | Passed. Only required `*_publicized=false` attestations mention protected categories. |
| Terminal output re-open/re-hash | Passed. |
| Forbidden ignore-list/mechanical-projection scan | No implementation match. |
| Scoped `git diff --check` | Passed. |

`ruff` was not available in the environment, so no `ruff` result is claimed.
TensorFlow emitted CUDA factory-registration and `cuInit` messages despite
`CUDA_VISIBLE_DEVICES=-1`; under repository policy these are CPU-hidden import
messages and are not GPU-health evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command class | CPU-hidden pytest, compilation, live replay reconstruction, serialization, and re-hash only |
| Python | `3.11.14` |
| NumPy | `2.1.3` |
| TensorFlow | `2.19.1` |
| TensorFlow Probability | `0.25.0` |
| CPU/GPU status | CPU-only by explicit `CUDA_VISIBLE_DEVICES=-1`; no GPU evidence |
| Data version | Governed deterministic T=120 LGSSM fixture hash above |
| Random seeds | No stochastic run; execution-contract seed values were serialized only |
| Wall time | Focused final pytest: approximately `4.0 s`; evidence generation: approximately `4.5 s` |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase3-integration-subplan-2026-07-11.md` |
| Result | This file |

Final scoped source/test exact hashes:

| File | SHA-256 |
| --- | --- |
| `bayesfilter/inference/hmc_identity_integration.py` | `48b7e67d047a4397bb6a9da9bae7c4961f6bf9161035c0114a90d92a20f3a0d0` |
| `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` | `772445871058066c4387173840f10ac6bb40fe8b90fffc5e29cf1fd4647f9ed4` |
| `tests/test_hmc_identity_integration.py` | `afec82dee09c2c50811c7082558bb60b934d91bdb106b416b81da56efdbb2d52` |
| Reviewed Phase 2 identity module | `420cc6434a8e94641c26fad74c545ddec306beb3dd7e6c37f7901c4dd5a24bd1` |
| Reviewed Phase 2 identity tests | `87383a386d7007a96e2a4f9b3e60a9d87bfa19fb706a400dcbf681f097153353` |

## Decision Table

| Decision | Status |
| --- | --- |
| Phase 3 primary criterion | Passed locally: typed candidate identities, sidecar, manifests, redaction, and live reconstruction agree. |
| Artifact-integrity vetoes | None fired; all governed inputs and new outputs re-opened and re-hashed. |
| Legacy Phase 7 gate | Failed as expected and remains binding. |
| Historical/refreshed transition equality | Unsupported and not claimed; the unavailable historical private transition cannot be reconstructed from the inspected artifacts. |
| Baseline adoption | Not authorized and not performed. |
| Phase 7 readiness | Not established. |
| Next justified action | Independent review of this result and the Phase 4 certificate-only subplan. |

## Post-Run Red Team

Strongest alternative explanation: typed reconstruction may be internally
consistent for the refreshed replay while still saying nothing about the
unavailable historical private transition. That is why this phase does not
issue a historical/refreshed equality certificate and does not adopt a
baseline.

The result would be overturned by a failed sidecar/public cross-link, changed
governed input, unclassified runtime consumer, public private-data disclosure,
or evidence that `validate_phase7_inputs` no longer emits its exact legacy
failure. The weakest evidence is historical comparison because only legacy
whole-payload pins, not the old private transition-bearing payload, are
available.

## Nonclaims

- No historical/refreshed transition equality is established.
- No legacy pin or baseline is superseded.
- No Phase 7 smoke, serious sampling, Phase 8, or NeuTra work ran.
- No posterior convergence, recovery, ranking, production, default, GPU, or
  scientific claim is made.

## Independent Review

Fresh bounded Codex substitute review found no material blocker and ended
`VERDICT: AGREE`. The review record is
`docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase3-implementation-result-codex-review-2026-07-11.md`.
