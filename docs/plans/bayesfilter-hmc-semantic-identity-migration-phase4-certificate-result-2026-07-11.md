# Phase 4 Result: Migration Certificate And Baseline Approval Stop

Date: 2026-07-11

Status: `PASSED_CERTIFICATE_ONLY_AWAITING_HUMAN_BASELINE_ADOPTION_APPROVAL`

## Direct Verdict

The certificate-only portion of Phase 4 passed its local engineering and
evidence gates. The protected certificate, redacted public proposal, and
terminal output manifest classify the available historical and refreshed
evidence without claiming that the unavailable historical private transition
was reconstructed.

Baseline adoption did not occur. The certificate status is
`proposal_only_pending_human_approval`, adoption is
`pending_human_approval`, and the active gate is
`legacy_gate_remains_binding`. The unchanged validator still raises exactly
`DeterministicLGSSMPhase7Error: public final kernel hash mismatch`.

## Implemented Scope

- Added strict proposal-only certificate schemas, constructors, parsers,
  source verification, redaction enforcement, and terminal-manifest handling
  in `bayesfilter/inference/hmc_identity_migration_certificate.py`.
- Added real-artifact and adversarial tests in
  `tests/test_hmc_identity_migration_certificate.py`.
- Bound the certificate to the exact active Phase 7 config, refreshed public
  kernel, refreshed private replay, and all four Phase 3 evidence artifacts.
- Required the three primary sources to be the exact files protected by the
  Phase 3 governed-input manifest, not compatible copies.
- Preserved all historical config pins, governed inputs, Phase 3 artifacts,
  `validate_phase7_inputs`, and runtime authority unchanged.

## Persistent Evidence

Protected migration certificate:

- path:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/hmc_semantic_identity_migration_certificate.json`
- embedded artifact hash:
  `sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`
- exact file SHA-256:
  `6b99187ed1e0444385caccc0745945431db69bfc4ca3cee54c835a07fc9c93bd`
- byte count: `13249`

Public redacted migration proposal:

- path:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/migration_certificate_proposal.json`
- embedded artifact hash:
  `sha256:ce7adc4deab1c813e5150f348e4bd7268ff50a8154b9c8af3f3179e82a263f18`
- exact file SHA-256:
  `9a495e0d555a463c5bb727f3539563bdbc5492d55785a91e648426268449b0af`
- byte count: `2730`

Terminal Phase 4 output-integrity manifest:

- path:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/migration_certificate_output_manifest.json`
- embedded artifact hash:
  `sha256:068e3afa1749e61c7a28ad63130d51020b6186b9e5f2c7eed1729112107830a7`
- exact file SHA-256:
  `8dde8fdd40fc62d8601e559e36e8967632a53dc23bddd9c1beb9d7d5f6dadcbd`
- byte count: `923`

The terminal manifest hashes the protected certificate and public proposal.
Neither hashed output references the terminal manifest.

## Classification Result

| Comparison | Classification | Evidence meaning |
| --- | --- | --- |
| Fixture hash | `equal` | Historical and refreshed pins match. |
| XLA compile hash | `equal` | Historical and refreshed pins match. |
| Geometry hash | `equal` | Historical and refreshed pins match. |
| Mass hash | `equal` | Historical and refreshed pins match. |
| Base adapter signature | `equal` | Historical and refreshed signatures match. |
| Selected step hash | `equal` | Both records contain `ec7db59e...0ab40`. |
| Public final-kernel whole-payload hash | `different` | Historical `8ddf25a3...2ffff`; refreshed `07910941...e7fbf`. |
| Private-loop whole-payload hash | `different` | Historical `391558a9...42039a`; refreshed `2823e200...c0168f`. |
| Selected-trajectory whole-payload hash | `different` | Historical `6eaf7a56...bd13b3`; refreshed `3f4b3368...aeb04b`. |
| Historical/refreshed typed transition identity | `unsupported` | The historical private transition-bearing payload is unavailable. |
| Historical/refreshed typed execution identity | `unsupported` | No historical typed execution contract exists. |
| Refreshed internal candidate reconstruction | `equal` | Phase 3 live reconstruction, sidecar, and public reference agree for the refreshed replay only. |
| Posterior convergence/recovery | `not_checked` | No sampler transition ran. |
| Baseline adoption | `not_checked` | Human approval has not been given. |

There are seven `equal`, three `different`, two `unsupported`, and two
`not_checked` classifications. Equal selected mechanics and refreshed internal
consistency do not establish historical typed-transition equality.

## Typed Candidate Identities

| Identity | Hash |
| --- | --- |
| Transition | `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` |
| Serious execution | `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4` |
| Smoke execution | `sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604` |
| Selection provenance | `sha256:8d154c31b403ae2415a83f05e4848dc9a37df59b8fa51d5c12d88f6bfe46d32f` |
| Complete tuning payload | `sha256:735e0178a76861382d49521aa726dcb3f73a8b0bd15a130775b6d74d978fc70d` |
| Legacy replay canonical payload | `sha256:14f252578e6d8e6f09194a0e1bef6af04f24fa18183dfae3620628c74f473c45` |
| Legacy replay exact file | `40c6a8d2cea8c55b5e923dace2bc3500a274b29a4abdd3ceed5b83c773f9ac0c` |

These are refreshed candidate identities. They are not a reconstruction of the
historical private transition.

## Checks Actually Run

Environment: deliberate CPU-hidden engineering checks with
`CUDA_VISIBLE_DEVICES=-1` and `MPLCONFIGDIR=/tmp/matplotlib`. No HMC transition,
worker, smoke, sampler, or experiment ran.

| Check | Result |
| --- | --- |
| Skeptical Phase 4 plan audit | Passed after source-ownership and cross-link gaps were repaired. |
| Python compilation | Passed for certificate module and focused tests. |
| Phase 4 adversarial suite | `11 passed`, two existing TFP `distutils` warnings. |
| Combined Phase 2-4/controller gate | `92 passed`, two existing TFP `distutils` warnings. |
| Phase 3 governed-input verification | Passed for all nine exact files. |
| Phase 3 terminal-output verification | Passed for all three outputs. |
| Phase 4 live-source verification | Passed for all seven sources. |
| Phase 4 terminal-output verification | Passed for both outputs. |
| Public exact-key secret scan | No forbidden exact key found. |
| Approval-state mutation tests | Passed; rehashed `approved`, `adopted`, active-gate, decision, and approval-boolean mutations were rejected. |
| Source path/byte substitution tests | Passed; identical-byte copies and whitespace-tampered sources were rejected. |
| Terminal-manifest byte tamper test | Passed; valid JSON with changed bytes was rejected. |
| Legacy validator check | Still fails exactly with `public final kernel hash mismatch`. |
| Forbidden projection/bypass scan | No identity implementation uses an ignore list or mechanical field projection. |
| Scoped `git diff --check` | Passed. |

`ruff` is unavailable in this environment, so no `ruff` result is claimed.
TensorFlow emitted CPU-hidden CUDA factory-registration messages on import;
these are not GPU-health or GPU-execution evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command class | CPU-hidden compilation, pytest, JSON construction, source re-hash, redaction scan, and unchanged-validator check only |
| Python | `3.11.14` |
| TensorFlow / TFP | Existing environment; imported by focused tests, no transition executed |
| CPU/GPU status | CPU-only by explicit `CUDA_VISIBLE_DEVICES=-1`; no GPU evidence |
| Data version | Nine Phase 3 governed inputs, exact hashes owned by the protected input manifest |
| Random seeds | No stochastic run; configured seeds were serialized only |
| Wall time | Phase 4 suite `7.88 s`; combined gate `9.88 s`; artifact generation/revalidation `7.0 s` |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase4-certificate-subplan-2026-07-11.md` |
| Result | This file |

Scoped source/test exact hashes before independent review:

| File | SHA-256 |
| --- | --- |
| `bayesfilter/inference/hmc_identity_migration_certificate.py` | `8dab913ead8aef72573105a6fb09d3f3bd3222e0836b2a37ca98ef0e59b1788a` |
| `tests/test_hmc_identity_migration_certificate.py` | `3a5a1bab7803c83548f2cb1e32667cc892c6cf201ff33022d6e3ce1f99fce421` |

## Decision Table

| Decision | Status |
| --- | --- |
| Certificate primary criterion | Passed locally: classifications, source ownership, strict schemas, redaction, and tamper gates agree. |
| Historical typed-transition equality | `unsupported`; it is not claimed. |
| Artifact-integrity vetoes | None fired. |
| Public-disclosure veto | Did not fire. |
| Legacy Phase 7 gate | Still binding and still fails exactly at the public final-kernel whole-payload hash. |
| Baseline adoption | Not approved and not performed. |
| Phase 7 readiness | Not established. |
| Next justified action | Independent review of this result and the Phase 5 subplan, then stop for exact human baseline-adoption approval. |

## Exact Adoption Proposal

After independent review, the human may approve exactly
`PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION` bound to protected certificate
artifact hash
`sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`.

That approval would authorize Phase 5 local implementation and adversarial
validation of a new versioned Phase 7 config and validator that:

1. adopts the refreshed transition, serious/smoke execution, selection
   provenance, complete tuning-payload, and replay-integrity identities listed
   above;
2. preserves the historical Phase 7 config and all legacy whole-payload hashes
   as immutable historical audit evidence;
3. keeps exact source/file/canonical artifact-integrity checks fail-closed;
4. rebuilds typed identities from the same live replay/runtime objects used by
   Phase 7 rather than trusting copied hash fields; and
5. runs CPU-hidden local preflight and adversarial tests only.

It would not authorize an HMC transition, two-worker smoke, serious burn-in or
sampling, Phase 8, NeuTra, a default/product-policy change, package/network
action, historical typed-identity equality claim, posterior claim, or
scientific claim. Those boundaries remain separate.

## Post-Run Red Team

Strongest alternative explanation: the refreshed typed identity may be
internally correct while the unavailable historical transition was materially
different. The certificate explicitly classifies that comparison as
`unsupported`, so adoption is a reviewed new-baseline decision, not an
equivalence finding.

The result would be overturned by a changed governed input, a source reference
that does not reopen to the recorded bytes, a public private-data disclosure,
an unclassified transition/execution input, or evidence that certificate
construction changed validator authority. None occurred in the local checks.

The weakest evidence remains historical comparison because the old private
transition-bearing payload is unavailable. No later test can repair that
historical evidence gap without locating a genuine historical source artifact.

## Nonclaims

- No historical/refreshed typed transition or execution equality is proved.
- No baseline, expected pin, validator, or active gate was changed.
- No HMC transition, smoke, sampling, Phase 8, or NeuTra work ran.
- No posterior convergence, recovery, ranking, production, default, GPU, or
  scientific claim is made.

## Independent Review

Fresh bounded independent Codex substitute review found no material blocker and
ended `VERDICT: AGREE`. After the supervisor clarified the Phase 5
V2-config/adoption-record graph, focused re-review also ended
`VERDICT: AGREE`.

The review record is
`docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase4-result-phase5-subplan-codex-review-2026-07-11.md`.
The managed Claude disclosure rejection remains binding and was not retried.
