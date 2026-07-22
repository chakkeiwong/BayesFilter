# Phase 4 Subplan: Migration Certificate And Baseline Approval Boundary

Date: 2026-07-11

Status: `REVIEWED_READY_FOR_CERTIFICATE_DRAFTING`

## Phase Objective

Construct and review a versioned migration certificate that classifies exactly
what the available historical and refreshed evidence proves, differs on, or
cannot establish. Phase 4 may prepare a baseline-adoption proposal, but it must
stop before changing any expected pin, active validator, config, artifact, or
runtime authority unless explicit human baseline-adoption approval is recorded
after the certificate review.

## Entry Conditions Inherited From Phase 3

- The Phase 3 implementation/result and this subplan pass independent review.
- The four Phase 3 evidence artifacts re-open, validate, and match their exact
  terminal-manifest hashes.
- All nine governed inputs still match the protected input manifest.
- `validate_phase7_inputs` still fails exactly with
  `public final kernel hash mismatch`.
- The public candidate record remains `blocked_legacy_gate`; no baseline has
  been adopted and no HMC runtime is authorized.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Preserve historical pins, refreshed legacy hashes, and typed candidate identities as separate fields; do not overwrite or relabel any of them. |
| Unsupported equality | The old private replay is unavailable. Classify historical/refreshed transition equality as `unsupported`, not equal or approximately equal. |
| Proxy promoted | Matching visible mechanics, acceptance, or lineage is explanatory only and cannot certify transition equality. |
| Certificate smuggles adoption | Certificate status is `proposal_only_pending_human_approval`; no expected pin, config, validator, or artifact changes are allowed in certificate construction. |
| Missing stop | Stop after reviewed certificate/proposal and request explicit human baseline-adoption approval. |
| Stale evidence | Re-open and re-hash all Phase 3 artifacts and governed inputs immediately before certificate construction. |
| Artifact cannot answer question | Certificate has separate `equal`, `different`, `unsupported`, and `not_checked` classifications with source anchors and bounded conclusions. |
| Boundary drift | No HMC transition, smoke, sampling, Phase 8, NeuTra, package/network action, or default-policy change is permitted. |

Audit verdict: `PASS_FOR_CERTIFICATE_DRAFT_ONLY_AFTER_PHASE3_REVIEW`.

## Required Artifacts

- Protected migration certificate:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/private_diagnostics/hmc_semantic_identity_migration_certificate.json`.
- Public redacted certificate reference/proposal:
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/migration_certificate_proposal.json`.
- Terminal Phase 4 output-integrity manifest that hashes the protected
  certificate and public proposal without circular references.
- Focused strict-schema, classification, redaction, tamper, source-hash, and
  approval-boundary tests.
- Phase 4 certificate result/approval-stop record.
- Drafted and reviewed Phase 5 adversarial-validation subplan, which remains
  non-executable until any required baseline-adoption approval is resolved.

## Required Certificate Contract

The protected certificate must bind:

1. exact Phase 3 sidecar, input-manifest, public-record, and output-manifest
   embedded and file hashes;
2. every historical Phase 6AA expected pin and every refreshed legacy hash;
3. the refreshed transition, serious execution, smoke execution, selection
   provenance, complete tuning-payload, and exact replay-integrity hashes;
4. the historical evidence availability statement;
5. a closed classification table whose only values are `equal`, `different`,
   `unsupported`, and `not_checked`;
6. the proposed migration action and the explicit human-approval requirement;
7. exact nonclaims and an embedded artifact hash.

Required classifications for the current evidence:

| Question | Required classification | Reason |
| --- | --- | --- |
| Historical vs refreshed selected step hash | `equal` | Both inspected records carry `ec7db59e...0ab40`. |
| Historical vs refreshed public final-kernel whole-payload hash | `different` | Historical `8ddf25a3...d92ffff`; refreshed `07910941...e7fbf`. |
| Historical vs refreshed private loop whole-payload hash | `different` | Historical `391558a9...f42039a`; refreshed `2823e200...c0168f`. |
| Historical vs refreshed selected trajectory whole-payload hash | `different` | Historical `6eaf7a56...bd13b3`; refreshed `3f4b3368...aeb04b`. |
| Historical vs refreshed typed transition identity | `unsupported` | No historical private transition-bearing payload is available for typed reconstruction. |
| Historical vs refreshed execution identity | `unsupported` | No historical typed execution contract exists. |
| Refreshed replay internal candidate reconstruction | `equal` | Phase 3 live reconstruction and sidecar/public cross-links agree for the refreshed replay only. |
| Posterior convergence/recovery | `not_checked` | No sampler ran. |
| Baseline adoption | `not_checked` | Human decision not yet made. |

The public proposal must be a strict redacted reference. It may expose bounded
schemas, classifications, hashes already public or approved for public
reference, the proposed action, approval status, and nonclaims. It must not
expose observations, transform arrays, HMC mechanics, seeds, runtime versions,
private paths, stage lineage, or adapter/mass signatures.

## Required Checks, Tests, And Reviews

- Re-run Phase 3 parsers and exact byte verification before constructing the
  certificate.
- Re-run the nine governed-input hashes against the protected manifest.
- Verify historical/refreshed hashes from their source artifacts rather than
  copying values only from prose.
- Strictly reject missing/extra certificate fields, unknown classifications,
  reordered fixed nonclaims, altered approval state, or altered fixed decision.
- Reject a certificate that classifies unavailable historical typed identity as
  `equal`, `different`, or `not_checked`; it must be `unsupported`.
- Reject any adoption state other than `pending_human_approval` during the
  authorized certificate-only portion.
- Verify public/private certificate references and terminal-manifest acyclicity.
- Run recursive public secret scans as defense in depth.
- Run Python compilation, focused pytest, forbidden bypass/repin scan, and
  scoped `git diff --check`.
- Obtain a fresh independent Codex implementation/result review because the
  managed Claude disclosure rejection remains binding and must not be retried.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can available evidence support a precise migration proposal without pretending the unavailable historical private transition was reconstructed? |
| Exact baseline | Historical Phase 6AA pins, refreshed governed artifacts, and the reviewed Phase 3 evidence bundle. |
| Primary criterion | Certificate/proposal classify each comparison correctly, bind exact source evidence, pass strict/redaction/tamper checks, and retain pending human approval. |
| Continuation vetoes | Unsupported equality, source-hash mismatch, changed governed input, invalid Phase 3 artifact, public disclosure, silent repin, validator switch, or adoption without explicit approval. |
| Explanatory only | Visible mechanics, acceptance, timings, and policy lineage. |
| Not concluded | No historical typed-transition equality, baseline adoption, Phase 7 readiness, convergence, recovery, production/default, GPU, NeuTra, or scientific claim. |
| Preserving artifact | Protected certificate, public proposal, terminal output manifest, Phase 4 result, and independent review record. |

## Forbidden Claims And Actions

- Do not claim historical/refreshed typed transition equality.
- Do not equate equal selected-step hashes or visible mechanics with complete
  transition equality.
- Do not change `expected_hashes`, `validate_phase7_inputs`, any governed input,
  or any current legacy artifact.
- Do not mark adoption approved, adopted, or active without explicit human
  approval after the reviewed certificate is presented.
- Do not launch HMC, actual-target smoke, serious sampling, Phase 8, or NeuTra.
- Do not install packages, fetch network resources, change default policy, or
  touch unrelated LEDH/QR work.

## Exact Next-Phase Handoff Conditions

The certificate-only portion of Phase 4 closes when:

1. Phase 3 evidence and governed inputs revalidate;
2. protected and public certificate artifacts pass all strict, source, redaction,
   and tamper checks;
3. the certificate states historical typed-transition equality is unsupported;
4. adoption remains `pending_human_approval`;
5. the Phase 4 result and Phase 5 subplan pass independent review; and
6. the supervisor stops and presents the exact adoption proposal to the human.

No baseline adoption or Phase 5 execution may occur until the human explicitly
approves the exact reviewed proposal. If approval is denied or absent, write a
visible approval-stop handoff and keep the legacy gate binding.

## Stop Conditions

- Any Phase 3 artifact or governed input fails revalidation.
- The historical source evidence cannot support one of the required
  classifications.
- A certificate implementation could switch the active gate before approval.
- Public output would expose a protected value.
- Scoped HMC files change unexpectedly.
- The same substantive review blocker remains after five repair rounds.
- Certificate construction completes and human baseline-adoption approval is
  not yet recorded.
