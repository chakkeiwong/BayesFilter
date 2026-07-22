# Phase 0 Result: Contract E Policy, Route Freeze, And Immediate Revocation

Date: 2026-07-13

Status: `PASSED_CLOSED`

Program ID: `contract-e-canonical-gradient-migration-20260713`

## Outcome

Phase 0 passed its local engineering and governance gate.

Contract E--Chol is now the only route eligible to seek canonical LEDH value,
total-gradient, admission, leaderboard, default-readiness, or HMC-facing status.
All raw-barycentric routes and all v1 forward/score artifacts are classified as
`historical_raw_barycentric_diagnostic_only` for canonical purposes.

The old v1 admission claims are **wrong relative to the owner-selected canonical
target** because v1 contains no mechanically bound reset identity. Existing v1
payloads remain readable as historical evidence, but a canonical-admission read
fails closed. Caller-added `contract_e_chol_v1` metadata does not change that
verdict.

No Contract E reset mathematics, schema v2, canonical factory, production
streaming composition, LaTeX correction, GPU benchmark, HMC run, or leaderboard
regeneration was performed in Phase 0.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Freeze `contract_e_chol_v1` as the sole eligible reset | `PASS`: policy and manifest agree | No policy conflict found | Implementation remains unproved | Derive normative finite program in Phase 1 | Not implementation correctness |
| Revoke canonical admission for v1 forward/score artifacts | `PASS`: central validators and shared emitter fail closed | Forged metadata, direct validator, emitter, and inclusive aggregator tests pass | Dirty other-lane harnesses remain historical code | Preserve them; replace through schema-v2 factory rather than editing in Phase 0 | Not removal of historical code |
| Preserve v1 diagnostic readability | `PASS`: formerly full v1 normalizes to historical; tiny/blocked payloads remain readable | Structural, precision, memory, row, and source checks still run before final veto | Model-specific old tests outside the focused central suite may still encode obsolete admission expectations | Phase 6 cleanup after canonical replacement, without weakening current revocation | Not backward-compatible canonical admission |
| Hand off complete mathematics before implementation | `PASS`: dedicated Phase 1 subplan exists | Complete-pullback, ridge, threshold, FD, and coverage gates are explicit | Exact derived threshold values and Kalman-gradient margin remain open | Execute Phase 1 derivation and design freeze | Not numerical/scientific validation |

## Claimed Target Versus Computed Quantity

| Item | Classification | Evidence |
| --- | --- | --- |
| Policy target | `correct`: Contract E--Chol is the only owner-authorized route eligible to seek canonical status | `AGENTS.md` Contract E policy and route-freeze manifest |
| v1 artifact route identity | `unsupported`: v1 does not bind reset semantics to an executed callable | v1 schemas at `bayesfilter/highdim/ledh_forward_contract.py` and `bayesfilter/highdim/ledh_score_contract.py` |
| Raw compact/full-history gradients as canonical Contract E gradients | `wrong relative to the stated target` | Route inventory and owner directive |
| Current dense Contract E helper | `not checked` for production correctness; retained as a later small-`N` reference | `docs/benchmarks/contract_e_reset_tf.py` inventory anchors |
| Complete Contract E total derivative | `not checked` in Phase 0 | Phase 1 handoff explicitly requires the derivation and independent checks |

## Artifacts

- Policy: `AGENTS.md`, section `Contract E Canonical LEDH Reset Policy`.
- Route manifest:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase0-route-freeze-manifest-2026-07-13.json`.
- Emergency revocation implementation:
  `bayesfilter/highdim/ledh_forward_contract.py`.
- Emergency revocation implementation:
  `bayesfilter/highdim/ledh_score_contract.py`.
- Emergency revocation implementation:
  `bayesfilter/highdim/ledh_score_artifact.py`.
- Focused new tests:
  `tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py`.
- Updated central historical-policy tests:
  `tests/highdim/test_ledh_forward_scalar_admission_guard.py`.
- Updated central historical-policy tests:
  `tests/highdim/test_ledh_score_contract_phase1.py`.
- Updated central historical-policy tests:
  `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`.
- Next subplan:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-normative-mathematics-subplan-2026-07-13.md`.

## Checks Actually Run

CPU-only choice: `CUDA_VISIBLE_DEVICES=-1` was set before Python/TensorFlow
imports for all focused tests. No GPU evidence was sought.

Final focused command:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py
```

Result: `75 passed, 2 warnings` in `3.98s`. The warnings are pre-existing
TensorFlow Probability `distutils.version` deprecations.

Additional checks:

- Python compilation of the three contract modules and new test: passed.
- JSON parse and semantic assertions over the route manifest: passed.
- Manifest path existence check: passed.
- `git diff --check` over code, tests, plans, and reviews: passed.
- Policy preservation check: the pre-existing `Academic Research Governance
  Profile` remains before the additive Contract E section: passed.
- Hash recheck of dirty/untracked other-lane model and harness files: unchanged
  from Phase 0 inventory; no Phase 0 edits made to them.

## Repair Record

The first full central contract run returned `15 failed, 54 passed`. Three
failures were obsolete assertions that v1 full admission should succeed. Twelve
were a real error-ordering regression: the emergency veto fired before malformed
historical payloads reached their structural/precision/memory checks.

The validator was repaired so all historical schema checks execute first and the
canonical-admission veto executes last. The next run returned `4 failed, 65
passed`; all four were the intended obsolete success assertions. Those clean
central tests were updated to assert historical normalization and fail-closed
canonical admission. One new emitter fixture then failed because it omitted
required precision/memory fields; the fixture was made structurally valid so it
reaches the intended final veto. The final `75`-test run passed.

This repair did not weaken the revocation. It preserved diagnostic validation
quality before returning the canonical policy verdict.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at inventory | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Branch | `main` |
| Command | Final focused pytest command above |
| Environment | `tf-gpu`; Python `3.11.14` |
| CPU/GPU status | CPU only; `CUDA_VISIBLE_DEVICES=-1`; GPU intentionally hidden |
| Data version | Existing checked repository JSON fixtures; no data regenerated |
| Random seeds | Existing serialized fixture seeds `81120` through `81124`; no stochastic execution |
| Wall time | Final test `3.98s`; Phase 0 visible work remained within campaign allocation |
| Output artifacts | Policy, route manifest, result, tests, Phase 1 subplan, ledger |
| Plan file | `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase0-policy-route-freeze-subplan-2026-07-13.md` |
| Result file | This file |

## Preserved Dirty Work

The pre-existing `AGENTS.md` addition was preserved and the Contract E policy was
inserted additively. The dirty model-specific score files, dirty experimental
transport file, and untracked compact GPU harness from the parallel leaderboard
lane were read only for inventory and hash-checked afterward. Their hashes did
not change during Phase 0.

## Remaining Gaps

- The exact finite Contract E equations and total pullback have not yet been
  independently derived in repository notation.
- Ridge target semantics, active sets, and scale-aware numerical vetoes remain
  to be frozen.
- FD step-ladder and near-zero rules remain to be derived.
- The coverage-preserving paired LGSSM statistical design remains to be frozen.
- A scientifically justified Kalman-gradient equivalence margin remains open;
  historical `1%` is not automatically reused.
- Schema v2, canonical route factory, cloud module, streaming composition,
  production feasibility, canonical graph, docs, LGSSM results, nonlinear rows,
  leaderboard, and integrity audit remain unimplemented or unrun.

## Handoff Verdict

Local Phase 0 gate: `PASS`.

The bounded fresh-Codex handoff review found no material issue and returned
`VERDICT: AGREE`. Phase 1 may begin. The review is recorded at
`docs/reviews/bayesfilter-contract-e-canonical-gradient-migration-phase0-result-phase1-handoff-codex-review-2026-07-13.md`.
