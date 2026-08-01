# Phase 2 Result: Schema V2 And Canonical Factory

Date: 2026-07-13

Status: `PASSED_CLOSED_SCHEMA_ONLY_NOT_ADMITTED`

Program ID: `contract-e-canonical-gradient-migration-20260713`

## Outcome

Phase 2 implemented separate forward and score schema-v2 candidates plus a
factory-bound semantic identity. Identity is computed from repository-owned
route semantics, realized prepared tensors, exact callables, loaded/current
source and code, behavior-bearing globals, dependency closure, TensorFlow
wrapper/JIT settings, and reviewed external package provenance.

The production factory is intentionally empty and write-once. Public issuance
therefore fails as unregistered, and public artifact builders reject identities
from the private schema-test factory. Every Phase 2 artifact remains
`admitted=false` and `canonical_admission_eligible=false`.

Current consumers are v1-only and reject v2 as an unknown/unadmitted schema.
That is fail-closed behavior, not completed v2-aware consumer migration.

## Claimed Target And Quantity Checked

The claimed Phase 2 target is mechanical identity and schema integrity. The
quantity checked is whether actual objects and prepared values produce one
immutable, reproducible identity and whether forged, raw, incomplete, mutable,
or mismatched inputs fail closed. This is `correct` on the tested paths.

Production Contract E reset math, total gradients, streaming composition,
same-scalar derivatives, Kalman agreement, and nonlinear validity were not
computed and remain `not checked`.

## Repair Loop

| Review | Verdict | Material repair |
| --- | --- | --- |
| Identity packet | `AGREE` | No material identity-coverage finding. |
| Artifact iteration 1 | `REVISE` | Replaced mutable canonical module dictionaries with private immutable tuples and fresh public copies. |
| Artifact iteration 2 | `REVISE` | Removed nested caller aliasing by reconstructing normalized state. |
| Artifact iteration 3 | `AGREE` | Artifact immutability and fail-closed admission converged. |
| Registry iteration 1 | `REVISE` | Mapping proxies did not stop wholesale registry replacement. |
| Registry iteration 2 | `AGREE` | Write-once factory plus mapping proxies and post-attempt issuance checks converged. |

Claude was not retried because the platform had already blocked repository
disclosure. Fresh bounded Codex reviewers were the approved substitute.

## Checks

- Final CPU-hidden Phase 0/2 compatibility suite: `104 passed, 2 warnings in 4.03s`.
- Phase 2 focused schema suite: `29 passed, 2 warnings`.
- Python compilation: passed.
- Manifest JSON parsing and exact post-repair source hashes: passed.
- Scoped `git diff --check`: passed.
- Final bounded reviews: `VERDICT: AGREE`.

The warnings are pre-existing TensorFlow Probability `distutils.version`
deprecations. No GPU, benchmark, HMC, nonlinear, or leaderboard command ran.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close schema/factory phase | Pass | No forgery, mutability, or v1-regression veto remains in focused tests | Conservative dependency walker supports only reviewed forms | Begin Phase 3 cloud reset implementation | Numerical correctness |
| Issue a canonical production route | Blocked | Public registry has zero routes | Reset/value/gradient symbols and later gates do not exist | Keep route issuance inert | Admission/default/HMC eligibility |
| Treat consumers as migrated | Blocked | Consumers reject rather than understand v2 | Phase 6 integration remains | Preserve fail-closed rejection | Complete v2 rollout |
| Promote any scientific claim | Ineligible | No numerical/scientific run occurred | All Phases 3-9 remain | Execute smallest Phase 3 parity checks | Kalman or nonlinear validity |

## Inference-Status Table

| Inference | Status |
| --- | --- |
| Hard veto screen | Mechanical forgery, mutability, aliasing, raw-callable, stale-code, missing-closure, and admission paths pass focused rejection tests. |
| Statistically supported ranking | None; no stochastic comparison ran. |
| Descriptive-only differences | Digest values and validation runtime are explanatory only. |
| Default-readiness | Not established; public production route count is zero. |
| Next evidence needed | Cloud-level TensorFlow forward/JVP/VJP parity, then streaming composition, one-graph FD, trusted GPU feasibility, and oracle tests. |

## Phase 3 Blocker Handoff

Phase 3 inherits unresolved pre-result requirements for residual-design
centering error, mean restoration, executed-kernel ridged-identity backward
error, raw ridge bias, conditioning/downstream error, and ridge magnitude/domain
adequacy. It may establish small-fixture parity without using those outputs to
invent promotion thresholds. It may not mark the production reset evidence gate
passed until every required Phase 3 numerical/scientific requirement is
independently justified and satisfied.

## Artifacts

- Schema/factory manifest:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase2-schema-v2-factory-manifest-2026-07-13.json`.
- Source modules under `bayesfilter/highdim/ledh_*_v2.py` and
  `bayesfilter/highdim/ledh_contract_e_identity.py`.
- Focused tests:
  `tests/highdim/test_ledh_contract_e_schema_v2_factory.py`.
- Check log and run manifest under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase2/`.
- Review records under `docs/reviews/`.

## Post-Run Red Team

Strongest alternative explanation: identity binding could still omit an
unsupported dynamic behavior. The factory responds by rejecting unresolved
owned/external dependencies rather than issuing an incomplete identity. That
conservative behavior is sufficient for Phase 2, but future production symbols
must pass the same closure audit.

What would overturn this close: a reproducible normal-access registry override,
identity collision after a behavior-bearing change, a v1 silent upgrade, or a
consumer accepting an unadmitted candidate.

Weakest evidence: real production callable coverage, because the public registry
is deliberately empty. That is a later-phase blocker, not a Phase 2 pass claim.
