# P0 Skeptical Pre-Execution Audit

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `PASS_FOR_CPU_METADATA_EXECUTION`

## Audit Findings

| Challenge | Finding |
| --- | --- |
| Wrong baseline | P0 compares planned route identities to current code, checked JMLR technical sections, and canonical author source. Historical readiness labels are not baselines. |
| Proxy promotion | Existing likelihood tests, scores, smokes, HMC diagnostics, and benchmark settings are inventory evidence only. They cannot issue a posterior signature. |
| Hidden assumptions | Missing prior, data hash, chart/Jacobian, filter settings, enhanced family, and comparator margins are explicit blockers or unresolved assumptions. |
| Stale context | The LGSSM result supplies controller mechanics only. SIR complete-data/scout work and generic retained-grid TT work remain inadmissible for full posterior claims. |
| Target conflation | Exact transformed SV and KSC are separate; every filter cell has a separate scope identity; scope identities are explicitly ineligible as posterior signatures. |
| Source mismatch | Current scalar/generic fixed-grid Zhao-Cui wrappers are classified as extensions/inventions unless an operation has both paper and author-source anchors. Structural application is invention by definition. |
| Stop and continuation | All eleven cells may be `TARGET_BLOCKED` while P1 proceeds with generic harness work. No cell may start HMC/training until its posterior contract is later admitted. |
| Environment mismatch | Builder and validator import only the Python standard library, use no TensorFlow/GPU, and make no numerical/scientific claim. |
| Artifact sufficiency | Builder emits cell, target, assumption, command, budget, event, run-manifest, source-hash, and artifact-hash evidence; validator fails on missing cells, signatures, classifications, future invented commands, or budget drift. |
| Dirty worktree | New uniquely named files/output roots avoid the extensive concurrent-lane changes. No existing file is modified by the P0 build. |

## Pre-Mortem

P0 could pass misleadingly if a likelihood-only route were hashed as a posterior,
if a generic fixed-grid wrapper inherited a Zhao-Cui fidelity label, or if missing
priors/data/charts were silently filled from unrelated fixtures. The builder
instead requires all incomplete cells to have `target_signature: null`, emits
only an explicitly non-admissible scope identity, and records the missing fields
as blockers.

The command could fail mechanically through a missing source, malformed JSON,
duplicate cells, or output overwrite. Those are fail-closed infrastructure
failures and may be repaired under P0 without changing its scientific contract.

## Allowed Execution

Run only the P0 standard-library builder, validator, JSON/Markdown/path checks,
and bounded read-only review. Do not import TensorFlow, probe GPU, train NeuTra,
or run HMC in P0.
