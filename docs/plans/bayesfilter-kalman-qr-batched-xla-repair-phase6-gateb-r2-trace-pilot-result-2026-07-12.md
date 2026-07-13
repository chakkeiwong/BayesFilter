# Phase 6 Gate B R2 Trace/Pilot Result

Date: 2026-07-12

Status: `GATE_B_R2_HARNESS_INVALID_REPAIR_REQUIRED`

Supervisor/executor: Codex in the current conversation.

Parent repair subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-subplan-2026-07-12.md`.

## Result

The fresh `r2` Gate B command was authorized and launched once, then failed in
less than one second before any target fixture, trace child, TensorFlow concrete
function, XLA compile, Kalman execution, trace ledger, or pilot ledger. The
failure was a common harness-schema defect:

```text
scripts.kalman_qr_benchmark_contract.ContractError:
Phase 6 bindings do not match closed schema
```

The exact failing call was
`contract.new_phase6_ledger(...)` for the trace census. Diagnosis reproduced the
real proposal bindings without target execution and localized the false
predicate:

- all four proposal authority inputs are regular, present, byte-counted,
  base64-preserved, SHA-256-valid blob records;
- the immutable `r1` archive input is strict JSON;
- the repair result, agreeing plan review, and agreeing result review are
  Markdown and therefore truthfully have `strict_json=null`;
- `_phase6_bindings_valid` required `strict_json is not None` for every
  authority input even though proposal construction and validation deliberately
  accept mixed-format path/digest records;
- exact path/SHA-256 projection of the four blob records equals the proposal's
  four inputs.

The same defect escaped local tests because real proposal construction tests
used text path/digest inputs but did not construct runtime bindings from them,
while binding tests used JSON evidence or an empty authority-input list.

The launch also exposed a second harness issue: the budget state is opened
before trace bindings are constructed and validated. Consequently this
pre-ledger exception left a truthful `running` budget record even though the
lease was released and no trace ledger exists. The record is evidence of the
failed authority and must not be closed, deleted, resumed, or reused.

This result invalidates the `r2` harness authority only. It is not a structural
trace rejection, CPU-XLA failure, memory result, performance result, candidate
rejection, or evidence against Kalman mathematics.

## Evidence Contract Assessment

| Field | Observed result |
| --- | --- |
| Engineering question | Not answered: execution stopped before construction of the initial trace ledger. |
| Exact baseline | Frozen `r2` proposal, proposal review, attestation, skeptical audit, import discovery, budget state, and released lease. |
| Primary criterion | Failed by common harness invalidity before target execution. |
| Hard veto | Fired: real proposal bindings could not satisfy the runtime binding schema despite passing proposal and attestation validation. |
| Repair trigger | Fired: permit arbitrary byte-valid proposal inputs while preserving strict JSON requirements only for semantically parsed proposal, attestation, Phase 4/5 evidence, and runtime predecessors; validate bindings before budget open. |
| Explanatory only | Sub-second wall time, traceback depth, file sizes, and monotonic timestamps. |
| Not concluded | No target trace structure, CPU-XLA viability, memory repair, performance, scalability, GPU, HMC/posterior, default/production, or scientific claim. |

## Preserved R2 Evidence

| Artifact | SHA-256 / state |
| --- | --- |
| Proposal | `187594f66a2a87e237d697d52085318731efea986e077b2972e7a1cf44b46359` |
| Proposal review | `c4c7055eb5c416310867c831e93fb3cb111d1da76c6d2b3e6cf25b409d940acf`; `claude_opus_max`; `VERDICT: AGREE` |
| Detached attestation | `4fa7b0cbef59c826804dc9e156fffe9660aabdb05d26b0d88458893b89d566cd` |
| Final skeptical runtime audit | `b95d56f31c5a5f47eba601df6695070c98e12ebac70e0ebc0f43f207709b566b`; `AUDIT: PASS` |
| Import discovery | `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac` |
| Budget state | `a4cc284b64d6527a7357171f4c47395a7f29f7fed7e50b15563257feae09390f`; `state=running`, command `state=running`, `update_index=0` |
| Released lease | `ae711efe84056ae416d5fe2d2d40751b91afaa7f3a2e3530f095fb501a03b456`; generation 1, `state=released` |
| Authority ID | `4807a429ce935c95392f6af62266ef53a2e6165c8b8cc5e0cb415ba80fb26096` |

The `r2` work root contains only the import discovery and the budget-state
directory with its state and lease. The repository trace and pilot outputs are
absent. No matching supervisor, benchmark child, or process group survives.

## Root Cause

The invalid predicate is at
`scripts/kalman_qr_benchmark_contract.py:1680`: authority inputs must be valid,
present blob records, but line 1684 additionally requires every blob to contain
strict JSON. That contradicts the proposal input contract, which binds exact
paths and SHA-256 digests and includes three reviewed Markdown artifacts.

The budget ordering defect is at
`docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py:3289`:
`phase6_budget_state_open` precedes `_phase6_bindings_for_gate` and
`new_phase6_ledger`. A deterministic prelaunch schema failure therefore consumes
authority state without producing a ledger that can classify the failure.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject `r2` as harness-invalid and preserve it | Failed before target execution | Common-harness veto fired | Whether any further pre-ledger mismatch remains after repairing mixed-format inputs and validation order | Review and execute the dedicated `r3` mixed-format-binding repair subplan; construct a new disjoint authority only after local closure | No CPU-XLA, graph, numerical, memory, runtime, or scientific conclusion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Failed for harness validity only |
| Viable candidates | Not evaluated; no target method launched |
| Statistically supported ranking | None |
| Descriptive-only differences | Sub-second failure duration and artifact sizes |
| Default readiness | Not evaluated |
| Next evidence needed | Fresh reviewed `r3` harness repair and authority, then the still-unanswered Gate B trace/pilot |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Failed at mixed-format authority-input binding and pre-ledger budget ordering |
| Numerical/runtime validity | Not evaluated |
| Scientific interpretation | Not evaluated |

## Post-Run Red Team

- Strongest alternative explanation: another binding predicate might also be
  false after the Markdown predicate is repaired. The next focused test must
  build bindings from the exact real-format four-input mix and enumerate every
  predicate before allowing a fresh proposal.
- Result that would overturn the root cause: a byte-identical real binding set
  that passes with only the Markdown strict-JSON requirement unchanged. The
  diagnostic instead showed all three Markdown inputs fail exactly that
  requirement while every blob and path/digest projection is valid.
- Weakest evidence: the supervisor emits only the aggregate binding error. The
  next repair should add focused predicate coverage rather than broadening
  runtime logging or weakening fail-closed validation.

## Handoff

Exact state:
`GATE_B_R2_HARNESS_INVALID_REPAIR_REQUIRED`.

Do not relaunch, resume, close, delete, rename, or overwrite `r2`. Proceed only
through the dedicated reviewed `r3` repair subplan. The research direction
continues because this result invalidated the harness, not the target,
implementation mathematics, CPU-XLA backend, or either candidate method.
