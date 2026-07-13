# Gate B R2 Final Skeptical Runtime Audit

Date: 2026-07-12

Status: `PASS`

Supervisor/executor: Codex in the current conversation.

Reviewer boundary: Claude Opus reviewed the proposal read-only at max effort
and returned `AGREE` after prompt clarification. Claude is not execution
authority. This audit, the user's execution request, and the reviewed subplan
govern the bounded runtime crossing.

## Runtime Question And Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the repaired supervisor produce a harness-valid target-grid structural trace and, only if that trace gate passes, a bounded two-row method-isolated CPU-XLA pilot? |
| Exact baseline | Immutable `r1` invalid-harness lineage archived at SHA-256 `caacd7144a0e6b7767487d7cc3a48145702983487ac1ab6885f5f97ba2f9607a`; it is not method evidence. |
| Primary criterion | Final trace and pilot ledgers satisfy their strict schema, proposal/attestation bindings, lifecycle, process-cleanup, and common-validity checks. A valid negative structural trace prunes the pilot and blocks Gate C without becoming harness invalidity. |
| Promotion vetoes | Authority/hash mismatch, malformed or non-final ledger, common invalidity, interrupted record, missing diagnostic, surviving process, stale namespace, non-finite target output, parity failure, compile crash, or timeout under the declared caps. |
| Continuation vetoes | Corrupt or ambiguous evidence, unsafe process ownership, inability to validate authority, missing required artifact, or a new human/package/network/model/default/scientific boundary. A method-local or structural candidate rejection is not by itself a research-direction veto. |
| Explanatory only | GraphDef nodes/bytes, trace duration, first-call time, warm-call time, memory telemetry, and individual method outcomes until later evidence supports interpretation. |
| Preserved artifacts | `r2` trace and pilot ledgers, budget state/lease, child artifacts/journals, runtime output, result note, and branch-specific next subplan. |
| Must not conclude | No speed or memory improvement, CPU/GPU scalability, method ranking, GPU readiness, HMC/posterior correctness, default/production readiness, or scientific validity from Gate B alone. |

## Authority Identity

| Artifact | SHA-256 / identity |
| --- | --- |
| Reviewed repair subplan | `575cf2fc7661bef2be6282ca57570d27bbf4490f2b01ecaad9e5cbb4c5efb004` |
| Repair result | `12518676d76c1497a070e3258463d66b7d81433918efcf7405d9816671dfc091` |
| Agreeing plan review | `39f41d15394e721a7a481d36abf65ac340dea656f6a96181f40e1577a0e1b9f5` |
| Agreeing repair-result review | `df3f732227dda254ea89f070da3f8f5058f14a08edc5f8613e9e43bf293d9ee0` |
| Fresh proposal | `187594f66a2a87e237d697d52085318731efea986e077b2972e7a1cf44b46359` |
| Proposal review | `c4c7055eb5c416310867c831e93fb3cb111d1da76c6d2b3e6cf25b409d940acf`; `claude_opus_max`; `VERDICT: AGREE` |
| Detached attestation | `4fa7b0cbef59c826804dc9e156fffe9660aabdb05d26b0d88458893b89d566cd` |
| Authority ID | `4807a429ce935c95392f6af62266ef53a2e6165c8b8cc5e0cb415ba80fb26096` |
| Import discovery | `8ae6086bd6b8bbebd7bf236536a80cb6b8befa993a9e686801c451e8fec4c8ac` |

Strict proposal validation and strict proposal-plus-attestation runtime-authority
validation both exited zero. All ten attestation checks were true. The exact
proposal command comparison, environment comparison, and budget comparison were
true. The proposal contains exactly four authority inputs, 36 trace schedule
rows, and two pilot schedule rows.

## Skeptical Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | Passed. The comparator is the byte-preserved `r1` invalid-harness lineage and the frozen repaired source, not a cleaned or reconstructed method result. |
| Proxy promoted to criterion | Passed. Unit tests, GraphDef size, trace duration, first-call time, and the internally successful `r1` child remain explanatory only. Only strict final ledgers can answer Gate B. |
| Missing stop conditions | Passed. Common invalidity, malformed evidence, authority drift, stale namespace, unsafe ownership, surviving process, outer timeout, and missing diagnostics are explicit stops. Structural trace rejection has its own valid-negative branch. |
| Unfair comparison | Passed. Target grid, method identities, order, dtype, device, JIT, child caps, cell cap, and outer budget are unchanged from the reviewed proposal. Gate B does not rank methods. |
| Hidden assumptions | Passed. The exact interpreter spelling and live supervisor argv are proposal-bound; raw `sys.orig_argv` behavior and negative mutations passed focused tests. Requested one-thread TensorFlow settings are recorded and are not called physical-core pinning. |
| Stale context | Passed. Frozen plan/result/review/archive hashes, implementation hashes, and protected Kalman source hashes were rechecked immediately before this audit. |
| Environment mismatch | Passed. This is a deliberate GPU-hidden CPU-XLA gate using `CUDA_VISIBLE_DEVICES=-1`, one requested intra/inter-op thread, `float32`, and `jit_compile=true`. It is reference/compiler evidence, not the repository production GPU target. |
| Artifact fitness | Passed. The trace ledger answers structural graph validity; the pilot ledger answers bounded two-row CPU-XLA viability; neither can establish full-grid memory or performance repair. |
| Pass while misleading | Controlled. A Gate B pass only permits a reviewed Gate C subplan; it cannot establish target-scale P150/B16 viability or speed. |
| Fail for implementation rather than mechanism | Controlled. Common invalidity returns to harness repair; valid structural/method-local failures are recorded separately and do not reject Kalman mathematics. |

## Prelaunch State

- no matching supervisor or benchmark worker is live;
- `/tmp/kalman_qr_phase6_cpu_xla_gateb_r2` is a real directory containing
  exactly its proposal-bound `import_discovery.json`;
- trace output, pilot output, budget state, lease, and child artifacts are
  absent;
- protected algorithm hashes remain:
  - `kalman_qr_tf.py`: `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b`;
  - `kalman_qr_derivatives_tf.py`: `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`;
  - `qr_factor_tf.py`: `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`;
- scoped `git diff --check` is clean;
- no package, network, credential, model-file, funding, product/default,
  release, or scientific-claim boundary is crossed.

## Decision

The reviewed evidence contract, authority, environment, process state, artifact
namespace, exact command, and stop conditions are internally consistent and
current. The exact Gate B runtime command may run once under its 3000-second
TERM deadline, 45-second outer KILL grace, 160-second paired-cell cap, and
3045-second total authority.

AUDIT: PASS
