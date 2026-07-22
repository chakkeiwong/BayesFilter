# Codex Substitute Re-Review: Phase 9 Gate A And GPU Manifest, Iteration 2

Date: 2026-07-10

## Scope And Limitation

Fresh local read-only re-review after the iteration-1 artifact-identity and
command-exactness blockers were repaired. Claude remains policy-blocked as
external repository disclosure. No GPU/CUDA command ran during this review.

This verdict authorizes only the trusted `nvidia-smi`/TensorFlow-XLA preflight
and the ten nonlinear Gate B commands. Gate C, Gate D, aggregation, and the
separate LGSSM lane remain blocked until the Gate B result receives another
bounded review.

## Blocker Resolution

| Iteration-1 blocker | Resolution checked |
| --- | --- |
| Gate C/D templates were not exact commands. | A deterministic generator emits a frozen JSON with 10 Gate B, 36 Gate C, 40 Gate D, and 5 aggregate commands. All 91 commands and output paths are unique, parser-valid, and governance-hashed. |
| Separate score/FD processes did not prove identical fixed inputs. | Every prepared tensor leaf is serialized and SHA-256 hashed. FD recomputes the tensor-tree fingerprint and must match the score reference before evaluation; aggregation rechecks it. Five-row determinism tests pass. |
| Dirty helper code was not content-addressed. | Each shard freezes recursive SHA-256 hashes for 50-53 reachable local Python sources plus governance artifacts, exact commands, source value artifact, and current HEAD. |
| Parser-equivalent unreviewed commands could run as trusted evidence. | Trusted runtime and aggregate paths require repository root, reviewed GPU/CPU device identity, frozen output/reference/shard paths, and literal argv equality with exactly one JSON command entry. |
| Long-run provenance could change mid-process. | Source, code, governance, and command hashes are cached/frozen for process lifetime; tests reject recomputation and stale/mismatched content. |
| Pipeline logging could hide Python failures. | Commands redirect stdout/stderr directly; shell exit status remains the runner's. |

## Gate A Evidence

- Syntax and generator-currentness checks pass.
- Focused harness: `76 passed, 2 warnings in 20.46s`.
- Final combined harness, five adapter parity cases, Phase 8 schedule, and
  shared score contract: `149 passed, 2 warnings in 20.27s`.
- Model-specific regression shards remain `39`, `45`, and `38` passing tests;
  no row adapter changed after those shards.
- All five tensor adapters match the existing tiny eager score/value routes at
  `atol=rtol=1e-10`.
- Source scan finds no `GradientTape`, `ForwardAccumulator`, non-JIT fallback,
  historical manual-total-VJP, or memory-style reverse route in the shared
  runner/generator.
- `git diff --check` passes.

## Gate B Safety Assessment

The trusted preflight has a durable JSON, explicit GPU 0 selection, XLA matmul,
TF32/device checks, and a stop on failure. Each Gate B score command is
singleton-seed, XLA-hard-coded, production-precision, atomic, reset-memory
instrumented, and terminal on ordinary exception or supervisor interrupt. A
matching FD command cannot start valid work from a wrong score shard: it checks
row/target/theta/transport/code/governance/source/output/hash identity and then
requires identical prepared tensor content.

Gate B remains a tiny compile/device/FD screen. Its sub-budget memory is not an
`N=10000` memory result and cannot admit a row. Runtime and continuous metric
differences remain descriptive only. No ranking is statistically supported.

## Residual Risks

- Actual GPU XLA compilation is not checked until Gate B runs.
- A tiny pass may fail at `N=10000` or longer `T`; the prefix ladder exists for
  that reason and remains separately gated.
- External package binaries and TensorFlow itself are identified by environment
  metadata/version rather than content-hashed; the managed environment is an
  execution prerequisite, not a scientific claim.
- Supervisor `SIGKILL` cannot be caught by Python. If a hard kill is required,
  the pre-existing nonterminal artifact is classified as
  `HARNESS_TERMINAL_ARTIFACT_FAILURE`; it cannot enter aggregation.
- LGSSM still lacks a hardened split-score/FD merge path and is not authorized
  by this verdict.

No material blocker remains for trusted preflight and nonlinear Gate B.

VERDICT: AGREE
