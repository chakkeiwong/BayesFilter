# Complete High-Dimensional Leaderboard Phase 2 Entry Review

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Reviewer role: current-session Codex supervisor, local read-only material
review. The active local-only runbook forbids Claude, child Codex, network, and
external API processes.

## Exact Scope

- Phase 1 result SHA-256
  `7bde673ab6b800f49a9d5dffd19aae61b2b86d6840fd6b011556182dc7fe250e`.
- Phase 1 machine manifest SHA-256
  `c26a897c7563092e59b417024e75b54c5ce2174681eff30f72110cc7a327bca0`.
- Phase 2 subplan SHA-256
  `9f7c288d07b0769d4e3663bd86f506940b15ded524b3fc7bec0427fa56b756ba`.
- Phase 2/3 exact-command manifest SHA-256
  `fa77f32fbf50333c0ae5e0e1a0c26e9772f9b568d9e0a017790e3d86c3d27433`.
- Trusted preflight script SHA-256
  `106f5c6fd382c374a25211ca09c693669934a6c9c4b10594fd3642d92eeb4629`.

## Review Question

Is Phase 2 safe and sufficient to test current-source one-seed GPU/XLA/TF32
feasibility and individual FD validity without admitting cells, crossing Phase
3 authority, or overrunning the common deadline?

## Findings And Repairs

| Round | Finding | Resolution |
| --- | --- | --- |
| 1 | `REVISE`: a command could start shortly before closeout and run beyond the hard deadline; preflight command and retry/overwrite semantics were underspecified; the Phase 1 material review was not durable | Added a common-clock start inequality using declared timeout plus 60-second reserve, one exact immutable preflight command with no retry-after-failure, preflight run-manifest fields and overwrite refusal, and a durable Phase 1 review artifact |
| 2 | `AGREE`: no remaining material baseline, proxy, environment, artifact, authority, timeout, or row-isolation defect | Entry conditions and exact hashes are final; Phase 2-only authority remains separate from Phase 3 |

## Final Assessment

- Baseline is current Phase 1 target/source/harness identity. Historical July
  artifacts cannot supply a shard.
- Trusted GPU preflight is a continuation veto, not numerical promotion.
- Prefix results are explanatory only; full-time seed `81120` pass makes only
  that row eligible for Phase 3.
- The exact command gate compares current target, source, configuration,
  route, and endpoint contracts and requires the Phase 2 authority receipt.
- Every runtime command has unique immutable JSON/Markdown/log paths and an
  external timeout.
- No command may start unless its full timeout plus reserve fits before epoch
  `1783867168` (`2026-07-12T22:39:28+08:00`).
- Row-specific compile, memory, numerical, or FD failure stops that row but not
  unrelated rows. Shared harness/target/manifest invalidity stops all rows.
- No Phase 3 command or aggregate is authorized by this entry review.
- No cell, ranking, HMC/posterior, coverage, source-faithfulness, default, or
  release claim is authorized.

## Residual Risk

The frozen per-command timeouts are conservative maxima, so the common-clock
rule may stop before later rows even if a command would have finished quickly.
That produces a truthful `deadline_incomplete` checkpoint and is preferable to
overrunning the binding eight-hour cap. Actual GPU/XLA behavior remains the
question Phase 2 is designed to answer.

VERDICT: AGREE
