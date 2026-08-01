# Complete High-Dimensional Leaderboard Phase 2 Subplan

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `REVIEWED_PHASE2_AUTHORITY_PENDING`

## Phase Objective

Run the frozen trusted GPU/XLA/TF32 seed-`81120` LEDH prefix ladders for all
six main rows, ending at each exact full-time shape. Preserve a paired total
value and total score from the same fixed-randomness route, then validate every
score coordinate using separately executed value-only finite differences.

Phase 2 is a one-seed feasibility and validity screen. Prefixes are explanatory
only, and even a passing full-time seed does not admit a leaderboard cell.

## Entry Conditions

- Phase 1 result is
  `PASS_PHASE1_SIX_ROW_LED_H_HARNESS_NO_NUMERIC_CELL_ADMISSION` at
  `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-result-2026-07-11.md`,
  SHA-256
  `7bde673ab6b800f49a9d5dffd19aae61b2b86d6840fd6b011556182dc7fe250e`.
- Phase 1 machine manifest SHA-256 is
  `c26a897c7563092e59b417024e75b54c5ce2174681eff30f72110cc7a327bca0`.
- Canonical target artifact SHA-256 is
  `1cc83076b491b7c059fadbef85cacbb138c974a39502f5418d9018c17ef8fec8`.
- P1-A and P1-B receipt SHA-256 values are respectively
  `41fafd0eed4abb10a002d525ccb10e3544a2f52ce81f7b0e76c1d57e040edaef`
  and
  `af0547a53097cb5af6579c8ae993c1868dcd75a570dcfe3e08c2248c57dd1718`.
- Exact Phase 2/3 command manifest SHA-256 is
  `fa77f32fbf50333c0ae5e0e1a0c26e9772f9b568d9e0a017790e3d86c3d27433`;
  its command-set SHA-256 is
  `985795301e33ccbdb18068f01c70d0231f7797c36033b2d61f156ef30954790d`.
- The command builder `--check`, independent canonical-target checker,
  dedicated harness suite, and row/cross-model suites pass.
- A local material entry review checks this subplan, the Phase 1 result, and
  the exact-command manifest and records `VERDICT: AGREE`.
- A Phase 2 execution-authority receipt binds the Phase 1 result, this
  subplan, the entry review, and the exact-command manifest. It does not
  authorize Phase 3.
- Current wall time is before the reserved closeout boundary
  `2026-07-12T22:39:28+08:00`; the hard run deadline remains
  `2026-07-12T22:59:28+08:00`.

## Research Intent Ledger

| Field | Binding |
| --- | --- |
| Main question | Can every current six-row compact LEDH route execute at seed `81120` on trusted GPU/XLA/TF32, preserve paired total value/score semantics, remain within the per-seed score-memory gate, and pass every individual FD direction? |
| Candidate/mechanism | Current schema-v5 compact no-autodiff score route plus the same-route value-only central-FD comparator |
| Exact comparator | Stored plus/minus total log likelihoods at frozen float32 endpoints for the same target, seed, prepared randomness, source, configuration, and route |
| Expected failure mode | GPU/XLA compilation or placement failure, nonfinite output, memory-budget failure, artifact/provenance mismatch, or individual-direction FD failure |
| Promotion criterion | For a row, all frozen prefix pairs execute in order and its exact full-time seed-`81120` score and FD shards pass every hard validator |
| Promotion veto | Any invalid target/source/config/route identity, CPU/non-XLA/non-TF32 substitution, nonfinite output, score peak above `14000 MiB`, malformed or mispaired shard, or failed individual FD direction |
| Continuation veto | Failed trusted GPU preflight; corrupted shared harness/manifest/target; unsafe overlapping worktree drift; required new authority; or closeout/hard deadline |
| Repair trigger | A bounded implementation, XLA-compatibility, artifact-schema, configuration, or numerical defect whose evidence does not invalidate the frozen target or shared harness |
| Explanatory diagnostics | Prefix values/scores, compile/runtime/memory below the hard gate, and descriptive coordinate errors |
| Must not be concluded | Five-seed cell admission, complete leaderboard, ranking, superiority, HMC/posterior correctness, confidence coverage, source-faithfulness, default readiness, or broad scientific validity |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does each seed-`81120` compact LEDH candidate remain valid and feasible through its exact full-time shape under the frozen production execution policy? |
| Exact baseline | Phase 1 canonical target and current-source compact route; old July GPU artifacts are historical context only and cannot supply a Phase 2 shard |
| Primary criterion | Current exact command; trusted visible GPU; XLA JIT; float32/TF32; finite paired total value/score; reset score peak `<=14000 MiB`; exact target/source/config/route/randomness identities; every individual FD direction passes `relative_error <= 0.05 * sqrt(p)` |
| Hard veto diagnostics | Preflight failure, wrong device/provenance, stale command or code identity, missing terminal artifact, nonfinite scalar/vector, memory failure, endpoint/score mispair, or failed per-direction FD policy |
| Explanatory only | Prefix results, sub-budget memory, wall/compile time, aggregate-looking one-seed summaries, and historical comparator values |
| What will not be concluded | Any claim listed in the research intent ledger's nonclaims |
| Preserved artifact | Trusted preflight JSON; every immutable score/FD JSON, Markdown, and log; Phase 2 result; repair notes; and Phase 3 draft |

Finite difference is validation only. It is never the admitted score. No
aggregate or mean may hide a failed direction. The `5% * sqrt(p)` rule is the
owner-selected FD-only threshold and is not a computed 95% confidence
interval.

## Skeptical Pre-Execution Audit

| Challenge | Finding and control |
| --- | --- |
| Wrong baseline | Historical July score/value artifacts use stale routes or incomplete evidence; every Phase 2 shard must be newly emitted from the current manifest |
| Proxy promotion | Prefix and singleton-seed results cannot admit a cell; only a full-time pass marks the row eligible for Phase 3 |
| Missing stop condition | Trusted preflight, shared-identity drift, invalid artifact, row-level hard gates, closeout boundary, and hard deadline are explicit |
| Unfair comparison | Rows are screened independently; runtime and memory are not cross-ranked |
| Hidden assumption | CPU-hidden Phase 1 tests did not establish GPU feasibility; preflight and the first exact rung test it directly |
| Stale context | Runtime validation compares the command-manifest target/source/config/route/endpoint identities against current source before execution |
| Environment mismatch | All GPU commands freeze `tf-gpu`, `CUDA_VISIBLE_DEVICES=0`, `/GPU:0`, XLA, float32, and TF32; CPU fallback is forbidden |
| Artifact insufficiency | Each command has unique JSON/Markdown/log paths, an external timeout, terminal artifact semantics, and exact command identity |
| Misleading pass | A prefix can pass while full time fails; only exact full-time evidence can mark a row Phase-3 eligible |
| Scientific overreach | This phase screens one fixed seed and cannot support ranking, posterior, HMC, coverage, or source-faithfulness claims |

Audit decision: `PASS_AFTER_PHASE1_AND_COMMAND_FREEZE_REPAIRS`. The frozen
commands now answer the stated Phase 2 question and remain fail-closed behind a
phase-specific authority receipt.

## Required Artifacts

- Trusted preflight:
  `docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh/gpu-preflight.json`.
- Exact Phase 2 score/FD JSON, Markdown, and logs under the paths in
  `docs/plans/complete-highdim-leaderboard-ledh-phase2-phase3-exact-commands-2026-07-11.json`.
- Per-row repair/result notes when a rung fails.
- Phase result:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-result-2026-07-11.md`.
- Refreshed Phase 3 subplan and a local close/handoff review.

Every serious artifact must record git commit/status, command argv,
environment, TensorFlow version, GPU visibility, XLA, TF32, dtype, target and
source hashes, seed, wall time, output paths, plan, and result path. `N/A` is
used only when a field genuinely does not apply.

## Execution Sequence

1. Recheck hashes, exact-command `--check`, current time, and absence of
   computation-relevant drift.
2. Run escalated/trusted `scripts/run_complete_highdim_leaderboard_gpu_preflight.py`
   at SHA-256
   `106f5c6fd382c374a25211ca09c693669934a6c9c4b10594fd3642d92eeb4629`
   with `CUDA_VISIBLE_DEVICES=0`, `MPLCONFIGDIR=/tmp`, and the `tf-gpu`
   interpreter using exactly:

   ```bash
   timeout --signal=TERM --kill-after=30 300 env CONDA_DEFAULT_ENV=tf-gpu CONDA_PREFIX=/home/chakwong/anaconda3/envs/tf-gpu CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp PYTHONNOUSERSITE=1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python scripts/run_complete_highdim_leaderboard_gpu_preflight.py --output docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh/gpu-preflight.json
   ```

   The preflight artifact is immutable and the script refuses overwrite. Stop
   all GPU work without retry if the command fails, times out, or does not
   record `preflight_pass=true`; write the Phase 2 blocker/result instead.
3. Execute exact Phase 2 manifest entries in this row order to maximize
   resumable evidence within the common deadline: `fixed-sir`,
   `predator-prey`, `lgssm`, `actual-sv`, `generalized-sv`, `ksc-sv`.
4. Within a row, execute rungs in the manifest's increasing-time order. Run
   `score-only` first; validate its terminal artifact before running paired
   `fd-only`.
5. After each FD artifact, run the raw-shard validators. Classify the outcome
   before deciding whether to continue.
6. A row-specific candidate failure stops that row's larger rungs but does not
   stop unrelated rows. A shared harness/target/manifest invalidity is a
   continuation veto for all rows.
7. For a bounded fixable defect, preserve the failed artifact, patch visibly,
   rerun focused CPU-hidden checks, regenerate/review the command manifest if
   computation identity changed, rebind authority, and retry only the affected
   smallest rung.
8. Before every command, compute the current epoch and require
   `now_epoch + command_timeout_seconds + 60 <= 1783867168`, where
   `1783867168` is the closeout boundary
   `2026-07-12T22:39:28+08:00`. The 60-second reserve covers timeout kill grace,
   terminal-artifact inspection, and state recording. If the inequality fails,
   do not start the command; classify the remaining row/rungs as
   `deadline_incomplete`. Write the Phase 2 result and resumable handoff before
   the hard deadline `2026-07-12T22:59:28+08:00`.

The exact runtime command is each selected manifest entry's `shell_command`.
Do not reconstruct, reorder, or hand-edit argv. Create only the manifest's
`required_directories` before execution.

## Required Checks And Reviews

- Escalated `nvidia-smi` and TensorFlow GPU/XLA/TF32 preflight.
- P1-D builder `--check` immediately before the first GPU command.
- Exact command-authority validation in the harness.
- Terminal JSON and log existence after every command.
- Raw score validator after every score shard.
- Raw FD validator after every FD shard.
- Current-source target/config/route/randomness/endpoint identity checks.
- Focused CPU-hidden regression after any code repair.
- `git diff --check` before phase close.
- Local material result/Phase 3 handoff review. The local-only runbook forbids
  Claude, child Codex, network, and external API processes.

## Forbidden Claims And Actions

- Do not admit a leaderboard cell from one seed.
- Do not run any Phase 3 seed or aggregate command under Phase 2 authority.
- Do not change targets, parameter order, seeds, particle count, transport,
  FD step/tolerance, XLA, TF32, or dtype after seeing results.
- Do not substitute CPU, eager, non-XLA, non-TF32, historical, or differently
  configured evidence.
- Do not use prefix evidence as full-time evidence.
- Do not rank rows or methods using runtime, memory, values, scores, or FD
  errors from this phase.
- Do not claim source-faithfulness; all six Zhao-Cui adapters remain
  `extension_or_invention`.
- Do not overwrite failed or historical artifacts.
- Do not launch child model processes or make network/API calls.

## Exact Next-Phase Handoff Conditions

Draft Phase 3 only after Phase 2 has a terminal status for every row:

- `eligible_for_phase3`: exact full-time seed-`81120` score and FD shards pass;
- `row_candidate_blocked`: a row-specific hard veto prevents further seeds;
- `shared_invalidity_blocked`: shared evidence is invalid and no row may
  advance; or
- `deadline_incomplete`: valid immutable checkpoints exist but the common
  deadline prevented completion.

Phase 3 authority may cover only rows marked `eligible_for_phase3`. It must
bind the Phase 2 result, refreshed Phase 3 subplan, its local review, and the
unchanged exact-command manifest. A row blocked in Phase 2 cannot be silently
included.

## Stop Conditions

- Trusted preflight fails or reports no physical/logical GPU, CPU placement,
  XLA failure, TF32 disabled, or nonfinite output.
- Any bound canonical target, P1 receipt, source, harness, builder, manifest,
  or Phase 1 result hash drifts before execution.
- An exact command is not recognized or its Phase 2 authority is invalid.
- Shared harness, target, data, math, or artifact evidence is invalid.
- A required repair changes target/default/public API/scientific claim or
  otherwise needs new owner authority.
- Overlapping dirty-work changes make the computation identity unsafe.
- Current time reaches the closeout boundary or hard deadline.

Row-specific numerical, memory, compilation, or FD failure stops that row's
ladder but is not automatically a research-direction or unrelated-row stop.
