# Complete High-Dimensional Leaderboard Phase 3 Subplan

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `BLOCKED_NO_PHASE2_ELIGIBLE_ROWS_NOT_AUTHORIZED`

## Phase Objective

For rows that have already passed exact full-time seed-`81120` score and FD
gates in Phase 2, execute the frozen full-time score/FD pairs for seeds
`81121..81124`, validate every seed and individual FD direction, then build the
exact five-seed offline aggregate required for LEDH cell admission.

The current eligible-row set is empty. This subplan is a fail-closed handoff,
not execution authority.

## Entry Conditions Inherited From Phase 2

- Phase 2 result exists at
  `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-result-2026-07-11.md`.
- A row may enter only with terminal, current-source, trusted GPU/XLA/TF32
  exact full-time seed-`81120` score and FD shards that pass raw validators,
  memory, identity, pairing, and every individual-direction FD gate.
- Phase 2 currently marks all six rows `row_candidate_blocked`; therefore the
  eligible-row set is `{}`.
- The repair1 exact-command manifest is SHA-256
  `bc8a8a9aa67b64b72ff5e9431bf8ea993bc8a97acbb62e52af0cef421bf4229f`.
  It is a deterministic command freeze, not authority.
- Before execution, a future reviewed Phase 2 repair result must explicitly
  add at least one row to `eligible_for_phase3`, and a refreshed Phase 3
  subplan/review must bind the then-current target, source, command manifest,
  and eligible-row allowlist.
- A separate Phase 3 execution-authority receipt must bind the refreshed
  subplan, Phase 2 result, local review, exact manifest, eligible rows, and a
  new hard deadline. No such receipt currently exists or may be created from
  the present evidence.

## Research Intent Ledger

| Field | Binding |
| --- | --- |
| Main question | For each Phase 2-eligible row, do exactly five paired full-time seeds remain valid under the same frozen target and computation identity? |
| Candidate/mechanism | Current compact no-autodiff score route plus separately executed same-route value-only central FD comparator |
| Exact comparator | Per-seed stored plus/minus total log likelihoods at the frozen float32 endpoint contract |
| Promotion criterion | Exactly seeds `81120..81124` pass score, memory, provenance, pairing, and every individual FD direction; offline aggregate passes the admitted artifact validator |
| Promotion veto | Any failed seed/direction, target/source/config/route drift, malformed/nonterminal artifact, wrong device/XLA/TF32/dtype, nonfinite value, score-memory failure, or aggregate mismatch |
| Continuation veto | Empty eligible-row set; missing Phase 3 authority; shared invalidity; unsafe source drift; or hard deadline |
| Repair trigger | A bounded row-specific failure that preserves prior valid shards and the exact target while motivating a reviewed Phase 2 repair |
| Explanatory diagnostics | Runtime, compile time, memory below the gate, seed dispersion, values, scores, and FD margins |
| Must not be concluded | Ranking, superiority, HMC/posterior correctness, confidence coverage, source-faithfulness, default readiness, complete leaderboard, release, or broad scientific validity |

## Required Artifacts

- Refreshed Phase 3 subplan and local material review.
- Phase 3 execution-authority receipt with a nonempty exact row allowlist.
- Immutable trusted GPU/XLA/TF32 score and FD JSON, Markdown, and log files for
  seeds `81121..81124` at each admitted row's exact full-time shape.
- Existing Phase 2 seed-`81120` score and FD shards, hash-bound without copying
  or rewriting.
- One CPU-hidden offline five-seed aggregate JSON and Markdown per eligible
  row, using exactly the manifest paths and exactly seeds `81120..81124`.
- Phase 3 result, run manifest, decision/inference tables, post-run red team,
  and Phase 4 handoff.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does a Phase 2-eligible LEDH row preserve its exact full-time score/FD validity across exactly five frozen seeds? |
| Exact baseline | The row's validated Phase 2 seed-`81120` full-time pair under the unchanged canonical target and current-source route |
| Primary criterion | Five unique paired seeds, trusted GPU/XLA/TF32 score/FD shards, finite paired total values/scores, reset score peak `<=14000 MiB`, exact identities, and every individual FD direction passing `relative_error <= 0.05 * sqrt(p)` |
| Hard veto diagnostics | Any missing/extra/duplicate seed, individual direction failure, stale or mismatched identity, wrong execution provenance, nonterminal artifact, nonfinite output, memory failure, or invalid aggregate |
| Explanatory only | Descriptive seed means/dispersion, values, runtime, compile time, sub-budget memory, and FD margin after the hard screen |
| What will not be concluded | Any claim in the research intent ledger's nonclaims |
| Preserved artifact | Per-seed shards, paired hashes, exact endpoints, aggregate, commands/logs, result, review, and next-phase handoff |

FD remains validation only and is never the admitted score. A mean or aggregate
may not hide one failed seed or direction. Passing five hard screens does not
support a ranking without predeclared uncertainty analysis.

## Required Checks And Reviews

- Verify a nonempty Phase 2 eligible-row allowlist and exact full-time passing
  seed-`81120` pairs before any runtime command.
- Run deterministic manifest `--check` and current-source target/config/route/
  endpoint identity checks.
- Run trusted GPU/XLA/TF32 preflight under the new authority and deadline.
- After every score shard, require terminal JSON/log and raw score validation.
- After every FD shard, require terminal JSON/log and raw FD validation.
- Run aggregate validators CPU-hidden with `CUDA_VISIBLE_DEVICES=-1`.
- Run focused adversarial tests, Python compilation, and scoped
  `git diff --check` after any repair.
- Write the Phase 3 result and refresh/review Phase 4 before any handoff.
- Local-only execution forbids Claude, child Codex, network, and external API
  processes unless a later owner instruction and reviewed runbook explicitly
  replace that boundary.

## Forbidden Claims And Actions

- Do not execute while the eligible-row set is empty.
- Do not create Phase 3 authority from the current Phase 2 result.
- Do not include a row absent from the exact Phase 2 eligible allowlist.
- Do not rerun, overwrite, substitute, or silently repair failed Phase 2
  artifacts inside Phase 3.
- Do not change target, parameter order, seeds, particle count, transport, FD
  step/tolerance, XLA, TF32, dtype, or endpoint policy after observing results.
- Do not substitute CPU, eager, non-XLA, non-TF32, historical, prefix, smoke,
  or differently configured evidence.
- Do not admit a cell if any seed/direction or aggregate validator fails.
- Do not rank rows or methods using descriptive evidence.
- Do not claim source-faithfulness; all six Zhao-Cui adapters remain
  `extension_or_invention` unless a separately grounded plan changes that
  classification.

## Exact Next-Phase Handoff Conditions

Phase 4 may be drafted for execution only after Phase 3 writes a terminal
classification for every allowed row and one of these applies:

- `admitted_five_seed_ledh_cell`: exactly five paired full-time seeds and the
  aggregate pass every gate;
- `row_candidate_blocked`: a row-specific seed, FD, memory, runtime, or
  artifact veto fires;
- `shared_invalidity_blocked`: shared target/harness/manifest evidence is
  invalid; or
- `deadline_incomplete`: immutable checkpoints exist but the reviewed deadline
  prevents completion.

The Phase 3 result must list admitted rows and blocked rows separately. Phase 4
authority may depend only on explicitly admitted Phase 3 artifacts and cannot
convert a blocked LEDH row into an admitted cell.

## Stop Conditions

- Current condition: stop because the eligible-row set is empty and no Phase 3
  authority exists.
- Stop on any target, source, harness, command, endpoint, or authority drift.
- Stop the affected row on any failed individual seed/direction, nonterminal
  artifact, nonfinite output, memory failure, or exact timeout.
- Stop all rows on shared invalidity, unsafe overlapping changes, or deadline.
- Candidate rejection does not establish rejection of the filtering research
  direction; return bounded failures to a reviewed Phase 2 repair loop.
