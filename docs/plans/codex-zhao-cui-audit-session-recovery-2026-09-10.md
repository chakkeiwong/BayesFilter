# Zhao-Cui Audit Session Recovery, 2026-09-10

Status: incident investigation complete; scientific audit unfinished.
Scope: recover the session in the owner's pasted conversation and establish
why it stopped. This note does not approve or execute the scientific program.

## Session Identity

The affected thread is `01a086f1-70ca-7663-8775-a17029b3e869`, titled
`The other codex agent with title "hi" got stuck. Can you lo...`.
It is the recovery conversation, not the original `hi` thread
`01a044d0-e6b9-7c63-b41b-abcaf78f065f`.

- Saved rollout: `/home/chakwong/.codex/sessions/2026/09/10/rollout-2026-09-10T00-12-45-01a086f1-70ca-7663-8775-a17029b3e869.jsonl`
- Session cwd: `/home/chakwong/BayesFilter`
- Research checkout: `/home/chakwong/BayesFilterZhaoCui`
- Research branch: `zhao-cui-tt-regression-20260908`
- Research HEAD: `47176bce7cdaa91ddd4466c39f90a11ad005c800`
- Recorded Codex version: `0.153.4`; model: `gpt-6-astra`; reasoning: `max`
- Saved-history mode: `legacy`. This thread has no rows in the paginated
  `thread_history_1.sqlite` tables; its rollout is the relevant history.

The matching messages are the master-program question at 13:23 and the full
code/LaTeX/analytical-derivative audit request at 13:38 on September 10.
All times in this note are Hong Kong time, UTC+8.

## Failure Diagnosis

The immediate blocker was failed remote context compaction. At 13:43:35,
Codex recorded `auto_compact_scope_tokens=247506` against
`auto_compact_scope_limit=Some(244800)`, with `token_limit_reached=true`.
Its full context window was 258400 and
`full_context_window_limit_reached=false`: the compaction threshold was
crossed, not the absolute model context limit.

Compaction ran with `reason=ContextLimit phase=MidTurn` and failed. Each later
continuation attempted `reason=ContextLimit phase=PreTurn` and failed before
the agent could read the new request and perform further work. This explains
why repeated `continue` or `hi` messages produced the same failure.

The provider is named `OpenAI` in local configuration, but its configured
`base_url` is `https://llm.visioncoder.ai`. The runtime logs confirm requests
to that gateway's `/responses` endpoint. The observed failures were:

- An ordinary sampling request at 13:39:40 returned HTTP 503, with
  `Service temporarily unavailable`.
- Most compaction retries ended with
  `stream disconnected before completion: Upstream request failed`.
- The 17:04 continuation reported
  `stream closed before response.completed`.
- The 17:58 continuation also reported
  `rate limit exceeded: Service temporarily busy. Please retry later.`

Fifteen compaction HTTP responses are recorded across the five failed turns,
three per turn. Each initially returned HTTP 200; the streamed response then
failed. Successful HTTP headers therefore did not establish successful
compaction. No successful `compacted` record exists in the saved rollout.

The evidence identifies a failure in the gateway/upstream response path.
It does not distinguish gateway capacity or compatibility problems from a
failure of its upstream service. It does not establish an account quota
problem, a GPU fault, or a particular server-side defect.

## Failure Timeline

| Request started | Last saved event | Failure stage | Error evidence |
| --- | --- | --- | --- |
| 13:38:25 | 13:47:09 | Mid-turn compaction after file reads | Upstream request failed |
| 13:50:20 | 13:54:04 | Pre-turn compaction | Upstream request failed |
| 17:04:40 | 17:06:14 | Pre-turn compaction | Stream closed before response.completed |
| 17:58:12 | 18:02:22 | Pre-turn compaction | Service temporarily busy, then upstream failure |
| 18:50:37 | 18:54:23 | Pre-turn compaction | Upstream request failed |

The rollout's empty `task_complete` events mark turn termination; they do not
mean the audit or compaction succeeded. Runtime errors establish failure.
The final code-read result was saved at 13:43:35, rollout line 263.

## Context Growth And Integrity

This second incident does not reproduce the missing tool-result defect found
in the original `hi` thread. All 281 JSONL records parse, and all 36 tool calls
have matching outputs, with no unmatched outputs. There is no evidence here
that repairing an orphaned tool result would resolve the stall.

The recovery conversation was already large before the audit. The last
master-program answer used 152916 input tokens. The audit then made eight
batches of file reads, whose stored output payloads total 314345 characters
when serialized as JSON. Several batches show explicit output-truncation
warnings. These are payload character counts, not token counts. The runtime
counter rose from 176390 after the first audit batch to 247506 after the last.

The large retained recovery history and broad reads explain why compaction
was needed so early in the audit. They do not prove why the remote service
failed to perform it. A fresh thread with a concise saved checkpoint avoids
this particular pre-turn compaction dependency; it cannot guarantee gateway
availability.

## Preserved Work And Continuation

Fresh SHA-256 checks of the following four files exactly match the values in
[the original recovery note](codex-hi-session-recovery-2026-09-10.md):

- `docs/plans/bayesfilter-zhao-cui-algorithm3-ukf-guided-tt-master-program-2026-09-09.md`
- `docs/plans/lean/zhao_cui_algorithm3_active_20260909.lean`
- `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`
- `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.pdf`

These paths are relative to the research checkout. The stalled audit performed
file reads and searches; it contains no implementation edits, test launches,
or completed audit report. The master still reports
`ACTIVE_PHASE0_REPAIR_AND_PHASE1_DIAGNOSTICS`. Its proposed Algorithm 3 output
directory, `docs/benchmarks/artifacts/zhao_cui_algorithm3_20260909/`, remains
absent. An escalated host process snapshot found no command matching the
Zhao-Cui checkout, Zhao-Cui/C2 runners, Lean, or LaTeX builds.

The interrupted objective is to audit the master program against the code,
LaTeX, paper, and author implementation before making a scientific verdict.
The last public progress message distinguished three derivative targets:
the recursively fitted TT likelihood, a frozen-proposal particle likelihood,
and the exact model likelihood. Their equivalence was not established.

Section 13 of the master records outstanding ancestor-selection,
adjacent-conditional-TT retention, sampler/density consistency, and reference
measure issues. These are recovered plan findings, not an independent
mathematical verdict from this incident investigation. The next research
work must check those findings and the exact analytical-score target against
the actual consumer call chains and source anchors. The retired APF Phase 8E
runner is not the active endpoint.

Continue from this concise note and the master program in a fresh research
conversation, preserving bounded file reads and saving findings between audit
sections. Reopening the entire stalled history has not been tested as a fix
and retains the observed compaction dependency.

## Evidence And Investigation Limits

The investigation matched the exact thread and timestamps, used read-only
SQLite queries and structured JSONL parsing, and compared saved-file hashes.
This avoids treating the old incident's hypothesis as the cause of the new
one or treating a terminated turn as a completed scientific audit.

Runtime evidence is in `/home/chakwong/.codex/logs_2.sqlite`, table `logs`:

| Log IDs | Evidence |
| --- | --- |
| 456807446 | Ordinary response HTTP 503 and actual gateway URL |
| 456808359 | 247506 tokens, 244800 compaction threshold, full limit not reached |
| 456808550, 456808715 | Mid-turn compaction retries |
| 456809737, 456809919, 456810123 | First failed pre-turn continuation |
| 456836375, 456836588, 456836617 | Incomplete-stream continuation |
| 456844304, 456844584, 456844778 | Service-busy and upstream failures |
| 456851988, 456852205, 456852360 | Last failed continuation |

Official-documentation search returned service errors, and direct retrieval
was denied in both sandboxed and escalated attempts. The incident findings
therefore rely on local session and runtime evidence, not an asserted
documented fix. No Codex configuration, session database, rollout, research
code, or scientific artifact was modified. This note is the only repository
file added by the investigation.
