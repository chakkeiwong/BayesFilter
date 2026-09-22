# Zhao-Cui Repeated Compaction Failure

Investigation date: 2026-09-11, Hong Kong time (UTC+8).
Scope: recover the latest stalled audit, diagnose why continuation fails, and
prepare a bounded restart. This is not completion of the scientific audit.

## Measured Tool-Output Contribution

A subsequent read-only parse of the exact rollout confirmed 657,799 characters
across its 28 saved tool outputs: 331,385 during recovery and 326,414 during the
algorithm audit. Seven outputs were truncated. The four largest results were
42,229 to 48,116 characters each and came from recovery batches mentioning
saved sessions. One six-command batch requested 31,600 output tokens in total.
Per-command caps therefore did not provide a small aggregate output budget.

This supports the owner's tool-use concern. Large retained recovery history,
broad reads, and printing every result made remote compaction necessary. The
upstream stream failures are separately established; their server-side cause
is still unknown. Character totals are not token counts and cannot be directly
subtracted from runtime context counters.

The reproducible parser and sanitized measurements are in
[session-context-summary.json](artifacts/zhao-cui-audit-20260911-03/session-context-summary.json)
and the adjacent session_context_audit.py. They emit sizes, flags, and selected
counters, not raw saved conversations. The updated
[audit checkpoint](zhao-cui-audit-restart-2026-09-11.md) records the completed
bounded checks and exact next repairs. Its full evidence is in
[the continuation result](artifacts/zhao-cui-audit-20260911-03/stage-result.md).

## Diagnosis

The latest failed audit is thread
`01a08be9-85ba-7f21-8e53-a2dbf566ab7d`, not the earlier implementation thread
`01a08af9-04e6-7ab3-bedf-7f52c2821b4d`.

Its saved rollout is
`/home/chakwong/.codex/sessions/2026/09/10/rollout-2026-09-10T23-22-12-01a08be9-85ba-7f21-8e53-a2dbf566ab7d.jsonl`.
All 219 records parse. All 28 top-level tool calls have saved outputs. The last
tool result is line 209 at 00:53:46. No successful `compacted` record exists.
The last three tasks ended with no final assistant message. A task-completion
event or the UI's compaction announcement does not establish successful work
or a usable compacted history.

| Time | Evidence | Consequence |
| --- | --- | --- |
| Sep 10, 23:31 | Recovery investigation ends at 142,331 context tokens. | The same conversation already carries a large incident history. |
| Sep 11, 00:38 | Context reaches 148,657 tokens. | The later scientific audit starts with much of its context consumed. |
| 00:43; 00:48-00:49 | Ordinary sampling streams fail and retry below the compaction threshold. | The service problem also affects normal model responses. |
| 00:53:46 | Local post-tool counter reaches 249,888; compaction threshold 244,800; effective full limit 258,400. | Automatic compaction is required, although the full limit has not yet been reached. |
| 00:54:54; 00:56:02 | Remote compaction v2 retries report `stream disconnected before completion: Upstream request failed`. | The audit turn ends at 00:57:12 without a final report. |
| 01:10-01:13 | Continuing the same thread fails in `run_pre_sampling_compact`. | No new tools can execute before compaction succeeds. |
| 01:13-01:18 | Manual compaction uses `gpt-5.6-sol` instead of `gpt-6-astra` and fails again. | The recorded model switch did not recover the conversation. |

The rollout's last model-reported token count is 237,083; the runtime's
249,888 counter includes the subsequent tool output and is the counter that
triggered compaction. Cumulative billing counts such as 4,397,021 are not the
current context size.

The provider configured as `OpenAI` has base URL
`https://llm.visioncoder.ai`. Runtime logs show compaction v2 sending streaming
POST requests to `https://llm.visioncoder.ai/responses`. This incident is not
evidence that `/responses/compact` is missing: that was not the observed path.
HTTP 200 and `text/event-stream` headers precede failed streams; they do not
establish that a response completed.

The failure is in the gateway/upstream response path as observed by Codex.
Gateway implementation, upstream availability, request size, and a server-side
timeout remain possible explanations. Local logs do not distinguish them.
There is no evidence here that a Zhao-Cui numerical operation killed the agent.
The failed turn, rather than loss of source files, explains the apparent death.

## Recovered Audit State

The source, diagnostic script, and result JSON survive. All six source hashes
in `docs/plans/artifacts/zhao-cui-audit-20260911-01/diagnostic-result-01.json`
still match the research checkout. The result reproduces a rank-activation
defect and acceptance of a nonfinite likelihood with invalid weight sums.
It also records passing frozen-score, density, Gram, and guide fixtures.
Those passes do not establish a complete algorithm audit or scientific accuracy.

The audit Markdown was saved before its findings section was populated. The
JSON contains results that the next stage must incorporate. The new entry point
is [the compact audit checkpoint](zhao-cui-audit-restart-2026-09-11.md).
No numerical tests or scientific campaigns were launched by this investigation.
A trusted process-list check found no matching Zhao-Cui/C2 runner, audit
diagnostic, or pytest process at inspection time.

## Practical Prevention

1. Start a NEW conversation from the compact checkpoint. Continuing a thread
   already over the trigger requires the same failed compaction first. Forking
   a conversation is not a reliable way to discard its inherited context.
2. Keep recovery investigations and scientific execution in separate fresh
   conversations. This audit inherited about 149,000 tokens before its source
   inspection began. Work in bounded stages and update the checkpoint after
   each meaningful result.
3. Bound file reads and tool output. Use exact source ranges, structured JSON
   field extraction, and indexed SQLite queries. Print summaries of long logs.
   Do not recursively print saved conversations, especially their embedded
   prior tool outputs. Instructions also occupy context; preserve applicable
   policy rather than silently dropping it to make space.
4. Trial an earlier compaction trigger and smaller tool-output budget for a
   new session. The launcher uses `model_auto_compact_token_limit=120000` and
   `tool_output_token_limit=2000`. These are conservative operational choices,
   not measured service limits. The trigger still invokes remote compaction;
   the main protection is completing each stage in a fresh small conversation.
5. Repair or validate the provider route for a durable service fix. Start with
   a tiny scratch conversation and manual compaction, then a bounded larger
   fixture if the tiny case passes. Successful ordinary chat alone does not
   test compaction. A different provider needs its own credentials and owner
   authorization; no provider switch or credential transfer was attempted.

The prepared launcher is:

```sh
bash /home/chakwong/BayesFilter/scripts/start_zhao_cui_audit_fresh.sh
```

It opens a new terminal Codex session in the research checkout, also granting
that session access to the BayesFilter checkpoint directory. It retains the
configured model, provider, and approval policy. Its settings apply only to
that invocation. `--print-command` previews it without launching an agent or
making a model request. For VS Code, start a new conversation and ask it to read
the same checkpoint; this shell launcher's settings do not alter VS Code.

Do not raise the declared context window or disable compaction to force this
overfull history through. That does not create backend capacity. Increasing a
client idle timeout is not a demonstrated repair for an explicit upstream
failure, and more retries do not repair a reproducible failure.
The installed build exposes `remote_compaction_v2`, but disabling it was not
validated and is not presented as an offline/local compaction workaround.

## Verification And Limits

Installed CLI and failed session version: 0.153.4. Local feature output reports
`remote_compaction_v2=true`. Both integer configuration overrides load
successfully with `codex ... features list`; intentionally supplying a string
to either key produces the expected typed configuration error. This checks
that the installed configuration loader recognizes them, not that the gateway
can compact a 120,000-token conversation. `--strict-config` is unsupported for
`codex features`, so it was not used as validation evidence.

Official documentation search/open returned HTTP 502, and direct retrieval of
the public configuration reference returned HTTP 403 both inside and outside
the sandbox. The diagnosis and option checks therefore rely on saved local
runtime evidence and the installed CLI. No successful remote compaction was
obtained during this investigation. No live model probe was launched.

No existing research source, Codex configuration, rollout, or session database
was changed. The new checkpoint and launcher reduce recovery effort; they do
not establish that the gateway has been repaired.

## Exact Evidence Anchors

Read-only database: `/home/chakwong/.codex/logs_2.sqlite`, table `logs`.
Query by the exact thread ID using its index; avoid dumping the database.

- Context: IDs 456905801, 456915089, 456919752.
- Ordinary failures: 456916455, 456917562, 456918726.
- Automatic compaction failures: 456919916, 456920065.
- Pre-turn compaction failures: 456923266, 456923537, 456923607.
- Sol manual-compaction failures: 456924112, 456924274.
- First compaction response: ID 456919775, server request ID
  `f83ee82a-a124-4cbf-a3af-df51605e907e`, 00:53:52 HKT.
- Last manual-compaction response: ID 456924293, server request ID
  `bb4445d3-6008-4d03-bbef-27a60cfb76e0`, 01:16:33 HKT.

Those request IDs and times can identify the failures for the provider operator.
No support message has been sent and no raw logs or credentials have been shared.
