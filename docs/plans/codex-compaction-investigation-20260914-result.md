# Why the observation-aware TT session stalled again

The immediate failure is remote compaction through the configured gateway,
preceded by an unexplained jump in reported input tokens. Broad tool reads and
duplicated instructions increase context pressure, but they do not explain the
last jump or establish why this particular compaction request failed. The
neighboring threads provide evidence against output volume being a sufficient
explanation.

This investigation matched the attachment to session
`01a091d6-769a-7901-90e0-80900daba20e`, parsed its saved rollout locally, and
checked indexed runtime logs and two neighboring threads. Research programs,
Codex configuration, saved sessions, databases, and running agents were not
modified. No model probes or GPU runs were launched.

## The immediate trigger

Times below are China Standard Time, 2026-09-14; source timestamps are UTC.

| Time | Recorded event | Meaning |
| --- | --- | --- |
| 04:20:07 | Successful compaction; next actual request has 43,181 input tokens | Compaction had reduced the working context. |
| 04:38:52 | Request reports 58,972 input tokens; runtime compaction counter is 59,072 | Below the configured 120,000-token threshold. |
| 04:39:03 | Next request reports 130,862 input tokens; runtime counter is 131,303 | A 71,890-input-token jump crosses the threshold. |
| 04:40:08 and 04:41:13 | Automatic compaction retries report `Upstream request failed` | The remote compaction stream is failing. |
| 04:42:18 | Turn ends with remote compaction error | Research supervision stops. |
| 04:47:18–04:51:25 | Explicit compact turn fails again, with zero tool calls | The retry never reaches work the agent could perform to reduce output. |

The tool between the last two ordinary requests was a 30-second `clock.sleep`;
its saved result occupies 46 serialized characters. The next request produced
a short progress update and a bounded log-tail call. There was no intervening
large tool result that accounts for a 71,890-token increase. The provider's
reported cached-input count also changed from 57,088 to zero. Cache status
alone does not prove replay, corruption, or bad accounting.

The client additionally logged `OutputTextDelta without active item` at
04:38:56, just before the anomalous response. An earlier ordinary stream had
ended before `response.completed` at 04:11:55. These are concrete stream
anomalies, not evidence that a long-running local command hung.

Both automatic and manual compaction exhausted their retries. The six associated
HTTP responses opened with status 200 on `cn.origincoder.com/v1/responses`,
then the operation failed while streaming. HTTP 200 therefore does not establish
successful compaction. The failure is not recorded as `context_length_exceeded`;
the client reports an effective context window of 258,400 tokens, distinct from
the 120,000-token automatic-compaction threshold.

The logs establish the trigger and failed operation. They do **not** establish
whether the unexplained increase reflects gateway accounting, request/state
handling, routing, or a client/provider compatibility defect. Determining that
requires inspection of the corresponding requests by the gateway operator or
a controlled reproduction. No conversation content was sent to an operator.

## What the agent's tool use contributed

The saved session contains 626 tool results with approximately 2.82 million
decoded text characters across its entire history. Of these, 130 exceed 8,000
characters and 30 exceed 20,000. These are character counts over time, not the
number of tokens simultaneously in context.

There are specific avoidable reads. At rollout line 1506, four commands each
received a 16,000-token output allowance, including hundreds of lines from
multiple source files. At lines 1532, 1545, and 1557, large overlapping portions
of `squared_tt.py`, `tt.py`, `bases.py`, and `fitting.py` were requested again.
Those cells lacked an explicit outer output budget. Tool allowances add across
the batch even when each command appears bounded. Several returned results
reached approximately 40,000 characters and were truncated.

This continued, less extensively, during the final research turn: 161 tool
results, 413,697 decoded characters, three successful compactions, and two
results larger than 20,000 characters. At line 4622 the agent requested nearly
the whole 470-line campaign driver with a 20,000-token allowance, immediately
after reading overlapping ranges. Filename discovery, exact function ranges,
and selected result fields would have answered the questions with less context.

However, the final window after the last successful compaction had only 42 tool
results totaling 42,292 decoded characters; none exceeded 20,000. Its normal
requests grew from 43,181 to 58,972 input tokens before the anomalous jump.
Attributing the immediate failure to a final giant tool dump would be wrong.

## Fixed overhead and repeated recovery

The last successful compacted replacement preserves an 84,164-character user
instruction message containing two copies of the global scientific-policy
heading. It also preserves about 16,001 characters of developer messages.
The global instructions file and repository instructions each contain the
shared global policy. There are no duplicate *whole messages* in that
replacement; repeated policy material within the instruction message is the
relevant distinction. This overhead survives successful compaction and leaves
the next real request at roughly 43,000 input tokens before much new work.

The last serialized permission update contains 583 approved command prefixes
occupying 156,777 characters. This is not a measured prompt-token count, and
the comparison threads have similarly large permission state. It is not a
demonstrated cause of this thread's anomaly.

Twelve successful compacted events are recorded in this session. The first
terminal error, on September 11, was a remote-compaction HTTP 504 through the
previous `llm.visioncoder.ai` endpoint. Work subsequently continued for many
turns. The gateway hostname changed; this did not eliminate later failures.
The current incident therefore combines repeated exposure to remote compaction
with a new, specifically observed token-count anomaly.

Continuing the same failing thread or issuing another compact request preserves
the state that is failing. Once automatic compaction blocks a turn, instructions
to “use less context” cannot be executed inside that turn. Existing policy
already calls for fresh research conversations at completed stage boundaries;
the long combined proof, implementation, audit, diagnostic, and campaign turn
did not provide that separation.

## Why other threads can work on this machine

The comparisons below are snapshots, not controlled experiments. Both neighbors
were still producing work when inspected. They use the same stored client
version, model, and recent gateway host: `0.154.0-alpha.6.2`, `gpt-6-astra`,
and `cn.origincoder.com`.

| Thread | Tool results | Results >20,000 characters | Successful compactions | Same-window input jumps >30,000 tokens |
| --- | ---: | ---: | ---: | ---: |
| Stalled TT, `01a091d6…` | 626 | 30 | 12 | 1: +71,890 |
| Active BayesFilter neighbor, `01a091eb…` | 624 | 71 | 12 | 0 |
| Active MacroFinance neighbor, `01a0956a…` | 990 | 49 | 14 | 0 |

The MacroFinance snapshot has no terminal-error turns. The active BayesFilter
neighbor has two historical terminal errors, so it is not an error-free control.
Both contain more large tool results than the stalled thread. This supports a
session/request-specific or intermittent failure rather than a rule that large
tool output always causes compaction failure. It does not identify the hidden
provider-side cause or rule out a shared intermittent service defect.

The standalone shell CLI is version `0.153.4`, whereas the affected session is
from the VS Code client build above. Changing or testing only the shell CLI
would not establish that the affected VS Code behavior was repaired.

## Practical remedies

1. Recover research in a fresh conversation from the current recovery note
   below, preserving the failed thread for diagnosis. Avoid importing its raw
   transcript or forking its oversized history as the recovery mechanism.
2. Enforce the existing output discipline in actual calls: around 2,000 tokens
   per read, around 4,000 across an entire batch, filenames before contents,
   exact source ranges, and selected JSON fields. After truncation, narrow the
   request instead of requesting the same large files again.
3. Update a short checkpoint after each substantive stage and before a campaign
   launch. Keep compaction investigation separate from scientific execution.
   A checkpoint must record completed launches and failures, not only intentions.
4. In a separately scoped policy cleanup, remove repeated copies of the shared
   instructions from the injected material while preserving every applicable
   rule. Do not remove scientific safeguards or inflate the model's declared
   context window. Instruction cleanup reduces overhead; it cannot certify
   gateway compaction reliability.
5. Use the sanitized request IDs and timing evidence for a gateway investigation
   of the usage jump and compact streams. Lowering the threshold is only an
   unvalidated mitigation: a jump from about 59,000 to 131,000 could leap over
   another threshold, and more frequent compaction creates more remote calls.

These changes address distinct causes. Better tool discipline reduces normal
context growth. A fresh thread avoids the currently failing saved state.
Resolving the upstream stream/usage anomaly requires evidence beyond this local
inspection. No global setting was changed or claimed to fix the provider.

## Research state recovered from disk

The campaign is no longer running. `campaign-01/run_manifest.json` records
`FAILED` after 151.322 seconds, and `failure.json` records
`ValueError: pair conditional failed finite/bracket/CDF check`. The traceback
ends at `observation_guided_tt_tf.py:217`. The partial `result.json` still says
`RUNNING`; the terminal manifest and failure record take precedence.

The representation retry, `diagnostic-02`, completed in 73.700 seconds; the
first diagnostic failed after 6.188 seconds. These facts recover execution
state, not scientific success. The old active checkpoint predates those runs.
Use [the current recovery note](observation-aware-tt-recovery-checkpoint-20260914.md)
before resuming work; the conditional validity failure must be assessed against
the existing plan's continuation conditions before another campaign.

## Reproducible evidence and limits

Evidence is in [the incident directory](artifacts/codex-compaction-20260914-01/):
`analyze_session.py`, three session-metrics JSON files, `failure-logs.json`,
`runtime-context-counters.json`, `tool-use-examples.json`, and
`peer-endpoints.json`. Metrics contain source byte counts, timestamps, and
SHA-256 hashes of the bytes parsed. Live neighbor snapshots may subsequently
grow. Counts exclude duplicate completion events and do not count archived
guardian histories as newly delivered tool output.

The decisive saved rollout entries are lines 5197–5208 (sleep, response usage,
and final log read), 5210 (automatic failure), and 5211–5212 (manual failure).
Runtime log IDs 457709387 and 457709500 preserve the actual compaction counters.
The six failed-operation HTTP request IDs are retained in the sanitized logs.

Official documentation search/open returned 502; direct official-page retrieval
returned 403 even outside the sandbox. Product behavior described here is based
on these local runtime records. The roughly 37 MB saved file and cumulative
usage totals are not evidence of a 37 MB active prompt. No provider-side request
body or internal gateway logs were available to prove the deeper cause.
