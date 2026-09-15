# Saved-session diagnosis, 12 September 2026

The final literature turn stopped because remote automatic compaction failed.
Avoidable context growth brought the session to the compaction threshold, but
the available logs do not establish why the upstream compaction service failed.
The full model context window was not exhausted. The earlier 12:43 AM attempt
also suffered a separate connection failure before any model response or tool
execution was recorded.

## Exact session and evidence

The pasted conversation matches session
`01a08f5c-e140-7ac1-93af-3e8ab3ac4417`, stored at
`/home/chakwong/.codex/sessions/2026/09/11/rollout-2026-09-11T15-27-04-01a08f5c-e140-7ac1-93af-3e8ab3ac4417.jsonl`.
It contains 923 JSONL records and was 7,346,314 bytes when inspected. Its SHA-256
is recorded in [session-metrics.json](session-metrics.json). Metadata identifies
VS Code, the BayesFilter checkout, and a session initially created with CLI
0.153.4. The final-turn application logs reference expected client version
0.154.0. This version difference alone establishes no cause.

The source session was read without resuming it. Application logs were queried
read-only by exact thread ID, using the existing SQLite indexes. Saved evidence
contains numerical summaries, source line numbers, and selected log IDs rather
than raw prompts, reasoning, or entire session/database copies. The original
session and databases were not modified.

## Final turn: compaction trigger and failure

All displayed times below use UTC+8 on September 12, matching the conversation.

| Time | Observed event | Evidence |
| --- | --- | --- |
| 02:54:23 | User asks for a derivation and literature survey of observation-aware TT proposals and UKF guidance. | Session lines 910–912 |
| 02:54:32 | Assistant announces a literature investigation. | Line 917 |
| 02:54:34 | Its single tool reads the scholarly-literature skill; the tool returns in about 0.1 seconds. | Lines 918–921 |
| 02:54:34 | App measures 124,873 tokens against a 120,000 automatic-compaction threshold; `full_context_window_limit_reached=false`. | App log ID 457159714 |
| 02:55:39 | Remote compaction stream fails; retry 1 of 2. | App log ID 457160022 |
| 02:56:42 | Remote compaction stream fails; retry 2 of 2. | App log ID 457160554 |
| 02:57:47 | Turn ends with a compaction error and no final assistant answer. | Session line 923 |

The terminal error is:

> Error running remote compact task: stream disconnected before completion: Upstream request failed

The final turn has no literature search, paper retrieval, numerical experiment,
or scientific implementation call. Its one tool had already returned when the
stall occurred. The logs show remote compaction v2, a mid-turn `ContextLimit`
trigger, and two retries. No final-turn thread log contains `504`; the error
does not identify an HTTP status or the upstream root cause.

The 245,203-token context, 244,800 threshold, and eleven 504 errors in the
pasted earlier answer refer to a previous diagnosis. They are not the measured
facts of this final incident.

## The context-use problem is independently supported

The last successful compaction occurred at 18:21:39 on September 11 (session
line 676). The following tool volume accumulated before the final failure:

| Subsequent question | Completed tool calls | Saved output text, characters |
| --- | ---: | ---: |
| Mathematical corrections, September 11 at 18:24 | 9 | 210,147 |
| UKF integration, September 12 at 02:45 | 14 | 392,132 |
| Literature survey, September 12 at 02:54 | 1 | 3,686 |
| Total since successful compaction | 24 | 605,965 |

These are character counts of the saved tool text, with outer content blocks
unwrapped. Nested JSON escaping remains included. They are not tokens, do not
measure unique information, and are not an exact reconstruction of what the
server retained in active context.

Nevertheless, the behavior is plainly oversized for the repository's output
discipline:

- Twenty-two of these 24 outputs exceed 8,000 characters. Median output length
  is 25,699.5 characters; several approach 40,000 characters.
- Every one of the 24 calls requests at least one inner output allowance above
  4,000 tokens. Their largest per-call requests range from 6,000 to 20,000.
  The final skill read explicitly requests 20,000, although its actual output
  is small. Thus that read triggered the next check; it did not produce most
  of the accumulated volume.
- Large reads include broad searches across source, tests, and plans, batches
  of manuscript and result-note ranges, and printing nested tool results.
  For example, call/output lines 771/776, 808/813, and 817/822 each preserve
  about 40,152 characters. The narrow UKF wiring question could have started
  at the named preparation helper and its consumers.
- Six post-compaction outputs contain truncation notices. A notice can occur
  inside a nested result; the metric records notice presence, not six proven
  server-context truncations. Repeated large reads after truncation are still
  visible in the source calls.
- The latest successful replacement history retains an 84,165-character user
  instruction message containing two copies of the global-policy banner and
  context-discipline section. All retained message text totals 100,978
  characters before the opaque compaction item. The first subsequent ordinary
  model request already uses 38,577 input tokens. That request includes other
  overhead too; it cannot all be attributed to policy duplication.

Across the complete session there are 128 tool outputs totaling 1,358,258
saved-text characters. That lifetime total must not be presented as the final
active context. There were three successful compactions earlier in this same
session, so the evidence does not support saying that compaction always failed.

## Counters that must remain distinct

| Counter at the final turn | Value | Meaning |
| --- | ---: | --- |
| Last ordinary request input | 111,672 tokens | Recorded model-request input before the new tool result |
| Last ordinary request output | 257 tokens | Includes reasoning; do not add reasoning again |
| Last ordinary request total | 111,929 tokens | Input plus output, session line 922 |
| App post-tool compaction measure | 124,873 tokens | Counter actually compared with the compaction threshold |
| Active automatic-compaction threshold | 120,000 tokens | Recorded in the final-turn app log |
| Recorded full model window | 258,400 tokens | App explicitly says it was not reached |
| Cumulative session usage | 10,501,808 tokens | Repeated request processing across the session, including cached input |

The difference between the server request count and the app's post-tool count
is not explained fully by the available records. They differ in timing and
accounting scope. It would be wrong to interpret the approximately 13,000-token
difference as the size of the skill file.

The current local configuration also has `model_auto_compact_token_limit =
120000` and `tool_output_token_limit = 2000`. The historical app log confirms
the threshold actually used in this incident; current configuration alone
would not prove that. The saved calls show that explicit large requests and
combined outputs defeated the intended small-output practice. The configuration
setting is not evidence that these tool results stayed within 2,000 tokens.

## Separate 12:43 AM connection failure

The first UKF question appears at 00:43:45 (lines 753–757). For that turn,
`01a0915a-8b2d-7280-84a2-5fd5aea01d77`, eight warning logs record
`Connection failed: error sending request`, with delays growing from 5 to 60
seconds. There is no token-usage record or tool result for that attempt, and
no recorded terminal event before the repeated question at 02:45. The repeated
question then completed successfully at 02:52. This earlier failure cannot be
classified as context exhaustion from the evidence inspected.

## Diagnosis and recovery

The immediate final failure is **confirmed remote-compaction failure**.
Avoidable context pressure is **confirmed contributing behavior**. A specific
provider defect, network cause, or size-dependent compactor bug is **not
established**. The separate connection failures make an infrastructure problem
plausible, but they do not prove one common cause for both incidents.

The practical next step is to continue the literature question in a fresh
conversation using [research-restart-brief.md](research-restart-brief.md).
Do not resume or fork this failed history for that work, and do not import this
incident report into the research conversation. Use exact file ranges, return
selected fields from structured tools, and keep each combined tool response
within the existing approximately 4,000-token budget. Explicit inner allowances
must fit that combined budget; printing entire nested results recreates the
problem. Preserve stage findings on disk and finish a bounded stage before
starting another large stage while compaction is unreliable.

The duplicated policy is a candidate for a separate, semantics-preserving
cleanup in its source/assembly mechanism. No policy was deleted, no context
window inflated, and no compaction threshold or other global setting changed
during this investigation. An earlier threshold is already installed; changing
it again is not an evidenced repair for a failing upstream compaction request.

This investigation produced no new scientific verdict about UKF or TT. Research
source and existing dirty changes in both checkouts were preserved. Attempts
to retrieve official documentation through the web tool returned HTTP 503;
all incident conclusions above rely on the inspected local records.

Reproduce the numerical summaries with:

```bash
python docs/plans/artifacts/codex-session-stall-20260912-01/analyze_session.py
```

The bounded output is saved in [analysis-run.log](analysis-run.log), with
machine-readable details in [session-metrics.json](session-metrics.json) and
[selected-app-events.json](selected-app-events.json).
