# Zhao-Cui stalled-session investigation

## Active checkpoint

Investigation complete. Excessive tool output and retained incident history
caused avoidable context pressure; the final stall was failed remote compaction
through `llm.visioncoder.ai`. Ordinary request failures also occurred below the
compaction threshold, so output volume does not explain every provider failure.
No mathematical audit, numerical experiment, or global configuration change was
performed during this investigation.

The attachment identifies session
`01a08d08-5bf7-7cb0-a8f9-d64660a69a8f`, saved at
`/home/chakwong/.codex/sessions/2026/09/11/rollout-2026-09-11T04-35-30-01a08d08-5bf7-7cb0-a8f9-d64660a69a8f.jsonl`.
The existing recovery report instead measures the earlier session
`01a08be9-85ba-7f21-8e53-a2dbf566ab7d`. Verify both separately.

Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`. Preserve the
existing dirty policy files, LEDH tests, Zhao-Cui notes, and other untracked work.

Preflight: compare recorded output sizes with runtime context counters, not
with cumulative billed usage. A large saved file or character count alone does
not establish active token use or the server-side cause. Match tool calls with
outputs, inspect compaction triggers, and distinguish normal response failures
from failures during compaction. Use exact-session structured parsing and
indexed read-only log queries; never print raw rollout or instruction payloads.

Budget used: bounded local incident inspection, no model probes, experiments,
GPU work, or environment changes. Official-documentation search/open failed with
HTTP 503; direct official configuration-page retrieval returned HTTP 403 even
with trusted permissions. Product conclusions must be supported locally or
explicitly left unverified.

Evidence directory: `artifacts/zhao-cui-context-investigation-20260911-01/`.
Authoritative measurements: `latest-session-final.json`; earlier-session
cross-check: `earlier-session-summary.json`; reproducible parser:
`session_diagnostic.py`. Intermediate summaries are preserved, but the final
summary additionally records decoded text sizes, HTTP responses, and terminal
errors. Runtime log queries use the exact thread's SQLite index in read-only
mode. They do not expose request bodies, credentials, or raw conversations.

Next research action: start a fresh conversation using
`zhao-cui-integrated-audit-restart-2026-09-11.md`. Do not load this incident report
into the research conversation. The launcher now selects that brief; it
previously selected the superseded aggregate-guard implementation stage.

## What the latest session actually did

The latest session started fresh, then accumulated six user turns spanning
recovery, algorithm inspection, incident diagnosis, prevention advice, policy
installation, and the integrated mathematical audit. Preparing a fresh-session
launcher did not restart the existing VS Code conversation. The recorded active
threshold remained 244,800, not the launcher's proposed 120,000.

| Measurement | Latest session | Earlier session covered by the old report |
| --- | ---: | ---: |
| Completed tool calls and matching results | 66 / 66 | 28 / 28 |
| JSON-serialized result characters | 543,546 | 657,799 |
| Decoded text characters | 516,665 | 615,967 |
| Results containing truncation markers | 6 | 7 |
| Successful saved compaction records | 0 | 0 |

The earlier 657,799-character finding reproduces exactly. It describes a
different session; reusing it as the measurement of the final failure would be
wrong. Serialized character sizes include JSON structure and escaping. Decoded
text sizes remove that outer representation, but neither measure is a token
count. The final cumulative usage total, 9,666,142, is accumulated usage across
requests, not the active context size or a verified monetary charge.

The first model-reported turn used 38,066 total tokens. The injected instruction
message alone occupied 82,344 serialized characters and contained the global
scientific policy twice, through global and project instructions. That is a
fixed startup cost, distinct from subsequent tool growth. Removing safeguards
or required project instructions is not an appropriate incident repair.

## Concrete tool-use defects

1. Rollout call line 19 loaded nine recovery paths with up to 280 lines per
   file and a 16,000-token allowance per command. It listed the same diagnostic
   script twice. The nominal sum of command allowances was 144,000 tokens;
   this was not the number actually returned. Result line 21 contained 41,314
   serialized characters and was truncated.
2. Call line 31 combined five source/status commands at 16,000 tokens apiece.
   Result line 33 was another truncated 41,322-character payload. The first
   recovery/audit turn alone accumulated 316,717 serialized result characters.
3. All 66 `exec` invocations omitted an explicit outer `max_output_tokens`
   pragma. An outer default still existed; it was not the promised 4,000-token
   aggregate cap. Per-command limits do not constrain a batch's total.
4. After installing bounded-output policy, calls 446 and 451 still requested
   aggregate command budgets of 7,000 and 5,200 tokens. The respective saved
   outputs contained 18,284 and 13,841 serialized characters. The agent printed
   complete selected command results instead of consistently extracting the
   fields needed for its next decision.
5. MathDev discovery call 434 printed 7,000 characters from `doctor` plus
   4,149 characters of tool descriptions. `doctor` returned. The session made
   no MathDev proof call; subsequent reads inspected its implementation and
   source files. There is no evidence of a hung mathematical proof tool.
6. At the integrated-audit boundary, the existing runtime counter was already
   226,575 tokens, leaving 18,225 before the active compaction threshold. Four
   subsequent batches added 59,708 serialized result characters and the
   counter crossed that threshold. Persisting the integrated plan did not
   remove the conversation's earlier history.

These are failures to follow the stated protocol, not evidence that the
protocol had never been installed. Each inspected global/project instruction
file has one installed context-discipline section. No global policy was
modified in this investigation.

The concrete correction is to request a small aggregate allowance at the
outer tool boundary, for example `// @exec: {"max_output_tokens": 3500}`, and
allocate smaller command budgets within it. Even that cap cannot make a broad
read useful: save the full result to disk and return only the selected status,
counts, source anchors, and evidence paths. Adding more policy text without
changing these calls repeats the failure. The launcher's tool-output setting
is a precaution; this investigation did not test its enforcement against
every tool type or explicit per-call override.

## Failure timeline

All times below are Hong Kong time on 2026-09-11. Runtime counters are Codex's
local post-sampling estimates; model-reported counters are a separate record.

| Time | Evidence | Interpretation |
| --- | --- | --- |
| 04:41:55 | Task error at rollout line 151; last runtime counter 130,518, log 456958530 | Ordinary request failure below threshold. |
| 04:57:39 | Task error at line 277; last runtime counter 170,782, log 456962496 | Another ordinary failure below threshold. |
| 12:07:31 | 226,575 tokens, log 457023951 | Integrated audit began in an already large conversation. |
| 12:10:47 | 241,233 tokens, log 457024887 | Only 3,567 tokens remained before compaction. |
| 12:11:04 | 245,203 tokens; threshold 244,800; `token_limit_reached=true`, log 457025048 | Automatic mid-turn compaction triggered. Full window 258,400 had not been reached. |
| 12:16:05 to 13:12:35 | Eleven HTTP 504 response records during compaction | Gateway repeatedly timed out. Retry-warning entries are additional log records, not additional HTTP responses. |
| 13:15:21 | HTTP 200 response headers, log 457035347 | Headers did not establish a completed compaction stream. |
| 13:15:47 | Terminal task error, rollout line 455 | `Error running remote compact task: stream disconnected before completion: Upstream request failed`. |

The interval from the compaction trigger to the terminal error was about
64 minutes 43 seconds. The two outer retry warnings are logs 457029817 and
457033001. The eleven 504 response IDs are preserved in the final JSON; the
last response request ID is `cfcf50c8-7ca9-47b0-b034-2c2a87972d5d`.

The final model-reported last-request usage was 215,123 total tokens. The
runtime's 245,203 counter, with `token_limit_reached=true`, is the direct
evidence of why Codex initiated compaction. Their difference must not be
interpreted as an independently measured amount of tool text.

The underlying reason the gateway or its upstream could not finish the request
is not established by client logs. Large history made compaction necessary;
it does not prove a particular server memory limit, model limit, timeout
configuration, or payload rejection. A provider investigation would need the
timestamp/request ID and its server logs. No external message or live model
probe was sent.

## Recovery state and verification

The dedicated checkout remains on `zhao-cui-tt-regression-20260908`, base
`47176bce7cdaa91ddd4466c39f90a11ad005c800`, with the saved uncommitted work.
The integrated-audit plan exists, but its promised artifact directory does
not. The completed `doctor` check and source inspection do not amount to a
completed mathematical audit. Earlier bounded findings remain in stage-03
artifacts and were not rerun or independently re-proved here.

The restart brief preserves the most recent scientific request: analyze code,
LaTeX, and the complete algorithm; propose coherent repairs; use MathDev for
mathematical assertions. It does not instruct an agent to immediately apply
the earlier isolated runtime repair. The existing launch script was adjusted
to select that brief and state an explicit outer-output policy. Its 120,000
compaction and 2,000 tool-output settings remain trial precautions, not proof
that the gateway is fixed. The script was not launched.

The parser was checked by independently aggregating decoded text sizes and by
reproducing the previous session's published 657,799-character result. All
66 call IDs have matching results, and the final saved task is terminal.
Launcher syntax and print-only behavior were checked without starting an
agent. No algorithm source, environment, session database, or saved rollout
was edited.
