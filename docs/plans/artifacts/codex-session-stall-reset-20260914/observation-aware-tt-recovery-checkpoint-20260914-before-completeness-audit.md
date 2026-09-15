# Observation-aware TT recovery checkpoint — 2026-09-14

This note updates execution state recovered during a separate Codex incident
investigation. It does not authorize a new campaign or assert scientific success.
The detailed research plan remains
`docs/plans/observation-tt-pair-block-remedy-20260914.md`.

Active scientific question: whether pair cores, variable ordering, and
importance-weighted training rows remedy the TT fitting problem and improve
the downstream recursive proposal under the plan's validation and heuristic
checks. The failed conversation is `01a091d6-769a-7901-90e0-80900daba20e`.
Worktree: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`; verify current
changes before editing and preserve unrelated work. No subagents were launched
by this investigation.

The previous `observation-aware-tt-active-checkpoint.md` predates the following
completed attempts. All paths below are relative to
`docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/`:

| Attempt | Terminal record | Recorded wall time |
| --- | --- | ---: |
| `diagnostic-01` | `FAILED`; harness called an unavailable fitting function | 6.188 seconds |
| `diagnostic-02` | `EXECUTED`; selected configuration and fresh downstream fixture saved | 73.700 seconds |
| `campaign-01` | `FAILED`; pair conditional finite/bracket/CDF validity check | 151.322 seconds |

Read `campaign-01/failure.json`, `run_manifest.json`, and `traceback.log` first.
The partial `campaign-01/result.json` incorrectly remains labeled `RUNNING`;
it is not a terminal success record and must not be used to infer that a process
is still running. The failing consumer is
`bayesfilter/highdim/observation_guided_tt_tf.py:217`, reached from the master's
particle filter. Diagnose which validity condition failed before interpreting
the method or resuming execution. Do not overwrite these attempt artifacts.

The recorded attempts consume about 231.210 seconds from the previous checkpoint's
approximately 2,275 remaining numerical seconds, leaving approximately 2,044
seconds under that inherited estimate. This subtraction is a recovery estimate,
not a reconciled campaign ledger. One full campaign was launched after the old
checkpoint. Verify the original plan's remaining attempts, other charged checks,
and continuation vetoes before any further experiment. No budget was spent on
new numerical work in the incident investigation.

The proof/document and MathDev audit completion status has not been rechecked.
Retain the original checkpoint's pending manuscript build, rendered review,
audit interpretation, and final evidence-table obligations until verified.

Exact next research action, when resuming the existing authorized work: inspect
the terminal failure and its corresponding diagnostics, classify the failure
against the plan, reconcile the budget, and record the smallest justified
diagnostic or repair. A new session should load this note and only the needed
plan/source ranges rather than importing the failed conversation.

Incident findings:
`docs/plans/codex-compaction-investigation-20260914-result.md`. That investigation
found an unexplained reported input-token jump followed by upstream compaction
stream failures; the scientific validity failure is a separate event.
