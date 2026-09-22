# Reset memo: repeated Codex compaction stalls in the observation-aware TT thread

Date: 2026-09-14  
Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`  
Purpose: recover the scientific and operational state without reopening the
oversized conversation that repeatedly failed during remote compaction.

## Active question

Why did this observation-aware TT thread stall twice during one investigation,
and was the cause the agent's tool use or the Codex/provider session state?
This memo records the two saved rollouts relevant to the incident, the work
they were trying to supervise, the governing master program, the recovered
research status, and the next safe starting point.

## Saved-session chain

### 1. Original research rollout

Session `01a091d6-769a-7901-90e0-80900daba20e`, saved as
`/home/chakwong/.codex/sessions/2026/09/12/rollout-2026-09-12T02-59-06-01a091d6-769a-7901-90e0-80900daba20e.jsonl`.

This was the long observation-aware TT implementation and experiment thread.
It added the pair-block representation, conditional sampler, weighted
regression rows, SGQF guidance, manuscript propositions, focused tests, and a
GPU/XLA campaign. The rollout contains 5,212 lines, 37,063,835 serialized
bytes, 626 tool results, 595 custom tool calls, 12 successful compactions, and
83 truncation notices. Across its lifetime the tool results decode to about
2.82 million characters; 130 results exceed 8,000 characters and 30 exceed
20,000 characters.

The important tool-use pattern is repeated broad and overlapping reads. For
example, one batched call requested large portions of `fitting.py` and tests
with a 16,000-token allowance; the next three calls reread overlapping ranges
of `squared_tt.py`, `tt.py`, `bases.py`, and `fitting.py` with several
10,000--12,000-token allowances. The outer calls did not declare a combined
output cap, so the nominal per-command limits accumulated. Several outputs
were truncated at roughly 40,000 characters. Near the end, the campaign driver
was reread in three overlapping ranges, including a 20,000-token allowance.
This behavior created avoidable context pressure and made compaction work
harder, but it does not by itself explain the terminal failure.

The original thread reached a successful compaction at 04:20:07 China time on
2026-09-14 and continued. Later, reported input usage jumped from 58,972 to
130,862 tokens after a short wait, with no intervening large tool result. The
runtime counter crossed the 120,000 automatic-compaction threshold. The client
also logged `OutputTextDelta without active item` shortly beforehand. Automatic
compaction retries then failed with `Upstream request failed`; a subsequent
explicit compact turn failed before any tool call could run. The saved rollout
therefore ends with no completed supervision message after the failure.

### 2. Investigation/recovery rollout

Session `01a09c96-afab-7662-89c0-9dc5737ec0ff`, saved as
`/home/chakwong/.codex/sessions/2026/09/14/rollout-2026-09-14T05-05-16-01a09c96-afab-7662-89c0-9dc5737ec0ff.jsonl`.

This was the first investigation of the original stall. It correctly matched
the attachment to the original rollout, measured tool output and neighboring
threads, and concluded that tool use contributed pressure but was not a
sufficient explanation. It then answered a follow-up asking what the research
agent had attempted, its result, current status, next step, and whether a
master program governed execution. At that point the rollout had 169 lines,
1,066,094 bytes, 624 tool results, 25 token-count records, and cumulative
reported input usage of 1,800,248 tokens (the effective context window was
258,400 tokens). The increasing values are cumulative accounting over many
turns, not a claim that 1.8 million tokens were simultaneously resident.

The follow-up turn started at line 121 and produced only a short progress
message at line 127. Its terminal event at line 166 failed with
`Error running remote compact task: stream disconnected before completion:
Upstream request failed`. A retry turn started immediately at line 167 and
failed with the same error at line 169. Neither failed turn made a tool call or
produced an agent answer. This is the first clear “twice in one roll” event:
the automatic/retry compaction path failed, then the newly started follow-up
also failed during compaction before execution.

### Present reconnect rollout

The current saved rollout is session `01a09dec-81e1-7002-9b22-3ec0dc604fc2`,
`/home/chakwong/.codex/sessions/2026/09/14/rollout-2026-09-14T11-18-37-01a09dec-81e1-7002-9b22-3ec0dc604fc2.jsonl`. It contains the owner request for this memo and bounded inspection so far. It has 31 lines and six token-count checkpoints; its latest reported input is 262,244 tokens against a 258,400-token declared window. It has not yet recorded a terminal compaction error, but it is already at the boundary where another automatic compaction may be attempted. This memo is therefore the handoff point; do not continue a large investigation in this thread.

## What the research agent was doing

The research target was an observation-aware TT proposal for the Zhao--Cui
stochastic-volatility filtering experiment. The earlier scalar workflow had
wrong initialization, bounded-support proposals, dependent innovations,
missing retained-TT propagation, unweighted guide rows, silently floored
weights, and no independent filtering reference. The repair implemented and
tested an adjacent-state pair-block TT route with SGQF observation guidance,
affine coordinate charts and Jacobians, L1-tuned weighted regression rows,
retained marginalization, particle-specific conditional KR sampling, and
importance correction using the actual proposal density. The manuscript was
being updated with propositions and proofs, and MathDevMCP was used for
individual symbolic checks; the document-wide audit returned a tool-execution
error and could not certify the integral/tensor claims.

The latest complete-program result is not a promotion. The 20-observation,
d=1 and d=4 GPU/XLA campaign passed finite/reference mechanics screens, but
the guided TT had descriptive conditional losses against simple proposals.
The separate pair-remedy campaign then failed a pair conditional finite,
bracket, and CDF validity check. The terminal manifest records `FAILED` after
151.322 seconds and `failure.json` points to
`observation_guided_tt_tf.py:217`; a partial `result.json` still says
`RUNNING` and is stale. The representation diagnostic retry (`diagnostic-02`)
completed in 73.700 seconds; its first attempt failed after 6.188 seconds.

## Governing master program and goal

The governing plan is
`docs/plans/observation-aware-tt-repair-complete-program-20260913.md`, with
the executable master
`docs/benchmarks/run_observation_aware_tt_complete.py`. Its question is
whether recursive SGQF observation guidance repairs adjacent-state
square-root TT regression sufficiently to produce usable, exactly weighted
conditional proposals. It requires SGQF checks, exact affine change of
variables, adjacent-state TT fitting including the previous TT law, retained
joint/marginal densities, conditional KR sampling, original-model particle
correction, independent reference comparisons, and frozen analytical scores.

The evidence contract makes finite proposal/weight behavior and agreement
with an independent filtering reference the primary screen. Density, CDF,
Jacobian, signed-mass, covariance, non-finite, and reference-disagreement
checks are vetoes. Fit residual, ESS, and timing are explanatory or repair
triggers, not correctness proofs. The campaign is bounded by its plan's
launch and numerical budgets; a failed TT arm is a repair signal, not evidence
against the entire research direction.

The older `observation-aware-tt-repair-full-master-20260913` result is only a
diagnostic candidate: its mechanics passed but it explicitly did not establish
source-faithful paper-scale KR, posterior correctness, convergence, HMC
readiness, or production readiness. The complete-program result similarly
blocks promotion because conditional comparisons and higher-dimensional
evidence remain unresolved.

## Diagnosis of the repeated stall

The evidence supports two contributors with different status:

1. **Agent-side context growth is real.** The original research rollout used
   too many overlapping broad reads, high per-command output allowances, and
   insufficient outer budgets. Duplicated governance/instruction text also
   remains in serialized context after compaction. This raises the frequency
   and difficulty of compaction and is a preventable process defect.
2. **The terminal failure is provider/client-side or session-specific until
   disproved.** The decisive 58,972-to-130,862 input jump had no corresponding
   tool result. Compaction requests received HTTP 200 from
   `cn.origincoder.com/v1/responses` but disconnected while streaming. The
   error was not `context_length_exceeded`. The manual retry failed before the
   agent could reduce context. Local evidence cannot distinguish gateway usage
   accounting, request replay/state handling, routing, or a client/provider
   compatibility issue.

This is not explained by “large files alone.” A same-client BayesFilter
neighbor had 624 tool results, 177 results over 8,000 characters and 71 over
20,000, with 12 successful compactions and no comparable input jump. A
MacroFinance neighbor had 990 tool results, 49 over 20,000, and 14 successful
compactions. These are snapshots rather than controlled experiments, but they
show that output volume is not a sufficient trigger. They also do not prove
the gateway is globally healthy.

## Reset procedure

Start the next research turn in a fresh conversation from
`docs/plans/observation-aware-tt-recovery-checkpoint-20260914.md`. Preserve
all three saved rollouts and the incident artifacts; do not fork or paste the
raw oversized transcript. First assess the pair conditional validity failure
against the frozen plan and remaining campaign budget. Use filename discovery,
small exact line ranges, selected JSON fields, and an outer output budget of
about 2,000 tokens per read and 4,000 per batch. Write a checkpoint after each
substantive stage and before any new launch.

The practical tool rule is to inspect names and counts first, then read only
the lines needed for the next decision. After truncation, narrow the query;
never reread the same large file in several overlapping ranges. Keep context
investigation separate from scientific execution. A gateway-side explanation
requires the sanitized request IDs and timing records in
`docs/plans/artifacts/codex-compaction-20260914-01/`; changing the compaction
threshold or global client settings is not justified by this local evidence.

## Evidence paths and limits

The incident analysis and metrics are in
`docs/plans/codex-compaction-investigation-20260914.md`,
`docs/plans/codex-compaction-investigation-20260914-result.md`, and
`docs/plans/artifacts/codex-compaction-20260914-01/`. Those records include
session hashes, event counts, tool-use examples, runtime counters, failure
logs, and peer endpoint snapshots. Official product documentation could not
be retrieved in that investigation (search/open returned 502; direct fetch
returned 403), so the provider-level diagnosis remains unverified. No code,
Codex configuration, saved session, database, or running process was changed
by this reset investigation.
