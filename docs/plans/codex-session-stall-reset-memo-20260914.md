# Reset memo: observation-aware TT work and two Codex compaction stalls

Verified and corrected on 2026-09-14. Checkout: `/home/chakwong/BayesFilter`,
branch `surrogate-hmc`. This memo replaces an incomplete and partly incorrect
first version, preserved in
[the pre-audit copy](artifacts/codex-session-stall-reset-20260914/memo-before-completeness-audit.md).
This audit inspected saved evidence and updated handoff documents; it ran no
new numerical experiments and changed no scientific implementation, Codex
configuration, saved session, or running process.

The latest work is the **pair-block SGQF/TT remedy**, not the earlier completed
scalar-TT campaign. The representation diagnostic completed and froze rank 3,
four sweeps, and 128 proximal steps. The downstream campaign **failed** during
4D pair-conditional sampling. Its completed 1D result already records a
heuristic-underperformance promotion veto. The remedy is implemented and
partly tested; it has not passed the complete downstream evidence contract.
The original request for proposition/proof exposition, MathDevMCP audit,
reviewed tests, execution, and a finished manuscript remains partly open.

For a fresh research conversation, start with the
[active checkpoint](observation-aware-tt-active-checkpoint.md), then read only
the relevant section of the
[current pair-block plan](observation-tt-pair-block-remedy-20260914.md).
Use this memo as a reference. Do not paste both saved sessions, all old plans,
or the entire incident investigation into the research conversation.

## 1. User intent and governing work

The research concerns the C2 stochastic-volatility filtering example and an
observation-aware tensor-train (TT) proposal. SGQF supplies a Gaussian
observation guide and coordinate charts; the retained TT filtering density
must propagate separately from the particle population. The goal is a correct,
useful recursive proposal with explicit importance correction and evidence
against simple alternatives, not merely a low regression loss.

The original session's relevant owner directives, in order, were:

1. Create, thoroughly review, and execute a master program implementing the
   remedy (saved-session line 1480).
2. Check whether all discussed remedies, including importance sampling, were
   implemented and tested (line 3399).
3. Identify the remaining TT-regression gaps (line 3468).
4. Trace the code and mathematics, especially the first-transition error and
   the joint treatment of current and previous state (line 3532).
5. Explain how the serious regression problem could be mitigated (line 3786).
6. Fully document the recommendation in LaTeX in proposition/proof form,
   audit it with MathDevMCP, and create, review, and execute a test plan
   (line 3995; 2026-09-13 19:17:24 UTC).

The user's supplied
[last-conversation attachment](/home/chakwong/.codex/attachments/f03ef00c-1393-4d74-b320-ec6ea4f5fa45/pasted-text.txt)
ends with the downstream campaign reported as running, followed by compaction
errors. The terminal files on disk now establish that the campaign failed.
The investigation and reset requests did not cancel the scientific objective.
They also did not request an unbounded new research campaign.

| Authority or implementation | Role at resumption |
| --- | --- |
| [Pair-block remedy plan](observation-tt-pair-block-remedy-20260914.md) | Latest scientific target, comparison design, frozen choices, budget, and stop conditions. |
| [Complete master](../benchmarks/run_observation_aware_tt_complete.py) | Still the downstream executable; the latest route requires `--pair-block`. |
| [Pair representation diagnostic](../benchmarks/diagnose_pair_block_tt_remedy.py) | Fresh d2/d4 representation and weighted-row comparisons; produces the selected downstream configuration. |
| [Pair implementation](../../bayesfilter/highdim/pair_block_tt_tf.py) | Pair fitting, evaluation, retained marginal, and conditional-sampling support. |
| [Observation-guided TT consumers](../../bayesfilter/highdim/observation_guided_tt_tf.py) | Recursive target construction, proposals, and the consumer that raised the current error. |
| [Gaussian/Hermite proposal support](../../bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py) | Shared proposal/sampling support used by the master. |
| [Complete-program plan](observation-aware-tt-repair-complete-program-20260913.md) and [result](observation-aware-tt-repair-complete-program-20260913-result.md) | Earlier scalar SGQF/TT stage and inherited campaign budget; not the latest next action. |
| [First-transition diagnosis](observation-tt-first-transition-root-cause-20260914.md) and [result](observation-tt-first-transition-root-cause-20260914-result.md) | Motivation for the pair/order/fitting repair. |
| [Regression gap assessment](observation-aware-tt-regression-gap-assessment-20260914.md) | Historical gap list; statements that weighted rows are entirely missing predate the pair implementation. |

The older
[full-master plan](observation-aware-tt-repair-full-master-20260913.md) and
[result](observation-aware-tt-repair-full-master-20260913-result.md), including
its historical runs 01–20, do not establish correctness of this repaired route.
Current `AGENTS.md` and the owner's instructions govern. Preserve unrelated
LEDH, policy, and manuscript edits in the dirty checkout.

## 2. Mathematical target and what the remedies actually change

The intended adjacent-state target is

\[
\gamma_t(x,z)=\widehat\pi_{t-1}(z)f_\theta(x\mid z)g_\theta(y_t\mid x).
\]

The separate invertible charts `x=m_x+L_x u` and `z=m_z+L_z v` do not make
current and previous state independent. Their determinants enter the pulled-back
density. The fitted amplitude is the square root of that density divided by
the product Gaussian base density `rho(u)rho(v)`. The earlier target already
contained the joint transition dependence. A diagnosis that it simply fitted
two independent filtering laws would be wrong.

The grouped scalar order `(u1,...,u4,v1,...,v4)` imposes a restrictive middle
TT-rank cut on temporal dependence. The saved first-transition diagnostic
reported approximately 0.461 relative error for the saved fit, only 0.032
relative change from replacing the inherited density, a degree-three grouped
rank-three floor near 0.280, a feasible grouped TT-SVD approximation near 0.283,
and a paired reference near 0.095. These are diagnostic results on the old
frozen target, not new holdout or downstream performance evidence. They support
both a representation obstruction and a remaining fitting gap. They do not
isolate optimization, sample coverage, L1 regularization, or initialization as
the sole fitting cause.

Pair cores carry `(u_i,v_i)` together with shape
`[left_rank, current_degree, previous_degree, right_rank]`. Their retained
marginal uses an exact positive matrix contraction. Conditioning on each actual
previous particle produces batched scalar current-state cores, which use the
shared Hermite conditional sampler. Arbitrary joint whitening that mixes both
state blocks cannot be substituted into the existing marginal and conditional
consumers without changing their mathematics.

Two distinct importance corrections must survive resumption:

- **Regression rows:** sample from `s = epsilon*rho + (1-epsilon)*s_joint`
  and weight the loss by `rho/s`. Here `epsilon=0.2`, so the ratio is bounded by
  5. This preserves the Gaussian population objective. The implemented loss
  divides by sample count, not the random sum of weights, and freezes the
  training target scale before validation/audit.
- **Filtering particles:** use the original model numerator divided by the
  actual particle-specific TT/Gaussian-mixture proposal, including physical
  chart Jacobians. This is distinct from changing where regression rows are
  drawn.

The pre-launch review repaired use of the SGQF smoothed marginal covariance
where the backward **conditional** covariance was required. It also repaired
paired-row indexing, inactive rank initialization, duplicated sampler logic,
and random-sum loss normalization. L1 selection and the factorization's gauge
still require the declared interpretation; an L1 penalty on one factorization
is not an invariant penalty on the represented function.

| Remedy or claim | Checked saved evidence and remaining limit |
| --- | --- |
| SGQF observation guidance, separate affine charts, retained-density recursion, and particle importance correction | Implemented in the earlier corrected master; its successful execution did not solve the regression problem. |
| Pair representation, exact marginal contraction, conditional dependence, weighted rows, and defensive Gaussian fallback | Implemented; 18 focused tests passed and a tiny compiled CPU/GPU parity check passed. The full 4D sampling path subsequently failed a validity guard. |
| Variable ordering, ranks 2/3/4, weighted versus unweighted rows, and one solver repair | Bounded fresh diagnostic completed. Selection used d4 validation; results do not establish that pair fitting dominates scalar interleaving. |
| All possible TT-regression remedies | Not completed. Broad degree/row-count studies, fitting-seed replication, independent-sequence validation, higher-dimensional scaling, and alternative methods such as CUT4 or DMIS/control variates were not all integrated and tested. |
| Zhao–Cui source faithfulness | The pair path is explicitly `extension_or_invention`, not a source-faithful TT-cross implementation. |
| Statistical superiority, HMC readiness, or default readiness | Not established. Four particle seeds do not replicate uncertainty in the learned TT construction. |

## 3. Verified execution state and frozen configuration

All latest run paths in this section are under
[`docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/`](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/).
Terminal manifests outrank partial result files and old chat updates.

| Attempt | Terminal status | Recorded wall time | Meaning |
| --- | --- | ---: | --- |
| `diagnostic-01` | `FAILED` | 6.187937 s | Harness called unavailable `fit_from_log` instead of the available fitting API. Repaired locally. |
| `diagnostic-02` | `EXECUTED` | 73.699802 s | Fresh representation comparisons completed; configuration and downstream fixture saved. |
| `campaign-01` | `FAILED` | 151.322206 s | 1D completed; 4D pair conditional sampling raised a validity error. |

The [focused-test log](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/focused-tests.log)
records **18 passed, 2 warnings, 8.00 s** for
`tests/highdim/test_pair_block_tt_remedy.py` and
`tests/highdim/test_observation_guided_tt_tf.py`.
The [parity record](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-01/cpu_gpu_parity.json)
records `PASS`, maximum difference `2.220446049250313e-16`. The parity check is
inside diagnostic-01's recorded elapsed time; do not charge it twice. These
checks establish their tested identities and fixture behavior, not complete
end-to-end validity. They were inspected, not rerun, during this handoff audit.

### Frozen data and selection

Use the actual
[selected-configuration file](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-02/selected_configuration.json)
and [downstream fixture](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-02/downstream_fixture.json).

| Field | Frozen value or rule |
| --- | --- |
| Physical model parameters | Frozen C2 parameters carried by the saved fixtures; do not substitute a new target. |
| Observation simulation seeds | Calibration `2026091401`; downstream `2026091402`. |
| Diagnostic dimensions and target | d2/d4, t=1 with SGQF Gaussian incoming density, to isolate representation. This is not the downstream recursive TT target. |
| Fitting grid | Degree 3, 1,024 rows per split, pair ranks 2/3/4, weighted/unweighted arms, L1 grid `{0, 1e-5, 1e-3}`. |
| Common independent Gaussian audit | Seed `58100+d`; never used to select configuration. |
| Selected downstream pair settings | Rank 3, four sweeps, 128 proximal steps, selected on **d4 validation**. |
| Bounded solver repair already exercised | Eight sweeps/256 steps; the selected d4 configuration remained the baseline solver. KKT threshold 0.001 is explanatory, not a convergence certificate. |
| d2 choice | Rank 4 independently selected; do not replace the downstream d4-selected setting with it. |
| Downstream dimensions/horizon | d1 and d4, T=20. The d4-selected setting applied to d1 is a transfer diagnostic, not scope-specific d1 tuning. |
| Particles/reference | 512 particles; 32,768 reference particles. |
| Training/reference seeds | `64100` / `74100`. |
| Particle seeds | `2101, 2102, 2103, 2104`. |
| Downstream CLI | `--pair-block`, pair rank 3, scalar rank 3, degree 3, rows 1,024, four sweeps/128 steps, internal wall cap 1,490 s. |

The exact executed command, environment, settings, and dependency hashes are in
the [campaign manifest](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/run_manifest.json).
The executable environment was `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`,
TensorFlow `2.20.0-dev0+selfbuilt`, float64, GPU/XLA, TF32 enabled, threads 2/1,
and `TF_FORCE_GPU_ALLOW_GROWTH=true`. Memory growth was verified on the RTX 5080
(`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`). The manifest records allocator peak
39,686,144 bytes. This scope's float64 configuration must not be silently
replaced with another repository lane's numerical default.

The recorded Git commit is `14a292098f35b6de450ffca29131ea34f4a4d8d7`, with dirty
source changes. At this audit, all five launch dependency hashes matched the
current master, observation-guided consumer, Gaussian/Hermite support, pair
implementation, and pair plan. A commit identifier alone would miss these
uncommitted additions. The [verification record](artifacts/codex-session-stall-reset-20260914/resumption-verification.json)
preserves the five checks and budget arithmetic. Recheck those hashes before
further implementation work.

The diagnostic's common-audit relative RMS values are descriptive: in d4,
scalar grouped was about 0.4425, scalar interleaved 0.1081, the selected pair
fit 0.1169, and the allowed solver repair 0.1316. The scalar and pair fitters
also differ in initialization. These values do not establish a statistically
supported ordering or uniquely attribute improvement to pair cores.

### Downstream result and precise failure

The [1D decision](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/d1/decision.json)
records successful reference evidence/moment screens for the six methods, but
the pair method's heuristic verdict is
`DESCRIPTIVE_UNDERPERFORMANCE_PROMOTION_VETO`. Default readiness is false and
statistical ranking is not established. Its recorded losses are to transition
bootstrap in ordinary and large observations, and to the SGQF Gaussian in large
observations. Those are descriptive losses that activate the predeclared veto;
they are not a statistically supported ranking. A frozen-score `PASS` concerns the
frozen finite program; it is not a total derivative through adaptive refitting.

In d4, all 20 saved pair proposal fits exist, along with comparator evidence,
but pair particle evaluation did not complete and there is no completed d4
decision. The log reports late audit residuals about 0.774 at t17, 1.308 at t18,
and 0.415 at t19. A completed fitting loop is therefore not proof that the
recursive regression problem is repaired.

Read [failure.json](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/failure.json)
and [traceback.log](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/traceback.log).
The exception is:

```text
ValueError: pair conditional failed finite/bracket/CDF check
```

The call chain is master `main` → `execute` → `particle_filter` →
`observation_guided_tt_tf.sample_pair_tt_step`, at line 217 of the audited
consumer. After `compiled_pair_sampler`, that consumer raises if
`diag["finite"]` is false or `diag["cdf_residual"] > 1e-8`.
**The raised error does not preserve which condition fired, the residual,
timestep, or particle seed. Those facts are still unknown.**

The [d4 saved pair proposals](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/d4/tt_pair_block_proposals.json)
contain the 20 sets of cores, mixture settings, and current/conditioning charts.
They support a deterministic sampling replay without rerunning regression.
The sampler derives mixture choices, uniforms, and Gaussian noise from
stateless seeds. The first pair-lane particle proposal is SGQF Gaussian; the
retained fitted initial law is used in subsequent regression. Preserve this
behavior when reconstructing the path.

The partial [result.json](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-01/result.json)
still says `RUNNING` and contains only d1. It is a stale progress snapshot,
not a terminal verdict or evidence of a still-running process. Preserve it
alongside the failed manifest rather than silently rewriting historical output.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the selected configuration frozen for diagnosis | Representation diagnostic completed; full downstream contract incomplete | d1 heuristic promotion veto and unresolved d4 sampling validity veto | Exact failed sampler condition and complete recursive accuracy | Reproduce the sampling failure from saved fits | Overall remedy success or rejection of the research direction |
| Withhold promotion | d1 reference screens passed; d4 pair decision missing | The existing vetoes remain active | Construction/sequence uncertainty beyond four particle seeds | Finish valid downstream evidence after a justified repair, within budget | Superiority, default readiness, or HMC readiness |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | d4 sampler validity failed; exact condition unresolved. d1 heuristic losses veto promotion under the plan. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Fit residuals and conditional heuristic losses; four particle seeds and one sequence. |
| Default readiness | False. |
| Next evidence needed | A valid completed recursive comparison, followed by the replication/uncertainty evidence required for any stronger claim. |

## 4. Budget and evidence boundaries

The inherited numerical budget was 2,400 seconds and three full launches. The
latest pair plan inherited approximately 2,275 seconds and two full launches,
and specified one fresh downstream test plus bounded representation work.

| Recorded research charge | Seconds | Manifest location under `docs/benchmarks/artifacts/` |
| --- | ---: | --- |
| Earlier complete-program campaign | 99.574079 | `observation_aware_tt_complete_20260913/campaign-01/run_manifest.json` |
| First-transition diagnostic run-01 | 13.103820 | `observation_tt_first_transition_20260914/run-01/run_manifest.json` |
| First-transition diagnostic run-02 | 11.546747 | `observation_tt_first_transition_20260914/run-02/run_manifest.json` |
| Pair diagnostic-01 | 6.187937 | `observation_tt_pair_block_remedy_20260914/diagnostic-01/run_manifest.json` |
| Pair diagnostic-02 | 73.699802 | `observation_tt_pair_block_remedy_20260914/diagnostic-02/run_manifest.json` |
| Pair campaign-01 | 151.322206 | `observation_tt_pair_block_remedy_20260914/campaign-01/run_manifest.json` |
| **Total of these charges** | **355.434591** | **2,044.565409 seconds remain under this ledger.** |

Additional documented checks are the earlier CPU mechanics (4.944046 s),
GPU/XLA mechanics (11.145976 s), and the 18-test suite (8.00 s). Charging these
as well leaves **2,020.475387 seconds**. The inherited research ledger did not
include those checks; neither figure is a guarantee that every historical
routine check has been metered. Reconcile the charging convention and any
other checks before spending more budget. This audit spent no numerical
campaign budget.

Two full downstream launches have now occurred across the original campaign.
The latest subplan's one fresh downstream launch is consumed. The unused slot
in the inherited three-launch ceiling is not permission for new candidate
selection on the exposed downstream observations. A localized repair/retry
must be documented under the current campaign-repair policy, respect both
limits, preserve the target and frozen settings, and use a fresh output directory.
Routine localized repair does not itself require renewed owner approval.

The sampler validity failure is a continuation veto for the current execution
until diagnosed and resolved. The d1 heuristic loss is a promotion veto; it
alone does not reject the research direction. Do not relax a CDF or mass check
to force a run through. Do not tune on the failed downstream data and then
present the same data as untouched confirmation. A new scientific selection
requires fresh calibration and untouched evaluation within a revised valid
plan and available budget.

## 5. Unfinished LaTeX, proof, and audit obligations

The manuscript is
[attempt05_observation_aware_tt_algorithm_note.tex](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex).
The protected pre-pair version is
[attempt05_note_before_pair_remedy_20260914.tex](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_note_before_pair_remedy_20260914.tex),
with its accompanying SHA-256 record. Preserve both.

The new pair section, around lines 4000–4374, contains eight propositions:
`pair-unitary`, `pair-rank`, `pair-marginal`, `pair-conditional`,
`sgqf-backward`, `pair-importance`, `pair-weighted-fit`, and `pair-gauge`
(each prefixed by `prop:` in the LaTeX labels). They cover the charted density,
representation/rank, marginal/conditional algebra, backward covariance,
importance weights, weighted fitting, and L1 gauge dependence.

The saved [MathDevMCP records](artifacts/observation-tt-pair-block-20260914/)
establish a narrower result than a proof certificate:

| Audit artifact | Actual outcome |
| --- | --- |
| `mathdev-tree-error.json` | Document-wide tool execution failed. |
| Eight `mathdev-prop-*.json` records | All theorem audits are `inconclusive`; the bounded backend could not encode the integral/tensor statements. An `ok: true` execution field is not a theorem verdict. |
| `mathdev-scoped-identities.json` | Nine finite scalar identities were reported equivalent: density cancellation, quadratic pair expansion, retained-polynomial expansion, conditional mixture, CDF derivative, backward variance, weighted loss, weighted core objective, and gauge cancellation. |
| `mathdev-weight-identity.json` | One additional scalar weight identity was reported equivalent. |

The plan records local skeptical review and pre-launch corrections. It does
not claim independent model review. Neither symbolic identities nor advisory
review certify the entire manuscript or the failed numerical path.

The LaTeX source was modified at 2026-09-13 20:29:19 UTC. The existing
[same-basename PDF](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf)
and build log are older (16:57:02 UTC); the 39-page PDF does not include the
latest pair section. The new section promises detailed audit/execution results
below, but is immediately followed by the appendix. The latest pair result
writeup is missing.

Remaining work is to diagnose the sampler, report the actual completed and
failed results and mathematical audit limits, finish the relevant derivations
and discussion, build the updated LaTeX, and inspect the rendered document.
Compare against the protected baseline to preserve equations, assumptions,
citations, and substantive qualifications. A successful build or a few symbolic
identities cannot be reported as full mathematical or scientific acceptance.

## 6. The two stalled sessions, correctly identified

Saved source paths and SHA-256 hashes are preserved in the linked summaries.
Times below use Asia/Shanghai (UTC+8); event timestamps in JSON are UTC.
These are two distinct conversations, each with a failed retry. The current
memo-writing conversation is not a third investigated stall.

| Record | Original research conversation | Investigation conversation |
| --- | --- | --- |
| Session ID | `01a091d6-769a-7901-90e0-80900daba20e` | `01a09c96-afab-7662-89c0-9dc5737ec0ff` |
| Saved JSONL | `/home/chakwong/.codex/sessions/2026/09/12/rollout-2026-09-12T02-59-06-01a091d6-769a-7901-90e0-80900daba20e.jsonl` | `/home/chakwong/.codex/sessions/2026/09/14/rollout-2026-09-14T05-05-16-01a09c96-afab-7662-89c0-9dc5737ec0ff.jsonl` |
| Saved size / lines | 37,063,835 bytes / 5,212 lines | 1,066,094 bytes / 169 lines |
| Tool results across saved conversation | 626 | 24 custom tool calls and 24 results |
| Successful `compacted` records | 12 | 0 |
| Final normal request's reported input | 130,862 tokens | 104,009 tokens |
| First terminal failure in the incident | Sept 14, 04:42:18 | Sept 14, 11:08:40 |
| Failed retry ended | Sept 14, 04:51:25 | Sept 14, 11:13:07 |
| Parsed evidence | [Original summary](artifacts/codex-session-stall-reset-20260914/01a091d6-769a-7901-90e0-80900daba20e-summary.json) | [Investigator summary](artifacts/codex-session-stall-reset-20260914/01a09c96-afab-7662-89c0-9dc5737ec0ff-summary.json) |

Both terminal failures have the same recorded message:

```text
Error running remote compact task: stream disconnected before completion:
Upstream request failed
```

### Original research incident

The last ordinary turn began at 03:17:24 local time and combined proof writing,
MathDevMCP calls, implementation, diagnostics, and the downstream launch. At
04:38:52 its request reported 58,972 input tokens, including 57,088 cached
input tokens. At 04:39:03 the next reported input was 130,862, with zero cached
input: a **71,890-token increase** after a small clock-wait response, not a
corresponding giant file dump. The saved runtime counter rose from 59,072 to
131,303, crossing the recorded 120,000 auto-compaction threshold. The effective
context window in the saved records was 258,400.

The original
[incident investigation](codex-compaction-investigation-20260914-result.md),
[runtime counters](artifacts/codex-compaction-20260914-01/runtime-context-counters.json),
and [failure logs](artifacts/codex-compaction-20260914-01/failure-logs.json)
record failed remote compaction through the then-configured gateway
`cn.origincoder.com`, using VS Code client `0.154.0-alpha.6.2` and
`gpt-6-astra`. Initial HTTP streaming success did not become a completed compact
response. The failed retry started at 04:47:18 and ran no research tool calls.

The TT campaign's numerical exception and the Codex compaction exception are
separate failures. The scientific process could write its terminal failure
while the agent lost its ability to report or investigate it in that turn.

### Investigation incident

This conversation initially completed its investigation at 05:22:24 local
time. At 10:57:03 the user asked for the scientific work/results/status/next
step/master program. Before failing, the agent made **seven tool calls**, at
saved lines 129, 135, 140, 147, 152, 157, and 162. These reread plans, results,
and source ranges; several cells requested four reads with large per-command
budgets. Six of the seven returned outputs were truncated.

The seven serialized output-text lengths range from 17,317 to 39,581
characters. These lengths include wrapper text, so they must not be directly
compared with the older analyzer's decoded-text counts or treated as token
counts. The [call index](artifacts/codex-session-stall-reset-20260914/investigation-tool-calls.json)
and session summary preserve the actual evidence. The investigating agent
repeated the broad-reading pattern it had just diagnosed.

The last normal request reported 104,009 input tokens. Its cumulative input
usage was 1,800,248 and cumulative total usage 1,831,424 across repeated
requests. Those cumulative figures are not simultaneous context size. The
saved conversation contains 29 token-count events and 25 token-usage records;
those are also different counters.

After the 11:08:40 compaction failure, a retry began at 11:08:53 and failed at
11:13:07 without executing another tool call. Thus only the retry was blocked
before any tools; the initial stalled follow-up had performed seven calls.
The exact internal compaction trigger for this second incident was not
reconstructed from a matching runtime log. Its last reported normal-request
input cannot by itself establish that trigger.

## 7. Why this recurred, and what is still unproved

The observed failure location is remote history compaction. The evidence
supports two interacting problems: avoidable context pressure from the
workflow, and a compaction response that the configured client/gateway/model
path did not successfully complete. It does not establish one exclusive cause.

**Tool use contributed avoidable pressure.** The first investigation counted
approximately 2.82 million decoded tool-output characters over the research
conversation: 130 results exceeded 8,000 characters and 30 exceeded 20,000.
It found overlapping large source reads and batches with four separate
16,000-token allowances and no explicit outer cap. The investigator later
repeated broad, truncated reads. These are concrete tool-discipline defects,
not merely the fact that a session was long.

**The immediate original failure is not explained by a final oversized read.**
After its last successful compaction, the original session had only 42 tool
results totaling 42,292 decoded characters; no result exceeded 20,000. Normal
input grew from about 43,181 to 58,972 before the unexplained jump to 130,862.
A zero-cache report alone does not explain why the input count increased.
Prompt reconstruction, instruction expansion, provider accounting, routing,
and client/provider compatibility remain hypotheses requiring further evidence.

**Fixed instruction overhead also survives compaction.** The prior evidence
found an 84,164-character retained user instruction message containing two
copies of the global scientific-policy heading, plus about 16,001 characters
of developer messages. A serialized permission update contained 583 approved
prefixes and 156,777 characters. These are character measurements, not measured
active-token contributions. Large permission state also appeared in comparison
threads; it is not a proved cause of this anomaly. Any policy deduplication
should preserve applicable requirements and be a separate bounded task.

**Other agents working on the machine is compatible with these findings.**
Saved comparison sessions successfully compacted, including histories with
large tool outputs. That weakens a simple machine-wide resource or total-file-size
explanation, while not ruling out an intermittent shared service defect. A
specific conversation can repeatedly encounter a problematic payload or
compaction path. No controlled replay isolates that payload, and no provider-side
request trace establishes the backend root cause. The exact reason these two
histories failed while those peers succeeded remains unresolved.

Official documentation defines `model_auto_compact_token_limit` as the
threshold that triggers automatic history compaction; it does not promise that
a gateway will successfully complete the compaction request. See the
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
The recorded 120,000 setting is local historical evidence, not a recommended
universal threshold. Inflating the declared context window, removing governing
instructions, or repeatedly retrying an already blocked turn would not establish
a fix. No configuration or provider repair was tested in this audit.

## 8. Exact resumption sequence

1. **Restore the task, not the entire transcript.** In a fresh research
   conversation, load the active checkpoint and applicable instructions.
   Verify checkout/branch, relevant working-tree changes, and the five launch
   dependency hashes. Use the latest pair plan's evidence contract and stop
   conditions; the old scalar master is not the current plan.
2. **Localize the 4D sampling validity failure from saved fits.** Inspect the
   terminal failure, exact consumer guard, saved d4 cores/charts, and frozen
   fixture. Plan the smallest deterministic replay to find the time and seed
   and distinguish non-finite/bracket failure from excess CDF residual.
   Preserve the already-computed diagnostics and reproducing inputs before
   raising. Regression need not be refit for this localization. This is the
   next research action; it was not executed by this memo audit.
3. **Repair only what the evidence supports.** Audit the proposed replay/repair
   against the target and budget before running it. If a localized harness or
   numerical implementation error is found, make the focused repair and the
   necessary identity, normalization, and CPU-reference/GPU-XLA checks. GPU
   work requires trusted/escalated execution and memory growth. Preserve the
   failure artifact and keep CDF/mass/finite checks active. A changed sampling
   measure, target, or numerics-altering protection needs its own justification
   and evidence; do not label it the same frozen method without analysis.
4. **Reconcile continuation and resources.** Record elapsed charges and launch
   allowance, which validity veto was resolved, and the smallest remaining
   comparison. A localized retry may continue under the existing authorization
   and unchanged campaign contract when those conditions are met. Use a new
   versioned directory. If new scientific tuning is needed, preserve exposed
   holdout results and obtain fresh partitions within a valid remaining plan.
   Stop when a true continuation veto remains or the budget is exhausted.
5. **Finish the scientific result, even if the candidate fails.** Report
   engineering correctness separately from numerical validity and scientific
   interpretation. Include all required d1/d4 comparator and observation-regime
   evidence, the d1 heuristic veto, uncertainty and missing evidence. Use the
   plan's decision and inference-status tables. No superiority claim follows
   from descriptive residuals, four particle seeds, or one observation sequence.
6. **Finish the original manuscript/audit request.** Replace the unfulfilled
   promise of results with the actual evidence, reconcile propositions with
   checked identities and inconclusive audits, rebuild and inspect the PDF,
   and preserve the protected baseline. Update the concise checkpoint after
   each meaningful result and before a large operation.

During recovery, discover filenames or counts first; inspect one selected
source range or a few structured fields per call. Aim for about 2,000 tokens
per tool response and 4,000 per combined batch, including the outer wrapper.
Save complete logs on disk and return status, key facts, and paths. A truncated
result calls for a narrower query. At completed research stages, use a fresh
conversation initialized from the checkpoint while compaction remains
unreliable. Keep further incident investigation separate from numerical work.
These practices reduce context pressure; they do not certify a provider fix.

## 9. Completeness and corrections to the first memo

This is a resumption record with an evidence index, not a lossless copy of
every message. It now captures the latest objective, governing plan, executable,
mathematical target, implemented remedies, completed and failed attempts,
frozen data/configuration, provenance, budget limits, document obligations,
both incident timelines, and a concrete next action. The exact failure
condition, final campaign validity, complete theorem verification, fresh PDF,
and backend compaction root cause remain explicitly open.

| First-memo defect | Correction now recorded |
| --- | --- |
| Earlier complete-program stage described as the active plan | Latest pair-block plan and `--pair-block` master route identified. |
| Current reconnect thread treated as one of the two stalled sessions | Correct research and investigator IDs, source paths, and event timelines recorded. |
| Investigator credited with 624 tool outputs | Its actual 24 outputs are recorded; the larger count belonged to a comparison history. |
| Both investigator failure turns described as having no tools | Seven calls before the first terminal failure; zero calls in the retry. |
| Cumulative 262,244 input tokens in the memo-writing thread treated as near-window context | That was cumulative accounting; the corresponding request input was 48,535. It was not evidence of a third imminent stall. |
| Latest selection, failed campaign, and remaining work insufficiently captured | Frozen rank/solver/data, d1 veto, d4 guard, saved replay inputs, budget, MathDev limits, and stale PDF documented. |
| Broad assurance that all context was preserved | Replaced by this explicit coverage statement and the unresolved facts above. |

The authoritative incident evidence remains the two saved JSONL files and the
small parsed records linked above. The earlier
[incident evidence directory](artifacts/codex-compaction-20260914-01/) also
contains the parser, original/peer metrics, tool-use examples, endpoint
comparisons, and runtime logs. These support targeted follow-up; they are not
startup reading requirements. Earlier conclusions remain historical unless
supported by the current plan and checked artifacts.
