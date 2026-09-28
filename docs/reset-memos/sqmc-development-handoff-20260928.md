> Completion update, 2026-09-28: the renewed expanded comparison is complete.
> Read `sqmc-development-checkpoint-20260928.md` for current state and
> `../benchmarks/sqmc-expanded-results-20260926.md` for results. All 32 units,
> 128 final cells and 8,480 scores were verified. The history below is preserved
> and must not be treated as instructions to relaunch completed work.

# SQMC development reset memo — 2026-09-28 Hong Kong

## Current task and authority

This is the detailed handoff for a fresh agent. The
[short checkpoint](sqmc-development-checkpoint-20260928.md) holds the next
action; this memo preserves the reasoning, evidence and unresolved problems.
It supersedes the active section of the older
[campaign reset memo](sqmc-4route-campaign-reset-20260922.md), whose historical
record remains intact. Applicable current repository instructions and the
owner's latest directions take precedence over older plans.

The immediate handoff request is to preserve the master program, current state,
problems, goals and context before starting a fresh conversation. No research
run, merge, push or environment change is being performed as part of writing
this memo. Preparing the prompt does not open a new VS Code conversation.

The outstanding scientific request is to test the existing model at T=10 and
T=120, then d=3 with a complete A/Q parameterization at T=2,10,120, then d=10
at those horizons. The owner authorized creating, reviewing and executing the
plan. The owner requires at least 1,000 particles for accuracy comparisons and
actual score values with absolute errors: an exact score may vanish at an MLE,
so relative score error is not a useful universal headline criterion.

## Checkout, provenance and preservation

| Item | Verified state |
|---|---|
| Working folder | `/home/chakwong/BayesFilter-SQMC` |
| Development branch | `sqmc-development` |
| HEAD | `f3995a06a467f16574f96bbc8a68ccbbc4e30dad` |
| Original Claude branch | `rqmc-sqmc-4route-comparison` |
| Original Claude worktree | `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909` |
| Recovered Claude session ID | `4b6ba369-219e-4a25-bbc3-547c5214d1e9` |
| Other main checkout | `/home/chakwong/BayesFilter`, branch `surrogate-hmc`; preserve unrelated work |
| Git state | Many modified tracked files plus untracked implementation, tests, plans and results |

Local `main`, cached `origin/main`, the original Claude branch and this
development HEAD were all at the SHA above during handoff inspection. This
confirms local ref agreement, not a new remote fetch. The earlier integration
work is recorded in the
[integration note](../benchmarks/sqmc-main-integration-20260924.md). Do not
repeat integration or switch the unrelated main checkout as a continuation
step. Git metadata for this linked worktree lives in the shared repository,
which may need permissions beyond the worktree folder itself.

The current HEAD alone cannot reproduce the repairs. The handoff evidence
directory contains [the state inventory](../plans/artifacts/sqmc-handoff-20260928/state.json),
[tracked changes](../plans/artifacts/sqmc-handoff-20260928/tracked-changes.patch),
[untracked working files](../plans/artifacts/sqmc-handoff-20260928/untracked-working-files.tar.gz),
and [file hashes](../plans/artifacts/sqmc-handoff-20260928/files.json).
The archive contains the selected untracked source, tests and text documents,
not the entire environment or all existing result directories. It is a
recovery aid, not permission to overwrite the live checkout. Preserve existing
dirty files, versioned results and failed attempts; do not reset, clean, stash
or replace them simply to obtain a clean branch.

## Which program is current

| Program or stage | State | Next implication |
|---|---|---|
| Original Claude four-route campaign | Closed with corrected, restricted conclusions | Historical record; not valid evidence for its former score-transfer claims |
| [Independent audit](../benchmarks/sqmc-independent-audit-20260925.md) and [repair master program](../plans/sqmc-repair-master-program-20260925.md) | Repair program complete within its stated local correctness scope | Read its result limits; do not equate completion with universal correctness |
| [93001 score diagnosis](../benchmarks/sqmc-score-93001-diagnosis-20260926.md) | Complete | Local derivative parity and particle approximation explain the checked anomaly |
| [N1008 diagnostic](../benchmarks/sqmc-n1008-results-20260926.md) | Complete | P44 d3 T2 comparison, not a full horizon/dimension study |
| [Expanded comparison plan](../plans/sqmc-expanded-comparison-20260926.md) | Active scientific master plan | Finish review, budget and GPU checks, then execute eight scopes |
| Full-matrix implementation and CPU checks | Implemented; 41 focused tests passed | Latest CPU checks do not replace GPU execution or long-horizon evidence |
| Full-d10 T2 timing pilot | Complete with one invalid final cell | Candidate failure and harness limitations must remain visible |
| Revised supervisor/worker runner | Implemented, not yet run after rewrite | Skeptical review and bounded GPU validation remain pending |
| Expanded result report | Not written; full ladder has not started | Produce only after execution or an honestly reported stop |

Older documents such as `sqmc-master-program-2026-09-12.md` and
`sqmc-control-generalization-master-program-2026-09-23.md` are historical
context. Do not revive their completion claims, procedural launch-token gates,
or untuned transfer assumptions. The newest applicable governance supersedes
retired ceremony; scientific validity and platform permissions still apply.

## What the audit established and repaired

The original campaign recorded 112 finite likelihood values: 16 in Phase 0,
32 in Phase 2 and 64 in Phase 3. Phase 1 was not executed. Phase 2/3 used
`with_score=False`; Phase 0's recorded scores were zero. Earlier serialization
failures were not successes. Those observations did not establish score
accuracy, transferability, no-retuning claims or production eligibility.

The independent audit found real bugs: missing parameter derivatives;
P44 covariance entries squared a second time and a changed initial law;
generator/executor disagreement on first-observation timing; duplicated
permutation ablation; missing score/oracle evidence; misinterpreted Fisher/HMC
metrics; accepted nonfinite sentinels; a multi-stage annealing regression;
overstated theory/equivalence/ranking claims; and incomplete scope, partition
and reproducibility enforcement. See A1–A12 in the linked audit rather than
reconstructing the findings from the old conversation.

The [repair results](../benchmarks/sqmc-repair-results-20260925.md) and
[mathematical corrections](../benchmarks/sqmc-repair-mathematical-corrections-20260925.md)
record their dispositions. Shared TensorFlow model specifications now define
the target, timing, analytical tangents and matched Kalman reference. Consumers
use shared route construction, complete analytical score assembly, distinct
cap ablation, validity rejection and exact-scope tuning. The multi-stage
annealed recursion was restored and checked. Unsupported historical claims
were withdrawn, and touched NumPy runtime paths were migrated.

Reported repair evidence: 73 CPU tests passed, one deliberately excluded
production-scale screen; eight final GPU graph/XLA compatibility cases passed;
a 32-cell diagnostic tuning workflow passed. Four old runner tests could not
collect because `docs/benchmarks/run_ledh_pfpf_genut_sqmc.py` is absent. Do not
represent those missing tests as passed. Review was by Codex, without an
independent Claude reviewer. These are checked local implementation results,
not a proof for every state, branch, model or horizon.

## Mathematical target and meaning of the score

All active comparison models use
`x0 ~ N(m0,P0)`, `xt = A x(t-1) + et`, `yt = H xt + vt`, with independent
Gaussian noises of covariances Q and R and H=I. The transition occurs before
**every** observation, including y1. The Kalman reference must use precisely
the same parameter coordinates, initial law, observations and timing.

The exact comparison target is the gradient of the Kalman marginal
log-likelihood. The candidate computes the analytical recursive derivative of
its finite particle program, with Contract-E reset and fixed numerical
controls. Those are different quantities at finite N and discretization.
Local finite-difference agreement checks differentiation of the finite
program; it does not make that program equal to the exact likelihood.
Sorting/ancestry branches restrict what a local derivative check establishes.

Claim-bearing candidate scores must use the shared analytical recursion.
Autodiff is confined to independent Kalman reference/parity diagnostics.
For the full model, `Q=L L^T` and `dQ=dL L^T+L dL^T`; Gaussian density
tangents include both covariance-determinant and inverse-covariance terms.
Callback tests exercise A, diagonal/off-diagonal L, R and initial-mean
dependence. The governing Contract-E total derivative includes direct source
moments/weights as well as transport dependence. A raw-barycentric or
transport-only derivative is not an acceptable replacement.

## Completed N1008 result and the 93001 concern

The large first-score discrepancy at N=12 was reproduced. Fixed-input local
analytical/finite-difference agreement supported the finite-particle likelihood
approximation as the explanation for that checked case, rather than an error
in its local derivative. This does not exclude bugs in untested cases.

The N1008 diagnostic retained the original P44 d3 T2 observations and reported
72/72 finite GPU cells. Four graph/XLA parity checks passed; maximum reported
absolute backend difference was 3.33e-16. Its numerical settings were frozen
diagnostic settings, not newly tuned N1008 controls.

Exact score order: transformed persistence, log Q scale, log R scale, initial
mean scale. For dataset 93001 the exact vector was
`[-0.148673923, -0.595155460, -0.561413331, 0.183929553]`; for 93002 it was
`[-0.284567727, -1.000254008, -0.541294107, -0.047953059]`.

| Route | 93001 original-input N1008 score | Absolute vector L2 error |
|---|---|---:|
| IID | [-0.076757352, -0.599967229, -0.582425275, 0.183581179] | 0.075078422 |
| Inverse CDF | [-0.150462238, -0.595976433, -0.556905652, 0.184010552] | 0.004919126 |
| Permutation | [-0.149003760, -0.595130222, -0.558301733, 0.183831866] | 0.003130657 |
| Permutation cap .97 | [-0.148860562, -0.595207149, -0.558270719, 0.183832747] | 0.003150062 |

The Hilbert routes' earlier sign discrepancy disappeared on that dataset at
N1008. IID remained noisier for that particular realization. The linked report
contains every score, the second dataset, and eight extra filter scrambles.
Permutation and its .97 cap ablation were close; “ablation is by far the best”
is unsupported. All stochastic differences here are descriptive. There is no
predeclared uncertainty analysis establishing a ranking, and no HMC or default
promotion follows.

## Expanded campaign specification

| Order | Model | d | T | N | Score coordinates |
|---:|---|---:|---|---:|---:|
| 1–2 | Existing P44 | 3 | 10, 120 | 1008 | 4 |
| 3–5 | Full A/full Q | 3 | 2, 10, 120 | 1020 | 17 |
| 6–8 | Full A/full Q | 10 | 2, 10, 120 | 1020 | 157 |

P44's fixed theta is `[.25, log(.18), log(.12), .04]`. At this point A is
diagonal approximately `[.1347052643,.1144994747,.0942936850]`, Q diagonal
`[.162,.198,.234]`, R diagonal `[.12,.144,.096]`, m0=`[.04,-.02,.01]` and
P0 diagonal `[.6,.8,1]`. These are fixed test points, not MLEs.

The full model has all d² entries of A; all d(d+1)/2 lower-Cholesky coordinates
of Q (log diagonal, unconstrained off-diagonal); log observation-variance
scale; and initial-mean scale. R=.12I, P0 diagonal linspace(.6,1,d),
m0=.04 times `[(-.5)^i]`. A diagonal is linspace(.65,.45,d), off-diagonal
`.12*(-1)^(i+j)/(d-1)`. L diagonal is sqrt(linspace(.16,.24,d)), lower entries
`.08*(-1)^(i+j)/sqrt(d-1)`. The row-absolute-sum bound for baseline A is .77;
the plan also calls for Q conditioning diagnostics. Score order comes from
the specification's `parameter_names`; never relabel Cholesky-Q scores as
derivatives with respect to covariance entries.

All four routes are mandatory:

| Route ID | Ancestry | Coordinate cap |
|---|---|---:|
| `iid_dual_cap` | `existing_one_to_one` | .98 |
| `previous_inverse_cdf` | `hilbert_inverse_cdf` | .98 |
| `repaired_permutation` | `hilbert_permutation_one_to_one` | .98 |
| `repaired_permutation_ablation` | `hilbert_permutation_one_to_one` | .97 |

The ablation changes a cap, not the presence of permutation. Each exact scope
calibrates flow substeps 2 versus 8 on data seeds 195001,195002; validates on
196001; and freezes settings before final data seeds 197001,197002 crossed
with filter seeds 198001,198002. There are four final cases per route/scope,
32 route/scope units and 128 final cases if every tuning unit is valid.
The nominal total is 288 numerical evaluations including calibration and
validation, before parity checks or retries. Report invalid units explicitly.

Other frozen controls: epsilon .4; Sinkhorn 24; balance 12; correction one
step, strength .12; pairwise one step, strength .03; Contract-E ridge 1e-5;
LM damping .01; scale floor 1e-4; trust radius .5; pairwise RMS cap 2;
coordinate-cap power 8; adaptive empirical chart; Hilbert bits 12.
These are inherited baseline hypotheses, not proven optimal or safety-tuned
defaults. The two-point flow calibration is limited. Preserve the quantitative
protection/calibration limitation in any conclusion.

Use FP64 GPU/XLA, TF32 disabled, memory growth verified before initialization;
this is an explicit reference-precision comparison, not FP32/TF32 production
evidence. N1020 satisfies the 2d residual design for both full dimensions.
Transport chunks obey K=N for these particle counts. CPU-only small-N tests
are mechanics/reference exceptions, not substitutes for these comparisons.

Primary reporting: exact and estimated likelihoods and all raw score entries,
signed/absolute component errors, per-component RMSE, vector L2 and parameter
block RMS. Keep dimensions separate because L2 grows with coordinate count.
Comparators: exact Kalman, IID route, zero score, and first-observation-only
Kalman score, measured against the **full-horizon** Kalman score. Losing to
cheap baselines vetoes promotion; no ranking is inferentially established by
two independent datasets. No predeclared universal accuracy threshold exists.

## Current implementation and checked call boundaries

| File | Role and current state |
|---|---|
| [sqmc_lgssm_tf.py](../../bayesfilter/highdim/sqmc_lgssm_tf.py) | Shared P44/diagonal specifications, simulation, matched cached CPU Kalman reference; `reference_value_and_score(theta, observations)` takes theta first |
| [sqmc_full_lgssm_tf.py](../../bayesfilter/highdim/sqmc_full_lgssm_tf.py) | Full A/Q specification and analytical model callbacks; untracked, must preserve |
| [sqmc_campaign_tf.py](../../bayesfilter/highdim/sqmc_campaign_tf.py) | Shared route settings, inputs, stable XLA kernels, complete score assembly, invalidity diagnostics |
| [sqmc_campaign_tuning.py](../../bayesfilter/highdim/sqmc_campaign_tuning.py) | Scope-bound issued tuning artifacts, disjoint partitions, final evaluation with separate filter seed, progress callback |
| [run_sqmc_expanded_comparison.py](../benchmarks/run_sqmc_expanded_comparison.py) | Revised standard-library supervisor and per-scope/route GPU workers; not executed since rewrite |

The score chain is expanded worker → shared tuner/evaluator →
`sqmc_campaign_tf.value_and_score` → cached `_kernel` →
`ledh_canonical_score_tf.canonical_value_and_analytical_score`, with model
callbacks supplied by the shared specification. The finite-program tests
exercise that shared boundary. A prose description of this chain does not
replace the still-pending GPU check through the revised consumer.

Full score assembly evaluates sequential parameter directions through the
analytical core, with stable TensorFlow signatures. It recomputes the primal
per direction; cost grows with the 157-coordinate model. Added diagnostics
record invalid coordinate indices and primal-value disagreement across
directions, while preserving the three-value public return convention.
Cached non-XLA CPU oracle graphs avoid repeated Kalman tracing; the oracle's
AD remains a diagnostic exception, never the candidate score engine.

Saved tuning JSON is evidence, not a caller-loadable authority. Final evaluation
requires the repository-issued in-process artifact with matching source and
scope. Do not construct a self-attested replacement to speed a resume.

The revised runner uses separate worker processes, external 90-minute unit
timeouts, incremental results, logs, observations, random-design hashes,
source hashes, GPU policy records, raw-score CSV and heuristic comparisons.
It retains invalid candidate rows instead of converting them into a successful
comparison. Worker exceptions still abort for localized repair. Resume/reuse
of completed units is not implemented in the supervisor and needs review
before relying on the plan's reuse language.

## Latest CPU validation

The completed command used the tftwogpu interpreter with
`CUDA_VISIBLE_DEVICES=-1`, `TF_CPP_MIN_LOG_LEVEL=2`,
`TF_NUM_INTRAOP_THREADS=4`, `TF_NUM_INTEROP_THREADS=2`, running:

```text
python -m pytest -q tests/highdim/test_sqmc_full_lgssm.py
    tests/highdim/test_sqmc_expanded_execution.py
    tests/highdim/test_sqmc_campaign_repairs.py
```

The three paths were arguments to one invocation (the display wraps it).
Result: **41 passed, 2 TensorFlow Probability deprecation warnings, 43.96 s**.
The [complete log](../plans/artifacts/sqmc-handoff-20260928/sqmc-expanded-focused-checks.log)
is now preserved outside `/tmp`.

Checks cover all-coordinate full-model callbacks, Kalman derivatives/timing,
diagonal reduction, Cartesian seed pairs, the correct heuristic error target,
and fixed-direction finite-program differences for d3/d10 across all four
routes. Directional tests use N=4d,T=2, central step 2e-6 with declared
tolerances; they are bounded mechanics tests, not high-particle accuracy
evidence. GPU parity after the latest wrapper/oracle/runner changes is pending.
No full-suite, universal-mathematics or production-readiness claim is made.

## Expanded pilot attempts and their limits

Evidence root: [sqmc-expanded-20260926](../plans/artifacts/sqmc-expanded-20260926/).
The old manifests retain their original state, including stale `running`
labels for crashed attempts. Do not rewrite them as if they had clean exits.

| Attempt | Observed outcome |
|---|---|
| 01-pilot | Started 2026-09-26 17:49:50 UTC without the old pilot-only flag, so began P44 d3 T10. Reporting failed with an undefined TensorFlow name after initial tuning |
| 02-pilot | Full d10 T2; a reversed Kalman-reference argument call caused a rank error after initial work |
| 03-pilot | Full d10 T2; inverse-CDF invalidity was wrongly treated by the old harness as a whole-run abort |
| 04-pilot | Full d10 T2,N1020,P157; finished 18:21:47 UTC, 317.424727 s, four routes, 28 tuning/validation/final evaluations |

Attempt 04's final cells used only two zipped pairs, 197001/198001 and
197002/198002. These are the raw full-vector absolute score errors:

| Route | First pair | Second pair |
|---|---:|---:|
| IID | 5.1322723145 | 9.0134362254 |
| Inverse CDF | 6.8382070561 | INVALID |
| Permutation | 3.8916849688 | 5.9423778117 |
| Permutation cap .97 | 3.8843620474 | 5.9255383244 |

The invalid cell was flagged as nonfinite or inconsistent across directions;
the exact cause is **not diagnosed**. New diagnostic fields were added later.
Seven valid cells do not establish superiority of permutation or its ablation.
All 157 raw coordinates remain in the
[pilot results](../plans/artifacts/sqmc-expanded-20260926/attempt-04-pilot/results.json).

Pilot limitations: old heuristic error compared the estimate to the first-only
score, rather than comparing that baseline to the full oracle; only two of
four final pairs were run; raw observations/designs were not fully preserved;
the runner and shared diagnostics subsequently changed. Its raw scores and
timing are pilot evidence, not completion of any requested final scope.
Previously viewed final pairs cannot be counted again as independent evidence.
The revised manifest discloses their reuse and that controls were not changed
in response to the pilot's final errors. Do not claim fresh untouched evidence
for those repeats or tune on them.

## Remaining problems, decisions and continuation program

| Issue | Evidence class and status | Required next action |
|---|---|---|
| Original elapsed budget expired | Verified timestamps; continuation boundary | Reconcile active resource usage and obtain/document a new elapsed window through the fresh-session instruction; no silent compute increase |
| Prior GPU usage | Exact total not reconciled; runner reserves 1200 s | Audit preserved attempts; the first-to-last pilot window spans about 32 minutes, so a 20-minute reserve is not automatically conservative |
| Revised runner not GPU-executed | Implementation exists, no latest end-to-end GPU evidence | Skeptical review, then one bounded parity/cost check within the reconciled budget |
| Inverse-CDF invalid final cell | Hard veto for that cell; root cause not checked | Use new diagnostics to localize; preserve candidate failure and continue unrelated valid scopes |
| Claim-data reuse | Two diagonal pilot pairs already observed | Label repeated evidence honestly; any claim of a new untouched holdout needs fresh predeclared seeds |
| Resume and retry | Plan permits reuse; supervisor currently starts all units | Review safe unit reuse/source matching and timeout/retry status; avoid rerunning completed units merely for convenience |
| Source coverage and input replay | New hashes implemented, full closure/replay not independently checked | Verify transitive source coverage, observations, designs, score lengths and shared baselines before treating provenance as complete |
| Fixed numerical protections | Inherited controls, narrow flow-only tuning | Retain limits; no default promotion without appropriate scope-specific numerical evaluation |
| Long-horizon/d10 cost | Only T2 pilot measured | Project with compilation/direction costs, monitor first expensive units; do not reduce N, coordinates, routes or datasets silently |
| Statistical ranking | Unsupported under the present descriptive design | Report raw differences and uncertainty limitations; a defensible ranking needs a separate predeclared replicated comparison |

The continuation program is:

1. Verify the new session root/permissions, branch, HEAD, dirty files and this
   checkpoint. Read only the exact source/plan sections needed for the next
   decision; do not reconstruct context by dumping old sessions.
2. Complete the skeptical runner/plan audit: baselines, full Cartesian design,
   oracle timing and coordinates, invalid-cell handling, source closure,
   unique outputs, timeouts, retry accounting, resume semantics and whether
   artifacts answer the scientific question. Fix confirmed localized defects.
3. Reconcile compute use and elapsed authorization. Record a single concise
   plan amendment if needed, maintaining the scientific target and aggregate
   resource cap. Only then run bounded GPU parity/cost validation.
4. Execute the eight scopes in the declared order, with exact-scope tuning,
   full score vectors, separate child logs and incremental evidence. Preserve
   failures; a rejected candidate does not alone reject the research direction.
5. Audit expected cases, invalidity, score dimensions, data/input identity,
   tuning provenance and exact Kalman comparisons. Write
   `docs/benchmarks/sqmc-expanded-results-20260926.md` (currently absent),
   detailed CSV/JSON results and a concise material-changes account.
6. Complete terminal review, decision/inference-status tables, remaining
   limitations and updated checkpoint. No automatic default, HMC, theorem,
   statistical-superiority or production promotion.

Hard validity failures include wrong target/timing, broken derivative/oracle
checks, missing coordinates, nonfinite output, inconsistent directional primal
values, mismatched data, failed required replay or tuning-scope mismatch.
They invalidate affected evidence and trigger repair. Shared implementation
failure, unsafe resource use or exhausted budget can stop continuation.
Finite but large error is a scientific result and repair trigger, not authority
to hide a route or skip later requested scopes.

## Environment, commands and permissions

Interpreter: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
Prior manifests report TensorFlow `2.20.0-dev0+selfbuilt`. The runner selects
GPU UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, previously identified as
RTX 4080 SUPER. Recheck availability/ownership; CUDA ordinal zero was not
equivalent to nvidia-smi index zero. No environment or package mutation is
needed to resume this task on current evidence.

All GPU/CUDA probes and jobs require trusted/escalated access under the local
policy. Set and verify TensorFlow memory growth before device initialization;
record allocator usage and device identity. CPU checks must hide GPU devices
before framework import. No pfor or NumPy candidate runtime, no new derivative
backend, and no silent eager/non-XLA replacement for the GPU comparison.

The supervisor's current invocation shape, **after** the outstanding checks,
is:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_sqmc_expanded_comparison.py --output docs/plans/artifacts/sqmc-expanded-20260926/attempt-05-full --budget-seconds REMAINING_SECONDS
```

`REMAINING_SECONDS` is a placeholder, not an executable argument. Choose a
fresh output directory after checking it does not exist. The current default
42000 assumes a 1200-second prior-use reserve and must not be taken as verified
accounting. The revised CLI has `--worker-scope` and `--worker-route`; the
historical `--pilot-only` option no longer exists. Review worker flags before
using one as a diagnostic because a worker includes tuning/final evaluations.

The old conversation's effective writable roots remain
`/home/chakwong/BayesFilter` and temporary directories. Its location is not
changed by using a different `workdir` argument or opening the SQMC folder in
VS Code. The user's latest environment message still shows that restriction.
This explains repeated write/GPU approval interruptions. Non-escalated calls
also sometimes failed with a bubblewrap mountinfo error; such failures are not
evidence of broken scientific code or CUDA.

Observed local Codex config uses `approval_policy="on-request"`,
`approvals_reviewer="auto_review"`, `sandbox_mode="workspace-write"`.
Automatic review changes who evaluates approvals, not writable roots. No live
policy was changed by this handoff. The installed extension's Full access
setting was verified locally; managed restrictions may still override it.
Use the [fresh-agent prompt](sqmc-new-agent-prompt-20260928.md) in a new SQMC
conversation and verify the actual profile. If permissions remain restricted,
use normal escalation rather than bypassing them.

Simple script-file invocations with the approved interpreter or narrow
repository commands avoid many unnecessary approval prompts. Complex shell
strings, heredocs and broad interpreter prefixes can require additional review.
Do not modify allow lists or credentials silently. User campaign authorization
already covers routine local implementation, repairs and planned experiments;
do not invent a fresh approval gate for every retry or report.

## Handoff acceptance and limitations

The handoff preserves current files, plans, known failures, actual result
boundaries, environment details, test evidence and a source recovery snapshot.
Its inventory records inspection time and matching-process checks. No matching
expanded-runner or pytest process was found during inspection; stale pilot
manifest labels are not evidence that a job remains active.

This is a checked reset memo, not a fresh complete mathematical or GPU audit.
The new agent should proceed from the pending stage, not erase the existing
audit or count historical checks as validation of later source changes.
No new VS Code primary agent has been launched from this conversation.

