> Update 2026-10-06: the owner replaced the integration-worktree/extraction stage with the direct local patch in `ledh-marginal-minimal-patch-2026-10-06.md`. Its parity checks are complete; see the corresponding result note. Retain this plan for the separate future PP/SIR canary and synchronization stages.

# Shared LEDH importance correction: merge and nonlinear canary plan

Status: proposed and self-reviewed, 2026-10-06. No implementation, canary,
merge, push or default promotion has been executed under this plan. The user
requested a merge plan after asking for predator–prey and SIR canaries. Proposed
experiment allocation: six aggregate job-hours within the additional 48 hours.

## Decision and checked starting point

Integrate marginal weighting through one shared value/analytical-score
implementation. Preserve the `ancestor` default and expose `marginal_mixture`
explicitly. Run PP and SIR d=18 canaries before accepting the integration.
Mechanical correctness and scientific accuracy are separate decisions.

The inspected marginal campaign tested LGSSM, KSC stochastic volatility and
range bearing. It supplies no PP/SIR marginal-weight result. Earlier PP/SIR
results tested different repairs using ancestor weights.

| Item | Read-only inventory |
|---|---|
| Main and cached origin/main | `a925f67a19d8c5d265b8ff71bfd2c78ac1964ff4`; no fresh remote fetch for this plan |
| Current branch | `sqmc-development`, `0b91a64f6d38009b5f081d7981f51fab958425c0`; dirty research work |
| Source branch | `ledh-graceful-failure-execution-20260917`, `f5e69d716818ff9ecf28b732a8952f5ec7f87739`; dirty work also present |
| Source versus main | Three commits ahead, zero behind; combined diff: 434 files |
| f5e69d716 alone | 173 files; 127,485 insertions, 718 deletions; includes other inference work and evidence |
| Main worktree | `/tmp/bayesfilter-sqmc-main-integration-20260924`; verify live status before use; another stale main registration exists |

**Do not merge the source branch or cherry-pick f5e69d716 wholesale.** The
previous method review covered the weighting capability, not every bundled
change. Extract and review its complete dependency set from immutable commits.
The source campaign helper became dirty during inspection; live source files
must not silently substitute for the pinned version.

Source inspection finds both canonical adapters eligible: PP has Q=4 I_2 and
SIR has Q=I_18, with no custom transition-density or structural-innovation
callback. Executed wiring remains to be checked. Earlier SIR T=20/N=1008 FP32
runs all failed; FP64 runs were finite but inaccurate. Distinguish shared old
failure from new regression. Evidence:
`docs/benchmarks/ledh-nonlinear-execution-results-20261002.md`.

## 1. One correction implementation

For one child per ancestor, preserve

\[
 V_i=\omega_i g(y\mid X_i)R_i,\qquad
 R_i(S_i)=\frac{\sum_{j\in S_i}\omega_j f_j(X_i)}
                 {\sum_{j\in S_i}\omega_j q_j(X_i)}.
\]

Only S_i={i} and S_i={1,...,N} are supported. The former reduces to f_i/q_i;
the latter is the full mixture. There is one incoming weight vector, and the
outer omega_i remains necessary. This notation does not justify arbitrary
subsets. Inactive zero-mass ancestors must not cause a literal 0/0 calculation.

For full-rank Gaussian transitions and admissible affine flows,
q_j=N(F_j(m_j), B_j Q B_j^T). Obtain F_j, B_j and their analytical tangents
from the actual flow loop. Include particle, incoming-weight, covariance,
flow-map and reset dependence in the total score of that same finite likelihood.
Verify the condition that flow coefficients are independent of the current
sampled innovation.

Use one correction interface and shared component-density/tangent calculations.
Diagonal and full support may have different indexing/reduction strategies:
ancestor evaluation stays O(N); full mixtures are exact O(N squared) with the
repository chunk policy. Do not compute the old correction before overwriting
it. Do not duplicate the filter, reset or score recursion.

Preserve broader ancestor support through the same interface. The marginal
setting still explicitly rejects unsupported non-Gaussian transitions,
structural/singular innovations, non-one-to-one ancestry and multiple annealing
stages. No approximate mixture, new ridge, clipping, altered probability model
or ancestry law enters this merge. Relaxing those guards needs its own derivation.

## 2. Isolate the feature and protect current work

1. Record heads, dirty paths, active jobs, worktree locations and source hashes.
   Preserve both dirty trees; do not indiscriminately stash, reset, clean,
   switch them, or stop another campaign.
2. Fetch origin and inspect main advancement. Create a clean integration branch
   such as `ledh-marginal-integration-20261006` in a new `/tmp` worktree from
   the intended main base. Resolve any local/remote main divergence there first.
3. Inventory required weighting hunks, dependencies, tests and docs from pinned
   commits. Initial candidates are `ledh_marginal_weights_tf.py`, policy and
   affine-map changes in `ledh_canonical_score_tf.py`, and forwarding in
   `ledh_canonical_batch_fused_tf.py`. Inspect imports and consumers recursively.
   Classify numerical-safety/small-matrix dependencies individually. Avoid whole
   file replacement that imports unreviewed adjacent behavior.
4. Extract/reimplement the smallest complete feature, retaining source provenance.
   Exclude unrelated HMC, NeuTra, structural-model and Zhao–Cui changes. Required
   dependencies must remain visible and tested; a smaller incomplete patch is
   not an improvement. Revise scope if extraction requires unrelated redesign.
5. Use explicit staging and reviewable commits: core correction; consumer and
   test wiring; documentation/results. Keep the integration tree clean.

Deliverable: dependency inventory and bounded final diff against current main.

## 3. Executable integration checks

Wire the option through the batch consumer and through the actual nonlinear
master path:
`run_ledh_nonlinear_master.py` -> `sqmc_campaign_tf.value_and_score/_kernel` ->
`canonical_value_and_analytical_score`. Forward it through the trace consumer
`sqmc_nonlinear_tf.trace_kernel` too. Include policy identity in kernel cache
keys, job IDs, manifests and resume checks. Extend the existing master runner;
never implement a separate filter inside the canary.

Before long runs, require:

- Primitive single-component reduction, identity flow, identical components,
  nonuniform/inactive weights, extreme log densities and invalid-covariance
  checks. N=1 is only a density-primitive fixture, not a full reset fixture.
- Actual affine-map/flow agreement; analytical derivatives for every mixture
  input against independent FP64 differences, including weight and covariance
  terms. Unsupported cases fail explicitly before execution.
- Ancestor parity against the unmodified integration base with identical inputs
  and existing tolerances fixed before observing results. Preserve broader
  transition and ancestry support. Investigate tolerance failures; do not relax
  bounds merely to pass a refactor.
- Marginal parity against the pinned implementation in previously validated
  LGSSM/KSC/M13 scopes; applicable canonical, batch, moment/reset and driver
  regressions on the merged implementation.
- Actual PP/SIR endpoints: T=1 and T=2, FP64/XLA, N=16 for PP and N=72 for SIR.
  Check every parameter direction with frozen inputs and central differences
  h_k=10^-4 max(1,abs(theta_k)) and h_k/2. Require finite values, consistent
  active branches, and |s_k-FD_k|/(1+|s_k|+|FD_k|)<=10^-5 at both steps.
  Failure triggers localization, not a post-hoc tolerance change. This bound
  concerns differentiation, not agreement with the model score.
- Two-row batch versus two scalar executions, trace versus untraced parity,
  and a wiring test proving the same cached consumer honors a policy switch.
- Chunk boundary density check N=4096,K=2048. Full canaries N<=3000 use K=N;
  no tiny alternative chunks to conceal resource failures.

CPU-only tiny checks hide GPUs before imports. GPU checks use trusted access
and verified memory growth. The executed score stays analytical; differences
are diagnostic only.

## 4. PP/SIR canary: question and evidence contract

Question: does marginal weighting execute correctly on both real consumers,
and what happens to likelihood/score accuracy and cost at matched inputs?
Expected failures include SIR reset/FP32 instability, inadequate particle
coverage, quadratic mixture cost and a correct derivative of an inaccurate
likelihood. These are distinct hypotheses to diagnose.

| Role | Predeclared interpretation |
|---|---|
| Engineering merge criterion | Correct shared call chain; reduction, regression, derivative, batch and validity checks; explicit applicability |
| Canary result | Actual likelihoods and every score coordinate, valid/attempted counts, reference errors/uncertainty and cost for every attempted scope |
| Promotion veto | Invalid results, derivative mismatch, inadequate reference, adverse precision or heuristic underperformance blocks scientific/default promotion in that scope |
| Continuation veto | Wrong target/data, corrupt evidence, missing required diagnostics, exhausted budget or unavailable trustworthy resources; stop affected stage |
| Repair trigger | New regression, wiring defect or localized harness/resource failure; repair with fresh attempt directory inside the same budget |
| Explanatory only | ESS, weight concentration, reset margins, quantiles, timing and memory; none replaces value/score accuracy |

An optional mathematically and mechanically correct method can be integrated
with explicitly documented unfavorable accuracy results and fail-closed scope
limits. An unresolved implementation defect blocks merge. Shared old FP32
failure blocks claims there but does not cancel independent FP64 diagnostics.
Do not drop invalid designs from paired summaries or conflate a failed candidate
with rejection of the research direction.

### Fixed comparison matrix

| Choice | Proposed setting and rationale |
|---|---|
| Models | PP d=2/p=6; Austria SIR d=18/p=3; existing canonical adapters |
| Target/timing | Existing NonlinearSQMCSpec initial law; transition before each observation; verify data identity |
| Parameters | PP (.6,114,25,.3,.5,.5); SIR (0,0,0) in its log-scale coordinates; existing nominal points, not tuned optima |
| Horizon/count | Small checks above; T=20,N=144 resource/localization pilot; main T=20,N=1008,K=1008 |
| Data | Archived seed 260401 and fresh seed 26100611 per model; verify freshness/hash and save exact arrays |
| Paired designs | Eight untouched seeds 261006101–261006108; same initial/process normals, observations and reset design for both settings |
| Ancestry/annealing | iid_dual_cap -> existing_one_to_one, one annealing stage; other SQMC ancestry settings remain unsupported |
| Moment repair | Existing guarded_pairwise arm and normal-quantile residual design, identical across weighting settings |
| Precision | FP64/XLA first; GPU FP32/TF32/XLA next; for precision tie-out promote exactly the same rounded input arrays to FP64 |
| Inference | Eight-design conditional canary on two datasets; exploratory uncertainty, no superiority/population/default conclusion |

The inherited moment/reset controls are an **untuned diagnostic warm start**.
Record exact values, provenance and prior safety rationale before launch. Small
first-step and N=144 diagnostics expose covariance, domain, cap and memory
failures. Do not tune on canary outputs. New protections/control changes need
an explicit non-harm comparison and fresh evaluation partitions. Admission
would additionally require scope-specific offline tuning and the repository
issued matching artifact. Cross-model reuse cannot establish a tuned default.

Construct three simple adversaries: bootstrap PF at N=1008 (unflowed baseline);
ancestor LEDH with the same moment controls (isolates weighting); ancestor LEDH
with covariance-only correction (tests whether high-moment fitting contributes
to failure). Report them separately by model, dataset and eligible precision,
including shared failures. Do not select controls to beat these checks.

Use `bayesfilter/testing/nonlinear_bootstrap_fisher_reference_tf.py` on identical
data, timing, parameters and initial law. Start with four independent replications
at N=32768 and N=131072. The reference allocation may fund one larger rung only
to resolve uncertainty. Preserve per-replication log likelihoods, Fisher scores,
ESS and MCSE. Reuse existing post-August references only after checking actual
files and target hashes. Missing files need verified recovery or new computation,
not reconstruction from rounded prose. Fisher and finite-filter derivatives are
different finite estimators; their agreement is an accuracy question.

Zhao–Cui quadratic scores may provide another comparison where targets match,
but unresolved support/particle bias and one SIR rank-40 proposal prevent treating
them as exact oracles. This merge changes no Zhao–Cui method or author source.

Report actual ell, all score coordinates, signed/absolute reference errors,
paired changes, reference MCSE, and every invalid trial. A dimensionless summary
uses fixed c_k=max(1,abs(theta_k)) and
sqrt(mean((c_k*(s_k-s_ref,k))^2)); retain coordinate tables to reveal mixed scales.
Use 10,000 paired-design bootstrap resamples, reporting seed 261006901,
explicitly exploratory with eight designs. Resample independent reference
replications separately for reference-noise sensitivity. MCSE/intervals do not
measure common reference bias. Unresolved reference refinement or conflicting
scores makes accuracy inconclusive.

Separate compilation from warmed timing and report peak memory. Same-N tests
isolate the weighting change; equal-cost efficiency and HMC suitability remain
later questions.

### Material default and assumption audit

| Choice / provenance | Failure risk | Earliest diagnostic / status |
|---|---|---|
| Gaussian scope from actual PP/SIR adapters | Different imported adapter or state transformation changes the density | Endpoint wiring and covariance/support check; reviewed mathematical scope |
| One-to-one ancestry, one stage from implemented method | A different allocator makes the outer-weight identity inapplicable | Preflight rejection and manifest identity; binding scope |
| Existing guarded_pairwise controls and caps | Untuned reset or protection obscures weighting effects | Verify prior safety rationale, first-step trace, covariance-only comparator; warm start only |
| N1008/T20 from recent nonlinear campaign | Existing count/horizon is inaccurate or too costly | N144 timing pilot, N1008 first complete trial and reference discrepancy; diagnostic scope |
| Two data seeds/eight designs, bounded-budget choice | Weak uncertainty, reuse or hidden selection | Check freshness, retain all attempts and condition by dataset; exploratory only |
| Two FD steps/10^-5 normalized bound, diagnostic choice | Truncation, cancellation or branch switches masquerade as bad differentiation | Compare both steps and branch diagnostics; no model-accuracy claim |
| Bootstrap/Fisher reference from inspected independent implementation | Resampling noise, particle bias or wrong observation timing | Target/hash tie-out, particle ladder and replicated MCSE; approximate reference |
| GPU FP32/TF32/XLA, repository target | Precision/compilation instability | Exact-input FP64 comparison, repeat fixed executable and record recompile separately; default execution target, not accuracy evidence |
| Rounded input replay and physical score coordinates | Different target or units create a false improvement | Hash exact arrays/parameters; show coordinate errors and apply stated chain rules; comparison requirement |
| Six-hour allocation / worker timeouts | Partial coverage or reference uncertainty remains | Timing pilot and running aggregate budget; convenience limit, incomplete is reported |

A missing protection rationale is an unresolved execution prerequisite; do not
invent one from successful canary results. No numerics-altering safety setting
is promoted or rejected on likelihood error alone.

### Budget, commands and artifacts

Cap: 21,600 aggregate job-seconds from the additional 172,800 seconds: pilots,
precision and derivative diagnostics 3,600; paired filters/heuristics 9,000;
references 7,200; repair reserve 1,800. Count compilation, failures and retries.
Use at most one GPU worker and two CPU threads per reference worker; default
worker timeout 1,200 seconds. Check occupancy and do not interrupt other jobs.
Do not start work beyond remaining budget. Unfinished planned cells remain
incomplete; no silent contraction followed by a claim the full canary passed.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow/TFP,
stable signatures, XLA. GPU commands use trusted access,
TF_FORCE_GPU_ALLOW_GROWTH=true and verified repository memory policy. Explicit
CPU exceptions set CUDA_VISIBLE_DEVICES=-1 before imports.

Add `--importance-weight-policies ancestor marginal_mixture` to the existing
master (planned option, not implemented). Before execution, prepare validated
dataset and parameter JSON files per model/data cell under the versioned root.
Use the archived parameter values exactly, including any recorded rounding;
nominal numbers in the matrix are labels, not permission to change the target.
The full matrix uses separate invocations so each can replay its own file.
Representative PP/archive cell after wiring and input preparation:

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  docs/benchmarks/run_ledh_nonlinear_master.py run \
  --models predator_prey --horizons 20 --particles 1008 \
  --data-seeds 260401 \
  --dataset-file docs/plans/artifacts/ledh-marginal-merge-20261006-01/inputs/predator_prey-260401.json \
  --theta-json docs/plans/artifacts/ledh-marginal-merge-20261006-01/inputs/predator_prey-260401-theta.json \
  --random-input-dtype float32 \
  --design-seeds 261006101 261006102 261006103 261006104 261006105 261006106 261006107 261006108 \
  --routes iid_dual_cap --arms guarded_pairwise \
  --importance-weight-policies ancestor marginal_mixture \
  --dtype float64 --device gpu --trace --worker-seconds 1200 \
  --budget-seconds 9000 \
  --output docs/plans/artifacts/ledh-marginal-merge-20261006-01/pp-260401-fp64-01
```

Before launch, the manifest must bind actual data files, frozen controls,
source commit, all argv and the **shared** FP64/FP32/heuristic budget. The 9,000
seconds above is a stage ceiling, not a new allocation per invocation. Freeze
full argv for `run_nonlinear_bootstrap_reference.py`, including exact saved
worker datasets, independent seeds and reference rungs, before execution. Parent
and child controllers must not each independently consume the entire allocation.

Versioned root: `docs/plans/artifacts/ledh-marginal-merge-20261006-01/`, increment
if it exists. Store dependency inventory, baseline outputs, manifests (commit,
command, environment, CPU/GPU, data, seeds, wall time, plan/result paths), logs,
hashes, per-attempt budget, raw JSON/CSV and continuous comparisons. Execution
will create `docs/benchmarks/ledh-marginal-merge-and-canary-2026-10-06.md` with
separate engineering, numerical and scientific conclusions, decision and
inference-status tables, and strongest-alternative-explanation review.

## 5. Documentation, final review and Git sequence

1. Review the final bounded diff and executed call chain. Check for duplicate
   filter/score code, dropped tangents, silent fallback, unsupported default
   change and unrelated imports. Review accuracy evidence separately.
2. Update the LaTeX algorithm and actual PP/SIR results: common formula, single
   incoming weight vector, outer factor, total derivative, applicability, cost
   and unresolved failures. Preserve pairwise skew/kurtosis repair and guards.
   Build the monograph, fix new references/citations and inspect changed pages.
3. Commit feature/tests and docs/results; require a clean integration branch.
   Fetch again. If main advanced, integrate the new main and resolve overlapping
   code by the mathematical contracts. Repeat affected checks and the document
   build; never choose ours/theirs blindly for scientific conflicts.
4. Merge the reviewed integration branch in the verified clean main worktree.
   Push normally to origin/main, never force. If remote advancement rejects the
   push, integrate it and rerun affected checks. Fetch and verify local main and
   origin/main identify the tested result.
5. Bring the integration branch to that same main head. Back-merge main into
   working branches only after their live dirty work is safely committed by the
   responsible execution step. Preserve unrelated research. Those branches can
   remain ahead: scoped feature synchronization does not imply identical whole
   branch heads. Earlier whole-project synchronization requires separately
   reviewing/integrating their other work.

Record tested/published commits and final clean status. Repair or revert a
post-merge problem with an ordinary commit, never rewritten remote history.
This turn proposes the sequence; it does not execute it or message another agent.

## Skeptical plan review

“Merge f5e69d716 and run two models” fails scope and baseline review: the commit
bundles other work; the current master does not forward the setting; old SIR
FP32 failures confound interpretation; nonlinear references are approximate.
This plan repairs those flaws with dependency extraction, executable consumer
checks, matched FP64/FP32 runs and explicit reference uncertainty.

Other risks checked: same-program differences are not an oracle; moment-repair
success does not test marginal weights; single-component reduction does not
justify arbitrary masks; inherited controls are not tuned; and same N is not
equal cost. Default/HMC promotion is outside this diagnostic integration.

Self-review conclusion: suitable for bounded diagnostic execution once Stage 2
confirms a complete extractable dependency set and the prelaunch data/control/
safety records are filled. These are concrete execution work items, not approval
tokens. No independent agent reviewed this plan. Remaining uncertainty is whether
SIR has a useful stable scope and whether accuracy gains justify mixture cost;
the canary measures those questions rather than assuming favorable answers.
