# Surrogate-HMC optimization: implementation and execution plan

## Active checkpoint

2026-10-21: owner requested implementation, review, adequate equivalence tests,
and execution of the four optimization proposals. Baseline is d189c4295 on
ledh-graceful-failure-execution-20260917. The checkout was clean. This plan
continues the SIR NeuTra integration plan and extends the finished numerical
repair with range-bearing and OT optimizations. It does not reopen the finished
Git integration. The preserved baseline has verified source hashes. The bounded
plan is complete: all four proposals were implemented and tested. The triangular
OT cache was rejected for slowdown; its source and evidence are preserved in
`rejected-triangular-cache/`. The original dense cache is restored, retaining
the invariant feature hoist. Final N4032 comparisons, paired timing, saved-tensor
review and the source-frozen SIR integration pilot are complete. Every saved
tensor for LGSSM, KSC SV, predator-prey and SIR is bitwise identical to baseline;
range bearing has the recorded small FP32 rounding differences. CPU diagnostics
cover 130 distinct tests, all passing in their latest execution, including
complete sharded updates/resume/rollback, independent mixture and polar
derivatives, correction on/off, matched SIR/pullback checks, mixed precision and
stale-map rejection. The final real-SIR pilot verifies two whole/sharded updates,
exact checkpoint continuation and a finite frozen-map HMC mechanics transition.
It does not establish trained-map quality or posterior convergence.

The six-hour ledger records 32 attempts: 16 complete, five failed and 11
superseded. Accounted worker time is 2,758.012 seconds (45.97 minutes), leaving
18,841.988 seconds; no jobs remain pending. Terminal interpretation and source
verification are recorded in
[the result note](surrogate-hmc-optimization-results-2026-10-21.md).
The next research phase is the parent target-specific SIR NeuTra training/search
and adaptive HMC assessment. It is outside this bounded implementation and
equivalence plan and was not launched. The five-model posterior objective
therefore remains incomplete, although this plan is finished.

## Research intent and evidence contract

The goal remains testing surrogate HMC: LEDH supplies proposal forces and the
declared independent endpoint supplies MH correction. The engineering question
here is whether we can reduce cost and memory while preserving the finite
LEDH likelihood, its complete analytical score, and the canonical NeuTra
training update. A trained map is a candidate for improved SIR mixing; training
loss or kernel speed cannot establish that improvement.

The baseline is the committed implementation above with identical observations,
parameters, random designs, N=4032, selected controls and priors. Ordinary FP32,
TF32 disabled, GPU/XLA remain requested. Historical mixed-precision SIR tuning
is not silently relabeled as a current uniform-FP32 selection. Any new scope
is explicitly identified and requires its own recorded check/tuning before
claim-bearing sampling. T50 applies to LGSSM, KSC SV, predator-prey and SIR;
range bearing uses the saved T20 fixture.

Promotion criteria for an algebra optimization are independent mathematical
equivalence tests, complete filter value/all-score and status comparisons,
and useful measured time or memory. For sharded training, compare the complete
objective, every gradient, updated weight, Adam slot and iteration with one
unsharded update at identical noise and starting state. No update between shards
is allowed. Apply clipping once to the mean gradient. The standard reverse-KL
objective has no added regularizer; any future regularizer must be applied once.

Nonfinite healthy outputs, invalid row contamination, omitted derivative terms,
wrong prior/endpoint/chart, checkpoint drift, scalar training fallback and
changed optimizer semantics veto a candidate and trigger repair. An OOM or
slowdown rejects that execution configuration, not the research direction.
Finite energy error and fixed percentage differences are not stopping criteria.
FP32 differences are reported against paired designs and the adapted Zhao-Cui
reference with its uncertainty when available; a tight FP64 algebra check is
an implementation test, not a replacement scientific likelihood tolerance.
Do not infer numerical equivalence from repeated calls on one random design.

Timing, allocator peaks, target failures, heldout training loss, map conditioning,
and gradient norms explain behavior. Passing them does not prove posterior
accuracy, convergence, source-faithful exact filtering or statistically superior
sampling. End-to-end HMC improvement requires subsequent cumulative sampling
against the same declared endpoint. No new long posterior campaign is hidden in
this bounded implementation run.

## Implementation sequence

1. **Matched SIR adapter and frozen-map connection.** Reuse
   `NeuTraTransport`, `NeuTraTransportTrainer`, the scientific prior/chart and
   `make_batched_score_filter`. Emit auxiliary value, corresponding analytical
   score, row status and completed horizon together. Bind the exact target,
   data/design, precision and selected controls to checkpoint/map provenance.
   Preserve the independent inference endpoint separately. Check the actual
   trainer consumer and nonlinear map pullback, including the log-Jacobian term.
   Reuse the canonical author IAF; do not substitute a historical trainer.
2. **Complete-update sharding.** Add optional native shard accumulation to the
   canonical trainer. Keep weights frozen throughout the logical batch; exclude
   padding from means and accumulate before clipping/Adam. Reject a whole invalid
   logical batch without changing weights or optimizer state. Preserve exact
   resume semantics including the external stateless-noise counter. Test shard
   sizes, a nondivisible final shard, invalid final rows and two resumed updates.
3. **Range-bearing algebra.** Share polar intermediates across analytical
   parameter directions. Replace nonshared Gaussian pair/state/parameter
   intermediates with exact weighted-moment contractions. Retain an independent
   direct implementation in diagnostic tests. Derive all covariance, location
   and normalization terms; test parameter-dependent covariances and common
   translations, not just fixed covariance or zero derivatives.
4. **OT memory traffic.** Evaluate symmetric primal cost storage and reuse of
   invariant packed features. Preserve finite Sinkhorn iteration order, epsilon,
   exact K2016 tiles, normalizers, barycentre derivatives and Contract-E reset.
   Never resurrect the rejected full derivative cache. A full streaming or
   custom fused kernel is only justified if this bounded exact candidate fails
   and profiling demonstrates the need; do not introduce a second reduced lane.

## Equivalence and execution ladder

* CPU diagnostic tests explicitly hide CUDA. Float64 independent dense formulas,
  finite differences over a step-size ladder, and existing directional callbacks
  check likelihood/score and all parameter directions on tiny fixtures. Test
  all five model families, shared/nonshared covariance, multiple rows, permutations,
  common translations, valid and invalid rows, correction on/off, and multistep
  recursions. Test XLA signatures and absence of graph expansion with batch size.
* Canonical IAF tests compare unsharded and sharded objective/gradient/Adam results
  with frozen noise, unequal final-shard sizes, clipped and unclipped gradients,
  invalid batches, checkpoint restore and nonlinear endpoint/force pullbacks.
  Tiny CPU optimizer tests are explicit diagnostic exceptions to GPU training.
* GPU preflight uses trusted access, configured/verified memory growth and TF32
  off before initialization. Save device/environment/source hashes. Use isolated
  processes for full-size baselines/candidates and synchronize result tensors.
  Record compilation separately from warmed calls and allocator peak separately
  from process reservation. Refresh the next-stage decision after each phase.
* Full N4032 fixed-design comparisons cover the five models and additional
  perturbed parameter rows. Start with one paired design; if a difference could
  change the decision, use up to four paired designs and matched reference
  uncertainty. No universal 1% promotion threshold. Arithmetic invariants remain
  strict; unexplained finite discrepancies require localization rather than a
  claim of scientific deterioration.
* Execute real SIR complete updates through the adapter, including resume and
  bounded heldout monitoring. This is an integration pilot, not a trained-map
  quality claim. Measure complete-update costs before the parent plan's six-arm
  architecture/LR search. A budget insufficient for that protocol is reported
  as under-budgeted for map promotion; an inherited recipe is not substituted.

The cheap adversaries are the current direct Gaussian calculation (correctness),
current dense cost cache (time/memory), unsharded canonical optimizer (update
semantics), and identity/affine map (transformed target mechanics). Evaluate
conditionally on model, covariance structure, batch/shard size and invalid-row
case. Exact Kalman toy checks and existing reference tests remain certifying
where applicable. A more elaborate candidate losing to the current calculation
is not promoted for speed. No learned-map ranking is intended in the pilot.

## Defaults, budget and stop conditions

N4032, FP32, TF32 off and current model horizons are owner choices; tests must
expose memory/cancellation risk. Exact-divisor K2016 is repository policy.
S4 is a memory-conscious starting hypothesis supported by the prior capacity
run, not an optimal size; compare S4/S8 when complete-update costs permit.
Canonical author IAF with three stages and explicit cap c=2 preserves the
architecture policy. A small width from the parent search is a pilot hypothesis,
not a tuned scientific default. Adam standard reverse-KL and its explicit
settings come from the existing audited trainer; use fixed noise for equivalence.

Bound local execution to six GPU hours, including failures/retries: up to three
hours for full-size numerical/performance comparisons, two hours for complete
updates and monitoring, one hour reserve. CPU diagnostics are bounded by per-job
timeouts and recorded separately. Each substantive candidate gets at most two
localized repair/retest attempts before recording a remaining issue. Do not
consume unused hours merely to exhaust the allocation. Stop a worker at its
deadline, preserve partial results, and advance independent checks. Budget
exhaustion limits evidence, not authority to change the target or skip tests.

Versioned output root:
`docs/plans/artifacts/surrogate-hmc-optimization-20261021-01/`.
Supervisor/worker command will be
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_optimization.py`;
its manifest records exact resolved argv, environment, source Git/hash identity,
seeds, devices, data hashes, timing, plan and result paths. Run this exact script
form for GPU work with escalated permissions, avoiding per-command approval churn.
No package changes, external messages, publication or automatic Git push occur.

## Skeptical review before execution

Reviewed against wrong baselines, proxy promotion, unfair timing, stale scope,
missing limits, environment differences and call-chain gaps. Two initial flaws
were corrected: filter sharding does not prove complete optimizer equivalence;
SIR capacity does not establish range-bearing capacity. Old mixed-precision
selection and current FP32 scope must remain distinguishable. Existing GPU
profiles contain no device events, so bottleneck fractions and promised speedups
are unsupported. Independent dense/derivative checks, full update comparison,
preserved baseline sources, per-model checks and bounded pilot answer the stated
questions. The plan passes this review for implementation and bounded execution.

Before terminal reporting, record a decision table, inference-status table,
strongest alternative explanation, unresolved evidence and next justified step.
Scientific ranking of maps and posterior improvement remain untested until the
parent training and adaptive inference phases have adequate evidence and budget.

## Algebra and final pre-launch audit

For Gaussian component j, let P_j=C_j^{-1}, r_j=x-m_j, and a_j be its
normalized mixture responsibility. Its complete differential is

    d log phi_j = -(dx-dm_j)^T P_j r_j
                  + 1/2 r_j^T P_j dC_j P_j r_j
                  - 1/2 tr(P_j dC_j).

Thus d log sum_j phi_j = sum_j a_j d log phi_j. Set H_j=P_j dC_j P_j
and expand r_j=x-m_j. Responsibilities multiply the component features
P_j, P_j m_j, P_j dm_j, m_j^T P_j dm_j, H_j, H_j m_j and
m_j^T H_j m_j-tr(P_j dC_j). Their weighted sums reproduce every term of
the differential while avoiding a P-by-B-by-N-by-N score tensor. Centering x
and m by the same particle, and their tangents by the same tangent, preserves
r_j and dr_j exactly in real arithmetic. Floating-point reassociation is
explicitly tested against the old program and independent dense FP64 formulas.

For OT, squared pair distances satisfy C_ij=C_ji^T; storing the upper block
triangle and reading transposes preserves the finite iteration count, order of
each reduction, and analytical cost derivative. At N4032/K2016 three primal
blocks replace four; this is a 25% reduction in cost-cache storage, not a
claim about total memory. The full-size run must exercise off-diagonal reads.

Runner review: baseline and candidate are separate processes with identical
data/design/controls and three warmed measurements. The range-bearing controls
come from the later flow16 preparation, not the superseded flow8 selection.
Training uses B8/S4 and an explicitly narrow affine scale .02 as a healthy
integration fixture, with standard-normal latent noise, author IAF and LR .001.
These are convenience choices for wiring, not the target-specific training
protocol. The whole/shard all-gradient diagnostic threshold .002 on
max(1,abs(component)) only triggers localization; it is not statistical
materiality or map promotion. Checkpoint continuation must be bitwise equal.
The pilot checks a real frozen-map SIR transition after both updates. It does
not replace the 1,000-point post-training diagnostic or authorize posterior use.

Trusted preflight sees two GPUs: RTX5080 (GPU0, about 5GB reserved and 8% use)
and RTX4080SUPER (GPU1, 86% use by an external workload). Bind this campaign to
GPU0 by its UUID, preserving memory growth and recording concurrent device use.
Both arms run on that same GPU. Earlier 4080 timings are not a fair comparator;
new timings remain descriptive because the machine is shared. The expanded
HMC adapter suite also passed (46 tests, including two checks counted above).

The first range-bearing candidate reduced peak allocation but ran more slowly.
Later models showed much larger time variation while independent CPU tests
were running. Before attributing those differences to code, use a focused
paired timing run in one process after those tests finish, alternate baseline
and candidate order, and retain both compiled graphs. This diagnoses cost, not
memory or scientific quality. It uses identical full-sized inputs and stays
within the existing budget. A genuine slowdown blocks speed promotion; memory
reduction remains a separate measured benefit. Do not use old, concurrently
measured timing as proof of a speedup or slowdown.

The alternating measurements confirmed the triangular-cache slowdown for LGSSM,
KSC and predator-prey (all six candidate calls slower than all six baseline
calls). Reject that storage layout as the default. After the remaining timing
workers finish, preserve its source and restore the original dense cache/read
layout while retaining the invariant feature hoist. Repeat the five full-sized
comparisons, paired timing and SIR integration on this final candidate. This is
the planned cost-regression repair, within the original six-hour budget. It
also tests whether SIR's intermediate sensitivity difference came from XLA's
changed reduction/layout choices: no nonshared-mixture or polar change is used
by SIR. If this isolated repair does not recover equivalence/cost, retain the
original OT implementation and report the triangular experiment as rejected.

The second integration attempt correctly rejected a stale map after source and
scope-identity checks changed while that worker was running. It is a harness
coordination failure, not a numerical training failure. Freeze numerical sources
before the final retry; include observation and chart-center identity rejection
in the final CPU checks. Do not weaken the check to reuse the stale map.
