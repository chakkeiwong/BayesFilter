# Five-model surrogate-HMC shared-cost repair master, 2026-10-11

Closeout update, 2026-10-21: the computational repair and native batching
campaign are complete. Four models completed their sampling rule; SIR ended
warm-up without retained samples. The fixed 1% numerical blocker is superseded
by the owner reference/uncertainty policy. See the
[final master summary](surrogate-hmc-master-final-summary-2026-10-21.md) for the
completed work and pending SIR NeuTra integration. The dated refreshes below
preserve their historical settings and budgets; they are not live run state.

Execution refresh, 2026-10-15: the executable master now supports the accepted
**epsilon=1/five-iteration OT continuation**, with fresh per-model mixture
calibration and automatic phase-local/campaign-level next-phase refresh.
The completed SIR reference comparison is settled and is not a new stopping gate.
See [current continuation plan](surrogate-hmc-accepted-ot-continuation-2026-10-15.md).

Remaining authorized compute before this phase: 51,669.679990 seconds (14.352689 h).
The 86,400-second campaign ceiling is unchanged; attempt cap is 140 to accommodate
fifteen planned workers and bounded repairs. N4032/K2016, FP32 default, TF32 off,
GPU/XLA and the existing SIR mixed-precision exception are retained. Latest saved
controls seed fresh tuning at offset 100; no old selection is current-source
admission. This continuation leads directly to actual HMC mechanics/cost checks.

The goal remains testing surrogate HMC. Full posterior inference is unfinished:
the normal prior in q remains a mechanics fixture pending the target decision.
That decision does not block this computational phase. Live execution state:
[phase plan](artifacts/surrogate-hmc-cost-repair-24h-20261013-01/next-phase-plan.json).

## Preserved earlier execution record

The following dated records describe their original sources and controls.
Their completion statements do not apply to the new OT continuation.

Execution refresh, 2026-10-14: **the corrected-source computation checks are
complete for all five models; the full surrogate-HMC campaign is not complete.**
The additional 24-hour allocation used 30,116.162 worker-wall seconds (8.3656 h)
across 99 attempts, including compilation and failed candidates. 15.6344 h remain.
No GPU worker is running. Prior evidence and the earlier 10-hour ledger are preserved.

The analytical all-parameter recursion, compact covariance contractions, and
retained-force repairs are implemented. The shared ancestor selector now uses
binary64 arithmetic before discrete inverse-CDF selection, preventing FP32
rounding across integer boundaries. Because this changes the old FP32 finite
program, all five filter scopes were freshly tuned with disjoint calibration and
validation partitions at seed offset 60, then checked in actual HMC consumers.

N4032/K2016, TF32 off and XLA are unchanged. LGSSM, KSC SV, predator-prey and range
bearing use uniform FP32. SIR uses the optional FP64 recursion with FP32 pair
kernels: uniform FP32 failed its 1% score arithmetic screen. Under fresh selected
controls, all five likelihood, score and actual-force comparisons pass the 1%
scaled tolerance against the same finite computation in FP64. Actual-binding
reconstruction errors are zero. This FP64 comparison is not a model oracle.

The terminal result is
[fixed-design-computation-result-01/result.md](artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-computation-result-01/result.md).
Its JSON links exact scopes, source hashes, settings, commands and worker evidence.
All 32 recorded source dependencies match the tested source and archived copies.
Paired consumer checks and two-transition-per-chain mechanics pass all five models.
Observed warmed four-chain force costs are 25.94 s (LGSSM), 9.37 s (KSC), 16.26 s
(predator-prey), 26.72 s (SIR), and 6.38 s (range bearing). Runtime differences from
few repeats are descriptive; long-chain practicality remains unresolved.

The executable master now includes a host-only finalize command. It refuses
completion for missing or failed model results, stale sources, mismatched scopes
or selections, relaxed arithmetic tolerances, failed health or active GPU work.
158 CPU-only tests and 3 subtests pass, including eleven new terminal guards.
Exact command and review: the extension plan and computation-terminal-review-plan.md.

The next posterior phase is part of this existing master. It has not started:
the saved independent-normal prior in standardized coordinates is explicitly a
mechanics fixture, and the final prior choice is unresolved. The owner has been
asked whether to retain those priors as benchmark posterior targets or use the
models' scientific priors. This is a target decision, not repeated compute approval.
After resolution, use the public tune_hmc_kernel authority with the repository's
typed position-force binding, retain adaptive warmup/cumulative sampling, and
apply the remaining budget. The low-level runner cannot issue tuning authority.
Do not substitute fixed L1 mechanics or precision checks for posterior inference.

The optional covariance-proposal force is not canonical Contract-E LEDH admission.
Rejected cache, reduction, tile-batch and transport-tangent candidates remain
recorded diagnostic failures; they were not promoted. A completed computation
screen does not resolve independent model-reference accuracy or posterior mixing.

## Research intent and evidence contract

Question: can the frozen covariance-guided defensive-mixture filter provide an
analytical proposal force for corrected HMC on LGSSM, KSC SV, predator-prey,
SIR (T=50), and the saved range-bearing case (T=20), with N=4032, ordinary
FP32, TF32 off, GPU/XLA and verified TensorFlow memory growth, at a tractable
cost? The candidate is a shared *implementation* of the existing finite
filter and its full parameter score. It does not change the MH endpoint, the
model, observations, particle design, transport chunk K=2016, or frozen
numerical controls. The exact/frozen endpoint, rather than the force, defines
the HMC target.

Comparator: the current per-parameter directional filter with identical
theta, observations, random design and controls, and the existing independent
endpoint for each model. For the practical-value claim, compare HMC with the
same-endpoint exact-gradient HMC where available and checked random-walk MH
otherwise; also construct prior-only and simple Gaussian proposal controls.
LGSSM Kalman likelihood/score is an exact reference for model accuracy. The
other reference limitations are in the existing five-model master.

Primary engineering pass: the new one-time-sweep force computes the derivative
of its own frozen finite likelihood program within declared precision, checked
against independent finite values using a converged width study when needed.
It preserves the baseline likelihood value and validity and has no new finite,
transport-mass, moment-restoration, or force-sign failures. Old/new analytical
score agreement is an implementation comparison, not an independent oracle;
a discrepancy triggers localization rather than declaring either route correct.
These checks must cover all five models, off-centre parameter points,
small and N4032 particle counts, and multiple rows/chains. The speed question
is answered only by separate compilation and warmed runtime, peak memory and
HLO/profile evidence. A statistically supported cost advantage requires
paired repeated timings, not a single ratio. A completed HMC chain needs the
existing sampler health, cumulative R-hat and posterior/reference checks;
parity and mechanics alone cannot promote inference or a new default.

Promotion vetoes: incorrect score component or chart sign, value/validity
mismatch beyond the declared tolerance, missing derivative through any finite
operation, XLA/pfor fallback, invalid endpoint or numerical health, overwritten
prior evidence, mismatched source/data scope, or missing own-scope tuning.
Continuation vetoes: corrupted input identity or artifacts, a mathematical
change to the MH target/finite filter, an exhausted total budget, or an
unresolved validity regression. Slower but correct candidates trigger
profiling/repair, not scientific rejection. Explanatory diagnostics are
acceptance, isolated ESS, individual HLO op counts, local timings and memory.
No claim follows about original-model posterior accuracy from a finite MH
endpoint or about canonical Contract-E LEDH admission from this optional
covariance-guided force.

## Implementation phases

1. Freeze the existing directional route as a paired diagnostic baseline. Build a
   vector-valued **analytical** sensitivity recursion carrying all p parameter
   columns through one observation-time loop. Share the primal particle,
   proposal and mixture computation; propagate derivatives of every finite
   normalization, transport update, covariance restoration and correction.
   No `tf.vectorized_map`, pfor, autodiff score, Python runtime particle loop,
   or p independent T-step filters. Keep the old route diagnostic-only.
2. Reuse transport distances and distance tangents within a Sinkhorn solve
   when measured HLO confirms repetition; respect K=2016, FP32 and the exact
   finite iteration schedule. Use tiled memory rather than a permanent dense
   N-by-N-by-p-by-d tensor. Preserve row/column normalization and derivatives.
3. Retain the mathematically required all-pairs marginal mixture. Share
   component Cholesky factors and primal quadratic forms across parameter
   columns; exploit structurally equal covariance/Jacobian only through an
   explicit general property with parity checks. Do not assume that SIR or a
   nonlinear model is linear merely for speed.
4. Evaluate batched chain forces against serial forces with identical inputs,
   first at small N and then N4032 subject to measured peak memory. Keep a
   serial memory-bounded fallback if batching is slower or exceeds capacity.
   A fallback is a documented execution limitation, not evidence of complete
   batching repair.
5. Separate compilation from warmed performance, persist per-phase telemetry,
   and update the shared master to refresh the next-phase plan after every
   phase. Compare frozen selected controls with cost-aware candidates on fresh
   calibration/validation only; do not retune using the untouched claim data.
   Run the five-model mechanics stage after parity and then the full HMC phase
   only after the prior and total GPU budget are fixed.

## Default and assumption audit

| Choice | Provenance/status | Failure mode and earliest check |
|---|---|---|
| N=4032, FP32, TF32 off, five saved scopes | Explicit owner request / fixed scope | Different numerical target or scope; manifest and data hashes before every run |
| p-axis forward sensitivities | Derivative recursion / implementation candidate | Missing cross-particle term or memory blowup; small finite-difference/parity tests, then GPU peak |
| Old directional route | Existing code / diagnostic baseline | Old route itself wrong; independent LG Kalman and finite-difference checks; parity is necessary but not sufficient |
| Existing flow/reset controls and beta | Own-scope preparation / frozen comparator | Poor cost-benefit; report unchanged baseline and use fresh data for new selection |
| K=2016 transport tiles | Mandatory chunk policy / reviewed default | Cache size or altered finite arithmetic; policy and parity tests |
| Batched four-chain force | Candidate, not default until measured | GPU memory/compile explosion; small batch parity and N4032 peak/warmed timing |
| Benchmark-normal-v1 prior | Prior mechanics fixture only; final prior choice pending owner | Wrong posterior claim; do not launch full inference until explicitly fixed |
| 2000/10000 warmup, cumulative R-hat <=1.01, 20000 retained max | Existing BayesFilter policy and owner instruction | False convergence or unaffordable run; health, ESS, reference agreement, budget stops |

## Skeptical audit, completed before execution

The current source-level direction loop is shared across all five cases; fixing
only LGSSM or SIR would be the wrong baseline. N-squared mixture evaluations
are intrinsic to the selected marginal-mixture law, so the plan preserves
them and attacks redundant factors/derivatives. XLA may already hoist static
transport tiles, so cache promotion requires optimized-HLO or paired timing;
otherwise a memory-heavy cache could make performance worse. A batched chain
can exhaust VRAM, and p-axis derivatives can multiply working memory even when
they eliminate repeated primal calculations. The existing four-chain pilot
included compilation in its runtime extrapolation; the new timing separates
first call from steady state. The current optional covariance-guided filter is
not Contract-E canonical LEDH; no result may silently claim that identity.
The old preparation budget is spent and the benchmark prior was mechanics-only,
so full inference waits for an explicit prior and campaign-wall decision. The
short parity/mechanics run cannot establish convergence. These are documented
risks with early checks, not reasons to suppress the repair.

## Execution, budget and artifacts

The repair-and-verification allocation is at most 3600 GPU-wall seconds and
up to thirteen model attempts (including localized compile repairs; revised below), plus focused CPU tests, under a fresh versioned root
`docs/plans/artifacts/surrogate-hmc-five-model-repair-20261011-01`. Keep
complete logs and a manifest (commit/source state, exact command, environment,
GPU and memory-growth status, TF32, seeds, wall time and artifact paths). A
failed attempt consumes this allocation; repaired retries get a new numbered
root. The separate full-HMC budget and prior are pending owner input. Stop
before consuming that budget without those answers. Use the absolute
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python` interpreter and direct
script arguments so the command matches the dedicated allow list. GPU device
checks and runs require escalated sandbox permissions. CPU-only checks set
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and label that choice.

The repair result must include a decision table and inference-status table,
paired value/score/validity and warmed timing per model, peak memory, remaining
weaknesses, and the next-phase plan. Refresh that plan after each executed
phase; an interrupted run must remain explicitly incomplete.

## Algebra and arithmetic checks

Paired FP64 fixture tolerance is `abs(error) <= 2e-7 + 2e-7*abs(reference)`;
paired FP32 tolerance is `1e-3 + 1e-3*abs(reference)` for both log likelihood
and every score component. These are implementation-equivalence screens,
stricter than the owner's 1% materiality allowance; they are not posterior
accuracy criteria. Near-zero scores are assessed with the absolute term.
Use finite differences as an independent local derivative check. Compare
multiple rows to independent scalar calls before wiring batched HMC.

The first N4032 attempt exposed an XLA dynamic-slice shape error before any
timing. Fixed-extent TensorFlow slices repair this infrastructure failure.
This retry consumes the same 3600-second allocation; the attempt limit is ten
to accommodate bounded compiler repairs across five models. Both methods
materialize the same small value/score/status outputs for warmed timings.

The first paired full-size LGSSM check preserved the value and score but
showed essentially equal warmed times (new 43.45--44.00 s, old 44.57--44.81 s).
That is a failed performance candidate, not a successful speed repair. The
next repair contracts squared-distance derivatives through each tile's primal
softmax with a packed matrix product. For `C_ij=-||x_i-x_j||^2/epsilon`,
`dC_ij=-2/epsilon (x_i.dx_i+x_j.dx_j-dx_i.x_j-x_i.dx_j)`; consequently its
weighted row derivative uses weighted means of `x`, `dx`, and `x.dx`. This
is the same finite Sinkhorn derivative without a P-by-N-by-N cached tensor
at each iteration. Validate in FP64 first, then paired FP32 GPU timing.

Offline filter-control nomination now minimizes the heldout importance-weight
second-moment proxy multiplied by warmed full-score seconds. This prevents
a negligible loss improvement automatically selecting twice as much work.
It is an engineering nomination hypothesis, not measured HMC ESS/second or
a statistically supported ordering. Each candidate must remain finite and
valid; beta fitting uses calibration only, the comparison uses disjoint
validation, and the claim dataset is unchanged and unused. Fresh tuning seeds
are 26101201--04. Compilation is recorded separately. The resulting source
change invalidates old selections; the consumer must not silently reuse them.

## Batch implementation review and GPU phase, 2026-10-12

The two-chain particle implementation passes all five N64/T2 FP64 fixtures
at two distinct off-centre positions, with moment correction enabled, against
the old directional filter and central finite differences of the finite value.
Evidence: artifacts/surrogate-hmc-five-model-unit-batch2-20261012-01.
The HMC consumer uses B=2 particle operations and pads an odd final group;
its force sign/chain rule and invalid-status wiring tests pass. Model callbacks
and small map assemblies remain fixed trace-time tuples because their public
protocol is single-position; no entire filter is mapped over chains or parameters.
This is a documented protocol limitation, not a claim that every callback is
batch-native. Two-chain GPU performance and capacity are still unmeasured.

The packed, unbatched LGSSM diagnostic preserves the FP32 value exactly and
scores within 2.4e-6, with warmed times 21.03--21.19 s versus 44.63--45.10 s.
These two repeats are descriptive, not a statistically supported speed ranking.
GPU wall consumed through this stage is about 526 s (three attempts). The next
five-model B=2 comparison is bounded to 2700 s total, at most 700 s per model,
under artifacts/surrogate-hmc-five-model-repair-batch2-20261012-01. This remains
within the 3600 s / ten-attempt repair allocation. The untouched full HMC
campaign is not launched by these diagnostics.

Skeptical review before this phase: preserve every finite operation and old
comparator control; include both chain rows and all score components, require
completed horizons and validity, and separate compile from warmed timing.
The SIR quadratic form may still allocate a large intermediate, so a timeout
or capacity failure is an implementation repair trigger. Source snapshots are
saved, not just hashes. A source or data mismatch vetoes artifact reuse.
The public tuning interface and bound position-only-force tuner are retained;
the force is a surrogate proposal field and the independent MH value fixes
the target. No new prior, target, canonical-LEDH claim, or convergence claim
is introduced by this implementation phase.

## SIR mismatch repair, 2026-10-12

The B2 SIR comparison returned finite T50 values, but score parity FAILED:
absolute component differences reached 9.91 while value differences were below
0.0004. Its timed repeat exceeded the 700s cap; this is not a passed run.
The packed transport derivative contains uncentred differences of large dot
products. Test translation invariance against direct pairwise differences,
then centre both states and sensitivities before packed contraction if the
fixture confirms cancellation. This is an algebraic implementation repair,
not damping or a changed filter target. Keep the direct pairwise primal cost.
The derivative identity is (xi-xj) dot (di-dj) =
(ui-uj) dot (vi-vj), with ui=xi-c and vi=di-e for any common c,e.
The heuristic comparator for this focused check is the direct dense contraction.
No timing or posterior promotion follows from this check. A failure of parity
is a repair trigger; skip expensive repeated timings of a known wrong score.
The controller must continue other independent model checks within its budget.
Skeptical audit: a tiny translation test does not establish SIR T50 correctness;
require full-size recheck before accepting the repair, preserve old evidence,
and retain FP32. The total GPU allocation remains 3600s, approximately 2652s used (about 526 earlier, 1633 in the four-model B2 run,
122 on range bearing, and 372 on the centered SIR recheck), leaving
about 948s and one of ten GPU attempts.

## Final bounded SIR discrimination, reviewed 2026-10-12

The centered packed transport passes a large-offset FP32 test against a
direct-difference FP64 reference, and all five N64/T2 FP64 filter fixtures pass.
It does **not** close the N4032/T50 SIR score gap. The centered full-size
recheck has value errors below 0.001 but score error as large as 20.95.
Per-observation errors grow from below 0.004 at t=4 to above 10 at t=9.
The old directional score for the same data, theta and controls changed by
21.94 in component zero between two compiled runs, while values differed
by less than 0.001. Thus old-vs-new score parity alone cannot identify which
FP32 derivative is correct. This is a continuation repair trigger, not a
posterior or research-direction veto.

Question: is the new analytical score's first SIR parameter component the
derivative of its own *finite FP32 value program* at full N4032/T50? The
independent comparator is central finite differences of fresh value
evaluations at theta0 +/- h e0 with common observations, particles, random
design, and selected baseline controls. Paired B=2 evaluates the two sides
together at h=0.005 and 0.001. Compare both total value differences and
per-observation increments with the independently propagated score. Values
come from the primal Jet route, which does not use the propagated tangent.
The selected controls are a historical performance comparator only because
their source scope predates this code; they cannot issue admission or HMC
evidence after source changes.

Interpretation: two stable finite-difference widths agreeing with the
analytical score would support one local component at one point. Width
dependence, invalidity, or disagreement leaves correctness unresolved and
blocks force promotion. This check does not establish all components,
off-centre correctness, or sampler convergence. No old directional score,
relative speed, or acceptance proxy may override a failed finite-value
derivative check. The direct dense transport large-offset test is the
constructed simple adversary; the paired directional code and finite value
are independent checks. The salient situation is the SIR high-dimensional
T50 filter, not the easier T2 unit fixture.

Skeptical audit: FP32 finite differences can suffer roundoff at small h and
curvature at large h; two widths and per-step traces expose this risk rather
than certifying a derivative from one scalar. A healthy bracket must complete
all 50 observations for both rows. The test intentionally keeps FP32, TF32
off, GPU/XLA, K2016, and source/data snapshots. It cannot prove posterior
accuracy and should not trigger full HMC. This final attempt is capped at
720s within the original 3600s GPU-wall allocation and ten-attempt limit.
An attempt timeout or budget exhaustion ends the current campaign without
claiming a full repair. After the result, refresh the checkpoint and
next-phase plan, including the independent KSC and range-bearing regressions.

## Full-size derivative diagnostic result and remaining local repair

The final allocated GPU attempt completed in 328.945 wall seconds. SIR
parameter-zero score was -50.6589; finite differences at h=.005 and .001
were -168.0420 and -173.3398. Both brackets completed 50 valid observations.
The differences between the two widths (5.30) are much smaller than either
disagreement with the score (117.38 and 122.68). This supports a material
score failure at this point; finite differences do not identify the exact
derivative or prove which internal operation is responsible. The largest
increment discrepancies are at observations 7 and 11--13. No HMC promotion.

Artifact: artifacts/surrogate-hmc-sir-finite-difference-20261012-01.
Approximately 2981.2 of 3600 GPU wall seconds and all ten attempts are spent.
The GPU phase stops at its declared attempt limit; this is not a claim that
the full repair is complete. CPU source review and focused regression continue.

The remaining output contraction of the transport still computes dC from
uncentred particle tangents. For a common tangent c,
(x_i-x_j)^T[(dx_i+c)-(dx_j+c)] is unchanged, so subtracting one shared tangent
origin before either dot product preserves the exact derivative. This is the
same analytic cancellation already used in the balancing loop. First strengthen
the CPU-only primitive test with a large tangent offset and test the *varying*
part of the answer against the direct FP64 reference, avoiding a relative
tolerance proportional to the common offset. Then make the algebraically
equivalent change only if it repairs that test. This check cannot establish
full SIR accuracy; source changes require a new full-scope check later.
Skeptical audit: preserve primal values, use identical quantized inputs, allow
for FP32 output ULPs, and do not equate primitive parity with whole-filter
accuracy or speed.

## CPU-only covariance contraction repair (reviewed)

A second shared kernel forms `u_{ij}^T dC_j u_{ij}` with a
parameter-by-model-by-all-particle-pairs-by-state intermediate. At SIR d=18,
N=4032 this can amplify GPU memory traffic by the parameter count. Replace
only that *derivative* contraction; leave the primal log Gaussian density
unchanged. For `r_i=x_i-a`, `m_j=mu_j-a`, and
`H_j=P_j^T dC_j P_j`, where `P_j=C_j^{-1}`, the same quadratic is
`r_i^T H_j r_i - 2 r_i^T sym(H_j) m_j + m_j^T H_j m_j`.
The shared anchor `a=x_0` cancels algebraically. Contract `r_i r_i^T`
against `H_j` and `r_i` against `H_j m_j` with batched matrix
products, so no P-by-N-by-N-by-d buffer is required. The output still has
P-by-N-by-N entries because the mixture needs those pairs.

Evidence contract: compare against the existing direct contraction on
small d=1,3,18 and both shared/particle covariance modes; use a separate
FP64 dense Gaussian-mixture finite-difference reference for all P directions,
then run the full CPU regression. Pair the same frozen inputs; do not change
the primal or the independent MH endpoint. A discrepancy is a hard repair
veto. No GPU time or HMC promotion is inferred from this CPU test. A new
full-size N4032 FP32 GPU derivative and timing recheck remains required
before claiming either correctness or speed. This is a bounded local
infrastructure repair after the SIR derivative failure, not a new target,
tuning scope, or posterior decision.

Skeptical audit: expanding the quadratic could worsen FP32 cancellation,
so anchor both x and means; covariance is symmetric mathematically but
symmetrize H only inside the quadratic identity. Shared-covariance shapes
must broadcast correctly, and all parameters must be evaluated together.
A small fixture may miss full-size XLA memory/compilation failure; keep that
as an unresolved risk and explicit future veto. The cheap independent
reference is direct cholesky/logsumexp on perturbed values, not the same
factorized algebra. No secondary runtime proxy can promote this change.

## Local repair verification and longer CPU recursion check

The strengthened transport test first failed at the large common tangent.
Centering only dC did not close it. Centering both parts of the final tangent
contraction, then restoring the shared tangent once, passed. This uses
sum(T)=1 and sum(dT)=0 in the column-normalized mathematical transport;
primal values are unchanged. Floating column-normalization error remains
observable and full-scope derivative agreement is still required.

The revised covariance contraction passed 44 CPU-only tests, including
FP32 and FP64 XLA mixtures against independent FP64 finite differences
(d=1,3,18; shared and particle covariance) and the failed-row isolation test.
Artifact: artifacts/surrogate-hmc-cpu-covariance-after-20261012-01.
All five N64/T2/B2 FP64 end-to-end call chains also pass finite differences
and old-route parity after both repairs:
artifacts/surrogate-hmc-five-model-unit-final-cpu-20261012-01.
These are engineering tests, not a new full-size GPU result.

One additional CPU-only SIR N64/T10/B2/FP64 check (theta offset .08,
flow2/reset4, all parameter directions) is allocated at most 420 seconds.
Use the same master with --unit --unit-horizon 10 --model sir_d18. It asks
whether recursive derivative errors already arise in double precision at
a longer small-particle horizon. The independent comparator is finite values
of the same frozen program, and the old directional result is a secondary
check. Success uses the existing unit tolerances and healthy complete
brackets. A failure localizes a correctness issue; a pass does not establish
N4032, FP32 or T50 accuracy. No GPU attempts or posterior budget are consumed.
Skeptical audit: the small particle count and reduced flow/reset schedule
can hide the full-size pathology; preserve that limitation rather than
transferring admission. This is the next diagnostic within the repair plan.

The FD diagnostic now also checks the unperturbed run's health and records
the width spread separately from the score gap; a completed diagnostic is
never an automatic accuracy pass.

## SIR T10 derivative-width diagnostic, 2026-10-12

The CPU-only N64/T10/B2/FP64 check under the final kernel snapshot gave
analytical old/new parity to about 1e-10 in score and 1e-12 in value. Its
row-2 central finite-value difference at relative width 1e-5 missed the
strict unit tolerance: parameter-0 gap 0.081913 versus score -310.874061,
while row 1 passed. The program now distinguishes this from old/new score
parity failure. The scientific classification remains unresolved because
one finite-difference width can suffer truncation, floating cancellation,
or a change of finite branch.

Next bounded diagnostic: rerun the identical saved SIR data, theta offset,
N64, T10, B2 and FP64 with width ladder
{1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6}. Record each parameter's value
difference, the observation-wise finite-value derivative, analytical
increment, bracket validity and completion. The primary question is whether
the gap shrinks toward zero with width rather than stabilizing at a nonzero
value; no single selected width is a promotion target. Budget 180 CPU seconds,
one versioned output root, with timeout 160 seconds. This diagnostic neither
uses nor extends the GPU campaign budget. A stable nonzero gap is a score
correctness repair trigger; a width-dependent unstable gap requires a
higher-precision or complex-step/reference check before concluding the
analytical recurrence is wrong. Old/new parity alone cannot certify either
score. Pre-mortem: small N or a reduced flow schedule may hide the N4032/T50
FP32 pathology, so even a clean ladder will not clear the full-scope veto.


### Width-ladder review and next numerical check

The completed SIR T10 ladder shows approximately quadratic small-width
truncation: row2/component0 gaps are .76550, .081913, .0073395 at widths
3e-5, 1e-5, 3e-6. Old/new scores agree to 2.42e-10. This evidence does not
support calling the small-fixture analytical derivative wrong. Preserve
its fixed-width FAIL_FINITE_DIFFERENCE artifact. A fresh CPU check will use
widths 1e-5, 3e-6, 1e-6 and the two adjacent Richardson estimates
D*=(r^2 D(h/r)-D(h))/(r^2-1). Admission of this local derivative check
requires both estimates agree with each other AND the analytical score
within the unchanged 2e-5+2e-5|D*| tolerance, healthy complete brackets,
and analytical implementation parity. This is a convergence check, not a
selection of the most favorable width. A failure remains unresolved; no
post-hoc tolerance increase is allowed. Maximum additional CPU wall120s.
The broader six-width artifact is preserved; the next three-width run
uses a fresh root. Skeptical audit: nearby non-smoothness or cancellation
can invalidate extrapolation; agreement across the two disjoint scales
and explicit health checks are required but still only local evidence.

The same-scope final GPU repair check remains inside the original 3600s
wall cap. The ten-attempt convenience cap is revised to twelve, solely to
check the transport-centering and covariance contraction fixes after the
small CPU tests. Approximately 619s remain; each launch must subtract its
actual wall time before another is considered. This is a localized repair
retry under unchanged data, target, FP32/N4032, hardware and total compute,
not a new HMC campaign or enlarged compute allocation. The first retry is
SIR --fd-sir, with a hard timeout no greater than the remaining balance.
Its purpose is final-source numerical diagnosis; passing health alone
cannot establish derivative correctness or promote HMC. No statistical
speed ranking will be made from these bounded checks.


The fresh SIR T10 convergence run passed all six parameter/row checks.
For row2/component0 the two extrapolants are -310.8740972 and -310.8740611,
against analytical -310.8740612; the default single-width failure is retained
in the artifact. This supports the local FP64 recurrence, not full-scope
FP32 accuracy. Artifact:
artifacts/surrogate-hmc-sir-unit-convergence-20261012-01.

The final-source GPU SIR diagnostic is allocated at most430 wall seconds
(worker timeout420), below the approximately619-second balance.
Exact command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py run --model sir_d18 --fd-sir --batch-size 2 --paired-repeats 0 --budget-seconds 430 --per-model-timeout 420 --output docs/plans/artifacts/surrogate-hmc-sir-final-source-fd-20261012-01

The master now records XLA/no-host-callback checks before diagnostic exits,
host peak RSS and GPU allocator state even on numerical failures, and preserves
nonfinite values explicitly in error artifacts. These wrapper repairs do not
change the scientific kernel or the local convergence result.


## Final-source SIR diagnosis and FP64 reference

The final-source N4032/FP32/B2 SIR diagnostic completes in104.915s
(worker wall; parent ledger is charged separately). Two perturbed evaluations
take25.63s and27.46s; peak TensorFlow allocator bytes1540003840.
All T50 steps are healthy. Score0=-50.706314 versus finite-value central
differences -168.041992 at h=.005 and -173.339828 at h=.001.
This does not clear the discrepancy. It is not enough to declare a
mathematical derivative error because the T10 CPU width study demonstrated
strong finite-difference curvature, and FP32 finite values may also limit
smaller widths. It is enough to veto a score-accuracy claim until resolved.

Next discriminating experiment, part of this master: evaluate the same
F32-quantized inputs, particle design, controls and theta once in FP64,
N4032/T50/B2/GPU/XLA. This is an explicit independent precision diagnostic,
not a change of default. Compare value, all scores and per-step trace with
the final-source FP32 artifact. A material dtype discrepancy is a numerical
repair trigger; dtype agreement would direct us to a refined FD-width study.
Neither result alone certifies a derivative or HMC. No tuning or posterior
data changes are made. One attempt, budget260s/worker timeout250s, within the
remaining approximately513s of the original allocation. The GPU is RTX4080
SUPER, so FP64 speed is uncertain; timeout is an inconclusive capacity result,
not evidence for either numerical explanation. Full trace fields that were
already computed are now always retained.
Output: artifacts/surrogate-hmc-sir-full-fp64-reference-20261012-01.


## Prefix localization after full FP64 timeout

The FP64 T50 attempt timed out after250.155s before any completed evaluation.
It gives no numerical evidence. The55-test CPU regression suite passed.
Approximately262s remain within3600s. One final bounded attempt uses the
same N4032 particles, data/design prefix and frozen controls, FP64/B2/GPU/XLA,
T10, with derivative widths .001, .0003, .0001 for SIR component0. This is
prefix localization only, not a newly tuned T10 candidate or full-scope
admission. The width sequence is fixed before launch. Compare its unperturbed
per-step scores and values with the corresponding first ten FP32 steps; use
the width convergence to determine whether local numerical precision or
derivative propagation is implicated. Any incomplete width ladder remains
inconclusive. All finite brackets must be healthy and complete. No speed
ranking follows. Cap250 parent seconds /245 worker seconds, preserving a
small remainder in the original total. The attempt convenience limit is
thirteen after this localized retry; total compute is unchanged.

Skeptical audit: using N64 or shortened flow/reset schedules would hide the
actual full-size pathology. This experiment retains both. FP64 may still be
too slow; partial completed evaluations are saved before the next bracket.
Strong curvature can invalidate wide differences, so require a converging
sequence rather than treating one bracket as an oracle. The strongest
possible result is a localized diagnosis, not a full repair.

Exact command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py run --model sir_d18 --fd-sir --fp64-reference --debug-horizon 10 --fd-widths 0.001 0.0003 0.0001 --batch-size 2 --paired-repeats 0 --budget-seconds 250 --per-model-timeout 245 --output docs/plans/artifacts/surrogate-hmc-sir-prefix-fp64-fd-20261012-01


## Additional arithmetic audit: mixture location derivative

Source review found the mixture location derivative still computes
-u^T dx +u^T dm as two large contractions. A common tangent origin o cancels
algebraically, so -u^T(dx-o)+u^T(dm-o) is the same real-arithmetic derivative
with smaller intermediate magnitudes. The transport path already uses this
identity; the mixture path does not. Test a common262144 tangent shift on a
small d18 mixture, comparing the FP32 result with independent FP64 finite
values of exactly the same quantized inputs. A failure is a primitive
cancellation defect, not proof it explains the full SIR discrepancy. If it
fails, center only the analytical location contractions, preserve all primal
values, and rerun the primitive plus broader regression. This is CPU-only
debugging and an exact algebra rearrangement, with no damping, target change,
new default, or GPU-budget extension. No full-scope claim follows without a
subsequent full-scope run.

## Terminal engineering checkpoint, 2026-10-12

The final mixture-centering source passes 59 CPU regression tests and all five
N64/T2/B2 independent finite-value derivative/parity fixtures. The same source
has not been checked at N4032. Exact GPU allocation use is 3586.062/3600
seconds, 13/13 attempts; no worker is active. The repair is unfinished.
See surrogate-hmc-five-model-performance-repair-result-2026-10-12.md and
artifacts/surrogate-hmc-performance-repair-summary-20261012-01/ for reviewed
findings, source hashes, budget ledger and proposed next phases. The proposed
additional 7200 GPU seconds require owner allocation before execution and
do not include full posterior HMC. Preserve FP32/N4032 and the same finite
filter; no scientific direction change is proposed.

## Owner-authorized continuation, 2026-10-12: additional ten GPU hours

The owner authorized 36,000 additional GPU-wall seconds with "you have 10 GPU
hours and continue". This replaces the proposed 7,200-second extension, not
the historical accounting. The original allocation used 3,586.062 seconds.
The new allocation starts at zero. Count whole GPU worker-launch wall time,
including compilation and failed attempts, conservatively. CPU-only unit
regressions do not consume GPU time. Use at most 60 GPU attempts, one GPU
worker at a time, and record each attempt under a fresh versioned directory.
No new launch approval is needed within this unchanged question and budget.

First resolve SIR numerical score validity at N4032, FP32/TF32 off: test the
latest common-tangent-centered mixture derivative against convergent finite
value differences at T50; localize in time with the identical T10 prefix and
FP64 reference, retaining the original FP32-quantized data/design. The existing
frozen controls are paired diagnostic controls, not retuned admission.
Allocate initially up to 3,600 seconds to this localization, up to 14,400 to
shared cost profiling/repairs and all-model checks, and reserve the remainder
for fresh scope tuning and feasible HMC mechanics. These are soft phase
allocations inside the hard total limit. Full inference still needs a
scientifically fixed prior; independent engineering work continues.

Evidence contract: own finite-value derivative convergence is the correctness
criterion; old/new parity and single-width finite differences alone cannot
certify it. FP64 is an explicit diagnostic comparator, not a default change.
The owner's 1% materiality tolerance applies to final likelihood/score precision,
with absolute scale reported for near-zero scores. Report finite-difference
width uncertainty rather than relabeling cancellation as score agreement.
Nonfinite or incomplete paths and significant stable derivative discrepancies
veto promotion and trigger localization/repair. Cost and graph counts explain
performance only. Short tests do not establish posterior convergence, improved
model likelihood accuracy, or canonical LEDH status. The original exact/
Gaussian/random-walk heuristic comparisons remain required for HMC practical-
value claims; precision repairs make no such claim.

Skeptical audit: checked branch, source changes, current checkpoint and
old budget; the latest mixture fix has no N4032 evidence yet. The old SIR
scores are not current-source evidence. A 245-second FP64 timeout left the
width study unfinished, so it cannot support a precision conclusion. Keep
the same data, controls and particle designs; allow a sufficient bounded
reference runtime. Profile warmed kernels separately from compilation.
All-P recursion still computes all-pairs mixtures and transport; eliminating
the parameter replay does not prove useful HMC speed. No material plan flaw
remains for these diagnostic phases. Save outcomes, budget and the next phase
under artifacts/surrogate-hmc-repair-continuation-20261012-01/.

## Shared transport derivative cache: reviewed bounded repair

Question: can the identical finite Sinkhorn reset compute its all-parameter
derivative with less repeated work and cancellation? The current implementation
caches each primal pairwise log cost but recomputes its derivative contraction
inside every row/column normalization. For log cost
C_ij = -||x_i-x_j||^2/epsilon, with fixed epsilon and parameter component r,
d_r C_ij = -2 (x_i-x_j)^T (d_r x_i-d_r x_j)/epsilon.
Both expressions are constant throughout a reset's balancing iterations.
Cache the derivative with the primal cost, using exactly the same tile policy.
For L_i = logsumexp_j(C_ij+a_j), the chain rule gives
d_r L_i = sum_j softmax_j(C_ij+a_j)(d_r C_ij+d_r a_j).
Column reductions use the transposed cached pair. The final barycentric
derivative consumes the same cache. This is an algebraic rearrangement, with
no changed iteration count, epsilon, residual design, ridge, clipping or
scientific target.

Baseline is the pre-cache source preserved by the LGSSM warmed-profile run;
the independent dense FP64 derivative and finite differences are correctness
comparators. Tests cover multiple parameters and batch rows, large common
sensitivities, all five tiny filters and failed-row isolation. Numerical
promotion requires these checks, full N4032 paired values/scores, and SIR
own-value derivative diagnosis. Timing is descriptive with multiple warmed
calls; it is not evidence of score correctness or HMC convergence.

Memory is the main failure mode: [P,B,N,N] costs about 520 MB for LGSSM
P4/B2/N4032/FP32, 780 MB for P6, and 780 MB for SIR P3/FP64. Compute
differences dimension by dimension in a single traced TensorFlow loop, if
needed, so a [P,B,K,K,D] temporary cannot dominate storage. Record allocator
peak and host RSS, first-call compilation and warmed wall times. An OOM or
poor timing rejects the candidate and triggers another shared-kernel repair;
it does not invalidate the research direction. Finite/derivative failure
vetoes use in HMC. All diagnostic settings remain frozen cross-source.

Skeptical audit: cache reuse is mathematically valid only within one reset,
not across time or theta; the implementation must keep that scope. It must
not change row/column ordering or omit potential derivatives. Exact scalar
and B2 consumer call chains already reach this shared routine. Cache memory
and XLA fusion are hypotheses to measure, not assumptions of speed. The
FP32 SIR discrepancy is not yet localized to transport, so this optimization
cannot be called its solution without new evidence. Continue within the
existing 36000-second ledger; retain enough time for all-model checks.

The derivative-cache candidate was rejected on 2026-10-12: N4032/B2/T50
LGSSM warmed times112.68/126.35s versus42.34/42.97s; allocator peak2.154GB
versus.603GB. Values agree within1.53e-5, scores within4.77e-6. These
single-process timings are descriptive but give no reason to retain this
candidate; restored the recorded pre-cache shared transport implementation.
No mathematical or posterior superiority claim was made. GPU profile had
zero device events (CUPTI error999), so device-kernel attribution is unavailable.

## Constant-Jacobian shared linear algebra

The model protocol already states that observation_jacobian_tangent_fn=None
is valid only for a Jacobian constant in both state and parameter along the
executed direction (ledh_canonical_score_tf.py, NonlinearScoreModel). When
this holds for all parameter directions, H, H'R^-1 H and the bridge covariance
P_lambda=(Q^-1+lambda H'R^-1 H)^-1 are shared across ancestors. Compute these
matrices, the flow's matrix product and its covariance once. Preserve all
ancestor-specific means, offsets and translations. The resulting third
mixture component has shared covariance; its existing general density
routine can use one covariance contraction instead of N redundant factors.

This is conservative use of an existing explicit model property, with no
model-name dispatch, new approximation or changed controls. Nonconstant H
continues through the same general equations with an ancestor matrix axis.
Check optimized maps against the full ancestor-matrix assembly for LGSSM,
KSC, predator-prey and SIR, and retain range-bearing regression. Verify all
values AND all parameter tangents, then the five finite filters and full-size
timings/precision. Exact callbacks, not a sample equality test, justify reuse.
Missing/incorrect model metadata would be a correctness veto; callback/map
parity is the earliest diagnostic. The main budget remains unchanged.

## Centered primal reset arithmetic: bounded precision candidate

Question: can equivalent normalized barycentric arithmetic reduce the FP32
SIR discrepancy without changing the real-arithmetic filter or its controls?
The current derivative already subtracts a common state/tangent origin, but
the primal barycenter still sums large absolute coordinates. A column-mass
rounding residual is consequently multiplied by the absolute SIR state.
For a normalized transport column t, use an origin o=x_0 and
s=sum_i t_i, a=sum_i t_i (x_i-o), y=o+a/s. In real arithmetic s=1, so
this is exactly sum_i t_i x_i. Propagate every derivative:
dy=do+(da-(a/s) ds)/s. Do not omit ds merely because its exact value is zero.
Retain the pre-renormalization column residual as an explanatory diagnostic;
positive, finite s is a validity requirement. No epsilon, iteration, cap,
ridge, particle design or selection rule changes.

For normalized cloud moments use the same centered expression:
s=sum_i w_i, mu=o+sum_i w_i(x_i-o)/s,
P=sum_i (w_i/s)(x_i-mu)(x_i-mu)'.
All call sites supply normalized exponential weights or uniform weights.
The explicit divisions remove floating-point normalization error and include
its derivative. This is algebraically equivalent on the documented domain;
it is a numerics-altering implementation candidate whose non-harm must be
checked before use. Apply one shared implementation across scalar/batch paths.

Comparators: saved pre-change full-scope outputs and FP64 evaluation of the
same FP32-quantized inputs; independent dense transport derivatives and direct
weighted-moment derivatives. First test offset states and sensitivities,
translation equivariance, both batch rows and all parameter columns, then all
five small filters, then N4032 SIR and the other full scopes. The candidate
passes only if healthy-reference agreement remains within existing tolerance,
finite derivatives agree, and the SIR final likelihood/score meets the owner's
1% precision criterion (near-zero components use an explicitly reported
absolute scale). A failed precision criterion rejects promotion and triggers
scope-specific tuning/localization; it does not authorize silently changing
the model or using FP64 by default.

Skeptical audit: this is not a new derivative formula or a clipping repair.
SIR T10 FP64 already matches fine-width finite differences, while full T50
has not yet been checked. Centering may fail because the finite filter is too
sensitive to FP32 rounding for these controls; tests must preserve this
alternative explanation. Centering only derivatives would not test the primal
rounding mechanism. This candidate stays within the existing 36000-second
budget and does not itself establish HMC readiness or a speed improvement.

## Fresh tuning call-chain repair and precision veto

The calibration evaluator still used the old directional implementation while
the timed force used the repaired batch implementation. This is wrong for the
claim that calibration and sampling evaluate the same finite FP32 algorithm.
Repair calibration to use the shared batch recurrence at B=1 with optional
primal ancestor history; proposal-context densities must use the same map,
mixture and observation routines. A trace-time history option must not add
particle-history storage to the default HMC hot kernel. Verify history against
the initial cloud and an independently run shorter prefix, then verify that
enabling history leaves every existing output unchanged for distinct rows.
This is a call-chain repair, not permission to change the target or prior.

Fresh calibration seeds26101211/12 and validation seeds26101213/14 replace
the previous unused-or-exposed partitions. Beta fitting uses calibration only;
validation nominates a candidate. Add a same-input FP32/FP64 final-value/score
screen on the fresh validation partition and store all component errors.
One percent with scale max(1,abs(FP64 component)) is a numerical promotion
veto, not the optimization loss. No candidate that fails this screen may
silently become the selected HMC force.

If the centered-primal full-scope screen fails, evaluate a dedicated smoother
reset family epsilon*[1,2,4,8], holding the other numerical controls fixed.
For entropic transport exp(-squared_distance/epsilon), increasing epsilon
directly reduces the cost derivative multiplier 1/epsilon. This supplies a
mechanistic candidate, not a guarantee that the whole recursion is stable:
measure final score precision, reset displacement, restored moments, mixture
weight variance, and independent likelihood/score error where a reference
exists. Epsilon is chosen offline on fresh partitions and frozen before a
new claim dataset; the failed saved dataset remains failed holdout evidence.
A healthier score alone does not prove accurate model likelihood or HMC mixing.

Ordinary viable scopes retain their existing declared three-control family
as hypotheses, with their own calibration/validation and the same precision
veto. A smoother scope is never transferred to another model without its own
checks. Time and memory remain bounded by the additional36000-second ledger;
every compile, failed attempt and retry counts. Continue independent scopes
after candidate rejection and refresh the next-phase JSON after each scope.

Skeptical audit: the current calibration call-chain mismatch must be repaired
before reusing the tuner; old selections are stale after source changes.
Precision of a single saved parameter/data point cannot promote a control;
fresh validation checks are required. Importance variance times warmed cost
only nominates an engineering candidate and cannot rank posterior samplers.
Full posterior inference still requires a defined prior and supported
authority; continue corrected-HMC mechanics and budget estimates meanwhile.

## Shared-covariance pairwise Gaussian contraction audit

The shared covariance case still forms a [B,K,K,D] residual and precision
product. The next bounded engineering candidate expands the same Mahalanobis
form into centered matrix products. Let z=x-o, m=mu-o, dz=dx-do and dm=dmu-do,
with a common origin per tile and P=C^{-1}. Then
q=z'Pz-2z'Pm+m'Pm and the location part of d log density is
-z'P dz+z'P dm+m'P dz-m'P dm. The existing covariance derivative contributes
one half times (x-m)'P(dC)P(x-m) minus one half tr(P dC). Thus the same total
derivative is retained, without a pair-by-state tensor in the shared case.
Ancestor-specific covariance continues through the general formula.

Baseline is the existing direct pairwise residual implementation. Promotion
requires dense value and derivative agreement for multiple rows, all parameter
directions and d=1,3,18, including offset clouds; then full-scope FP32/FP64
precision and measured time. No clipping, ridge, changed probability law, or
dropped mixture term is introduced. Expansion can lose precision for nearly
coincident points; small cancellation fixtures and final-score comparisons
can veto it. Dense tests are engineering checks, not likelihood oracle or HMC
evidence. Keep failed candidates and the same remaining 36,000-second ledger.
Fresh tuning remains after this numerical source change, so stale selections
cannot be consumed. Skeptical audit: this targets allocation and redundant
arithmetic, but N-squared interactions remain; no asymptotic speed claim.

Execution audit before the next GPU phase:100 CPU regressions passed99.33s,
including history invariance, independent context density, finite-value
derivative comparisons, phase continuation and invalid CLI combinations.
The19 standalone shared Gaussian dense tests also pass. The prior paired
phase finished four precision passes and an800s SIR FP64 timeout. This is a
resource failure; budget1800s for each new precision worker and7000s total.
Use one warmed repeat, B2, same quantized inputs, same source, N4032/K2016.
No tuning or mechanics admission occurs until the relevant precision check.

## Tuning observability and cold-run health audit

Before fresh tuning, retain the already-computed ESS, maximum weight,
transport mass residual, reset moment errors, reset margin and displacement
for every calibration/validation trajectory. These are explanatory diagnostics;
the existing finite/valid/full-horizon checks remain numerical vetoes.
For LGSSM include exact Kalman value/score on the same validation observations
and parameter positions. For KSC include both 1201/48 and 1601/56 integration
references and their refinement differences. These report model approximation
error separately from FP32-versus-FP64 numerical error; they are not posterior
validation and do not change the control-selection objective. Other models
have no independent score oracle in this tuner and must say so.

Audit finding: the mechanics pilot checked only its warmed trajectory even
though the cold timing call also executes HMC transitions. Check and archive
both calls. This closes an observability/validity gap without altering any
accepted trajectory or changing the target, prior, controls, or sampling law.
Keep timing compilation separately; no convergence claim follows from either
short trajectory. Focused regressions precede execution. These additions are
part of the authorized master repair and use the same remaining GPU budget.

## Frozen Gaussian contraction choice: execution review

The shared-covariance matrix expression is algebraically equivalent to direct
residual evaluation: for z=x-a and m=mu-a, the quadratic is
z'Pz + m'Pm - 2z'Pm; its location derivative follows by the product rule.
The covariance derivative remains the existing complete -trace(P dC)/2
+ (x-mu)'P dC P(x-mu)/2. Neither route omits covariance dependence.
FP32 rounding may differ, so equivalence is tested rather than asserted from
algebra alone. No clipping, ridge, model, proposal law, reset or MH target changes.

Full-scope descriptive timing reveals a material performance tradeoff:
the matrix expression helps SIR but slows LGSSM. Make direct/matrix a frozen
execution control in ProposalControls, passed through both the batch score and
scalar tuning-context density consumer. Direct restores the prior arithmetic;
matrix is optional. There is no model-name branch in the numerical kernel.
The selected controls and source hashes bind the arithmetic to each fresh
tuning result, preventing a later mechanics worker from changing it.

The pending change is reviewed before execution: test both contractions
against independent dense finite Gaussian values and derivatives, check scalar
and batch call-chain wiring, and repeat the small full-filter regression.
Existing full-scope FP32/FP64 pairs nominate execution choices only. Fresh
scope tuning retains the one-percent final value/score precision veto on
disjoint validation inputs for every eligible candidate. Do not claim a
statistical speed ranking from one warmed observation. Poor performance blocks
promotion of that expression for the scope but does not stop the next repair.
All runs consume the same remaining 36,000-second allocation; no new budget.

### Executable per-scope continuation

The master advance command reads frozen scope-execution.json in the continuation
root: direct contraction for LGSSM, KSC and range bearing; matrix for predator
and SIR. This nominates arithmetic from the paired timing results without
claiming a statistically supported ranking. The kernel has no model-name
dispatch. Fresh own-scope precision, validity and tuning selection remain
mandatory. Tuning caps are 4000,2200,4000,9500,2600 seconds respectively; each
mechanics cap is 1200 seconds. Cumulative GPU wall time remains the controlling
limit. Unsupported modes, missing scopes, non-finite or non-positive caps fail
before a GPU launch. Exact configuration is copied into phase JSON.

Skeptical audit: baseline precision is nomination evidence, not new-source
admission. The actual new-source selection is generated before mechanics and
verified by its source/data identity. Failed SIR precision invokes the already
planned smoother-reset family; it is not silently admitted. Independent scopes
continue after a candidate fails. Full inference is still separate from short
mechanics and remains subject to the declared prior and public tuner authority.

## Current allocation update — 2026-10-13T11:48Z
Final-source precision is complete: LGSSM, KSC, predator-prey and range bearing pass the owner 1% screen. SIR all-FP32 score discrepancy reaches48.6936% while its value agrees; mixed recursion remains an explicit candidate, not a changed default. Fresh scope tuning/mechanics/consumer cost continues automatically in final-scope-advance-01. The actual consumer now supports an optional paired preserved-source check (--consumer-baseline). GPU contention during initial LG tuning is archived and costs from that interval are explanatory only. See the concise checkpoint and current24h plan for exact next actions.

## 2026-10-14 current computation phase

All five selected actual-consumer arithmetic checks pass the 1% same-program
FP64 screen. KSC repeated cost ratios 1.0054 and 1.0141 do not reproduce the
initial 11% slowdown and show no measured gain. Three transport primitive
candidates failed their predeclared cost nomination thresholds; none changed
the runtime. Continue phase 2 with sir-local-arithmetic-plan.md to localize
SIR's all-FP32 discrepancy (48.69% at the saved nominal row). The selected
mixed route passes, but restoring FP32 could reduce its remaining cost.
This is computation repair; posterior HMC remains unexecuted.

## 2026-10-14 fixed-design repair

The SIR localization exposed a real comparison/implementation defect: FP32
rounding before floor selected different proposal ancestors than FP64 from the
same representable uniforms. The events at observations 1 and 7 coincide with
the large local proposal differences. Follow
`artifacts/surrogate-hmc-cost-repair-24h-20261013-01/ancestor-selection-repair-plan.md`.
Both actual draw paths now share binary64 discrete-index selection; continuous
kernels remain in their requested precision. This changes the old FP32 finite
design at boundaries and therefore requires refreshed arithmetic comparisons,
fresh tuning identities and consumer checks. It is phase 2 of the existing
master, not a change of scientific direction. 150 CPU regressions passed.
The corrected SIR T10 GPU localization is running; full-horizon all-model
precision checks follow, then any remaining targeted repair. No posterior
convergence or canonical admission is implied.
