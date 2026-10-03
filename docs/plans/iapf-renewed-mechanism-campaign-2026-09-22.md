# Renewed iAPF mechanism and implementation campaign

Status: COMPLETE for this bounded mechanism campaign; the wider master
program remains open. See the [terminal result and review](artifacts/iapf-renewed-mechanism-20260922-01/result.md)
and [budget ledger](artifacts/iapf-renewed-mechanism-20260922-01/budget.json).
Owner authorized another 48 CPU hours and 48 GPU hours on
2026-09-22. This is a new allocation, not a revival of the expired campaign
deadline. Entry checkout: `surrogate-hmc`, commit
`6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`, with unrelated dirty work preserved.
The master remains [the model-score master program](younis-kdm-score-master-program-2026-09-14.md).
Results: `artifacts/iapf-renewed-mechanism-20260922-01/`.

## Research intent and source audit

The immediate question is why a numerically valid learned guide can give a
poor likelihood estimate in GJL's first linear-Gaussian study. Distinguish
error imposed by a diagonal guide, finite-cloud estimation, concentrated
regression weights, inherited backward-recursion error, and the positive
floor. The wider purpose is a trustworthy reference for debugging the
TensorFlow filtering and model-score work. A good likelihood reference does
not establish a correct score, KDM implementation, or canonical LEDH route.

Primary source: local tractable PDF/text of Guarniero, Johansen and Lee,
*The iterated auxiliary particle filter*, arXiv working-paper version,
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.*`.
Checked anchors: Propositions 1–3, Section 3.3 / Algorithm 3, Section 3.4 /
Algorithm 4, Section 5.1 / Equations 15–16 and Algorithm 5, Section 5.2.
Inspect the variance appendix before using its result. Record version limits;
the authors' numerical fitting procedure and floor remain unrecovered.
The existing source audit and numerical replay are evidence, not authority
to silently identify a log-regression extension with Equation 15.

The checked R call chain is `iapf_choice_run` or
`iapf_constrained_diagnostic_run` → `iapf_iterate` → `iapf_apf` →
`iapf_backward_log_target` and the corresponding fitter. Both use newly
fitted next-time guides. The TensorFlow chain is `execute_iapf` →
`make_fitted_twist_kernel` / `make_density_recursive_fit_kernel`; the consumer
explicitly rejects dimensions other than d=o=1, uses a different model/time
boundary and fitting/controller choices, and has only component parity with
R. Whole-filter equivalence is not established.

Checked previous evidence: 160/160 independent learners completed; 51,100
repaired regressions passed their numerical checks, but some weight ESS values
were approximately one. All five paired 99% accuracy intervals included zero.
The repaired reference failed 14 conditional heuristic screens. No accuracy
ranking was established. The old validation inputs are now diagnostic data,
never a new untouched claim set.

## Mathematical diagnostic

Let the exact predictive state density before observation t be `p_t`, the
exact future guide be `h_t`, and the learned guide be `g_t`. Define
`s_t ∝ p_t h_t` and `q_t ∝ p_t g_t`. The former is the exact smoothed marginal.
For a Gaussian guide, both densities are Gaussian, so `KL(s_t || q_t)` is
available without particle noise. This measures guide discrepancy on the
relevant state distribution, not only residual at training points.

If `p_t=N(a,P)` and `s_t=N(b,S)`, a diagonal-precision Gaussian guide yields
`q_t=N(b,(P^-1+diag(j))^-1)` after optimizing its mean. Its best nonnegative
precision j minimizes

`0.5 * [tr((P^-1+diag(j)) S) - logdet(P^-1+diag(j))]`, `j >= 0`.

The gradient is `0.5*(diag(S)-diag((P^-1+diag(j))^-1))`; the Hessian is
`0.5*(C ∘ C)`. This convex projection is a diagnostic lower bound for the
Gaussian component family, including its closure at zero precision. It is
not an executable general-model learner, nor a lower bound for the broader
Gaussian-plus-floor family. Check the projection's KKT residual and objective
against feasible moment/precision-diagonal comparators.

The actual floored guide produces a known mixture of this Gaussian and p_t.
Measure its discrepancy separately with independent smoothing draws and
paired Monte Carlo uncertainty. Do not silently discard its floor. A
one-step fitting error must use `g_t^target = observation_t * F(g_{t+1})`,
not the ideal h_t; otherwise recursion error is incorrectly blamed on the
current optimizer. Decompose these quantities on common heldout draws.

## Execution sequence and evidence contract

1. **Saved-guide diagnosis.** Freeze and hash the three existing R reference
   sources. Use all 80 saved paired cells and their final guides. Compute
   exact Gaussian-component KL through the full horizon. At t=1,25,50,75,99,
   100 compute the actual floored-mixture KL with common independent draws,
   the diagonal-family optimum, local backward residual and exact-future
   residual after removing irrelevant additive constants. Report by dimension,
   dataset, time and method. Averages must not hide d80 or late-time failures.
   Validate smoothing moments against an independent RTS recursion, Gaussian
   projection gradients/KKT, exact-guide zero discrepancy and the shared
   APF/Kalman oracle. These identities are pass/fail criteria; guide distances
   and correlations with likelihood error are explanatory only.
2. **Controlled mechanism intervention.** On fresh data compare the frozen QR
   and repaired weighted-log references with the same constrained solver at
   exponent zero. This isolates regression weighting while preserving the
   solver, positive-curvature guard, floor, filter and iteration controller.
   An intermediate exponent may be used only as an explicitly labeled
   explanatory arm, not selected after seeing final errors. Replay selected
   one-step clouds with both a learned next guide and an exact next guide;
   compare training and independently drawn heldout log-shape errors. Use
   fresh paired full learners to test downstream consequences, not just
   regression loss. Begin with two datasets and four seeds per dimension;
   inspect validity before expanding to an independent confirmation batch.
   Freeze any nominated change before confirmation. Failed weighted fits are
   repair triggers, not vetoes on continuing the unweighted test.
3. **TensorFlow/GPU bridge.** Audit exact model, time indexing, initial
   distribution, resampling, fitting objective, floor and stopping choices at
   the actual consumers. Test compatible shared computations under GPU/XLA
   FP64 and FP32, with memory growth and explicit device provenance. Implement
   the smallest reference comparison needed to expose or resolve a concrete
   mismatch. Do not remove the scalar guard until an actual multidimensional
   consumer has executable tests. A diagnostic comparison cannot promote a
   separate production algorithm, score or default.
4. **Confirmation and closeout.** If the intervention yields a viable
   reference, freeze it and test fresh observations/seeds with paired
   uncertainty and the heuristic ladder. Otherwise preserve the negative
   mechanism result and retain the viable QR reference. Reconcile the master,
   checkpoint, decision/inference tables, source ledger and remaining budgets.
   Stop this program when these questions are answered or a true veto fires;
   unused budget is not a requirement to run uninformative experiments.

Primary downstream criterion: finite, valid fresh terminal likelihood
estimates against exact Kalman, with absolute log error and relative-likelihood
RMSE reported separately. Statistical comparisons use dataset-stratified
paired resampling, 20,000 resamples and simultaneous 99% intervals over five
dimensions. A numerical solver pass is not a performance pass. Rare-tail and
bias claims require more evidence than a small sample of likelihood ratios.

Constructed heuristic adversaries: constant guide/BPF (no learning),
observation-only guide (current information), exact-guide moment-diagonal
(known Gaussian marginal variances), exact-guide precision-diagonal (known
diagonal curvature), and full exact future guide (Kalman equality oracle).
The analytic controls deliberately use unavailable model knowledge; they
diagnose avoidable approximation error, not fair learning cost. Evaluate
conditionally on each dimension and observation dataset. A heuristic loss
vetoes promotion, but does not itself prove statistically worse performance.
All learning passes, reruns and fresh final filters count in cost.

Validity / continuation vetoes: inconsistent target or RNG pairing, failed
independent identity, nonfinite supposedly valid result, corrupted/missing
inputs, unreviewed scientific scope change, exhausted compute. Local harness,
serialization or solver failures trigger bounded repair and a fresh attempt.
Candidate accuracy failure blocks its promotion, not later diagnostic phases.
No conclusions about author identity, exact paper replication, unbiased
log-likelihood, gradient correctness, LEDH, KDM, HMC or production readiness.

## Defaults, assumptions, and pre-mortem

| Choice | Provenance / role | Risk and earliest check |
|---|---|---|
| d=5,10,20,40,80; T=100; alpha=.42; identity noises | Paper Section 5.2, baseline | Verify model construction and Kalman tie-out |
| N0=1000,k=5,tau=.5,kappa=.5 | Paper, baseline | Preserve exact windows and account for ambiguous first doubling |
| tail8 floor, sample SD, after-k doubling | Existing independent reference hypotheses | Log floor mass; do not claim author identity |
| Max 12 iterations / N=4000 initially | Bounded validation choice | Report censoring as failure, expand only within budget |
| Weighted exponent 1 versus 0 | Existing extension versus mechanism ablation | Same-cloud targets and same solver; zero is not selected by heldout error |
| Condition target eps^-1/2, curvature guard | Frozen numerical repair | KKT/residual plus downstream guide error; conditioning alone cannot recover information |
| Exact Gaussian KL / projection | Local derivation, explanatory oracle | RTS and finite-difference checks; positive floor analyzed separately |
| Selected times and 1024 smoothing draws | Coverage / bounded MC convenience | Full-horizon component KL; MC uncertainty, no tail certification |
| CPU single-thread R | Explicit independent reference exception | Record CUDA_VISIBLE_DEVICES=-1 and thread controls |
| GPU/XLA | Repository default backend | Elevated probes, growth, precision and parity before benchmark |

Pre-mortem: the convex projection could be mistaken for a realizable learned
filter; its oracle status is explicit. A successful local fit could inherit a
poor next-time guide; the one-step/exact-next decomposition separates them.
Old validation data could be reused for promotion; fresh confirmation is
required. A Gaussian-component result could miss floor tails; preserve both.
A TensorFlow primitive could pass while its consumer is unreachable; test the
call chain. Broad searches and repeated review could consume the campaign;
use concise checkpoints and one substantive audit, followed by terminal review.

## Compute, artifacts and skeptical review

New total pools: 172,800 CPU worker seconds and 172,800 GPU process seconds.
Debit every experiment attempt including failed runs, verification and repairs;
single-thread CPU workers make worker time an approximate core-time bound.
GPU process wall time is a conservative occupied-device accounting rule.
Initial phase caps: diagnosis 7,200 CPU seconds / 20 attempts; intervention
28,800 CPU seconds / 250 attempts; confirmation 72,000 CPU seconds / 500
attempts; bridge 7,200 CPU and 14,400 GPU seconds / 30 attempts. Remaining
CPU 57,600 and GPU 158,400 seconds are a recorded repair/extension reserve,
not permission for a different scientific target. No additional wall deadline
was specified by the owner. Do not inherit the old expired one.

Every launch writes a unique attempt directory, source/input hashes, command,
commit, environment, seeds, elapsed time, status and output paths. Save full
logs on disk; summarize only relevant fields. No external messaging, package
changes or destruction is required.

Skeptical pre-execution review: PASS with the following corrections embodied
above. Existing analytic diagonals are not the best attainable diagonal
projection; compute the appropriate convex oracle. The paper's variance
formula describes its stated resampling setting and must not be asserted as
the finite-N variance of the adaptive implementation. Numerical KKT does not
certify statistical information. R/TF component parity does not certify the
scalar consumer at d80. The prior budget/deadline are historical. Primary
criteria, conditional heuristics, heldout separation and repair conditions are
explicit. This is a Codex self-review, not an independent review certificate.

### Bridge execution specification, recorded before launch

The full-filter test reconstructs the *actual TF fixed-guide consumer* in R:
its six-parameter LGSSM, an x0 draw followed by a transition before y1,
initial lookahead resampling and every-step multinomial resampling. It uses
d=o=1,2,5, N=64, T=5 with both constant and positive-floor guides and shared
supplied Gaussian/uniform draws (seeds 93600001, 93600002, 93600005).
R central differences at 1e-5 and 5e-6 must retain every discrete label and
mixture choice and agree within 1e-5. This derivative is only the fixed-label
finite-program derivative, not the physical model score.

Compare actual `make_fitted_twist_kernel` values, all clouds and derivatives
under GPU/XLA FP64, FP32 without TF32 and FP32 with TF32. Predeclared
absolute/relative tolerances: FP64 values/clouds 1e-9/1e-9, derivatives
1e-5/1e-7; FP32 values/clouds 1e-4/1e-5, derivatives 5e-4/1e-4. A failure
triggers localization, never silent tolerance relaxation. Also execute the
multidimensional adapter veto. Passing the lower-level filter does not remove
that higher-level limitation or establish adaptive-learner parity.

`diagnose_iapf_tf_filter_bridge.py` snapshots the scientific sources, exports
the R reference and invokes the actual TF kernel. An elevated NVIDIA probe
identified the idle RTX 4080 SUPER by UUID
`GPU-68251639-fe82-8f81-3ccc-2953c32e805b`; use the UUID because CUDA ordinal
1 resolved to the other device. Enforce and record memory growth before
initialization, actual tensor placement, compilation evidence and allocator
peak. Up to 1200 seconds in this attempt, within the bridge cap. Debit a
conservative 30 GPU seconds for the preliminary device-only probe. These
small fixtures diagnose correctness; they are not performance benchmarks.

### Intervention decision and frozen confirmation

The 40-cell intervention completed all 120 learners. Guarded unweighted
regression matches QR terminal log values within 3.64e-12; all 25,200
unweighted fits use zero ridge and inactive curvature guards. The weighted
arm needs 1,685 ridges in 25,300 fits. Saved-guide and fresh guide discrepancies
nominate the unweighted guard as a numerically protected reference hypothesis,
not a statistically superior learner. Keep the exact existing controls and
exponent fixed; make no error-based hyperparameter selection.

Confirm on four new datasets per dimension, eight independent learner seeds
per dataset (160 cells / 480 learners), using `confirmation` in
`diagnose_iapf_weight_intervention.R` and its +1,000,000 seed offset. Primary
paired contrast is weighted minus guarded-unweighted terminal absolute log
error. Use dataset-stratified paired 99% intervals with 20,000 resamples,
conditional heuristic screens and the fixed N=1000 probes. QR parity remains
a numerical check. Four datasets limit population generalization; bootstrap
conditional on them does not certify unseen-observation tails. Up to four
single-thread R workers and 10,800 wall seconds, conservatively charged as
43,200 CPU worker seconds, inside the 72,000-second confirmation cap. No
retuning on these new observations. Continue GPU localization independently.

GPU first attempt: FP64 and FP32 without TF32 pass all full-filter comparisons;
TF32 fails 12 value/score/cloud comparisons. Treat this as a comparison veto
and localization trigger. Inspect actual resampling labels and cumulative
probabilities before claiming a mathematical or implementation defect. TF32
can change a discrete ancestor when a uniform draw is close to a boundary;
large path disagreement alone does not establish biased likelihood estimation.

### TF32 localization review

Expose ancestor indices, their cumulative probabilities and Gaussian-mixture
probabilities from the actual fixed-guide kernel through an optional
`include_numerical_trace=False` argument. This is observability only: no
precision, guard, filter decision or default changes. Six new trace checks
and three existing consumer tests pass on CPU (9 tests total). Each GPU case
must also show bit-for-bit equality of the original outputs with and without
tracing; cumulative probabilities must reconstruct the actual indices.
Failure of either check invalidates attribution to the ordinary kernel and
triggers a localized observability repair.

For a valid trace, compare TF32 with FP32 without TF32 in chronological order:
initial resampling, each mixture decision, then each observation's resampling.
Record the first changed choice, the uniform draw, the old/new probability
boundary, the rounding perturbation and the pre-perturbation margin. A
boundary crossing explains that paired path discontinuity; it establishes
neither likelihood bias nor a general failure of TF32. Same-label numeric
differences remain separate. Preserve the failed original parity verdict
even when localization succeeds. Run `diagnose_iapf_tf_filter_bridge.py
--trace` in a new source-snapshotted GPU attempt, capped at 1200 seconds.

Skeptical review: PASS. The comparator, random inputs and scientific target
are unchanged, extra outputs are explicitly checked for numerical effects,
and a diagnosis-complete status cannot replace a failed precision comparison.
This is a bounded explanation of observed failures, not a precision-default
change or a new performance claim.

### Confirmation reporting audit

Before confirmation completes, resolve the wording conflict between
"simultaneous 99%" above and the shorter "99% intervals" execution note.
Use the stricter interpretation: Bonferroni 99.8% percentile intervals for
each of the five primary dimension contrasts, giving a nominal 99% family
level. Also preserve marginal 99% intervals, clearly labeled. Use the already
declared 20,000 dataset-stratified paired resamples; at the adjusted tails
only about 20 resamples determine a quantile, so retain their Monte Carlo
limitation. These are conditional bootstrap intervals, not exact finite-sample
coverage guarantees. No selection or retuning follows these results.

The conditional heuristic screen uses the same N=1000 fresh probes and
compares absolute log error and relative-likelihood RMSE separately for every
dimension and dataset. Any observed loss is a conservative promotion veto,
not a statistically supported ranking. Preserve all observations and all
candidate failures; do not drop failed learners from the primary comparison.
The report must verify complete pairing, exact-guide Kalman agreement,
guarded-zero/QR numerical parity and the status of every learner before
interpreting intervals. This correction tightens the original evidence
standard and does not use the still-running confirmation results.

### Unexpected pre-resampling TF32 displacement

GPU attempt003 preserves clouds as well as decisions. In the d=2 fitted
fixture, at the second transition, 60 Gaussian-branch particles move by
approximately (+0.1575,-0.2498) relative to FP32, before any changed discrete
choice. This is too large and too uniform to attribute to the observed
small probability perturbations alone. The displacement agrees approximately
with K times (the previous guide center minus the current guide center).
Treat stale-center use as an unproved compiler/data-dependence hypothesis.

Next discriminating check: invoke the actual `twisted_transition` on supplied
means/noise and time-varying centers, using a stable-signature GPU graph with
and without XLA and a short statically unrolled comparison. Compare FP64,
ordinary FP32 and TF32. These are diagnostic execution variants of the same
routine, not algorithm candidates or a new production fork. Preserve both
returned values and tangents so pruning cannot silently change the question.
Use a fixed T=5, N=64, d=2 fixture and contrasting supplied centers; compare
to an independent closed-form Gaussian update. Classify loop-only, XLA-only
and TF32-only discrepancies separately. Cap each probe at 1200 GPU seconds
inside the existing bridge budget. A reproducible wrong time center would
be an implementation/compiler failure, not evidence against iAPF mathematics.

Skeptical review: PASS. This extends the stated localization repair, keeps
the actual routine under test, freezes inputs to prevent resampling feedback,
and distinguishes an actual wrong update from legitimate changed random
choices. No numerical default or scientific target is changed.

GPU attempt004 reproduces a loop-specific wrong result: for frozen inputs,
TF32/XLA with the actual transition has errors up to 1.88 on alternating
centers, while the statically expanded TF32 calls stay within 0.0012 and
non-XLA FP32 within 3e-7. FP64 agrees across routes to 1.3e-15. The error
matches use of the first center at subsequent times. This is a local
implementation/compiler failure, not just divergent resampling genealogy.

Evaluate a shared-routine repair that expands K(c-m) as Kc-Km, keeping the
time-dependent center in a matrix-vector operation. Expand its analytical
tangent consistently. This preserves the mathematical transition, guide,
random inputs and numerical mode; it does not change TF32 policy. Preserve
the old source snapshots as the baseline. Before retaining the repair,
require focused existing and trace tests, the frozen-center loop probe against
an independently evaluated precision-form Gaussian product, FP64/ordinary
FP32 full-filter parity, and removal of the large TF32 center-reuse error.
Ordinary TF32 rounding and resulting resampling discontinuities may remain;
they cannot be relabeled as full FP32 parity. Accept by correction and
non-harm on these numerical contracts, not likelihood-accuracy optimization.
The scope remains this shared finite-filter routine; no canonical LEDH,
adaptive iAPF or physical-model-score claim follows.

Skeptical repair review: PASS. A weaker blanket precision disable would not
answer the identified data-dependence defect. The proposed expression is
algebraically identical, includes the tangent, and is tested in the actual
consumer as well as a discriminating time-varying-center fixture. A failed
non-harm check triggers repair, not tolerance relaxation.

Attempt005 removes the large TF32 center error, but its independent FP64
reference check fails at 2.24e-8. The diagnostic constructed means with
`tf.cast(.3, float64)`, which first rounded the Python literal through FP32;
R used a double literal. Repair that supplied diagnostic input to
`tf.constant(.3, dtype)`, preserve the failed attempt, and rerun unchanged
tolerances. This is a comparison-harness mismatch, not a reference tolerance
exception. The repaired shared routine passes all 10 focused CPU tests.

Confirmation finished 480 attempts: 478 complete learners and two weighted
d80 iteration-cap outcomes (dataset 1, replications 2 and 8). Both QR and
guarded-zero complete 160/160. The first report correctly refuses its
all-complete check; it must now distinguish candidate incompletion from
invalid input. Preserve every failed row and capped history, recover the
fit diagnostics from the saved capped attempts, and emit no primary d80
accuracy interval or partial-sample heuristic ranking. The four complete
dimension comparisons retain the originally allocated five-comparison
Bonferroni adjustment. Treat the two caps as a promotion veto for the frozen
weighted protocol, not proof that additional iterations could never succeed.
This reporting repair changes neither the data nor the frozen learners.

### Bounded explanation of iteration censoring

The two capped weighted runs have finite fits and last-window CVs 0.577 and
0.562, versus the frozen stopping threshold 0.5, after 12 iterations at
N=2000. To distinguish limited iteration budget from persistent failure,
rerun exactly these two seeds with a diagnostic maximum of 24 iterations and
the unchanged N<=4000 bound. Verify that all 12 original likelihoods and
particle counts reproduce before interpreting any continuation. Save these
as new diagnostic attempts and never substitute their terminal values into
the frozen confirmation or its intervals. A completion establishes only that
more iterations resolved these selected capped cases. A repeated cap remains
censoring, not a proof of divergence. Up to two CPU workers and 600 wall
seconds (1200 worker seconds) inside the confirmation/repair allocation.

Skeptical review: PASS. This is a predeclared-type cap expansion used only to
explain a failure, with an executable identical-prefix check and no rescue
of the original promotion result. Doubling the iteration ceiling is a bounded
diagnostic choice, not a newly tuned default. No further expansion is implied.

### Compiler mechanism isolation

The standalone `--broadcast-probe` in GPU attempt008 reproduces the wrong
time dependence without filtering, resampling, covariance factorization or
derivatives. On this local TensorFlow 2.20.0-dev0+selfbuilt / CUDA 12.8.1 /
RTX 4080 SUPER stack, TF32/XLA computes `K(c_t-m)` with errors up to 1.92;
the split expression stays within 0.000167. Ordinary FP32 and static unrolling
are controls. Optimized HLO retains the correct dynamic index but fuses its
slice, broadcast and subtraction into `__triton_gemm`.

One additional diagnostic attempt will disable that compiler fusion using
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` and rerun the same standalone
probe. This is a process-local debugging exception, not a numerical default
change. Record the flag in the manifest. Primary criterion: the original
broadcast expression recovers the correct time dependence with maximum error
below 0.003, all controls pass, and its optimized HLO has no Triton GEMM.
The 0.003 bound already used in the center diagnostic separates the O(1)
wrong-center defect from TF32 rounding; it is not full-filter precision
admission. A failure narrows attribution without invalidating the tested
algebraic repair. No upstream software version other than this installed
build, or exact compiler source-level cause, is established by this test.
Use one source-snapshotted attempt, at most 1200 GPU seconds within the bridge
allocation; preserve the original failed expression and HLO.

Skeptical review: PASS. The control changes only a locally verified compiler
flag, uses identical mathematical inputs, and cannot tune the learner or
replace the original precision verdict. The additional evidence answers
whether the observed failure is specific to the fused compilation route.

### Final call-chain review

The shared `twisted_transition` has one production call site, inside
`make_fitted_twist_kernel`. Both `fitted_twist_adapter` and `iapf_adapter`
consume this kernel, and the nonlinear adapter delegates to those consumers.
The existing `test_younis_score_master_nonlinear_iapf.py` explicitly exercises
that adaptive selection/data/call-chain endpoint and its finite-difference
checks. Run this additional seven-test CPU reference file with devices hidden
and a 300-second cap before closeout; the prior nonlinear-kernel test file
does not replace this consumer coverage. Skeptical review: PASS; this checks
a distinct dependent endpoint without changing the experiment or its criteria.

Earlier GPU/XLA/TF32 results from the original affected fitted transition
must be revalidated before reuse as evidence. This is a route-specific
quarantine, not a statement that every earlier scalar or FP64 result is
numerically wrong. Independent frozen R evidence is outside this compiler
failure. Preserve prior artifacts and record the limitation in the master.

## Terminal execution and review

All four planned phases and their localized repairs completed. The 480-attempt
confirmation preserves two weighted iteration caps; their separate extensions
complete at iterations 13 and 14 with identical original prefixes. No complete
paired d80 accuracy interval is reported. All four estimable adjusted intervals
contain zero. Each learned arm remains blocked by conditional heuristic screens.

The GPU defect is reproduced in a standalone dynamic-center broadcast GEMM.
Disabling Triton GEMM for one control process removes it and selects cuBLAS.
The retained shared-routine repair expands K(c-m) into Kc-Km and preserves
the analytical tangent. Fifty-four focused and dependent consumer tests pass.
Final FP64 and ordinary FP32 full-filter comparisons pass 18/18 each; the
strict TF32 comparison still fails 11/18, with that veto preserved. The seven
additional nonlinear adaptive tests close the final call-chain coverage gap.

Terminal skeptical self-review: PASS for the bounded engineering and mechanism
conclusions. No accuracy ranking, paper/author identity, model-score, canonical
LEDH, KDM or HMC promotion follows. This is a self-review, not independent
review certification. The original comparison criteria, failed attempts and
frozen confirmation are retained. All attempted repairs addressed real
infrastructure or implementation findings without changing the scientific
target. See the terminal decision and inference-status tables for residual
uncertainty and the next evidence required.

Execution consumed approximately 2.072 CPU hours and 0.08246 GPU hours.
The unused allocation remains available; no worker is running. The stopping
reason is completion of this plan's questions, not exhaustion, a candidate
rejection promoted into a continuation veto, or a new approval requirement.
