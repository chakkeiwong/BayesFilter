# Renewed iAPF mechanism campaign: terminal result

The learned references still lose conditional comparisons with analytic
diagonal-guide controls, so none qualifies for promotion. There is no
statistically supported accuracy ranking between weighted and unweighted
fitting. The campaign nevertheless identifies two concrete causes of failure:
concentrated regression weights lose useful fitting information, and a
GPU/XLA fusion computes the wrong time-dependent Gaussian transition.
The shared transition has been repaired and checked against an independent
R calculation. Full paper replication remains open.

All four phases of the [reviewed plan](../../iapf-renewed-mechanism-campaign-2026-09-22.md)
are complete: saved-guide diagnosis, controlled fitting intervention, actual
TensorFlow consumer comparison, and fresh confirmation. The larger
[master program](../../younis-kdm-score-master-program-2026-09-14.md)
remains open. No research worker is running.

## What was tested

The R reference uses GJL's first linear-Gaussian study: dimensions
5, 10, 20, 40 and 80; 100 observations; A_ij=0.42^(|i-j|+1);
identity initial, process and observation covariances. The three learners
are the frozen log-quadratic QR reference, safeguarded unweighted log
regression, and safeguarded weighted log regression. All retain the same
local floor, adaptive resampling and stopping choices. Log regression is
an extension; it does not implement the paper's density objective in
Equation 15. The authors' initialized numerical fitting procedure and
numerical floor rule have not been recovered.

The GPU comparison executes the actual shared fixed-guide TF consumer at
d=o=1,2,5, N=64, T=5 against an R reconstruction of that consumer's model,
time indexing, mixture and resampling. This TF model is not the paper's
first-study model. Its score check differentiates the same finite program
with fixed discrete labels; it does not validate the marginal model score.
The adaptive TF adapter still explicitly rejects d=o>1.

## Fitting mechanism and confirmation

A diagonal quadratic omits the exact guide's cross terms. Weighted fitting
then solves D'WD beta=D'Wy using D=(1,z,z²). At d=80 this has 161
coefficients, while measured weight ESS can approach one. ESS is not
algebraic rank, but the measured singular values and active ridges confirm
the loss of numerical information. A small KKT residual establishes that
the constrained optimization was solved; it cannot recover omitted terms
or information absent from the weighted cloud.

Across the previously saved d80 guides, Gaussian-component KL is 0.198683
for QR, 0.497642 for weighted regression, and 0.164935 for the best
attainable diagonal-precision projection. These are explanatory guide
distances, not likelihood rankings. The independent floor-mixture check
finds floor probabilities below 3.5e-23 in the inspected regime. Local
backward-fit discrepancy exceeds inherited recursion discrepancy.
The [source and mathematical audit](source-and-math-audit.md) gives the
derivations, paper anchors and interpretation limits, including the
vanishing-density degeneracy of the unrestricted Equation 15 objective.

The intervention completed 120/120 learners over 40 paired cells. The
fresh confirmation used four new datasets per dimension and eight seeds
per dataset: 160 cells, 480 learner attempts, and 1,278 fresh filter/control
probes. Exact full-guide controls agree with Kalman to 5.46e-12.

| Frozen confirmation arm | Complete learners | Recorded fits | Positive ridges | Active-bound fits | Minimum weight ESS |
|---|---:|---:|---:|---:|---:|
| QR | 160/160 | 99,300 | 0 | 0 | 1,000 |
| Safeguarded unweighted | 160/160 | 99,300 | 0 | 0 | 1,000 |
| Safeguarded weighted | 158/160 | 102,400 | 6,650 | 4 | about 1 |

Unweighted safeguarded terminal values agree with QR within 3.64e-12.
Thus removing weighting removes the observed need for ridge repair in
these cases while retaining the guard. It does not demonstrate uniformly
better likelihood accuracy. The largest recorded KKT residual is
6.73e-13 for unweighted and 1.91e-12 for weighted fitting.

The primary contrast is weighted minus safeguarded-unweighted mean absolute
terminal log-likelihood error. Negative values favor weighted descriptively.
Dataset-stratified paired bootstrap intervals use 20,000 resamples and
99.8% per-dimension intervals, preserving the predeclared five-comparison
Bonferroni allocation for nominal 99% family coverage.

| Dimension | Observed difference | Adjusted interval | Interpretation |
|---:|---:|---:|---|
| 5 | 0.01406 | [-0.00740, 0.04139] | Unresolved |
| 10 | 0.01329 | [-0.00934, 0.03358] | Unresolved |
| 20 | -0.00235 | [-0.04422, 0.04304] | Unresolved |
| 40 | 0.03411 | [-0.03593, 0.11233] | Unresolved |
| 80 | Not estimated | Two incomplete weighted learners | No complete paired accuracy comparison |

Every estimable interval includes zero. These intervals are conditional on
four datasets, with only eight seeds per dataset. Adjusted bootstrap tails
contain about 20 resamples; finite-sample coverage and unseen-data tails are
not certified. [Primary intervals](attempt006-report/results/primary-intervals.csv)
preserve the unrounded values and separately labeled marginal intervals.

At the common N=1000 evaluation budget, the heuristic screen compares
constant/BPF, observation-only, exact-guide moment-diagonal, and
exact-guide precision-diagonal controls separately for each dimension and
dataset, using both absolute log error and relative-likelihood RMSE.
QR and safeguarded unweighted each incur eight promotion vetoes: five
against moment-diagonal and three against precision-diagonal controls.
Neither loses an observed screen to the constant or observation-only
control. Weighted incurs twelve observed accuracy vetoes plus four
incomplete-cell vetoes. The analytic diagonals use privileged Gaussian
future information, so this is an adequacy screen, not an equal-training-cost
competition. Its observed losses block promotion without establishing a
statistical ranking. All 240 conditional comparisons are in
[the heuristic table](attempt006-report/results/heuristic-screen.csv).

The two weighted d80 caps reached iteration 12 at N=2000, with CV 0.577
and 0.562 against the threshold 0.5. A separate diagnostic extension
reproduces both original 12-iteration histories exactly, then completes
at iterations 13 and 14, with N=2000 and N=4000 respectively. These were
iteration-budget failures, not demonstrated divergence. Their new terminal
values do not replace the frozen confirmation or fill its missing interval.
See [the extension](attempt007-cap_extension/results/cap-extension.csv).

## Compiled transition defect and repair

The Gaussian transition mean is m+K(c_t-m), with K=Q(Q+V_t)^-1.
In the original GPU/XLA/TF32 loop, the error agrees with K(c_0-c_t):
the first center is effectively reused at later times. The large
displacement occurs before the first changed resampling choice. It is
wrong relative to the stated Gaussian update, not a valid alternative
particle trajectory arising only from rounding near a probability boundary.

The standalone expression in GPU attempt008 reproduces the defect without
filtering, covariance calculations, derivatives or resampling. On the
installed TensorFlow 2.20.0-dev0+selfbuilt / CUDA 12.8.1 / RTX 4080 SUPER
stack, its error grows to 1.92007. Optimized HLO incorporates a correctly
indexed dynamic slice and broadcast into a Triton GEMM fusion. Disabling
that fusion only for GPU attempt009 selects cuBLAS and reduces the error
below 1.69e-7. This localizes the failure to the fused compilation route;
the exact compiler source-level defect and other builds remain untested.

The shared implementation now computes m-Km+Kc_t and expands the analytical
tangent consistently. It preserves the mathematical update and keeps TF32
enabled under the existing repository default. The
[isolated campaign patch](shared-kernel-repair.patch) excludes unrelated
pre-existing edits. Optional numerical tracing records actual decisions and
is bit-for-bit neutral on all 18 checked GPU cases.

| Verification | Result |
|---|---|
| Focused repair/trace tests | 10 passed |
| Adaptive consumer, Fisher and nonlinear consumer tests | 44 passed |
| Independent R center-product comparison, FP64 | Max error 7.55e-15 |
| Same comparison, ordinary FP32 | Max error 4.76e-7 |
| Same comparison, TF32 | Max error 0.000808, below the diagnostic 0.003 bound |
| Final full-filter FP64 / ordinary FP32 | 18/18 comparisons pass in each mode |
| Final strict full-filter TF32 | 7/18 pass; 11/18 fail, veto retained |

The remaining TF32 differences include rounding-level discrepancies with
identical labels and larger changes after an actual probability-boundary
crossing. For example, the repaired d2 fitted case stays within 0.000805
cloud error before its late changed ancestor; its final cloud error is
0.11651. The original premature O(1) displacement is removed. These
diagnostics do not establish likelihood accuracy or unbiasedness under
TF32, and the stricter original comparison tolerances were not relaxed.

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Retain the shared transition repair | Independent formula and actual-consumer checks pass | Strict TF32 precision veto remains | Other compiler builds and rounding-sensitive trajectories | Preserve regression and compiler reproducer; test downstream distributional accuracy separately | Whole-filter TF32 equivalence or general GPU correctness |
| Keep safeguarded unweighted as an optional reference | Matches QR; all learners complete | Eight conditional heuristic vetoes | Accuracy ranking unresolved | Use for controlled diagnostics with explicit extension identity | Better method or default readiness |
| Keep weighted failures as evidence | 158/160 under frozen cap | Twelve observed heuristic losses and four incompletion vetoes | Weight concentration; finite cloud; slow stopping | Test a source-grounded fitting change on fresh partitions before another claim run | Rejection of iAPF as a research direction |
| Retain paper/TF integration gaps | Lower-level bridge established | Adaptive d>1 guard remains | Model/time/controller and Equation 15 differences | Specify and test the actual multidimensional adaptive call chain | Full paper replication, KDM, canonical LEDH, model score or HMC readiness |

| Inference question | Status |
|---|---|
| Hard veto screen | All report-integrity checks pass; the two frozen iteration caps and TF32 comparison failures remain visible. |
| Statistically supported ranking | None. Four estimable primary intervals contain zero; d80 is incomplete. |
| Descriptive-only differences | Likelihood errors, guide KL, heuristic losses, ridge counts and runtime do not establish superiority. |
| Default-readiness | No learned-reference or precision promotion. Existing GPU/TF32 direction is unchanged; only the algebraically identical shared repair is retained. |
| Next evidence needed | Actual multidimensional adaptive parity, a clearly identified paper-fitting procedure, fresh replication with more independent datasets, and downstream precision evidence. |

## Integrity, budget and terminal review

[Verification](verification.json) checked 413 source hashes, 272 input hashes
and 2,201 output hashes across 2,851 distinct files. All match; the three
frozen R references are unchanged. The final checked shared kernel matches
the repaired GPU attempts and all three current CPU test records. Scoped
`git diff --check` passes. Exact commands, environments, seeds, device
placement, memory growth, source snapshots, logs and output hashes are
preserved in each attempt manifest and indexed by [the run manifest](manifest.json).

The report's first attempt rejected its all-complete assumption, then was
repaired to preserve incomplete candidates and suppress incomplete accuracy
comparisons. One GPU comparison failed because a diagnostic Python literal
was rounded through FP32 before FP64 conversion; the input construction was
fixed and rerun without changing tolerances. Both failed attempts remain
preserved and charged. Earlier wrong-result and strict-precision failures
are preserved separately from successful localization.

[Accounting](budget.json): 7,458.685461 CPU worker seconds (2.071857 hours)
and 296.839526 GPU process seconds (4.947325 minutes), including failures,
checks and the conservative initial device probe. Remaining allocation:
45.928143 CPU hours and 47.917545 GPU hours. CPU worker time is a conservative
single-thread-worker bound; GPU parent wall time includes fixture preparation.
The first GPU attempt's full elapsed time is also charged to CPU because its
fixture split was not separately recorded. No old expired deadline is reused.

Terminal skeptical self-review: PASS for these bounded conclusions, with
no independent-review claim. The comparison preserves the actual R and TF
boundaries; no diagnostic becomes a promotion criterion; the safeguarded
fit's numerical validity is separated from its statistical quality; no
censored result is silently removed or repaired into the primary sample.
The strongest alternatives remain finite-data regression variability,
privileged oracle controls and sensitivity specific to this custom compiler
build. Wider untouched replication could reverse descriptive differences;
a counterexample to the repaired Gaussian update would invalidate its
engineering acceptance. Four datasets per dimension and unresolved author
choices are the weakest parts of paper-replication evidence.

The candidate failures did not invalidate the harness, model or iAPF idea.
The planned repair phases continued through completion. This bounded
mechanism program ends because its stated questions have been answered;
the unused allocation remains available for the master program's next
specified experiment. It does not require spending the remainder on
additional repetitions of already answered diagnostic questions.
