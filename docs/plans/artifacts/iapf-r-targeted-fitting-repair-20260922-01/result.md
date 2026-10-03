# Targeted repairs complete the eight diagnosed d80 failures

The repaired diagonal-guide learner completes all eight original d80 failures
after one localized optimizer repair. It also completes all six preassigned
fresh-data runs, alongside six completed QR comparators. This closes the
demonstrated execution and numerical failures for these cases. It does not
establish an accuracy advantage: the repaired learner has larger observed
terminal error than QR in three of the six fresh comparisons, and its guide
shape is close to the unweighted regression on the examined heldout regions.
The conditional heuristic checks therefore veto promotion. Full paper
replication remains unresolved.

This is an independent CPU R reference extension. Its weighted-log objective
and numerically calibrated ridge are explicitly different from GJL's printed
Equation 15. The original author implementation/settings remain unavailable.
No TensorFlow/GPU, LEDH, KDM, score, HMC or production default changed.

## What changed, and what the experiment actually tested

The master program's source/code diagnosis separated three mechanisms: a QR
prerequisite that prevented warm starting, an absolute-density objective that
rewards disappearing density, and weighted diagonal regression that amplifies
omitted cross terms. The present plan tested those mechanisms in that order.
The source derivations and paper anchors remain in
`../iapf-r-root-cause-audit-20260922-01/result.md`.

The optional `f2_independent_strict` and `f2_independent_loose` arms in
`docs/benchmarks/reference_iapf_author_choices.R` now obtain their optimizer
anchor, coordinate scales and fixed density/loss scales from the previous
valid Gaussian. They never call QR in that case. When no previous Gaussian
exists they retain the original QR dependency. Existing arm outputs were
checked against the preserved source and remain identical on the fixtures.

All 25 previously blocked cases reached the optimizer under both tolerances,
and all 50 returned through the existing numerical guards. That is an execution
repair, not successful guide learning: shape error worsened in 18/25 cases
under each tolerance; the median final shape residual was 0.9999998. The fixed
Equation 15 loss became small without useful proportionality to the target.
The prior guard catches complete density underflow, but finite small densities
can still exploit the same objective geometry. Increasing iteration budgets or
using previous fits does not itself remove that mathematical weakness.

For the weighted-log extension, standardized points give a design
`D=[1,z,z²]`, centered log targets `y`, and normalized weights
`w_i ∝ exp(y_i)`. The repaired fit solves

\[
 \min_\beta\;\frac12\|A\beta-b\|^2
        +\frac{\lambda}{2}\|\beta-\beta_0\|^2,
 \qquad A=W^{1/2}D,\quad b=W^{1/2}y,\quad q_j\leq q_{\max}<0.
\]

Here `beta0` is the unweighted SVD regression on the same bootstrap learning
cloud, and `q_j` are the squared-coordinate coefficients. Negative curvature
ensures a positive Gaussian variance. The bound uses the explicit precision
reference and relative roundoff rationale recorded in the plan; it is not a
parameter tuned for likelihood accuracy.

The ridge is likewise calibrated to the numerical system, not its likelihood
result. If `smax,smin` are the extreme singular values of A, then

\[
 C=\epsilon^{-1/2},\qquad
 \lambda=\max\!\left(0,\frac{s_{\max}^2-C s_{\min}^2}{C-1}\right)
\]

bounds the Hessian condition number by C. Exactly representable diagonal
responses retain their unweighted solution; a feasible unconstrained solution
is returned unchanged by the constraint-only arm. The ridge changes the
statistical fit where it is positive and remains an explicit extension.

## Saved-cloud and heldout-region results

The 40 saved records contain 32 distinct point/target inputs, not 40 independent
replications. Every record was preserved and checked under all five arms.
These are the original solver results before the later active-set fallback:

| Fitting arm | Numerical acceptance /40 | Exact-diagonal controls failing 1e-6 | Largest scaled response to the fixed tiny target perturbation |
|---|---:|---:|---:|
| Unweighted SVD |40|0|3.43e-10|
| Half-weight SVD |40|0|3.63e-9|
| Target-weighted SVD |34|6|7.15e-9|
| Target-weighted, curvature bound |36|6|5.38e-3|
| Target-weighted, calibrated ridge and bound |40|0|3.96e-10|

The bound-only solver's sensitivity is a numerical finding, not evidence of
superior statistical regularization. Its original rejection records remain
unchanged. All 40 ridge solutions were interior to the curvature constraint:
the ridge toward the unweighted fit, rather than active clipping, provided the
repair on these saved clouds.

Fresh predictive and smoothing draws were generated independently of those
training particles. The following are centered log-shape RMS errors on the
two checked smoothing regions; the full conditional tables and Gaussian KL
values are in `attempt03-solver/results/heldout-shape.csv`.

| Guide | Saved replicate 1 | Saved replicate 3 |
|---|---:|---:|
| Constant |5.743|5.786|
| Observation only |1.971|1.999|
| Unweighted fit |1.697|1.626|
| Repaired weighted fit |1.740|1.632|
| Analytic moment-diagonal |0.621|0.679|
| Analytic precision-diagonal |0.638|0.690|
| Exact full Gaussian |0|0|

The analytic diagonal guides have much smaller error on these points. Thus
the diagonal family can represent a substantially better guide than the
current learned approximation here. The missing interactions are real, but
they do not by themselves explain all the learned-guide error. These two
purposefully selected clouds do not establish population performance or a
ranking of fitting procedures. Exact-diagonal controls and analytic gradients
provide numerical checks; they are not scientific accuracy evidence.

## Full filter and complete learning checks

The full Gaussian guide agrees with the Kalman terminal log likelihood in all
twelve analytic runs. With 1,000 particles, both analytic diagonal projections
have absolute terminal log errors below 0.848 across the four d80 runs. The
observation-only errors range from -12.744 to -2.523, and the constant-guide
errors from -3552.152 to -3100.962. These are descriptive conditional controls,
not estimates of average performance from a sufficiently large sample.

The preassigned complete-learning experiment used two new data sets at each
of d=5, 20, 80, with T=100 and N0=1,000. Both QR and the frozen repaired learner
completed every data set:

| Dimension | Data seed | QR terminal log error | Repaired terminal log error | Repaired final particles |
|---|---:|---:|---:|---:|
|5|92210005|+0.06669|-0.00489|1000|
|5|92220005|+0.02688|+0.04971|1000|
|20|92210020|-0.08130|-0.29977|1000|
|20|92220020|-0.13665|-0.04378|1000|
|80|92210080|+0.16098|+0.13027|2000|
|80|92220080|+0.01064|-0.10178|2000|

Three absolute-error comparisons favor each fitted method descriptively; no
ranking is statistically supported. The conditional heuristic ledger records
nine underperformance observations across the declared terminal controls.
An example is d=20, data seed 92210020, where the repaired error exceeds QR and the
observation-only guide. The model-specific analytic guides also use privileged
information, and some comparisons have different final particle counts and
learning costs. The ledger exposes those differences; it does not claim a
matched-cost performance comparison. Its promotion veto remains in force.

The subsequent bridge check ran the unchanged weighted-log consumer on both
original d80 data sets and four original seeds per data set. All eight failed
at exactly the saved time, with exactly matching failure points and targets.
The repaired consumer was checked to begin with the same particle cloud.
Seven completed immediately; one failed later at iteration 4, time 72 because
L-BFGS-B reached its iteration limit with scaled KKT residual 1.508612e-6.
That failure was retained as `attempt22-bridge`.

The localized repair adds a direct SVD active-set fallback only when the
constrained optimizer fails its unchanged KKT criterion. It solves the same
convex objective with the same bounds. A small exhaustive active-set/QR oracle
agreed with the implementation. The saved failed fit reached KKT 4.28e-14 in
two active-set iterations; its full identical-seed retry then completed at
iteration 11 with 2,000 particles and terminal log error +0.00327.

All eight original cases therefore complete with the final repair. Their
terminal log errors range from -0.63124 to +0.45019. This is conditioned on the
known failures and is not untouched validation. The authoritative eight
records and the preserved nine-attempt history are separate CSV files in
`report-v1`. The recorded 9,800 fits from the fourteen successful repaired
learning runs have maximum KKT 5.83e-13; exactly one used the active-set fallback.

## Verification, decisions and remaining evidence

Forty-three focused checks cover gradients, unchanged comparator outputs,
healthy-fit preservation, actual warm-start/controller wiring, active-bound
behavior, the exhaustive convex-solver oracle and the saved failed fit.
The existing author-choice regression suite passes. Recorded-output checks
verify all saved bounds and the derived Hessian condition cap. Across all
attempts 464 check evaluations passed; this count includes repeated regression
checks and is not 464 independent scientific tests. Source, input and output
hashes were rechecked during report construction.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept independent warm-start wiring |50/50 optimizer entries, zero QR calls |Shape remains inadequate; promotion veto |Different objective/constraints could help |Preserve repaired wiring; do not expand raw Equation 15 runs unchanged |Warm starting alone repairs learning |
| Accept localized numerical repair |Exact controls, KKT checks, 8/8 same-seed full retries complete |Original optimizer failure preserved and repaired |Unseen data may expose other failures |Freeze the explicit reference candidate for independent validation |General correctness from eight cases |
| Retain both learned fits as reference candidates |6/6 fresh completions each |Conditional heuristic underperformance |Few seeds, learning approximation and cost |Larger paired validation with all baselines |A statistically superior or default-ready method |
| Keep paper replication open |Paper-scale repeated evidence absent for this extension |Author identity/objective differ |Unrecovered original numerical choices |Maintain explicit paper-versus-extension separation |Reproduction of the published numerical procedure |

| Inference status | Finding |
|---|---|
| Hard veto screen |Original QR/rank/curvature failures reproduced; one later optimizer failure repaired without relaxing criteria |
| Statistically supported ranking |None |
| Descriptive-only differences |Terminal errors, shape RMS/KL, perturbation response, timing and conditional comparison tables |
| Default readiness |No promotion; heuristic underperformance remains a veto |
| Next evidence needed |Independent paired replications across all five dimensions, uncertainty intervals for likelihood error, and matched-accuracy cost |

Post-run skeptical review: the strongest alternative explanation for the good
completion rate is that the ridge mostly preserves the unweighted guide, which
already works on these fresh cases. The heldout shape comparison supports
taking that explanation seriously. Analytic diagonal controls show room to
improve learning but cannot establish that a general learner can achieve it.
The weakest evidence is the small number of complete runs and deliberate
conditioning of the bridge on known failures. New failures, uncertainty-aware
comparisons favoring simple controls, or a verified original-author routine
would change the next research decision. The demonstrated implementation
repair stands independently of those unresolved performance questions.

Numerical execution used 494.012354 aggregate worker seconds in 29 launches,
within the 4,800-second/30-launch cap and original deadline. No timeout or
infrastructure launch failure occurred. The candidate failure and automatic
repair are explicitly preserved. The remaining prior allocation is 4,513.601570
seconds; cumulative campaign use is 110,076.194834/172,800 seconds. No worker
remains active. The exact commands, CPU-only environment, R version, seeds,
source snapshots, logs and hashes are in per-attempt manifests and the root
manifest. The next action is recorded in the master and concise checkpoint.
