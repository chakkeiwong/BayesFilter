# Lorenz bounded covariance repair — 2026-10-01

**CLOSED after attempt126, 2026-10-02: no candidate admitted.**
See [terminal result](controlled-smc-lorenz-bounded-spd-result-20261001.md).
The specifications below preserve the executed hypotheses; dependent full
validations with no surviving candidate are ineligible.

Owner authorization: “do as you suggested”, following the proposed independently
validated positive-definiteness repair. This is part of the existing 48 GPU-hour
extension, not a new compute grant. The replication master remains authoritative.

## Question and evidence contract

Can a bounded covariance projection repair the saved invalid full-quadratic ADP
update while preserving healthy trajectories and agreeing with an independent
calculation throughout fitting and filtering? The comparator is the unchanged
centered-QR implementation, with independent full-rank SVD as the numerical
reference. Attempts 78/80 remain rejected; their finite outputs are not admission
evidence. Attempt95 reproduced invalid precision with the author's fitting body.

The claimed target is an OPTIONAL constrained controlled-SMC policy, not a
bitwise replication of unconstrained author code. Paper section6.2, equations
43–44 and footnote4 require an integrable Gaussian twist and allow an SPD
projection; the author full-quadratic branch at cSMC_Resample.m:213–239 does not
apply one. We do not claim our relative floor is specified by that paper or is
Higham's exact algorithm. The source observations, N=512, d=8, T=100, maximum
four refinements and 90% adaptive stopping remain unchanged.

For raw precision P=Q^{-1}+2(A_old+A_fit)=V diag(lambda) V', form the symmetric
inverse C=V diag(1/lambda) V'. For rho>0 set tau=rho max|1/lambda| and
C_rho=V diag(max(1/lambda,tau)) V'. Thus C_rho is the Frobenius projection onto
the set of symmetric matrices C>=tau I for this fixed tau: orthogonal
invariance reduces the squared distance to independent squared eigenvalue
differences, each minimized by clipping at tau. Its condition number is at most
1/rho. Convert P_rho=C_rho^{-1} and add (P_rho-P)/2 to A_fit. Raw b,c stay fixed;
the centered polynomial receives the exactly equivalent quadratic, linear and
constant correction. This changes the fitted policy and must be recorded.
Singular raw P is rejected, not inverted. Healthy P with relative margin>=rho
is untouched. No clipping changes an accepted unprotected baseline.

Primary admission criteria: finite and integrable complete recursion; independent
coefficient and fixed-input forward agreement at the existing tolerance
1e-7+1e-9*abs(reference); full-rank design; identical stopping and bitwise
unchanged healthy outputs when the protection never fires. Stable XLA signatures,
shared scalar/batch consumer wiring and trace counts are engineering criteria.
Rank, integrability, nonfinite, coefficient or consumer discrepancy veto a
candidate. They trigger the next declared floor/representation repair, not
rejection of controlled SMC. Corrupt inputs, a broken reference, exhausted budget
or an unexplained implementation mismatch stop dependent work for repair.

Diagnostics: activation count, eigenvalue margins, coefficient magnitude, projection
size, log-integral error and rank ratios explain behavior; likelihood variance,
ESS and runtime MUST NOT select the safeguard. No variance superiority, original
Figure6 replication, KDM completion, production/default or HMC claim follows.
Heuristic dominance is deferred to the downstream efficacy experiment: this
stage calibrates arithmetic/non-harm, not efficacy. APF and its three source
regimes (F=2.5,4.8801,8.5) remain mandatory downstream comparators.

## Defaults, assumptions and pre-mortem

* FP64, centered QR, full quadratic and fixed saved inputs: inherited checked
  reference lane; independently validated in77/92. Risk: conditioning and
  eigenvector sensitivity; compare identical coefficients before different fits.
* Historical rho=sqrt(eps64) is a rejected baseline. Candidate rho values
  1e-6,1e-4,1e-3 correspond to covariance condition limits 1e6,1e4,1e3.
  These are calibration hypotheses, not tuned defaults. Their justification is
  a measured accuracy/perturbation curve, not filtering performance. Take the
  smallest rho that passes the complete prescribed numerical and non-harm
  checks; if none passes, reject this family and inspect the first discrepancy.
* Retaining raw b,c preserves the existing candidate's defined projection but
  may cause large means/cancellation. Inspect local log integrals and the full
  forward consumer; finite SPD alone cannot admit the candidate.
* The known failure is calibration data. Reusing it cannot establish generality;
  freeze any surviving candidate before new seeds and downstream comparison.
* A run can pass finite-value checks while computing the wrong policy. Independent
  SVD, centered/raw equivalence and fixed-input Gaussian/weight checks prevent
  that interpretation. A later failure can reflect ill-conditioning rather than
  failure of the paper's idea. Preserve the first offending update.

## Executable phases and bounds

1. c1_lorenz_bounded_reference: save independent healthy and first failed
   regression coefficients from76, expected floor projections and Gaussian
   correction integrals. CPU only, <=30 s, at most2 attempts/45 s total.
2. c2_lorenz_bounded_calibration: identical-coefficient GPU/XLA checks against1,
   centered/raw polynomial equivalence, no-fire equality, trace and call-chain
   checks. <=120 s, at most2 attempts/180 s total.
3. c2_lorenz_bounded_safety: evaluate floors that pass2 on the healthy69 design
   and failed76 four-row full recursion. Retain all outcomes; no ESS selection.
   <=900 s, at most2 attempts/1500 s total. Save all histories and safety fields.
4. c1_lorenz_bounded_validation: independent SVD backward refits and Gaussian
   forward values for every surviving candidate, stage and row. <=45 s,
   at most2 attempts/45 s total. Freeze only the smallest fully passing floor.

Total this stage <=90 CPU seconds and <=1800 GPU seconds; attempt ceiling106
reassesses95 completed attempts and leaves room for localized repairs, without
enlarging the existing500 CPU/18000 GPU second tranche. Opening balances at95:
119.240781 CPU and3533.843673 GPU seconds in tranche; inherited372.407717 CPU
and116485.144024 GPU seconds. Dispatcher enforces the smaller available budget.
Every launch uses a fresh versioned attempt directory under
docs/plans/artifacts/controlled-smc-kdm-replication-20260928, source snapshots,
input hashes, exact command, environment, wall time and device provenance.

Run each phase with:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase NAME`.
GPU phases use trusted GPU access, memory growth and XLA; CPU reference phases
explicitly hide GPUs. Existing source code and rejected results are preserved.

## Skeptical pre-run audit

Reviewed against wrong baselines, proxy promotion, missing stopping conditions,
source claims, silent floors, holdout reuse, environment and consumer wiring.
PASS for a bounded repair experiment: the baseline remains unprotected, every
floor is explicit, numerical non-harm decides survival, both backward and forward
consumers are checked, and the output cannot close Figure6. The principal risk
is cancellation from preserving raw b,c; the local integral check and complete
reference are required before any fresh-seed efficacy run.

After this stage: record a decision/inference-status table and strongest alternative
explanation. If a floor survives, freeze it before untouched seeds, then run
conditional APF/heuristic comparisons, scalar/batch parity and a cost pilot under
a separate bounded continuation section. Do not launch the full Figure6 matrix
until correctness and remaining cost are known.

## Local execution repair

Attempt96 computed the calibration inputs, then failed serializing a NumPy boolean
in the reference summary. Converted the diagnostic predicate to a Python bool;
no numerical change. Charged2.619960 CPU seconds. Retry uses a new directory
within the same two-attempt/reference cap and total campaign budget.

Attempt98 hit an XLA-incompatible LogMatrixDeterminant operation introduced in
this diagnostic harness. Replaced logdet of the known SPD base precision by
2*sum(log(diag(Cholesky(P)))), the same mathematical quantity using the existing
XLA-compatible factorization. The shared runtime was unchanged. Full exception
text is now retained. Charged2.869241 GPU seconds; retry remains within180 s.

## Discriminating repair after99: XLA eigensolver accuracy

Attempt99 rejects every tested floor on IDENTICAL coefficients. At rho=.001 the
matrix error still reaches606 times the existing tolerance; healthy inputs remain
unchanged. Thus independent least-squares solvers cannot explain this difference.

Upstream TensorFlow v2.20.0 xla_self_adjoint_eig_op.cc:50–59 lowers eigh to
2*ceil(log2(n)) sweeps and tolerance1e-6, regardless of FP64 input. Local source
copy: .localresources/code/tensorflow-v2.20.0-xla-self-adjoint-eig-op.cc.
The installed binary is2.20.0-dev0+selfbuilt; exact build-source identity is not
established. Measure its residual directly before attributing its behavior.

Add c2_lorenz_bounded_eigen_calibration: compare the same input matrices with
the default eigensolver and XlaSelfAdjointEig(max_iter=100,epsilon=8*eps64).
The latter is an explicit OPTIONAL solver setting in the same shared helper,
not an eager/NumPy fallback. Its epsilon follows FP64 rounding; 100 is an iteration
cap, with acceptance determined by residual and orthogonality <=256*d*eps64.
Failed convergence rejects the fit. Record both residuals and exact consumer
wiring. Repeat all identical-input coefficient/integral/no-fire checks unchanged.
This changes numerical accuracy, not the defined spectral projection. Its
qualification remains non-harm and reference agreement, never ESS tuning.
At most2 attempts/180 GPU seconds; total bounded-stage GPU cap remains1800 s
(the safety ceiling1500 plus two calibration ceilings180 each is constrained by
that aggregate cap). One extra diagnostic phase does not enlarge the grant.
Only if this passes may safety run with that explicit frozen eigensolver setting.

Attempt100 encountered a generated-edit literal-newline syntax defect before
numerical work. Corrected it and syntax-checked all five touched files before
retry; scientific settings and budget are unchanged.

## Residual repair after101

Tightening the exposed XLA tolerance did NOT change the result: maximum measured
eigen residual7.83336e-9, orthogonality1.33e-15, no floor admitted. The local
TensorFlow build source was found at /home/chakwong/tf_dual_gpu_build/tensorflow.
Its XLA eigh_expander.cc:130–136 uses0.1*float32 epsilon to suppress small Jacobi
rotations regardless of input dtype. The same code is present in the saved
upstream xla-eigh-expander.cc. Thus exposing tolerance alone cannot guarantee
FP64 accuracy. This is measured eigenpair inaccuracy, not retracing or NumPy in
the runtime. Exact binary/source revision identity remains a provenance limit.

Add a native TensorFlow residual refinement, explicitly optional: begin with the
XLA eigenbasis V, form B=V'PV, and apply symmetric Jacobi rotations to B and V
until its relative off-diagonal Frobenius norm <=8*eps64 or at most20*d*d
rotations. For pivot(p,q), tau=(Bqq-Bpp)/(2Bpq), t=sign(tau)/
(abs(tau)+sqrt(1+tau^2)), c=1/sqrt(1+t^2), s=t*c, and
R=I+(c-1)(ep ep'+eq eq')+s(ep eq'-eq ep'). Update B=R'BR,V=VR.
This orthogonal similarity preserves the matrix; after sorting the diagonal,
accept only residual and orthogonality <=256*d*eps64. A zero pivot is a no-op.
The procedure is batch-native, uses tf.while_loop, has no pfor/NumPy callback,
and retains explicit stable XLA signatures. It changes the eigensolver's
accuracy, not the defined covariance projection. The full identical-input
and non-harm criteria are unchanged.

c2_lorenz_bounded_refined_calibration uses the same frozen9 cases and independent
reference, plus matrix scales and repeated eigenvalues for a direct residual
check. At most2 attempts/180 GPU seconds. All bounded phases together retain
the1800 GPU-second cap. Rejected raw-eigensolver evidence is preserved.
A new candidate failure remains a repair trigger; no full recursion starts
without a numerical survivor.

## Attempt104 failure and bounded localization
The 0.001 candidate passes fixed-coefficient checks and finishes the frozen
recursions, but independent full backward checks fail six coefficient checks
in row0 refits3/4. Forward fixed-input checks and rank checks pass. This vetoes
admission, not continued numerical repair. Do not loosen the 1e-7+1e-9|reference|
criterion. c1_lorenz_bounded_localization records every backward-time error and
projection, correction/target scale and singular ratio for the same saved
inputs. One attempt, at most30 CPU seconds, inside the unchanged90/1800 repair
caps and tranche. No new random data or scientific promotion.
Skeptical audit: the comparator re-solves the full backward sequence by SVD
on the saved TF cloud, so it can distinguish a first projection discrepancy
from propagation; it does not by itself separate every floating-point cause.
The next repair requires an algebraic derivation and fixed-input checks.

## Algebraically equivalent projected-integral repair (after105)
Attempt105 locates the first failure at time11 immediately after projection at
time12 (refit3), and at time12 after projection at13 (refit4). The regressions
remain full rank. Keep the frozen0.001 floor and the same raw polynomial.
The candidate changes its evaluation only; no floor tuning or tolerance change.

Write the unprojected fitted polynomial as q0(x)=z'Az+b'z+c,
z=(x-mu)/s, and the projection addition as x'Dx. Under the old twisted
Gaussian N(m,C), let P=C^-1, H=A/(s s'), h=b/s-2H mu,
S=P+2(H+D), K=S^-1, and n=K(Pm-h). Completing the square gives
log E exp[-q0(X)-X'DX] =
0.5(log|K|-log|C|)-q0(n)-n'Dn-0.5(n-m)'P(n-m).
This follows by evaluating the combined quadratic at its minimizing mean n;
its Hessian is S. It equals the existing Gaussian formula exactly but avoids
expanding x'Dx into enormous centered linear and constant terms. Preserve
unprojected centered coefficients plus D through the backward loop. When D=0,
call the existing formula to preserve healthy output.

Assumption audit: this is an optional arithmetic representation, not a new
projection or author implementation claim. The original eigenbasis repair,
floor, QR objective, target, data and comparator remain frozen. Cancellation
within q0(n), ill-conditioned S, or a wiring error can still veto it. Test the
identity first against80-decimal mpmath Gaussian integration by completing the
square in raw coordinates on the exact frozen coefficients; then run shared
GPU/XLA kernels, healthy bitwise checks and independent entire recursions.
The high-precision reference is diagnostic-only; no runtime dependency.

Four added phases: c1_lorenz_bounded_integral_reference (30 CPU seconds),
c2_lorenz_bounded_integral_calibration (120 GPU seconds),
c2_lorenz_bounded_integral_safety (900 GPU seconds),
c1_lorenz_bounded_integral_validation (45 CPU seconds). Each allows at most2
attempts, but aggregate90 CPU/1800 GPU repair caps and the remaining tranche
override per-phase limits. Reassess attempt ceiling106 to112 to accommodate
bounded repairs; this creates no new compute allocation.
Promotion still requires all1e-7+1e-9|reference| checks, finite valid outputs,
full rank, eigen residual/orthogonality, one trace and healthy bitwise identity.
No efficacy, replication, HMC or default claim. Candidate failure triggers
localization; corrupted artifacts, exhausted caps or mathematical mismatch
stop dependent execution.

Skeptical pre-run review: a reference using the same expanded polynomial could
share the cancellation, so an independent high-precision raw-coordinate
Gaussian expression is required before the full run. Fixed-input success is
only a necessary condition; full backward parity remains required. The
comparator is unchanged in mathematical target. No heuristic ranking is made.

## Attempt109: partial arithmetic repair, full admission still vetoed
The new integral passes80-digit fixed-input checks and removes all refit3
coefficient failures. Full refit4 still fails A/b/c (max absoluteA0.05851).
All other checks pass. c1_lorenz_bounded_integral_localization repeats the
per-time diagnostic on108 with the same stable reference. One attempt<=30 CPU
seconds inside unchanged caps. A mismatch-free stage is a valid diagnostic
outcome, not an infrastructure failure. The next discriminating check will use
the earliest mismatch, not increase the tolerance or change the floor.

## Discriminating local-input crossover after110
Attempt110 places the remaining refit4 discrepancy at time12 after projection13.
The singular ratio at12 is0.000747: full rank but less well conditioned. Separate
three possible causes before another repair: (a) Gaussian evaluation at fixed
coefficients; (b) QR on an identical response; (c) propagation of small changes
through the projected recursion. Pure observability adds the unprojected
centered coefficients, fitted coordinates, projection addition and correction
integral to the shared fit result. The original consumer may discard them;
a bounded diagnostic saves them from its exact shared call chain.
c2_lorenz_bounded_crossover_capture <=120 GPU seconds, one attempt; compare
captured A/b/c with the saved full-controller fit, then save row0 refit4.
c1_lorenz_bounded_crossover_reference <=30 CPU seconds, one attempt; on identical
inputs compare each SVD regression and correction integral, including80-digit
raw-coordinate Gaussian checks at the earliest failing boundary. Artifacts
separate local computation from recursive sensitivity. Existing tolerances,
floor, total90/1800 caps and tranche unchanged. These checks cannot promote a
candidate or rank filters; they choose the next justified arithmetic repair.
Skeptical review: consumer parity must pass before interpreting captured
internals; independent SVD must consume exactly the saved GPU target.

## Reference arithmetic audit after112
The shared consumer capture111 agrees with the full controller. In112, every
same-target SVD/QR fit and same-increment Gaussian integral passes. At time11
the80-digit oracle exposes maximum cancellation error1.39e-6 in the NumPy
expanded formula versus3.33e-8 in the GPU formula (both inside local tolerance).
The reference must use the independent completed-square evaluation for all
increments, including D=0, then verify it against the same80-digit check before
rerunning full parity. This changes neither the mathematical target nor the
acceptance tolerance and does not automatically absolve the candidate.
c1_lorenz_bounded_reference_arithmetic: <=30 CPU seconds, at most2 attempts,
within the unchanged90 CPU/1800 GPU stage caps; ceiling114, unchanged tranche.
It repeats the local checks/high-precision oracle and the entire independent
backward/forward comparison of108. Full comparison is still decisive.
Skeptical review: a locally more accurate formula can still leave recursive
sensitivity. Preserve109 as rejected evidence and accept only a complete pass;
no fresh-seed or full Figure6 launch on local checks alone.

Attempt113 completed local checks but failed to import the full-validation wrapper from its source snapshot. Added the missing snapshot dependency. No mathematical change; retry114 remains inside the two-attempt/45-second phase and unchanged aggregate caps.

## Next conditioning rung after114
The0.001 candidate remains rejected:114 passes every fixed-input check and the
80-digit oracle, but the full recursion fails row0/refit4 A,b,c (max abs A0.0989).
Two independently evaluated recursions are sensitive despite locally accurate
kernels. This is evidence to test a stronger conditioning bound, not authority
to relax coefficient tolerance. It does not yet prove the sole error source.

Next fixed calibration grid:rho=0.01 and0.1, giving covariance condition bounds
100 and10 instead of1000. These are explicit new Class-C hypotheses in the same
spectral projection family, with provenance in the failed1000-condition rung.
Use the same frozen nine cases and80-digit raw Gaussian reference. The smaller
passing rho is eligible only if all healthy coefficients and complete healthy
histories remain bitwise unchanged. If0.1 activates on a healthy fixture it
fails immediately. No likelihood variance, ESS, runtime or failed claim holdout
selects rho; all saved designs here are numerical-repair calibration only.
The mathematical integral representation and eigen refinement remain fixed.
A stronger floor changes a failed policy; it is not an unprotected source replay.

Phases c1_lorenz_bounded_condition_reference (<=30 CPU seconds),
c2_lorenz_bounded_condition_calibration (<=120 GPU seconds),
c2_lorenz_bounded_condition_safety (<=900 GPU seconds),
c1_lorenz_bounded_condition_validation (<=30 CPU seconds), at most2 attempts
each, aggregate90 CPU/1800 GPU stage caps unchanged. Reassess ceiling120 without
new compute. Existing tranche and inherited-grant limits still apply.
Skeptical review: an overlarge floor can appear stable by changing valid fits;
the existing no-fire/bitwise comparator prevents that. Even a passing repair
still needs fresh seeds, conditional APF controls and a cost pilot before any
Figure6 claim. If neither rung passes, preserve the rejection and diagnose the
projection family's conditioning rather than declare the paper invalid.

## Feasibility-trigger repair after118 (2026-10-02)

Attempts115/116 pass fixed-input and80-digit checks at rho=0.01.
Attempt117 rejects it: three activations change a healthy full trajectory.
Attempt118 performs zero full validations because there is no survivor.
Experiment completion is not candidate acceptance.

Separate activation from correction strength. Let m=lambda_min(P) /
max(abs(lambda(P))) for the proposed precision. Activate only when
m<=64*epsilon, the existing shared covariance validity boundary; then retain
the independently derived covariance projection at rho=0.01 (condition100).
Every numerically valid proposed fit is unchanged. This is an explicit Class-C
repair hypothesis, not the author's unconstrained full-quadratic code and not
a new default. Preserve prior coupled-trigger candidates.

Question: does feasibility-only activation preserve healthy trajectories and
remove full recursive parity failure? Comparators: unprotected healthy
controller and independent SVD/eigensolver implementation of this candidate.
Primary criterion: all full backward coefficients and forward checks within
unchanged1e-7+1e-9|reference|. Promotion vetoes: any healthy activation/change,
nonfinite value, invalid covariance, rank or eigen-residual failure, unmatched
call chain or retracing. Above/at/below64epsilon cases test activation.
Fixed-input80-digit Gaussian checks precede full execution. ESS, likelihood
and runtime are explanatory only. No ranking, full replication, HMC, default
or superiority claim follows. Heuristics remain a later efficacy gate.

Defaults audit:64epsilon comes from the existing validity guard, not efficacy
tuning. rho=0.01 is the tested condition100 hypothesis, not a claimed optimum.
Near-zero rounding may change activation; boundary checks and independent
recursion can veto it. Retained eigen refinement and stable integral have
fixed-input evidence in102/106/112. The failed design is repair calibration,
never a fresh efficacy holdout.

Execute c1_lorenz_bounded_trigger_reference (<=30 CPU seconds),
c2_lorenz_bounded_trigger_calibration (<=120 GPU seconds),
c2_lorenz_bounded_trigger_safety (<=900 GPU seconds), then
c1_lorenz_bounded_trigger_validation (<=30 CPU seconds), at most2 attempts each.
Use unique dispatcher attempt roots. After118 approximately32.17 CPU/1557.10
GPU seconds remain in the unchanged90 CPU/1800 GPU stage. Reassess ceiling126;
no additional compute. Corrupt artifacts, changed target, mathematical
mismatch or exhausted caps stop dependent execution. Local failures permit
bounded retry; candidate rejection permits diagnosis, not admission.

Skeptical audit passed: the optional repair semantics change explicitly in
both independent reference and actual controller. Finite-only success and a
zero-check validation cannot pass. A local oracle does not prove recursive
agreement. No tolerance is relaxed; no Figure6 run starts before numerical
and later cost gates pass.

Attempt121: feasibility-only activation passes healthy bitwise identity with
zero activations, but the failed design becomes nonfinite in refit4 despite
full-rank regressions and small eigen residuals. No candidate survives.
Before another parameter hypothesis, execute c1_lorenz_bounded_trigger_history
(<=10 CPU seconds, max2 attempts, unchanged stage cap) to locate the first
nonfinite saved field and nearby state magnitudes. This is explanatory
inspection only, not promotion. Full reference validation remains ineligible.
The safety harness's generic REJECTED_NONHARM_SCREEN label includes numerical
failure; here the healthy non-harm checks themselves all passed.

Attempt122 locates nonfinite weights at refit4/time0 and nonfinite states at
time1, row0. The initial cloud reaches637.05 despite finite coefficients and
positive-definite covariance. Next c1_lorenz_bounded_trigger_transition
(<=12 CPU seconds, max2, same caps) evaluates unchanged author RungeKutta4.m
and the independent NumPy equation on the previous/current initial clouds.
Require matching finite masks, finite-value agreement at1e-10+1e-12|reference|,
finite previous-refit control, and a nonfinite current refit. This is an
explanatory cause check, not an efficacy comparison. No observation model,
RK4 step size or simulation target is changed.

## Final bounded conditioning candidate after123

Both unchanged author RK4 and the independent equation become nonfinite on all
512 particles of refit4's initial cloud; the previous-refit control is finite.
Positive covariance alone does not bound the proposal mean or keep fixed-step
RK4 in a numerically stable region. This is a failed repair, not evidence
against the Lorenz model or paper.

The last candidate in this stage activates at margin<=0.001 (condition1000)
and repairs with floor0.01 (condition100). The activation threshold has healthy
no-fire evidence from103/108; the stronger correction has fixed-input oracle
evidence from115/120. Their combination is a new Class-C hypothesis, not a
default or source-identical operation. It is chosen by numerical non-harm and
accuracy, never ESS or likelihood. It still does not bound the mean, so the
full finite forward check and independent backward parity are mandatory vetoes.

Phases c1_lorenz_bounded_margin_reference, c2_lorenz_bounded_margin_calibration,
c2_lorenz_bounded_margin_safety, c1_lorenz_bounded_margin_validation retain
limits30 CPU/120 GPU/900 GPU/30 CPU seconds and max2 attempts each. The existing
90 CPU/1800 GPU aggregate overrides them; after123 approximately19.09 CPU and
1497.60 GPU seconds remain. Ceiling130 adds no compute. Preserve the rejected
positivity-trigger evidence. After this last candidate, close the bounded
stage with its verdict; any new safeguard family requires a distinct repair
specification using the still-authorized campaign balance.

Skeptical audit passed: no-fire support is measured, not universal. A mean/
forward failure vetoes the candidate even if precision improves. Same
tolerance, source, observation model, RK4, random design and stopping rule.
Full independent recursion remains decisive. No efficacy or replication claim.
