# iAPF fitting failure: paper, mathematics and executed code

> Source correction,27September2026: the author's thesis now identifies
> R optim/BFGS and zero explicit penalty for the PartIII state-space examples,
> and explicitly discusses the degeneracy derived here. Unknown-optimizer
> statements below are superseded by the
> [source recovery](../iapf-thesis-reconstruction-20260927-01/source-recovery.md).
> The saved numerical failures and derivation remain unchanged; they concern
> our reconstruction, not a demonstrated failure of the author's procedure.

Completed 27 September 2026. Scope: the four saved d=5, T=100, N=1000
Equation (15) failures. Both scientific replications remain incomplete.
This audit concerns iAPF; it establishes no new KDM/MDPF failure diagnosis.

## Verdict

The current fitting route follows an escape direction in the paper's stated
objective: Gaussian density and the fitted target multiplier collapse together,
so the squared error falls without a better proposal shape. This mechanism is
supported by derivation and by the saved optimization trajectories. Independent
checks found no target-construction or derivative error on these inputs.

The mathematical fitting prescription is known. The practical procedure that
made the authors' simulations work is not fully recovered: the inspected source
does not specify the optimizer, numerical-fit initialization, parameter-domain restrictions or
the exact positive floor function. Original author code and the final publisher
version remain unverified. This is a failed reconstruction of those practical
choices, not evidence that every implementation of iAPF fails.

## What the paper says, and which code ran

The locally held accepted manuscript specifies backward targets in Algorithm 3,
then density least squares in Equation (15), followed by adding a positive
constant in Equation (16). It explicitly uses a diagonal covariance matrix and
reports successful simulations. Its discussion motivates the constant as tail
protection. We retain those statements; the global-minimization issue below is
a separate derivation for the displayed objective.

Source anchors:
[Algorithm 3](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt:656),
[Equations (15)–(16)](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt:911),
the diagonal restriction at line 943, and the Section 5.2 model at line 1023.
The public [arXiv record](https://arxiv.org/abs/1511.06286) was also checked.
No retrieved source closes the numerical-choice gaps above.

The actual call chain is:

1. `diagnose_iapf_scale_aware_reconstruction.R:143` generates the Section 5.2
   data and constant-twist particle clouds, then calls `sa_backward`.
2. `reference_iapf_scale_aware_fit.R:97` visits times in reverse order.
   `reference_iapf_paper.R:334` builds the target from the observation density
   and the integral of the next twist. Its line 53 evaluates the Gaussian
   convolution plus the positive constant.
3. `reference_iapf_paper.R:196` supplies a log-quadratic initialization.
   `reference_iapf_scale_aware_fit.R:7` profiles out lambda and constructs
   the residual and Jacobian; line 44 runs the scale-aware LM fit.
4. Line 56 checks the relative gradient. Lines 61–74 accept raw-loss descent.
   Line 81 checks curvature after a stationary candidate; line 92 constructs
   the positive-floor twist. Lines 112–119 save and reject a failed fit.

The sources actually executed are frozen in
[attempt01/sources](attempt01/sources). The diagnostic reads the historical
attempt03 source and six input RDS files after checking their hashes. It verifies
the original clouds and independently reconstructs all twelve targets at times
100, 99 and 98. This is executable call-chain evidence, not an inference from
the existence of a fitting function.

## Why decreasing this loss can make the fit worse

Write the Gaussian density values on the finite cloud as a vector p, and the
positive backward-target values as y. Equation (15) minimizes

`L(m,V,lambda) = sum_i (p_i(m,V) - lambda*y_i)^2.`

For fixed Gaussian parameters, differentiation in lambda gives

`lambda* = (p' y)/(y' y) > 0,`

and substitution gives

`L* = ||p||^2 - (p' y)^2/(y' y) = ||p||^2 * S,`

where `S = 1 - (p' y)^2/(||p||^2 ||y||^2)` measures disagreement in shape
on these points. Profiling lambda is therefore correct for the stated
objective. The implementation divides the sum by N, which does not change its
minimizers.

On any finite cloud, spreading a normalized Gaussian indefinitely, or moving
its mean sufficiently far away, makes all p_i approach zero. Lambda* also
approaches zero, and L* approaches zero regardless of whether S approaches zero.
If no finite Gaussian is proportional to the sampled target, the unrestricted
problem has infimum zero without a finite attaining minimizer. This establishes
global degeneracy; it does not rule out useful finite local minima.

The code uses changing numerical scales to evaluate this descent without
underflow and has overflow guards. Those guards do not prevent the observed
finite-parameter escape. Its relative-gradient criterion correctly rejects the
result. An ordinary absolute-gradient or optimizer-success flag can instead
mistake small amplitude for convergence.

For seed 9262001 and the N^-1 floor, the saved trajectory starts with log mean
raw loss about -28.90 and shape residual 0.00151. The final mean raw loss has log
-1621.51 while the shape residual is 0.00837. Shape initially reaches about
0.0000129 and then deteriorates. The fitted variance reaches about 277,532.
These are numerical observations of the derived escape mechanism, not useful
convergence.

## Why the recursive targets are difficult

With Gaussian transition covariance Q and next twist
`N(z;m,V) + c`, the backward target is

`h_t(x) = g_t(x) * [N(Ax;m,Q+V) + c].`

For this model, C=R=I. The no-floor product is a Gaussian with covariance W
satisfying

`W^-1 = I + A' (Q+V)^-1 A,`

and mean `W [y_t + A' (Q+V)^-1 m]`.
A dense A produces off-diagonal precision terms even when V is diagonal.
With c positive, the complete target is a mixture of this product and
`c*g_t(x)`. The floor is already added to the terminal twist, so both t=99
and t=98 targets are mixtures.

At t=100 the target is exactly a diagonal Gaussian. At t=99 and t=98, fitting
log-target values to the full diagonal-quadratic design leaves RMS residuals
about 0.19–0.23 and 0.22–0.26 respectively. Exact proportional representation
on these clouds is therefore unavailable to numerical precision.

Mismatch alone does not explain why t=99 succeeds and t=98 fails. At t=98,
the concentration statistic `(sum y_i^2)^2 / sum y_i^4` is about 1.26–5.98
out of 1,000, compared with 11.5–26.2 at t=99. This shows that a few large target
values dominate the squared-target scale. It is explanatory only: it is not
Jacobian rank, a literal count of usable observations, or proof that
concentration independently causes failure. The exactly representable terminal
target also has high concentration and is successfully fitted.

## Discriminating checks and interventions

One structured CPU/R/FP64 diagnostic completed successfully. All 48
source-cloud, target, Gaussian-product, gradient and Schur-complement checks
passed. The largest analytic-identity error was 1.78e-14; the largest
finite-difference error was 2.39e-8. Thresholds were 1e-10 and 1e-5 respectively.

Each intervention uses the same saved particles and starting parameters.
Pass means the existing relative-gradient threshold 1e-6 and curvature check;
it does not mean full-filter qualification.

| Target and fit | Cases passing | What it establishes |
| --- | ---: | --- |
| Actual target, saved profiled LM | 0/4 | Replays the current failure |
| Remove the floor, same profiled LM | 0/4 | The floor is not necessary for failure |
| Exactly diagonal Gaussian target, same profiled LM | 4/4 | The code can recover an exactly representable target on the same clouds and starts |
| Actual target, independent profiled BFGS | 0/4 | Changing to this optimizer does not repair the four cases |
| Actual target, joint Gaussian/lambda LM | 0/4 | These failures are not removed by jointly optimizing lambda |

Three BFGS runs reported their own successful termination, yet failed the
independent stationarity screen. The fourth reached its budget. An optimizer
return code is consequently insufficient evidence here. The BFGS `iterations`
CSV field contains function evaluations as returned by R, not a universal
iteration count; its optimization limit was 800 iterations.

The Schur check establishes equivalence of the undamped joint and profiled
Gauss–Newton elimination at the checked starts. It does not assert identical
damped trajectories. The alternative solvers are bounded probes, not an
exhaustive search over algorithms or initializations.

Files:
[checks](attempt01/checks.csv),
[geometry](attempt01/geometry.csv),
[solver probes](attempt01/solver_probes.csv),
[saved trajectory](attempt01/saved_trajectory_excerpt.csv),
[run manifest](attempt01/manifest.json).
Earlier exact-twist/Kalman, direct/profiled-objective and rescaling controls are
preserved in the [previous result](../iapf-scale-aware-reconstruction-20260926-01/result.md);
the current run does not relabel them as newly executed.

## Cause classification and remaining uncertainty

| Candidate cause | Verdict |
| --- | --- |
| Free amplitude plus non-proportional target | Confirmed global escape mechanism and observed trajectory |
| Target outside diagonal-Gaussian family | Confirmed on the checked clouds; not sufficient alone to explain failure time |
| Raw-loss LM acceptance | Permits and follows the escape; it decreases its stated objective |
| Positive floor | Changes the target but is not necessary for the observed failures |
| Particle-cloud concentration and initial basin | Plausible contributors; concentration measured, their individual causal effects not isolated |
| Incorrect target or derivatives | No defect found by the independent checks on these inputs |
| XLA, GPU, retracing, Python or NumPy | Not the cause of this reproducer, which fails independently in R/FP64 on CPU |
| Absence of every useful finite local fit | Not established |

The R reference has loops, as expected for this independent diagnostic.
That does not identify a TensorFlow performance fault. This audit does not
reassess every historical XLA execution incident or certify the TF runtime.

## Decision, next step and evidence limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Root-cause audit complete | Derived escape plus replay and controlled checks | All numerical/provenance checks pass | Other finite basins and original-author choices | Recover practical source choices; if unavailable, test explicitly labelled finite-local-fit reconstructions | No universal failure claim |
| Current adaptive iAPF candidate rejected | No complete fitted backward sweep | All four actual-input fits fail stationarity | Whether a different local procedure is useful downstream | Require fresh full sweeps and adaptive/Kalman evaluation after a candidate qualifies | Neither paper replication nor TF/GPU readiness |
| KDM/MDPF status preserved | Outside this audit | No new KDM test | Prior source/comparator/uncertainty gaps | Follow its existing master-program phases | No KDM root-cause conclusion |

A finite-domain bound changes the feasible set; a penalty, normalized shape
loss or fixed amplitude changes the fitting objective. Those may be worthwhile
separate hypotheses, but must not silently replace Eq. (15) or be called
recovered author choices. A source-justified local stopping/initialization
procedure could retain Eq. (15), but still needs evidence that it produces a
useful twist. No revised numerical default is selected here.

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Actual saved fits and tested actual-target alternatives fail the declared stationarity requirement |
| Statistically supported ranking | None; deterministic interventions on two fixed datasets do not rank stochastic filters |
| Descriptive-only differences | Loss, gradient, shape, concentration and runtime within these saved cases |
| Default-readiness | Not established; independent R reference only |
| Next evidence needed | Finite local-fit evidence, fresh backward sweeps, downstream adaptive likelihood comparisons with conditional heuristics and uncertainty, then TF/GPU parity |

The strongest alternative explanation is that a different initialization or
local procedure finds useful finite minima that these starts miss. That would
overturn any broader claim that Eq. (15) is unusable, while leaving the
vanishing-amplitude derivation intact. The weakest evidence is attribution of
the t=98 timing to concentration versus basin geometry. Self-review corrected
an early draft that incorrectly distinguished t=99 from t=98 by the presence of
a floor; both have one. No independent reviewer is claimed.

The exact command was
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_iapf_root_cause_audit.py`.
Attempt01 exited 0 in 7.438382 seconds, with 60 additional seconds charged
conservatively for preliminary inspections without preserved exact timing.
No GPU was used; CUDA_VISIBLE_DEVICES=-1 and one BLAS thread were set before R.
One of two permitted diagnostic attempts was needed; the audit is closed.

Combined research CPU charge is now 199.100153 / 1800 seconds, leaving
1600.899847 seconds. The conservative remaining GPU allocation is unchanged,
about 36.567 hours. No further fit campaign was launched by this audit.
