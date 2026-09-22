# iAPF analytical Fisher-score comparison

Completed 2026-09-18 under the [reviewed plan](../../younis-iapf-fisher-score-2026-09-18.md).

The raw genealogical Fisher score still fails the practical comparison: at
N4096 its score MSE exceeds UKF on both weakly nonlinear datasets, and at
N256 it exceeds UKF on all four nonlinear datasets. It also has greater MSE
than the fixed-label derivative on every dataset at N4096. No estimator is
promoted. The useful finding is narrower: the new Fisher estimator passes
the predeclared mean-bias screen on all four nonlinear datasets, whereas
the fixed-label derivative fails on all four. The remaining observed error
in the Fisher estimator is dominated by sampling variance.

## What was computed

For the physical model
\(p_\theta(x_{0:T},y_{1:T})=\mu_\theta(x_0)
\prod_t f_\theta(x_t\mid x_{t-1})g_\theta(y_t\mid x_t)\), define
\[
A_T=\partial_\theta\log\mu_\theta(X_0)+
\sum_{t=1}^T\{\partial_\theta\log f_\theta(X_t\mid X_{t-1})+
\partial_\theta\log g_\theta(y_t\mid X_t)\}.
\]
Each partial holds the latent states fixed. Fisher's identity is
\(\nabla_\theta\log p_\theta(y)=E_\theta[A_T\mid y]\).
The optional output propagates these analytical terms through the actual
iAPF ancestors and returns \(\sum_i W_T^i A_T^i\), using terminal weights
before final resampling. The initial-distribution terms are included.
The plan derives cancellation of the auxiliary twist factors, so the terminal
weighted genealogy targets the physical posterior.

The existing output differentiates the finite particle likelihood while
holding categorical labels fixed. Its earlier finite-difference agreement
checks that derivative. It does not establish equality with the physical
model score. The new estimator addresses that target distinction, but is
neither finite-N unbiased nor the gradient of the reported finite likelihood.
It therefore cannot silently replace an HMC force. The source anchor is
Poyiadjis, Doucet and Singh (2011), Section 2.1, Eq. (4), Algorithm 1/Eq. (7);
the local paper and the iAPF cancellation derivation are linked in the plan.
Original-author implementation parity for this extension is not checked.

## Conditional results

T=2, 64 independent final particle streams per rung, frozen offline fits,
GPU FP32/TF32/XLA. The table reports the mean squared Euclidean error of the
six-component score against the FP64 oracle. These MSE values are descriptive;
the accompanying paired intervals provide the stated conditional inference.

| Dataset | Regime | N | iAPF fixed-label | iAPF Fisher | Bootstrap Fisher | No resampling | UKF |
|---|---|---:|---:|---:|---:|---:|---:|
| 1390 | Affine | 4096 | .0009024 | .0030215 | .0035753 | .0015266 | 1.24e-13 |
| 1400 | Weak | 256 | .0613964 | .1689140 | .3192516 | .3596057 | .0075210 |
| 1400 | Weak | 1024 | .0141405 | .0427519 | .0746033 | .0981482 | .0075210 |
| 1400 | Weak | 4096 | .0039825 | .0100147 | .0187827 | .0255705 | .0075210 |
| 1401 | Weak | 256 | .0708660 | .2701401 | .5284280 | .4873477 | .0077707 |
| 1401 | Weak | 1024 | .0147857 | .0626097 | .1084860 | .0963740 | .0077707 |
| 1401 | Weak | 4096 | .0044674 | .0143343 | .0263673 | .0334257 | .0077707 |
| 1410 | Curved | 256 | .0471042 | .1163789 | .1632155 | .2121052 | .0167799 |
| 1410 | Curved | 1024 | .0140941 | .0290764 | .0467151 | .0586840 | .0167799 |
| 1410 | Curved | 4096 | .0041061 | .0080764 | .0123620 | .0129100 | .0167799 |
| 1411 | Curved | 256 | .0333070 | .0922294 | .1313913 | .1201740 | .0429724 |
| 1411 | Curved | 1024 | .0100775 | .0294248 | .0273604 | .0286460 | .0429724 |
| 1411 | Curved | 4096 | .0039545 | .0060966 | .0087468 | .0071444 | .0429724 |

The [full conditional table](conditional-errors.csv) also includes EKF and
the bootstrap fixed-label derivative. [analysis.json](analysis.json) preserves
117 paired comparisons and the empirical MSE decomposition. The raw 4,160
score vectors and fitted coefficients are in [results.json](attempt02/results.json).

At N4096, the four nonlinear Fisher means pass all 24 component screens using
the approximate Bonferroni Student-t 99% criterion (critical value 3.7271),
with the separately recorded FP32 numerical allowance. Their largest absolute
mean-error/MCSE is 3.1443. The corresponding fixed-label maxima by dataset are
5.493, 3.872, 9.614 and 13.962, and each dataset fails at least one component.
Bootstrap Fisher passes all four datasets; bootstrap fixed-label fails all
four and also fails the affine check. Affine iAPF Fisher and fixed-label
both pass. Passing the screen means no bias was detected at this resolution;
it is not proof of zero bias. The Fisher intervals are wider because its
sampling variance is larger.

For each nonlinear dataset, the N4096 iAPF Fisher-minus-fixed-label MSE
interval is strictly positive: respectively [.00408,.00833],
[.00664,.01392], [.00232,.00579] and [.000886,.00341]. The Fisher-minus-UKF
intervals are positive in the weak regime, [.000285,.00505] and
[.00294,.01078], and negative in the two curved cases. These are individual,
exploratory paired-bootstrap 99% intervals, not a simultaneous ranking over
the entire study. Fisher-minus-no-resampling at dataset1411 is unresolved:
[-.00437,.00175]. There is no defensible universal winner.

On curved dataset1411, the empirical MSE decomposition illustrates the tradeoff:
fixed-label squared mean error is .001142 and variance contribution .002813;
Fisher squared mean error is .0000260 and variance contribution .006071.
This is the exact decomposition of the observed sample MSE, not an unbiased
estimate of squared population bias. It motivates variance reduction; it
does not justify mixing the two scores without a target/centering argument.

## Engineering and execution

All 29 distinct focused CPU checks pass across the recorded invocations:
independent scalar and multivariate density derivatives, full genealogy replay,
initial-law contribution, default-output parity, terminal-resampling invariance,
and existing consumer checks. The KDM mixture and SGQF consumers required
repairs for the shared quadrature API's new validity outputs. Injected-invalid
tests now verify propagation to the actual LEDH consumer's (-infinity, zero
score) result. These repairs change neither quadrature formulas nor controls.

The first GPU attempt exposed an invalid harness assumption: separate FP32
XLA compilations with/without the extra score need not preserve categorical
choices exactly. One of 4096 particles differed in the second cloud; value
and score differences were 1.62e-5 and at most 2.00e-4. FP64 replay of the same
inputs gave identical clouds/value and score agreement within 2.23e-16.
This supports rounding amplified by categorical selection; the specific
compiler operation was not isolated. A non-XLA GPU probe returned empty tensors
in the self-built TensorFlow environment and is not usable evidence.

The [plan amendment](../../younis-iapf-fisher-score-2026-09-18.md)
records the assumption repair before resumption. No tolerance was raised and
no numerical kernel was altered. Both estimators come from one joint invocation,
so their actual trajectories and likelihood values are identical. The failed
attempt remains preserved. Complete affine results and the frozen1400 fit were
reused only after numerical source hashes matched; partial particle results
were repeated. Final source hashes still matched. Every retained scored-kernel
specialization has tracing count 1; the generic TensorFlow retracing warning
reflects creation of the declared shape/regime specializations.

All five grid mesh/domain checks are below 2.18e-15; recorded tail diagnostics
are below 3.34e-16. Affine grid/Kalman agreement is 1.67e-15. These are numerical
checks on these fixtures, not rigorous global quadrature error bounds.
All five fits were valid and had no active parameter boundary in this fresh
sample; maximum projected gradient was 9.73e-8. Dataset1401 needed 2404 fitting
steps, again exceeding the old 2000-step cap. These observations do not clear
previous failed holdouts or establish general adequacy of the frozen controls.

The two [manifests](attempt01/manifest.json) and
[resumed manifest](attempt02/manifest.json) retain exact commands, source hashes,
seeds, data, environment and runtime. Both use the RTX5080, TensorFlow
2.20.0-dev0+selfbuilt, XLA and verified memory growth. References deliberately
use CPU FP64; final filters use GPU FP32. No timing ranking is made on this
shared device. Consumption: 2/4 attempts, 128.756/1800 summed driver seconds,
5/8 adaptive fits, 3000/3000 filter-call charges; tests/probes conservatively
charged 240/600 seconds. The comparison is complete and its call budget is
exhausted. No further launch belongs to this stage.

## Decision and next action

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain Fisher as a checked diagnostic comparator | Density/genealogy checks and all four N4096 bias screens pass | No unresolved numerical/reference continuation veto | Finite-N bias and wider confidence intervals | Derive target-preserving variance reduction | Exact model score, finite-N unbiasedness or finite-program gradient |
| Reject raw genealogy as a default candidate | Conditional MSE comparison complete | UKF losses in weak cases and lower-N cases | Only two datasets per nonlinear regime | Fresh, separately calibrated conditional-integration or exact-mean control-variate study | Rejection of iAPF or Fisher's identity |
| Keep wider master open | This diagnostic stage complete | Old fit-bound and missing integration evidence remain | Other targets, longer horizons, cost and defaults | Keep fitting repair open; finish required evidence before iAPF-moment/LEDH integration | Canonical LEDH or HMC readiness |

| Inference status | Conclusion |
|---|---|
| Hard veto screen | Conditional heuristic losses veto promotion; no corruption or physical-model invalidity found |
| Statistically supported ranking | Only the individual conditional intervals above; no simultaneous or general ranking |
| Descriptive-only differences | Particle-count trends, maxima, empirical bias/variance decomposition and absence of fit boundaries |
| Default-readiness | No; diagnostic output only |
| Next evidence needed | A derived same-target variance reduction, fresh calibration/confirmation, longer horizons and broader model coverage |

Post-run skeptical review: the strongest alternative explanation for an apparent
bias repair is that the noisier Fisher estimator simply has wider intervals.
Independent density/genealogy checks and the physical identity support its
target, but many more replications or a limiting argument would be needed to
establish its bias behavior. Bias at larger N, a failed identity, or a reference
failure would overturn a stronger claim. Two short nonlinear datasets per regime
and a single frozen fit per dataset are the weakest generalization evidence.
The review is local; no independent reviewer is claimed.

Next derive a conditional-integration or exactly centered variance-reduction
candidate under the actual sampling law. For a self-normalized score,
integrating numerator and denominator separately is not automatically the
conditional expectation of their ratio. Prove the relevant identity first;
keep this stage's final streams and old failed holdouts out of calibration.
This is the master's bounded score diagnosis, not activation of its deferred
general backward-smoothing program. Prior fitting-bound repair remains open
before promotion and iAPF-moment/LEDH integration.
