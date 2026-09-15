# Review of the amended Younis/KDM score master program

Date: 2026-09-14

Reviewed document:
[master program](../plans/younis-kdm-score-master-program-2026-09-14.md).
This checks the mathematical specification, target labels, phase dependencies,
test coverage, decision rules, and selected current source prerequisites. It
does not establish filter runtime readiness or constitute a MathDevMCP audit.

## Claude's clarification

Claude's bounded Phase 4C review gave `REVISE` and explicitly marked the wider
proposal, SGQF, and KDM implementation questions unchecked. Treating that as
whole-program approval was an unfair characterization. The master accepts the
agreed FD corrections and identifies the additional dependency and integration
findings as findings of this amendment review.

The ratio-bias witness was already in the manuscript. The OT question asked
whether the existing flow determinant enters the likelihood; it did not demand
a new OT determinant. Historical replies are preserved, but these stronger
characterizations are withdrawn for the present review. The actual safeguards
remain: distinguish derivative consistency from expectation ratio bias and
derive any new density correction from the sampling law.

## Mathematical review

| Issue | Amended specification and verdict |
|---|---|
| FD MSE | Correct: \((T_h+B_{N,h})^2+c^\mathsf T\Sigma(h)c/h^2\), retaining \(2T_hB_{N,h}\). Bias cancellation is possible. |
| Coupling | Correct: the covariance numerator is measured. Perfect shared additive noise cancels, mean-square differentiability can yield bounded variance, and independent nonvanishing noise gives \(h^{-2}\) variance. No universal U-curve. |
| Deterministic order | Correct sufficient assumptions: bounded third derivative with \(C^3\) for second order; bounded fifth derivative with \(C^5\) for fourth order. No stochastic slope veto. |
| Cubic fit | Odd-block normal equations give the stated eleven-point weights. Fitting against \(j\) requires division by \(h\); fitting against \(jh\) does not. The larger span changes the truncation constant. |
| Direction solve | For \(V\in\mathbb R^{d\times m}\), solve \(V^\mathsf Ts\approx b\) by QR/SVD; the full-row-rank solution is \((VV^\mathsf T)^{-1}Vb\). Propagate directional bias and covariance. |
| Step/conditioning choices | No fixed universal ranges or cutoffs. Require valid and representably distinct parameters, coordinate scaling, numerical error bounds, and rank/error amplification checks. |
| Two biased estimators | The bias is the weighted sum; scalar covariance has the factor two, whereas vector covariance needs both cross-covariance matrices. Oracle MSE-optimal mixing and variance-only mixing are different objectives. |
| KDM control variate | Exact centering preserves baseline expectation and bias. IWSG's own expectation gradient is not automatically a valid center under the LEDH joint sampling law. |
| Ratio bias | If \(\widehat D=\nabla\widehat Z\), then \(\widehat D/\widehat Z=\nabla\log\widehat Z\) pathwise. Their agreement cannot establish equality of their expectation with \(\nabla\log Z\). |
| Proposal covariance and OT | Moment restoration is not distribution preservation. Sampling density, correction, moment lifecycle, and total derivatives must be checked separately. |

The final sweep also removed an inherited unconditional Jacobian requirement
for jitter after OT. A mixture centered on transported points requires its
actual jitter density; an additional change-of-variables determinant depends
on the sampling construction. Gate C now distinguishes a declared approximation
from a broken density or correction, which cannot be fixed by relabeling.

The FD argument is now taught in order: deterministic approximation, particle
bias and covariance, measured MSE, selection, then full-vector reconstruction.
The target reader is a particle-filter researcher familiar with conditional
expectations, importance sampling, Taylor expansion, and least squares.
Numerical/call-chain tests remain operational requirements rather than
substitutes for the derivation. Human readability feedback is still pending;
this is an author review, not a readability certification.

## Dependency and coverage review

| Task | Required coverage and current status |
|---|---|
| Phase 0 | Exact Gaussian oracle, baseline ladder, target metadata, initial law, and reusable scope tuning are prepared before quality claims. A high-\(N\) reference needs a bias bound, not replication alone. |
| Phase 1 | Tests are chosen by target: finite scalar, IWSG expectation, unnormalised pair, or model-score estimate. Initial and recursive terms and actual endpoint wiring must be tested. |
| Phase 2 | Actual sampler/denominator, support, normalization, twisted correction, covariance lifecycle, and SGQF signed-weight/moment checks precede comparisons. Score improvement needs score MSE. |
| Phase 3 | Conditional comparisons on common clouds are repeated across independent clouds. KDM combinations keep exact centering, heuristic bias correction, and different expectation targets distinct. |
| Phase 4 | Each new horizon has its own tuning scope. Normalizer, derivative, and ratio are measured separately; unresolved causal attribution does not invalidate a sound held-out MSE measurement. |
| Phase 4B | Oracle calibration and held-out correlation uncertainty precede transfer. Explanatory warnings and promotion vetoes cannot silently become continuation vetoes. |
| Phase 4C | Exact weights, polynomial moments, smooth deterministic functions, fit units, coupling marginals, step validity, reconstruction rank, and propagated uncertainty are specified separately from stochastic performance. |
| Phases 5--6 | Smoothing and DSGE derivations are deferred to separately assigned programs; no active dependencies or rows require them. |
| Phases 7--9 | Construct conditional heuristic adversaries, use paired/hierarchical uncertainty, preserve failures, compare equal \(N\) and equal compute, verify scope artifacts and source/implementation evidence. |

The revised coordinator checks prerequisites per row and keeps planned repairs
available after a promotion failure. It no longer waits until Phase 8 to tune,
nor does an unrelated KDM failure stop deterministic FD mechanics.

## Statistical and default audit

Calibration, validation/selection, and final claim data and streams are disjoint.
Pilot-fitted controls are frozen, but parameter dependence during their
evaluation still needs the declared derivative terms. Claim data cannot choose
the winner. Particle replicates nested inside a dataset are not independent
datasets. Optional replication increases require an appropriate sequential
rule or a fixed terminal sample size.

ESS, covariance accuracy, and weight tails do not substitute for downstream
score MSE. A normalizer-accuracy claim uses normalizer MSE; variance alone
requires a common expectation. MSE times runtime is not a universal comparison,
since replication reduces variance and leaves bias. Rank cutoffs, FD scales,
covariance ridges, and other numerical choices need scoped justification.

The large initial grid is a coverage proposal. A serious launch still needs a
finite row/attempt/compute budget and an explicit precision target. This is
ordinary campaign planning, not an additional approval-token mechanism.

## Remaining obligations

| Obligation | Inspected evidence | Required next evidence |
|---|---|---|
| KDM callback compatibility | Scalar executor rejects trace and observation/post-reset callbacks; integrated/resampling KDM endpoints request them. | Recheck the ongoing canonical repair and run consumer-to-provider regressions. Runtime call-chain correctness is not checked here. |
| Initial-law sensitivity | Scalar executor seeds supplied state/covariance tangents to zero. | Nonconstant initialization fixture and verification of the complete outer dependency. Fixed-input derivatives are not themselves wrong. |
| SGQF integration | Standalone module declares an eager/Python runtime; its LEDH call chain is unverified. | Multistep filtering moments, persistent state/reset, batch/device/XLA, and analytical sensitivity parity. |
| Twisting/iAPF | The master now requires the actual ancestor/state law, integral, correction, and derivative terms. | Technical paper/author-code anchors and executable correction/endpoint tests for the selected implementation. |
| Companion LaTeX | The directional-FD section still has the four-derivative claim and stochastic slope conflation. The SGQF sketch omits the full lifecycle and contains literal `,qquad`. | Localized manuscript correction, PDF build, and rendered inspection before it serves as the revised implementation specification. |

The master records these obligations explicitly. None can be represented as
implemented merely because its requirement is now written.

The source checks above refer to the current
[scalar executor](../../bayesfilter/highdim/ledh_canonical_score_tf.py)
(trace/callback guards at lines 174--193 and initial tangents at 221--224),
[integrated KDM consumer](../../bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py)
(call at 522--531),
[resampling KDM consumer](../../bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py)
(call at 443--453), and
[standalone SGQF](../../bayesfilter/nonlinear/fixed_sgqf_tf.py)
(runtime marker at line 17). The manuscript observations are in
[the score note](../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex),
around lines 1477--1483 and 1525--1570. Recheck these locations after concurrent
repairs; this review records a snapshot, not a permanent capability verdict.

## Evidence and decision

The protected baseline, complete amendment diff, checkpoint, reference checks,
and validation record are under
[the amendment artifact directory](../plans/artifacts/younis-score-master-amendment-20260914-01/).
Validation is limited to document consistency and independent mathematical
reference calculations. No particle campaign, GPU run, external reviewer call,
or MathDevMCP audit was performed. Unrelated working-tree edits are preserved.

Exact rational checks passed for the three stencil moment systems and leading
Taylor coefficients, additive-noise cancellation, the bias cross-term witness,
the vector mixing coefficient, and a rectangular direction solve with a
singular incorrect Gram matrix. The document-anchor checks passed. The amended
Markdown also built to a 22-page PDF; pages 16--17 were visually inspected for
the smoothness, MSE, calibration, and reconstruction equations. This PDF is a
preview of the amended master, not an updated companion research manuscript.
The first LuaLaTeX render failed on a missing font; the PDFLaTeX retry exposed
and led to correction of two inherited math delimiters, then built successfully.
Logs preserve those attempts.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the amended active program | Corrected mathematical/experimental specification | Named implementation prerequisites remain open | Actual KDM/SGQF/twist call chains | Implement and check the prerequisites, beginning with exact Gaussian mechanics | Score improvement, unbiased finite-particle model score, default/HMC readiness |
| Retain KDM control variates and heuristic combinations | Separate center/target requirements and held-out MSE | Unknown centering blocks an exact-CV claim | Center estimation cost and correlation with LEDH error | Test centered statistics and oracle-calibrated combinations separately | An IWSG variance result automatically improves the LEDH score |
| Defer smoothing and DSGE | Scope follows the user's separate assignments | No automatic active rows | External applicability evidence | Import only after target/support/provenance checks | Regular-track results solve a degenerate transition |

Strongest remaining alternative explanation for an eventual apparent gain is
a change of target or tuning scope. The earliest discriminating evidence is
the actual sampling-law/endpoint identity followed by untouched oracle error.
The weakest present evidence is runtime integration: this amendment specifies
its tests but does not execute them.
