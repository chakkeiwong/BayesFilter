# Particle-filter score literature gap audit

Date: 2026-09-14  
Question: whether the current LEDH/PFPF/GenUT/KDM score program has missed a
literature family that changes its target, comparator set, or treatment of
degenerate transitions.

## Finding

Yes. The program has the right central distinction—marginal model score,
finite-program derivative, KDM fixed-mixture expectation gradient, and
unnormalised derivative must remain separate—but its literature matrix is too
generic in six places. The omissions are actionable additions to the plan,
not a reason to discard the current target registry or the regular/singular
track split.

The most important omission is Nemeth, Fearnhead, and Mihaylova's published
KDE/Rao--Blackwell score estimator. It is the closest prior method to the
proposed KDM row and directly addresses the familiar choice between
path-space score variance that grows badly with horizon and quadratic-cost
backward calculations. The current program should reproduce it under its own
fixed-bandwidth and regular-transition assumptions before describing a KDM
score route as new or improved.

There is an acronym trap here. Nemeth's “kernel” is a KDE/Rao--Blackwell
approximation of the score-statistic recursion. It is not Younis's continuous
kernel-mixture proposal, and it does not provide the Younis fixed-mixture IWSG
identity. The registry must use distinct identifiers so that these two ideas
are not accidentally presented as one method.

The second omission is that “backward pair” is not a sufficient algorithm
description. PaRIS supplies a backward-draw parameter and a linear-cost
additive-functional smoother with a theorem for its long-time variance under
mixing assumptions. Fearnhead--Wyncoll--Tawn supplies a separate sequential
smoothing construction with linear- and quadratic-cost variants. Both should
be explicit Phase 3 arms, and both should be evaluated as estimators of
additive complete-data quantities before any claim about the marginal score.

There is also a direct debiasing route that the current program does not name:
Jacob, Lindsten, and Schön couple conditional particle-filter chains and apply
Rhee--Glynn debiasing to obtain unbiased smoothing expectations with finite
random cost under meeting and moment assumptions. They explicitly note that
Fisher's identity then gives an unbiased score when the transition density is
tractable. This is the most relevant published answer to the question “can two
biased estimators be combined into something better?” It should be a separate
regular-model comparator, with random cost and meeting-time variance reported.
It does not make an intractable or singular DSGE score exact.

The third omission is the continuous-likelihood econometric line: Malik--Pitt,
DeJong et al., and the related smoothly jittered particle-filter work of
Flury--Shephard. Malik--Pitt is not intrinsically one-dimensional: the paper's
abstract explicitly states that it proposes two multivariate approaches and
includes a neoclassical-growth application. These papers address
discontinuities in likelihood evaluation that motivated differentiable
resampling. They are needed to establish what OT/KDM contributes beyond
earlier continuous particle representations. A continuous finite particle
likelihood still needs a target and correction analysis; differentiability
alone does not make its derivative the exact model score. In particular, the
Malik--Pitt multivariate likelihood result is not evidence that an exact
parameter score exists for a parameter-dependent singular DSGE support.

The fourth omission is iterated filtering. Ionides et al. derive a likelihood
derivative approximation by augmenting the model with an artificial parameter
random walk. This is the standard comparison when a transition can be
simulated but its density or derivative is unavailable, so it belongs in the
DSGE/simulation-only branch. Its vanishing-noise limit has a visible
bias--variance--mixing trade-off and must not be called an exact finite-particle
score.

The fifth omission is proposal and discretization variance control. Twisted
particle filters preserve the likelihood through a change-of-measure
correction while targeting a better asymptotic normalizer-variance rate.
Multilevel particle filters use coupled telescoping levels to reduce
discretization/filtering work. They do not supply score identities, but without
these arms the program cannot tell derivative error from proposal or
discretization variance. A score-level coupling proof is required before
calling a multilevel difference unbiased for a score.

Finally, the program needs an explicit target-boundary row for FIVO/AESMC/VSMC
and learned smoothers, plus a continued search for constrained/manifold and
deterministic-transition SMC. The inspected literature does not provide a
generic ambient Gaussian KDM or PaRIS result for singular DSGE transitions.
Murray et al.'s disturbance-state representation supports a possible
lower-dimensional base-measure route, but the Jacobian, parameter dependence,
regime nonsmoothness, and variance properties remain local derivation and
experiment questions.

## Recommended ordering

1. Add Nemeth KDE/Rao--Blackwell, PaRIS, and Fearnhead smoothing to the direct
   score comparator set.
2. Add Malik--Pitt/DeJong continuous-likelihood and Ionides iterated-filtering
   rows with distinct target identifiers.
3. Add twisted proposals and multilevel coupling as variance-control arms,
   with normalizer, finest-level bias, increment variance, and work diagnostics.
4. Add variational-objective and PMCMC/HMC boundary checks so likelihood,
   ELBO, and derivative evidence cannot be transferred across consumers.
5. Keep Phase 6 blocked for promotion until the singular support derivation and
   a score estimator in the declared disturbance/base-measure coordinates pass
   their own checks.

## Evidence and limitations

Primary technical text was inspected locally for Nemeth, PaRIS, Fearnhead,
Ionides, Whiteley--Lee, Jasra et al., Poyiadjis, Murray, Corenflos, and the
Younis papers. Crossref/OpenAlex metadata and retraction flags were refreshed
for the main omitted papers; citation counts are coverage metadata only. The
publisher full technical texts for DeJong et al. and Malik--Pitt were not both
available in this pass (Malik--Pitt publisher XML was obtained), so their
algorithm-specific claims remain limited until full local copies are acquired.
The public search endpoint returned transient 502/503 errors; recent
constrained/manifold, coupled-score, weak-derivative, and learned-controlled-
SMC papers remain a bounded inspection queue. This is therefore a prioritized
gap audit, not an exhaustive bibliometric census.

## Source links

- [Poyiadjis, Doucet & Singh (2011)](https://doi.org/10.1093/biomet/asq062)
- [Nemeth, Fearnhead & Mihaylova (2016)](https://doi.org/10.1080/10618600.2015.1093492)
- [Olsson & Westerborn, PaRIS (2017)](https://doi.org/10.3150/16-BEJ801)
- [Jacob, Lindsten & Schön, coupled smoothing (2020)](https://doi.org/10.1080/01621459.2018.1548856)
- [Fearnhead, Wyncoll & Tawn (2010)](https://doi.org/10.1093/biomet/asq013)
- [Malik & Pitt (2011)](https://doi.org/10.1016/j.jeconom.2011.07.006)
- [DeJong et al. (2013)](https://doi.org/10.1093/restud/rds040)
- [Ionides et al. (2011)](https://doi.org/10.1214/11-AOS886)
- [Whiteley & Lee (2014)](https://doi.org/10.1214/13-AOS1167)
- [Jasra et al. (2017)](https://doi.org/10.1137/17M1111553)
- [Murray, Jones & Parslow (2013)](https://doi.org/10.1137/130915376)
