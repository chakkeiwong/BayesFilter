# Particle-filter score literature gap audit

Date: 2026-09-14

## Question

Which important primary-literature methods or theory are missing from the LEDH/PFPF/GenUT particle-filter score research program, and would any omission change the program's targets, comparators, or treatment of degenerate transitions?

## Scope

Audit model observed-data scores and derivatives of finite particle-filter programs, with emphasis on differentiable/continuous resampling, Fisher/score identities, tangent and smoothing recursions, variance reduction, unbiasedness, and singular or disturbance-state models. This is a source audit, not an experiment or claim of exhaustive bibliometric coverage.

## Evidence contract

The baseline is the current master program `docs/plans/younis-kdm-score-master-program-2026-09-14.md`: exact observed-data score, finite-program derivative, KDM fixed-proposal expectation gradient, unnormalised derivative/normaliser pair, canonical LEDH-PFPF-OT route, and separate regular/singular tracks. Primary support requires inspected technical sections/equations/algorithms. An omission is actionable if the source offers a distinct estimator, theorem, or failure mode that changes a phase comparator or validity gate. Citation counts and venue are metadata only. No experiment is run in this audit; the artifact is this plan, the ledgers, and the final audit note.

## Pre-mortem

A search could miss a key method because terminology varies (`tangent filter`, `online smoothing`, `score`, `gradient`, `disturbance`). Cheap mitigation is backward/forward snowballing from Poyiadjis, PaRIS, differentiable-PF, OT-resampling, and disturbance-state seeds. A paper may prove a result only for regular dominated transitions; every source is therefore classified by support assumptions before transfer to DSGE. A promising method may target a variational or discriminative objective rather than the marginal score; target identity is checked explicitly.

## Audit actions

1. Inventory existing local papers and citations, then identify gaps by method family.
2. Verify primary technical text for the highest-risk omitted families and record six ledgers in `docs/plans/artifacts/particle-filter-score-literature-gap-audit-20260914/`.
3. Snowball backward and forward where public primary sources are available; record blocked metadata rather than fabricate coverage.
4. Produce a gap note with severity, exact target relation, assumptions, and a proposed insertion point in the master program.

## Decision rule

A missing family is a program-level omission when it is (a) a direct estimator or variance theorem for the same score target, (b) a materially different way to handle resampling/normalisation/singular support, or (c) a standard comparator needed to interpret a claimed improvement. It is background when it only supplies generic autodiff or unrelated objectives.

## Skeptical audit outcome

The initial matrix passed the target-identity and support-split checks, but it
did not pass the comparator-coverage check: “KDM”, “backward pair”, and
“coupled multilevel” were labels without the named published algorithms and
assumptions needed to interpret their results. The audit therefore adds the
Nemeth KDE/Rao--Blackwell estimator, explicit PaRIS/Fearnhead smoothers,
Jacob--Lindsten--Schön coupled debiasing, continuous-likelihood comparators,
and iterated filtering before any claim-bearing score comparison. Twisted and
multilevel filters are retained as proposal/discretization variance controls,
not as score identities. The singular DSGE branch remains separately gated.

## Manifold and constrained-state branch

A bounded search on 2026-09-14 confirmed a substantial literature on particle
filtering with states constrained to manifolds, but did not identify a
published parameter-score estimator for a degenerate state transition. The
papers found are primarily filtering and state-estimation methods: Snoussi and
Mohammad-Djafari propose sampling along manifold geodesics; Koval et al.'s
manifold particle filter handles contact manifolds of changing dimension; and
Zhang, Taghvaei, and Mehta derive a feedback particle filter on Riemannian
manifolds and matrix Lie groups. Their targets are filtering distributions or
state estimates, not the observed-data parameter score used in this program.

The correct implication is conditional. If a fixed manifold $\mathcal M$ has a
parameter-independent reference volume $\mu_{\mathcal M}$ and the transition
has a differentiable density $k_\theta(x,dx')$ with respect to that measure,
then a Fisher/Poyiadjis/PaRIS-style additive score can in principle be
re-derived intrinsically on $\mathcal M$. This is a new support-specific
derivation, not a transfer of the ambient-space formula. It is not the generic
DSGE case: structural equilibrium restrictions and solved policy functions
usually define a parameter-dependent path set $\mathcal M_\theta$, and the
shock-to-state map itself depends on $\theta$. If the manifold or constraint
depends on $\theta$, or if the transition is a deterministic map from
lower-dimensional shocks, the score must instead be written in charts or
innovation coordinates and must include the relevant Jacobian, volume, and
support terms. A Euclidean Gaussian/KDE insertion or an autodiff derivative of
a projected particle program does not establish that identity.

The search also found manifold score-matching and Riemannian-gradient papers,
but those estimate a spatial density score $\nabla_x\log p(x)$ or define a
geometric optimizer. They are not particle estimators of
$\nabla_\theta\log p_\theta(y_{1:T})$. This terminology distinction is now an
explicit omission check. The literature branch is therefore a mathematical
control and possible special-case extension, not the current DSGE route. The
DSGE claim-bearing route remains the disturbance/innovation-coordinate
derivation.

Primary records inspected:

- Snoussi and Mohammad-Djafari, *Particle Filtering on Riemannian Manifolds*,
  DOI [10.1063/1.2423278](https://doi.org/10.1063/1.2423278).
- Koval et al., *The Manifold Particle Filter for State Estimation on
  High-dimensional Implicit Manifolds*, DOI
  [10.1109/ICRA.2017.7989543](https://doi.org/10.1109/ICRA.2017.7989543).
- Zhang, Taghvaei, and Mehta, *Feedback Particle Filter on Riemannian
  Manifolds and Matrix Lie Groups*, DOI
  [10.1109/TAC.2017.2771336](https://doi.org/10.1109/TAC.2017.2771336).
- Koval, Pollard, and Srinivasa, *Pose Estimation for Planar Contact
  Manipulation with Manifold Particle Filters*, DOI
  [10.1177/0278364915571007](https://doi.org/10.1177/0278364915571007).

These records support the existence and scope of manifold particle filtering;
they do not support a claim that a manifold score estimator has already been
provided for the DSGE problem. Full technical-source inspection remains
required before importing any intrinsic smoother or claiming a literature
precedent for parameter-score computation.
