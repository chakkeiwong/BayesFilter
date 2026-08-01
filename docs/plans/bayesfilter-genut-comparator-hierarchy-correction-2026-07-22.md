# GenUT Comparator Hierarchy Correction

Date: 2026-07-22

Status: `ACTIVE_COMPARATOR_POLICY_CORRECTION`

## Decision

The previous Cubature/GenUT master plan used the centered-Gaussian Contract E
residual route as a universal baseline. That is the wrong comparator policy for
the intended nonlinear-model testing program. The centered-Gaussian residual
route is retained only as an optional ablation to isolate residual-design
effects. It is not the default baseline, an oracle, or a promotion gate.

The active comparator hierarchy is:

1. **Exact model-specific oracle, when available.** Examples are the analytic
   Kalman value/score for the LGSSM and the refined dense integration value/
   score for the exact transformed-SV DGP. Oracle comparisons may support
   model-scope value/score accuracy claims under their declared uncertainty
   design.
2. **Fixed-variant Zhao-Cui diagnostic, when no exact oracle is available.**
   This is the closest independent algorithmic comparator for the intended
   high-dimensional filtering use. It is diagnostic evidence about cross-method
   behavior, stability, and scale. It is not an oracle and cannot certify the
   GenUT value, score, bias, likelihood, posterior, MLE, HMC, or scientific
   correctness.
3. **Centered-Gaussian Contract E residual ablation, when useful.** This
   isolates the effect of replacing a random Gaussian residual cloud by a
   deterministic GenUT cloud while retaining the same staged OT and
   Contract E-Chol restoration. It is not the scientific baseline.

## Mathematical And Evidence Boundary

The current GenUT implementation still uses the staged positive-OT and
Contract E-Chol restoration. The change is the residual-cloud policy and the
comparison hierarchy, not a claim that the restoration mathematics has been
replaced.

For a no-oracle model, the following are valid diagnostic questions:

- Do GenUT and fixed-variant Zhao-Cui produce finite, replayable values and
  recursive scores for the same declared target and data?
- Do their value/score trajectories, per-time increments, and parameter
  perturbation responses agree or disagree in a reproducible way?
- Are differences stable over independently generated datasets, particle
  seeds, and declared tuning scopes?
- Do both routes satisfy their own internal measure, derivative, residual,
  branch, and resource contracts?

The following conclusions are forbidden from the Zhao-Cui comparison alone:

- Zhao-Cui is the truth or an exact likelihood/score oracle;
- agreement proves GenUT correctness;
- disagreement proves GenUT bias or Zhao-Cui correctness;
- one route is statistically superior without a predeclared target-specific
  ranking criterion and uncertainty analysis;
- HMC, MLE, posterior, leaderboard, or production readiness.

Every Zhao-Cui row must retain the project source-anchor classification:
`source_faithful`, `fixed_hmc_adaptation`, or `extension_or_invention`.
The generic retained-grid route remains diagnostic/historical only. A
fixed-variant implementation that is `extension_or_invention` may be used as a
clearly labeled diagnostic if the target and scope are explicit, but it cannot
close a source-faithful Zhao-Cui admission gate without the required human
approval and paper/source anchors.

## Required Per-Model Comparison Contract

| Model situation | Primary comparator | What it can establish | What it cannot establish |
|---|---|---|---|
| Exact oracle available | Model-specific exact value/score | Value/score accuracy under the oracle's stated assumptions | Broad nonlinear or cross-model superiority |
| No exact oracle, fixed Zhao-Cui available | Fixed-variant Zhao-Cui diagnostic | Reproducible cross-method behavior, trajectory/stability diagnostics, implementation disagreements | Truth, accuracy, unbiasedness, or method superiority |
| No exact oracle or Zhao-Cui route | Internal GenUT contract plus explicit blocked status | Finite/replayable execution and declared derivative/measure/resource mechanics | Any external accuracy claim |
| Residual-design study | Centered-Gaussian Contract E ablation | Effect of residual-cloud choice within the shared staged restoration | Baseline truth or default-readiness |

## Plan Audit

The correction resolves four material risks in the old plan:

- **Wrong baseline:** Contract E Gaussian is no longer silently treated as the
  strongest comparator for models where it is merely another approximation.
- **Oracle substitution:** Zhao-Cui is explicitly diagnostic and cannot become
  a hidden truth source.
- **Source-boundary drift:** fixed-variant Zhao-Cui retains paper/source anchor
  and classification requirements.
- **Scope mismatch:** exact-oracle evidence and no-oracle diagnostic evidence
  are reported in separate ledgers and cannot be pooled into one accuracy gate.

Audit verdict: `PASS_WITH_EXPLICIT_ORACLE_DIAGNOSTIC_SEPARATION`.

## Next Action

Revise the active Cubature/GenUT nonlinear master plan and future model plans to
use this hierarchy. For each no-oracle model, first determine whether a
target-matched fixed-variant Zhao-Cui route exists and record its classification
and anchors. If it does not, leave the comparator cell blocked rather than
substituting centered Gaussian Contract E or calling an internal diagnostic an
oracle.

This correction changes the comparator policy. It does not by itself promote
GenUT to every model, certify Zhao-Cui, or alter the separate owner-directed
runtime policy until the default decision is recorded under the corrected
hierarchy.
