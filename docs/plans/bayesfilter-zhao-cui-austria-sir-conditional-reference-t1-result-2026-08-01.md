# Zhao-Cui Austria SIR Conditional-Reference T1 Result

Date: 2026-08-01

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-repair-plan-2026-08-01.md`.

Artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-conditional-reference-t1-20260801/authority-two-seed-n8192-retry02/`.

## Result

The independent origin proposal authority passed its mechanics and numerical
gates.  Each cloud has 8,192 rows generated at `theta_ref=0` from fixed
standard-normal innovations and evaluated with the exact finite importance
program

\[
\widehat Z(\theta)=N^{-1}\sum_i
\exp[\log p_\theta(z_{0i})+\log f_\theta(z_{1i}|z_{0i})+
\log g_\theta(y_1|z_{1i})-\log p_0(z_{0i})-\log f_0(z_{1i}|z_{0i})].
\]

The analytical complete-data ratio score and `GradientTape` derivative of this
same finite program agree to machine precision for both seeds.  The payload
records value MCSE, score MCSE, ESS, cloud hashes, and source-observation hash.

| Seed | log value | value MCSE | score | score MCSE | ESS |
|---:|---:|---:|---|---|---:|
| 92001 | -31.1313338600 | 0.00460 | `[-6.04346, 2.35302, -4.89090]` | `[0.86837, 0.22922, 0.00969]` | 6981.98 |
| 92002 | -31.1309072106 | 0.00453 | `[-3.89973, 1.55798, -4.88739]` | `[0.85510, 0.22618, 0.00956]` | 7011.16 |

The paired score difference is `[2.1437, 0.7950, 0.0035]`.  The corresponding
three-combined-MCSE screen is approximately `[3.65, 0.97, 0.041]`, so the
independent seed check passes.  The value difference is `0.000427`, below the
three-combined value MCSE screen of approximately `0.0194`.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Admit independent T1 reference authority | Analytical score equals autodiff derivative of same finite program; two-seed value/score agreement; ESS >= N/2 | Finite, shape, backend-mechanics, and ESS checks passed | Monte Carlo error and proposal-to-target transfer away from origin | Design a parent-conditioned residual interface in `(z0, epsilon)` and prove zero-slice parity | No Zhao-Cui source-faithful child, no full-horizon score, no proposal-quality claim, no HMC |

## Evidence classes

- Hard veto evidence: none for this mechanics/reference artifact.
- Descriptive evidence: the two seed estimates and MCSEs above.
- Statistical evidence: only the predeclared paired three-MCSE viability screen;
  it does not rank methods or establish superiority.

## Scientific boundary

The target actually computed is the sealed latent pre-clipping Austria T1
observed-data value under a frozen origin proposal.  It is equal to the exact
finite target definition used by the model at `theta=0`; it is not yet a
Zhao-Cui assembled child.  The innovation coordinates are sampling variables,
not an added Jacobian term in the physical score.

This result repairs the target/authority gap exposed by the failed centered
rank ladder.  It does not repair that representation.  The old child remains
rejected, and no score recursion beyond T1 is authorized.

## Post-run red team

The strongest alternative explanation is that both clouds share a model or
observation-definition error.  The next discriminating check is scalar-model
parity at symmetric nonzero theta rows and an independent direct complete-data
ratio calculation.  A future child pass would still not prove source
faithfulness without the Zhao-Cui paper and author-code anchors.
