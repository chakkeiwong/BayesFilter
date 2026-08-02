# Zhao-Cui Austria SIR Conditional-Reference Sample-Growth Result

Date: 2026-08-01

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-repair-plan-2026-08-01.md`.

Artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-conditional-reference-t1-20260801/authority-two-seed-n65536/`.

## Outcome

The fresh `N=65,536` two-seed authority passes the predeclared precision and
validity screens.  It uses the same frozen origin proposal law and the same
finite importance value/score program as the `N=8,192` authority; only sample
count and seeds changed.

| Seed | log value | value MCSE | score | score MCSE | ESS |
|---:|---:|---:|---|---|---:|
| 92003 | -31.1312981308 | 0.001608 | `[-4.89556, 1.91978, -4.88762]` | `[0.30345, 0.08027, 0.00339]` | 56034.96 |
| 92004 | -31.1319630944 | 0.001602 | `[-5.31054, 1.95957, -4.88499]` | `[0.30491, 0.08055, 0.00338]` | 56103.33 |

The score difference is `[0.41498, 0.03979, 0.00263]`; the three-combined-MCSE
screen is approximately `[1.29, 0.34, 0.014]`, so the paired seed gate passes.
The value difference is `0.000665`, below the corresponding three-MCSE screen
of about `0.0068`.  Analytical complete-data scores and autodiff derivatives
of the same finite program agree for both clouds.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Pass T1 reference precision | Two fresh seeds agree within `3` combined MCSEs; ESS >= `N/2`; finite/autodiff parity | No hard veto fired | The exact fixed-variant parent is an approximation to the physical target away from its fitted slice | Design a reviewed representation bridge with an explicitly chosen base measure | No source-faithful child, no full-horizon score recursion, no HMC |

## Mathematical boundary

The authority computes the exact finite derivative of the importance program
under the frozen origin law.  A proposed parent-preserving correction of the
form

\[
q_\theta(r)\propto q_0(r)\,
\frac{\pi_\theta(z_0,z_1,y_1)}{\pi_0(z_0,z_1,y_1)}
\]

would preserve the parent at `theta=0`, but its normalized origin derivative
is an expectation under `q_0`, not automatically under the exact physical
law `p_0(z_0)f_0(z_1|z_0)g_0(y_1|z_1)`.  Therefore it cannot be advertised as
the correct observed-data score without a separate bridge proof or an exact
target-weighted base measure.  This is the current real blocker.

## Post-run red team

The strongest alternative explanation is a shared target implementation error
in the latent pre-clipping model.  That risk is reduced, but not eliminated,
by analytical/autodiff agreement because both use the same model code.  The
next bridge plan must include an independent scalar log-density decomposition
and a declared measure identity before fitting any residual child.
