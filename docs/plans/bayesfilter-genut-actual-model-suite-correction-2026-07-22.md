# GenUT Actual-Model Suite Correction

Date: 2026-07-22
Status: `active_suite_classification`

## Correction

`reduced_continuous_preclip_sir_j1_v1` was accidentally treated as an actual
model in the GenUT feasibility sequence. That classification is wrong. The
constructor defines a `J=1` fixture chosen so the clipping boundary has material
probability. It changes the dimension, graph, population scale, recovery rate,
covariances, and RK4 convention relative to Austria SIR. It is useful only for
local transition/tangent and clipping-boundary mechanics tests.

The actual nonlinear model omitted only from that ad hoc GenUT sequence is the
already-implemented and already-tested Chapter 18b five-parameter structural
model. It was not missing from BayesFilter's model tests or executable registry.
No new structural model should be invented, reimplemented, or treated as a new
suite addition.

## Active classification

| Target | Model status | Existing implementation/evidence | GenUT-suite status |
|---|---|---|---|
| `artificial_reduced_preclip_sir_j1_mechanics_fixture_v1` | artificial mechanics fixture | `bayesfilter/highdim/sir_latent_preclip_reference_tf.py`; dense-grid and tangent tests | permanently ineligible for actual-model suite, feasibility, leaderboard, default, or HMC evidence |
| `zhao_cui_spatial_sir_austria_j9_T20` | actual Austria SIR row | `bayesfilter/highdim/models.py`; canonical generated dataset and existing SIR routes | actual-model entry; GenUT integration not supplied by the reduced fixture |
| `STR-UKF-five-probit-T100-structural-innovation-v1` | existing actual Chapter 18b structural target, already in the tested suite and executable registry | `bayesfilter/testing/structural_ukf_neutra_target_design_tf.py`; frozen T=100 dataset; manual analytical value/score; CPU/GPU XLA and negative-control tests; executable `STR-UKF` registry cell | retain existing suite status; reuse the model and tested comparator unchanged; only a target-faithful GenUT route is absent |

## Chapter 18b target

The existing target is

```text
m_t = rho * m_(t-1) + sigma * epsilon_t,  epsilon_t ~ N(0,1)
k_t = phi * k_(t-1) + gamma * m_t^2
y_t = m_t + k_t + e_t,                    e_t ~ N(0,R)
```

with physical parameter order `(rho,sigma,phi,gamma,R)`, truth
`(0.8,0.5,0.7,0.4,0.25)`, one stochastic innovation, a deterministically
completed second state, and frozen `T=100` observations. Every propagated point
must obey `k_t-phi*k_(t-1)-gamma*m_t^2=0`; artificial independent noise in `k_t`
is a wrong-model negative control.

The GenUT integration must use the existing structural transition and frozen
dataset. It must not use the older four-parameter Contract-E structural fixture,
add independent `k` noise, or create another convenience DGP.

## Evidence consequences

- Both `genut_three_model_simple_feasibility_20260722` attempts have invalid
  actual-suite composition. Their generalized-SV and KSC phase files remain
  model-specific diagnostic evidence; their reduced-SIR phase is mechanics-only.
- Aggregate statuses such as `pass_all_three` are revoked relative to an
  actual-model suite claim.
- Historical reduced-SIR campaigns remain readable but cannot nominate tuning,
  claim, leaderboard, default, or HMC work.
- Future GenUT comparison plans must preserve the Chapter 18b target's existing
  suite status and must not count reduced SIR as a replacement model.

## Implementation guard

`reduced_sir_candidate_adapter` now fails closed unless the caller explicitly
passes `mechanics_fixture_only=True`. Historical fixture scripts use that opt-in.
The mistaken three-model runner is retired and fails before execution. A
regression test verifies both the reduced-SIR fail-closed behavior and the
existing `STR-UKF` executable registry entry.

## Next justified action

The GenUT interface is now adapted to the existing Chapter 18b target through
shared primitives, and the `N=1002,T=100` scope was tuned with structural
residual as a hard veto. Capacity, structural wiring, route identity, and the
representative recursive-score audit passed, but the fresh full-horizon
multi-seed claim produced a non-finite value or score. The GenUT leaderboard
extension therefore includes `STR-UKF` as
`blocked_nonfinite_full_horizon_multiseed_claim`; it is not admitted. The next
action is a new diagnostic campaign that serializes the first failing seed/time
and repairs the numerical instability before another untouched claim.
