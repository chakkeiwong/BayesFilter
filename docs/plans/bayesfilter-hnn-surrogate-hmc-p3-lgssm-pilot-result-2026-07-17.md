# P3 Result: Exact-Likelihood LGSSM Pilot

Date: 2026-07-17

Decision: `HNN_VALIDITY_CONFIRMED_ONE_EXACT_LGSSM_FIXTURE`

Performance decision: `DESCRIPTIVE_PERFORMANCE_SCREEN_PASS`

## Outcome

The corrected learned-force kernel passed the prospective health, modern
rank/folded R-hat, bulk/tail ESS, tuned plain-HMC agreement, generating-truth,
and full joint-energy gates on the registered 18-parameter, 120-observation
exact-Kalman-likelihood LGSSM fixture.

The separate endpoint program uses the same posterior scalar without computing
parameter gradients. Its maximum transformed-target parity error was
`4.2633e-14` on the final parity set under a predeclared `5e-7` float64 GPU/XLA
tolerance. Every transition used one new endpoint batch evaluation. Archived
traces reconstruct `delta_h` from both endpoint potentials and kinetic energies
with maximum error `0.0` in all three arms.

## Arm Results

| Arm | Step/L | Warm-up | Retained | Acceptance | Max R-hat | Min bulk ESS | Min tail ESS | Sampling seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero residual | `0.8/10` | 2,000 | 2,000 | 0.6474 | 1.00688 | 3433 | 2000 | 232.3 |
| Learned residual | `0.8/6` | 2,000 | 2,000 | 0.6784 | 1.00673 | 2069 | 3005 | 227.2 |
| True gradient | `0.4/6` | 2,000 | 1,000 | 0.9420 | 1.00940 | 7865 | 1582 | 1792.3 |

All three arms passed their hard sampler gates. The learned arm agreed with
the preserved tuned plain-HMC posterior with maximum combined-MCSE discrepancy
`1.5463`; maximum absolute mean-minus-truth in posterior-SD units was `1.6361`.
Minimum posterior truth-tail was `0.06637`, with no marginal or severe
parameters.

## Training

The 16-arm frozen P0 screen selected fresh recipe
`w4_l3_lr0.005_b512`. Heldout standardized force RMSE was `0.15288`, centered
standardized potential RMSE was `0.11411`, and mean force cosine was `0.99160`.
Tail force error was larger than central force error, as expected; these
metrics remain explanatory and did not promote the force.

The learned arm's reuse-scenario seconds per minimum bulk ESS was `0.2483`,
versus `0.6193` for same-chart true-gradient HMC, and its sampling-only value
was `0.1098` versus `0.2279`. This satisfies the predeclared descriptive screen.
It is one campaign, not a statistically supported superiority claim.

The zero-residual arm was descriptively better than the learned arm on both
reuse and sampling seconds per minimum bulk ESS (`0.1367` and `0.06767`). Thus
P3 establishes that the corrected learned kernel is valid and can avoid exact
interior-gradient cost; it does not show that residual learning improves this
already strong NeuTra chart.

## Repair History

Attempt 1 never reached training or sampling. Supervision accidentally routed
through the base adapter's scalar `tf.map_fn` score surface. The repair bound
the repository's admitted batch-native LGSSM score kernel explicitly, including
transport score and log-Jacobian score pullback. A 128-row trusted GPU/XLA
regression completed in 3.48 seconds with finite output. Attempt 2 retained the
same target, data, criteria, seeds, hardware, and budget and passed.

## Decision Table

| Field | Status |
| --- | --- |
| Primary criterion | passed |
| Target/value-only parity | passed |
| Full energy correction | passed, exact archived identity |
| Learned sampler health/convergence | passed |
| Plain-HMC posterior agreement | passed |
| Generating-truth diagnostic | passed one seed |
| Descriptive performance vs true gradient | passed |
| Residual learning benefit vs zero residual | not demonstrated |
| Main uncertainty | one favorably truth-centered exact-likelihood fixture |
| Next justified action | test independent nonlinear Tier A targets |
| Not concluded | broad validity, statistical superiority, residual-learning necessity, or default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | clear |
| Statistically supported ranking | none |
| Descriptive-only differences | all runtime and ESS-normalized cost differences |
| Default readiness | false |
| Next evidence | predator-prey UKF and SGQF under independent target-specific protocols |

## Post-Run Red Team

The strongest alternative explanation is that the existing NeuTra chart has
already removed nearly all useful residual geometry. The zero-residual result
supports that explanation. P4 therefore keeps zero residual mandatory and does
not assume the selected LGSSM architecture or learning rate will help another
posterior.

True-gradient tuning was unusually expensive because each static candidate
compiled separately and evaluated exact filtering gradients at every kick.
That measured cost is real for this harness but is not a universal lower bound
for optimized exact-gradient HMC. It must remain a descriptive comparison.

## P4 Handoff Review

P4 remains justified as a generality test, not because P3 proved learning is
better than a Gaussian chart force. UKF and SGQF cells retain separate target,
training, tuning, truth, and cost decisions. Their endpoint programs must be
value-only and parity-certified before sampling. The P3 measured wall time was
1.63 hours, leaving the P4 ceiling and total campaign budget intact.

Status: `CONTINUE_TO_P4`.

