# Corrected Parameter-Authority Phase 52 Repair and Refresh

Date: `2026-08-28`  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-result-2026-08-28.md`  
Branch: `fresh_geometry_uncertainty_incompatible`  
Campaign state: `STOP_NEXT_VALID_PHASE_UNDER_BUDGET`

## What failed

The implemented target, measure, proposal correction, pairing, numerical
kernel, device policy, and artifacts passed. The frozen geometry candidate
failed its scientific promotion criterion. In the report notation,

`D_m = range_m(geometry) - range_m(support)`.

The gate required `U_m <= 0` for every primary metric, where `U_m` is the 95%
bootstrap upper endpoint. For `theta_mean_0`, `U_m = 0.1823817845 > 0`.
Therefore the conjunction is false even though the point estimate is negative
and the other two primary upper endpoints are nonpositive.

This is not an implementation failure. It is also not evidence that the
broader proposal idea is impossible. It rejects only the frozen equal-weight
geometry with `rho=0.50`, `kappa=2.0`, inherited mode representatives, and
local curvature covariances as a Phase 52 spread-reduction nominee.

## Why the favorable diagnostics are insufficient

The geometry arm produced a smaller covariance range and a higher ESS level in
all six paired rows. Neither quantity identifies target accuracy. A cloud can
have compact covariance because it is underdispersed, and normalized weights
can have high ESS while all particles occupy the wrong region. Likewise, a
stable negative-mode fraction is useful only if the correct target mode mass
is independently known.

The equal-weight local mixture encodes two neighborhoods but does not identify
their global probability masses or the density between them. Exact
independent-MH correction preserves the declared target as the invariant law
of each finite kernel, conditional on valid evaluation, but it does not imply
that eight proposals per stage erase finite-run initialization and mode-mass
error. The Phase 52 variability is therefore compatible with a correct finite
kernel and an inadequate finite proposal design.

## Repair decision

Do not:

1. tune `rho`, `kappa`, mode weights, or representatives on the six Phase 52
   claim banks;
2. discard the two unfavorable banks or pool them with Phase 50/51 rows;
3. replace the failed theta-mean gate with covariance or ESS after seeing the
   result;
4. train NeuTra on these rows or launch HMC; or
5. describe the geometry arm as superior, whitening, or posterior-correct.

The next valid research phase would be a newly reviewed,
reference-anchored proposal-calibration protocol:

1. establish an independent target-level reference for selected posterior
   moments and mode mass in `theta_R4`, with its own convergence and uncertainty
   checks;
2. generate calibration-only particle banks and compare a bounded proposal
   family, including the isotropic support baseline and mode-aware candidates;
3. freeze all proposal parameters and selection rules before validation;
4. generate untouched validation banks with fresh seeds;
5. use reference error as the primary criterion and retain ESS, acceptance,
   covariance, roots, and range diagnostics as explanatory quantities; and
6. require downstream NeuTra validation before any learned-transport claim.

This design prevents the current proxy metrics from becoming substitutes for
posterior accuracy. It also gives ETPF, GenUT covariance correction, or a
particle-flow proposal a legitimate role only if its output is evaluated
against the same target-level reference; empirical moment restoration alone
cannot pass the phase.

## Budget blocker

The conservative remaining campaign pool is `4483.656990259071 s`. The prior
three-bank Phase 51 boundary used `5558.9085 s`, and the six-bank Phase 52
boundary used `23316.68617718201 s`. A disjoint calibration/validation phase
cannot fit the remaining pool even before reference construction, fixture
work, pilot generation, and reporting.

Running a smaller post hoc screen would be underpowered and would reuse the
held-out evidence. The current campaign therefore stops at a real blocker:
the next scientifically valid phase is under-budgeted. A future phase requires
a new concise experiment plan, an independently justified compute budget, and
fresh versioned roots. It does not require reviving the retired approval-token
ceremony.

## Terminal classification

| Question | Answer |
|---|---|
| Was the harness invalidated? | no |
| Was the target or theta measure invalidated? | no |
| Was the finite MH math contradicted? | no |
| Did the current candidate pass promotion? | no |
| Is a statistically supported ranking available? | no |
| Is NeuTra whitening or HMC admitted? | no |
| Why stop now? | insufficient remaining budget for the next valid disjoint protocol |

No Phase 53 subplan is activated under the current campaign.
