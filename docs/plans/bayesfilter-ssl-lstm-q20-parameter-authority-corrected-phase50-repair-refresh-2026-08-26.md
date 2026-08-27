# Corrected Parameter-Authority Phase 50 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-result-2026-08-26.md`  
Branch: `support_broadened_does_not_reduce_variability`  
Next version: `v3.3-mode-aware-proposal-geometry`

## Interpretation

Phase 50 passed the fixture, target/status, theta-measure, pairing/replay,
finite-artifact, candidate-validity, and GPU/XLA gates. Its scientific branch
was negative for the tested candidate method: an isotropic `Normal(center,
4^2 I)` component mixed into the independent proposal reduced only the
negative-mode spread, while covariance, theta-mean, root-count, and ESS spreads
were not reduced versus the frozen Phase 49 depth-eight arm.

This is a candidate repair trigger. It is not a continuation veto. The target
is still available in `theta_R4`, the exact q-base/r-proposal identity passed,
and all three paired replay boundaries were reproduced.

## Repair decision

Do not tune `rho` or `tau` blindly. The Phase 50 result is consistent with a
proposal component that has broad isotropic mass but little density where the
two calibrated target-mode neighborhoods are narrow and separated. The next
test will replace only that component with an equal-weight mode-aware Gaussian
mixture using the existing stationary representatives and their stable local
precision matrices:

`s_geom(theta) = 0.5 N(theta; m_minus, kappa^2 C_minus)
                 + 0.5 N(theta; m_plus, kappa^2 C_plus)`,

`r_geom(theta) = (1-rho) q(theta) + rho s_geom(theta)`.

The first geometry hypothesis is `rho=0.50` and `kappa=2.0`. The multiplier is
deliberately larger than one because the raw curvature covariances are local;
it is a reviewed hypothesis, not a claim that the Laplace approximation is a
posterior covariance. The target, q-based bridge, initial clouds, seeds,
resampling schedule, eight mutation steps, and primary spread branch remain
unchanged. The exact correction is

`bridge_q(theta') - bridge_q(theta)
 + log r_geom(theta) - log r_geom(theta')`.

The component covariance and density are evaluated with the repository
full-covariance Gaussian-mixture routines. No density is assigned to an ETPF
or GenUT transform, and no simplified LEDH route is introduced.

## Required gates and stop rules

1. A CPU-hidden fixture verifies normalized mode-mixture density evaluation,
   exact q-base/r-geometry correction at beta zero and beta one, finite states,
   nonzero movement, and the declared eight-step depth.
2. The q=20 runner reproduces Phase 47 initial and identity hashes and loads
   passing Phase 49 and Phase 50 receipts as frozen comparators.
3. Candidate rows are sampled from `r_geom`, while q remains the annealing
   base; current and candidate `log r_geom` values are both used.
4. Invalid target/status candidates are never accepted; all retained rows are
   finite `[256,4]` tensors in `theta_R4`.
5. The report preserves raw replicate rows, compares geometry with Phase 50
   and Phase 49, and makes no statistical ranking from three replicates.

An unavailable target/proposal, exact fixture contradiction, three unrepaired
infrastructure failures, or exhausted remaining campaign pool is a
continuation veto. A failed geometry candidate is a repair trigger and does
not close the parameter-space research direction by itself.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| mode means `m_minus,m_plus` | existing stationary geometry artifact | stale or mode-biased representatives | geometry hash and source status | reviewed warm start |
| local covariances `C_j=P_j^{-1}` | stable positive-definite source-curvature records | curvature is only local and may under-cover tails | eigenvalue/SPD fixture and log-density finiteness | diagnostic construction |
| `kappa=2.0` | bounded scale-up of local covariance; not inherited as a default | still too narrow or overly diffuse | component fractions, acceptance, log-ratio tails | hypothesis |
| `rho=0.50` | Phase 50 paired-support mixture weight | candidate law dominates or is redundant | sampled component fraction and exact r recomputation | hypothesis |
| equal mode weights | two calibrated representatives with comparable diagnostic log mass | true mode masses may differ | retain raw per-mode occupancy; no mass claim | hypothesis |
| q=20, N=256, three banks | frozen Phase 49/50 paired design | finite-replicate uncertainty | raw rows and no ranking | comparator design |
| GPU/XLA boundary | repository policy and prior valid receipts | allocator/compile failure | pre-import memory-growth receipt | required execution choice |

## Skeptical pre-execution audit

The proposed repair passes the audit for a bounded candidate-method experiment:

| Audit question | Finding | Control |
|---|---|---|
| Does the candidate change the target? | No; q remains in the bridge and only the independent proposal changes. | Store q and `r_geom` separately and assert target signature. |
| Is the density normalized? | Yes if both covariance matrices are SPD and mixture weights sum to one. | Full-covariance routine plus fixture normalization/finiteness checks. |
| Is the comparison fair? | Initial clouds, resampling, depth, seeds, and target are frozen. | Reproduce Phase 47 identity hashes; freeze Phase 49/50 reports. |
| Could a proxy become a promotion gate? | Spread, ESS, mode mass, and acceptance are descriptive. | Only algebra, measure, status, replay, finite, device, and artifact gates can pass. |
| Could local curvature be mistaken for posterior truth? | Yes. | Label it a proposal component; explicitly forbid Laplace/posterior claims. |
| Is the experiment under-budgeted? | The measured campaign lower-bound is `28960.22324898499 s`, leaving about `35839.776751015015 s` of the `64800 s` pool before this phase. | Reserve no more than the declared `7200 s` local cap and record actual wall time. |

The audit therefore authorizes the fixture and, if it passes, the GPU boundary
under the unchanged campaign budget.

## Next subplan

`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase51-subplan-2026-08-26.md`

No Phase 50 result authorizes whitening, NeuTra promotion, HMC, canonical LEDH,
posterior claims, or a default change.
