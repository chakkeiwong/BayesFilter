# SSL-LSTM q=20 physical global repair reset memo (2026-08-10)

## State

Stage 2 is complete.

- Direct corrected importance sampling failed: central mean `0.468`, interval
  `[0.302,0.635]`, maximum normalized weight `0.572`, and strong covariance-scale
  sensitivity.  Do not quote `0.468` as a posterior mode weight.
- The physical chart passed local HMC checks in both known regions.
- Six-temperature physical replica exchange produced five pre-swap hot local-HMC
  sign changes and two cold sign transitions, with finite/status-valid traces.
- No replica completed a cold-hot-cold round trip.  The transition candidate is
  viable but not globally validated.
- No posterior archive exists and predictive validation remains blocked.

Terminal result:

`docs/plans/bayesfilter-ssl-lstm-q20-physical-global-repair-result-2026-08-10.md`

Artifacts:

`docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/`

## Next repair

1. Run annealed importance sampling from the normalized two-local-Gaussian mixture
   to the exact physical target.  Use independent worker batches, preserve AIS log
   weights and terminal signs, and require weight ESS, maximum normalized weight,
   independent-batch uncertainty, and schedule sensitivity.
2. If AIS fails, use annealed SMC with conditional-ESS temperature placement,
   resampling, ancestry tracking, and fixed-HMC rejuvenation.  Do not average failed
   direct-IS and AIS point estimates.
3. After stable mass evidence exists, repair temperature travel.  The current ladder
   communicates and hot replicas change sign, so the smallest repair is more travel
   opportunities or a reviewed nonreversible even/odd schedule, not a return to
   NeuTra coordinates.
4. Require repeated full round trips and cold convergence under frozen settings
   before issuing a posterior archive.
5. Only then train/retrain NeuTra from globally weighted coverage and run the
   posterior-predictive distribution diagnostic.

## Failure repairs preserved

- `weight-canary.log`: launch had `CPUQuota=25%`; scientific output valid, timing
  ineligible for capacity estimates.
- `weights.log`: all 1,200 target rows completed; terminal aggregation failed on
  `tf.constant(list_of_scalar_tensors)`.  `weights-aggregate` verified 84 receipts,
  recomputed zero target rows, and wrote `weights.json` using `tf.stack`.
- `physical-local.log`: CPU XLA rejected an in-graph fixed `slogdet`.  The constant
  log determinant moved outside the graph.
- `physical-local-r2.log`: repaired local gate passed.
- `physical-transition.log`: detached transition completed in `4600.29 s` and wrote
  the terminal artifact.

## Resilient execution

Long runs use transient user services with unique unit names, explicit timeouts,
absolute log paths, runner-side caps, atomic progress artifacts, and overwrite
refusal.  Monitor with short artifact reads and occasional trusted `systemctl --user
show`.  Sandbox-local `ps` cannot be trusted for service liveness because the service
may live in a different process namespace.

