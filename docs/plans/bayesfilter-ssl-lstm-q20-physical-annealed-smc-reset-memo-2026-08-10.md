# SSL-LSTM q=20 physical annealed-SMC reset memo (2026-08-10)

## Current state

The mass-repair lane is complete.

- Direct importance sampling failed and its `0.4683` estimate remains unsupported.
- Sparse AIS repaired 64-bridge weight concentration but failed sign movement and
  schedule stability; its `0.5253` estimate remains unsupported.
- Adaptive global-resampling SMC passed all material gates for the two known
  proposal-supported sign regions.
- Eight central estimates average `0.47087`, with 95% independent-batch interval
  `[0.40573,0.53602]`.
- Every central terminal ESS fraction is at least `0.8783`, every maximum weight is
  at most `0.03854`, and every run retains at least 23 initial roots from each sign.
- cESS `0.70` vs `0.80` mass difference is `0.05192`, below the `0.08` gate.
- The original manifests linked 780 child receipts plus two aggregates.  A flat
  stage-map defect hid 210 same-named pre-resampling entries, although their
  immutable files existed.  Post-run recovery verified all 990 child tensors and
  two aggregates, including weights, ancestry, stage continuity, and reproduced
  estimates.  Material wall time was `2487.58 s`.
- No HMC mutation changed sign.  SMC authority is limited to the two known regions;
  exhaustive mode discovery is not established.

Material result:

`docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-result-2026-08-10.md`

Terminal artifact:

`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/material.json`

Receipt recovery artifact:

`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/receipt-recovery-v1.json`

Recovery SHA-256:

`3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97`

Future SMC stages use schema v2 with nested `receipts.pre` and `receipts.post`.
Do not rewrite the historical v1 stage JSON; bind it to the recovery inventory.

## Next lane

Return to physical replica-exchange transition validation.

1. Keep the exact physical target and chart.  Do not return to the failed NeuTra
   coordinates.
2. Design a travel-focused ladder/schedule using the existing six-temperature
   candidate as the baseline.  The prior 12-transition run had valid swaps, five
   hot local-HMC sign changes, and two cold sign transitions but zero full round
   trips.
3. Require repeated cold-hot-cold identity round trips, hot sign forgetting, valid
   target/status telemetry, and modern cold-chain convergence diagnostics under
   frozen settings before a posterior archive.
4. Preserve SMC mass evidence separately from transition samples.  Do not treat raw
   replica occupancy as a replacement weight estimate.
5. Only after an eligible posterior archive exists should NeuTra be retrained from
   globally weighted coverage and the posterior-predictive output-distribution test
   be run.

## Claim boundary

The current result supports: finite relative mass over the two known sign regions,
approximately `0.471` negative with measured interval `[0.406,0.536]`.

It does not support: exhaustive full-posterior mass, a stationary HMC archive,
NeuTra convergence, or predictive equivalence.

## Resilient workflow

Continue using detached transient user services for runs longer than about one
minute.  Require unique unit names, versioned output roots, explicit service/runner
caps, GPU-hidden CPU environment, atomic progress, append-only logs, receipt hashes,
and overwrite refusal.  This campaign survived the session-stream risk without
losing any completed child or active computation.
