# Scalar SSL-LSTM Filtering HMC Validation Phase 1 - 2026-07-09

## Decision

- phase1_short_chain_screen_passed: `False`
- vetoes: `['seed_2_acceptance_outside_phase1_screen']`
- passed_seed_count: `3` / `3`
- zero_divergence_claim_made: `False`
- next_justified_action: write Phase 1 blocker/repair result before Phase 2

## Phase 1 Gate

- acceptance rates: `[0.9375, 0.75, 1.0]`
- acceptance range: `0.75` to `1.0`
- native divergence statuses: `['not_exposed_by_kernel', 'not_exposed_by_kernel', 'not_exposed_by_kernel']`
- native divergence interpretation: native divergence unavailable for at least one seed; unavailable is not zero divergences
- log-accept threshold used as native divergence: `False`

## Aggregate Summary

- max abs u by seed: `[3.9364129599327216, 6.603472829746424, 10.427120225910855]`
- target log-prob overall range: `-42.653608440660555` to `-37.81306363456166`
- log-accept max abs by seed: `[2.1187000672423397, 48.620365994974954, 0.580877228359129]`
- interpretation: descriptive only; no ranking, convergence, posterior correctness, or default-readiness claim

## Seed Rows

| seed index | seed | status | vetoes | acceptance | finite samples | native divergence |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | [20260709, 6101] | passed_short_smoke | none | 0.9375 | 16 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 1 | [20260709, 6102] | passed_short_smoke | none | 0.75 | 16 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |
| 2 | [20260709, 6103] | passed_short_smoke | none | 1.0 | 16 | {'available': False, 'status': 'not_exposed_by_kernel', 'nonclaim': 'unavailable native divergence telemetry is not zero divergences'} |

## Inference Status

| field | value |
| --- | --- |
| hard_veto_screen | failed |
| native_divergence | native divergence unavailable for at least one seed; unavailable is not zero divergences |
| zero_divergence_claim | not made |
| statistically_supported_ranking | none; no method comparison and no uncertainty interval |
| descriptive_only_differences | per-seed acceptance, target-log-prob range, log-accept range, sample range, and runtime |
| default_readiness | not assessed |
| gpu_xla_readiness | not assessed; CPU-hidden debug/reference exception |
| hmc_readiness | not assessed; Phase 1 finite/acceptance screen only |
| next_evidence_needed | reviewed Phase 2 scalar reference agreement before any posterior agreement interpretation |

## Nonclaims

- Phase 1 finite/acceptance short-chain validation screen only
- not HMC readiness evidence
- not HMC convergence evidence
- not posterior correctness evidence
- not a zero-divergence claim when native divergence is unavailable
- not a tuned-kernel claim
- not sampler superiority evidence
- not statistically supported ranking evidence
- not GPU/XLA production-readiness evidence
- not default-readiness evidence
- not Zhao-Cui source-faithfulness evidence
