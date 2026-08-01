# Complete High-Dimensional Leaderboard Phase 0 Owner-Authority Amendment

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `OWNER_APPROVED_AUTHORITY_OVERLAY`

## Purpose

This additive overlay resolves the two target/source decisions that blocked the
local continuation at Phase 1 P1-A/P1-B. It does not erase or retroactively
change the reviewed historical records. The Phase 0 machine freeze must be
regenerated and independently audited against this overlay before P1-A may be
reissued.

Historical identities preserved by this amendment:

- original Phase 0 freeze SHA-256:
  `4115ef55114ffd73255363f0c62c4a19dd85d7ca3241d002c48409cb9004f878`;
- original reviewed Phase 1 subplan SHA-256:
  `ff75b73fdbc2f75c0d5f05c0ac835fdfec69cc7ccd1448b47c6f66b2d9ebb62b`;
- failed P1-A receipt SHA-256:
  `3299d3b797aa41b028fe77ec4d3aabb639fa176da73767fe3b4905a2d614ff67`.

## Owner Decisions

### Fixed SIR Observation Identity

For the main row `zhao_cui_spatial_sir_austria_j9_T20`, the authoritative
observations are the deterministic bytes produced by the current BayesFilter
`_sir_dataset(81103)` route and sliced to the row's declared `T=20` horizon.
The seed is part of the target-generation identity, not an LEDH execution seed.

This identity is explicitly **not** a claim that those bytes reproduce MATLAB
`rng(1)`, the author companion-code observations, or an author-distributed data
file. The exact authority token is:

`fixed_bayesfilter_sir_observations_from_dataset_seed_81103_not_author_matlab_rng1_reproduction`

This decision supersedes only the prior token
`fixed_austria_j9_source_observations_no_synthetic_seed_declared` and any prose
that said no synthetic target-generation seed was declared for this row.

### Zhao-Cui Exact-Row Classifications

The following exact leaderboard adapters are owner-approved as
`extension_or_invention`. None may be called `source_faithful`,
`fixed_hmc_adaptation`, a paper reproduction, or an author-code reproduction:

| Row | Binding classification |
| --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `extension_or_invention` |
| `zhao_cui_sv_actual_nongaussian_T1000` | `extension_or_invention` |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `extension_or_invention` |
| `zhao_cui_spatial_sir_austria_j9_T20` | `extension_or_invention` |
| `zhao_cui_predator_prey_T20` | `extension_or_invention` |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `extension_or_invention` |

For predator-prey, this approval specifically covers the exact BayesFilter
physical target `(r,K,a,s,u,v)=(0.6,114,25,0.3,0.5,0.5)` and its current model
semantics even if the pinned author code encodes a different parameter order,
scaling, or physical model from the paper/BayesFilter row. The P1-B ledger must
still record the exact discrepancy and source anchors; approval does not turn
the discrepancy into source faithfulness.

## Boundary

This amendment authorizes only:

1. regeneration and independent audit of the Phase 0 machine freeze;
2. regeneration and checking of the P1-A canonical-target artifact;
3. completion of the P1-B source-availability/classification ledger and its
   receipt; and
4. continuation to P1-C only after both superseding gate receipts pass.

It does not authorize a target substitution beyond the decisions above, a
public/default-policy change, a GPU benchmark, a leaderboard cell admission,
release, source-faithful language, ranking, HMC/posterior correctness, or a
scientific-validity claim.

The retained-grid Zhao-Cui route remains diagnostic/historical only. Any later
implementation still requires the fixed-variant production route, exact target
signatures, fail-closed paired value/score evidence, and all downstream gates.

## Evidence Classification

| Item | Classification |
| --- | --- |
| Owner decision | Authority for the exact target and adapter labels above |
| Current deterministic SIR bytes | Engineering target identity after regenerated P1-A checks |
| Paper and pinned source comparison | Source-availability and mismatch evidence only |
| Extension approval | Permission to implement the labeled adapters, not evidence of correctness |
| P1-A/P1-B pass | Pre-implementation gate evidence only |

## Stop Conditions

Stop before P1-C if the regenerated SIR bytes do not equal the independently
reconstructed `_sir_dataset(81103)` bytes, if any adapter is labeled more
strongly than `extension_or_invention`, if an exact source discrepancy is
hidden, if a historical hash is overwritten without a supersession record, or
if the frozen benchmark harness drifts before the superseding P1-A receipt.

