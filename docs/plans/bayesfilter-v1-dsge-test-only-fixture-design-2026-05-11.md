# Design: BayesFilter v1 DSGE Test-only Fixtures

## Date

2026-05-11

## Scope

This design belongs to the BayesFilter v1 external-compatibility lane.  It does
not edit `/home/chakwong/python`, does not import DSGE modules from
BayesFilter production code, and does not promote SGU to a production filtering
target.

## Decision

Use DSGE targets only as optional live external fixtures until a separate v1
integration lane is opened.

```text
Rotemberg: candidate_test_only_fixture
EZ: metadata_only_test_only_fixture
SGU: blocked_sgu_causal_locality
```

## Rotemberg Fixture Requirements

A future optional live Rotemberg fixture must check:

1. external checkout path and commit hash;
2. structural state names and partition metadata;
3. stochastic and deterministic blocks;
4. DSGE-owned deterministic `dy` completion;
5. deterministic completion residuals before and after completion;
6. singular completion fails closed;
7. observation map and covariance shape;
8. BayesFilter structural value likelihood on the exported target;
9. support/rank diagnostics for the stochastic integration block;
10. no production import from DSGE into BayesFilter.

Allowed claim:

```text
rotemberg_optional_live_structural_fixture_passed
```

Blocked claims:

```text
rotemberg_bayesfilter_production_adapter_ready
rotemberg_hmc_ready
rotemberg_client_default_switch_over_ready
```

## EZ Fixture Requirements

A future optional live EZ fixture must check:

1. external checkout path and commit hash;
2. state names and all-stochastic metadata;
3. transition and observation shape;
4. local analytical stability metadata;
5. absence of BK/QZ determinacy certification;
6. BayesFilter metadata compatibility without determinacy or posterior claims;
7. no production import from DSGE into BayesFilter.

Allowed claim:

```text
ez_optional_live_metadata_fixture_passed
```

Blocked claims:

```text
ez_bk_passed
ez_qz_determinacy_passed
ez_hmc_ready
ez_client_default_switch_over_ready
```

## SGU Stop Rule

SGU remains blocked for production filtering until DSGE supplies a causal local
one-step predictive target:

```text
sgu_causal_filtering_target_passed
```

Timing fixes, residual-closing two-slice projections, and future marginal
utility repairs are diagnostic evidence only.  They do not justify a
BayesFilter production filtering adapter.

## Future Optional Test Shape

Future tests should live in BayesFilter as optional live tests, for example:

```text
tests/test_dsge_rotemberg_optional_live_tf.py
tests/test_dsge_ez_optional_live_tf.py
```

These tests must skip cleanly when `/home/chakwong/python` is unavailable and
must never become required local CI before an explicit v1 integration decision.
