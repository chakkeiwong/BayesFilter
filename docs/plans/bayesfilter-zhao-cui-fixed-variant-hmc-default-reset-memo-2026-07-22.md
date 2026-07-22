# Zhao-Cui Fixed-Variant HMC Default Reset Memo

Date: 2026-07-22

Status: `ACTIVE_POLICY_FIXED_VARIANT_HMC_DEFAULT`

## Decision

All BayesFilter filters are intended for HMC-facing use. Therefore every
admitted Zhao-Cui route must be a fixed-variant value program with an analytical
score of that same program. The high-dimensional source-route default is:

```text
fixed_variant_zhao_cui_source_route
```

The repository-owned policy identifier is:

```text
zhao_cui_fixed_variant_hmc_default_v1
```

The fixed-variant requirement freezes the approximation branch, fitted objects,
discrete schedules, bases/ranks, prepared samples, and other construction
choices while retaining the declared analytical parameter derivative of the
same finite value program. This fixed-branch property is a prerequisite for
an HMC target and for an analytical score claim. It does not require every
model-specific Zhao-Cui adapter to use the same `source_route.py` implementation.

The following model-specific fixed-variant routes remain eligible in their
declared target scopes:

| Route | Scope |
|---|---|
| `zhao_cui_exact_transformed_sv_fixed_branch_tt` | exact-transformed SV |
| `zhao_cui_ksc_mixture_fixed_branch_tt` | KSC transformed-SV mixture |
| `zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt` | generalized-SV prior-mean target |
| `zhao_cui_sir_d18_local_complete_data_manual_component` | scoped SIR local complete-data target only |

## Demoted Routes

The following routes remain readable for historical comparison, debugging, and
source-audit purposes only:

| Route | Active status |
|---|---|
| `adaptive_author_full_sol` | `historical_diagnostic_only` |
| `adaptive_author_pre_sol` | `historical_diagnostic_only` |
| `diagnostic_historical_retained_grid` | `historical_diagnostic_only` |
| `zhao_cui_fixed_adjacent_state_squared_tt_v1` | `historical_diagnostic_only` |

The generic retained-grid functions
`multistate_nonlinear_fixed_design_tt_value_path` and
`multistate_nonlinear_fixed_design_tt_score_path` are not HMC defaults and
cannot be used as Zhao-Cui leaderboard evaluators. The adaptive author
`full_sol`/`pre_sol` programs are source references, not accepted analytical-
gradient runtime evaluators.

No historical artifact is deleted or rewritten. Its route identity remains
part of the historical evidence; it must not be relabeled as the fixed-variant
default after the fact.

## Enforcement

`bayesfilter.highdim.source_route.zhao_cui_hmc_route_policy()` is the
repository-owned selector. With no explicit route it returns the fixed variant.
`require_zhao_cui_hmc_route()` fails closed for every historical, unknown, or
otherwise non-fixed route. Callers must not implement a fallback after that
failure.

The selector is a policy boundary, not a claim that every model-specific fixed
variant is already implemented. A leaderboard row must additionally declare
both the fixed-variant route identity and an implemented full-filtering
evaluator (`hmc_filtering_route_admitted=true`). A local complete-data component
is not sufficient. For example, predator-prey still requires its target-specific
fixed analytical-gradient filtering evaluator. Until it exists, the HMC-facing
request remains blocked; it must not fall back to the retained-grid result.

Consequently, the scalar fixed-branch TT SV routes remain analytical HMC-facing
routes for their declared targets. The LGSSM exact-oracle adapter is not a
Zhao-Cui algorithm result and remains diagnostic. The scoped SIR component is
an analytical HMC target in its conditioned local-complete-data scope, but it
is not a full observed-data filtering score and cannot serve as the comparator
for a GenUT filtering likelihood.

## Evidence Boundary

This reset promotes route identity and selection policy. It does not certify:

- exact likelihood correctness;
- full filtering score correctness;
- source-faithful reproduction of every author implementation detail;
- HMC convergence or posterior correctness;
- GPU/XLA performance or broad model coverage; or
- leaderboard completion for models whose fixed variant is still missing.

Those are model- and scope-specific gates after the fixed route is implemented.

## Verification

The focused policy tests are in:

```text
tests/highdim/test_zhao_cui_hmc_default_route_policy.py
```

They verify fixed-variant default selection, historical demotion, unknown-route
failure, and fail-closed HMC enforcement.
