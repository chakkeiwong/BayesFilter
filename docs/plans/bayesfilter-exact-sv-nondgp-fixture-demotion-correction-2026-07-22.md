# Exact-SV Non-DGP Fixture Demotion Correction

Date: 2026-07-22

Status: `ACTIVE_CORRECTION_NONDGP_SV_EVIDENCE_REVOKED`

## Owner Decision And Direct Verdict

The observation sequences produced directly as

```text
z_t ~ Normal(0,1)
```

by `run_cubature_genut_exact_sv_claim_diagnostic.py`,
`run_cubature_genut_exact_sv_n_scaling.py`, and
`run_cubature_exact_sv_score_ladder.py` are not draws from the declared
stochastic-volatility data-generating process

```text
h_t = gamma h_(t-1) + sigma eta_t
z_t = h_t + 2 log(beta) + log(e_t^2).
```

Using those direct-normal transformed observations for an SV accuracy, bias,
tuning, score, method-ranking, or promotion conclusion was a planning error.
Those conclusions are **wrong relative to an SV scientific claim** and are
revoked.  The data are out-of-distribution arbitrary-input fixtures, not SV
datasets, replications, validation data, or holdouts.

This correction supersedes every 2026-07-21 plan, result, report, and artifact
that treated these observations as scientific exact-SV evidence.  Historical
files remain preserved for provenance; their old status fields do not override
this correction.

## Eligibility Boundary

| Evidence from the non-DGP fixtures | Status after correction |
|---|---|
| TensorFlow/XLA execution and device placement | Eligible engineering evidence only |
| Finiteness, memory growth, allocator use, runtime, replay | Eligible engineering evidence only |
| Sinkhorn/reset residual mechanics | Eligible engineering evidence only |
| Recursive score versus finite difference of the same finite scalar | Eligible derivative-mechanics evidence only |
| Particle-seed or antithetic variance conditional on the arbitrary input | Historical engineering diagnostic only; irrelevant to SV performance |
| Dense conditional likelihood/score calculation | Mathematically defined for that arbitrary input, but irrelevant to SV-DGP performance |
| Control tuning against dense value on the arbitrary input | Ineligible for every active SV scope |
| Value accuracy or bias conclusion | Revoked and ineligible |
| Score accuracy, bias, OPG, or per-time mechanism conclusion | Revoked and ineligible |
| Particle-count scientific scaling conclusion | Revoked and ineligible |
| Cubature/GenUT/Contract E scientific comparison | Revoked and ineligible |
| Default, leaderboard, MLE, HMC, or nonlinear-model decision | Forbidden |

## Affected Scientific-Looking Campaigns

The following are demoted to
`HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`:

- `docs/plans/bayesfilter-cubature-genut-model-claim-plan-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-model-claim-phase3-result-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-exact-sv-n1000-plan-2026-07-21.md`;
- `docs/plans/bayesfilter-cubature-genut-exact-sv-n1000-result-2026-07-21.md`;
- `docs/plans/bayesfilter-exact-sv-cubature-score-bias-variance-ladder-plan-2026-07-21.md`;
- `docs/plans/bayesfilter-exact-sv-cubature-score-bias-variance-ladder-result-2026-07-21.md`;
- their JSON artifacts under `cubature_genut_model_claim_20260721/`,
  `cubature_genut_exact_sv_n1000_20260721/`, and
  `cubature_exact_sv_score_ladder_20260721/`.

The Phase 3 adapter pilot, Phase 4 antithetic diagnostic, and Phase 5 GPU/XLA
fixtures retain only their explicitly mechanical evidence.  They are not SV
scientific evidence and cannot nominate a variance policy for SV.

## Corrected Fixed-GenUT Interpretation

The paired artifact

`docs/benchmarks/artifacts/exact_sv_fixed_gaussian_genut_paired_20260721/attempt02/result.json`

contains two data arms.  They are no longer coequal:

- `original`: non-DGP direct-normal fixture; excluded from all scientific
  criteria and interpretation;
- `fresh_dgp`: genuine stationary SV-DGP simulation; the only scientifically
  eligible arm.

Therefore the old cross-dataset mechanism criterion and its failure are
invalid.  The corrected result is:

`SINGLE_DGP_GENUT_SIGNAL_PROMISING_REPLICATION_REQUIRED`.

On the single genuine DGP sequence at `N=1998`, fixed Gaussian GenUT reduced
the absolute gamma mean-score error and the mean per-seed OPG diagnostic under
both 4 and 8 Sinkhorn steps, while its value interval included the dense value.
This is one-dataset evidence only.  It does not establish a population ranking,
default readiness, MLE/HMC readiness, or broad SV validity.

## Active SV Evidence Contract

All future claim-bearing SV experiments must generate or load observations
from the declared SV DGP or a documented real-data observation source.  A
fixture is DGP-valid only when its manifest binds at least:

- parameter point or data-source identity;
- initial-state law;
- transition equation and innovation seed/source;
- observation equation and observation-noise seed/source;
- raw/transformed observation relation;
- horizon and exact serialized observation hash.

Directly generating transformed observations from an unrelated convenience
distribution is allowed only in paths and artifacts explicitly named
`historical_nondgp_engineering_only`.  Such a path must fail closed unless the
caller supplies the explicit historical-engineering flag.  It must never emit
`target_accuracy_valid`, `score_accuracy_valid`, `hard_valid` as a scientific
accuracy status, or a tuning artifact eligible for an SV claim.

## Next Scientific Action

Run a multi-DGP paired Cubature versus fixed-Gaussian-GenUT replication with:

- independently generated stationary SV datasets as the outer sampling unit;
- common particle seeds nested within dataset;
- frozen shared controls and rank coupling;
- dense value/score references for each dataset;
- dataset-level uncertainty for method differences;
- posterior-region value-difference diagnostics only after the DGP result
  reproduces.

The non-DGP fixture must not appear in the plan, criterion, veto, table, or
decision except as historical provenance.

