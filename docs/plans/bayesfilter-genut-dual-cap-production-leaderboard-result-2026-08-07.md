# GenUT Dual-Cap Production Leaderboard Result

Date: 2026-08-07  
Plan: `docs/plans/bayesfilter-genut-dual-cap-production-leaderboard-plan-2026-08-07.md`  
Artifact: `docs/benchmarks/artifacts/genut_dual_cap_production_leaderboard_20260807/attempt02/result.json`

## Outcome

The repository-owned selector now resolves `algorithm="default"` to
`dual_cap`, while explicit historical algorithm names retain their own
controls. The repaired six-model GPU/XLA campaign completed all six GenUT
rows with `N=1008`, FP32, TF32 enabled, verified memory growth, and 16 common
claim seeds. The Austria SIR Zhao-Cui cell was intentionally excluded because
it is owned by a separate implementation program.

Verdict: `SIX_MODEL_DUAL_CAP_FEASIBILITY_PASS_NOT_HMC_READY`.

This is a production leaderboard implementation/feasibility result. It is not
a claim that the dual-cap score is exact, statistically superior, or ready for
posterior/HMC/NeuTra promotion.

## Campaign Attempts

| Attempt | Role | Outcome | Repair |
|---|---|---|---|
| `attempt01` | First six-row run | Four rows completed; Austria diagonal baseline failed | Restored Austria's scope-specific tuning seeds `98301,98302`; the first five scopes use `98501,98502` |
| `attempt02` | Repaired final run | Six rows completed; hard-valid pass | No further repair under this contract |

The first failure was a tuning-scope harness error, not evidence against the
dual-cap method. The failed four checkpoints remain preserved in `attempt01`.

## Scope And Runtime

| Scope | Horizon | State dimension | Parameter dimension | Selected pairwise strength | Dual valid |
|---|---:|---:|---:|---:|---:|
| LGSSM | 50 | 3 | 5 | 0.05 | yes |
| KSC SV | 10 | 1 | 2 | 0.02 | yes |
| Exact transformed SV | 10 | 1 | 2 | 0.02 | yes |
| Generalized SV | 10 | 1 | 3 | 0.02 | yes |
| Predator-prey | 20 | 2 | 6 | 0.05 | yes |
| Austria SIR | 20 | 18 | 3 | 0.05 | yes |

The selected dual family is diagonal correction plus four pairwise steps,
pairwise radial RMS cap `2`, standardized coordinate cap `b=.98,p=8`, and
affine source mean/covariance restoration. Pairwise and radial operations are
BayesFilter extensions; this is not a source-faithful Zhao-Cui claim.

## Value And Score Evidence

The following are 16-seed means with sample SD. All method differences are
descriptive; the per-seed paired summaries and MCSEs are in the JSON artifact.

| Model | Diagonal value | Dual-cap value | Dual score mean |
|---|---:|---:|---|
| LGSSM `T=50` | `-136.33349 (0.46797)` | `-136.34637 (0.47538)` | `[5.54990,-3.95812,0.16620,-2.13987,5.56885]` |
| KSC SV `T=10` | `-19.95395 (0.04760)` | `-19.95785 (0.04894)` | `[-0.70681,0.57547]` |
| Exact transformed SV `T=10` | `-19.99441 (0.04883)` | `-19.99795 (0.05048)` | `[-0.70954,0.54087]` |
| Generalized SV `T=10` | `-16.01754 (0.01555)` | `-16.01806 (0.01567)` | `[-0.12431,-0.15417,0.02219]` |
| Predator-prey `T=20` | `-102.73954 (0.29234)` | `-102.72772 (0.30553)` | `[-27.74831,0.07550,-0.08700,1.01471,18.24351,-23.50232]` |
| Austria SIR `T=20` | `-683.36381 (0.63669)` | `-681.76285 (0.51977)` | `[28.67902,-102.12267,11.16216]` |

Austria is the important diagnostic: the dual-cap score dispersion is much
smaller than the diagonal score dispersion in this run, but the score remains
without an exact observed-data authority. The dual value shifts by about
`1.60` log units, roughly eight diagonal value MCSEs, so this is not a
value-preserving change.

For scalar SV, pairwise and radial operations are structurally no-ops; the
coordinate cap is the only dual-cap difference. Cap activity is high across
the scopes (roughly `0.71-0.80` of standardized coordinates), so the cap is
not tail-only in this finite program.

## Derivative Evidence

The result contains centered same-program finite-difference ladders at
`h=0.001,0.01,0.03`. The small-step diagnostic remains poor for LGSSM,
predator-prey, and Austria, while SV/generalized-SV are substantially better.
This diagnostic is explanatory only. It blocks an exact derivative-admission
or HMC-readiness claim; it does not invalidate the finite value/score program
or the six-row feasibility artifact.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Promote `default -> dual_cap` selector | selector tests and replay-preserving names pass | none | future callers outside this runner | integrate selector at additional public factories/CLIs as bounded work | universal statistical superiority |
| Admit six GenUT rows to feasibility leaderboard | all six candidate rows finite and hard-valid | none in `attempt02` | few-seed stochastic comparison and nonlinear score authority | retain artifact as production feasibility baseline | exact likelihood/score |
| Austria dual-cap score | finite and lower descriptive dispersion than diagonal | no hard veto | no exact `T20` observed-data score; value shift | keep as descriptive proposal/filter diagnostic | correct score or posterior |
| HMC/NeuTra production | not tested by this campaign | hard status veto | derivative admission, training, and sequential HMC remain open | separate target-specific NeuTra/HMC program | HMC readiness |
| Zhao-Cui Austria cell | explicitly out of scope | external continuation owned by another agent | observed-data marginal score route | wait for that agent's result | whole-method coverage |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Pass for six GenUT rows in `attempt02`; Austria Zhao-Cui excluded by contract |
| Statistically supported ranking | None; 16 seeds and no exact nonlinear score authority |
| Descriptive-only differences | All value/score gaps, MCSEs, cap activity, runtimes, and FD residuals |
| Default readiness | Selector/feasibility readiness established for this bounded route; broad public-factory migration remains separate |
| HMC/NeuTra readiness | Not established |
| Next evidence needed | Public-factory replay tests, target-specific posterior/HMC validation, and the separately owned Austria Zhao-Cui cell |

## Post-Run Red Team

The strongest alternative explanation is that the dual-cap score changes reflect
the altered finite objective rather than improved score estimation. The large
Austria value shift and high cap activity support that caution. A result that
would overturn the current feasibility conclusion is any replay mismatch,
non-finite row, or failure of the selector/explicit-option compatibility tests.
The weakest evidence remains nonlinear score accuracy: Austria has no exact
score authority and finite-difference behavior is poor at the smallest step in
several models.

