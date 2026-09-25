# Experiment plan: is the 1D T=3 LGSSM score error bias or Monte Carlo variance?

Date: 2026-08-29
Status: ACTIVE
Scope: diagnostic (no promotion, no default change)

## Question

On the 1D AR(1) linear Gaussian model at `T=3` with Contract-E reset, the
analytical score in the `phi` direction was reported 31-44% away from the exact
Kalman score in single-seed runs. Is that a persistent bias of the finite
LEDH-PFPF-OT score, or is it Monte Carlo variance of a single replication?

A first 10-seed run at `N=3000` on a different parameter vector returned score
error mean `+0.0253` with standard error `0.1531` (`z=0.17`), i.e. per-seed
score standard deviation of roughly the same magnitude as the score itself.
That observation motivates a paired ladder on the exact ablation fixture.

## Mechanism under test

The analytical score is the total derivative of the executed finite-N filter
value, already verified against central finite differences of that same value
to six decimals. Therefore any disagreement with the exact Kalman score is a
property of the finite value program, decomposable into

- a bias term: `E[score_hat] - score_exact`, and
- a variance term: `Var[score_hat]`.

The value-lane ablation established a value bias of `+0.136 +/- 0.060` at
`N=1008` and `+0.046 +/- 0.018` at `N=2016` on this fixture. The score's
bias/variance split has not been measured.

## Fixture (matched to the ablation artifact)

- model: 1D AR(1) linear Gaussian, `theta = [phi, sigma_w, sigma_v] = [0.9, 0.6, 0.8]`
- horizon: `T = 3`
- observations: `[-0.9911206904960701, -2.7246862541942303, -0.30490733274540416]`
- exact Kalman value: `-5.903444873434653`
- score direction: `phi` (coordinate 0), exact Kalman score by central FD at `h=1e-5`
- reset: `contract_e`, `epsilon=1.0`, Sinkhorn/balance `8/8`, ridge `1e-5`
- flow substeps: `12`; `correction_steps=1`, `pairwise_steps=1`; `annealed_stages=1`
- dtype `float64`; GPU intentionally hidden (`CUDA_VISIBLE_DEVICES=-1`), CPU-only
- particle ladder: `N in {504, 1008, 2016, 3000}`; all `N <= 3000`, so the
  exact-divisor chunk policy gives `K = N` and the dense reset route is used
- seeds: the same 16 seed values at every `N` (paired across the ladder)

## Diagnostics and their roles

| Diagnostic | Role |
|---|---|
| score error mean and standard error per `N` | primary descriptive estimate of bias |
| per-seed score standard deviation per `N` | primary variance estimate |
| `z = mean / SE` per `N` | screen for separation from zero (2-SE) |
| value error mean and SE per `N` | explanatory: links to the ablation result |
| ratio of score SD to `|score_exact|` | explanatory: is a single seed informative |

## Success criteria

This run answers the question if, at every rung, the score error mean, its SE,
and the per-seed SD are all recorded, and the paired seed set is identical
across rungs. It does not require any rung to pass a threshold.

## What would change the next step

- If the score error mean is not separated from zero at any rung while the
  per-seed SD is large and roughly `N`-independent, the score problem is
  primarily variance, and the next step is a variance-reduction question
  (common random numbers, more replications, or a lower-variance estimator),
  not a bias repair.
- If the score error mean is separated from zero and shrinks like the value
  bias, the score inherits the value bias, and the next step is the value-bias
  repair already indicated by the ablation.
- If the score error mean is separated from zero and does not shrink with `N`,
  that points at a fixed finite-program defect (route, control family, or
  tangent composition), which is the region the 2026-08-29 codex audit flagged
  as C1-C4.

## Failure modes and pre-mortem

- Nonfinite or wildly heavy-tailed score draws would make the mean and SE
  uninformative; the artifact records per-seed values so this is visible.
- 16 seeds bound the SE only loosely; a null result here is "not separated at
  this precision", never "unbiased".
- A single observation path is one realization of the data; nothing here
  generalizes across observation paths, horizons, dimensions, or directions.

## What will not be concluded

No claim of unbiasedness, convergence, Kalman equivalence, HMC readiness,
statistical superiority, or default readiness. No per-model claim, since this
scope has no score-lane tuning artifact.

## Artifacts

- runner: `docs/benchmarks/lgssm_1d_t3_score_variance_ladder.py`
- result: `docs/benchmarks/lgssm_1d_t3_score_variance_ladder.json`
