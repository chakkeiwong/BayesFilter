# Predator–prey and SIR master tester

`run_ledh_nonlinear_master.py` evaluates both models through the shared canonical
LEDH executor and its recursive analytical total score. It writes actual log
likelihoods and every score coordinate, with standard errors across independent
particle designs. The program is diagnostic: its initial controls are warm
starts, and it does not issue tuning or admission artifacts.

## Models and data

| Model | State / observation dimensions | Parameter coordinates | Data-generating truth |
|---|---|---|---|
| `predator_prey` | 2 / 2 | r, carrying capacity, half saturation, s, u, v (physical units) | 0.6, 114, 25, 0.3, 0.5, 0.5 |
| `sir_d18` | 18 / 9 | log infection-rate scale, log removal-rate scale, log observation-noise scale | 0, 0, 0 |

Fresh synthetic data use the current canonical adapters, the declared initial
N(mean, I) law, and a transition before each observation y1,...,yT. Dataset JSON
records the observations and their hash. Historical fixtures with observation at
time zero or a different parameter chart are different targets. Both initial
clouds have parameter-independent identity local covariance; the executor then
carries each particle's updated UKF covariance and its derivatives.

The arms are `covariance_only`, `original` (repeated-axis residual and original
cap), `richer_marginal`, `richer_pairwise` (normal-quantile residual and identity
cap region), and `guarded_pairwise`. The last two differ only in the safety flag.
The first three are simple comparison arms. Results remain conditional on model,
dataset, parameter point, horizon, particle count and ancestry route. With T=1,
the final reset cannot affect the current likelihood: use T>=2 to exercise its
effect on subsequent likelihoods and scores.

## Commands

Run from the repository root with the `tftwogpu` Python interpreter. `plan` is
standard-library only; it prints the exact proposed jobs without using a GPU:

```bash
python docs/benchmarks/run_ledh_nonlinear_master.py plan
```

The default plan uses both models, T=20, N=1008, two datasets, eight particle
designs, three ancestry routes and all five arms. A real run requires an explicit
wall-time budget, including compilation; choose it under a recorded campaign
plan. For example, the following is a bounded mechanics check, not the full plan:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_ledh_nonlinear_master.py run \
  --models predator_prey sir_d18 --horizons 2 --particles 72 \
  --data-seeds 260201 --design-seeds 260301 --routes iid_dual_cap \
  --arms guarded_pairwise --budget-seconds 900 --worker-seconds 440 \
  --output /tmp/ledh-nonlinear-smoke-unique
```

GPU/XLA, FP32 with TF32 and verified memory growth are the defaults. Run GPU
commands with trusted/escalated device access. `--dtype float64` selects an
explicit reference arm. `--device cpu` hides GPUs before TensorFlow import;
`--no-jit` is an explicit debugging exception. The output directory must be new.
A timeout or invalid numerical result produces nonzero exit status and retained
partial results; it is not silently discarded. Workers execute sequentially so
the global wall-time limit includes all attempted work.

`--trace` saves reset and safety diagnostics for the first particle-design seed
in each cell, checking that tracing preserves both value and score coordinate
zero. `--controls-json` accepts model-keyed dictionaries of shared numerical
controls; `--theta-json` accepts model-keyed lists of parameter vectors. Neither
constitutes scope-specific offline tuning. Particle count must be divisible by
2d, and the repository exact-divisor transport chunk policy is enforced.

## References and interpretation

No exact nonlinear observed-data oracle is built into this program. An absent
reference is `null`/unavailable, never zero. A conditional complete-data score,
a UKF approximation, or a particle teacher must not be supplied as an exact
observed-data oracle. Independently computed references can be loaded with
`--reference-file`, a JSON list whose entries have this form:

```json
[{
  "target_id": "canonical_sir_d18_x0_then_transition_observe_v1",
  "observation_sha256": "COPY_FROM_DATASET_JSON",
  "horizon": 20,
  "theta": [0.0, 0.0, 0.0],
  "parameter_names": ["log_kappa_scale", "log_nu_scale", "log_observation_noise_scale"],
  "kind": "numerically_converged",
  "source": "path/to/independent-reference-result.json",
  "verification": "Describe target alignment, refinement and residual uncertainty",
  "log_likelihood": -123.0,
  "score": [1.0, 2.0, 3.0]
}]
```

The numbers above illustrate the schema, not SIR results. Allowed kinds are
`exact`, `numerically_converged`, and `approximate`. The runner verifies exact
identity fields, dimensions and finite numbers; it cannot certify the scientific
validity of the supplied verification. The reference producer must establish
that independently. A provided file missing a requested scope fails closed.
Float32 parameter coordinates must match the serialized evaluated coordinates.

`rows.json` retains each raw value and score, full controls, validity and optional
reference errors. `values-and-scores.csv` reports every coordinate alongside the
reference and replication standard error. `results.md` is a compact table;
`summary.json` retains conditional means, errors and paired changes against the
original arm. Failed rows and unfinished workers are counted. Source hashes,
Git state, commands, settings and memory policy are in manifests; complete logs
remain beside every worker.

Raw differences without a reference do not measure accuracy. Small smoke runs
establish mechanics only. A scientific comparison still needs scope-specific
calibration/validation, an untouched test, same-target reference uncertainty,
paired statistical analysis, conditional heuristic comparisons and a declared
budget. No result from this driver alone changes a default or establishes HMC
readiness. See `docs/plans/ledh-nonlinear-master-20261002.md`.

## Completed campaign

The first campaign finished on 2 October 2026; this closeout was written on
3 October. It completed the planned diagnostic screening, an independent
reference ladder, exact-input FP64 replay and localization of the first SIR
failure. Six GPU launches used 1289.335 of the 3600-second budget. Code, raw
results and the monograph update were archived in commit
`77be36b951de9b9a98225758554fe895cdf5a505`. The
[execution manifest](../plans/artifacts/ledh-nonlinear-execution-20261002/execution-manifest.json)
records every attempt, including runs that returned numerical vetoes.

At T=20 and N=1008, predator–prey produced finite likelihoods and all score
coordinates in 70/70 evaluations; 69 passed the trace checks. SIR d=18 failed
all ten FP32 evaluations. Exact-input FP64 replay produced four finite results,
three of which passed the trace checks, but the likelihood and score errors
remained large. Representative likelihoods show the scale of the discrepancy:

| Model / data seed | Covariance-only log likelihood | Guarded pairwise log likelihood | Approximate reference log likelihood | Reference MCSE |
|---|---:|---:|---:|---:|
| Predator–prey / 260401 | -97.438702 | -97.350349 | -97.342974 | 0.006575 |
| Predator–prey / 260402 | -99.297594 | -99.287128 | -99.311525 | 0.003656 |
| SIR d=18 / 260401, FP64 | -982.119424 | -1040.878062 | -678.077466 | 0.014864 |

Predator–prey entries are mean log likelihoods across two and four IID designs,
respectively. The SIR entries are individual valid evaluations using design
260502. References use N=524288 and four independent replications, aggregated
as log mean likelihood; MCSE is a delete-one-replication jackknife estimate.
The reference ladder also includes N=8192, 32768 and 131072. Its analytical
Fisher score estimates the observed-data score by averaging complete-data scores
over particle paths. It is not the derivative of the finite stochastic particle
likelihood estimate, and its MCSE does not account for finite-particle bias.
Every score coordinate, uncertainty and flagged row is available in the
[full numerical results](ledh-nonlinear-execution-results-20261002.md) and
[likelihood-and-score CSV](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/actual-values-and-scores.csv).

The SIR FP32 failure first appears at observation 4 in the shared Contract E
reset. The upstream UKF and flow checks pass. Covariance-only reset states remain
finite while their analytical tangents fail; the guarded correction receives
invalid input. This localizes the next diagnostic without identifying the exact
offending factorization or derivative operation. Double precision alone does
not resolve the observed accuracy discrepancy.

The original Zhao–Cui filter was not a comparator in this campaign. The local
[author-source snapshot](../../third_party/audit/zhao_cui_tensor_ssm_p10/README.md)
is pinned to upstream commit `80034dccb99eb1d86284a1839b4a12067d13b9da` with
documented Octave compatibility patches. In
[full_sol.m, method smooth](../../third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m),
`w = logpdf_t - logpdf_e`, and the returned `lml` is the mean of finite `w`
values; the log-mean-exp calculation is commented out. That statistic is not
the observed-data log likelihood used here. A useful comparison must align
saved observations, model timing and fixed parameter values, then verify the
likelihood normalizer and a same-target score estimate. No agreement or
disagreement with the original algorithm has been established on these data.

The campaign supports the numerical vetoes above and leaves the valid
predator–prey configurations available for further evaluation. All differences
between correction arms are descriptive: no ranking is statistically supported.
The controls were untuned, and there was no untouched claim run. Next work is
the SIR reset diagnostic and a non-harm evaluation of any numerical protection,
followed by scope-specific calibration, held-out replication and an independent
nonlinear reference cross-check. These results do not establish a new default,
HMC readiness or an exact nonlinear oracle. The full result note retains the
decision and inference-status tables and all 25 conditional simple-arm comparisons.

Engineering validation is complete: 25 focused CPU tests passed, including the
three independent-reference tests, and the scientific commit passed three
oracle-contract hook tests. GPUs were intentionally hidden for those CPU checks.
Both LaTeX chapter copies agree; the monograph compiled to 606 pages without
unresolved references or citations, and the changed pages were visually
inspected. [Build verification](../plans/artifacts/ledh-nonlinear-execution-20261002/monograph-build/verification.json)
and the [reset memo](../reset-memos/ledh-nonlinear-execution-20261002.md) preserve
the completed state and the next scientific action.
