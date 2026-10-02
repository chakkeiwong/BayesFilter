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
