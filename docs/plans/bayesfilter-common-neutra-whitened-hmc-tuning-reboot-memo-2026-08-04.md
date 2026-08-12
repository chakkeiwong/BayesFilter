# Common NeuTra tuning reboot memo

Date: 2026-08-04

## Recovery point

Work stopped **after** promoting the repaired state-continuing tuning route to
be the common/default NeuTra-whitened HMC tuning procedure in shared code, but
**before** launching the full cross-target sweep.

The build/sweep plans to resume from are:

- `docs/plans/bayesfilter-common-neutra-whitened-hmc-tuning-function-build-plan-2026-08-03.md`
- `docs/plans/bayesfilter-common-neutra-whitened-hmc-tuning-sweep-plan-2026-08-03.md`

## What is already implemented

### New / modified shared infrastructure

- `bayesfilter/inference/neutra_shared_procedure.py`
  - `DEFAULT_COMMON_VARIANT = STATE_CONTINUING_EPSILON_REPAIR_V1`
  - repaired route is now the default common path
  - legacy operational route remains explicit/reference only
- `bayesfilter/inference/neutra_state_continuing_broad_grid.py`
  - extracted repaired PP-style state-continuing epsilon-repair mechanics
- `bayesfilter/inference/neutra_broad_grid.py`
  - shared fixed-screen health helpers and variant metadata passthrough
- `bayesfilter/inference/neutra_end_to_end.py`
  - wrappers delegate through shared procedure / shared handoff parser
- `docs/benchmarks/run_neutra_shared_procedure_20260803.py`
  - default variant is now the common repaired route
  - target-specific status-key and warm-start hooks resolve automatically from shared metadata
- `bayesfilter/testing/neutra_model_registry_tf.py`
  - `CellSpec` now includes:
    - `common_tuning_status_keys`
    - `common_tuning_initial_epsilon_by_l`
  - metadata populated for:
    - `LGSSM-EXACT`
    - `PP-UKF`
    - `PP-SGQF`
    - `SIR-SGQF`
    - `SVX-ZC`
- `docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_end_to_end_20260802.py`
  - KSC special spec now carries `common_tuning_status_keys`

### Tests already updated

- `tests/test_neutra_shared_procedure.py`
- `tests/test_neutra_all_models_end_to_end_contract.py`

## Verified test status before reboot

These were run and passed **after** the default-common-route refactor:

- `python -m pytest tests/test_neutra_shared_procedure.py tests/test_neutra_all_models_end_to_end_contract.py -q`
  - **59 passed**
- `python -m pytest tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py tests/test_hmc_operational_broad_grid.py -q`
  - **27 passed**
- `python -m pytest tests/test_neutra_shared_procedure.py -q`
  - **13 passed** after the SVX-ZC metadata correction

One pre-existing repo debt remains unchanged:

- `python -m pytest tests/test_neutra_hmc_route_policy.py -q`
  - still **fails** due old unledgered-route debt across the repo
  - not introduced by this common-tuning refactor

## Important audit finding before reboot

I ran a CPU telemetry probe and found that **LGSSM and SVX-ZC both expose the
strict UKF-style telemetry keys through `BatchNativeBoundAdapter.target_status_telemetry`**, even though earlier I thought they might need only the basic two-key contract.

Observed telemetry keys for all four probed lanes:

- `PP-SGQF`
- `SIR-SGQF`
- `SVX-ZC`
- `LGSSM-EXACT`

all included:

- `status_code`
- `valid_pre_regularized_score`
- `floor_count_value`
- `min_innovation_eigenvalue`
- `innovation_condition_estimate`
- plus `*_available` flags in some cases

**But note:** the current metadata change only updated `SVX-ZC` back to the
strict five-key contract. `LGSSM-EXACT` is still set to the basic two-key
contract in `neutra_model_registry_tf.py` because I had not yet decided whether
to preserve that historical distinction or align it with the observed adapter
surface. This is the first thing to re-check after reboot before launching the sweep.

## Active / blocked / excluded lanes snapshot

### Active executable registry lanes

- `LGSSM-EXACT`
- `PP-UKF`
- `PP-SGQF`
- `SIR-SGQF`
- `STR-UKF`
- `SVX-ZC`

### Special active lane outside registry

- `KSC-UKF-GAUSSIAN-SUM-T20`

### Blocked

- `SVX-SGQF`
- historical `KSC-UKF`
- `PP-ZC`
- `STR-ZC`
- `SIR-ZC`

### Owner-excluded

- `SIR-UKF`

## Target lanes in scope for the sweep

Run only these:

- `PP-UKF`
- `PP-SGQF`
- `SIR-SGQF`
- `SVX-ZC`
- `KSC-UKF-GAUSSIAN-SUM-T20`
- `LGSSM-EXACT`

Do **not** run blocked/excluded lanes.

## Frozen transports and hashes already resolved

### PP-UKF
- path:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`

### PP-SGQF
- path:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-SGQF/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `9f1080d42eb5f2f34a9dc3e8278f7110c3424ca829685a4a10ea2c8bfa783dd9`

### SIR-SGQF
- path:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/SIR-SGQF/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `26246c564faaa429a1772b5df464ffb8a7b218ff394ba4209d9a3f235942f673`

### SVX-ZC
- path:
  `docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/neutra-hmc-attempt01/SVX-ZC/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `c816de3d7101444bdeead2e9d43b0ca49de8d426ebd650c0efcf73068d9decff`

### KSC-UKF-GAUSSIAN-SUM-T20
- path:
  `docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/final-training-attempt01/KSC-UKF-GAUSSIAN-SUM-T20/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `dbbaba3735404d9dd98b233e9419ab4fd3d82c8ac9a5922c9e47712d42e8bddb`

### LGSSM-EXACT
- path:
  `docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json`
- sha256:
  `b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`

## Immediate next step after reboot

1. Re-open this memo and the two plan files above.
2. Check whether `LGSSM-EXACT.common_tuning_status_keys` should remain basic or be promoted to the strict five-key contract, given the CPU telemetry probe. This is the only unresolved metadata decision before the sweep.
3. Re-run the focused audit tests:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_neutra_shared_procedure.py \
  tests/test_neutra_all_models_end_to_end_contract.py \
  tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py \
  tests/test_hmc_operational_broad_grid.py -q
```

4. If those pass, launch the bounded tuning-only sweep in this order:
   - `PP-UKF`
   - `PP-SGQF`
   - `SIR-SGQF`
   - `SVX-ZC`
   - `KSC-UKF-GAUSSIAN-SUM-T20`
   - `LGSSM-EXACT`

5. Use fresh roots under:
   `docs/plans/artifacts/bayesfilter-neutra-common-tuning-sweep-20260803/`

## Sweep commands template

Use the shared driver:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_shared_procedure_20260803.py \
  --cell <CELL_ID> \
  --output-root <FRESH_ROOT> \
  --frozen-transport <PATH> \
  --frozen-transport-sha256 <SHA256> \
  --root-seed <seed0> <seed1>
```

Notes:
- omit `--variant` to use the repaired common default;
- omit `--required-status-keys` to use shared metadata auto-resolution;
- only pass `--initial-epsilon-by-l` when overriding or testing a warm-start hypothesis explicitly.

## Working-tree caution

There are unrelated user changes and many untracked docs/artifacts in the repo.
Do **not** revert or mass-stage by glob. Stage only the common-tuning-function
files and the new sweep artifacts.
