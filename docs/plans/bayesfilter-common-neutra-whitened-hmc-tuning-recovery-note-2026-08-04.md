# Common NeuTra whitened HMC tuning recovery note

Date: 2026-08-04
Status: `REBOOT_RECOVERY_NOTE`

## Purpose

This note is the short restart handoff for the next agent after reboot. It
summarizes exactly where to resume the common NeuTra-whitened HMC tuning work
without rediscovering scope, transports, or test status.

## Primary restart files

Read these first, in order:

1. `docs/plans/bayesfilter-common-neutra-whitened-hmc-tuning-reboot-memo-2026-08-04.md`
2. `docs/plans/bayesfilter-common-neutra-whitened-hmc-tuning-function-build-plan-2026-08-03.md`
3. `docs/plans/bayesfilter-common-neutra-whitened-hmc-tuning-sweep-plan-2026-08-03.md`

## Current implementation state

The repo is already partly refactored toward the common repaired tuning route:

- `bayesfilter/inference/neutra_shared_procedure.py`
  - repaired state-continuing route promoted as the common/default variant
- `bayesfilter/inference/neutra_state_continuing_broad_grid.py`
  - extracted repaired PP-style tuning mechanics
- `bayesfilter/inference/neutra_broad_grid.py`
  - shared fixed-screen health helpers / metadata passthrough
- `bayesfilter/inference/neutra_end_to_end.py`
  - wrappers delegate through the shared procedure / shared handoff parser
- `bayesfilter/testing/neutra_model_registry_tf.py`
  - `CellSpec` now carries common tuning metadata
- `docs/benchmarks/run_neutra_shared_procedure_20260803.py`
  - shared driver updated toward common-route defaults
- `docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_end_to_end_20260802.py`
  - KSC special spec now exposes common tuning status-key metadata

## Focused tests already known to pass

These passed after the refactor work before reboot:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_neutra_shared_procedure.py \
  tests/test_neutra_all_models_end_to_end_contract.py -q
# 59 passed

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py \
  tests/test_hmc_operational_broad_grid.py -q
# 27 passed

CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_neutra_shared_procedure.py -q
# 13 passed after the SVX-ZC metadata correction
```

## Known pre-existing failing test

This is still failing from older repo debt and was **not** introduced by the
common-tuning refactor:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_neutra_hmc_route_policy.py -q
```

Do not confuse this route-ledger failure with the current refactor.

## One unresolved metadata decision before the sweep

A CPU telemetry probe showed that all of these expose the strict UKF-style
telemetry surface through `BatchNativeBoundAdapter.target_status_telemetry`:

- `PP-SGQF`
- `SIR-SGQF`
- `SVX-ZC`
- `LGSSM-EXACT`

The strict keys observed include:

- `status_code`
- `valid_pre_regularized_score`
- `floor_count_value`
- `min_innovation_eigenvalue`
- `innovation_condition_estimate`

`SVX-ZC` was already corrected back to the strict five-key metadata.

**Open question:** whether `LGSSM-EXACT` should remain on the historical basic
2-key contract in metadata, or be promoted to the strict 5-key contract since
its adapter surface now exposes those fields.

Resolve that one issue first before launching the sweep.

## Active lanes to run in the sweep

Run only these common-tuning lanes:

- `PP-UKF`
- `PP-SGQF`
- `SIR-SGQF`
- `SVX-ZC`
- `KSC-UKF-GAUSSIAN-SUM-T20`
- `LGSSM-EXACT`

Do **not** run:

- historical `KSC-UKF` (blocked)
- `SIR-UKF` (owner-excluded)
- blocked `SVX-SGQF`, `PP-ZC`, `SIR-ZC`, `STR-ZC`

## Frozen transports already resolved

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

## Exact next commands after reboot

### 1. Re-run the focused audit

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest \
  tests/test_neutra_shared_procedure.py \
  tests/test_neutra_all_models_end_to_end_contract.py \
  tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py \
  tests/test_hmc_operational_broad_grid.py -q
```

### 2. If the audit passes, launch the bounded tuning-only sweep

Use fresh roots under:

`docs/plans/artifacts/bayesfilter-neutra-common-tuning-sweep-20260803/`

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
- only pass `--initial-epsilon-by-l` when deliberately overriding the shared
  hook metadata.

## Sweep order

Run in this order:

1. `PP-UKF`
2. `PP-SGQF`
3. `SIR-SGQF`
4. `SVX-ZC`
5. `KSC-UKF-GAUSSIAN-SUM-T20`
6. `LGSSM-EXACT`

## Working-tree caution

There are many unrelated user changes and untracked docs/artifacts in the repo.
On resume:

- do **not** mass-stage by glob,
- do **not** clean untracked files broadly,
- do **not** revert unrelated work,
- stage only the common-tuning-function files and the new sweep artifacts.
