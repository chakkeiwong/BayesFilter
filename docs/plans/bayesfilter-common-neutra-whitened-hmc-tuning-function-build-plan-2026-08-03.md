# Common NeuTra-Whitened HMC Tuning Function Build Plan

Date: 2026-08-03
Status: `ACTIVE_REFACTOR_PLAN`

## Question

Can BayesFilter collapse its existing NeuTra/HMC machinery into one common
Python tuning function for successfully trained NeuTra-whitened targets, using a
single repaired state-continuing epsilon-repair procedure with only small
model-dependent hooks?

## Motivation

Recent shared-procedure infrastructure work proved that:

- the PP repaired route can be extracted into shared inference code and still
  reproduce its historical viable-set artifact;
- KSC gaussian-sum SV and LGSSM both fail under the weaker generic operational
  broad-grid route in a way that indicates a **procedure mismatch**, not an
  infrastructure failure.

The user’s clarified direction is correct: imperfect whitening changes the
**result** of tuning, not the need for one common **procedure**. If whitening
were exact, NeuTra HMC in latent coordinates would reduce to the IID Gaussian
case; tuning exists precisely to absorb residual mismatch. Therefore the repo
should have one common NeuTra-whitened HMC tuning function in Python.

## Target procedure to implement

Promote the repaired state-continuing epsilon-repair route to the common/master
procedure. For every active NeuTra-whitened target lane, the common function
should:

1. verify target signature and frozen transport SHA-256;
2. bind the fixed-transport batch-native adapter;
3. run the common primary `L` grid;
4. for each primary `L`:
   - dual average epsilon toward target acceptance 0.70,
   - verify one frozen post-adaptation epsilon,
   - if acceptance is outside the calibration region, run a bounded epsilon
     repair loop,
   - continue from the calibrated state,
   - run fresh final screens;
5. run one-hop same-epsilon coverage probes from the calibrated parent state;
6. preserve the complete unranked viable set;
7. emit a normalized artifact that records the exact common procedure id.

No sequential HMC will be launched in the validation sweep for this build plan.
The question is only whether the common tuning procedure can produce a valid
complete viable-set artifact.

## Existing pieces to reuse

- target/frozen-transport binding and manifests:
  `bayesfilter/inference/neutra_end_to_end.py`
- reusable HMC runners and dual averaging:
  `bayesfilter/inference/hmc.py`, `bayesfilter/inference/hmc_tuning.py`
- barrier and viable-set semantics:
  `bayesfilter/inference/hmc_operational_broad_grid.py`
- repaired state-continuing calibration/repair mechanics:
  `bayesfilter/inference/neutra_state_continuing_broad_grid.py`
- shared top-level procedure scaffold:
  `bayesfilter/inference/neutra_shared_procedure.py`
- canonical sequential controller (not used in the sweep, but preserved for
  downstream integration): `bayesfilter/inference/neutra_hmc.py`

## Small model-dependent hook surface

The common procedure should vary only by:

- required status telemetry keys
- optional warm-start `epsilon_by_L` hints
- target/plan/procedure metadata

Requested active lanes for the validation sweep:

- `LGSSM-EXACT`
- `PP-UKF`
- `PP-SGQF`
- `KSC-UKF-GAUSSIAN-SUM-T20`
- `SVX-ZC`
- `SIR-SGQF`

Blocked / excluded lanes are recorded, not forced:

- historical `KSC-UKF` blocked
- `SIR-UKF` owner-excluded
- blocked `SVX-SGQF`, `PP-ZC`, `SIR-ZC`, `STR-ZC` out of scope

## Code changes

### 1. Promote repaired route to common default
- Refactor `bayesfilter/inference/neutra_shared_procedure.py` so the repaired
  state-continuing route is the common/default path.
- Keep the old operational route only as `legacy/reference` mode.

### 2. Add target hook metadata
- Extend `CellSpec` in `bayesfilter/testing/neutra_model_registry_tf.py` with
  optional common-tuning metadata:
  - status-key contract
  - optional warm-start `epsilon_by_L`
  - procedure id / enabled flag if needed
- Teach the KSC gaussian-sum shared driver path to provide the same metadata.

### 3. Update the shared driver
- Refactor `docs/benchmarks/run_neutra_shared_procedure_20260803.py` so the
  common repaired route is the default and shared target metadata is resolved
  from one place.

### 4. Keep the generic route but demote it
- Preserve `operational_broad_grid_v1` for compatibility, tests, and historical
  comparison.
- Do not route the new sweep through it.

### 5. Strengthen tests
- `tests/test_neutra_shared_procedure.py`
- `tests/test_neutra_all_models_end_to_end_contract.py`
- `tests/test_hmc_operational_broad_grid.py`
- `tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py`
- likely `tests/test_neutra_hmc_route_policy.py`

## Success criteria

1. The repaired route becomes the common default in the shared driver and
   shared procedure module.
2. Shared target metadata determines status-key contract and warm-start hints.
3. Focused tests pass before the sweep.
4. Each active lane can be launched through the common repaired tuning function.
5. The terminal sweep result can answer, for each lane: did the common tuning
   procedure produce a valid complete viable-set artifact?

## Nonclaims

- no sequential/posterior claim
- no convergence claim
- no ranking between targets or filter families
- no default-readiness claim
- no evidence that any blocked/owner-excluded lane is fixed
