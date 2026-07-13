# Phase 3 Result: LGSSM Fixture Driver

Date: 2026-07-09

Status: `PASSED`

## Scope

Phase 3 added and ran the deterministic fixture stage of the serious LGSSM HMC
tuning driver. It generated a `T=120` lower-triangular LGSSM fixture from the
Phase 2 config. It did not compile the HMC target, run HMC, train NeuTra, tune
kernel parameters, sample posterior chains, or make posterior recovery claims.

## Implementation Artifacts

- Driver:
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
- Focused test:
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Config:
  `docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json`
- Fixture:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/fixture_T120_seed20260709_301.json`

## Fixture Summary

| Field | Value |
| --- | --- |
| Fixture schema | `bayesfilter.deterministic_lgssm_hmc_tuning_fixture.v1` |
| Fixture artifact hash | `sha256:346d2932ac329a477b35530010cf2dff6d4cf2022f003216b1a46a19bbca54ac` |
| Config hash | `sha256:683e45cef9a46e14a3ee2de3e51d5fc19a0512feb43e376e30c2da19e1a2ccb0` |
| Horizon | `120` |
| State shape | `[120, 4]` |
| Observation shape | `[120, 4]` |
| Transition spectral radius | `0.62` |
| Stationary covariance min eigenvalue | `0.03457789017637346` |
| Lyapunov max residual | `1.3877787807814457e-17` |
| Max abs observation | `0.8536776776031644` |

## Checks Run

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `4 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage fixture
```

Result: fixture JSON written. TensorFlow emitted CUDA plugin/cuInit warnings
despite `CUDA_VISIBLE_DEVICES=-1`; this is recorded as CPU-hidden environment
noise, not GPU evidence.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/fixture_T120_seed20260709_301.json
```

Result: JSON parse check passed.

```text
git diff --check -- <Phase 3 files>
```

Result: passed.

## Evidence Contract Assessment

| Field | Assessment |
| --- | --- |
| Question | Can the LGSSM fixture be generated deterministically from prior means? |
| Primary criterion | Met: fixture hash is stable in focused tests and output records `T=120`, truth, states, observations, stationarity diagnostics, and hashes. |
| Veto diagnostics | No nonstationarity, invalid covariance, nonfinite data, or hash instability observed in Phase 3 checks. |
| Explanatory diagnostics | Eigenvalues, stationary covariance eigenvalues, Lyapunov residual, moments, max absolute values. |
| Not concluded | No target score correctness, XLA compile readiness, HMC tuning success, convergence, posterior recovery, or scientific adequacy claim. |

## Decision Table

| Field | Decision |
| --- | --- |
| Phase decision | `PASS_TO_PHASE4` |
| Primary criterion status | Deterministic fixture generation passed |
| Veto diagnostic status | No Phase 3 veto triggered |
| Main uncertainty | The XLA value/score target has not yet been checked against this new `T=120` fixture |
| Next justified action | Execute Phase 4 XLA value/score gate |
| What is not concluded | No HMC run, no posterior recovery, no convergence, no runtime tuning feasibility, no scientific/default/product claim |

## Plain-Language Gate

Claimed target: generate a deterministic `T=120` lower-triangular LGSSM
fixture from the prior-mean truth config.

Computed quantity: one JSON fixture with raw truth, constrained truth, states,
observations, stationarity diagnostics, config hash, and artifact hash.

Verdict: `correct` for deterministic fixture generation under the named config;
`not checked` for XLA target score, HMC tuning, posterior recovery, NeuTra, and
scientific validity.
