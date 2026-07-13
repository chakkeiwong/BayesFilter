# Phase 5 Result: Geometry And Mass Driver

Date: 2026-07-09

Status: `PASSED`

## Scope

Phase 5 added and ran deterministic geometry/mass initialization for the
multidimensional LGSSM HMC tuning program. It used BayesFilter's quadratic
geometry initializer and mass-matrix helper. It did not run HMC, tune a kernel,
generate posterior samples, train NeuTra, or make posterior recovery claims.

## Skeptical Pre-Run Audit

| Risk | Audit Finding |
| --- | --- |
| Wrong baseline | Inputs were the Phase 3 fixture and Phase 4 XLA value/score gate. |
| Agent tuning | Geometry and mass settings came from the fixed JSON config, not manual diagnostic edits. |
| Proxy promotion | Geometry/mass pass is treated only as initializer readiness for later tuning. |
| Missing stop condition | Geometry rejection, non-SPD mass, condition-cap failure, or reconstruction failure would veto. |
| Environment mismatch | Command used `CUDA_VISIBLE_DEVICES=-1`; this was CPU-hidden initialization, not GPU evidence. |
| Artifact mismatch | Geometry and mass artifacts include arrays, summaries, hashes, vetoes, and nonclaims. |

Verdict: `PASS_TO_EXECUTE_PHASE5`.

## Implementation Artifacts

- Driver:
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
- Focused test:
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Geometry artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/geometry.json`
- Mass artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/mass.json`

## Artifact Summary

| Field | Geometry |
| --- | --- |
| Schema | `bayesfilter.deterministic_lgssm_hmc_tuning_geometry.v1` |
| Artifact hash | `sha256:e2b9531e86f85a662c4da26595e0ab082dd8a1a29d2dbb83b31b076bbf7683ac` |
| Passed | `true` |
| Vetoes | `[]` |
| Low-rank status | `usable` |
| Accepted | `true` |
| Holdout RMSE | `0.12444574204988731` |
| Holdout threshold | `0.6641473363290465` |
| Center score norm | `15.247046803491266` |
| Precision condition number | `1.9640743886787067` |
| Covariance condition number | `1.9640743886787073` |
| Center refinement accepted | `false` |

| Field | Mass |
| --- | --- |
| Schema | `bayesfilter.deterministic_lgssm_hmc_tuning_mass.v1` |
| Artifact hash | `sha256:92536fbd13e1ba89c53bfcc874355194b8d2d097ea498d22b5ccd7c318490d8e` |
| Passed | `true` |
| Vetoes | `[]` |
| Precision condition number | `1.9640743885607335` |
| Mass covariance condition number | `1.9640743885607346` |
| Precision/covariance identity max error | `6.661338147750939e-16` |
| Factor/covariance max error | `1.3877787807814457e-17` |
| Clipped eigenvalue count | `0` |
| Effective eigenvalue floor | `0.00016049785411397907` |

## Checks Run

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `7 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage geometry_mass
```

Result: geometry and mass JSON artifacts written. TensorFlow emitted CUDA
plugin/cuInit warnings despite `CUDA_VISIBLE_DEVICES=-1`; this is recorded as
CPU-hidden environment noise, not GPU evidence.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/geometry.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/mass.json
```

Result: JSON parse checks passed.

```text
git diff --check -- <Phase 4/5 modified files and artifacts>
```

Result: passed.

## Evidence Contract Assessment

| Field | Assessment |
| --- | --- |
| Question | Can existing BayesFilter tools deterministically produce an HMC mass initializer? |
| Primary criterion | Met: geometry accepted, mass covariance is SPD, condition caps are respected, and hashes are recorded. |
| Veto diagnostics | No geometry rejection, nonfinite samples, non-SPD mass, condition cap failure, or reconstruction failure. |
| Explanatory diagnostics | Center score norm, holdout RMSE, eigen summaries, regularization report, factor reconstruction. |
| Not concluded | No certified MAP covariance, no HMC tuning success, no convergence, no posterior recovery, no sampler superiority, and no production/default readiness. |

## Decision Table

| Field | Decision |
| --- | --- |
| Phase decision | `PASS_TO_PHASE6_WITH_APPROVAL_REQUIRED` |
| Primary criterion status | Deterministic geometry/mass initialization passed |
| Veto diagnostic status | No Phase 5 veto triggered |
| Main uncertainty | Kernel tuning has not been run and may fail under XLA/runtime constraints |
| Next justified action | Review Phase 5 result and Phase 6 subplan, then request explicit runtime approval before any serious HMC tuning |
| What is not concluded | No HMC run, no burn-in, no retained sampling, no convergence, no posterior recovery |

## Plain-Language Gate

Claimed target: produce a deterministic geometry/mass initializer using the
BayesFilter quadratic initializer and mass conversion tools.

Computed quantity: one accepted low-rank SPD quadratic geometry artifact and
one dense SPD mass covariance/factor artifact.

Verdict: `correct` for Phase 5 deterministic initializer generation under the
named CPU-hidden command; `not checked` for HMC tuning, posterior recovery,
NeuTra, GPU execution, and scientific validity.

## Next Boundary

Phase 6 is serious HMC kernel tuning. The runbook says Phase 6 requires
explicit user runtime approval before launch. Do not run Phase 6 until that
approval is granted.
