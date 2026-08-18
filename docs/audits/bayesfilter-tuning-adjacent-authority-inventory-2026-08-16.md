# Adjacent Tuning Authorities

Generated 2026-08-16. These modules do not all have tuning names, but they are imported by active tuning routes or provide execution/evidence/geometry authority. They are audited as adjacent dependencies, not included in the 1,417 core-definition count.

| Path | Definitions | Active dependency reason |
|---|---:|---|
| `bayesfilter/inference/hmc.py` | 204 | TFP full-chain execution, config, reusable runner, mass artifact and adapter signatures |
| `bayesfilter/inference/hmc_verification.py` | 47 | acceptance/health evidence and veto classification |
| `bayesfilter/inference/hmc_convergence.py` | 12 | rank-normalized R-hat/ESS diagnostics |
| `bayesfilter/inference/hmc_warmup.py` | 65 | warmup schedules and nonclaims |
| `bayesfilter/inference/mass_matrix.py` | 10 | covariance/whitening/mass construction |
| `bayesfilter/inference/fixed_transport_hmc.py` | 10 | fixed transformed-coordinate adapter and transport identity |
| `bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py` | 31 | fixed-transport numerical mechanics |
| `bayesfilter/inference/fixed_transport_hmc_candidate_discovery_tf.py` | 66 | transport candidate discovery and evidence |
| `bayesfilter/inference/posterior_adapter.py` | 37 | value/score capability and target scope contract |

Required review: inspect these modules for changes to target identity, trace shape, XLA, memory policy, or evidence semantics before extracting tuning mechanics.

## Direct Dependency Closure Addendum

The nine-module table above is the named execution/evidence subset and totals
482 definitions. A fresh import walk from the core tuning modules found these
additional direct dependencies that must also be reviewed before extraction:

| Path | Definitions | Import role |
|---|---:|---|
| `bayesfilter/hmc_budget_contract.py` | 16 | Work-budget contracts imported by `hmc_kernel_tuning`. |
| `bayesfilter/hmc_route_contract.py` | 7 | Route identity contracts imported by tuning and artifact modules. |
| `bayesfilter/inference/batched_value_score.py` | 46 | Value/score authority used by fixed-transport and budget-ladder routes. |
| `bayesfilter/inference/hmc_coordinates.py` | 31 | Coordinate and metric contracts used by tuning/artifact code. |
| `bayesfilter/inference/hmc_diagnostics.py` | 10 | HMC diagnostic summaries used by tuning policy. |
| `bayesfilter/inference/hmc_posterior_diagnostics.py` | 23 | Posterior diagnostic helpers used by staged fixed-kernel code. |
| `bayesfilter/inference/neutra_shared_procedure.py` | 8 | Shared procedure used by NeuTra broad-grid routes. |
| `bayesfilter/runtime/__init__.py` | 2 | `stable_config_hash` and runtime artifact identity. |
| `bayesfilter/inference/native_tfp_hmc.py` | 69 | Native TFP execution authority named by active routes. |
| `bayesfilter/inference/neutra_hmc.py` | 59 | NeuTra sequential HMC execution authority. |
| **Total** | **271** | Additional direct dependencies. |

The full tuning dependency closure is therefore not represented by a single
number: 1,417 core definitions, 482 named adjacent execution/evidence
definitions, and 271 additional direct dependencies. These sets may overlap by
imports but are intentionally reported as separate inventories, not summed as a
deduplicated closure.
