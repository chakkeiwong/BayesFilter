# P0 Target And Route Ledger

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Machine-readable authority:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/target_registry.json`

## Registry Decision

No mandatory cell currently has a complete claim-bearing posterior contract.
P0 therefore issued zero posterior target signatures. Each row has a
deterministic `scope_identity` for inventory continuity, but that identity is
explicitly ineligible for HMC, NeuTra training, frozen-transport loading,
posterior comparison, or scientific claims.

A future target signature requires all of: frozen observations and data hash,
parameter prior, constrained parameter support, unconstraining chart and full
log-absolute-Jacobian, exact filter settings and dependency closure, batched
value/score adapter, and independent posterior total-value/total-score
recomposition with wrong-substitution negative tests.

## Cell Matrix

| Cell | Current route evidence | Source/route class | P0 state | Load-bearing blockers |
| --- | --- | --- | --- | --- |
| `SVX-SGQF` | Exact transformed direct-likelihood SGQF value/score in `bayesfilter/highdim/sv_mixture_cut4.py:663` and `:771` | BayesFilter approximation | `TARGET_BLOCKED` | Prior, serious data hash, independent posterior recomposition, serious SGQF settings, batched posterior adapter |
| `SVX-ZC` | Factorized scalar fixed-design TT wrapper at `bayesfilter/highdim/sv_mixture_cut4.py:1267` and `:1329` | `extension_or_invention` for the current wrapper | `TARGET_BLOCKED` | Prior/data/recomposition plus source-route mismatch and no admitted production fixed route |
| `KSC-UKF` | Component-enumerated principal-square-root UKF at `bayesfilter/highdim/sv_mixture_cut4.py:2062` and score at `:2151` | BayesFilter approximation; distinct KSC target | `TARGET_BLOCKED` | Prior, serious data hash, posterior recomposition, posterior-region UKF admission, batched posterior adapter |
| `PP-SGQF` | Test closure at `tests/highdim/test_p47_predator_prey_filtering.py:258` | BayesFilter approximation | `TARGET_BLOCKED` | Prior/data, bounded HMC chart/Jacobian, independent posterior recomposition, serious SGQF settings, registered adapter |
| `PP-UKF` | Structural closure and UKF test route at `tests/highdim/test_p47_predator_prey_filtering.py:214` and `:275` | BayesFilter approximation | `TARGET_BLOCKED` | Prior/data, bounded chart/Jacobian, posterior recomposition, posterior-region UKF admission, registered adapter |
| `PP-ZC` | Generic all-axes retained-grid diagnostic at `tests/highdim/test_p47_predator_prey_filtering.py:201` | `extension_or_invention`; production-ineligible generic route | `TARGET_BLOCKED` | Prior/data/chart/recomposition, source-route mismatch, no admitted source-route same-target value/score posterior |
| `STR-UKF` | NumPy worked fixture and identity test only | Planned BayesFilter approximation | `TARGET_BLOCKED` | Graph-native model/data, inferred parameter subset, prior/chart, posterior adapter/recomposition, naive-noise negative control |
| `STR-ZC` | No implementation | `extension_or_invention` by definition | `TARGET_BLOCKED` | All structural blockers plus missing extension design/value/score route |
| `SIR-SGQF` | No full observed-data parameter-posterior route | Planned BayesFilter approximation | `TARGET_BLOCKED` | Prior/data/recomposition and missing full observed-data SGQF value/score posterior |
| `SIR-UKF` | Lower-rung/scout evidence only | BayesFilter scout | `TARGET_BLOCKED` | Prior/data/recomposition and scout is not a full observed-data posterior |
| `SIR-ZC` | Parameterized local/component model and source-route substrate | Incomplete `fixed_hmc_adaptation` substrate for an extension target | `TARGET_BLOCKED` | Paper SIR fixes parameters, no full observed-data parameter posterior, no retained-object value/score posterior, prior/data/recomposition missing |

## Exact Target Separations

- `SVX-*` uses exact log-chi-square transformed observation noise and zero
  transform offset. `KSC-UKF` uses a seven-component Gaussian mixture and
  `log(y^2 + 1e-8)`. Evidence cannot cross those signatures.
- Each filter likelihood defines a separate approximate posterior even when the
  model, data, and prior are later shared.
- SIR complete-data component scores do not equal an observed-data filter
  posterior. The paper's Section 6.3 SIR example has no inferred parameters;
  BayesFilter's three-scale SIR target is an extension.
- The Chapter 18b route must preserve
  `k_t - phi*k_(t-1) - gamma*m_t^2 = 0`; a full-state UKF with artificial
  `k_t` noise is only a required negative control.

## P1 Eligibility

P1 may proceed because its generic target/signature/recomposition/state-machine
harness can be built and tested with synthetic canaries while all model cells
remain blocked. P1 may not issue a target signature or run a model-cell HMC or
training job. Each model phase must return to `P0_TARGET_FREEZE` for its cell
after closing the listed target fields.

## Review Instruction

A bounded reviewer checking Zhao-Cui classifications may inspect only these
explicit cited paths in addition to this ledger:

- `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`
  at Sections 2.1, 2.3, 3, 5, and 6.2-6.4;
- `third_party/audit/tensor-ssm-paper-demo/models/full_sol.m:21`;
- `third_party/audit/tensor-ssm-paper-demo/models/pre_sol.m:16`;
- `bayesfilter/highdim/sv_mixture_cut4.py:1267`;
- `tests/highdim/test_p47_predator_prey_filtering.py:201`.

No other repo-wide review is requested.

