# Direct-Factor SR-UKF Model Coverage Execution Result

Date: 2026-08-17  
Plan: `docs/plans/bayesfilter_direct_factor_srukf_all_model_coverage_and_latex_update_plan_2026_08_17.md`  
Artifact root: `docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/`  
Status: `EXECUTED_WITH_EXPLICIT_CLASSIFICATION_BOUNDARIES`

## Review disposition

The plan was reviewed against the actual model registries and adapter source.
The important correction was to distinguish an existing principal-root UKF
target from a certified direct-factor `TFFactorSRUKFModel` contract. PP-UKF and
STR-UKF were certified through one-time pre-trace factor conversion; their
principal-root/eigen routes remain historical comparisons, not temporal
fallbacks. SGQF, DPF/LEDH,
multiplicative-SV, domain-clipped SIR, and generic covariance-provider routes
remain non-applicable to this campaign. Singular and rank-changing routes are
value-only and carry no analytical score claim.

## Inventory and execution

The machine-readable inventory has 24 unique rows:

- `eligible_score`: 5 rows, all executed: `model_a_affine`,
  `model_b_nonlinear_accumulation`, `model_c_nonlinear_growth`, `PP-UKF`, and
  `STR-UKF`.
- `eligible_value_only`: 1 row, executed: `structural_ar1_quadratic_h16`.
- `adapter_required`: 4 rows: Common V2 LGSSM, range-bearing, predator-prey,
  and the LGSSM-EXACT registry candidate.
- `not_applicable_contract`: 6 rows after the managed-GPU correction:
  multiplicative SV, spatial SIR, PP-SGQF, SIR-SGQF, active frozen-T10
  SVX-ZC, and the generic MacroFinance adapter.
- `blocked`: 5 rows: SVX-SGQF, KSC-UKF, PP-ZC, STR-ZC, and SIR-ZC.
- `historical_only`: 1 row: the legacy actual-SV independent-panel route.
- `owner_excluded`: 2 rows: SIR-UKF and SSL-LSTM.

Every row has a source path/anchor, source checksum, status, and reason. The
inventory validator rejects duplicates, missing canonical rows, unknown status,
missing source metadata, and score claims outside the factor contract.

The original artifact snapshot classified SVX-ZC as blocked. The later GPU 3
gate and matching terminal sequential-HMC evidence restored that registry lane
for its frozen target. It remains non-applicable to direct-factor SR-UKF, so no
direct-factor numerical row or score claim was added.

## Numerical evidence

The score rows were run in float64 eager and host-XLA modes. Historical
principal-root values/scores were recorded for comparison, while the direct
route used block QR and retained direct stack/derivative residual diagnostics.
The affine row agrees to floating-point precision. Nonlinear historical deltas
are recorded rather than relabeled as failures, because factor gauge and
sigma-point orientation can change a nonlinear approximation:

| Model | Value max abs delta | Score max abs delta | Minimum QR pivot | XLA |
|---|---:|---:|---:|---|
| `model_a_affine` | `0.0` | `2.22e-16` | `1.272e-1` | pass |
| `model_b_nonlinear_accumulation` | `1.691e-4` | `8.423e-4` | `7.051e-2` | pass |
| `model_c_nonlinear_growth` | `3.002e-2` | `2.472e-2` | `2.798e-1` | pass |
| `PP-UKF` | `2.221e-6` | `7.775e-6` | `1.441` | pass |
| `STR-UKF` | `1.657e-1` | `1.584` | `2.211e-1` | pass |

PP-UKF and STR-UKF analytical scores match centered finite differences of the
same direct-factor programs with maximum absolute errors `2.15e-9` and
`1.62e-8`, respectively. STR-UKF's larger historical delta is retained as a
factor-gauge-sensitive nonlinear comparison, not represented as exact parity.

The rectangular structural route returned a finite on-support value
`-1.4250004806894907`, minimum observation rank one, zero support residual,
and eager/XLA absolute delta `8.88e-16`. It emits
`value_only_rank_discovery` and no score.

The robustness matrix covers direct QR scales `1`, `1e-4`, `1e-8`, `1e-12`,
`1e-14`, and `1e-15`, all finite with zero direct-stack reconstruction
residual. An explicit relative pivot floor of `1e-8` fails closed on a
`1e-14` pivot. Direct-stack SVD diagnostics cover exact rank-zero, exact
rank-one, and repeated singular values, all finite with zero reconstructed Gram
residual; these remain value-only branches.

## Verification and survey

Executed:

```text
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_model_coverage_20260817.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_direct_factor_srukf_model_coverage_inventory.py tests/test_block_qr_conditional_tf.py tests/test_rectangular_factor_tf.py tests/test_rectangular_srukf_tf.py tests/test_factor_srukf_tf.py tests/test_factor_srukf_model_parity.py tests/test_factor_srukf_route_guard.py tests/test_srukf_backend_policy.py
```

Result: `31 passed, 1 warning`; the warning is the pre-existing local HDF5
build/runtime mismatch. The survey LaTeX document was compiled twice with
`pdflatex`; both passes exited zero. The PDF, log, and SHA-256 checksums are in
the artifact root. Non-fatal overfull/underfull box warnings remain in the
documentation layout and do not affect mathematical or numerical claims.

## Claims and nonclaims

This execution establishes the direct block-QR route on the six certified
fixture contracts and classifies the rest of the repository inventory. It does
not establish that every repository model was executed as SR-UKF, exact
nonlinear Bayesian inference, broad posterior correctness, HMC readiness, or
GPU production readiness. Adapter-required rows can only be promoted after
their transition, observation, factor, parameter ordering, likelihood measure,
and derivative contracts are independently certified.
