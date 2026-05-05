# Result: structural SVD final execution path

## Plan reference

- `docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`
- `docs/plans/bayesfilter-structural-svd-final-execution-audit-2026-05-06.md`

## Scope

This result records the execution of the final tool-gated plan.  It validates
the BayesFilter-local value and adapter gates, records tool-gated source and
derivation status, and preserves blockers for model-specific DSGE residuals,
derivatives, JIT, and HMC.

## Phase status summary

| Phase | Status | Evidence | Interpretation |
| --- | --- | --- | --- |
| Preflight | passed | `git status --short --branch`; `git log -5 --oneline --decorate` | Workspace has pre-existing dirty docs/assets; scoped staging is required. |
| 1. Source and derivation audit | partially passed | New tool-gated audit addendum; ResearchAssistant searches; MathDevMCP label lookups | BayesFilter-local value work may proceed; literature/HMC/SVD derivative promotions remain blocked. |
| 2. Code reuse and doc-code audit | passed for current scope | Existing 2026-05-04 source/code audit plus code review | Reuse existing BayesFilter core; do not rewrite passing backend code. |
| 3. Structural sigma-point core | passed value gate | `tests/test_structural_partition.py`, `tests/test_structural_sigma_points.py`, `tests/test_structural_ar_p.py`, `tests/test_filter_metadata.py` | Eager NumPy structural sigma-point backend remains `value_only` and `approximation_only`. |
| 4. Exact Kalman spine | passed value gate | `tests/test_degenerate_kalman.py`, `tests/test_filter_metadata.py` | Exact covariance Kalman path remains distinct from nonlinear structural approximation. |
| 5. Generic structural fixtures | passed | Structural AR and sigma-point tests | Generic lag/structural fixtures support the contract. |
| 6. MacroFinance adapter classification | passed as gate | `tests/test_macrofinance_adapter.py` | BayesFilter consumes readiness metadata; no MacroFinance economics migrated. |
| 7. DSGE adapter integration | passed adapter gate | `tests/test_dsge_adapter_gate.py`; client contract tests in `/home/chakwong/python` | SmallNK/Rotemberg/SGU metadata ready; EZ blocked. |
| 8. Model-specific DSGE residual evidence | blocked | No Rotemberg/SGU/EZ residual tests in this pass | Structural nonlinear promotion remains blocked. |
| 9. Derivative/Hessian safety | guardrails pass, certification blocked | `tests/test_derivative_validation_smoke.py`; `tests/test_backend_readiness.py` | Gate logic exists; SVD/eigen analytic gradients are not certified. |
| 10. JIT/static-shape gate | blocked | Backend metadata says eager/eager_numpy | No compiled production target claim. |
| 11. HMC ladder | blocked | Phases 8--10 not closed | No HMC convergence claim. |
| 12. Docs/provenance/release | passed for this pass | New plan/audit/source addendum/result/reset memo; YAML parse; diff check | Release-level production/HMC claims remain blocked. |

## Tests run

```bash
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py tests/test_structural_ar_p.py tests/test_filter_metadata.py
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
cd /home/chakwong/python && PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q tests/contracts/test_structural_dsge_partition.py
pytest -q tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
```

## Decision

The BayesFilter-local value and adapter gates are stable enough to commit the
tool-gated execution artifacts.  The next implementation phase should not be
HMC.  It should close the first blocked promotion gate: model-specific
Rotemberg/SGU/EZ residual and timing evidence, followed by derivative and JIT
gates.

## Hypotheses for next phases

H1: Rotemberg can pass a pointwise second-order/pruned completion identity test
for `dy_next = y_next - y_current` on sigma-point grids; failure would mean the
current metadata is adapter-ready but not structurally promotable.

H2: SGU deterministic coordinates `d,k,r,riskprem` require model-specific
residual derivations; if residuals fail near boundary parameters, SGU must
remain fail-closed for nonlinear structural sigma-point filtering.

H3: EZ timing can be classified without changing BayesFilter by auditing the
client model state ordering and exposing metadata only after the timing map is
explicit.

H4: SVD/eigen derivative certification will fail near small spectral gaps
unless the backend records validity regions, uses custom derivatives, or
switches factorization policy.

H5: A compiled static-shape target can match the eager BayesFilter value path
on toy structural fixtures, but this must be shown before any sampler ladder.

H6: HMC diagnostics will be meaningful only after value, residual, derivative,
and compiled-target gates pass; smoke tests should remain labeled as finite
smoke only.
