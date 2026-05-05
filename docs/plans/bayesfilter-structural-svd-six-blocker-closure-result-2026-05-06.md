# Result: structural SVD six-blocker closure

## Plan reference

- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-plan-2026-05-06.md`
- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-audit-2026-05-06.md`

## Summary

This pass executed the six-blocker closure plan as an evidence and blocker
evaluation.  The strongest new evidence comes from `/home/chakwong/python`,
where the DSGE strong residual gates pass as explicit blocker assertions:

- Rotemberg second-order/pruned measurement equations match the H-S form, but
  the `dy` pathwise identity is materially blocked.
- SGU's linear bridge does not close nonlinear equilibrium-manifold residuals.
- EZ remains fail-closed pending source-backed timing metadata.

Because the model-specific residual gates remain blocked, derivative,
compiled-target, and HMC work for those DSGE models is not justified yet.
BayesFilter guardrails were revalidated.

## Phase status

| Blocker | Status | Evidence | Allowed label |
| --- | --- | --- | --- |
| 0. Preflight | passed | BayesFilter and DSGE status/logs recorded. | `scoped_pass` |
| 1. Rotemberg residual gate | closed as blocker | DSGE strong residual tests pass; second-order/pruned `dy` residual is materially nonzero. | `blocked_pruned_second_order_dy_identity_residual` |
| 2. SGU residual gate | closed as blocker | DSGE strong residual tests pass; nonlinear canonical residual remains materially nonzero. | `blocked_nonlinear_equilibrium_manifold_residual` |
| 3. EZ timing/partition gate | closed as fail-closed blocker | Code-level probe sees two states/two shocks but no metadata; BayesFilter adapter fails closed. | `blocked_pending_source_backed_timing_metadata` |
| 4. SVD/eigen derivative certification | blocked | BayesFilter guardrails pass; no model/backend residual gate is promotable. | `value_only_blocked` |
| 5. Compiled static-shape parity | blocked | Backends remain eager/eager_numpy; no compiled parity target exists. | `compiled_parity_not_started` |
| 6. HMC diagnostics ladder | blocked | Residual, derivative, and compiled gates are not closed. | `hmc_not_justified` |

## Tests run

BayesFilter baseline/guardrails:

```bash
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

Result:

```text
10 passed
```

DSGE strong residual and metadata gates:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

Result:

```text
19 passed, 3 warnings
```

Warnings were two TensorFlow Probability deprecation warnings and one
read-only pytest cache warning.

Final BayesFilter validation:

```bash
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
```

Result:

```text
63 passed, 2 warnings
YAML parse passed
git diff --check passed
```

The stale-claim search found only policy text, blocker text, tests, and
allowed label definitions; it did not introduce a new production/HMC claim.

## Interpretation

The closure pass did not promote Rotemberg, SGU, or EZ.  It produced a better
answer: the first three blockers now have executable evidence and named
blocker labels.  That prevents future agents from mistaking adapter metadata or
linear bridge tests for nonlinear structural correctness.

Since the target definitions remain blocked for the mixed DSGE models, it is
not justified to implement derivative certification, compiled parity, or HMC
for those targets yet.  BayesFilter should stay at value/guardrail status until
the client repo supplies model-specific completion maps or timing metadata.

## Next hypotheses

H1. Rotemberg needs a second-order/pruned deterministic completion design for
`dy`; the next test should demonstrate that the new map drives
`dy_next - (y_next - y_current)` below tolerance on deterministic sigma-point
grids.

H2. SGU needs nonlinear equilibrium-manifold completion, not only an `h_x`
linear bridge.  The next test should reduce the canonical residual below
tolerance for `d,k,r,riskprem` on representative and boundary-adjacent grids.

H3. EZ should remain fail-closed until a source-backed timing/provenance note
explains the state ordering, shock timing, and measurement timing.  A code-level
two-state/two-shock probe is not sufficient metadata evidence.

H4. Derivative/Hessian certification should wait for a promotable model target.
Once one exists, the first backend hypothesis should be that spectral gradients
require minimum-gap telemetry plus finite-difference and JVP/VJP checks.

H5. Compiled parity should be tested first on exact LGSSM and generic
structural fixtures, then on the first DSGE model whose residual gate passes.

H6. HMC should start only after the same model/backend pair has residual,
derivative, and compiled parity evidence.  Until then, HMC labels remain
`hmc_not_justified`.
