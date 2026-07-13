# Phase 0 Subplan: Baseline And Score Governance

metadata_date: 2026-07-07
status: `DRAFT_LAUNCH_GATE`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 0

## Phase Objective

Freeze the score program baseline before implementation:

- the six-row value artifact is the only score row-set anchor;
- score means no-tape total derivative of `log_likelihood`;
- `GradientTape`, `ForwardAccumulator`, hidden autodiff, and stopped partial
  derivatives are banned for admitted LEDH score evidence;
- prior LGSSM/fixed-SIR diagnostics are inventory items, not automatic score
  admissions for this program.

## Entry Conditions Inherited From Previous Phase

- The value-only runbook closed locally with six admitted value rows.
- Phase 8 value integration artifact exists:
  `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- This score launch package must pass local checks and read-only review before
  Phase 0 execution.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-subplan-2026-07-07.md`
- Phase 0 result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-result-2026-07-07.md`
- Phase 1 score schema subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-subplan-2026-07-07.md`
- Phase 0/1 review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase0-result-phase1-subplan-review-bundle-2026-07-07.md`
- Updated execution ledger.

## Required Checks/Tests/Reviews

Launch-gate checks:

```text
python -m py_compile docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Phase 0 checks:

```text
python -m json.tool \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/test_ledh_fixed_sir_manual_score_phase4.py -q
```

Static Phase 0 no-tape inventory check:

```text
python - <<'PY'
import ast
import importlib
import inspect
from pathlib import Path
targets = {
    'docs.benchmarks.benchmark_ledh_same_target_lgssm_m3_t50_value': {
        '_compact_value_and_score_from_components',
        '_compact_forward_transport_jvp_tf',
        '_manual_score_diagnostic',
        '_manual_transport_vjp_tf',
        '_manual_forward_transport_tf',
    },
    'docs.benchmarks.benchmark_ledh_same_target_fixed_sir_score': {
        '_fixed_sir_manual_score_diagnostic',
        '_fixed_sir_same_scalar_fd_diagnostic',
        '_require_fixed_sir_score_args',
    },
}
for module_name, helper_names in targets.items():
    module = importlib.import_module(module_name)
    for helper_name in helper_names:
        source = inspect.getsource(getattr(module, helper_name))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in {'GradientTape', 'ForwardAccumulator', 'gradient', 'jacobian', 'batch_jacobian', 'watch'}:
                    raise SystemExit(f'{module_name}.{helper_name}: banned attr {node.attr}')
            if isinstance(node, ast.Name):
                if node.id in {'GradientTape', 'ForwardAccumulator'}:
                    raise SystemExit(f'{module_name}.{helper_name}: banned name {node.id}')
print('PHASE0_NO_TAPE_INVENTORY_OK')
PY
```

Review:

- bounded read-only review of Phase 0 result and Phase 1 subplan before Phase
  1 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What rows are eligible for LEDH score repair, and what score evidence is currently admitted, diagnostic, or blocked? |
| Baseline/comparator | Phase 8 value integration artifact, prior LGSSM/fixed-SIR score diagnostics, and current score tests. |
| Primary criterion | Phase 0 writes a baseline result that freezes exactly six eligible value rows, states no score row is admitted by Phase 0, distinguishes diagnostics from admissions, and drafts a Phase 1 schema/guard plan. |
| Veto diagnostics | Diagnostic SIR promoted; value artifact missing or invalid; row-set mismatch; score admitted at baseline without replayable score artifact; tape/autodiff allowed; HMC/runtime/scientific claim. |
| Explanatory diagnostics | Existing tiny score tests, historical score-memory artifacts, exact/FD comparators, and current test status. |
| Not concluded | No new score admission, no score correctness beyond previously scoped diagnostics, no leaderboard rebuild, no HMC readiness, no posterior correctness, no scientific superiority, and no runtime ranking. |
| Artifact | Phase 0 result, Phase 1 subplan, ledger entry, and review bundle. |

## Step-By-Step Plan

1. Read the Phase 8 value integration artifact and record the exact eligible
   row ids.
2. Inventory existing LGSSM and fixed-SIR score diagnostics as prior evidence,
   not automatic admission.
3. Confirm the parameterized SIR diagnostic row is excluded from the score row
   set.
4. Run the required local checks.
5. Write the Phase 0 result with a decision table and nonclaims.
6. Draft Phase 1 score schema/guard subplan.
7. Send the Phase 0 result and Phase 1 subplan for bounded read-only review.

## Forbidden Claims/Actions

- Do not admit any score in Phase 0.
- Do not implement score code in Phase 0.
- Do not promote diagnostic SIR evidence as full main-row evidence.
- Do not use or authorize tape/autodiff score routes.
- Do not rebuild a leaderboard.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  runtime ranking, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 1 may start only if:

- Phase 0 result exists;
- local checks pass or any failures are recorded as blockers;
- Phase 1 subplan exists;
- read-only review agrees that the baseline and Phase 1 guard plan are safe.

## Stop Conditions

Stop and write a blocker result if:

- the Phase 8 value artifact is missing or invalid;
- the value row set is not exactly the six main rows listed in the master
  program;
- a current score artifact is ambiguous enough that it could be mistaken for
  score admission without replay;
- any Phase 0 check fails and the failure is not a simple documentation repair;
- read-only review finds a material issue that does not converge after five
  rounds;
- a human approval boundary is reached.
