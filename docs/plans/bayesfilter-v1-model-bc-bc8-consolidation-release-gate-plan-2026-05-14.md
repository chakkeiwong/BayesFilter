# BayesFilter V1 Model B/C BC8 Consolidation And Release-Candidate Gate Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC8 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Consolidate BC0-BC7 evidence and decide what Model B/C claims can be carried
into a V1 release-candidate checkpoint.

## Entry Gate

BC8 may start after BC0-BC7 are complete, blocked with explicit labels, or
deferred by documented decision.

## Evidence Contract

Question:
- What can V1 claim for Models B-C value, score, reference, HMC, GPU/XLA, and
  Hessian status for each filter?

Baseline:
- BC0-BC7 artifacts and the V1 master execution summary.

Primary criterion:
- Final result states every B/C/filter status without exceeding artifacts.

Veto diagnostics:
- claims exceed artifacts;
- out-of-lane files are dirty or staged;
- default CI depends on GPU, HMC, external projects, or long experiments.

What will not be concluded:
- External client switch-over, exact full nonlinear likelihood, broad GPU
  speedup, HMC convergence, or nonlinear Hessian production support unless
  their specific gates passed.

Artifact:
- Final B/C testing summary, reset memo update, optional source-map update, and
  scoped commit.

## Execution Steps

1. Write the final B/C evidence matrix.
2. Update the V1 reset memo with BC0-BC7 decisions and continuation state.
3. Update `docs/source_map.yml` only if new durable artifacts need source-map
   registration under the V1 lane.
4. Run fast public API tests.
5. Run focused nonlinear suite.
6. Run opt-in HMC diagnostics only if the final claims require them.
7. Run full default CPU.
8. Tidy generated artifacts and stage only V1-lane files.
9. Commit only after gates pass and no veto diagnostics fire.

## Primary Gate

BC8 passes only if the final matrix states B/C value, score, reference, HMC,
GPU/XLA, and Hessian status for each filter.

## Veto Diagnostics

Stop and ask for direction if:
- any claim lacks an artifact path;
- out-of-lane files are edited or staged;
- GPU/HMC/external checks are added to default CI;
- stale fixed-null terminology replaces structural fixed-support language for
  default Model C.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-final-testing-summary-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional if durable artifacts are added:

```text
docs/source_map.yml
```

## Default Validation Commands

Fast public API:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Focused nonlinear B/C:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Full default CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

## Continuation Rule

After BC8, open a new plan only for a named unresolved hypothesis.  Do not
continue automatically into client integration, nonlinear Hessians, exact
references, GPU production policy, or HMC convergence promotion.
