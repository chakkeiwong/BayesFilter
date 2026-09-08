# Phase 0 Launch and Contract-Closure Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_GATE`  
Budget cap: `1800 s`  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase0`

## Objective

Turn the reviewed documentary direction into an executable, reproducible
campaign boundary without sampling, GPU work, replay replacement, or HMC.

## Entry and exit

Entry requires the Fable and Grok reply files to be present and the active
modular plan status to identify both reviews. Exit requires a clean source/path
inventory, a valid environment/import check, a unique artifact root, and a
recorded skeptical audit. A missing runner is an implementation gap to repair,
not a reason to claim the science is blocked.

## Work items

1. Reconcile the modular plan status and review tail with the preserved Grok
   reply; do not alter its scientific role boundaries.
2. Record commit, environment, CPU/GPU visibility, TensorFlow version, and the
   exact source/review paths.
3. Verify that the planned runner and output roots are absent/present as
   expected; create only the new versioned campaign root.
4. Run focused existing tests for `annealed_smc_tf`, importance sampling, and
   q20 receipt/harness imports with `CUDA_VISIBLE_DEVICES=-1`.
5. Write the master-program skeptical audit before Phase 1 implementation.

## Gates and diagnostic roles

| Check | Role | Pass condition |
|---|---|---|
| Review/path closure | promotion veto for execution layer | both reply files resolve and are documentary-only |
| Environment/import | hard engineering veto | TensorFlow/TFP imports and focused tests pass |
| Device policy | hard artifact veto | CPU lane explicitly hides GPU; no GPU initialized |
| Existing q20 harness import | explanatory/repair trigger | import or test failure is classified and localized |
| Stale status text | repair trigger | active plan points to master program |

## Exact checks

```text
git status --short
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_annealed_smc_tf.py tests/test_importance_sampling_tf.py
```

The q20 physical harness tests may be added only if the focused tests complete
within the phase cap; they remain historical comparators and do not authorize
reuse of their six-bank output.

## Repair boundary

Repair stale documentation, imports, test fixtures, or artifact setup in scope.
Do not tune q20, change the target, install packages, or run GPU/HMC. After the
phase, update the Phase 1 subplan with the actual environment and any failures.

## Required artifacts

- `phase0-result-2026-08-25.md`
- `phase0-repair-and-refresh-2026-08-25.md`
- `master-program-review-2026-08-25.md`
- machine-readable manifest with command, commit, environment, and wall time
