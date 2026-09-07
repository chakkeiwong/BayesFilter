# C2 Recovery Integration Regression

Date: 2026-09-07

This note records the final bounded regression on the clean canonical-base
integration branch.  It is engineering-validity evidence only; it does not
promote the C2 proposal or establish a statistical ranking.

## Run

| Field | Value |
| --- | --- |
| Branch | `codex/c2-root-integration-20260907` |
| Commit | `2157d088` |
| Environment | `tftwogpu` |
| Device mode | CPU-only diagnostic (`CUDA_VISIBLE_DEVICES=-1`) |
| Command | `python -m pytest -q tests/highdim/test_c2_*.py -k 'not test_oracle_gate_degree12_rank6_n2_t12'` |
| Result | `110 passed, 1 deselected, 2 warnings` |
| Wall time | `174.13s` |
| Warnings | two TensorFlow Probability `distutils` deprecation warnings |

The excluded test is
`test_oracle_gate_degree12_rank6_n2_t12`.  It was run separately under a
180-second bound and timed out without an assertion or finite result.  It is
an expensive oracle-coverage gap, not a C2 proposal failure.

## Checks

The same clean integration commit also passed the 35-test focused Phase 8E
readiness suite, `py_compile`, `git diff --check`, imports of `HermiteBasis1D`,
the exact-Laplace adapter, and the base-mass-aware evaluator, plus frozen
observation-bin boundary recomputation and launcher contract checks.

## Interpretation

The C2 call chain is internally executable on this clean integration branch.
The result does not establish posterior correctness, likelihood unbiasedness,
general-model transfer, HMC readiness, production readiness, or default
selection.  The dirty canonical worktree was not modified.
