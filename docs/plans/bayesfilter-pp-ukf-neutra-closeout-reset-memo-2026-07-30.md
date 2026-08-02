# PP-UKF NeuTra Closeout Reset Memo

Date: 2026-07-30

Status: `PP_UKF_DECLARED_SCOPE_COMPLETE`

## Terminal Decision

PP-UKF is finished for the current NeuTra tuning and validation question.
The repaired true-HMC campaign produced ten HMC-valid candidate archives. The
subsequent chain-aware posterior comparison established same-target
distributional compatibility for `L=12` and `L=17` on all 30 declared checks.
Retain both as viable, unranked kernels. Do not spend more compute qualifying
the other eight inconclusive kernels unless a later use specifically requires
one of them.

Terminal evidence:

- HMC result:
  `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/public_result.json`
- HMC manifest:
  `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/run_manifest.json`
- Posterior-validation result:
  `docs/plans/artifacts/bayesfilter-pp-ukf-posterior-validation-20260730/attempt-07/public_result.json`
- Result interpretation:
  `docs/plans/bayesfilter-pp-ukf-posterior-validation-result-2026-07-30.md`

## Evidence Boundary

The completed claim is compatibility with the archived affine plain-HMC
comparator for the same approximate PP-UKF target. It is not exact-posterior
correctness, a ranking of `L=12` versus `L=17`, sampler superiority,
cross-model transfer evidence, or default/production readiness.

## Multi-Model Restart

Reusable master machinery already exists:

- campaign CLI:
  `docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py`;
- target-agnostic composition:
  `bayesfilter/inference/neutra_end_to_end.py`;
- model/filter registry:
  `bayesfilter/testing/neutra_model_registry_tf.py`; and
- contract tests:
  `tests/test_neutra_all_models_end_to_end_contract.py`.

The CLI supports a selected `--cells` subset, so PP-UKF need not be rerun.
However, do not launch the July 18 campaign unchanged. Its plan and terminal
state predate the repaired tuning protocol and this PP-UKF closeout. Before the
next serious run:

1. refresh the executable/blocked inventory against current target code and
   evidence;
2. classify `PP-UKF` as completed and exclude it from the next campaign;
3. bind the current generic per-scope tuning and sequential HMC semantics;
4. preserve each remaining model's target-specific training protocol rather
   than transferring PP-UKF or LGSSM settings as defaults; and
5. write one refreshed multi-model plan with a bounded budget, fresh output
   root, evidence contract, and model-local stop/repair conditions.

The next campaign should reuse the generic orchestration, not rebuild it. Its
first phase is a registry and stale-contract audit; the first GPU launch comes
only after that audit identifies the currently executable non-PP-UKF cells.
