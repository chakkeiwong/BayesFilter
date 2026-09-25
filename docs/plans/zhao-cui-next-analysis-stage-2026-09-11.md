# Zhao-Cui continuation: aggregate validity repair

Read this brief first in a new research conversation. It contains the state
needed for this stage; old conversations and incident reports are unnecessary.
Preserve all applicable repository instructions and unrelated worktree edits.

Research checkout: /home/chakwong/BayesFilterZhaoCui.
Base: 47176bce7cdaa91ddd4466c39f90a11ad005c800, with uncommitted implementation.
Notes/output workspace: /home/chakwong/BayesFilter.
The previous bounded audit returned REVISE; no overall correctness or filtering
accuracy approval exists. Frozen-score, conditional-law, real-C2 mechanics,
and finite Lean algebra checks passed on their recorded fixtures.

## One stage to execute

Repair aggregate validity in bayesfilter/highdim/zhao_cui_algorithm3_tf.py and
add focused checks in tests/highdim/test_zhao_cui_algorithm3_tf.py, both in the
research checkout. Inspect the current diff first. Compare current source hashes
with the selected fields of
docs/plans/artifacts/zhao-cui-audit-20260911-03/closeout-provenance.json in the
notes workspace; investigate changes rather than restoring the saved version.

Known counterexample: FrozenAlgorithm3Program with observations zeros([4,1]),
states zeros([4,2,1]), proposal log densities zeros([4,2]), the test NormalModel,
theta=[1e154,0], float64, jit_compile=False returns valid=true,
log_likelihood=-inf, and per-time weight sums 2.0. The evaluator checks local
factors, tangents, normalized logs, and increment scores but misses aggregate
overflow and lost normalization. The saved result is the finite_output_guard
field of artifacts/zhao-cui-audit-20260911-01/diagnostic-result-01.json under
docs/plans in the notes workspace.

The target stays the analytical derivative of the finite importance likelihood
with coefficients, charts, uniforms, sampled states, and proposal densities
frozen. This stage adds rejection/diagnostics; it does not silently change
normalization, clipping, damping, or the derivative target. Check increments
and total likelihood, local/aggregate/centered scores, finite normalized weights,
filtering means, and finite valid ESS. Derive a scale- and dimension-aware
normalization tolerance; document its provenance and failure mode.

Before editing, record a brief skeptical check: the counterexample must be
rejected by both the compiled evaluator's valid flag and the checked host
boundary, while existing healthy frozen-score/value fixtures must pass with
unchanged accepted outputs. This is a Class B guard, not an approximation-quality
test. Finite-difference parity alone cannot certify the validity flag.

## Execution and completion

Use the existing /home/chakwong/anaconda3/envs/tf-gpu/bin/python with
CUDA_VISIBLE_DEVICES=-1, BAYESFILTER_TEST_DEVICE_SCOPE=cpu,
BAYESFILTER_PRELOAD_CUSTOM_OP=0, PYTHONDONTWRITEBYTECODE=1, and a writable
MPLCONFIGDIR before framework import. These are explicit CPU/reference tests;
no GPU/XLA or C2 campaign is required. Allow three focused invocations of at
most 180 seconds each, including one localized repair/retry. Record each command,
exit code, wall time, environment, and outcome. Stop additional execution if
the allowance is exhausted and save the exact unresolved issue.

Use a fresh directory under docs/plans/artifacts/zhao-cui-validity-repair/ in
the notes workspace. Preserve complete logs on disk. Output only the exit
status, test summary, key counterexample result, and artifact paths. Completion
requires the failing-input regression, healthy no-fire regression, reviewed
diff, and a saved result; a plan or successful file edit alone is insufficient.

Update this brief by replacing completed-task instructions with the next task;
retain detailed evidence in the versioned result. Subsequent stages are rank
activation through the shared fitting API, manuscript/master reconciliation,
then a generic consumer and separately gated GPU/C2 validation. Constant
initializer perturbation probes from the prior audit did not select a repair.
The UKF/sigma-point guide remains a local extension. Candidate defects do not
reject the Zhao-Cui research direction.

## Keep context bounded

- Read exact files/ranges, normally at most 2,000 returned tokens per command
  and 4,000 per combined tool call. Per-command limits add up; cap outer output.
- For searches, first return file names or counts, then selected matching lines.
  A truncated result triggers a narrower read, not a larger output allowance.
- Extract needed JSON fields instead of printing complete logs, sessions, or
  nested tool results. Use explicit workdirs and absolute cross-checkout paths.
- Save each result promptly. Keep this brief near one page; do not append the
  incident history or every completed stage to the next session's entry text.
- Finish at the stage boundary and continue from the revised brief in a new
  conversation while compaction is unreliable. Do not spawn or resume another
  agent unless the user authorizes it. An earlier compaction trigger is only a
  backup; it does not repair the provider.
