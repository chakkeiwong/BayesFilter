# Five-model posterior audit and Git closeout

The owner requests actual 95% posterior interval coverage of simulation truths,
all-parameter R-hat below 1.01, chain acceptance between 65% and 75%, then commit,
merge into main, merge remote main, push main and synchronize the working branch.
This is postprocessing of completed runs; no new tuning or sampling is planned.

## Evidence contract and skeptical audit

Reconstruct all retained draws from the archived TensorFlow tensors. Verify their
manifest hashes, undo the saved nomination affine transformation, and for SIR
apply the exact frozen NeuTra map used by the run. Verify reconstructed means,
standard deviations and acceptance against the original HMC report. Map draws
through the run's literature-prior chart to physical model parameters, and
compute pooled equal-tailed 2.5% and 97.5% empirical quantiles. Warm-up is excluded.
Use simulator defaults only after tracing the saved data generation call chain;
retain the FP64 simulation truth and FP32 nominal chart center separately.

The requested descriptive checks are interval inclusion, each recorded modern
parameter R-hat strictly below 1.01, and each retained-chain acceptance within
[0.65, 0.75]. A checksum, dimension, map identity, or reconstructed-summary
mismatch invalidates this reconstruction and triggers repair before reporting.
Coverage on one dataset, R-hat and acceptance do not certify exact posterior
accuracy, repeated-dataset coverage or superiority. ESS and MCSE remain
diagnostic only. A failed requested check is reported, not hidden by retuning.
There is no competing method or heuristic ranking in this audit.

Defaults: 95% equal-tailed intervals are the conventional interpretation of the
owner's request; pooling the four retained chains uses the campaign's inference
draw policy. Both are reporting choices. Nonlinear map omission or a wrong
parameter chart is the main failure risk; agreement with saved means and standard
deviations provides the earliest discriminating check. Physical parameter names
and transformations are taken from the inspected model code. The saved HMC
reports, rather than an unrelated run or approximate normal interval, are the
comparators. These checks make the plan adequate for the stated question.

Budget: at most five CPU minutes for reconstruction, plus focused regression
tests and Git operations; zero GPU hours. Explicit CPU-only TensorFlow reporting
is a diagnostic exception, not a new execution default. Results, exact command,
input hashes, source revision and runtime go under
`docs/plans/artifacts/surrogate-hmc-five-model-completion-20261023-01/posterior-audit/`.

## Repository closeout

Preserve unrelated worktrees. Commit the current branch's campaign implementation,
tests, records and audit. Use the existing clean main worktree at
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-document-main-sync-20261015`.
Merge this branch into main, fetch origin, merge origin/main, resolve conflicts
while preserving current behavior, run relevant checks and push main. Merge main
back into `ledh-graceful-failure-execution-20260917` and verify matching commits
and clean tracked/untracked status. User authorization covers this entire sequence.

## Checked posterior result

The successful command was
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/audit_surrogate_hmc_five_model_posterior.py --output docs/plans/artifacts/surrogate-hmc-five-model-completion-20261023-01/posterior-audit-03`.
It took 4.01 CPU wall seconds, using TensorFlow 2.20.0-dev0+selfbuilt with GPU
devices intentionally hidden. An initial attempt and a logged retry used
`tf-gpu`, whose TensorFlow ABI cannot load the existing Sylvester custom op;
neither produced a scientific result. The repair used the original campaign's
`tftwogpu` environment without changing packages or libraries. The failed
retry log is preserved as `posterior-audit-02.log` beside the successful log.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| All truths included | 17/17 empirical 95% intervals contain the generating truth | All input checksums and reconstruction checks pass | One dataset per model; marginal intervals | Commit evidence and synchronize Git | Repeated-dataset coverage or exact posterior agreement |
| All parameter R-hat below 1.01 | Maximum 1.009603 in SIR | Physical-coordinate and original chart checks pass | Finite retained chains | No additional sampling phase | A mathematical proof of convergence |
| Acceptance band has one exception | Four models pass; SIR 75.9–79.3% exceeds 75% | No artifact validity failure | Observed finite-chain rate | Report the exception; no unrequested retuning | That all five meet the requested band |

No candidate ranking is attempted. Coverage and acceptance differences are
descriptive. The strongest alternative explanation for interval inclusion is
wide marginal uncertainty (visible particularly in predator–prey); inclusion
alone does not measure information gain. A wrong data or coordinate identity
would invalidate the audit, so those identities and reconstructed summaries
were checked before interpretation. Full numerical results and provenance are
in `posterior-audit-03/report.json` and `report.md`.

## Regression checks

The final CPU-only focused suite passed **71 tests** in 26.34 seconds:
`tests/test_sir_neutra_training_master.py`, `tests/test_neutra_hmc_route_policy.py`,
`tests/test_hmc_tensorflow_resume.py`, `tests/test_hmc_sequential_verification.py`
and `tests/test_surrogate_hmc_scope_source_compatibility.py`.
Command: `env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q` followed by those five paths.
The full log is `closeout-pytest-repaired.log` in the completion artifact root.
The first suite run had 70 passes and one stale test requiring the October 18
runtime hash. The repaired test verifies that the newer NeuTra runtime retains
its actual source identity and cannot inherit the historical tuning identity.
The compatibility implementation and its exact historical hash exception were
not broadened. Source and documentation whitespace checks passed. Four archived
log/error lines retain TensorFlow's original trailing spaces so the saved
execution evidence remains byte-for-byte unchanged.

The commit includes the otherwise ignored JSON run records and logs needed to
reproduce the archived results, including SIR's frozen map. Temporary lock files
are excluded. No experiment was rerun or tuned for the requested acceptance band.

## Git integration record

Campaign commit `1cd67e5d2` contains the implementation, tests, archived training
and sampling evidence, and full posterior audit. The repository's CPU oracle
contract pre-commit hook passed all three checks in 116.93 seconds.
Main merge commit `ba3d9916d` incorporates this branch without conflicts.
`git fetch origin` succeeded; merging `origin/main` then reported already up to
date, so no remote conflict required resolution. The numerical source tree on
main is identical to the tested campaign branch.

This documentation-only main commit records those checked results. Delivery
uses `git push origin main`, followed by `git merge --ff-only main` in the
campaign worktree. Final verification requires main, origin/main and
`ledh-graceful-failure-execution-20260917` to resolve to the same commit, with
both participating worktrees clean. The final response reports those checks.
