# Controlled SMC/KDM commit, main synchronization and restart

## Scope and skeptical audit

The owner requests committing the LEDH work, merging it into main, resolving
conflicts, committing main and bringing main back to the LEDH branch, followed
by a master-program refresh. Execution checkout:
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
branch `ledh-refactor-with-policy-fix`, initial HEAD `90a3acebd`.
Main checkout: `../ledh-document-main-sync-20261015`, initial HEAD `4a1990d7d`.
The root checkout is a different, dirty `surrogate-hmc` branch and is outside
the write set. Main's untracked LaTeX audit plan is also outside the write set.

Audit: main and LEDH have diverged by four and three commits respectively.
Both sides contain shared numerical changes, so resolving a conflict by taking
one entire side would discard required functionality. Preserve the shared
checked analytical implementation, guide integration and main's updates; inspect
each overlapping hunk and run focused regressions. This is integration, not a
new performance or guide-efficacy experiment. No campaign or N64 result is
promoted by a passing merge test. Existing N64 evidence remains disqualified.

## Steps and validation

1. Commit authored source, tests, LaTeX/PDF, plans and retained result/manifest
   evidence from the LEDH worktree. Keep duplicate generated source trees,
   training datasets, profiler dumps and render images locally. Retain compact
   source indexes/diffs where available. Do not rewrite recorded experiments.
2. Fetch the main ref, reconcile it in the existing main worktree, merge LEDH,
   inspect conflicts and commit the resolved integration. A local merge does
   not imply a remote push; the current request does not ask for one.
3. Merge main back into LEDH and verify ancestry plus the shared source hashes.
4. Refresh the active master/checkpoint around fresh N>=1008 simulated-moment
   training and downstream evaluation of controlled-SMC and SIR neural-KDM
   proposals. Preserve the completed slowdown trace as engineering evidence.
   Record the next bounded preparation step and distinguish it from an
   adequately budgeted fresh training campaign.
5. Commit the refreshed planning state and synchronize that state through main
   and back, so both requested branches contain the final master program.

Validation: whitespace checks on authored files, focused CPU-only harness and
numerical regressions appropriate to any conflict, ancestry checks and exact
source comparison between main and LEDH. CPU checks intentionally hide CUDA;
they cannot establish GPU performance, N1008 training feasibility or efficacy.
Stop integration only for an unresolved semantic conflict, failed targeted
regression or concurrent edits to the same affected files. Preserve all work
and investigate a localized failure before asking for direction.

## Execution record

The LEDH snapshot is committed as `9bc3e7ec7` (355 files: authored source,
tests, paper/PDF, plans and compact evidence). Bulk generated datasets, source
copies, profiler dumps and images remain local. Essential N1008 analysis,
parity, campaign-manifest and disqualification JSON files are also explicitly
retained with this planning refresh; their ignored status is not an evidence
policy.

Fetched `origin/main`; its tip was `4a1990d7d`, matching local main. Merge
`90515228c` integrates LEDH into main and has been fast-forwarded back to LEDH.
No textual conflict required manual resolution. The only automatically merged
code file was `covariance_proposal_score_batch_tf.py`: main's nonshared Gaussian
helper and invariant transport packing coexist with LEDH's pre/post-reset
diagnostics and scale-aware mean validity check. Neither side was replaced
wholesale.

During integration main acquired nine unrelated modified LaTeX/bibliography
files. Their paths did not overlap the incoming change, and they remained
unstaged. The existing untracked LaTeX audit plan and new results chapter were
also excluded. The dirty root `surrogate-hmc` checkout was not modified.

Validation used the `tftwogpu` environment with `CUDA_VISIBLE_DEVICES=-1`,
`TF_CPP_MIN_LOG_LEVEL=2`, `TF_NUM_INTRAOP_THREADS=2`,
`TF_NUM_INTEROP_THREADS=2`, and `PYTHONPATH=.`:

- Before merge, `python -m pytest -q` on
  `tests/test_sir_n1008_runtime_diagnostic.py`,
  `tests/test_sir_runtime_repair_diagnostic.py`,
  `tests/highdim/test_ledh_checked_flow_determinant.py`,
  `tests/highdim/test_ledh_flow_guard_batching_tf.py`,
  `tests/highdim/test_ledh_neural_kdm_tf.py`, and
  `tests/highdim/test_ledh_physical_proposal_tf.py`: 97 passed in 340.72 s.
  Log: `/tmp/ledh-main-sync-20261023-premerge-tests.log`.
- On the merged main tree, `python -m pytest -q` on
  `tests/highdim/test_covariance_proposal_packed_score.py`,
  `tests/highdim/test_covariance_proposal_precision_guard.py`, and
  `tests/highdim/test_covariance_proposal_native_batch.py`: 56 passed in 217.03 s.
  Log: `/tmp/ledh-main-sync-20261023-merged-tests.log`.
- The existing CPU-only oracle commit hook passed; it was not bypassed.
  Whitespace and unresolved-conflict checks passed. Small fixture results
  establish integration mechanics only, not N1008 scientific or speed claims.

The active master and checkpoint now point to
[the reviewed fresh restart](controlled-smc-kdm-n1008-restart-20261023.md).
It identifies the remaining hard-coded N64 callers, the row-loop neural
training lane, the need for actual-consumer gradient/validity checks and the
fresh calibration/train/heldout separation. This documentation update is
committed on LEDH and synchronized through main; final verification compares
the two branch tips and trees and checks that unrelated work remains unstaged.

No new GPU budget is spent by this synchronization. Historical campaign
allowances are not reset or reassigned to the new N1008 study. No push is
included in this request.
