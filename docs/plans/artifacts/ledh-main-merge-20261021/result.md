# Main merge into the LEDH branch, 21 October 2026

The requested merge is committed as `90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`
on `ledh-refactor-with-policy-fix`. Its parents are the previous LEDH head
`4d6915cc66e701c4c7b657e094306d298fdd3e76` and fetched main
`d189c429542eb3a682677ce7fc463c3b796b69ea`. Worktree:
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
The separate `surrogate-hmc` checkout was not changed. Nothing was pushed.

The pre-merge audit required preserving all unfinished work, combining main's
native batch analytical computation, compact Gaussian operations and exact
transport blocks with backward guidance, and checking shared numerical and
shape contracts. This is integration work; no performance comparison or new
scientific campaign was run.

The merge preserves the FP32 covariance ceiling, the LEDH branch's FP64
half-significand floor, scale-aware mean checks and main's particle-count
allowance. Both scalar and native batch paths use the shared covariance
tolerance. Batch traces retain the branch's additional diagnostics. Exact-block
transport retains configurable denominator regularization and finite checks;
scalar adapters explicitly select the single batch validity element.

Tracked work was saved in stash
`5c2c75eaa130429e36e9a6a6572f2a7e6f8f1ad1` and restored after the merge commit.
The stash remains available. Restoration combined three overlapping files:
`higher_moment_contract_e.py`, `ledh_canonical_score_tf.py` and
`ledh_unified_correction_tf.py`. It preserves offset-based moment restoration
and physical proposal mixtures together with main's checked factorizations and
failure status. The physical proposal call extracts the affine tensors from
main's expanded flow return tuple. Restored work remains uncommitted and
unstaged. Main's checked standardization uses finite placeholders when a
covariance is invalid. The restored `materialized_shape_available` diagnostic
therefore now includes the factorization validity flag, so finite placeholders
cannot be reported as available moments. The collapsed-cloud regression now
checks main's explicit rejection and unavailable-shape flag.

SHA-256 verification found all 21,647 original untracked files unchanged and
all 12 non-overlapping tracked files byte-identical to their saved versions.
Only the three explicitly reconciled tracked files differed. The active
checkpoint was subsequently updated to record the merge and changed source
identity. One original untracked test, `test_affine_restoration_offsets.py`,
was then updated for the checked-standardization contract; its original is
also backed up. Existing experimental results and PDFs were preserved.

Validation uses `tftwogpu` (`2.20.0-dev0+selfbuilt`) with
`CUDA_VISIBLE_DEVICES=-1`: intentionally CPU-only integration checks.
Before WIP restoration, 131 distinct focused reset, guidance, moment-guard,
native-batch, packed-score and shared-Gaussian checks passed. All three oracle
contract pre-commit checks passed. Eleven additional compact-map diagnostic
tests fail in unchanged incoming code: ten monkeypatches have an outdated
Cholesky signature, and one protected-baseline mutation targets the former
scalar route. They were not repaired as part of this merge.
The restored physical/neural proposal and moment-training suite passed 48
checks and exposed the collapsed-cloud diagnostic mismatch described above.
All 16 focused moment and restoration checks pass after that repair, completing
the 49-check restored-work suite. No unresolved merge conflicts or staged
changes remain; `git diff --check` passes.

Backups and exact saved-file hashes are in
`/tmp/bayesfilter-ledh-main-merge-20261021-kk9uockg/`. Copies of verification
and full test logs are preserved beside this result. The relevant logs are
`merge-regressions-repaired.log`,
`guided-affine-regressions.log`, `batch-regressions.log`, `merge-commit.log`
and `restored-wip-regressions.log`, followed by
`restored-moment-guard-regressions.log`. No speedup or scientific improvement is
inferred from these checks. Prior training evidence retains its original
source hashes; it does not validate this merged source revision.
