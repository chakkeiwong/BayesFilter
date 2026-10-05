# HMC repair Git integration, 2026-10-05

The owner requested committing the HMC repairs, merging remote changes,
resolving conflicts, and pushing to `main`. Integration uses
`/tmp/bayesfilter-hmc-v7-merge-20261005` on
`integrate/hmc-v7-20261005`, starting at
`70a6d7e9611a9f859a2519f37f236b61cac35c1e`. The fetched comparator is
`origin/main` at `a925f67a1`. The shared development checkout and the frozen
confirmation/monitor services continue independently.

## Scope and evidence

Commit the candidate-set acceptance repairs, retained-runner bridge, state-space
validation and supervision, their regression tests, the official tuning
chapter and agent reference, and the HMC plans. Include compact manifests and
results, with hashes of copied files. Raw simulation output, build products,
archived source trees, and unrelated NeuTra/q20 edits remain outside this
commit. Live ledgers are explicitly point-in-time snapshots, not terminal
results. The original campaign files remain the authority for the active run.

The engineering question is whether the combined source preserves both the
local repairs and remote HMC execution functionality. The baseline is the
recorded local commit plus the explicit HMC overlay, compared with the fetched
remote commit. Promotion to the Git branch requires resolving every conflict,
passing focused merged-source regressions and guide checks, and a clean
integration index. A failed behavioral regression or missing dependency blocks
the push until repaired. Test duration is explanatory only. These checks do
not establish GPU parity, statistical calibration, release qualification, or a
new default; the live campaign uses its own unchanged scientific contract.

## Execution and skeptical audit

1. Snapshot the scoped files into the isolated worktree and commit them.
2. Merge `origin/main`, preserving remote chain execution, TensorFlow migrations,
   and the local acceptance-policy versions and retained-candidate semantics.
3. Run CPU-only engineering regressions with the `tfgpu` interpreter,
   `CUDA_VISIBLE_DEVICES=-1`, growth enabled, and bounded threads/affinity.
   Validate the rendered route tables and official tuning chapter linkage.
4. Record the actual commands, receipts, conflicts, and limitations. Fetch again,
   incorporate any additional remote commit, then push `HEAD:main` without force.

The validation cap is 2,000 enclosing CPU wall seconds, drawn from the already
authorized release balance (2,235.56 uncommitted seconds at planning), with no
GPU allocation and no new campaign. This is a convenience safety cap, not a
scientific threshold; use focused failures to choose any follow-up within it.
Charge enclosing receipts once, without adding nested test timings. Git I/O
does not consume an experiment allocation. Stop validation on budget exhaustion
and report the unresolved check rather than infer a pass.

Audit passed before execution: the baseline and explicit overlay are identified;
neither short tests nor a successful push will be treated as release evidence;
the environment is deliberately CPU-only; an isolated checkout avoids changing
the running source or unrelated work; and the remote overlap is tested rather
than resolved by choosing one entire version. The main risks are missing
untracked dependencies, interaction with remote TensorFlow migrations, stale
generated documentation, and accidental inclusion of active/raw artifacts.
The snapshot inventory, clean-tree checks, behavioral tests, and documented
artifact selection address those risks. The live run's original deferrals are
preserved and cannot be replaced by recovery outcomes for the release criterion.

## Result

Integration in progress. No release or default-promotion verdict is made here.
