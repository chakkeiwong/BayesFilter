# Surrogate-HMC repository integration — 2026-10-21

The owner authorized committing the campaign, merging it into main, incorporating
origin/main, pushing main, and bringing the work branch up to date. This is a
repository closeout, with no new sampling or GPU experiment. Scientific status
is recorded in `surrogate-hmc-master-final-summary-2026-10-21.md`: four models
passed their recorded sampling rule; SIR exhausted its run budget in warm-up.
The five-model sampling objective remains incomplete.

## Evidence contract and skeptical check

The engineering question is whether both the campaign changes and concurrent
main/remote changes survive integration. Comparators are the committed work
branch `091ee89bc`, pre-integration main `ea1018de1`, and fetched remote main
`3eb9462d3`. Success requires preserving both implementations, resolving every
Git conflict, passing focused call-chain regressions and the repository commit
hook, an ordinary push, equal final branch/main/remote refs, and clean worktrees.
Unresolved conflicts, syntax errors, failed relevant regressions, or a failed
push block completion. Runtime and test counts are explanatory; these tests do
not establish posterior accuracy or convergence. No numerical defaults change.

The skeptical check found an actual merge risk: a text-only resolution could
lose either backward guidance from main or structured value/force failures from
the campaign. The merged call chain retains both and has two targeted failure
tests. The next remote integration changes the public tuning interface, so its
modern-HMC integration tests are included. CPU-only execution is deliberate,
with CUDA hidden; it checks software integration, not GPU performance. Focused
test commands have a 15-minute bound per invocation; no campaign GPU budget is
used. A missing fixture in main's sparse checkout was an environment failure,
repaired by materializing the tracked fixture directories before rerunning.

## Completed branch and main integration

- `091ee89bc` commits the campaign code, evidence, materiality correction, and
  closeout documentation. The branch was clean after committing.
- Branch checks: 301 changed active Python files parsed; 71 focused tests
  passed; the oracle pre-commit hook passed all 3 tests (141.76 seconds).
- `a32f00ccb` merges the work branch into main. The canonical score recursion
  preserves main's backward guidance and the branch's structural innovations,
  precision diagnostics, and separate value/force validity. Bibliography entries
  from both branches remain present.
- Main merge checks: 399 changed active Python files parsed; 103 distinct
  focused tests passed, including two new backward-guidance failure regressions.
  The oracle pre-commit hook passed all 3 tests (140.15 seconds). The missing
  sparse fixture caused one initial test failure; its rerun passed.

## Remote integration checks

Fetch advanced origin/main to `3eb9462d3`. Its modern-HMC changes merged
automatically in the public exports and tuning contract. The sole conflict was
the bibliography tail; both the marginal-particle-filter entry and remote
comment were retained, with unique entry keys and balanced braces checked.
All 82 focused remote integration tests passed (140.76 seconds), covering the
modern public tuning pipeline, tuning contract and dispatch, verified TensorFlow
tuning, and acceptance accounting. All 84 changed active Python sources parsed
without errors. The two warnings were dependency deprecations.

Commands, environment, timings, and preserved logs are in
[the integration validation record](artifacts/surrogate-hmc-closeout-20261021-01/integration/validation.json).
The original branch validation remains in the parent artifact directory.

The remaining delivery actions are the remote-merge commit (with its oracle
hook), ordinary push to origin/main, fast-forward of the work branch, and a
final check of both worktrees and remote refs. These happen after this record
is written; the final user report supplies their observed outcome.

The strongest integration risk was silent loss of one side's behavior despite
a clean textual merge. The executable backward-guide failures and both public
tuner families reduce that risk. These bounded tests cannot rule out every
interaction and do not establish GPU speed or scientific validity.
