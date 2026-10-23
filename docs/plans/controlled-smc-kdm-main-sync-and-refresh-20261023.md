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

Pending. No new GPU budget is spent by this synchronization. Historical
campaign allowances are not silently reset or reassigned to a new N1008 study.
