# Active checkpoint — nonlinear master, 2026-10-02

User authorized LaTeX update, task commits, conflict resolution, remote-main
push, synchronization back to sqmc-development, and nonlinear master tester.
Plan: docs/plans/ledh-nonlinear-master-20261002.md.
Result: docs/benchmarks/ledh-nonlinear-master-results-20261002.md.

Stage: implementation, merged validation and manuscript complete; finish Git.
Development committed at 1ccf9375e. Main integration worktree:
/tmp/bayesfilter-sqmc-main-integration-20260924. Main fast-forwarded to remote
3673aebf1 and development merged without committing yet. Conflicts resolved;
source/tests/docs staged next. Preserve other worktrees and background Git GC.

Checked merged code: 75 distinct focused CPU tests pass; both models T2/N1008
FP32 GPU/XLA/TF32 guarded pairwise valid, trace/value/score parity exact. Earlier
small SIR failures retained, not erased. No nonlinear accuracy campaign,
scope-specific tuning, oracle certification or statistical ranking established.
GPU smoke budget used 539.916/1800 s. CPU checks plus first commit hook
about 224/600 s. Remaining hook is expected under the remaining CPU allowance.

Merged monograph: 605 pages; no undefined refs/citations or duplicate labels;
chapter mirrors equal. Restored five tracked sparse-checkout figures. Updated
table page inspected; existing global overfull boxes remain. Evidence root:
docs/plans/artifacts/ledh-nonlinear-master-20261002/.
PDF: merged-monograph-build-02/main.pdf (generated, not committed).

Next: stage integration records, commit main, push origin main, fast-forward
sqmc-development to main, copy final generated PDF into the development workspace,
verify identical main/development/origin-main/remote SHA and clean worktrees.
No remote development branch existed; create no extra branch solely for syncing.
