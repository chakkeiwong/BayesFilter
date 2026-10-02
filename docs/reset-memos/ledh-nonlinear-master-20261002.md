# Active checkpoint — nonlinear master, 2026-10-02

User authorized LaTeX update, all task commits, conflict resolution, remote main
push, and synchronization back to sqmc-development; also a nonlinear master tester.
Plan: docs/plans/ledh-nonlinear-master-20261002.md.

Stage: implementation and monograph complete; final reporting/Git integration.
Branch sqmc-development started at 688878c8f. Main clean worktree:
/tmp/bayesfilter-sqmc-main-integration-20260924. Remote fetched; main has advanced.
Preserve other worktrees. No commits or merges yet in this task.

Checked: 49 CPU reference regressions pass. Shared executor runs both adapters.
Predator–prey T2/N72 FP32 original and guarded arms valid; SIR T2/N1008 FP32
with declared controls valid, including trace/value/score parity. Small SIR
T1/N36 and T2/N72 reduced-control FP32 fixtures invalid, preserved. SIR T1/N72
FP64 valid. Different controls/precision/count prevent causal localization.
GPU smoke wall time 442.0s of 1800s; CPU checks about 41s of 600s. No full
scientific comparison, tuning or exact nonlinear oracle established.

Monograph 605 pages, both chapter copies equal; no undefined refs/citations or
duplicate labels. Added T120 actual KSC tables and uncertainty, score-error
identity and nonlinear testing discussion. PDF pages 216–219 inspected.
Evidence: docs/plans/artifacts/ledh-nonlinear-master-20261002/.

Next: finish result note and reporting review, commit our complete dirty work,
fast-forward main to origin/main, merge sqmc-development, resolve conflicts,
validate affected files, push main, fast-forward and push development, verify
identical local/remote commit IDs and clean worktrees.
