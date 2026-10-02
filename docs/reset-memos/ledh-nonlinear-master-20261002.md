# Completed delivery — nonlinear master, 2026-10-02

Question: document the revised higher-moment correction, synchronize Git, and
provide a master tester for predator-prey and Austria SIR with d=18.
Plan: docs/plans/ledh-nonlinear-master-20261002.md.
Result: docs/benchmarks/ledh-nonlinear-master-results-20261002.md.
Runner: docs/benchmarks/run_ledh_nonlinear_master.py.
Guide: docs/benchmarks/ledh-nonlinear-master.md.

Implementation, manuscript and bounded validation complete. Development commit
1ccf9375e merged with remote 3673aebf1 in 20e70c9e8. Both code conflicts resolved;
all shared analytical-score, richer-design and safety features retained.
20e70c9e8 was pushed; main, sqmc-development and remote main were checked identical
with clean worktrees. This completion record adds documentation only; publish it
through main and fast-forward development, then recheck live refs. Live Git refs
are authoritative for the final delivery SHA.

Checked: 75 distinct CPU reference tests plus 3 required oracle-contract checks;
both models T2/N1008 FP32 GPU/XLA/TF32 guarded-pairwise execution valid, exact
trace/value/first-score parity. Earlier small SIR failures preserved. GPU smoke
wall time 539.916/1800 seconds. CPU validation and two commit hooks about
308/600 seconds; no scientific comparison campaign consumed this budget.

Merged monograph: 605 pages, no undefined refs/citations or duplicate labels;
chapter mirrors equal. Five tracked figures restored in sparse checkout. Updated
table page inspected; existing global overfull boxes remain. Final PDF:
docs/plans/artifacts/ledh-nonlinear-master-20261002/merged-monograph-build-02/main.pdf.
Evidence and manifests are under the same dated artifact root.

Remaining scientific work: same-target nonlinear reference construction,
scope-specific calibration, replicated held-out likelihood/score comparisons,
and localization of small-SIR failures. Current smokes establish execution only;
no nonlinear accuracy ranking, broad numerical safety or HMC readiness.
Next justified action is a bounded reference/calibration plan, not an unplanned
full default-grid launch. Preserve unrelated worktrees and historical evidence.
