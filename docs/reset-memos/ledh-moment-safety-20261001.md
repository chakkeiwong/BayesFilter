# Completed checkpoint — LEDH moment safety, 2026-10-01

Question: document the richer-design marginal/pairwise reset and repair its
fitting instability with a bounded, observable optional guard.
Stage: implementation, reviewed plan, bounded execution, monograph and result
record complete. Branch sqmc-development; baseline 688878c8f; changes uncommitted.
User authorized plan, review and execution; review was by the executing agent.

Plan: docs/plans/ledh-moment-safety-repair-20261001.md.
Results: docs/benchmarks/ledh-moment-safety-results-20261001.md.
Evidence: docs/plans/artifacts/ledh-moment-safety-20261001/.
Final PDF: evidence root / monograph-build/main.pdf (602 pages; new section PDF
pages 209–216, printed 191–198; resolved references; final render inspected).

Implemented: shared covariance and post-whitening loss guard, eight trials,
protected complete-map no-fit fallback, analytical selected-branch total JVP,
retained marginal/pairwise caps and shared/batched/canonical/settings wiring.
Existing defaults unchanged. Full projected cumulant basis is explicitly outside
this guard's supported scope.

Checked: 55 focused CPU tests; 112 broad GPU cases + 14 CPU reference cases;
after a stricter trial-eigenvalue repair, 8 final GPU/XLA mechanics checks.
Broad and final-source evidence are distinguished in the result note. Healthy
GPU values exact; tangent difference at most 4.34e-19; FP64 FD discrepancy below
1.18e-10. No invalid guarded result in these checks. Original cap lower combined
loss in 26/112 cases: no accuracy/default promotion. Original explosive input not
replayed; global boundary smoothness/HMC validity remain unproved. An unrelated
stale test module failed initial collection and is recorded, not called a pass.

Conservative charge: 324/1800 GPU seconds, 200/1800 CPU seconds; three campaign
attempts used. Build within separate 20-minute allowance. No running job or
pending execution. Next action: await the user's next scientific scope; any
filtering-quality/default/HMC claim needs fresh scope-specific tuning and evidence.

## Reporting preference and numerical readout — 2026-10-02

Owner requests actual values, effect sizes, per-dataset trade-offs and uncertainty
for scientific comparisons, rather than reducing the answer to binary verdicts.
Preserve predeclared experimental criteria in the evidence, but present the
measured results first. Existing KSC full-mixture likelihood/score oracles are
available; the missing work is a matched oracle comparison of the latest guard,
not obtaining reference likelihoods or scores.

The September design/cap results have been re-expressed quantitatively, with
all 16 untouched cells, validation data, raw likelihood/score coordinates,
paired intervals, CSV and figure under evidence root / oracle-readout-20261002/.
This is saved-result reporting only; no new filtering experiment was run.
