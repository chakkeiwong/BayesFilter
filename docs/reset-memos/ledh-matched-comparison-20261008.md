# Completed checkpoint: matched T=50 comparison (2026-10-08)

Owner request: test the old and covariance-guided mixture filters again under
exactly the same conditions. Branch `sqmc-development`; baseline commit before
this stage `729e21825`. The prior N=128 comparison was not matched to the old
N=1008/P44 campaign.

Plan: `docs/plans/ledh-matched-comparison-20261008.md`.
Driver: `docs/benchmarks/run_ledh_matched_comparison.py`.
Result: `docs/benchmarks/ledh-matched-comparison-results-20261008.md`.
Artifacts: `docs/plans/artifacts/ledh-matched-comparison-20261008-01/`.

The campaign ran T=50, N=1008, FP64 GPU/XLA, TF32 disabled, data seed 26100611,
P44 LGSSM, KSC, predator--prey and SIR d=18, and four paired design seeds
261006101--104. Old and new received identical initial/process tensors; the
new method received separately hashed branch/ancestor uniforms. All four
calibrations produced scope-valid beta values; all 50-step workers completed.
The 20 focused CPU regression tests passed and the GPU memory-growth/XLA
preflight passed on GPU 1.

Protected old replay passed for LGSSM, KSC, predator--prey, and saved nonlinear
rows. No comparison row was invalid. Descriptively, the new candidate reduced
mean likelihood error for LGSSM, KSC, predator--prey, and SIR; score L2 error was
lower for LGSSM/KSC/SIR and essentially unchanged for predator--prey. The four-
design paired intervals include zero for LGSSM/KSC/predator--prey; SIR's
likelihood interval is favorable but its score interval is wide. Keep the new
filter optional and diagnostic only. Do not promote a default, HMC route,
production claim, or general accuracy claim from this run.

Remaining gap: repeat independent T=50 data/design partitions and investigate
SIR reference uncertainty before ranking or selecting controls. The old/new
controls differ as complete algorithm variants, so this run does not isolate a
single code-line change.
