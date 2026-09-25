# M21 complete ordinary-HMC confirmation

Source and launch details are superseded by [the merged-source continuation](bayesfilter-hmc-merged-source-continuation-2026-09-22.md). This original inventory was stopped before numerical launch; preserve it as planning history. The targets, denominators and screens carry forward to the new inventory.

All four public pilots completed with intact candidate inventories. Unit
Gaussian retained 19 verified members and its preselected L=3 member passed at
1500 retained draws; beta-binomial retained 16 and passed at 2000; LGSSM
location retained 16 and passed at 1000. Rotated Gaussian retained 19, while
its preselected member reached the 10000-draw precision cap. These are single
development observations and do not establish operating characteristics.

Measured whole-worker times, including initialization, were 183.23,
230.91, 227.95 and 256.02 seconds respectively
for Gaussian, beta-binomial, LGSSM and rotated Gaussian. The run-index records
preserve the unrounded measurements. Reallocate four unspent CPU
hours from M20 to M21: M20's CPU ceiling becomes 3600 seconds, above its
1397.7616405547597 actual charge; M21's becomes 79200 seconds. The campaign
ceiling remains 48 CPU and 24 GPU worker-hours. GPU allocations are unchanged.

Freeze 128 fresh complete fits each for Gaussian and beta-binomial, and eight
each for rotated Gaussian and LGSSM. This is 272 complete fits with independent
fixed-count comparators. The two larger groups estimate conditional interval
coverage and delivery; at probability 0.95, the normal planning half-width at
128 fits is about 0.038. Report exact pointwise Clopper--Pearson intervals.
Eight-fit stress groups provide engineering and failure replication only.
Neither their coverage rates nor cross-model differences support a ranking.

The Gaussian and beta-binomial primary screens are complete engineering
inventories, posterior-delivery lower 95% bound at least 0.90, and reported
mean/median interval delivery-and-coverage lower bound at least 0.90 in each
arm. The floor is inherited from M21's declared diagnostic adequacy screen,
not a new numerical default or simultaneous coverage theorem. It may remain
inconclusive at this count. Do not extend a denominator until a screen passes.
For rotated Gaussian and LGSSM, preserve every outcome and check public-path
invariants; statistical adequacy remains open with only eight observations.

Use the exact pilot targets, data, quantities, ordinary preparation settings,
broad L grid, selection rule, precision tolerances and fixed comparators.
Only phase, root seed (2026092231), replication count and worker budgets change.
Select the first verified candidate ID at L=3 before posterior sampling and
retain all siblings. Missing selection, warmup/precision caps, numerical vetoes
and missing intervals stay in the planned denominator. Truth is assessor-only.

Reserve 28000 seconds for Gaussian, 35000 for beta-binomial and 3000 each for
rotated Gaussian and LGSSM: 69000 CPU worker-seconds. Measured pilot scaling
projects about 56890 seconds, leaving about 21% total overhead within this
reservation. Each group has its own firm ceiling; all partial outcomes remain
visible if a ceiling fires. The M21 opening ledger plus these ceilings fits
the revised 79200-second allocation. No source-bound checkpoint is imported.

Run from immutable m21-r1/source-r1, with GPUs hidden, one TensorFlow intra/inter-op
thread, and no more than two numerical workers across this and M22. A queue
may overlap independent null confirmation with one HMC group. Record actual
commands, source hashes, seeds, CPU status, manifests, time and complete run
indices under public-confirmation-cpu-r1. Each result receives a single fixed
denominator. The existing stopping engine archives raw tuning and posterior
observations and computes the independent fixed-arm comparison.

Skeptical review: raw AR(1) accuracy cannot certify public tuning; these are
fresh complete public fits. Reallocating unused budget preserves the authorized
ceiling and is driven by measured full-fit cost. The smaller stress groups
cannot close coverage, and 128 fits can still leave a pointwise screen
inconclusive. An oracle/reference failure triggers arithmetic or target
diagnosis; valid numerical caps trigger the existing member/precision repair
agenda. A null result is not posterior correctness or default promotion.
