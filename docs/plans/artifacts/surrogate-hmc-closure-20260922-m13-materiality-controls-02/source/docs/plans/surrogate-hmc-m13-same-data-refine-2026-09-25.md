# M13 same-data nested resolution repair — 2026-09-25

Continue the existing surrogate-HMC master after same-data-posterior01.
The endpoint/reference discrepancy passed the prospective0.05SD screen, but
33/65 grid shifts exceeded0.01SD. Preserve those findings and repair only
quadrature resolution. This stage adds129-grid nodes to all four saved65 grids:
fixedN4096/8 seed2026092402 and referenceN16384/8 seeds2026092501..2503.
Reuse16900 node evaluations, compute49664 new ones. Same data1704, prior,
wrapped likelihood, box +/-4 prior SD, seeds, particles and replicate counts.

Evidence contract, mathematical uncertainty calculation, diagnostics and
nonclaims are exactly those of surrogate-hmc-m13-same-data-posterior-2026-09-25.md.
Compare129 against65 with the same ratio-estimator paired MCSE. The four
mean/SD endpoint envelopes must be<=0.05SD and grid envelopes<=0.01SD.
These are local precision screens, not unbounded posterior admission.
Missing, duplicate, changed or invalid saved cells stop reuse. Verify actual
numerical dependency hashes, data checksum, grid hashes and complete65x65x8
shape for every group. New finite fit-invalid cells are preserved and suppress
summaries; never discard them. Save every new row before continuing. No prior
artifact is changed. Retain control qualification and posterior tails as gaps.

Skeptical audit: refining a shared quadrature isolates the observed error;
independent seeds and increased particles are already present. Re-evaluating
old nodes would waste budget and introduce an unnecessary replay discrepancy.
129 is a new resolution hypothesis, not borrowed qualification from data1701.
Passing can still hide omitted tails; finite-box results stay explicitly limited.
This numerical repair cannot certify HMC convergence or tune proposal controls.

CPU/GPU budget: separate bounded pass4500CPU/5000GPU within remaining
5901.521923CPU/189495.994144GPU. Worker4000CPU/4400wall; review120CPU.
Measured predecessor costs predict roughly3200 seconds for the new cells.
Up to3 localized attempts within these bounds; exhausted global budget stops
further execution. User campaign authorization persists. Reference FP64,
GPU/XLA, memory growth; default force remains FP32/TF32. Result note is the
same-data posterior result with an explicitly dated refinement addendum.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-same-data-refine --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-refine-01
```
