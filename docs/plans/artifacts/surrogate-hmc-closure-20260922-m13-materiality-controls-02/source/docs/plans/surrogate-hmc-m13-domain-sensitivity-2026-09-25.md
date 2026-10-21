# M13 same-data domain sensitivity — 2026-09-25

This is the next bounded stage in the surrogate-HMC master. The129-grid
precision repair passed. The question is whether expanding the integration
box from4 to5 Gaussian priorSD materially changes posterior means or SDs
on data1704. The target remains the Gaussian-prior posterior on R².

Reuse each checked129x129 core; add the coarse Simpson shell between4SD and
5SD at spacing priorSD/8. The outer81x81 grid contains the prior65x65 grid;
2336 new nodes per group,9344 total, with the same baseline4096x8 and
independent16384x24 seeds. Core129 weights plus outer81 weights minus inner65
weights form positive weights on the union. Reused boundaries receive the
sum of core and shell weights. No old node is recomputed.

Evidence contract: compare each extended-domain mean/SD with its own4SD
value, using paired replicate influence MCSE. A local sensitivity screen
passes if each absolute shift+3MCSE is<=0.01 reference posteriorSD, and
new-shell probability+3MCSE<=0.0001. These are diagnostic allocations from
the earlier1%SD grid margin and a prospective small-mass screen. They do not
bound mass outside5SD or qualify an unbounded posterior. Also preserve
fixed-endpoint vs independent-reference differences against the existing
0.05SD margin. Comparator ladder: frozen endpoint, independent importance
replicates, same-target domain expansion. Evaluate nominal/core and newly
added outer nodes separately. No sampler performance ranking is attempted.

Validity vetoes: changed data or numerical dependencies, damaged/missing
cached grids, invalid likelihood fit, nonfinite values/weights or nonpositive
normalization. Record each new row and invalid node; do not discard bad tail
cells. Expected finite fit failures trigger localization within remaining
budget, not rejection of surrogate HMC. No tuning or claim data enter.

Default audit: radius5SD is a diagnostic hypothesis, not posterior-support
truncation. Coarse shell resolution may miss narrow remote modes; its
contribution and uncertainty are reported, not asserted as a rigorous tail
bound. We retain the previously checked fine core. The same mixture proposal,
wrapped likelihood, seeds and particle counts preserve comparability.
FP64 is the explicit independent-reference exception; force default stays
FP32/TF32/GPU/XLA. Mathematical check: positive quadrature weights integrate
constant and quadratic functions over the expanded square exactly; ratio
influences propagate the same replicate evidence weights as the previous
reference. CPU-only analytic Gaussian tests check domain mass and moments.
Skeptical audit: this answers domain sensitivity, not infinite-tail control
or HMC mixing; fixed shell spacing is an acknowledged limit.

Reserve1000CPU/1100GPU from the2795.212640CPU/186402.833577GPU global balance.
Worker800CPU/1000wall; focused review120CPU/120wall. Forecast approximately
600CPU seconds from the completed refinement. Up to3 localized attempts
within this cap. Global54000CPU/226800GPU unchanged.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-domain-sensitivity --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-domain-sensitivity-01
```

Save raw new rows, same-data grids, checksums, source snapshot, conditional
weight diagnostics, paired MCSE, screen, timing and master receipt. Result
note: surrogate-hmc-m13-domain-sensitivity-result-2026-09-25.md.
Then continue scoped controls and the four-chain public-tuner adapter/cost
check within the remaining budget. Never treat a diagnostic pass as a
completed surrogate-HMC test.

Pre-run test audit: independent separable1D Simpson calculation matches the
2D union-weight implementation to2e-14. Relative to analytic Gaussian
truncation, coarse-shell errors are4.943e-8 in SD and3.630e-8 in probability.
The original2e-9 analytic test tolerance conflated quadrature approximation
with implementation error; it was replaced by the independent-rule equality
and a1e-7 absolute analytic accuracy check. Scientific screens are unchanged.
