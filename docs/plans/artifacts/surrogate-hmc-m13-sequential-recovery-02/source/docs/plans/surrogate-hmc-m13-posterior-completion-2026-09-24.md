# M13 posterior-reference completion — 2026-09-24

This is a bounded completion retry for the existing M13 independent posterior
reference refinement. The preceding attempt
`surrogate-hmc-closure-20260922-m13-reference-extension-04` completed the
full 65 x 65 grid and 125 of 129 rows before its worker CPU allowance stopped
it. The preserved 129-grid rows contain 12,030 finite evaluations. The only
missing cells are the 386 new cells in rows 0, 1, 127, and 128. The preserved
files are evidence and must remain unchanged.

The question is whether the finite-box independent reference is numerically
resolved enough to be used as a later comparator for the M13 surrogate-HMC
endpoint. This completion stage does not run HMC, does not tune a force, and
does not qualify a posterior or an endpoint.

Evidence contract:
- Target: the M13 calibration likelihood integrated against the declared
  Gaussian prior on the finite box [-4 log(2), 4 log(2)]^2.
- Comparator: positive composite Simpson integration on nested 65 and 129
  grids, using the same full-support wrapped importance reference, 4,096
  particles, 8 replicates, seed [2026092402, 9101], and FP64/XLA reference
  execution.
- Primary diagnostic: all 386 missing cells complete with finite likelihoods
  and the 129-grid summary is complete; resolution is reported by the change
  in posterior means and standard deviations in units of the 129-grid
  posterior standard deviations.
- Vetoes: any source/data/grid hash mismatch, missing or duplicate preserved
  cell, changed seed/particle/replicate scope, non-finite or invalid cell, or
  incomplete 129-grid prevents a resolution judgment.
- Explanatory diagnostics: effective sample size, maximum importance weight,
  evidence MCSE, and posterior tail/boundary mass. They do not qualify HMC.
- Nonclaims: this stage does not establish exactness outside the finite box,
  posterior correctness for the production consumer, surrogate-force
  equivalence, HMC mixing, acceptance, R-hat, or default readiness.

Skeptical audit: reusing the saved rows answers the resource failure without
changing the scientific object. Re-evaluating all cells would waste the
remaining extension budget and would discard valid evidence. A complete finite
grid can still be insufficient if box or quadrature error is material, so the
result remains a diagnostic and the next independent seed, sample-size, tail,
and endpoint checks remain required.

Budget and command:
- Existing extension pass cap: 600 CPU-worker seconds and 600 GPU seconds.
- Completion reservation: 80 CPU-worker seconds, 150 GPU seconds, plus the
  bounded review tests; no new campaign budget.
- Output:
  `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05`.
- Execution:
  `/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-posterior-completion --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-05`.

The completion artifact is a finite-reference diagnostic only. A resolution
failure is a repair trigger for the independent reference; it is not evidence
against the surrogate-HMC research direction.
