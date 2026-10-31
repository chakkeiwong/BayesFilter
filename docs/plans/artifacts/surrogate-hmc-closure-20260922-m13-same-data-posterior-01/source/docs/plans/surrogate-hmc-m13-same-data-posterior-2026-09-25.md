# M13 same-data posterior precision — 2026-09-25

## Intent and evidence contract

Continue the existing surrogate-HMC master after the passed local endpoint
materiality check. Test whether the fixed importance endpoint used by M13
mechanics changes posterior means or standard deviations materially on the
same seed1704 observations. The model, Gaussian prior on R^2, wrapped
observation law and proposal-force target remain unchanged. No HMC is run.

Baseline: fixed seed[2026092402,9101], N4096,8 replicates, exactly the mechanics
likelihood before FP32 casting. Comparator: N16384, three independent groups
of8 replicates with seeds[2026092501..2503,9101]. Both include the same prior.
Use positive Simpson grids33 and65 on [-4log(2),4log(2)]^2, nested at identical
nodes. This is a finite-box diagnostic of the unbounded target, not a truncated
replacement posterior. Old data1701 grids cannot be reused. Baseline and fresh
reference are each evaluated at all4225 nodes; no surrogate force enters the
reference and no interpolation replaces likelihood evaluation.

Primary diagnostic: absolute differences in the two posterior means and two
standard deviations, plus3 approximate comparator MCSE, below .05 comparator
posterior SD. Numerical grid changes plus3 paired grid MCSE should be below
.01 SD. These prospective precision allocations (5% and1% of posterior SD)
nominate a useful comparator; they cannot establish posterior correctness,
endpoint admission, convergence, force-control selection or default-readiness.
Failure triggers reference repair within the master, not direction rejection.

Validity vetoes: changed data/sources, nonfinite likelihoods, missing/duplicate
nodes or invalid reference fit. Persist all rows including failures; do not
silently discard invalid tail cells. Deliberately finish the grid after finite
fit-invalid nodes to localize failure, but suppress posterior summaries.
Uncaught numeric errors stop the worker. Stop at the master CPU/GPU allowance.

Uncertainty: integrate likelihoods before normalizing. Replicate evidence
weights propagate uncertainty through normalization. For expectation m, the
delta-method influence is (Z_r/mean(Z))*(m_r-m). For variance use the influence
of E[x^2]-E[x]^2; divide by2SD for SD. Paired grid differences use common replicate
identities. These approximate MCSEs do not bound importance bias, grid error or
omitted tails. Fixed baseline uncertainty is reported, but the discrepancy is
against its realized deterministic endpoint, so its MCSE is not added to the
fresh reference comparison. FP32 casting was separately immaterial on the
mechanics pairs; it is not silently certified on the whole grid here.

Explanatory diagnostics: likelihood ESS/max weights by nominal/intermediate/
outer regions, posterior mass beyond3 prior SD, replicate evidence dispersion,
and fine/coarse differences. Low ESS alone is a warning, not posterior harm;
missing tail mass outside4SD remains an unresolved limitation. Cheap comparator
set: fixed-seed endpoint, same-target independent seed groups, and coarse/fine
quadrature. Evaluate both center and outer regions and retain all raw estimates.

## Defaults, skeptical audit and execution

Radius4SD and grids33/65 come from the earlier reference protocol; they are
starting hypotheses, not transferred qualification. N4096/8 exactly reproduces
the saved endpoint; N16384/24 is a bounded higher-precision check, not infinity.
A .05SD difference is a local materiality hypothesis and does not replace the
final campaign posterior-equivalence contract. No data are selected using a
favorable result. Calibration1704 is kept separate from validation2704 and
untouched claim3701. Falsely reassuring outcomes include common importance
failure or omitted posterior tails; a pass therefore triggers domain-sensitivity
and qualified-chain comparisons rather than admission. Skeptical audit passes
for this explicitly limited diagnostic. Failure of a strict tail fit may be a
numerical failure rather than model or surrogate-HMC failure.

Use existing TensorFlow/TFP FP64 independent reference on GPU/XLA, with verified
memory growth. The force/HMC default remains FP32/TF32. Budget: reserve2500CPU /
3000GPU seconds from unchanged global54000CPU/226800GPU; worker1800CPU/2400wall,
review120CPU. Current balance7004.064910CPU/190589.401049GPU. At most3 attempts,
all charged and uniquely archived. No source edits while a pass runs.

Command:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-same-data-posterior --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-posterior-01
```

Save grid rows and complete grid per seedgroup, same-data checksum, source
snapshot, diagnostics, posterior summaries, result report and master receipts.
Result note: surrogate-hmc-m13-same-data-posterior-result-2026-09-25.md.

Pre-run audit:33 focused CPU-only regressions passed. The ratio-estimator
influence agrees with central finite differences of replicate weights; common
multiplicative error cancels from posterior moments, and constant likelihood
recovers the analytic truncated Gaussian prior. Public tuner capability was
inspected; this reference diagnostic issues no tuning artifact.
