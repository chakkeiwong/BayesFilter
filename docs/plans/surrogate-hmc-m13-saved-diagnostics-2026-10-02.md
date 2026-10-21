# M13 saved-sample convergence diagnostics, 2026-10-02

Question: what do acceptance, sampling precision, and chain behavior say about
the saved M13 surrogate-HMC posterior samples beyond the R-hat stopping rule?
This is post-run inspection under the user's request for other diagnostics.
There is no new sampling, validation phase, tuning, or master-policy change.

The target remains M13/data1704/T20's frozen importance-likelihood posterior.
Compare the previously reported 9,000-draw stopping point with all 12,750
post-warmup draws per chain. Include all four chains and exclude warmup. Use
`surrogate-hmc-m13-reclassified-20261002-02/report.json` and its checksum-bound
sample archive, reconstructing acceptances from the original saved history.
The existing public-tuner history loader verifies archive and state continuity.

Evidence contract and intent: describe whether autocorrelation, chain means,
tails, and acceptance changes corroborate the existing mixing warning. All new
diagnostics are explanatory; none creates a promotion or continuation veto or
reclassifies the earlier stopping-rule success. Invalid checksums, nonfinite
samples, or disagreement with the saved summary invalidate this inspection.
No superiority, exact-likelihood equivalence, or all-model conclusion follows.
There is no method ranking, so a heuristic-adversary comparison is inapplicable.

Pre-run skeptical audit: prefix and full histories overlap, so their mean
differences must not receive independent-sample significance tests. Good
acceptance does not establish mixing. Within-chain autocorrelation may miss
between-chain separation. ACF lags up to 1,000, 1,000-draw display blocks, and
running means are display choices, not acceptance thresholds. Preserve raw
traces, including repeats and all suffix draws, so averaging cannot hide a
long excursion. Plot choices have no effect on the master. No trace-energy
diagnostic may be invented from proposal energy errors: E-BFMI needs a
Hamiltonian-energy series that these archives do not save. Audit passes with
these limitations recorded.

Budget: one CPU-only saved-data inspection, at most five minutes, with one
localized repair if necessary; no HMC transitions. Deliberately hide GPUs
before TensorFlow import. TensorFlow/TFP computes numerical diagnostics;
Matplotlib renders them. Non-XLA diagnostic graphs are an explicit reporting
exception, not evidence about production performance. No new numerical default.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/plans/artifacts/surrogate-hmc-m13-saved-diagnostics-20261002-01/analyze.py`.
Results, plots, script, log, manifest, and a result note go under that versioned
directory. Preserve all prior artifacts. The result note records supported
health checks, descriptive findings, uncertainty, and the actual next action.
