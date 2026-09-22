# A09 covariance-collapse debugging note

Question: why did d4-s04/t8 pass covariance validation, and which protections
actually ran? This is a bounded replay of an observed failure, not a new
method comparison, tuning run or promotion decision. Preserve A09 evidence.

Before execution: inspect the actual call chain from the A09 driver through
build_guide_path, sgqf_update, Chart.from_moments and spd_factor, then through
both Gaussian and TT consumers. Replay only the level-2 node weights using
the saved t7 moments and t8 observation. CPU float64, GPU intentionally hidden,
eager execution is an explicit debugging exception; no production timing or
bitwise GPU-reproduction claim. Cap the replay at 120 seconds. The saved GPU
covariance is authoritative; tiny CPU/reconstruction differences are expected.

Evidence contract: compare saved covariance scale with predictive covariance;
report the exact active eigenvalue tolerance, normalized node concentration,
and existing higher-moment/level diagnostics. Verify the current guide module
hash against the executed manifest. Nonfinite diagnostics or source mismatch
invalidate replay. This explains failure and missing guards; it does not
validate a covariance floor, an alternative guide, or downstream repair.
Skeptical review: no baseline selection, new thresholds or method mutation;
the saved failure answers this debugging question without a new campaign.

Command: CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2
TF_NUM_INTEROP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/plans/artifacts/observation-tt-warm-improvement-20260916-01/covariance-collapse-diagnostic.py
Full stdout/stderr: covariance-collapse-diagnostic.log; result JSON:
covariance-collapse-diagnostic.json, beside this note.

## Checked diagnosis

Replay passed in 4.249339 seconds; the guide source hash matches the executed
A09 manifest. At d4-s04/t8, the likelihood at the dominant level-2 node exceeds
every other node by at least 72.4968 log units. Its normalized quadrature weight
is 1.0 in float64; the sum of all other absolute weights is 1.37806e-31.
The nine-node quadrature has effectively become a point mass. This is a
quadrature-resolution failure, not evidence that the true posterior is that
concentrated. One saved 131072-particle reference repetition has covariance
diagonal [1.07004, .24732, .70674, .89713] at this time.

The saved covariance eigenvalues span 3.71644e-32 to 2.43588e-31. The actual
guard is lambda_min > 64*machine_epsilon*d*max(abs(eigenvalues)); its bound
is only 1.38464e-44, so this covariance passes. Predictive eigenvalues span
1.20191--1.31925. In predictive coordinates the posterior eigenvalues are
2.94739e-32--1.97960e-31. The guard tests relative conditioning within the
candidate covariance, not loss of physical scale against an independent
reference covariance. An arbitrarily small multiple of the identity passes.

Levels 3--5 fail SPD checks because their signed weighted covariances have
negative eigenvalues. The selector therefore accepts level 2. With only one
accepted level, successive-level discrepancy is None. Standardized fourth
moments reach 2.22e31--7.21e31; these were saved but did not reject the chart.

The active call chain is A09 driver build_guide_path -> sgqf_update ->
Chart.from_moments -> spd_factor. The SGQF joint consumer reuses spd_factor.
The pair TT consumer draws its polynomial/defensive-Gaussian mixture in chart
coordinates and then applies current_chart.forward. Thus its defensive
Gaussian shares the collapsed physical covariance. Process noise is added
before observation reweighting and is not a posterior covariance floor.
The exact importance correction cannot supply missing proposal coverage.

Decision: the active guide protections are incomplete for this failure. A09
explicitly froze the guide without a ridge for comparability, but its review
missed the distinction between positive definiteness and adequate scale.
No runtime behavior has been changed by this diagnosis. The next reviewed
amendment should first detect scale loss/node concentration and record usable
resolution diagnostics, then evaluate a principled guide repair or defense
whose physical spread is independent of the failed chart. A covariance ridge
alone is not established as sufficient: the guide mean and integral also fail.
Do not tune or make new promotion claims on this exposed confirmation case.
