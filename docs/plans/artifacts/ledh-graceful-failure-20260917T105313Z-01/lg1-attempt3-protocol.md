# LG1 attempt 3: non-harm-calibrated coordinate cap

Attempt 2's fixed Halton designs passed no approximation screen; preserve
`lg1-halton-curves-01/curves.json`. A subsequent dedicated Class C test found
that the inherited standardized cap 0.98 changes a healthy Gaussian cloud's
kurtosis from 2.99334 to 1.58792 and moves a coordinate by 2.37228 standard
deviations. Thus that inherited warm setting fails the non-harm criterion.
It was never a justified default. Fixing mean and covariance does not repair
that shape distortion. The independent scalar flow/weight check had passed.

Use cap=16, selected as the smallest positive value passing the predeclared
1e-4 maximum healthy standardized displacement screen, with finite flagged
stress behavior. Observed displacement was 4.53e-6. The bound and limitations
are in `ledh_cap_protocol.md`, measurements in `coordinate-cap-nonharm-01`.
This calibrates a protection, not likelihood performance. Smaller N does not
automatically inherit qualification: preserve actual pre/post cap magnitudes,
displacements, active fraction and derivatives at every observation, and
require downstream calibration and derivative checks in each exact scope.

Keep all canonical features, including the positive coordinate and RMS caps,
full UKF lifecycle, Contract E-Chol and analytical total derivative. Retain
the fixed Halton candidate as a hypothesis, not a demonstrated improvement.
Frozen arms: N256/1024/4096, flow10/3/1, cap16, all other controls unchanged.
Use calibration replicates 0/2 and phi -0.9,0,0.9 for nomination only. Stop at
the first passing scope under the existing 0.05 moment and 0.02 centered
log-ratio criteria; then require full four-dataset/seven-parameter calibration,
scope-specific derivative checks and untouched validation. Do not tune on
validation/claim observations or manufacture tuning authority.

This is substantive attempt 3, the final attempt under the master ceiling.
The preceding 18 groups took 229 seconds; forecast 344 seconds with 1.5x
headroom, bounded by a 900-second job. If no scope passes, report LG1 unready
and its downstream posterior claims blocked under the declared attempt limit.
Candidate rejection does not invalidate the method or justify changing the
prior, canonical features or thresholds. Independent engineering closure and
the public binding diagnostic remain authorized.
