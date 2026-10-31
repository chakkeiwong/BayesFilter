# M13 nested posterior-reference refinement — 2026-09-24

The existing master goal remains testing surrogate HMC. The 33x33 localization
pilot has1089/1089 valid likelihood points but a material quadrature discrepancy:
lambda_R mean changes.04432, versus importance-sampling MCSE.0000369.
This triggers the planned reference refinement, not an HMC or model verdict.

Use the same calibration1701 data, full-support N(0,log(2)^2) parameter prior,
T20 likelihood, FP64 independent-reference exception,4096x8 importance samples,
common seed[2026092402,9101] and radius4log(2). Reuse verified saved nodes,
then evaluate the missing nodes at65x65 and129x129. The actual full-trajectory
defensive-mixture calculation remains the likelihood; no Gaussian surrogate
replaces it. Simpson integrals and paired replicate differences describe
successive resolutions. Reused source/data/grid hashes must match.

Primary question: is quadrature error now small relative to posterior scale
and importance-sampling uncertainty? Report changes in means divided by the
finer posterior SD, relative SD changes, evidence changes and replicate MCSE.
These are explanatory diagnostics for allocating endpoint/reference work,
not posterior qualification. No data-dependent thresholds or setting selection
occur in this run. Invalid cells prevent a posterior summary. Changed source
or data, corrupted evidence or nonfinite summaries veto the affected result.
A proposal failure triggers localization; no missing cell is assigned zero mass.
No result certifies omitted tails or turns the finite box into a prior boundary.

Assumption audit: the original four-prior-SD radius is a localization choice,
not a tail theorem. Common random numbers reduce noise in grid differences but
can conceal integration bias, so both sample-size/seed refinement and a tail
argument remain required before model-posterior promotion. Reusing nodes makes
the nested comparison paired and saves compute; their full provenance must
remain inspectable. The eight-replicate MCSE remains descriptive, with no
statistical ranking. The primary comparator is the actual previous grid.
Constant and tilted Gaussian checks, shift invariance and invalid-grid rejection
remain the constructed cheap sanity checks. Report central/boundary regions
separately; this reference prerequisite has no sampler efficiency claim.

Skeptical audit passes for reference refinement only. The earlier pilot's
material change demonstrates that a successful likelihood calculation alone
cannot establish a resolved posterior. Final-grid stability would still leave
tail omission, Monte Carlo approximation and downstream sampler checks open.
No untouched claim data or DCTR control-validation data are consumed.

Reserve350CPU-worker/440GPU-wall seconds plus60CPU for focused preflight in
the existing extension pool (494.187540CPU/545.148222130GPU remain).
Measured pilot numerical cost39.46CPU/36.04GPU for1089 points includes
compilation; the new15552 evaluations are forecast near300seconds. Budget
timeout is a continuation veto for this allocation; preserve completed65-grid
and partial129 rows if interrupted. Global campaign ceilings are unchanged.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-posterior-refine --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-04

Preserve source hashes, exact command/environment/seeds/GPU memory growth,
run manifest, per-row checks,65/129 summaries and actual charges. This is part
of the existing master. Continue to the smallest remaining reference/endpoint
or correction repair after inspecting the result; the campaign is incomplete.
