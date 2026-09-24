# A11 execution record

The owner-authorized A11 plan passed skeptical review before implementation.
Preserved manuscript and budget baselines are in this directory. The optional
chart constructor calls the existing stable route exactly at lambda=1; other
values blend checked physical covariances in a stable-signature TF/XLA kernel.
The same charts reach warm.fit_step, PairTTStep and the physical-mixture
particle consumer. The independent reference routine has explicit precision
parameters with its previous defaults preserved.

Initial CPU diagnostic run: 33 passed, one new test failed because its field
name was condition_chart rather than conditioning_chart. Corrected the test;
focused suite now 34 passed, including actual fitting/consumer wiring, bounds,
collapse protection, Cholesky derivative finite differences, reference
replicate policy, calibration selection and seed separation. GPU was hidden
before framework import. See tests-initial.log and tests-focused.log.

GPU/XLA smoke-01 completed in 118.649815 seconds on physical GPU 1, an RTX
4080 SUPER; growth configured and verified. Both d1 and d4 references pass,
baseline and maximum-capacity fits complete, and all four heuristic consumers
execute. Zero candidate numerical, CDF or log-evidence failures. Smoke MSE and
ESS are engineering diagnostics, not ranking evidence. Manifest, code
snapshots and full logs are under the numerical attempt-smoke-01 directory.
The covariance/density protections and actual consumer tests support proceeding
to the predeclared fresh calibration panel without changing the plan.

LaTeX build recovery: latexmk is unavailable; pdflatex is installed. The first
direct compile used the repository root and could not resolve an existing
relative figure path. Running two pdflatex passes from the manuscript's own
directory succeeds. These are build-environment failures, not mathematical
findings. Full logs retained.

MathDevMCP derivation review parsed four of fourteen requested equation targets.
Ten compound relations were quarantined as unknown_row_shape; two broad rigor
requests failed at retrieve_label with KeyError. A bounded audit of the exact
new section is in progress. These tool limitations do not invalidate the
hand-derived proofs or authorize a formal certification claim.

Numerical time charged so far: 118.649815 seconds; included in active A11 time,
not an additional campaign charge. Calibration attempt-01 is the next action.
# Completed regression coverage

The focused suite has 34 passing tests. The inherited SGQF initialization,
pair-block density/sampler and SGQF consumer suites add 19 passing tests
(`tests-additional-a10.log`, 6.88 seconds). All 53 pass; GPU devices were
intentionally hidden for these CPU-only checks. The combined coverage includes
the 38 A10 regression cases plus A11 and independent-inference checks.
