# A11 implementation review

Scope: optional protected charts, the reviewed fitting comparison, and actual
importance-corrected filtering. This is a local extension for d=1 and d=4;
no author-route, high-dimensional scaling, total-gradient or default claim.

The call chain is `run_fit -> fit_path -> robust.blended_charts ->
warm.fit_step`; its current and conditioning charts are inspected by the new
consumer-wiring test. `evaluate -> a10.evaluate -> base.particle_filter ->
sample_physical_defense` draws a mixture component and evaluates both component
densities at the selected state before applying physical transition/likelihood
weights. The inherited tests exercise both components, normalization and the
actual particle consumer. These are executable wiring checks, not inspection
of an unused general function.

The lambda=1 path returns the original stable charts exactly. Other lambda
values blend PSD posterior-factor products with the independently propagated
SPD covariance. Both chart factorization and relative spectral bounds are
checked; no eigenvalue clipping or new ridge is introduced. Local analytical
Cholesky derivatives have an independent finite-difference check.

The density/normalization identities are exact mathematical identities;
sampling and evaluation use float64 arithmetic. Each TT inversion records
its CDF residual and bracket validity, with a 1e-8 residual rejection guard.
The open-unit-interval RNG uses machine epsilon at the endpoints. The final
report must count actual CDF checks and quote observed residuals, rather than
equating an absence of emitted errors with proof of exact arithmetic.

All runtime numerical components inspected here use TensorFlow. The full
Gaussian coefficient projection retains an explicitly diagnostic, bounded
d<=4 setup route and CPU TensorFlow SVD; it is not evidence of scalable
high-dimensional TT initialization or end-to-end GPU compilation. Saved
manifests disclose these existing setup exceptions. No NumPy numerical path
was found in the touched implementation or initializer.

Engineering validation: 34 focused checks and 19 additional inherited checks
pass (53 total, CPU-only with GPU intentionally hidden). The trusted GPU/XLA
smoke passes in both dimensions. Full calibration and confirmation validity,
reference precision, inference and timing remain pending until their terminal
artifacts are complete. This review cannot pre-approve their scientific result.

Reporting clarification during the run: the driver's dimension-level
`promotion_veto` summarizes any failure in that panel. It must not be read as
the verdict for every candidate. The diagnostic reporter now derives separate
arm decisions from each arm's own validity checks, heuristic comparisons and
predeclared uncertainty interval. A synthetic check confirms that a baseline
heuristic loss does not veto a passing nominee, while a nominee's own invalid
density does veto that nominee. This changes no fit, sample, selection rule,
scientific criterion or frozen driver dependency. See `check_reporting.py`.
