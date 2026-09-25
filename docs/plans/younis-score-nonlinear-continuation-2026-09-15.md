# Nonlinear score-master implementation and pilot

This is Phase 0G of the authorized master, following the recovered affine
provider execution. It uses the remaining allocation in
`artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/phase-0g-repair-execution-plan.md`.
The implementation is in the persistent `younis-score-nonlinear-20260915`
worktree. Concurrent main-checkout loop changes remain preserved.

The question is whether the checked analytical score consumers, nonlinear
references and conditional reports execute together. The scalar physical model
is x[t]=a*x[t-1]+c*sin(x[t-1])+sigma_Q*epsilon[t] and
y[t]=H*x[t]+b*x[t]^2+sigma_R*eta[t]. The six parameters retain the affine
fixture's definitions, including the initial mean and variance. The curvature
coefficients c,b are fixed model settings. All shocks are independent standard
normal. This regular scalar fixture diagnoses covariance and proposal effects;
it makes no statement about DSGE transitions or high-dimensional feasibility.

The reference integrates the physical transition and observation density on a
fixed trapezoidal grid and differentiates that recursion analytically. Initial
and transition tail mass is retained as an error diagnostic, never silently
renormalized. The finite grid score approximates the model score; its derivative
identity alone would not establish reference accuracy. Before every consumer,
compare 301 and 601 points on [-10,10] and 901 points on [-15,15]. The relative
change in log likelihood, six scores and terminal moments must be at most
1e-7, with maximum lost or edge mass at most 1e-9. These are explicit reference
vetoes. The reference and data generation use CPU FP64 even in a GPU study.

The constructed heuristic set is EKF (local first-order Gaussian inference),
UKF (nonlinear Gaussian moments), bootstrap PF (exact physical simulation) and
a locally linear Gaussian proposal corrected by the physical f*g/q ratio.
Compare shared canonical LEDH, SGQF moments in the same executor and persistent
Gaussian-mixture moments in that executor with every heuristic separately.
The grid reference provides the common error reference. For resampling PFs,
the derivative holds sampled ancestor labels locally fixed; it is the derivative
of a finite program and is not an unbiased model-score estimator.

Use three fixed situations: weak curvature (c=.1,b=.03,log sigma_R=-.6),
stronger curvature (.45,.4,-.6), and concentrated observations (.25,.2,-1.3).
Each has two independent datasets, two particle replicates and eight methods:
96 numerical rows total. N=32,T=4, one CPU thread, FP64 and XLA are mechanics
choices. The two datasets cannot support a scientific ranking. Each study is
limited to 600 seconds and 40 attempts; total CPU and GPU use remains inside
the parent campaign. Stop before any parent allocation is exhausted.

The primary pass criterion is complete requested-row accounting, finite total
derivatives, valid physical proposal corrections, passed reference checks and
correctly separated method configurations in the report. Broken identities,
nonfinite results, reference failure, corrupt provenance and budget exhaustion
are continuation vetoes for dependent work. Candidate underperformance is a
promotion veto and a repair trigger, not a continuation veto. Score error and
timing are descriptive here; covariance fit, ESS and compilation time explain
results but cannot promote a method. No default, unbiasedness, scientific
superiority or full-master completion follows from this pilot.

Default audit: the model coefficients construct distinct situations rather than
represent a calibrated application. Their failure risk is insufficient stress;
report all three separately. LEDH controls come from the earlier mechanics
fixture and remain unpromoted hypotheses, held equal to isolate covariance
providers. SGQF level 2 and mixture within fraction .5 are named candidates,
not tuned choices. Numerical protections retain the previously documented
canonical controls; the new grid and local Gaussian proposal introduce no
clipping or covariance ridge. An invalid covariance fails closed. Before any
claim row, perform fresh scope-specific calibration and untouched evaluation.

Skeptical audit: Gaussian-only baselines would not answer the nonlinear question;
actual nonlinear EKF/UKF/bootstrap/local-proposal kernels are implemented and
checked against the affine limit and same-program finite differences. Merely
reporting a small grid difference could hide common domain truncation; the
separate domain and tail checks address that risk. The report must preserve
mixture/twist controls when grouping variants. Passing a smoke while losing to
cheap heuristics is a plausible result and must be the result note's headline.

Run from the persistent worktree with the `tftwogpu` Python, GPU deliberately
hidden for CPU runs, `BAYESFILTER_PRELOAD_CUSTOM_OP=0` and both TF thread counts
set to one. Execute each `phase-0g-nonlinear-*-study.json` using
`scripts/run_younis_score_master.py --action run`, then `--action compare` at
the unchanged source revision. Preserve exact command, source fingerprint,
runtime and row evidence in `nonlinear-*-cpu-01` under the active artifact root.
Repairs use new versioned directories. After this pilot, refresh scope tuning,
high-dimensional feasibility and the ratio/consistency work in the master.
