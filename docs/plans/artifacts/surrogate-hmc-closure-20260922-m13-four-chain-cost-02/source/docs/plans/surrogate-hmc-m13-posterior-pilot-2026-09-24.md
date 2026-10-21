# M13 posterior-reference pilot — 2026-09-24

The campaign goal is testing surrogate HMC: a frozen canonical analytical DCTR
proposal force with separately specified endpoint acceptance. This existing
master-program prerequisite locates the two-parameter posterior and tests the
independent likelihood integrator over that region. It launches no HMC chains.

## Evidence contract and intent

Use the unchanged M13 T20 calibration data seed1701, once rounded to FP32, and
the fixed initial law. The parameters are log standard-deviation multipliers
lambda_Q and lambda_R, with independent N(0,log(2)^2) priors on all R^2. The
independent full-trajectory defensive-importance reference evaluates the
normalized Gaussian radial and wrapped-Gaussian angular observation law.
Its FP64/GPU/XLA arithmetic is an explicit independent-reference exception;
the campaign's DCTR force remains FP32/TF32/XLA by owner directive.

The primary diagnostic is a finite, valid likelihood at every node of a 33x33
grid on [-4log(2),4log(2)]^2, followed by comparison of posterior integrals
using the nested 17x17 grid. Both use positive composite Simpson weights. Each node uses 4096 samples in each of eight
independent importance replications. Common random numbers across parameter
nodes smooth the integration comparison; seed[2026092402,9101] is fixed. Replications also give descriptive
Monte Carlo standard errors for the normalized integrals and their paired
grid differences. They do not measure omitted-tail or discretization error.

The comparator is the coarser integration of the same likelihood, not DCTR or
a Laplace substitution. Record means, standard deviations, covariance,
standard-deviation multipliers, evidence, and probability in the outer grid
region. An outer-region probability is a localization diagnostic, not a bound
on probability outside the grid. This grid does not truncate the declared
prior and cannot define an HMC acceptance endpoint.

Invalid likelihoods, failed proposal checks, nonfinite output, mismatched model
law, missing diagnostics or changed sources veto reference qualification.
Missing cells must never be silently assigned zero mass. Integration changes,
MCSE, ESS, maximum weights and outer-region mass are explanatory diagnostics.
A failed proposal is a repair trigger; it does not reject surrogate HMC or
stop independent planned repairs. Every output says posterior qualification
is false, including a fully successful pilot.

Constructed sanity checks: constant likelihood must recover the truncated
Gaussian prior; an analytically tilted Gaussian likelihood must recover its
known mean/variance inside a sufficiently wide diagnostic box; multiplying
likelihoods by a common constant must preserve normalized summaries; missing
or invalid cells must prevent a posterior summary. These are falsification
checks on integration, not tuning objectives. Conditional situations are
central, low-noise, high-noise and boundary grid regions. No heuristic
efficiency comparison or stochastic ranking is claimed by this prerequisite.

## Assumptions and skeptical audit

The integration grid is a pilot hypothesis: its spacing may miss a narrow
posterior, and its extent may omit material tails. Nested integration and
boundary diagnostics expose those risks but cannot certify their absence.
The 4096x8 reference size inherits the successful broad-scale diagnostic only
as a warm start. Per-node proposal checks and weight diagnostics expose transfer
failure. The 1/16 defensive prior mixture and 1.2 proposal-covariance expansion
are unchanged reference choices; their exact normalized mixture density remains
in every importance weight. Poor efficiency triggers proposal repair or more
replications, never replacement by the fitted Gaussian likelihood.

The audit passes for localization only. This experiment would be misleading
if a finite-box posterior, eight-replication MCSE, or all-valid grid were
reported as the continuous model posterior. The implementation explicitly
prevents that promotion. Calibration data are reused deliberately; the
untouched claim seed3701 and DCTR control-validation partitions are not read.

## Execution and budget

Use the existing reference-extension pool, which has554.930702 CPU-worker
and581.189720343 GPU-wall seconds remaining. Reserve180CPU for focused tests
and360CPU/480GPU for this worker; unchanged global campaign ceilings apply.
Each evaluation is checkpointed; failed attempts and timeouts consume budget.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-posterior-pilot --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-03

The master preserves source hashes, exact command, environment, seeds, GPU
memory-growth policy, wall/CPU charge, per-node logs, numerical summaries,
and result/report manifests in that versioned directory. Inspect the result
before declaring the next reference refinement or endpoint design. This is
part of the existing master; the campaign is not complete after this pilot.
