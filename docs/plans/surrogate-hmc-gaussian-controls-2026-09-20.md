# Anisotropic Gaussian surrogate-HMC control

The campaign tests surrogate HMC. This C01 scope asks whether deliberately
imperfect deterministic position-only fields can sample a known anisotropic
Gaussian when the shared kernel uses the correct per-chain endpoint energies.
It completes the controlled-force experiment already specified in master
section 10. The earlier toy had a chain-coupled acceptance error; its results
are invalid and supply neither a baseline nor tuning evidence.

## Target, arms and evidence contract

Use q in R^3, precision Q=diag(1,4,9), potential U(q)=q'Qq/2, and independent
four-chain Metropolis decisions. The normalized target is N(0,Q^-1). The
target and normalization are analytic; q_j*sqrt(Q_jj) is standard normal.
Use the actual shared neural_force_hmc proposal/transition, repository-issued
binding and public tune_hmc_kernel with VerifiedTensorFlowHMCConfig. An
ordinary stage helper alone cannot authorize retained sampling.

Freeze four fields: exact Qq; half-strength 0.5Qq; rotational Qq+0.2Aq with
A=[[0,-1,0],[1,0,-1],[0,1,0]]; and zero. Coefficients come from the master
mechanism design, not tuning. A is antisymmetric: the rotational field's
Jacobian Q+0.2A is nonsymmetric, so it is not the gradient of this or another
twice differentiable potential. It remains a deterministic position-only
field. Each kick and drift is a triangular shear with determinant one;
reversing the kick/drift sequence by momentum negation gives the inverse map.
Thus volume preservation and reversibility do not require force agreement
with the endpoint gradient. The Metropolis step still needs the exact
per-chain U and kinetic energy. Executable reversal, energy and consumer checks
must pass before dependent sampling.

Primary inference criterion: simultaneous equivalence intervals for 15
functions of w=q*sqrt(diag(Q)): three means (reference 0), three second moments
(reference 1), and the three marginal CDFs at each of Phi^-1(.1), 0 and
Phi^-1(.9). References are analytic; allowance zero records no quadrature error,
with ordinary floating-point error still possible. Margins are 0.1 for means
and second moments and 0.02 for CDF probabilities, the master pilot margins.
Apply the existing stationary-chain batch-means method with its unchanged
R-hat, ESS, batch-lag and MCSE sensitivity screens. Bonferroni allowance covers
four arm comparisons, 15 functions and two looks at 1000/2000 retained draws
per chain. Even an exact-force arm may remain inconclusive at this budget.
No extra looks, post-hoc margin changes or selection among successful arms.

Numerical endpoint/force failure, per-chain energy mismatch, loss of replay,
row coupling, wrong proposal reversal or corrupt archives invalidate the
affected branch and trigger repair. A tuning failure or unhealthy retained
chunk vetoes that arm. A discrepancy or unresolved equivalence is a candidate
result, not a veto on independently tuned arms. Acceptance and timing explain
the result only. Passing this control establishes no DCTR accuracy, nonlinear
model inference, statistical superiority or default readiness.

Reference expectations, rather than a finite reference sample, determine the
intervals. Archive 8192 independent exact draws under a separate seed as a
reference sanity diagnostic; do not substitute their empirical means for the
analytic expectations. Inspect lower/central/upper marginal regions through
the frozen CDF functions; starts span near/far regions in every coordinate.
The cheap adversary set is endpoint-only random-walk MH, coordinate slice
sampling and exact Gaussian Laplace approximation (constructed as the target's
quadratic Hessian inverse). Their efficiency and conditional comparisons remain
NOT_RUN in this bounded mechanism stage and veto any practical-usefulness or
efficiency promotion. The exact independent oracle remains a reference.

## Tuning, assumptions and budget

Each field receives its own public adaptation, fresh verification and artifact.
Initial physical scales are (1,1/2,1/3), derived from Q, with initial position
zero. Fresh standardized starts are (-2,-1,-1.5),(-1,2,.5),(1,-2,-.5),(2,1,1.5).
Nomination uses mass windows 64/128, 128 step adaptation, 128 candidate
verification, candidate leapfrog counts 1/2/4, initial step 0.2, target
acceptance 0.7, no repair round, mass shrinkage 0.1, jitter/floor 1e-9,
max condition number 1e8. Acceptance bands [0,1] make acceptance explanatory;
health and fresh R-hat admission still apply. Fresh warm-up 256 and verification
2000 are archived and excluded from posterior estimates. Retention grows in
250-draw chunks to the two frozen looks. Seeds are frozen per arm and phase
in the driver manifest before execution.

These short nomination settings come from the qualified public-tuner control,
as a warm-start hypothesis. The fresh verification can reject inadequate
adaptation. Scale and prior law are exact here; the target is an unbounded
Gaussian with no model-data or Jacobian terms. FP64 GPU/XLA is an explicit
mechanism-reference exception, consistent with the campaign integrator; it is
not default FP32 DCTR evidence. The covariance ridge is a tuner stability
setting: relative to the known minimum variance 1/9, 1e-9 is below 1e-8
relative scale. It changes proposal geometry only; exact endpoint correction
leaves the declared target unchanged. Report selected covariance and health.
The 1000 energy-error veto is the existing public qualification threshold,
a catastrophic-failure guard; it cannot certify mixing.

First run mechanics and cost probes with all four fields, using longest L=4,
compile plus two timed 256-transition chunks. Estimate each arm's maximum
nomination (192+3*256 transitions), fresh verification/warm-up (2256), and
retention (2000), times the larger observed per-transition wall/CPU cost,
plus measured compilation and a 90-second allowance for the larger tuner
graph, archive graphs and analysis, all multiplied by 1.5. This is a conservative
planning estimate, not a runtime bound. One first-batch cost overrun defers
inference rather than reducing evidence. The master watchdog enforces actual
wall and CPU caps. Time forecast is recorded before public tuning.

C01 initial allocation: 7200 GPU seconds / 1800 CPU-worker seconds, within P3.
Qualification cap 300 GPU / 400 CPU seconds; inference cap at most 900 GPU /
1000 CPU seconds, and focused tests at most 100 CPU seconds. At most one
qualified four-arm tuning/retention execution and one localized infrastructure
retry within these aggregate limits. Candidate failure is not retried on the
same retained seed. Preserve each run in a new versioned campaign directory.

Skeptical audit before implementation: the exact endpoint and analytic
reference answer the mechanism question; force quality is not a gate. The
full DCTR is intentionally absent in a control explicitly classified by the
master. Bounds do not promise all arms will mix or pass equivalence. Larger
dimensions and heuristic comparisons remain separate obligations. Extend the
equivalence utility only for explicit static parameter/function counts while
preserving LG1 defaults and formulas; regression-test the original results and
family multiplicity. Keep fresh tuning authority, source binding, per-chain
energies and warm-up exclusion through the actual public call chain.

Planned artifacts: scripts/qualify_surrogate_hmc_gaussian_controls.py and
scripts/run_surrogate_hmc_gaussian_controls.py, wired to P3-C01-qualify and
P3-C01-inference in the master. Save exact command, Git/source snapshot,
environment/device/memory policy, fields, references, seeds, checks, forecast,
tuning artifacts, every retained chunk/look, budget, decision and result note.

## Implementation audit and exact launch

The implementation uses the shared proposal, transition, public tuner, verified
artifact loader and bound retained archive runner. Focused CPU reference tests
passed 54/54 (GPUs deliberately hidden) in
`surrogate-hmc-all-models-20260920-c01-tests-02`; the first run found two stale
orchestration-test expectations, preserved in `c01-tests-01`. The new probe
uses an absolute 1e-12 FP64 mechanics tolerance on this fixed order-one control,
and bitwise equality only for replay and independent unaffected rows. Its
epsilon 0.1 is a short stable cost-probe hypothesis, not a selected HMC setting.
Actual posterior tuning still uses independent adaptation for each arm.

The all-model master now exposes the two C01 stages, keeps their comparisons
separate from DCTR/model comparisons, and reports existing retained evidence
without another attempt. Stage retry allowances subtract previously charged
C01 costs. The initial skeptical audit remains satisfied; no scientific
threshold, posterior margin, retained look or campaign budget changed.

Use the `tftwogpu` environment and trusted/escalated GPU access. GPU 1 (RTX 4080
SUPER) was idle at the preflight check. Set
`CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `BAYESFILTER_PRELOAD_CUSTOM_OP=0`,
`TF_NUM_INTRAOP_THREADS=2`, and `TF_NUM_INTEROP_THREADS=1` before import.

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --scope C01 --stage P3-C01-qualify --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-c01-qualify-01
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --scope C01 --stage P3-C01-inference --qualification-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-c01-qualify-01/P3-C01-qualify --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-c01-inference-01
```

The second command remains conditional on the first command's actual mechanics
checks and measured forecast. Master exit 2 means the overall all-model program
is still incomplete; inspect the job exit and structured decision separately.
