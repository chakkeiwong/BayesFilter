# Phase 17: diagnose score-fit failures and bounded remedies

Status: reviewed for execution, 2026-09-23. Owner authorized continued execution
and code tracing under the existing campaign allocation. This is a CPU-only,
independent R diagnostic reference, not LEDH, original-author iAPF, Equation 15,
a parameter-likelihood score, or a production backend.

## Question and evidence contract

Why did four d80 learners reject precision and four reach the iteration cap in
phase 16? Can an unchanged controller with a longer cap, and an explicitly
identified QR rescue of a rejected backward fit, resolve those failures without
changing healthy score runs?

Baseline: the frozen phase-16 `score_after_k`, same data, seeds, N0=1000,
Nmax=4000, k=5, tau=.5, six-value sample-CV window, kappa=.5, diagonal covariance,
tail8 floor, and `after_k` doubling. The baseline cap is 12; the diagnostic cap
is 18. Baseline source is commit 89065bc6354801cb368d5e163ba49fd9c4372d10.
Original evidence remains read-only at
`/home/chakwong/BayesFilter/docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01`.

Primary engineering criteria: reproduce all four precision rejections; verify
the first 12 histories of capped replays exactly; determine whether the same
stopping rule subsequently passes; retain rejection events in every rescued
result; and require bitwise identical histories, guides, and independent final
estimates when rescue does not fire. These are diagnostic/non-harm criteria,
not evidence that one stochastic filter is more accurate.

Numerical validity vetoes: nonfinite output, failed covariance/precision guard,
failed finite-difference checks, mismatched replay prefix, or changed healthy
output. Invalid source/data provenance or exceeded budget stops execution.
An individual failed fit or further cap rejects that candidate/case and triggers
interpretation, not abandonment of the research direction. If QR also rejects,
retain the failure; do not clip eigenvalues or invent an unreviewed fallback.

Explanatory diagnostics: floor probability, empirical score precision spectrum,
cloud spectrum, exact target curvature, same-target floor continuation, stopping
CV, particle count and per-case errors relative to Kalman. Removing/lowering the
floor is an explanatory intervention only. It is not promoted: the paper gives
the positive floor a tail-robustness role (GJL 2017, Section 5.1, Eqs 15–16).

Fresh probes use two untouched d80/T100 datasets (seeds 981080 and 991080), four
learner labels each (seeds 1972001–1972004 and 1982001–1982004). No choice is
selected by likelihood accuracy on these probes. Compare original score,
score-with-QR-rescue and standalone QR at cap18. Preserve fresh final APF calls.
Constructed cheap adversaries, on each dataset separately: BPF N10000 (unguided
proposal), fully adapted APF N5000 (exact one-step proposal), current-observation
guide N1000 (myopic information), and full backward oracle N1000 (exact Gaussian
control). Kalman supplies exact log likelihood. Record heuristic vetoes per
dataset and arm. With four samples per dataset, errors/tails are descriptive;
there is no supported accuracy ranking, unbiasedness test, reliability-rate
claim, paper replication, default readiness or HMC claim.

## Implementation and order

1. Reconstruct first bootstrap clouds for labels 291,300,694,722. Call the actual
   score backward routine with observing wrappers, saving the failing cloud,
   score and next guide. No alternate APF implementation is permitted.
2. Derive and test the score and Hessian of observation density times a
   Gaussian-plus-floor future integral. Check directional finite differences,
   the empirical negative precision, and the Gaussian-component limit on the
   same cloud. Plot/tabulate log-floor shifts 0,-2,-4,-8,-16,-32,-64,-128,-1000
   without treating them as accuracy tuning.
3. Replay labels 844,891,947,989 with cap18 and exact prefix checks. Replay the
   four rejected labels with an explicitly named `score_qr_rescue` callback:
   attempt the unchanged score backward routine; only on its rejection, run
   the existing complete QR backward routine on the same clouds and return its
   guide. Subsequent iterations again attempt score fitting. Record both the
   rejection and replacement objective. This hybrid is an extension, not a fix
   establishing correctness of unconstrained Gaussian score regression.
4. Execute the frozen fresh probes and simple controls. Healthy paired runs
   must be bitwise identical. Negative fresh evidence is recorded, not tuned
   away. Review the result, update the master and concise checkpoint.

## Defaults, risks and skeptical audit

| Choice | Provenance / status | Failure risk | Early check |
|---|---|---|---|
| Phase16 settings | frozen baseline, not universal defaults | inherited controller/floor can fail | exact seeded failure replay |
| Cap18 | diagnostic budget bound, six more iterations | may still be too short | full CV/N history; cap remains a failure |
| QR rescue | existing independently tested extension; conditional candidate | different fitting objective, can also reject | preserved rejection plus guarded QR and actual APF downstream |
| Unchanged positive tail8 floor | frozen baseline with paper tail-robustness rationale; exact exponent remains a hypothesis | nonconcave target, weak future information | curvature and floor-continuation curve |
| Zero precision damping / no clipping | exact unconstrained affine-score regression target | invalid Gaussian precision | fail-closed positive-definiteness guard; prompt rescue evaluation |
| Two datasets, four seeds each | bounded convenience probe, never accuracy certification | rare failures/tails easily missed | no rate/ranking claim; retain earlier 1000-label failures |
| R FP64, one thread/worker | authorized independent reference | no GPU/TF32 or nonlinear transfer | explicit manifest and nonclaims |

Skeptical audit PASS for this bounded question: baselines and target are explicit;
replayed failures are selected diagnostic cases, not a new success-rate sample;
the old failures cannot be replaced by successful retries; the rescue changes
the fitting objective openly; stopping thresholds and particle cap stay fixed;
healthy identity is tested at the consumer; and fresh conditional controls can
veto but cannot nominate settings. A finite-difference match verifies the local
derivation, not the paper's Equation 15 solver. The strongest alternative cause
is cloud-dependent regression geometry rather than local curvature alone; save
both exact curvature and empirical covariance/precision to distinguish them.

## Execution and budget

Workdir: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Branch: `ledh-refactor-with-policy-fix`; root surrogate-hmc is occupied.
New output: `docs/plans/artifacts/iapf-score-failure-remedies-20260923-01/`.
Driver: `python3 docs/benchmarks/diagnose_iapf_score_failures.py --execute`.
Worker: `/usr/bin/Rscript docs/benchmarks/diagnose_iapf_score_failures.R`.
All workers set CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.

Reserve at most 7200 single-worker wall seconds (2 CPU hours), two concurrent
workers, 300 seconds/attempt, at most 48 attempts including repairs; zero GPU.
Charge every launch by worker wall time, including failures. Before launching,
reserve each active worker's full cap. Starting campaign balance is
65708.04461661764 CPU seconds and 171882.52032958285 GPU seconds. No wall expiry.
Localized harness repairs within these unchanged bounds need no reapproval.
Each attempt has a fresh directory, command/environment/source hashes, Git
identity, data and RNG seeds, wall time, exit status, preserved RDS/CSV/logs.
Manifest, budget, checkpoint, mathematical note and terminal result live in the
new output root. No prior artifacts are overwritten.
