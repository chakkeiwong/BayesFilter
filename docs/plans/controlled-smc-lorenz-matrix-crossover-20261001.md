# Lorenz regression matrix crossover — 1 October 2026

## Question and evidence contract

The controlled-SMC master remains active. Attempt 90 located a discrepancy in
single regressions on the author's actual inputs, before backward-recursion
differences are needed. Determine whether forming the normal equations, solving
them, or both explains this discrepancy. This is deterministic localization,
not a new empirical comparison or admission of a numerical default.

Baseline: the pinned author's `Lorenz/cSMC_Resample.m:187–215`, using the exact
captured X, Value and BetaHat from attempt 88. The paper's Section 4.3 and
Section 6.2, equations 42–44, define the regression and quadratic policy. The
author code forms D with `x2fx(X,'quadratic')` and computes `(D'*D)\(D'*Value)`.
Preserve this code and its actual coefficients; a lower residual is not itself
source reproduction. No covariance projection or precision-policy change is
authorized by this diagnostic.

Use times 100 and 99 (zero based) from every one of the 16 captured refinements:
32 regressions across all six native cases. Time 100 is the terminal control;
time 99 contains the first observed difference in the first case. This fixed
selection includes every refinement, rather than selecting favorable examples.

1. CPU reference: reconstruct the author's full design, Gram matrix and right
   hand side in the pinned Octave/statistics environment. Require exact replay
   of captured BetaHat before accepting this as the matrix baseline.
2. GPU/XLA: call the unchanged shared `regress` endpoints for raw normal
   equations and centered QR. Form matrices with the shared design routine.
   Cross the original/TF Gram matrices and original/TF right hand sides in all
   four combinations, solving each with `tf.linalg.solve`. Compare the
   separately formed solve with the actual compiled regression endpoint; XLA
   arithmetic must not be assumed unchanged by diagnostic extraction.
3. CPU reference: solve the four identical matrix pairs with Octave backslash,
   explicit LU and, when factorization succeeds, explicit Cholesky. Preserve
   the source direct least-squares and standardized QR reference as explanatory
   checks of the same least-squares objective. Do not infer the exact Octave
   internal branch merely from similar results.

The primary diagnostic completion criterion is complete saved matrices,
coefficients and comparisons for all 32 regressions and all four crossed
inputs, with exact native replay and checked shared-endpoint wiring.
Existing `1e-7 + 1e-9*abs(reference)` screens describe coefficient/prediction
agreement; do not relax them or interpret them as a well-conditioning theorem.
Full-design equality, matrix rounding differences, residuals, singular values,
condition estimates and solver agreement are explanatory diagnostics. Finite
status is recorded separately for every solver. A failed optional factorization
rejects that solver on those inputs; it is not a continuation veto. Missing or
corrupted inputs, failed exact source replay, wrong endpoint, missing diagnostics,
unstable tracing, timeout or exhausted budget veto continuation of this stage.

The research-direction repair trigger is a localized matrix/solve discrepancy.
After the crossover, specify the smallest justified repair or source-reproduction
route and validate it on complete and previously failed trajectories. Do not
stop the research merely because a solver differs. If extraction changes the
compiled result, inspect that difference before attributing the original failure
to a separately compiled diagnostic. No conclusion about Figure 6, KDM training,
MATLAB equivalence, solver efficacy, or the cause of attempt 76 follows by itself.

## Defaults, skeptical audit and resources

| Choice | Provenance and justification | Failure mode and early diagnostic | Status |
|---|---|---|---|
| All 16 refits, final two times | Frozen attempt 88; includes first mismatch and terminal controls | Transfer to earlier times can fail; no such claim | Diagnostic scope |
| FP64, TF32 off, GPU/XLA | Existing port/reference comparison contract | GPU/compiler arithmetic differs; record actual device and cross matrices | Reference comparison |
| Raw normal equations | Exact author operation | Squared condition number; save singular values and residuals | Source baseline, not promoted default |
| Centered QR | Existing optional shared implementation | Different floating-point coefficients; check actual objective, never call it literal replay | Repair hypothesis |
| Zero added damping/projection | Needed to isolate existing arithmetic without changing the target | Indefinite precision remains possible; preserve existing guards and full-consumer vetoes | Diagnostic-only baseline |
| Existing coefficient tolerance | Attempt 90, retained without tuning | Ill-conditioned coefficients can disagree despite a common objective; report residuals separately | Localization screen |

Skeptical pre-run audit passes: the actual author output is the comparator;
both matrix formation and solve are crossed; a traced diagnostic is tied back
to the real consumer endpoint; source and TF inputs stay identical. No residual,
smoke result or internal consistency check is a promotion criterion. The main
pre-mortem is an apparently explanatory standalone solve whose arithmetic differs
from the real XLA endpoint; the explicit endpoint comparison exposes it. These
are deterministic operator checks, so stochastic ranking and heuristic filtering
efficacy comparisons are inapplicable here. Independent review is not performed.

Allocate at most 90 CPU seconds (45 per CPU phase) and 600 GPU seconds, with two
attempts per phase including localized harness retries: six attempts total,
ceiling 90 → 96. These are within the unchanged current allocation of 500 CPU /
18000 GPU seconds and inherited grant; no additional compute is granted. At
opening, current-allocation remaining CPU/GPU seconds are 130.615 / 3540.422;
inherited remaining GPU hours are 32.35881. Record each attempt and its charge.

Use unique attempt roots under
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/` with source/input
hashes, command, git commit, environment, device, seeds, wall time and plan/result
paths. CPU references intentionally hide GPUs. GPU numerical kernels use stable
input signatures, XLA, verified memory growth and escalated device access.
Binary I/O and reporting stay outside the kernels. NumPy/SciPy are confined to
the explicitly independent CPU reference; no production code is changed.

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_matrix_reference
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_lorenz_matrix_crossover
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_matrix_crosssolve
```

Result: `controlled-smc-lorenz-matrix-crossover-result-20261001.md`.
Existing failed precision/batching dependencies remain active; a diagnostic PASS
cannot unblock a full empirical launch.

## Follow-up: identify Octave's automatic solver branch

Attempts 91–93 show that identical matrices produce different answers with
backslash and explicit LU/Cholesky. Exact Octave 6.4.0 dMatrix.cc lines
1448–1461 and 1530–1567 reject a reciprocal condition estimate when
rcon + 1 == 1; lines 1625–1628 invoke lssolve, which calls DGELSD with its
default rank threshold. Source: .localresources/code/octave-6.4.0-dMatrix.cc.
The official 6.4 solver-selection manual and installed help matrix_type
document the minimum-norm fallback and explicit singular selector.

Question: does this fallback explain ordinary backslash on the 128 frozen
matrix/RHS pairs? Compare ordinary backslash against the forced singular
minimum-norm branch. Preserve warnings, cached matrix types, spectra and
coefficients. Exact equality identifies branch behavior on these inputs; the
existing coefficient screen is secondary. Mismatch triggers further branch
attribution, not a runtime change. Baseline is the same Octave executable and
frozen inputs; exact replay of captured source coefficients remains required.

No threshold is tuned or protection introduced into the runtime. This cannot
establish MATLAB equivalence, solver efficacy, or full adaptive validity.
Forced singular is a diagnostic hypothesis, not a proposed default. Missing
inputs, changed hashes, nonfinite mandatory coefficients or failed baseline
replay invalidate the run. Cached types and spectra explain, not promote.
Heuristic efficacy comparisons do not apply to branch identity; the full
filtering comparison retains its previously declared baselines.

Skeptical audit: Cholesky success does not imply Octave used that answer.
Its conditioning test is a concrete competing explanation. Forced selection
does not perturb G or r. Preserve both answers and do not infer MATLAB
behavior. The failed full-trajectory precision veto remains. Audit passes.

Allocate at most 20 CPU seconds total and two attempts within the unchanged
90 CPU-second matrix-stage allowance, tranche and attempt ceiling 96. GPUs
are intentionally hidden. Snapshot the Octave C++ source with the wrapper.

Command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_matrix_fallback

## Follow-up: original backward fit on the failed TensorFlow cloud

Attempt 94 identifies 15/32 native matrices taking Octave's minimum-norm
fallback, with exact forced-branch equality; the other 17 are cached as positive
definite. Across four crossed arms this is 60/128 exact fallback matches.
Backend-dependent numerical semantics do not by themselves establish that the
full source recursion repairs the previously failed cloud.

Replay the author's ApproxDP body from cSMC_Resample.m lines 142–239, extracted
verbatim into a callable wrapper, on the three recorded refits of row 0 from
attempt 76. The third refit failed at zero-based time 12 with minimum precision
eigenvalue about -1730. First require exact returned-policy equality on all 16
healthy author-native refits from attempt 86. Only observability and a stop
before an invalid Gaussian inverse/log-determinant may be added. Assert that
stripping these additions restores the extracted body exactly. No solver or
coefficient changes. Record pinned x2fx and source hashes.

Primary diagnostic: does the same author fitting code produce a valid policy
on the frozen failed TensorFlow cloud and incoming policy? Reconstruct the
author's Gaussian shift/covariance summaries from those inputs with original
RungeKutta4. This is a fixed-cloud replay, not end-to-end native randomness.
Compare with saved TF QR policies and independent SVD result 77. Exact healthy
replay is a harness validity veto. Negative precision on the failed-cloud
replay is a candidate veto and repair trigger, not an experiment failure.
No efficacy ranking, paper refutation, default change or Figure 6 completion
follows from either outcome. Full filter comparisons retain their existing
heuristic and APF baselines; this diagnostic tests validity, not efficacy.

Default audit: source backslash is the source-runtime hypothesis; old TF
policies are fixed input conditions, not presumed source policies. Checking
precision uses the existing 64-epsilon relative margin as a fail-closed
diagnostic and never alters accepted coefficients. Cheap healthy replay tests
whether extraction or instrumentation changes the target. Paper section 6.2,
equations 43–44, conditions feasibility on positive precision; the source's
full-quadratic branch does not enforce it.

Skeptical audit passes for attribution only. This tests the neglected input
regime with unchanged source code and exact healthy controls. It does not
conflate source success on other clouds with safety on this cloud, or residual
fit with filter validity. Stop for stale input, failed healthy replay, missing
diagnostics, timeout or source drift. Candidate failure triggers a constrained
fit repair plan; it does not invalidate the research direction.

Budget: 45 CPU seconds total, two attempts, within the remaining matrix-stage
90-second allocation and attempt ceiling 96. No new GPU or compute grant.

Command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_failed_source_replay

## Next repair phases after terminal attribution

Stages 91–95 are complete. Source/native/backend localization is closed for
the tested inputs; do not restart it merely because the full repair is open.

1. Specify the full-covariance constraint and numerical representation. Start
   from the paper's section 6.2 feasibility condition and footnote 4. Inspect
   its cited Higham method before calling any new implementation source-faithful.
   The existing square-root-epsilon projection failed; it is a rejected
   comparator, not a default. Explicitly distinguish a covariance projection,
   a precision projection, and a constrained least-squares objective.
2. Calibrate the proposed floor/conditioning parameter on the fixed first
   failed update, using identical fitted coefficients before allowing separate
   QR/SVD fits. Predeclare a scale-aware conditioning curve, finite Gaussian
   integrals, matrix agreement and healthy no-fire equality. Do not select
   using likelihood variance, ESS, or this known failed seed's performance.
3. Validate the entire backward recursion and forward consumer independently
   on healthy and failed trajectories. Any rank, integrability, nonfinite,
   coefficient or reference failure rejects the candidate and identifies the
   next repair; an apparent finite run cannot close the phase.
4. Freeze a viable candidate before untouched seeds. Apply the existing
   conditional heuristic/APF comparisons and uncertainty criteria. Recheck
   shared scalar/batch call-chain parity, memory, XLA tracing and full-matrix
   cost before a Figure 6 launch within the remaining grant.

These repair phases need concrete implementation and a bounded execution
specification before launch. The current executable phases perform diagnosis;
none automatically supplies an independently validated SPD repair. Keep the
RBF/Table 2 and KDM fitting/training gaps distinct in the authoritative master.
