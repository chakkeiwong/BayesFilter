# Author/source-port matched-input result — 1 October 2026

The TensorFlow weight calculation agrees with the original source on all22
saved stages. Literal regression-coefficient reproduction fails for both the
raw normal-equation TF path and centered QR. The first mismatch in the first
case occurs in refinement1 at zero-based time99, on identical captured X and
Value. A backward-recursion difference is not needed to produce this first
difference: it already appears at the single-regression endpoint.

The source capture is valid: attempt88 reproduces all fields of all six
attempt86 source returns exactly and captures1616 regressions across16 refits.
Original source/data/dependency hashes agree. Instrumentation consists solely
of recording existing variables after the author regression. Preserve the
minimal diff and source checksums in88. The original source checkout is not
edited.

## Execution and measured differences

Attempt90 completes354 harness checks and306 comparisons in55.983336118 GPU
wall seconds. There are82 passing comparisons (66 weight/logZ/ESS and16 design
ordering) and224 coefficient/prediction mismatches. Diagnostic execution PASS
is separate from scientific verdict PORT_MISMATCH_REQUIRES_LOCALIZATION_REVIEW.

| Compared quantity | Result | Largest absolute difference |
|---|---|---:|
| Incremental weights,22 source stages | All pass existing tolerance | 9.750e-10 |
| Cumulative log normalizer,22 stages | All pass | 1.637e-9 |
| ESS,22 stages | All pass | 3.382e-8 |
| Polynomial design edge rows,16 refits | Exact agreement | 0 |
| Raw TF local A,b,c/predictions,64 comparisons | All mismatch source somewhere in time | See saved per-time comparisons |
| Centered QR local A,b,c/predictions,64 comparisons | All mismatch source somewhere in time | See saved per-time comparisons |
| Full backward coefficients,48 comparisons per solver | Both mismatch source | See saved tensors |

Both solvers remain finite and pass their current validity checks on these
source clouds; these checks do not establish source parity. All11 used compiled
endpoints traced once. Numerical tensors ran on GPU0 (RTX4080 SUPER), FP64,
TF32 disabled, XLA enabled, with verified growth on both physical GPUs. Peak
TensorFlow allocator usage was73,884,160 bytes. Binary decoding and tensor/file
serialization are explicitly CPU operations outside the numerical functions.

For F2.5, seed20260929, refinement1, the sum of squared residuals against the
same captured targets is16.085380884 for source coefficients,179426.812909 for
raw TF normal equations, and0.193181359 for centered QR. The minimum raw TF Gram
eigenvalue ratio is7.197e-17 versus0.01032 after standardizing the design.
These are deterministic diagnostics, not stochastic performance rankings.
Centered QR solves the same stated least-squares objective but does not reproduce
the author's floating-point coefficients.

Ill-conditioning is supported as an explanation for sensitivity. We have not
yet separated different rounding in Gram/RHS formation from different
factorizations or solving behavior. Therefore it is premature to say the exact
backend operation responsible is known, that QR is the right empirical repair,
or that these source-cloud differences cause attempt76's indefinite precision.
Those are different questions. No source-run failure or paper invalidity follows.

## Decision and terminal review

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept capture88 for reference diagnostics | Exact equality of all original returns | No source/capture veto | Octave is not MATLAB | Reuse captured inputs | Full Figure6 replication |
| Accept90 as completed localization | All declared shared endpoints and comparisons saved | No harness/device/trace veto | Successful source trajectories only | Compare shared Gram/RHS across solvers at the first mismatch | Failing TF trajectory explained |
| Keep literal TF-source regression parity open |224 existing-tolerance screens fail | Source-parity promotion veto | Matrix formation versus solve behavior | Preserve source baseline; no tolerance relaxation | That either solver is admitted |
| Keep full empirical launches blocked by existing dependencies | Independent repair/full-consumer validation incomplete | Existing precision/batch vetoes remain | A numerically faithful, valid full consumer | Specify and validate the localized repair before batching/cost | Whole-master completion |

| Inference status | Assessment |
|---|---|
| Hard veto screen | No invalid numerical output here; literal coefficient parity fails |
| Statistically supported ranking | None; this is deterministic implementation localization |
| Descriptive differences | Residuals and conditioning above; no efficacy ranking |
| Default readiness | Unsupported |
| Next evidence needed | Crossed identical-Gram/RHS comparison, then common-forward-input validation and the existing failed-trajectory repair |

Skeptical terminal self-review: the test holds regression inputs fixed and
checks real shared endpoint calls. Weight parity and single tracing rule out
those particular mismatches in this scope. The strongest alternative explanation
is backend arithmetic amplification by an ill-conditioned Gram matrix, rather
than an incorrect regression formula. A common-matrix cross-solve could
distinguish this. The weakest evidence is transfer to the failed TF trajectory:
it has not been tested here. Independent review was not performed.

The readiness request is complete for these two implemented diagnostic phases.
They have run to completion. The master is not an unattended fix-all program:
the exact next numerical repair still depends on resolving the matrix/solver
difference and must keep its empirical claim separate from original-source
reproduction. No new precision projection or scientific default was installed.

## Provenance and resources

Plan: controlled-smc-lorenz-source-port-localization-20261001.md.
Artifacts: artifacts/controlled-smc-kdm-replication-20260928/attempt88_c1_lorenz_author_capture
and attempt90_c2_lorenz_source_port, each with frozen sources, commands, hashes,
inputs, logs and manifests. The latter also preserves all computed local/full
fit diagnostics as tensors and all per-time comparison errors in summary.json.

Attempt87: source equality passes all six cases; MAT decoding fails,10.539320001
CPU seconds. The dictionary/mat_struct boundary repair passed a focused saved-file
check. Attempt88 passes,11.191161323 CPU seconds.
Attempt89: DecodeRaw under strict GPU placement fails before numerical comparison,
5.729742912 GPU seconds. Explicit CPU I/O boundaries repair this harness issue.
Attempt90 passes diagnostic execution,55.983336118 GPU seconds.

Total additional charge:21.730481324 CPU and61.713079030 GPU seconds. No compute
allocation increased. There is no running campaign job. Attempt ceiling90 is
consumed; a new experiment specification and bounded reassessment are required
for another campaign launch, without a new human approval under the unchanged
authorized scientific scope and total budget.

Remaining inherited CPU seconds: 383.782092395; GPU seconds: 116491.722039794 (32.35881 hours).
Remaining current-tranche CPU/GPU seconds: 130.615156774 / 3540.421689074.
