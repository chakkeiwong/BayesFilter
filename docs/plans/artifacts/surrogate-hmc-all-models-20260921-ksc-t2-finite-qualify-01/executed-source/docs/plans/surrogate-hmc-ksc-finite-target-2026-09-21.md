# KSC T2 independent finite-target comparison

## Research intent and evidence contract

The next question is whether the full canonical DCTR proposal field, corrected
by a frozen DCTR endpoint, samples that endpoint's parameter posterior. This is
C/D. D/R separately measures the finite target's approximation to the exact
KSC-mixture model posterior. A D/R discrepancy does not cancel C/D. The completed
A/R and B/R attempt remains closed, with both comparisons inconclusive.

Keep the existing nominal T2 claim data, independent Uniform(0.1,0.9) parameter
priors, probit chart, N=1024, K=1024, fixed claim particle bank, and independently
selected protection multiplier 0.1. All filtering and analytical recursion must
call the shared canonical batch implementation. No force replacement, altered
data, new control selection, or retrospective posterior look is authorized by
this continuation. The existing user instruction to execute and continue covers
this independent arm within the unchanged campaign budget.

D is a named finite density. For physical theta in the open prior rectangle,
round theta to FP32 and clamp only coordinate-rounding excursions to the nearest
FP32 interior boundary. Evaluate the full canonical DCTR physical value, including
its fixed uniform prior. For q in R², theta = 0.1 + 0.8 Phi(q) and add the FP64
Jacobian sum(log(0.8) - q²/2 - log(2 pi)/2). This defines the endpoint even when
floating point Phi rounds to a boundary. It is not the old fully FP32 chart
program. Endpoint status requires valid values; a separate force-status failure
must remain visible and must not relabel a valid endpoint as an exact score.

Force and endpoint bindings are independently immutable. The force remains
the already qualified FP32 position-only DCTR field with FP64 integration.
Their input banks may coincide, but detailed balance does not require that.

Endpoint qualification requires physical/chart agreement, deterministic replay,
direct shared-callable parity, correct boundary/nonfinite status, unchanged force
binding, endpoint-cache independence, and a healthy reversible four-chain
proposal at a diagnostic step. These are engineering prerequisites, not evidence
of posterior agreement. Invalid values, corrupted provenance or a shared-kernel
call-chain mismatch veto continuation until repaired. A failure of a candidate
or a reference precision screen triggers a bounded diagnosis; it does not reject
the research direction.

An independent parameter quadrature will integrate D over the full proper prior.
Use nested 16/32/64/128 interval grids, extending to 256 only in a prospectively
budgeted precision repair. Reuse checked exact-model predictive values at the
matching knots of the existing 256-interval R reference. Freeze R's means,
scales, marginal cut points and 13 function definitions for all comparisons.
Predictive functionals mean exact-model conditional prediction averaged over
the named parameter posterior; this does not claim a DCTR filtering predictive.

For each functional, require both twice the maximum of the last two refinement
changes <= one tenth of its equivalence margin and final change <= half the
preceding change + 1e-5. These are empirical numerical allowances, not rigorous
global error bounds. D/R uses the sum of D and R allowances and the existing
margins. A D/R discrepancy is an approximation finding, not a sampler veto.

C requires its own public typed position-field tuning and verification against
this new endpoint. No A/B tuning artifact transfers. A later inference launch
must reserve all tuning, four chains, 250 discarded transitions, 1,000 fresh
verification transitions and the complete frozen retained schedule of 1,000 and
2,000 draws per chain. C/D and C/R use the predeclared simultaneous uncertainty
and health checks. No new C posterior look begins until its complete forecast
fits the remaining allocation. The heuristic comparisons and cross-model
confirmation remain separate requirements for usefulness and default claims.

## Assumptions and skeptical audit

| Choice | Provenance and status | Failure mode | Earliest discriminating check |
|---|---|---|---|
| Full DCTR, selected 0.1 controls and fixed bank | Existing KSC scope selection; frozen candidate | A different endpoint chart changes the computed target | Explicit physical/chart identity and independent D reference |
| FP64 probit chart with FP32 interior rounding | Named finite-target definition, not an exact-score claim | Silent support exclusion or unrecorded clipping | Boundary and nonfinite probes; rounding telemetry |
| Nested full-prior quadrature | Existing independent two-parameter reference machinery; accuracy hypothesis | Discontinuous finite density or insufficient resolution | Per-functional contraction/precision, raw grids retained |
| 128 intervals initially | Cost hypothesis informed by R refinement, not transferred accuracy | D needs more resolution than R | D's own refinement screen; no admission on R's accuracy |
| Batched evaluation | GPU/XLA default; no scalar fallback | Memory growth or compilation dominates | Small batch timing and allocator telemetry before full grid |
| Existing margins and two-look schedule | Frozen master protocol | Short chains fail to qualify MCSE or narrow intervals | Record inconclusive without relaxing the screen |

The audit finds no wrong comparator: C is compared first with its own independently
integrated D. Numerical parity and short transitions cannot promote inference.
The primary uncertainty is finite-reference cost and precision, so endpoint and
cost qualification precedes quadrature and any C tuning. The expected failure
of interest is C mixing poorly despite valid endpoint correction; a reference
failure instead invalidates that comparison and must not be called sampler
failure. A run could appear successful while using the exact-model endpoint
accidentally; direct endpoint call-chain parity and distinct identities guard
that error. D/R never substitutes for C/D.

## Bounded execution

Campaign ceilings remain 54,000 CPU-worker and 226,800 GPU-wall seconds. KSC's
current cumulative ceilings remain 4,000 and 7,200. At this plan's start,
KSC has used 1,744.545861 / 1,619.123895 seconds. No allocation is enlarged here.
Endpoint qualification/cost work is limited to 180 CPU / 240 GPU seconds in total,
including retries. A full-grid forecast must fit a provisional 750 CPU / 1,200
GPU reservation and leave room for the measured complete C inference forecast;
otherwise record under-budgeted and prospectively review internal allocation.
Tests and reporting consume the same cumulative family budget.

Use the tftwogpu Python environment, designated GPU1
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, escalated access, verified memory growth,
TF32, stable-signature XLA, four CPU-core affinity and the existing live CPU/wall
watchdog. CPU-only unit checks explicitly hide the GPU. No new dependencies.

Commands and source snapshots will be recorded in each run manifest. Output roots
are fresh directories under docs/plans/artifacts named
surrogate-hmc-all-models-20260921-ksc-t2-finite-qualify-NN and
surrogate-hmc-all-models-20260921-ksc-t2-finite-reference-NN. Preserve failures,
account actual resources, and stop before a stage whose full reservation cannot
fit. A later retained C command and forecast must be written here before launch.

Result notes must distinguish engineering validity, reference precision, sampler
validity and statistical interpretation. A passing endpoint check establishes
only the first; one small KSC scope cannot establish all-model recovery.

## Endpoint launch record

The independent coordinate/status tests and affected master regressions passed
53 tests (finite-tests-01). The public master dry-run validates the unchanged
qualification, control selection, exact-model reference and cumulative budgets.
The launch reserves 180 CPU-worker / 240 GPU-wall seconds. The audit passes for
this bounded preparation only; no inference or reference-precision claim follows.

```bash
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b \
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py \
  --stage P3-KSC-finite-qualify --scope KSC \
  --qualification-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-qualify-01 \
  --selection docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-controls-01/selection.json \
  --posterior-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-posterior-01 \
  --output docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-finite-qualify-01
```
