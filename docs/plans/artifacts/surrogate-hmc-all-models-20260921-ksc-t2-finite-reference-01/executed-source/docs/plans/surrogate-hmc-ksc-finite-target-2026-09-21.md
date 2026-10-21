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

Campaign ceilings remain 54,000 CPU-worker and 226,800 GPU-wall seconds. At this
plan's start KSC's cumulative ceilings were 4,000 and 7,200, with
1,744.545861 / 1,619.123895 seconds used. The prospective redistribution and
replacement reference reservation below supersede these initial allocations.
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

## Measured endpoint result and prospective reference reservation

Finite-qualify-01 passed all 27 checks, with reversal error 5.55e-17 and peak
TensorFlow allocator use 16,900,096 bytes. Actual cost was 118.217901 CPU-worker
/ 122.684598 GPU-wall seconds. This qualifies endpoint mechanics only.
The measured complete forecasts are reference 911.682672 CPU / 1,001.345051 GPU
and C tuning plus sampling 2,983.730842 CPU / 3,139.577287 GPU seconds. C uses
the maximum two-step trajectory, both looks, all tuning and verification, a
1.5 safety factor, and 330 seconds of compilation/analysis allowance inside
that factor. The reference forecast uses the finest nested grid's 1,041 batches,
twice the slowest measured batch and additional compilation/host allowance.

The original provisional 750 CPU reference reservation is insufficient and is
not used. Before reference execution, redistribute 2,500 CPU seconds from unused
P5 replication to P3 within the authorized campaign total. KSC's ceiling becomes
6,500 CPU / 7,200 GPU seconds; P3 becomes 15,500 CPU and P5 becomes 6,100 CPU.
The other currently ready families already received their initial feasibility
turns. This funds the independent C/D question while retaining explicit limits
on later replication. It creates no extra B attempt or retrospective look.

Reserve 1,000 CPU / 1,100 GPU seconds for the reference, including retries.
Its launcher also checks that this reservation plus the complete C forecast
fits the remaining family and campaign balances. Use one-sided endpoint limits
for trapezoid boundary knots, retaining the declared interior-rounding rule.
Reuse the finest grid's nested subsets rather than reevaluating identical
points. Persist full log density, all grids and boundary/force-validity counts.
Invalid values anywhere on the full-prior grid veto this reference. Force-only
invalidity is reported separately because D is defined by the endpoint value.

The skeptical audit still passes: the reference is independent of chains, uses
the same frozen D density at every knot, and cannot be promoted from mechanics
alone. Its empirical precision may fail even after a successful run; such a
failure requires a prospective refinement/cost decision before C launches.

## Reference launch record

Finite-tests-02 passed 58 tests, including exact integrals of uniform and tilted
densities, endpoint boundaries and master dispatch. Review then caught a loader
count typo: qualification has 27 required checks, not 28. The loader now checks
their exact names, including the canonical call-chain check; a substituted check
cannot pass. Finite-tests-03 passed 35 affected tests. Costs were 10.373699 and
8.785164 CPU seconds, with GPUs deliberately hidden. The numerical endpoint is
unchanged, so the prior qualification remains applicable.

The public reference dry-run passed with 1,000 CPU / 1,100 GPU reserved and the
complete C forecast also fitting the remaining balances. Trusted nvidia-smi
showed the assigned GPU idle. The audit passes: analytic-integral tests check
the quadrature independently; this run evaluates the full declared prior and
archives nested refinements, with no sampler inference claim. The reference's
empirical precision screen remains decisive for whether C/D is interpretable.

```sh
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b \
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py \
  --stage P3-KSC-finite-reference --scope KSC \
  --qualification-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-qualify-01 \
  --selection docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-controls-01/selection.json \
  --posterior-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-posterior-01 \
  --endpoint-root docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-finite-qualify-01 \
  --output docs/plans/artifacts/surrogate-hmc-all-models-20260921-ksc-t2-finite-reference-01
```
