# Optional mixed precision integration and fresh validation

This continues the authorized shared surrogate-HMC repair. FP32/TF32 off remains
the default, N4032/K2016, SIR T50 and range-bearing T20. The ten-hour ledger has
14486.3173 GPU seconds remaining. This is the phase-entry budget; SIR is now running. This phase may use at most
9000 GPU seconds and eight attempts, including compilation and failures, leaving
the balance for the next planned cost/HMC mechanics repair. Every launch uses a
new directory and the existing campaign budget controller.

## Question and evidence contract

Does FP64 recursive state/sensitivity, local model arithmetic and recolouring,
with centered FP32 Gaussian/transport pair kernels, repair precision on fresh
data? Baseline: identical finite filter in uniform FP64, on identical data,
particle design, parameters, weights and controls. Preserve the previous uniform
FP32 failure. FP64 is an arithmetic reference, not an exact-model oracle.

Primary screen: every value/score component within 1% of max(1,abs(reference))
and healthy completed trajectories. Nonfinite results, failed factorizations,
incomplete steps, source/scope mismatch, or a mixed FP64 comparator veto it.
Runtime and importance variance are descriptive nomination diagnostics, not
evidence of superiority, posterior convergence or canonical LEDH admission.
Calibration and validation use fresh data. The saved campaign dataset has already
been used for diagnosis; a later frozen run on it is exploratory mechanics, not
an untouched generalization claim. Such a claim would require additional fresh
data after selection (for example seeds26101235/36).
Candidate rejection triggers the next planned repair; invalid comparisons or
budget exhaustion stop the affected experiment.

## Implementation, mathematical basis and executable checks

Add an optional precision policy to the shared batch factory; default uniform.
Pair boundaries call the existing kernels without copied formulas. Subtract a
moving origin a and its derivative da from points/tangents BEFORE casting.
Gaussian points/means share the origin. Transport adds a/da after returning to
FP64. Pair distances and Gaussian residuals are translation invariant, so these
changes preserve the real-arithmetic expression. Finite precision needs tests.
The analytical derivative is of these algebraic operations, not discontinuous
machine rounding.

Consumer/tuning scope binds the policy, recurrence dtype and pairwise dtype.
The tuner constructs a separate uniform FP64 reference, never comparing the
mixed implementation to itself. Calibration uses the same shared density
primitive and precision policy as the force. Fresh seeds: calibration
26101231/32, validation26101233/34; earlier26101211--14 are preserved failed-
candidate evidence. The existing candidate family (baseline, doubled steps,
half epsilon) is a warm-start hypothesis. Tune each model separately.

Before GPU launch: CPU/XLA parity against the isolated diagnostic, tiny-model
health, unchanged default-path regression, scope identity and pure-FP64 wiring
checks. Preserve intermediate comparisons/health before subsequent work can
time out. SIR first, range-bearing next within budget. A filter selection is not
a canonical HMC tuning artifact; later mechanics must use the supported public
arbitrary-force controller.

## Assumption audit and skeptical review

FP32, N4032, 1% are owner choices; accumulated roundoff is exposed by the paired
reference. Mixed precision is a candidate supported by saved SIR T10/T50, not
fresh-data evidence. Translation centering follows algebra; large common
offset/tangent tests expose incorrect cast order. Flow16/reset40/epsilon0.5
are inherited warm starts, tested by separate candidate families and health.
Equal and fitted defensive mixtures are constructed cheap adversaries in each
calibration/validation partition, reported separately. Exact Kalman and refined
KSC references remain model-specific authorities. For SIR/range, absent exact
oracles prohibit model-accuracy/default-readiness claims from precision alone.

Pre-mortem identified two material hazards: labeling mixed arithmetic FP32,
and accidentally propagating its policy into the pure FP64 reference. Explicit
scope and executable wiring checks address both. Fresh partitions avoid
selecting on the saved successful SIR case. Passing arithmetic checks may still
leave a poor posterior force; downstream HMC remains required. Plan passes
skeptical audit for optional integration/numerical screening only.

Artifacts: new surrogate-hmc-mixed-scope-* directories; this plan, focused test
logs and result with decision/inference tables. Existing master records command,
git/source, environment, seeds, device/memory policy, wall time and budget.
The next phase is part of the master: frozen-selection mechanics/cost checks
or another repair of failed precision. Candidate rejection alone is not a
continuation veto.

## Prelaunch audit result

Shared runtime versus isolated diagnostic and protected uniform baseline passed
for all five N64/T2 CPU/XLA fixtures. Initial suite:84 passed, one test API misuse
(force wrapper called without .function); corrected test plus calibration and
existing consumer checks:9 passed. Tests include exact factory policy/dtype
forwarding, distinct uniform FP64 comparator, fresh seed identity and rejection
of a mismatched selection. Full logs are alongside this plan. No scientific
source changes will be made while a worker runs.

An unrelated observation-TT GPU workload was visible before launch. It is not
owned by this campaign and is left running. GPU memory growth remains mandatory.
Any timing is descriptive under possible contention; no relative speed claim
or default promotion follows. The precision comparison remains paired on data
and parameters. One campaign worker runs at a time within the total wall budget.

Actual SIR command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py run --model sir_d18 --output docs/plans/artifacts/surrogate-hmc-mixed-scope-sir-20261012-01 --filter-tuning --runtime-precision fp64_recursion_fp32_pairs --tuning-seed-offset 20 --shared-gaussian-contraction direct --budget-seconds 6500 --per-model-timeout 6500 --campaign-ledger docs/plans/artifacts/surrogate-hmc-repair-continuation-20261012-01/budget.json

## Master phase-plan wiring repair

The v1 execution-plan schema binds only contraction and a time cap. Add v2 with
explicit precision, seed offset and candidate family per model; preserve v1.
Both phases must receive identical precision/seed settings, and mechanics must
consume an exact-scope selection. Unknown modes, missing fields, invalid seed
offsets and conflicting global overrides fail before launch. Include the mixed
runtime tests in the master regression suite.

Skeptical audit: orchestration only; no numerical candidate or failed-tuning
bypass. SIR numerical dependencies stay frozen. Preserve its original launcher.
Focused mocked phase/command tests must prove propagation and rejection before
use. This continues the authorized master repair without added GPU budget.

Phase-plan wiring regression:83 tests passed (CPU only,0.24s). Both v2 phases
receive the declared precision and fresh partition; invalid inputs fail before
launch. v1 remains supported. Log:master-scope-v2-regression.log.
