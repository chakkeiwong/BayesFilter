# GenUT Dual-Cap Beta-Release Master Program

Date: 2026-08-08

Status: `PLANNED_TEST_FIRST_REPAIR`

## Objective

Turn the current scalar research implementation of dual-cap GenUT into one
coherent, testable TensorFlow/TFP algorithm that can be offered for bounded
beta testing across scalar value/score evaluation and batch-native NeuTra
targets.

The selected algorithm family is:

```text
diagonal third/fourth-moment correction
+ pairwise co-skewness/co-kurtosis correction
+ rowwise radial RMS cap
+ standardized coordinate cap
+ affine mean/covariance restoration
```

The initial selected family constants are four pairwise steps, radial cap
`2.0`, coordinate cap `0.98`, and coordinate-cap power `8`. Pairwise strength,
transport controls, ridge, and other numerical controls remain scope-specific
and require target-specific tuning.

This program does not claim an exact nonlinear likelihood, an unbiased score,
Zhao-Cui source-faithfulness, posterior correctness, or HMC readiness. The
pairwise and cap operations are BayesFilter extensions.

## Current Verdict

The current test suite is useful but insufficient for refactoring or beta
release.

Existing strengths:

- independent forward-accumulator checks cover the scalar diagonal, pairwise,
  radial-cap, coordinate-cap, projected-cumulant, and affine-restoration JVPs;
- scalar finite-value/score recursion has a same-program finite-difference
  fixture;
- diagonal batch evaluation has scalar-row and value-only/value-score parity
  tests; and
- focused scalar/filter/batch tests currently pass.

Release-critical gaps:

- batch-native evaluation does not implement pairwise correction or either
  cap;
- `GenUTControls` and NeuTra target signatures cannot represent dual-cap;
- no public algorithm selector maps `default` to `dual_cap`;
- no scalar/batch dual-cap parity test exists;
- no full finite-program dual-cap score parity test exists;
- no extreme-float32 or zero-correction-plus-cap semantic test exists;
- no test binds route identity and tuning identity to the selected algorithm;
- no GPU/XLA runtime or memory regression baseline exists for dual-cap; and
- old NeuTra admission artifacts describe a diagonal-only target.

Therefore tests that characterize current behavior and specify intended
dual-cap behavior must precede structural refactoring.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can the selected dual-cap family become one scalar/batch-consistent, numerically stable, scope-tuned TensorFlow/XLA beta route? |
| Candidate | Shared batch-native dual-cap shape map, with scalar evaluation defined by the same semantics and stable named configurations. |
| Exact baseline | Frozen pre-refactor scalar implementation on ordinary valid fixtures, plus exact Kalman value/score where the LGSSM target permits it. |
| Independent derivative authority | TensorFlow forward accumulator for primitive JVPs and scale-aware centered finite differences of the same finite scalar for end-to-end checks. |
| Promotion criterion | All mandatory correctness, scalar/batch parity, identity, GPU/XLA, capacity, and target-specific validation gates pass before internal beta. |
| Promotion veto | Scalar/batch objective mismatch, partial derivative, silent selector fallback, stale tuning/admission identity, cap formula failure, invalid branch accepted as valid, or missing required diagnostics. |
| Continuation veto | Target law changes, irreconcilable scalar-reference contradiction, unavailable GPU for required phases, campaign budget exhaustion, or corrupted artifacts. |
| Repair trigger | Focused test failure, XLA compile failure, cap overflow/saturation defect, conditioning failure, excessive memory/runtime, or validation instability. |
| Explanatory only | Raw speed, descriptive comparator proximity, cap-active fraction, one-seed score, small-step FD residual without a scale-aware ladder, and approximate UKF/SGQF/Zhao-Cui differences. |
| Must not be concluded | Exact nonlinear score, lower bias, universal superiority, posterior correctness, HMC correctness, or production readiness. |

## Evidence Contract

The claimed score is the total derivative of the same deterministic finite
dual-cap value program under fixed random innovations, design, target data,
algorithm configuration, and tuning artifact. The batch route and scalar route
must compute the same quantity row by row within declared float32/TF32 and
float64 reference tolerances.

Mandatory evidence before internal beta:

1. primitive value/JVP tests for both caps, pairwise correction, affine
   restoration, and invalid branches;
2. end-to-end scalar dual-cap value/score parity against same-program finite
   differences using a scale-aware step ladder;
3. batch dual-cap row parity against the frozen scalar reference;
4. batch value-only versus value/score scalar parity;
5. repository-owned selector and immutable algorithm/configuration identity;
6. exact-scope tuning and admission artifacts carrying that identity;
7. GPU/XLA correctness, deterministic replay, runtime, and memory evidence;
8. target-horizon multi-seed validation for every beta model; and
9. a versioned beta result and limitations note.

No approximate nonlinear comparator becomes a correctness oracle. LGSSM
Kalman is an exact model-specific authority. Other comparator values and
scores remain explanatory unless a target-matched exact authority is added.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| radial cap `2.0` | August four-model campaign | family hypothesis | unnecessary overhead or target-specific variance increase | paired coordinate-only/dual timing and multi-seed diagnostics |
| coordinate cap `0.98`, power `8` | Austria cap experiments and four-model campaign | family hypothesis | changes the bulk, overflows in naive FP32 power, or flattens scores | active fraction, stable extreme-value tests, derivative-floor diagnostics |
| four pairwise steps | prior pairwise campaigns | family hypothesis | excess cost or overcorrection | step-count ablation during scope tuning |
| pairwise strength | model-specific prior campaigns | warm start only | cross-model transfer bias or instability | disjoint target-specific tuning |
| `N=1008` | existing comparison scope | compatibility baseline | inadequate target accuracy or misleading capacity | `N` ladder including intended beta counts |
| FP32/TF32 GPU/XLA | repository execution policy | required target backend | small-step derivative noise or overflow | FP64/no-TF32 reference ladder and stable cap tests |
| scalar implementation | current research path | frozen characterization authority only | contains current bugs or blocks batch design | golden tests plus independent mathematical checks; do not preserve known defects as intended behavior |

## Pre-Execution Skeptical Audit

1. **Wrong implementation risk.** Optimizing the existing batch path would
   optimize a diagonal-only algorithm. Batch dual-cap semantics must be
   implemented and parity-tested before performance work.
2. **Golden-test risk.** Characterization tests can freeze defects. Golden
   outputs are limited to ordinary valid fixtures and are paired with
   independent invariants and JVP/FD checks. Known defects such as cap overflow
   and the zero-step early return receive corrected-spec tests, not golden
   preservation.
3. **Proxy-promotion risk.** Passing primitive tests or approximate-reference
   proximity cannot establish posterior or HMC correctness. Internal beta is
   limited to algorithm execution and diagnostics.
4. **Configuration-drift risk.** A bag of keyword arguments permits scalar,
   batch, benchmark, and NeuTra divergence. One immutable versioned config and
   one repository selector must own semantics.
5. **Stale-artifact risk.** Existing NeuTra admission artifacts predate
   dual-cap. They are historical and cannot admit the new target.
6. **Numerical-stability risk.** Direct eighth powers overflow in float32.
   Stable cap value/JVP formulas and extreme-value tests precede batch
   promotion.
7. **Performance-measurement risk.** Current campaign artifacts omit per-arm
   warmed timings. Compile time, warm execution, peak allocator memory, and
   scalar/batch throughput must be reported separately.
8. **Unfair comparison risk.** Performance comparisons must hold target,
   particle count, batch size, dtype, TF32, XLA, device, and diagnostics
   constant. Diagonal, coordinate-only, pairwise, and dual arms use common
   inputs.
9. **Dirty-worktree risk.** The active workspace contains unrelated research
   edits. Implementation phases should use an isolated worktree or a carefully
   bounded branch and must not absorb unrelated changes.

Audit decision: `PASS_FOR_TEST_FIRST_PHASES_ONLY`. No public/default or NeuTra
admission change is authorized until the corresponding phase gates pass.

## Phase Program

### Phase 0: Boundary, Inventory, And Reference Freeze

**Question:** Can we state exactly what is being repaired without confusing
dual-cap with bounded-teacher or projected-cumulant research paths?

**Actions:**

- inventory scalar, batch, NeuTra, benchmark, identity, tuning, and admission
  entry points;
- classify dual-cap, bounded-teacher, and projected-cumulant paths separately;
- record current callable signatures, control schemas, target signatures, and
  artifact dependencies;
- freeze small ordinary-valid scalar fixtures for `d=1`, `d=2`, `d=3`, and
  `d=18`, including value, score, particles, residuals, and cap diagnostics;
- record known defects as non-golden cases; and
- work in an isolated branch/worktree for implementation.

**Pass:** a machine-readable inventory and reference-fixture artifact exist;
known defects and intended behavior are distinguished explicitly.

**Veto:** target ambiguity, inability to reproduce the current scalar route,
or unresolved overlap with another agent's files.

### Phase 1: Test Harness And Missing Coverage

**Question:** Do tests protect both the current scalar behavior and the
intended release contract before refactoring?

**Actions:** add tests in this order:

1. stable cap value/JVP tests over negative, zero, ordinary, threshold, large,
   and extreme float32/float64 values;
2. zero-correction plus coordinate-cap semantics;
3. degenerate, nearly singular, non-finite, and extreme-cloud validity tests;
4. composed pairwise plus radial plus coordinate-cap plus affine-restoration
   JVP check;
5. full scalar finite-value/score dual-cap FD ladder at well-conditioned
   fixtures;
6. scalar `d=1` structural no-op for pairwise/radial while coordinate cap
   remains active;
7. a fail-closed test proving the current batch API cannot silently accept or
   claim dual-cap before Phase 4;
8. executable scalar/batch parity fixtures whose batch assertions are enabled
   in the same Phase 4 change that implements the capability;
9. selector/configuration validation tests enabled with Phase 2; and
10. target-signature and stale-admission rejection tests enabled with Phase 5.

Do not leave permanent `xfail` coverage for release requirements. Before the
capability exists, test explicit rejection. When it is implemented, replace
that rejection assertion with positive parity assertions in the same change.

Use branch coverage as a diagnostic, not a promotion metric. Required semantic
cases must be named explicitly; a high aggregate line-coverage percentage is
not a substitute.

**Pass:** current regression tests remain green; new scalar correctness and
fail-closed capability tests pass; every critical finding has at least one
test owner and a named future positive-parity fixture.

**Veto:** independent JVP/FD tests contradict the intended mathematics or
ordinary valid fixtures cannot be made deterministic.

### Phase 2: Versioned Configuration And Repository Selector

**Question:** Can every route select the same declared algorithm without a
keyword bag or silent fallback?

**Actions:**

- introduce immutable `GenUTAlgorithmConfig` and a schema/version identifier;
- include diagonal, pairwise, radial, coordinate-cap, projected, and
  bounded-teacher policy fields with cross-field validation;
- add repository-owned names `default`, `dual_cap`, `coordinate_cap`,
  `pairwise`, `diagonal`, and `none`;
- map `default` to `dual_cap` only in the new internal selector contract;
- preserve explicit historical options without semantic changes;
- make unsupported scalar/batch capabilities fail closed; and
- bind the selected algorithm name and full config into route and target
  identities.

Do not export this selector as a public beta API until Phase 10. Phases 2-9
use it internally so every route shares one semantics owner.

Keep scope-tuned numerical controls separate from family constants where the
policy requires tuning. The config should distinguish family identity,
scope-tuned values, and experimental features.

**Pass:** selector/config tests pass; serialization round-trips exactly;
invalid combinations and unknown names fail closed; identities change when any
semantic field changes.

**Veto:** caller-stamped identity, implicit option fallback, or algorithm-name
collision with bounded-teacher/projected routes.

### Phase 3: Stable Shared Cap And Shape Primitives

**Question:** Are the cap and affine-restoration primitives numerically stable,
differentiable on their declared domain, and reusable by scalar and batch
routes?

**Actions:**

- extract overflow-safe coordinate-cap value/JVP and rowwise radial-cap
  value/JVP primitives supporting arbitrary leading batch dimensions;
- remove direct large eighth-power evaluation or guard it with a mathematically
  equivalent stable formulation;
- return structured diagnostics including saturation/overflow status,
  derivative minimum, cap activity, displacement, and affine restoration;
- define the zero-step behavior explicitly so an enabled coordinate cap still
  executes;
- centralize covariance conditioning and Cholesky validity policy; and
- retain a no-runtime-autodiff TensorFlow implementation.

**Pass:** Phase 1 primitive tests pass in float64 and float32; manual JVP agrees
with forward accumulator; intended asymptotes and odd symmetry hold; extreme
finite inputs do not silently map to the wrong limit.

**Veto:** non-finite output/JVP for finite in-domain values, wrong asymptote,
or an invalid covariance branch reported as valid.

### Phase 4: Batch-Native Dual-Cap Implementation

**Question:** Does one genuine leading-batch TensorFlow/XLA path implement the
same dual-cap finite program as scalar evaluation?

**Actions:**

- implement batch-native pairwise moment targets, residuals, projection,
  radial cap, coordinate cap, affine restoration, and complete JVP;
- avoid `map_fn`, `vectorized_map`, scalar Python loops, and scalar fallbacks;
- use the shared config and primitives from Phases 2-3;
- update both batch value-only and value/score routes; and
- consider defining scalar evaluation through the same core with leading batch
  size one after parity is established.

**Pass:** for all named algorithm variants and `d={1,2,3,18}` fixtures:

- every batch row matches the frozen scalar/reference value and score within
  declared tolerances;
- batch value-only equals the value returned by value/score;
- JVP and FD checks pass on well-conditioned fixtures;
- no sample-wise loop or scalar fallback appears in the training graph; and
- invalid rows fail independently without contaminating valid batch rows.

**Veto:** scalar/batch objective mismatch, partial batch tangent, cross-row
coupling, or XLA-incompatible control flow.

### Phase 5: NeuTra Identity And Admission-Infrastructure Migration

**Question:** Can NeuTra training and endpoint evaluation consume exactly the
same dual-cap target with truthful provenance?

**Actions:**

- extend `GenUTControls` or replace it with the versioned algorithm/scope
  configuration;
- forward all dual-cap controls through `_core_kwargs`;
- update target signatures, adapter signatures, status schemas, and manifests;
- expose cap and pairwise diagnostics in batch status;
- invalidate historical diagonal-only admission artifacts for dual-cap use;
- add tuning/admission schema support for the complete algorithm identity;
- exercise the loader and target factory with test-only/smoke identities, not
  claim-bearing tuning artifacts; and
- add discovery tests preventing unversioned or unbound GenUT target routes.

**Pass:** a NeuTra batch call and its endpoint call share the same target
signature, algorithm identity, controls, value, and status under a test/smoke
identity; stale or mismatched artifacts fail closed; batch size is greater than
one. No old artifact is promoted and no claim-bearing dual-cap admission
artifact is issued yet.

**Veto:** old admission silently reused, target signature omits a semantic
field, or training and endpoint paths execute different algorithms.

### Phase 6: Correctness And Numerical Admission

**Question:** Is the complete finite value/score program numerically trustworthy
on representative target regions?

**Actions:**

- use a scale-aware FP32 finite-difference policy, with multiple step sizes and
  effective denominators, rather than only `h=1e-3`;
- run FP64/no-TF32 reference arms on reduced but representative scopes;
- test values and scores at centers, local perturbations, and declared chart
  boundaries;
- predeclare derivative, conditioning, affine-restoration, cap-derivative,
  and additivity vetoes; and
- distinguish failure of the cap/JVP from inherited transport/reset
  sensitivity.

**Pass:** every required coordinate has a stable FD window or an independent
JVP authority; scalar/batch differences fit the arithmetic tolerance; no veto
diagnostic fires.

**Veto:** no stable derivative window, unexplained one-sided score error,
non-finite boundary behavior, or cap derivative below its declared admissible
floor without an explicit status failure.

### Phase 7: Performance Refactoring And Capacity

**Question:** Can dual-cap meet beta throughput and memory budgets without
changing semantics?

**Actions:** first profile, then optimize:

- separate compile time, warm scalar time, warm batch time, and per-target
  throughput;
- record TensorFlow allocator current/peak bytes and device placement;
- profile pairwise `O(N d^2 P)` contractions, Cholesky/standardization counts,
  cap/restoration overhead, and time-loop state size;
- benchmark a fair ladder over `N`, `d`, parameter count, horizon, and batch
  size using common inputs;
- remove duplicate moment/Cholesky computations where a mathematically
  identical factor can be reused;
- fuse or algebraically simplify contractions only behind parity tests;
- specialize the `d=1` pairwise/radial no-op at trace time; and
- evaluate chunking or structured pair masks only if profiling shows a real
  capacity need.

Every optimization must pass the Phase 4 parity suite and the Phase 6 numerical
suite before retention.

**Pass:** predeclared beta scopes compile and run on GPU/XLA with verified
memory growth, no CPU fallback, bounded peak memory, deterministic replay, and
recorded throughput. Regression thresholds are based on warmed paired runs,
not a single noisy timing.

**Veto:** semantic change, graph explosion, out-of-memory at the beta scope,
or material regression without a justified tradeoff.

### Phase 8: Target-Specific Tuning And Multi-Model Validation

**Question:** Does the repaired route remain viable on every intended beta
model under exact-scope tuning?

**Initial beta models:** LGSSM, KSC SV, exact transformed SV, generalized SV,
predator-prey, and Austria SIR. A model may remain blocked rather than being
silently replaced by another target or comparator.

**Actions:**

- perform disjoint calibration/validation tuning per complete scope;
- tune transport controls, pairwise strength/steps, and any declared family
  hypotheses allowed by the beta contract;
- freeze controls before untouched multi-seed claim runs;
- issue fresh tuning and admission artifacts only after the Phase 7
  implementation and performance dependency closure is frozen;
- report hard vetoes first, then descriptive value/score/variance/runtime
  differences; and
- use exact oracles only where available and label all other comparators as
  approximate diagnostics.

**Pass:** every admitted model has finite/program-valid multi-seed results,
scope-bound controls, complete diagnostics, and no correctness veto. Passing
does not establish cross-model superiority.

**Veto:** claim-seed leakage, cross-scope tuning reuse, wrong event order or
target hash, missing required diagnostics, or statistically unsupported
ranking presented as fact.

### Phase 9: NeuTra Training Smoke And Downstream Beta Check

**Question:** Can the dual-cap target support the intended batch-native GPU
NeuTra workflow without target mismatch?

**Actions:**

- run a bounded target-specific GPU training smoke for each admitted model;
- verify every optimizer step consumes a genuine batch greater than one;
- check loss/value/score finiteness, deterministic target replay, heldout
  target diagnostics, and endpoint identity;
- run short proposal/HMC mechanics only as a beta integration diagnostic; and
- do not claim posterior convergence from a smoke.

**Pass:** training uses the admitted dual-cap target, produces a reusable
artifact with exact signatures, and passes heldout numerical/integration
checks.

**Veto:** scalar fallback, batch size one, target-signature mismatch,
non-finite training, or endpoint target different from training target.

### Phase 10: Internal Beta Packaging And Release Gate

**Question:** Is the code safe and clear enough for bounded internal beta
testing?

**Actions:**

- expose the stable selector and configuration API;
- provide a minimal runnable example and configuration reference;
- document algorithm order, diagnostics, tuning requirement, supported
  models/scopes, expected computational scaling, and limitations;
- add a changelog/version entry and migration note for old diagonal-only
  artifacts;
- run the focused suite, broader affected suite, GPU/XLA regression suite, and
  artifact/readback checks from a clean commit; and
- obtain one terminal code/result review focused on correctness and truthful
  release claims.

**Internal-beta pass:**

- no critical or high correctness finding remains;
- scalar and batch routes are semantically identical;
- all identities and tuning artifacts are current;
- required GPU/XLA scopes pass correctness and capacity gates;
- documentation calls the route experimental/internal beta and states all
  nonclaims; and
- rollback to explicit historical algorithms remains available.

**Production and public-beta nonclaims:** Internal beta does not establish
posterior correctness, HMC convergence, universal superiority, or production
support. Those require a separate downstream evidence program.

## Test Matrix Required Before Refactoring

| Layer | Required cases | Current state |
|---|---|---|
| Coordinate-cap primitive | odd symmetry, threshold, large/extreme values, float32/64, JVP, derivative floor | partial; extreme stability missing |
| Radial-cap primitive | zero/ordinary/extreme row RMS, bound, JVP, batch leading dimensions | scalar partial; shared batch primitive missing |
| Pairwise map | `d=1` no-op, masked pairs, `d=2/3/18`, JVP, affine invariants | scalar partial; batch missing |
| Composition | diagonal + pairwise + radial + coordinate + restoration | scalar mechanics partial; full composed JVP/FD missing |
| Scalar filter | deterministic replay, value/score FD ladder, invalid rows, cap diagnostics | basic route exists; dual-cap end-to-end coverage missing |
| Batch filter | scalar-row parity, value-only parity, row isolation, XLA | diagonal only |
| Selector/config | named algorithms, default alias, serialization, invalid combinations, identity mutation | missing |
| NeuTra target | batch size >1, target signature, endpoint parity, stale artifact rejection | diagonal only |
| Performance | compile/warm timing, memory, `N/d/T/B` ladders, regression thresholds | missing for dual-cap |
| Model validation | exact-scope tuning, multi-seed claims, hard vetoes, uncertainty | scalar descriptive campaign only |

## Refactoring Order

The required logical dependency is:

```text
freeze boundaries and ordinary-valid reference fixtures
-> add missing semantic tests
-> create versioned config and selector
-> extract stable shared cap primitives
-> implement batch-native dual-cap
-> prove scalar/batch and value/value-score parity
-> migrate NeuTra identity and admission infrastructure
-> run full numerical admission
-> profile and optimize behind parity tests
-> freeze the implementation, then tune, admit, and validate each model
-> run batch-native NeuTra integration smokes
-> package as internal beta
```

Performance refactoring before batch semantic parity is forbidden because it
would optimize the wrong target. Public default wiring before new admission
artifacts is forbidden because it would expose a documented algorithm that the
training path does not execute.

## Artifact Layout

Use fresh versioned roots:

```text
docs/plans/artifacts/genut-dual-cap-beta-release-20260808/phase-*/
docs/benchmarks/artifacts/genut_dual_cap_beta_release_20260808/phase-*/
```

Every serious run manifest records Git commit, command, environment, GPU and
memory-growth status, dtype/TF32/XLA, algorithm/config identity, tuning scope,
target/data identity, particle count, batch size, horizon, seeds, wall time,
allocator memory, and artifact paths.

## Compute Budget

- Phases 0-5: routine tests and focused GPU smokes, at most 60 GPU minutes
  total.
- Phase 6: numerical admission ladder, at most 90 GPU minutes.
- Phase 7: performance/capacity campaign, at most 120 GPU minutes.
- Phase 8: model-specific tuning/validation budget must be declared per model
  before launch; do not borrow an under-budgeted protocol from another model.
- Phase 9: bounded training smokes only, with a separate per-model plan and
  budget.

Stop or revise the relevant phase when its budget is exhausted. Do not relax a
correctness gate to preserve schedule.

## Decision Table

| Decision | Primary criterion | Veto status now | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Start refactoring | test-first characterization exists | blocked until Phase 1 tests are added | hidden scalar edge cases | execute Phases 0-1 | beta readiness |
| Promote shared config | selector/identity tests pass | not checked | compatibility with historical options | Phase 2 | runtime correctness |
| Promote batch dual-cap | scalar/batch and endpoint parity pass | currently blocked: batch route is diagonal-only | high-dimensional tangent cost | Phases 3-4 | target accuracy |
| Admit NeuTra target | exact algorithm/tuning/target identity | currently blocked: stale diagonal admission | target-specific training behavior | Phase 5 infrastructure, Phase 8 admission, then Phase 9 | posterior correctness |
| Optimize | Phase 4 and Phase 6 correctness frozen | not authorized before parity | pairwise `O(Nd^2P)` cost | Phase 7 | scientific superiority |
| Internal beta | all Phase 10 gates pass | currently blocked | downstream robustness | complete Phases 0-10 | public/production readiness |

## Inference Status

| Evidence class | Current status |
|---|---|
| Hard veto screen | Scalar mechanics are finite on tested campaign scopes; batch semantic mismatch is a hard beta veto. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Four-model scalar value/score and variance comparisons. |
| Default readiness | Policy-selected family only; runtime/public default readiness is not established. |
| Next evidence needed | Test-first scalar specification, batch implementation/parity, identity migration, numerical admission, and target-specific validation. |

## Post-Plan Red Team

The strongest alternative explanation for a successful implementation is that
the dual-cap program is internally consistent but still approximates the
underlying nonlinear likelihood poorly. This program can establish that one
finite algorithm and its derivative are implemented consistently; it cannot
establish scientific adequacy without downstream target-specific posterior
evidence.

The conclusion should be overturned if scalar/batch parity requires changing
the declared scalar objective, if no stable derivative window exists after
separating transport sensitivity, if target-specific tuning repeatedly fails,
or if GPU capacity makes the pairwise route unusable at intended model sizes.

The weakest current evidence is the absence of an exact nonlinear score
authority outside LGSSM and the absence of batch dual-cap execution. Both must
remain explicit limitations throughout the beta program.
