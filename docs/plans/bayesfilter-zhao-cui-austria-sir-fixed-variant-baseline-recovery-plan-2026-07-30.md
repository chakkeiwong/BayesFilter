# Zhao-Cui Austria SIR Fixed-Variant Baseline Recovery Plan

Date: 2026-07-30

Status: `LANE_A_EXECUTED_BLOCK_EXACT_P88_RECOVERY_EXHAUSTED`

Lane-A terminal result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-result-2026-07-30.md`.

Lane B remains proposed and is not authorized. It requires the explicit owner
decision described below.

Parent master plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.

Phase-0 evidence:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-2026-07-30.md`.

## Decision

Use two non-interchangeable lanes:

1. Run a bounded, read-only historical recovery audit. Promote only an artifact
   that proves the exact missing P88 coordinate, transport, retained, input,
   and source identities.
2. If exact recovery fails, stop. Constructing a usable fixed-variant filter
   then requires an explicit owner decision selecting a newly named baseline.
   The new baseline may preserve the P88 training-base algorithm family, but it
   is not P88 reconstruction.

Lane A is the next recommended action because it is cheap and can preserve the
historical identity if the missing artifact exists. Lane B is the practical
fallback. This document proposes Lane B but does not authorize it.

No parameter training, total-score implementation, GPU claim run, method
comparison, or HMC is part of Lane A. HMC remains forbidden until the T20 value
and total score of one admitted fixed-variant program pass their own gates.

## Gap Classification

Phase 0 established an exact P88 T1 squared-TT density, not an exact P88
fixed-TTSIRT retained filter. The gaps are:

| Gap | Current status | Required closure |
|---|---|---|
| P88 state density `phi^2 + tau*lambda` | Exact | Preserve all 36 cores, basis, measure, `tau`, floors, and normalizer. |
| Coordinate semantics | Missing | Exact `mu` and full 36-by-36 frame, with dtype and hashes. |
| KR numerical program | Missing | Complete `KRCDFConfig` and callable/source identity. |
| Frozen transport randomness | Missing | Exact reference arrays and hashes, not seeds alone. |
| T1 retained branch | Missing | Repository-issued identity from the actual density, frame, CDF settings, references, inputs, and code closure. |
| T1 input/observation lineage | Missing | Exact target input and event-order identity. |
| T2 previous-marginal boundary | Not checked | Reloaded T1 retained object accepted by an independently checked T2 target assembly. |
| T2 trained density and value | Absent | Scope-specific training-base fit and same-scalar fixed-value gate. |
| Parameter conditioning | Not started | External theta, state-only integration, exact zero slice. |
| Observed-data total score | Not started | Forward total derivative through the same retained recursion and scalar. |
| T20 streaming and GPU/XLA | Not started | Horizon ladder with bounded live state and final trusted GPU artifact. |

The historical fit residual and holdout residual are explanatory diagnostics.
They do not establish a correct retained filter, value, or score.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can the missing exact P88 fixed-variant retained program be recovered; if not, what is the smallest new fixed-variant baseline needed before parameter and score work can resume? |
| Candidate mechanism | Exact artifact recovery first; otherwise a newly named, fully serialized TensorFlow training-base T1/T2 filter. |
| Expected failure | No historical artifact binds the missing arrays and call graph; or a new baseline cannot produce a coherent, reproducible T1/T2 scalar within memory. |
| Promotion criterion | Lane A: exact identity proof and replay. Lane B: deterministic reload plus independent same-scalar T1/T2 value parity under a coherent declared measure. |
| Promotion veto | Guessed settings, current-code recomputation called P88, changed coordinate meaning, APF/source-replica/retained-grid substitution, UKF insertion, non-finite or mismatched values, unbound randomness, or memory breach. |
| Continuation veto | Corrupt evidence; incoherent target measure; no reproducible T1/T2 finite program; or no compact representation within the memory contract. |
| Repair trigger | Loader/schema/harness faults, deterministic serialization faults, fit instability, or a failed candidate under an unchanged target. |
| Explanatory diagnostics | Fit/validation residuals, correction spread, inverse-CDF residuals, ranks, wall time, and peak memory. |
| Must not be concluded | Recovery or same-program parity does not prove exact likelihood, posterior correctness, statistical superiority, HMC readiness, whole-route source faithfulness, or production readiness. |

Candidate rejection is not research-direction rejection. A failed fit or rank
arm triggers the next predeclared training-base repair unless a continuation
veto fires.

## Mathematical Target

For a fixed-variant time step, the admitted density must be the configured
finite program

\[
  \rho_t(z;R_{t-1})
  = \phi_t(z;R_{t-1})^2 + \tau_t\lambda_t(z),
  \qquad
  Z_t = \int \rho_t(z;R_{t-1})\,d\nu(z).
\]

The retained update and operational value are

\[
  R_t=\Phi_t(R_{t-1}),
  \qquad
  L_T=\sum_{t=1}^{T}\{\log Z_t-c_t\}.
\]

At `t>1`, `rho_t` depends on the previous retained marginal. A collection of
independent one-step fits is not this sequential scalar. An APF log-sum-exp
normalizer is also a different scalar.

Later, theta is an external conditioning input:

\[
  R_t(\theta)=\Phi_t(\theta,R_{t-1}(\theta)),
  \quad
  L_T(\theta)=\sum_t\{\log Z_t(\theta)-c_t(\theta)\}.
\]

Only state variables are integrated. The score must be the total derivative

\[
  D_aL_T
  =\sum_t\left\{\frac{D_aZ_t}{Z_t}-D_ac_t\right\},
\]

with compact forward sensitivities for the three parameters,

\[
  D_aR_t
  =\partial_a\Phi_t
   +(D_R\Phi_t)D_aR_{t-1}.
\]

This forward recurrence is the selected score path. It needs only the current
retained object and its three directional sensitivities. A reverse tape over
all time steps, a full sample-by-axis-by-parameter Jacobian, or a local
transition-plus-observation score is not admissible.

## Source And Adaptation Ledger

The inherited operations remain anchored as follows:

| Operation | Classification | Anchor |
|---|---|---|
| Sequential previous-marginal, transition, likelihood target | `source_faithful` | Zhao-Cui paper Algorithm 2(a), Eq. (15), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:693`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72`. |
| Squared-TT density and marginalization | `source_faithful` | paper Algorithm 2(b-c), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:703`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:1`. |
| Conditional KR proposal and importance correction | `source_faithful` | paper Algorithm 3, `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:890`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43`. |
| Sequential `log Z_t-c_t` accumulation | `source_faithful` | author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:84`. |
| Frozen randomness and repository identity | `fixed_hmc_adaptation` | author randomness at `full_sol.m:22`; local retained-object identity machinery. |
| TensorFlow training-base optimizer, L1 tuning, and validation/audit split | `extension_or_invention` | P86/P88 artifacts and current Zhao-Cui training policy. |
| External theta conditioning and state-only integration | `extension_or_invention` | parent master plan. |
| Manual forward total derivative of the finite retained program | `extension_or_invention` | parent master plan; equality is checked against the same local scalar. |

These anchors constrain inherited operations. They do not make the TensorFlow
trainer or assembled BayesFilter route source-faithful, and they do not permit
reviving author TT-cross/ALS.

## Lane A: Exact Historical Recovery

### A0. Inventory Without Recalculation

Search read-only surfaces inside the repository workspace:

- all Git refs, reflogs, dangling objects, and registered worktrees;
- `.complete-highdim-source-snapshot-complete-highdim-leaderboard-20260711-221500`;
- `.claude/worktrees`, ignored artifact roots, archived run outputs, logs, and
  checkpoints;
- P86/P88 manifests, command ledgers, and predecessor artifacts; and
- serialized frame, transport, reference, and retained-object payloads under
  names that do not mention P86/P88 but bind the P88 core or branch hash.

Hash every candidate before inspection and write a recovery inventory. Do not
modify, normalize, or regenerate a candidate. The preserved complete-source
snapshot has already been seen to contain the same P88 fit JSON and source
history; its presence is not itself closure.

### A1. Exact Admission Schema

A candidate passes only if one artifact or a provable immutable artifact chain
binds all of:

- P88 file SHA-256
  `ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e`;
- density branch
  `265f9a06877e9babbba22dde187487fde4b50d08d8ecb98cd26b16467b6c1f10`;
- all 36 P88 core arrays, order, dtype, shapes, and hashes;
- `coordinate_frame_mu` and the full frame matrix arrays and hashes;
- every `KRCDFConfig` field;
- exact frozen reference arrays and hashes;
- target input/observation, time index, event order, measure, `tau`, defensive
  reference, shift, floors, and normalizer convention;
- retained branch identity or enough historically bound fields to reproduce
  and verify its recorded hash; and
- the actual execution source/dependency closure, including the fit/transport
  callables and framework versions.

Seeds, a frame log determinant, current source, an unverified command string,
or agreement of one scalar are insufficient. If no candidate supplies an
independent expected hash for a regenerated field, regeneration cannot prove
historical identity.

### A2. Fail-Closed Replay

For an admitted candidate only:

1. Load it through a repository-owned non-overridable factory.
2. Reproduce the exact P88 density normalizers.
3. Reproduce the recorded transport/retained identity without caller-stamped
   canonical fields.
4. Compare `log Z_1-c_1` with an independent direct TT contraction of the same
   `phi^2 + tau*lambda` program.
5. Reload the serialized T1 retained object in a fresh process and pass it to
   the T2 previous-marginal target boundary without fitting T2.

The independent bridge must not call the value implementation it checks.
Inverse-CDF residuals, transport samples, or likelihood fit residuals are
diagnostics; they cannot replace same-scalar value equality.

### A3. Lane-A Exit

Allowed exits are:

- `PASS_EXACT_P88_FIXED_VARIANT_BASELINE_RECOVERED`; or
- `BLOCK_EXACT_P88_RECOVERY_EXHAUSTED`.

The pass reopens Phase 0 of the parent master plan. The block preserves P88 as
an exact density artifact only and returns the owner decision for Lane B.

### Lane-A Budget And Artifacts

| Resource | Bound |
|---|---|
| Inventory attempts | Two; a second is only for a repaired search manifest. |
| Inventory wall time | 30 minutes total, read-only, no TensorFlow/GPU. |
| Candidate replay | One primary plus one infrastructure repair, each at most 10 CPU minutes. |
| CPU memory | 12 GiB peak; stop before loading unbounded artifact sets. |
| Output root | `docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/recovery-attempt-NN/` |

CPU replay must intentionally set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow
import and record that choice. No package, environment, or source mutation is
allowed in Lane A except the focused recovery loader/tests and versioned result
artifacts needed to perform the replay.

## Owner Decision Boundary

If Lane A blocks, the owner must choose whether the target changes from exact
P88 reconstruction to a new fixed-variant baseline. The recommended new
identity is:

```text
zhao_cui_austria_sir_fixed_variant_training_base_v1
```

This name must never normalize to, alias, or claim `P88` identity.

The decision permits reuse of the P88 algorithm family, not silent reuse of
the P88 coefficients. TT coefficients are coordinates in the missing affine
frame. With a newly generated frame, their physical meaning changes. Therefore
P88 cores may be used only when either:

1. the exact P88 frame is recovered; or
2. an exact basis/core coordinate transformation is derived and independently
   proves density equality.

Otherwise the cores are at most an initialization hypothesis and the new
baseline must be retrained. Calling a new frame plus old cores "P88" is wrong.

## Lane B: New Fixed-Variant Baseline

Lane B starts only after the owner approves the changed baseline identity.

### B0. Freeze The Finite Program

Implement a repository-owned baseline factory and artifact schema that bind:

- model, state/observation axes, sealed observation hashes, and event order;
- target density and measure convention;
- full affine-frame arrays and hashes;
- basis definitions, TT ranks, core arrays, and hashes;
- `tau`, defensive density, shift, floors, and direct normalizer policy;
- complete `KRCDFConfig`;
- frozen reference arrays and hashes;
- training, validation, and untouched audit identities;
- TensorFlow/TFP versions, dtype, XLA/TF32/device posture, Git commit, and
  transitive repository-source hashes;
- retained-object and previous-retained identities issued from actual
  callables; and
- the value definition `sum_t(log Z_t-c_t)`.

Reject missing, stale, cross-target, or caller-stamped fields. Add tamper tests
for every identity-owning group.

Before fitting, state the operational measure exactly. If clipped pushes,
Gaussian densities, reference coordinates, and Jacobians do not define one
coherent finite scalar, exit
`BLOCK_NEW_FIXED_VARIANT_TARGET_MEASURE_INVALID`. Do not hide a mismatch by
calling the value approximate likelihood.

### B1. Deterministic T1 Construction

1. Seal disjoint training, validation, and untouched audit inputs.
2. Construct and serialize the T1 frame before core fitting.
3. Use the TensorFlow training-base optimizer. Do not use ALS, source replica,
   APF, retained grid, or UKF initialization.
4. Tune L1 explicitly with rank, degree, learning rate, batch size, stopping
   rule, and capacity treated as hypotheses rather than inherited defaults.
5. Freeze the selected settings before untouched T1 evaluation.
6. Serialize T1 density, transport, frozen references, retained object, source
   closure, and repository-issued identity.

P88 order 3/rank 4/LR `3e-4`/zero L1 is one warm-start comparator, not the new
default. Current Zhao-Cui policy requires L1 tuning; audit data cannot select a
candidate.

### B2. T1 Value And Reload Gate

Require all of:

- exact reload identity in a fresh process;
- finite positive square and defensive normalizers;
- direct TT contraction versus operational `log Z_1-c_1` agreement in FP64:
  `abs(delta) <= 1e-9 * (1 + abs(L_ref))`;
- deterministic replay of retained samples, weights, and identity;
- inverse-CDF and mass checks within predeclared numerical tolerances; and
- peak-memory compliance.

A validation residual may veto a fit, but it cannot pass this value gate.

### B3. T2 Previous-Marginal Boundary

Before training T2:

1. Reload T1 from disk.
2. Evaluate the T1 previous marginal at deterministic points using both the
   retained-object API and an independent TT marginal contraction.
3. Bind that marginal into the source-order T2 target
   `previous * transition * likelihood`.
4. Verify axes, reference/physical maps, Jacobians, event order, and input
   identities.

This phase must not synthesize T2 cores. Its pass proves the boundary only.

### B4. Scope-Specific T2 Fit And Fixed-Value Gate

Train T2 under a separate recorded scope using the same trainer family and the
same validation/audit discipline. A T1 setting is only a warm start for T2.
Freeze a T2 tuning artifact before untouched evaluation.

Require:

- T1 and T2 fresh-process reload;
- independent marginal and normalizer contractions at both steps;
- operational value
  `L_2=(log Z_1-c_1)+(log Z_2-c_2)` agreement under the B2 FP64 tolerance;
- deterministic retained lineage from T1 into T2; and
- bounded current/previous live memory.

The exit is `PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE`, not an exact
likelihood, P88, score, T20, or HMC claim.

### B5. Parameter Extension After Fixed-Value Admission

Only after B4:

1. Add three centered external conditioning coordinates.
2. Preserve the fixed slice algebraically at theta zero.
3. Evaluate parameter cores first and integrate state axes only.
4. Keep parent hashes and the new child identity distinct.
5. Train with disjoint calibration/validation/audit data over the declared
   theta domain.

Origin equality is checked before nonzero-theta fit quality. A child that fits
nonzero theta but changes the origin scalar is rejected.

### B6. Total Score At T1/T2

Implement the three-parameter forward recurrence, with an owner ledger for
every direct and retained dependency. The runtime score must be an analytical
contraction; TensorFlow autodiff and central finite differences are independent
diagnostics only.

At sealed FP64 diagnostic points require:

- manual versus total autodiff infinity-norm error
  `<= 5e-7 * (1 + ||g_ad||_inf)`;
- manual versus central-FD infinity-norm error
  `<= 5e-5 * (1 + ||g_fd||_inf)`; and
- the same value, branch, frozen randomness, and retained lineage for all three
  computations.

Do not loosen tolerances after inspecting untouched failures. A failed gate
triggers derivation/implementation repair or a reviewed tolerance study on
separate calibration points.

### B7. Horizon And Backend Ladder

Proceed `T1 -> T2 -> T5 -> T10 -> T20`, recording value, total score, retained
hashes, rank, live tensor estimates, measured peak allocation, and wall time at
each rung. Fit and serialize each new time-step object; do not build a time-axis
tensor product or retain all sample clouds in memory.

Only after the FP64 value/score ladder passes, run the identical program in
FP32/TF32/XLA on trusted GPU access with verified TensorFlow memory growth.
Predeclare FP32 parity tolerances in that phase's execution note before the
claim run. A GPU failure triggers an unchanged-program backend repair, not a
scalar or estimator change.

GenUT, SGQF, and UKF may be compared only after this admission. They are
different approximate likelihood programs, not score authorities for this
fixed variant. Compare each method to the same independent reference where one
exists, report value/score error and resource use, and do not rank stochastic
results without uncertainty. UKF remains a comparator or separately reviewed
warm-start A/B, not baseline initialization.

## Lane-B Memory Contract

For `d` state axes and three parameter axes, core storage is

\[
  O\!\left(\sum_{k=1}^{d+3}r_{k-1}b_kr_k\right),
\]

not `O(B^(d+3))`. The score adds three compact directional carries, so its
intended overhead is a small constant multiple of current retained storage,
not a full Jacobian.

At runtime, only these may be live:

- one bounded training or validation microbatch;
- current and immediately previous retained objects;
- one conditional transport microbatch;
- three compact retained sensitivities; and
- immutable compact frame/core/config manifests needed by the current step.

Completed step artifacts and sample clouds are written to versioned storage
and released. The following are hard vetoes:

- theta-by-state or all-time tensor-product grids;
- generic all-axes retained-grid transition;
- full reverse history or full sample Jacobians;
- all-time particle/training histories in device memory;
- a runtime dense theta grid; or
- APF history introduced to avoid retained-state derivatives.

Initial bounds, treated as hypotheses until the preflight measures them:

| Resource | Bound |
|---|---|
| CPU reference process | 12 GiB peak |
| One conditional/training microbatch | 512 MiB estimated live tensors |
| Concurrent trainers | One |
| GPU live allocation | At most `min(12 GiB, 50% of trusted-probe available memory)` during T1/T2 campaign |
| GPU allocator | Verified memory growth before initialization; no whole-device preallocation |
| Persistent step artifact | Compact arrays/configs/hashes; no unbounded histories |

If the GPU bound cannot be enforced with memory growth alone, use a reviewed
logical-device memory limit and record the policy exception. A microbatch may
shrink only if it preserves the exact objective and update semantics.

## Lane-B Campaign Budget

This is the maximum budget for new-baseline T1/T2 closure, not authorization to
spend it:

| Stage | Maximum |
|---|---|
| CPU engineering smokes | Four launches, 10 minutes each |
| Per-step pilot screen | 12 arms, 10 GPU minutes each |
| Per-step continuation | Four arms, 30 GPU minutes each |
| Per-step finalists | Two arms, two seeds, 60 GPU minutes each |
| Per-step untouched claim | One launch, 120 GPU minutes |
| Total for T1 and T2 | 20 planned GPU-hours; hard campaign cap 24 GPU-hours including two localized repairs |

Every launch uses a fresh directory under
`docs/plans/artifacts/zhao-cui-austria-sir-fixed-variant-baseline-recovery-20260730/new-baseline-v1-attempt-NN/`.
The phase execution note must freeze the exact command, seed ledger, arm table,
environment, data hashes, and remaining budget before any serious run. No HMC
or T20 training is inside this budget.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| Exact P88 recovery first | Phase-0 blocker | Recommended baseline-preserving action | Search misses differently named archive | hash search across all workspace artifact surfaces and Git objects |
| P88 cores with missing frame | Historical artifact | Density exact only; not reusable in a new frame by default | changed physical density | require exact frame or exact coordinate-transform proof |
| Training-base optimizer | P86/P88 local fixed variant | Required Lane-B family | current helper drift changes objective | dependency closure plus objective parity test |
| P88 rank/degree/LR/L1 | Historical comparator | Warm-start hypothesis | target/time-specific underfit or instability | successive-halving calibration/validation |
| L1 tuning | Current Zhao-Cui policy | Required | zero-L1 inherited as an untested default | explicit candidate grid with audit separation |
| Frozen samples | Fixed-HMC adaptation | Required for deterministic program, not HMC authorization | unbound randomness changes value/score | array hashes and fresh-process replay |
| UKF | Separate P76/P77 line | Excluded from baseline | silent estimator/initialization change | dependency exclusion test |
| Forward sensitivities | Three-parameter recurrence | Selected score design | omitted retained dependency | manual/autodiff/FD T1/T2 triangle |
| T1/T2 first | Smallest sequential discriminator | Required gate | T1 success hides broken recursion | independent T2 previous-marginal check |
| FP64 CPU before GPU | Numerical reference policy | Required diagnostic | CPU program differs from final graph | shared identity and later FP32/XLA tie-out |
| Memory caps | P88 measured peak plus current policy | Planning hypothesis | actual transport/training exceeds cap | allocation forecast and T1 microbatch smoke |

No rank, degree, learning rate, L1 value, batch size, CDF grid, bisection count,
or stopping rule is silently promoted from historical code.

## Evidence Contract

| Field | Lane A | Lane B |
|---|---|---|
| Question | Is the exact historical P88 retained program recoverable? | Can a newly named training-base T1/T2 fixed variant be made deterministic and internally value-correct? |
| Baseline | P88 JSON/core/branch identities | New repository-issued baseline identity |
| Primary criterion | Complete identity proof plus exact replay | Fresh reload plus independent same-scalar T1/T2 parity |
| Hard vetoes | Any missing/guessed field or unverified regenerated identity | Incoherent measure, identity mismatch, value mismatch, drift, non-finite result, or memory breach |
| Explanatory only | Search matches and scalar similarity | Fit residuals, inverse-CDF residuals, ranks, runtime, correction spread |
| Nonclaim | No correctness beyond recovered finite program | No exact likelihood, score, T20, HMC, superiority, or production claim |
| Preserved artifact | Versioned recovery inventory/replay/result | Versioned factory, tuning, T1/T2 objects, manifest, and result |

## Stop And Repair Matrix

| Observation | Classification | Action | Forbidden response |
|---|---|---|---|
| No complete historical artifact | baseline recovery failure | emit Lane-A block and request owner Lane-B decision | guess CDF/frame/reference fields |
| P88 cores fail coordinate-semantics gate | baseline identity failure | retrain under new identity if Lane B is approved | call new-frame cores P88 |
| Target measure cannot be stated coherently | target/math failure | continuation veto and owner-level target correction | hide mismatch with approximate wording |
| T1 direct value disagrees | implementation/numerical failure | repair same scalar | use fit residual as correctness evidence |
| T2 marginal disagrees | retained recursion failure | repair before T2 fit | train around the mismatch |
| Candidate residual/rank gate fails | fit/capacity failure | continue predeclared tuning ladder | reject whole fixed-variant direction |
| Rank or allocation exceeds cap | representation/resource failure | revise TT ranks/bases/microbatch within the same estimator | retained-grid or dense theta fallback |
| Manual score disagrees with total diagnostics | derivative ownership failure | audit every recurrence owner | report local component score |
| GPU/XLA differs after CPU pass | backend failure | unchanged-program backend repair | change value definition |
| UKF appears in baseline closure | baseline drift | fail dependency guard | describe it as inherited initialization |

## Skeptical Plan Audit

| Risk checked | Finding and correction |
|---|---|
| Wrong baseline | P88 is an exact density artifact, not a proved retained filter. The plan never promotes it without missing identity evidence. |
| Proxy promotion | Fit/holdout, inverse-CDF, and transport residuals are explanatory or veto diagnostics; independent same-scalar value equality is primary. |
| Hidden coordinate assumption | P88 cores depend on the missing frame. New-frame reuse is prohibited without an exact transformation proof. |
| Stale context/source | The P88 introducing commit lacks the named fit script. Current-code recomputation cannot prove history; Lane A requires an immutable closure. |
| Missing stop condition | Both lanes have terminal blocks for absent identity, invalid measure, bad recurrence, and memory infeasibility. |
| Unfair comparison | GenUT/SGQF/UKF are not used as Zhao-Cui score authorities; later comparisons require the same target/reference and uncertainty-aware interpretation. |
| Environment mismatch | Recovery/replay is explicitly CPU-hidden. Serious GPU work is deferred and requires trusted access, XLA, TF32, and verified memory policy. |
| Memory blow-up | Forward three-direction carry and current/previous streaming replace full history, full Jacobians, and tensor-product grids. |
| Source-faithfulness drift | Every inherited operation retains paper and author-code anchors; local trainer/conditioning/score work remains labeled extension. |
| UKF drift | UKF is neither silently added nor removed from P88; it is excluded from closure and remains an optional later comparator. |
| Command cannot answer question | Lane A inventory answers existence/identity; replay answers finite-program equality; training residuals cannot answer value correctness. |

Audit verdict: `PASS_FOR_PROPOSAL`. The plan has a valid next action and does
not authorize Lane B implicitly. It must be re-audited after the Lane-A
inventory and before any Lane-B implementation or serious run because the
recovered evidence or owner decision may change the baseline.

## Definition Of Done

The current gap is closed only when one of these is true:

1. exact P88 retained-program recovery passes Lane A; or
2. the owner approves Lane B and the newly named baseline passes deterministic
   T1/T2 reload, independent value, previous-marginal, identity, measure, and
   memory gates.

Only then may the parent program resume parameter conditioning, total score,
horizon scaling, and final GPU/XLA validation in that order. HMC is outside
this plan.
