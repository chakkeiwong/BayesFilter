# Zhao-Cui Austria SIR Material-Replay XLA Repair Plan

Date: 2026-08-02

Status: `ACTIVE_IMPLEMENTATION_AND_BOUNDED_VALIDATION`

## Owner Direction

Exact bitwise core replay is retired as an admission requirement for this
score-closeout lane. Numerical replay that preserves at least five significant
digits is sufficient when value, analytical-score, finite-difference,
provenance, input, and memory gates also pass.

The claim-bearing numerical replay must be TensorFlow/XLA native. NumPy, eager
`.numpy()` decisions, and Python-controlled optimizer, microbatch, TT-axis,
parameter-direction, or finite-difference loops are forbidden inside the
numerical computation. Python remains allowed only at static configuration,
artifact I/O, hashing, manifest construction, and reporting boundaries that do
not influence numerical runtime decisions.

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | Can the admitted Lane-B T1 and T2 finite programs be materially replayed and differentiated by one graph-native TensorFlow/XLA route? |
| Candidate | Packed rank-4 TT cores with inactive boundary entries masked, `tf.while_loop` over optimizer steps, T2 microbatches, and TT axes, plus compiled three-direction XLA tangent issuance and independent centered-FD diagnostics. |
| Expected failure | Packed/tuple arithmetic differs beyond the material gate; XLA cannot differentiate the nested loops; memory exceeds 6 GiB; or value/score/FD parity fails. |
| Promotion criterion | T1 then T2 material functional replay, scalar parity, manual issued-tangent parity, independent step-halving centered-FD parity, strict issuer reload, and allocator peak at most 6 GiB. |
| Promotion veto | Any failed criterion, stale input/source identity, nonfinite result, omitted carried T1 marginal derivative, NumPy/Python numerical fallback, or score/value scalar mismatch. |
| Continuation veto | Invalid admitted artifact/input, failed focused XLA parity, two failed claim launches per issuer, hard-cap breach, or failed strict reload. |
| Repair trigger | Localized packing, masking, XLA lowering, serialization, or source-closure failure with the scalar and evidence contract unchanged. |
| Explanatory only | Runtime, compiler logs, physical Fisher score, UKF/GenUT/SGQF comparisons, and residuals below the material gate. |
| Must not be concluded | Bitwise identity, exact physical likelihood, arbitrary-theta correctness, T3+, HMC readiness, source-faithful parameter estimation, or method superiority. |

## Material Replay Contract

TT core elements are gauge-dependent and are not promotion evidence. Apply the
five-significant-digit rule to gauge-invariant positive functional values on
the complete frozen T1 training and calibration clouds:

- normalized full 36D defensive density;
- normalized retained 18D prefix marginal needed by T2; and
- the calibrated scalar value as a separate parity check.

For every parent functional value `p` and replayed value `r`, require

```text
abs(r - p) <= 5e-12 + 5e-6 * abs(p)
```

This is a conservative five-significant-digit minimum. It is stricter than a
plain five-decimal-place rule and handles small entries without division by
zero. Record both the maximum absolute residual and the maximum normalized
residual

```text
abs(r - p) / (5e-12 + 5e-6 * abs(p)).
```

The normalized residual must be at most `1` for every screen. Core absolute and
normalized residuals remain explanatory gauge diagnostics only. This gate
applies independently at T1 and T2. It does not weaken manual/JVP or
same-program centered-FD gates.

New issuer schemas, IDs, statuses, replay-gate fields, and hard-gate names must
say `material`, not `exact`. Historical v2 exact-replay schemas remain
historical and must not be silently reinterpreted.

## XLA Contract

- Use one packed core tensor of shape `[36,4,5,4]` with a repository-built
  active-entry mask. Packing/unpacking is a static artifact boundary.
- Precompute frozen basis values and mass matrices with TensorFlow at setup;
  claim computation consumes tensors only.
- Use `tf.while_loop` for all 96 optimizer steps, all TT-axis contractions,
  T2's 16 microbatches, three JVP directions, and three centered-FD directions.
- Evaluate active and zero-reference target values in the same compiled graph.
- Keep optimizer operation order and mixed FP32/FP64 constants unchanged.
- Mask inactive packed entries before clipping and after every Adam update.
- Use no NumPy, `tf.py_function`, eager `.numpy()` decisions, scalar target
  mapping, or Python numerical fallback in the claim computation.
- Artifact hashing, JSON conversion, tensor serialization, device queries, and
  static source-closure assembly remain host-side boundaries and must not feed
  numerical runtime decisions.

## Skeptical Plan Audit

Verdict: `PASS_FOR_IMPLEMENTATION_THEN_BOUNDED_T1_T2_VALIDATION`.

- Wrong baseline: rejected. The admitted T1/T2 parents and frozen clouds remain
  unchanged.
- Proxy promotion: rejected. Functional materiality is necessary but insufficient;
  value, manual score, JVP, FD, loader, and memory gates remain mandatory.
- Ambiguous tolerance: repaired by the elementwise mixed formula above.
- Small-entry blindness: repaired by the absolute floor and normalized residual.
- T2 inconsistency: rejected by applying the same schema and material gate to
  both issuers.
- Fake-XLA risk: rejected by graph inspection and tests requiring TensorFlow
  `While` control flow with no PyFunc operations or Python numerical loops in
  the owned replay functions.
- Packed representation drift: controlled by tuple-versus-packed full-density,
  prefix-marginal, mass, one-step Adam, full-primal, JVP, and FD parity tests
  before claim execution.
- Missing stop conditions: bounded below.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| `rtol=5e-6`, `atol=5e-12` | Owner requested five significant digits; conservative reviewed interpretation | Too loose near zero or too strict for benign backend drift | Parent-core scale audit and synthetic boundary tests |
| Packed rank 4 | Frozen parent rank; implementation representation only | Inactive padded entries affect loss/clip/update | Mask invariant and tuple/packed one-step parity |
| Same-graph active/origin target | Prior failure hypothesis | Changes theta derivative or origin arithmetic | Origin ratio exactly zero plus eager value/score parity |
| `tf.while_loop` replay | XLA compliance requirement | Operation order changes or autodiff disconnects | One-step/full-primal parity and JVP/FD regression |
| 6,144 MiB hard cap | Existing campaign resource policy | Packed autodiff exceeds memory | Trusted probe and allocator peak veto |

## Evidence Contract

- Exact baselines remain T1 identity
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`
  and T2 identity
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`.
- Primary T1/T2 pass: material full-density, prefix-marginal, and scalar replay
  under the formula above, manual issued-tangent parity, independent
  step-halving centered-FD parity,
  strict reload, and the 6 GiB peak gate. Core residuals are explanatory only.
- Diagnostics that veto: nonfinite tensors, mask violation, source/input/hash
  mismatch, graph fallback/PyFunc, failed prepared-cloud load, missing carried
  marginal, or any criterion above.
- Explanatory only: residual magnitude after it passes, runtime, and compiler
  tracing/optimization messages.
- No bitwise-replay, physical-likelihood, later-horizon, HMC, default-readiness,
  or superiority claim will be made.
- Versioned output root:
  `docs/plans/artifacts/zhao-cui-austria-sir-material-replay-xla-20260802/`.

## Execution Sequence And Budget

1. Implement T1 packed/XLA replay, material schemas, loaders, runner, and tests.
2. Implement the corresponding T2 packed/XLA replay and chained schema.
3. Run CPU-hidden focused parity, graph, loader, tamper, and memory-policy tests.
4. Run trusted GPU hard-cap probes and the smallest compiled T1 diagnostic.
5. If all preflights pass, run T1 in a fresh directory. Budget: two launches
   and 90 minutes total.
6. Strictly reload T1. Only then run T2. Budget: two launches and 120 minutes
   total.
7. Preserve every attempt, update the result/reset notes, and stop at the first
   continuation veto.

No HMC or later horizon is authorized.

## Execution Attempt Ledger

| Attempt | Classification | Result | Repair/next action |
|---|---|---|---|
| T1 primal diagnostic 1, `t1-primal-diagnostic-01` | Localized XLA lowering failure before numerical replay | CUDA/XLA initialized under the 6,144 MiB cap, but reverse-mode lowering through nested TT `tf.while_loop` contractions rejected dynamically bounded TensorLists. The preserved directory is empty. | Add static `maximum_iterations` to every numerical `tf.while_loop`, execute a nested-gradient XLA regression, then use a fresh diagnostic directory. No material replay metric was computed. |
| T1 primal diagnostic 2, `t1-primal-diagnostic-02` | Localized XLA static-shape failure before numerical replay | The statically bounded nested-gradient CUDA regression passed, but the full primal used a cyclic-batch `tf.range` whose start depended on the optimizer loop counter; CUDA XLA requires compile-time range bounds. The preserved directory is empty. | Precompute the full `[96,512]` cyclic index schedule at setup and gather one statically shaped row inside the compiled optimizer loop. No material replay metric was computed. |
| T1 primal diagnostic 3, `t1-primal-diagnostic-03` | Gate-design defect exposed by numerical evidence | Full CUDA/XLA replay succeeded with about 96 MiB peak and scalar residual `7.1e-15`, but core residual was `0.678`. TT cores are gauge-dependent, so core identity is not a valid material-equivalence target after changing representation. | Preserve the failed result. Replace the core promotion gate with normalized full-density and retained-prefix-marginal parity on complete frozen training/calibration clouds. Keep core residuals explanatory only. |
| T1 primal diagnostics 4 and 5, `t1-primal-diagnostic-04` and `t1-primal-diagnostic-05` | Valid functional gate failure | Both packed CUDA/XLA routes matched the calibrated scalar but failed every density-shape screen with maximum log residuals from `0.823` to `1.518`. Restoring true-shape boundary contractions and per-core regularization/global-norm reductions did not materially change the failure. | Do not relax the functional gate. Use an admitted-tuple versus packed trajectory diagnostic to identify the first divergent numerical quantity. |
| T1 trajectory diagnostic 1, `t1-packed-trajectory-diagnostic-01` | First-step mismatch localized | Before accumulated optimizer drift, the packed step-1 `rho` range and gradient norm already differed from the admitted tuple authority. The packed route used eagerly precomputed basis tables while the authority evaluated basis algebra inside its XLA step. | Compile the complete setup-static basis tables with XLA before the optimizer loop; retain tensors as frozen replay inputs. |
| T1 trajectory diagnostic 2, `t1-packed-trajectory-diagnostic-02` | Localized repair passed | XLA basis precomputation made objective terms agree at step 1 and kept all five checkpoints functionally material through step 96. At step 96 the maximum functional normalized residual was `1.19e-8`; peak allocator use was about 158 MiB. | Proceed to one fresh claim-shaped primal diagnostic, then schema/loader tests and the T1 JVP/FD issuer. |
| T1 primal diagnostic 6, `t1-primal-diagnostic-06` | Claim-shaped primal preflight passed | All four functional screens passed; maximum normalized residual `1.55e-8`, scalar log residual `3.55e-15`, and peak allocator use about 233 MiB under the 6 GiB cap. Core residual `3.13e-13` is recorded as explanatory only. | Bind functional and scalar evidence into the v3 issuer identity, run strict loader/tamper tests, then launch the bounded T1 JVP/FD issuer. |
| T1 issuer attempt 1, `t1-material-jvp-issuer-01` | Localized CUDA/XLA higher-order autodiff failure | The origin primal compiled, but forward-over-reverse differentiation through the optimizer `tf.while_loop` failed because CUDA/XLA emitted `XlaDynamicUpdateSlice` without a registered outer gradient. The preserved directory is empty; no score or tangent artifact was issued. | Do not claim JVP/autodiff. Issue offline core tangents by centered differences of the same full XLA primal at `h=5e-5`, compare the resulting manual scalar score against an independent centered difference at `h=1e-4`, retain the existing `2e-4 + 2e-4*abs(score)` tolerance, and label the derivative finite-difference/heuristic only. Runtime remains manual contractions with no autodiff or finite differences. |

## Pre-Mortem

- A packed route could pass functional tolerance while differentiating a
  different target. Eager value/analytical-score and same-program FD checks
  veto it.
- Padded entries could absorb gradients and alter clipping. The active mask is
  applied before global-norm clipping and after every update.
- A graph could contain a `While` but still call host code. Graph inspection
  rejects `PyFunc`, `EagerPyFunc`, and numerical callbacks.
- A five-digit gate could hide cumulative score error. Manual/JVP/FD gates are
  independent and retain their existing tighter tolerances.
