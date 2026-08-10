# Zhao-Cui Bounded-Reference GenUT Austria T1/T2 Plan

Date: 2026-08-06

Status: `EXECUTED_STOPPED_AFTER_HARD_VETO_AND_BUDGET_EXHAUSTION`

## Research intent ledger

| Field | Frozen decision |
|---|---|
| Main question | Can the current-source Lane-B Zhao-Cui T1/T2 squared-TT objects provide finite, differentiable third/fourth-moment targets that can be used by GenUT without empirical particle moments, and does the resulting T2 finite program remain numerically valid? |
| Candidate mechanism | Contract moments of the retained bounded Zhao-Cui reference coordinates, apply diagonal/pairwise corrections in those same coordinates, map the corrected cloud back to physical Austria state coordinates, and exactly restore the weighted physical mean/covariance. |
| Exact baseline | The same Austria T2 GenUT finite program, particles, observations, Contract-E controls, arithmetic, and seeds with no higher-moment correction. |
| Expected failure | Wrong retained-axis orientation; non-finite inverse algebraic map; teacher/core-tangent contraction error; physical mean/covariance drift; manual-score/finite-difference disagreement; or a cap that is inactive or destroys moment reduction. |
| Promotion criterion | None. This is an implementation-validity and mechanism diagnostic, not a default, HMC, T20, or superiority run. |
| Promotion veto | Any stale/invalid T1/T2 issuer chain, non-finite target/value/score, failed reference-moment JVP check, incorrect time-index target selection, failed physical affine restoration, invalid Contract-E step, or failed same-program score finite difference. |
| Continuation veto | The bounded reference moments are mathematically undefined, the current-state block cannot be identified, the issuer artifacts fail strict reload, the corrected finite program cannot preserve physical mean/covariance, or the two-attempt serious-run budget is exhausted. |
| Repair trigger | A localized loader, dtype, shape, XLA tracing, artifact-writing, or tolerance-calibration failure that does not change the target, data, arms, seeds, hardware class, or total budget. |
| Explanatory only | T2 value/score displacement, target residuals, tail RMS, cap scale, runtime, and cross-seed ranges. |
| Must not be concluded | T20 benefit, posterior correctness, HMC/NeuTra readiness, an exact physical likelihood, unbiasedness, a statistically supported ranking, a production default, or that Zhao and Cui proposed this GenUT composition. |

## Correction to the originally proposed experiment

The originally proposed direct call

\[
  x = \mu + A r,
  \qquad
  \texttt{squared\_tt\_shape\_targets\_jvp}(\widehat p,\mu,A,\ldots)
\]

is wrong for the current Lane-B artifacts. Lane B evaluates its TT basis on the
unbounded algebraic coordinate

\[
  r = \frac{u}{\sqrt{1-u^2}}, \qquad u\in(-1,1),
\]

and its defensive squared-TT component has positive constant density in the
bounded reference coordinate `u`. Consequently, the defensive component gives
infinite second and higher raw moments in `r`; physical third/fourth moments
cannot be obtained by pretending that the physical state is affine in the TT
reference coordinate. The existing affine moment contractor also accepts the
Legendre basis, whereas Lane B uses a piecewise Lagrange basis.

The repaired target is explicitly the standardized shape of the retained
bounded coordinates `u`, not the physical raw moments. For each time `t`:

1. contract the Zhao-Cui T1/T2 squared-TT marginal over retained axes `0:18`;
2. compute its standardized diagonal and pairwise third/fourth moments in `u`;
3. transform the weighted source cloud and Contract-E reset cloud from physical
   `x_t` to the same `u_t` chart;
4. apply the target correction and optional radial cap in `u_t`;
5. map back through the algebraic and affine maps; and
6. restore the original weighted physical mean and covariance, including their
   total JVP.

This is a bounded-feature teacher. It does not claim equality to physical
third/fourth moments.

## Source grounding and classification

Checked primary/source anchors:

- Zhao and Cui, *Tensor trains for sequential learning of dynamical systems*,
  JMLR 2024, Equation (13) and Proposition 2 / Equation (14), local full text
  `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539`;
- sequential adjacent-state target and retained marginal, paper Equation (15)
  and Algorithm 2(a), same local source `:655`;
- author squared-TT marginal contraction,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25`;
- author/source event order and previous-marginal recursion,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72`;
- Lane-B T1 identity declares `axis_order=(z1,z0)` and retained axes `0:18` in
  `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_tf.py:831`;
- Lane-B T2 identity declares `axis_order=(z2,z1)` and retained axes `0:18` in
  `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_tf.py:682`.

Classification:

| Operation | Classification |
|---|---|
| Squared-TT density and prefix marginal contraction | `source_faithful` operation inside the teacher |
| Frozen current-source T1/T2 Lane-B artifacts and first-order parameter children | `fixed_hmc_adaptation` / existing `extension_or_invention` artifact semantics |
| Bounded-reference third/fourth-moment contraction | `extension_or_invention` |
| Zhao-Cui teacher plus GenUT/Contract-E correction and radial cap | `extension_or_invention` |

The assembled route must never be called source-faithful Zhao-Cui.

Literature metadata, forward snowballing, and a broad omission audit are not
needed for this bounded implementation test. The local primary paper and pinned
author code answer the source-operation question. No retraction or quarantine
notice is present in the repository source ledger; live external metadata is
not used as evidence.

## Evidence contract

### Exact artifacts

- T1 parent identity:
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`.
- T2 parent identity:
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`.
- T1 current-source tangent issuer:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-training-jvp-20260806/attempt-01-current-closure/`.
- T2 current-source tangent issuer:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-training-jvp-20260806/attempt-01-current-closure/`.

### Baseline ladder

All arms use identical `T=2`, `N=1008`, observations, particle seeds, design,
Contract-E settings, FP32 storage/arithmetic, TF32 disabled, and XLA execution.

| Arm | Teacher correction |
|---|---|
| `no_shape` | no diagonal or pairwise correction |
| `teacher_diagonal` | Zhao-Cui bounded-reference diagonal third/fourth moments only |
| `teacher_pairwise_uncapped` | diagonal plus all ordered pairwise third and unordered pairwise fourth moments, no cap |
| `teacher_pairwise_cap8` | same pairwise target with cap `c=8` |
| `teacher_pairwise_cap2` | same pairwise target with cap `c=2` |

The inherited correction settings are `4` iterations, diagonal strength `0.2`,
pairwise strength `0.02`, and floors `1e-5`. They are warm-start hypotheses from
the historical Austria diagnostic, not tuned defaults. Cap 8 and cap 2 are
hypotheses selected to span weak and material damping. No arm can be promoted
from this run.

### Pass/veto checks

- Exact strict reload and identity checks for both parents and both tangent
  issuers.
- T1 and T2 retained-axis/frame assertions, including zero top-right frame
  dependence and physical/local/reference round trip.
- Reference-coordinate operator matrices agree with independent piecewise
  Gauss quadrature on a small fixture.
- All three parameter-direction teacher JVPs agree with centered differences of
  the same linear core child on a focused fixture.
- The bounded correction manual JVP agrees with TensorFlow forward autodiff on
  a small deterministic cloud.
- Each executed arm is finite, program-valid, and restores physical mean and
  covariance within FP32 tolerances.
- One predeclared seed and every nontrivial arm pass a same-finite-program
  centered-FD score check using `h=1e-3`, with a step-halving explanatory row.
  The prospective gate is maximum absolute residual `<=0.08` and maximum
  normalized residual `<=0.03`; if step halving shows roundoff domination, the
  candidate is not admitted and the tolerance is not relaxed after inspection.
- Caps must satisfy post-cap per-particle RMS `< c` when active. An inactive cap
  is reported and cannot support a damping claim.

### Descriptive run

Use common seeds `98601, 98602, 98603`. Report every value, score coordinate,
increment, target residual, cap statistic, and cross-seed range. Three seeds can
identify crashes, one-sided samples, and gross variance, but cannot rank viable
arms statistically.

Versioned output root:
`docs/benchmarks/artifacts/zhao_cui_bounded_reference_genut_austria_t1_t2_20260806/`.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| T1/T2 only | Current validated fixed-variant coverage; baseline | Later-time behavior differs | Explicit T20 nonclaim |
| Retained axes `0:18` | Repository identities and source event order; reviewed | Wrong marginal | Frame/block/round-trip assertions |
| Bounded `u` moments | Mathematical repair; hypothesis | Useful shape in `u` may not improve physical filtering | T2 value/score and residual diagnostics |
| All off-diagonal pairs | Existing full pairwise GenUT semantics; hypothesis | Excess cost/noisy correction | finite/residual/tail diagnostics |
| N=1008 | Historical exact cubature-compatible Austria diagnostic; convenience baseline | Particle-count interaction | N-specific nonclaim |
| Strengths/steps | Historical empirical-target warm start; unpromoted hypothesis | Under/over-correction | no-shape and diagonal ladder plus displacement diagnostics |
| Caps 8 and 2 | Earlier reset damping diagnostic; unpromoted hypotheses | inactive or over-damped | cap scale and target residuals |
| FP32, TF32 off | Prior derivative-parity decision for GenUT; reviewed diagnostic choice | slower execution or FP32 FD noise | GPU smoke and FD step-halving |
| Three seeds | Bounded diagnostic budget | cannot support ranking | inference-status table and nonclaim |

## Skeptical plan audit

Audit verdict: `PASS_AFTER_MATERIAL_REVISION`.

- **Wrong baseline:** repaired. Empirical moment targets are excluded. The exact
  no-shape arm and diagonal/pairwise ladder share all non-candidate inputs.
- **Proxy promotion:** controlled. Moment residuals, cap activity, and short T2
  value/score differences are explanatory; there is no promotion criterion.
- **Missing stop conditions:** repaired with strict artifact, moment, JVP,
  restoration, finite-program FD, and two-attempt continuation vetoes.
- **Unfair comparison:** controlled with common random numbers and identical
  numerical controls outside the declared teacher/cap mechanism.
- **Hidden assumption:** the first draft incorrectly treated Lane-B physical
  state as affine in the TT coordinate. The algebraic map and infinite
  defensive physical moments invalidate that route. This plan uses bounded
  coordinates on both teacher and particle sides and states the changed target.
- **Stale context:** current-closure T1/T2 issuers dated 2026-08-06 are required;
  historical recursive-T20 teacher artifacts are excluded.
- **Environment mismatch:** serious execution is GPU/XLA with verified memory
  growth before device initialization. CPU-hidden runs are mechanics tests only.
- **Artifact sufficiency:** the JSON result binds source hashes, artifact
  identities, command, environment, device, seeds, controls, diagnostics, wall
  time, and output path.

## Pre-mortem

- The run could pass while matching moments in different coordinates. Prevent
  this by applying the identical algebraic reference transform to TT observables
  and particle clouds and by recording the coordinate semantics in the result.
- It could pass score FD while omitting teacher tangents if the correction is
  weak. Prevent this with independent target JVP checks and a target-tangent
  ablation in the focused test.
- It could reduce bounded-reference residuals while damaging physical moments.
  Prevent this with post-inverse exact physical affine restoration checks.
- It could fail because FP32 FD is noisy rather than because the manual score is
  wrong. Preserve the fixed prospective tolerance and a step-halving row; do not
  reinterpret a failure as a pass.
- It could look favorable at T2 but fail after repeated nonlinear resets. The
  result must stop at T2 and trigger construction/validation of T3+ teachers.

## Execution sequence and budget

1. Implement the piecewise-Lagrange bounded-reference moment operators and
   three-parameter target assembly without changing existing defaults.
2. Implement the opt-in time-indexed bounded-feature teacher path and physical
   mean/covariance restoration.
3. Add focused operator, target-JVP, coordinate, cap, restoration, and
   finite-program tests; run them CPU-hidden.
4. Run an escalated GPU/XLA one-seed smoke after `nvidia-smi` and a TensorFlow
   memory-growth/device probe.
5. If the smoke passes, run the three common seeds in one fresh result
   directory. At most two serious launches total, including one localized
   repair/retry, and at most 45 minutes aggregate GPU wall time.
6. Record the exact results, inference status, strongest alternative
   explanation, and next justified action in a result note. Run the focused
   tests and `git diff --check` after the final code state.

## Execution revision and attempt ledger

The exact eager all-moment contraction did not complete safely on the actual
36-core Lane-B child. After strict T1/T2 loading, it exceeded the bounded setup
interval and the diagnostic process crashed during TensorFlow eager operation
construction before producing any target. This is an implementation failure of
that contraction route, not evidence against the Zhao-Cui density.

Before GPU execution, the implementation was revised to use a fixed independent
Lane-B retained sampler with exact `log p_TT - log q` correction weights. The
teacher moment tangent is the total derivative of this sampled self-normalized
TT-marginal estimator, obtained from the issued marginal score. It never uses
the GenUT particle cloud. A 256-row setup exceeded the bounded setup interval;
the predeclared minimum of 64 rows was used and its ESS/correction diagnostics
were frozen in a separate artifact. This sampled estimator is a material
qualification relative to the exact contraction originally sought and cannot
support an exact-moment claim.

| Attempt | Classification | Result | Decision |
|---|---|---|---|
| `teacher-attempt01-n64` | CPU-only fixed teacher setup | Pass; T1/T2 ESS `63.869/64` and `63.834/64`; wall time `57.49 s` | Freeze and reuse identical target tensors for all GPU arms. |
| `smoke-attempt01` | Localized harness failure before any arm | Memory growth was configured after imported TensorFlow constants initialized the GPU | Preserve empty directory; repair ordering under unchanged contract. |
| `smoke-attempt02` | GPU/XLA candidate smoke | Hard veto: diagonal arm invalid; uncapped parameter-0 FD absolute residual `0.1073 > 0.08` | Stop the three-seed run. Two serious-launch budget exhausted. |

Post-run review also found that `no_shape` unnecessarily passed through the
bounded map even with zero correction steps, creating a tiny nonzero cloud
displacement. Current code now bypasses the bounded teacher exactly when all
shape steps are zero and a regression test binds this parity. The preserved
smoke artifact remains evidence for its recorded source closure only; it is not
silently upgraded to current-code baseline evidence, and no additional claim
run is authorized under this plan.
