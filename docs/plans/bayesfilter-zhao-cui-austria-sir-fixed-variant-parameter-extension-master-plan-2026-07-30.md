# Zhao-Cui Austria SIR Fixed-Variant Parameter Extension Master Plan

Date: 2026-07-30

Status: `LANE_A_EXHAUSTED_OWNER_DECISION_REQUIRED_FOR_NEW_BASELINE`

Phase 0 terminal result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-2026-07-30.md`.

Historical recovery terminal result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-result-2026-07-30.md`.

Phase 1 and all later phases are inactive. The exact P88 squared-TT density was
reconstructed, but the complete P88 transport/retained baseline was not.
The bounded Lane-A recovery search is also exhausted. Continuing now requires
an explicit owner decision selecting a newly named baseline; it cannot be
described as exact P88 reconstruction.

## Owner Direction

Extend the developed BayesFilter Zhao-Cui fixed-variant line, starting from its
strongest exact serialized artifact. Do not reconstruct the original Zhao-Cui
estimation algorithm and do not replace the fixed-TTSIRT retained-object scalar
with a frozen-proposal APF scalar. Current evidence does not prove that the
P88 artifact was ever an end-to-end T2 or T20 filter; this plan must close that
gap without pretending it was already closed.

The intended delta is limited to:

```text
exact fixed-variant parent
    + three external log-scale parameters
    + parameter-conditioned TT representation
    + analytical total score of the same fixed-variant scalar
    + streaming extension from demonstrated T1/T2 mechanics to T20
```

If that extension cannot be implemented without changing the estimator, stop
and report the exact blocker. Do not silently substitute another algorithm.

## Baseline Correction

The July 30 discussion conflated two separate local lines:

1. P86/P88 is the strongest serialized fixed-variant fit artifact. It uses the
   BayesFilter `training-base` TensorFlow optimizer, serialized 36D TT cores,
   fixed ranks/basis/schedules, L1 policy, and no ALS revival. It is not by
   itself a proved end-to-end filter.
2. P76/P77 is a separate one-step UKF-warm-start training experiment. It was
   not wired into the P86/P88 fixed-TTSIRT retained-object filter as a complete
   working pipeline.

Therefore UKF is not silently inserted into or removed from the canonical
baseline. The baseline is whatever the exact P88 artifact and its callable
lineage prove. P76/P77 may be evaluated later only as an explicitly separate
warm-start candidate; it is not inherited baseline behavior.

The repository also has a retained-object sequential-loop implementation and a
bounded Austria T1/T2 assembly, but these are separate from the exact P88
artifact. The P88 fit script fixes `time_index=1`; the executed P90 bridge is a
separate deterministic `time_index=2` fixture with different branch hashes.
Therefore Phase 0 validates P88 at T1 and carries its retained output only to
the T2 target-input boundary. It must not claim a P88-trained T2 transport or a
complete T20 callable.

## Superseded Directions

The following July 30 documents are historical only:

- `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-implementation-handoff-2026-07-30.md`;
- `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-active-implementation-plan-2026-07-30.md`;
- `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-reset-memo-2026-07-30.md`;
- `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-result-2026-07-30.md`;
- `docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-2026-07-30.md`;
- `docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-reset-memo-2026-07-30.md`; and
- `docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-result-2026-07-30.md`.

Their APF/source-replica mechanics and failures remain historical evidence.
They are not continuation gates or repair triggers for this plan.

## Canonical Fixed-Variant Evidence

The baseline must be reconstructed from these exact surfaces:

| Component | Required anchor | Status entering this plan |
|---|---|---|
| Fixed Austria model | `bayesfilter.highdim.models.zhao_cui_sir_austria_model` | implemented and source-anchored |
| Fixed-variant trainer | `scripts/p86_author_lagrangep_phase5_budget_fit.py`, `training_base_optimizer` | implemented |
| Strongest fit artifact | `docs/plans/bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-27.json` | serialized T1 36D rank-4 cores; degree evidence only |
| Current training policy | TensorFlow training-base, target-specific L1 tuning, validation/audit separation, no ALS | binding for new training |
| Squared TT/TTSIRT | `SquaredTTDensity` and `FixedTTSIRTTransport` | implemented primitives |
| Retained-object loop | `source_route_run_sequential_fixed_hmc` | implemented sequential mechanics |
| Austria T1/T2 assembly | `p59_author_sir_step_spec_assembly` | bounded two-step mechanics; not P88-core integration evidence |
| Proposal correction | `source_route_generate_retained_samples` | implemented target-minus-proposal correction |
| Previous marginal | `source_route_previous_marginal_log_density` | implemented value operation |
| Operational value | `SourceRouteSequentialResult.log_marginal_likelihood` | sum of transport-normalizer increments |
| Value bridge | P90 author-formula replay | separate deterministic T2 same-scalar fixture passed; reusable comparator architecture, not P88 branch evidence |
| Local parameter score | `ParameterizedZhaoCuiSIRSSM` plus P90/P91 component carry | implemented local components only |

The P88 artifact is not correctness, T20, score, HMC, or production evidence.
It is the strongest concrete fixed-variant training artifact to preserve while
closing the missing integration and parameter-extension work.

Its current file SHA-256 is
`ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e`;
its recorded Git introduction is commit
`c815edc52162779e969b2982723b2f52770fd849`. Phase 0 must fail on a different
payload unless a reviewed plan revision deliberately selects a different
parent. The artifact binds target id `zhao_cui_sir_austria_d18`, reference
measure, 36 axes, rank 4, order 3, 25 basis functions per axis, zero-L1
comparator status, seeds and training backend. It does not bind the July 30
observation hashes, and its fit script fixes `time_index=1`. Phase 0 must record
both facts rather than inventing an observation lineage or T2 cores.
Observation binding and the first observation-specific T2 fit belong to Phase
1.

## Source Relation And Extension Ledger

The inherited mathematics was checked against the paper and author code. These
classifications apply operation by operation; they do not make the assembled
BayesFilter route wholly source-faithful.

| Operation | Classification | Paper and author-code anchor | Binding interpretation |
|---|---|---|---|
| Joint sequential target `previous marginal * transition * likelihood` | `source_faithful` | paper Algorithm 2(a), Eq. (15), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:693-719`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-80,132-135` | Preserve the prior/previous, transition and likelihood factors and their event order. |
| Squared-TT nonnegative density and state marginalization | `source_faithful` | paper Algorithm 2(b-c), `:703-722`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:1-87` | Preserve squared-TT normalizer/marginal semantics; the local implementation still needs same-scalar verification. |
| Conditional KR proposal and importance correction | `source_faithful` | paper Algorithm 3, `:890-924`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100` | Conditional state generation and correction remain inherited mechanics; deterministic samples and identities are classified separately below. |
| Sequential normalizer increment `log(sirt.z) - const` | `source_faithful` | author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:84-124` | Preserve `log Z_t-c_t`; do not substitute APF log-sum-exp increments. |
| Frozen seeds, samples, ranks, frames, schedules and repository-issued identities | `fixed_hmc_adaptation` | author randomness at `full_sol.m:22-37,53-65,90-98`; local P90 binding contract | Freeze the inherited route for a deterministic differentiable finite program; this does not authorize HMC. |
| P88 TensorFlow `training_base_optimizer`, degree-3 comparator, L1 policy and validation/audit split | `extension_or_invention` | P88 artifact fields `classification=extension_or_invention`, `training_backend=training_base_optimizer`, and explicit `not source-faithful Zhao-Cui` nonclaim | This is the working local trainer family. Never replace it with author TT-cross/ALS while claiming baseline preservation. |
| Treat theta as an external conditioning input and integrate state only | `extension_or_invention` | this plan's outer-likelihood target; unlike paper Algorithm 2, which represents theta jointly | Required for `L(theta)`; it is not the paper's joint parameter-estimation program. |
| Centered child TT with an exact P88 origin slice | `extension_or_invention` | this plan | Must preserve the parent at theta zero; cannot close a source-faithfulness claim. |
| Manual total derivative through parameter cores, retained marginal and applicable transport/normalizer terms | `extension_or_invention` | this plan, checked against the exact local value program | Correctness is same-program derivative equality, not presence in the author implementation. |
| P76/P77 UKF geometry, if later tested | `fixed_hmc_adaptation` | separate P76/P77 experiment | Optional candidate A/B after origin preservation; not part of P88 and not likelihood evidence. |

Every Phase-0/1 implementation record must include this classification for each
touched operation plus its exact local callable and source anchor. A missing or
contradictory classification stops with `BLOCK_SOURCE_UNGROUNDED`. Source
anchors justify inherited operations; they never authorize a return to the
author random TT-cross/ALS trainer.

## Forbidden Drift

The active route must not:

- call the original-author random TT-cross/ALS estimator;
- use the July 30 `zhao_cui_austria_sir_source_replica_tf` path;
- use the July 30 frozen-proposal APF value as the canonical scalar;
- use the generic all-axes retained-grid evaluator;
- remove or add UKF while claiming unchanged baseline behavior;
- change the fixed-variant trainer family, target measure, event order,
  normalizer convention, or retained-object recursion without a new reviewed
  owner decision;
- integrate over theta or generate theta particles when computing an outer-
  parameter likelihood;
- run HMC before the value and total score are correct.

Source-anchor policy classifies inherited operations. It does not require a
return to the original-author estimator.

## Research Question Guardian

| Field | Contract |
|---|---|
| Main question | Can the existing P86/P88 BayesFilter fixed-variant TT training and retained-object filtering route be extended to the three-log-scale Austria parameter surface while preserving the fixed value at theta zero and computing the analytical total derivative of the same operational scalar? |
| Mechanism | Preserve the fixed state-space TT slice and add three external conditioning coordinates plus derivative carry through the existing normalizer and previous-marginal recursion. |
| Exact baseline | Repository-issued Phase-0 artifact binding the P88 cores to the actual T1 fixed-TTSIRT scalar, an independent author-formula replay, and the T2 previous-marginal input boundary. It does not claim P88 T2 cores. |
| Expected failure modes | P88 cores cannot be reloaded into the loop; fixed scalar has an invalid measure; active-data event order differs; parameter TT ranks grow; previous-marginal or normalizer derivatives are missing; T20 memory grows beyond budget. |
| Primary promotion criterion | At theta zero, the parameterized child program reproduces the fixed baseline value under predeclared dtype tolerances; its manual score matches the derivative of that same child program. |
| Promotion veto | Baseline mismatch, changed scalar, changed trainer, source-replica/APF/retained-grid substitution, missing derivative owner, theta integrated as a random axis, caller-stamped identity, runtime autodiff/FD score, or memory breach. |
| Continuation veto | Baseline is not reconstructible; baseline scalar is mathematically invalid under its claimed measure; parameter conditioning cannot preserve the estimator; or no tractable total derivative exists within the reviewed memory representation. |
| Repair trigger | A fit, rank, event-order, derivative, or backend failure under an unchanged target triggers a same-route repair, not an estimator switch. |
| Explanatory diagnostics | Fit/holdout residuals, UKF diagnostics if separately tested, ESS, correction spread, FD/autodiff residuals, rank, runtime, and memory. |
| Must not be concluded | No exact nonlinear likelihood, posterior correctness, HMC readiness, whole-route source faithfulness, statistical superiority, or production readiness follows from a mechanics pass. |

Every failure result must distinguish baseline reconstruction, target/measure,
implementation, tuning, derivative ownership, numerical validity, and resource
feasibility. A failed candidate is not automatically a rejection of the fixed-
variant direction.

## Target Contract

| Field | Required value |
|---|---|
| Parameterized row | `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` |
| Fixed parent row | `zhao_cui_spatial_sir_austria_j9_T20` |
| Comparison alias | `austria_sir_T20` |
| State | `(S1,I1,...,S9,I9)`, dimension 18 |
| Observation | infectious components, dimension 9 |
| Event order | draw `x0`; for `t=1,...,20`, transition, then observe `y_t` |
| Theta | `(log_kappa_scale,log_nu_scale,log_obs_noise_scale)` |
| Theta domain | reviewed diagnostic domain `[-0.5,0.5]^3` |
| Reference/truth | `(0,0,0)` |
| Origin identity | the parameterized model at zero equals the fixed Austria model |
| Source observation SHA-256 | `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07` |
| Runtime FP32 observation SHA-256 | `40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009` |
| Runtime backend | TensorFlow/TFP; no NumPy numerical runtime |
| Final execution | FP32, TF32, XLA, GPU memory growth after CPU reference gates |
| Score | manual analytical total derivative; autodiff/FD diagnostic only |

Phase 0 must bind the baseline state/measure convention explicitly. The author
clipped sampling push and an ordinary Gaussian transition density are not the
same probability measure. If the operational fixed scalar uses them together,
the artifact must describe it as the exact finite approximation it computes,
not as an exact observed-data likelihood. If no coherent operational measure
can be stated, stop with `BLOCK_FIXED_VARIANT_TARGET_MEASURE_INVALID`.

## Mathematical Program

### Fixed baseline

For each fixed-variant step, let `R_{t-1}^0` denote the exact compact retained
object carried from the previous step and let `phi_t^0` be the TT amplitude
represented by the serialized cores. The density is not assumed to be merely
the square of that amplitude. It is the exact configured defensive program

\[
  \rho_t^0(z;R_{t-1}^0)
  =\left[\phi_t^0(z;R_{t-1}^0)\right]^2
   +\tau_t^0\lambda_t^0(z),
\]

where the P88 artifact currently records `tau=1e-8` and a reference-measure
defensive term. Phase 0 must bind the actual `tau`, reference measure and local
normalizer implementation rather than simplifying them away. Let `Z_t^0` be
the integral of this exact density and `c_t^0` the recorded shift. The
deterministic fixed program has a retained-state update

\[
  R_t^0=\Phi_t^0(R_{t-1}^0),
\]

and operational value

\[
  L_0=\sum_{t=1}^{T}\left(\log Z_t^0-c_t^0\right).
\]

At `t>1`, `rho_t^0` includes the previous retained prefix marginal evaluated
from `R_{t-1}^0`. This dependency is part of the value and cannot be replaced
by an independent local density.

### Parameter-conditioned extension

Theta is an external input, not a latent random variable. Choose a centered
TensorFlow chart `u(theta)` with `u(0)=0` and construct a parametric TT
amplitude

\[
  \phi_t(u,z;R_{t-1})
  =\phi_t^0(z;R_{t-1})+C_t(u,z;R_{t-1}),
  \qquad C_t(0,z)=0.
\]

The parameterized density is the same family of finite program,

\[
  \rho_t(u,z;R_{t-1})
  =\left[\phi_t(u,z;R_{t-1})\right]^2
   +\tau_t(u)\lambda_t(u,z).
\]

At the origin, `tau_t(0)=tau_t^0` and
`lambda_t(0,z)=lambda_t^0(z)`. Prefer keeping both theta independent; any
theta dependence is a reviewed design choice whose value and derivative must
be included. Setting `tau` to zero, dropping `lambda`, or changing its measure
would change the estimator and is forbidden under baseline preservation.

The parent state cores and their hashes remain unchanged in the origin slice.
The new parameter cores and coupling channels form a repository-issued child
identity that binds:

- the P88 parent artifact and all parent state-core hashes;
- coordinate frame, basis, ranks, defensive mass, shift and normalizer policy;
- observation/event-order identity;
- previous retained parent hashes; and
- the new parameter-core hashes and training/tuning artifact.

For a fixed theta, evaluate parameter cores and integrate only state axes. The
child program carries its retained object recursively:

\[
  R_t(\theta)=\Phi_t\bigl(\theta,R_{t-1}(\theta)\bigr),
\]

\[
  Z_t(\theta)
  =\int \rho_t\bigl(u(\theta),z;R_{t-1}(\theta)\bigr)\,d\nu_z(z),
\]

\[
  L(\theta)
  =\sum_{t=1}^{T}
    \left[\log Z_t(\theta)-c_t(\theta)\right].
\]

A joint integral over theta and state is wrong. The conditional TTSIRT adapter
must bind theta as a fixed prefix and generate/evaluate only state coordinates.
It must not generate theta samples or introduce APF ancestors/normalizers.

The algebraic origin requirement is exact:

\[
  C_t(0,z)=0,
  \quad \tau_t(0)=\tau_t^0,
  \quad \lambda_t(0,z)=\lambda_t^0(z),
  \qquad L(0)=L_0.
\]

Numerical origin comparisons use predeclared FP64 and FP32 tolerances because
adding identity-core contractions may change floating-point operation order.

The frozen parametric TT is compiled offline over the theta domain. At runtime,
theta changes parameter-core evaluation, not optimizer state or fitted core
coefficients. The result is a declared parametric fixed-variant approximation;
it is not claimed to equal refitting a separate TT at every theta or the exact
nonlinear likelihood.

### Total score

For parameter `a`, use `D_a` for the total derivative of the deterministic
finite program, not a local partial derivative:

\[
  D_aL(\theta)
  =\sum_{t=1}^{T}
   \left[
     \frac{D_aZ_t(\theta)}{Z_t(\theta)}
     -D_ac_t(\theta)
   \right].
\]

Writing `S_{t,a}=D_aR_t`, the recursive derivative ownership is

\[
  S_{t,a}
  =\partial_a\Phi_t
   +D_R\Phi_t\,S_{t-1,a},
\]

\[
  D_a Z_t
  =\partial_a Z_t
   +D_R Z_t\,S_{t-1,a}.
\]

Here the partial terms hold the incoming retained object fixed; the second
terms carry all dependence inherited from earlier observations. An
implementation that returns only `partial_a Z_t`, or only transition and
observation component scores, is wrong relative to the claimed total score.

The derivative-owner ledger must resolve:

- transition-density parameter score;
- observation-density parameter score;
- zero initial-density parameter score;
- parameter-core basis evaluation and all parameter/state coupling channels;
- squared-TT state normalizer;
- shift, chart, affine, `tau` and defensive-reference terms when theta
  dependent;
- the carried previous retained marginal at every `t>1`;
- conditional/proposal/transport terms only where they enter the exact value
  program; and
- child/parent branch identity and retained-object lineage.

Frozen coefficients do not imply a zero score: the frozen parametric TT slice
still depends on theta through its parameter cores. Optimizer-iteration
derivatives are not part of this declared runtime scalar.

The P90/P91 local score carry is reusable, but these blockers remain until the
new scalar gives them an implementation or audited not-applicable proof:

```text
BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED
BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED
```

They may not be erased by using an APF or reporting a local complete-data score.

## Memory Contract

The representation must remain TT-linear and horizon-streaming:

\[
  M_{cores}
  =O\!\left(\sum_{k=1}^{39}r_{k-1}b_kr_k\right).
\]

At most the following may be live together:

- one training/calibration batch and one bounded validation batch;
- one current conditional transport/microbatch;
- the current and immediately previous retained objects;
- compact parameter-core derivatives for three parameters; and
- compact immutable core/frame/manifests for completed time steps.

Forbidden allocations:

- tensor-product grids across theta and state;
- a runtime dense `B^3` theta grid;
- full retained particle/training histories for all time steps;
- full Jacobians over samples, grid points, axes and parameters;
- the generic all-axes retained-grid transition; and
- an APF particle history introduced to avoid the existing derivative owners.

Parameter cores are evaluated at theta before state contraction. Time steps and
conditional inverse operations are microbatched only when doing so preserves
the exact fixed-variant operation and identity.

Initial planning caps, to be revised only from measured Phase-0/1 evidence:

| Resource | Cap |
|---|---|
| CPU reference process | 12 GiB peak |
| One conditional microbatch | 512 MiB estimated live tensors |
| Concurrent trainers | one |
| GPU allocation | verified memory growth; no whole-device preallocation |
| Stored step payload | compact cores/maps/manifests, not clouds/grids |

## Anti-Drift Guards

The guards are cumulative. Phase 0 must implement and pass guards 1-5. Each
later guard must be implemented before the phase that first creates the
relevant surface; all 13 must pass before Phase 3 training or any claim run.
The required guards are:

1. The baseline factory loads the exact P88 artifact and hashes its 36 cores.
2. The baseline uses the recorded training-base family and contains no ALS,
   source-replica, APF, or retained-grid dependency.
3. The P88 cores can be consumed by the actual fixed-TTSIRT/retained-object
   interfaces, or Phase 0 stops instead of synthesizing a replacement.
4. The value schema is the transport-normalizer scalar.
5. The event order is `x0 -> transition -> y1`; no `y0` slot shifting occurs.
6. The parameterized model at zero equals the fixed model on deterministic
   transition/density/observation fixtures.
7. The child identity binds, but does not impersonate, the fixed parent branch,
   state cores and retained lineage.
8. Theta cores are evaluated as conditioning inputs and only state axes are
   integrated.
9. The conditional adapter generates only state coordinates and preserves the
   previous-marginal and normalizer semantics.
10. Parent state-core hashes, defensive `tau` and defensive reference measure
    remain unchanged at the origin slice.
11. Origin value tolerances are declared before evaluation.
12. Score admission fails for any missing derivative owner or runtime
    autodiff/FD dependency.
13. No HMC entry point can consume the route before the T20 value/score gate.

Any change of estimator family, scalar, target measure, trainer, parameter
semantics, or baseline identity requires an owner-approved plan revision.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| P88 order-3/rank-4/zero-L1 cores | strongest serialized fixed-variant artifact | historical baseline only | artifact not compatible with sequential transport | Phase 0 load/wire replay |
| Training-base optimizer | P86/P88 working route | mandatory trainer family | helper drift changes objective | callable/dependency identity guard |
| L1 tuning | current Zhao-Cui lane policy | mandatory for new scope; zero-L1 remains baseline comparator | transferred L1 value harms active target | target-specific calibration/validation |
| T1/T2 retained loop | P57/P59/P90 | working mechanics/value bridge | P88 artifact never integrated and supplies T1 cores only | Phase 0 P88 T1 execution plus T2-boundary compatibility |
| Active observation timing | current target | required | older data used initial observation or different seed | Phase 1 event-order/hash test |
| Parameter chart/basis | not selected | Phase 2 design choice | origin slice not exact or corner instability | centered-basis algebra test |
| Parameter TT conditioning | project extension | primary design | ranks grow or theta accidentally integrated | T1 partial-contraction tests |
| P76/P77 UKF warm start | separate experiment | optional later A/B only, not baseline | splicing it in changes baseline | excluded before baseline/origin gates |
| Theta domain `[-0.5,0.5]^3` | reviewed July 2 contract | diagnostic domain | nonfinite/capacity failure at corners | truth plus eight corners |
| T20 | target contract | final horizon | cumulative rank/memory growth | T1/T2/T5/T10/T20 ladder |
| FP32/TF32/XLA | repository policy | final backend | normalizer/score cancellation | FP64 CPU tie-out first |

No inherited degree, rank, LR, L1 value, basis, batch size, or stopping rule is
a new active-data T20 default. P88 settings define the historical baseline;
new claim settings require scope-specific tuning with disjoint calibration,
validation and audit data.

## Baseline Ladder

1. Exact P88 serialized fixed-variant density artifact.
2. Exact P88 artifact executed as T1 fixed-TTSIRT and independently replayed.
3. Exact P88 T1 retained object accepted by the T2 previous-marginal target
   boundary, without claiming T2 trained cores.
4. Same training-base family fitted for T1/T2 on sealed active observations.
5. Three conditioning axes with zero correction and exact origin slice.
6. Trained parameter correction with the parent slice locked.
7. Complete manual total score of the same parameterized scalar.
8. Streaming T1/T2/T5/T10/T20 execution.

Original TT-cross/ALS, source replica, frozen-proposal APF, and retained grid are
not ladder arms. P76/P77 UKF may become a separate warm-start A/B only after the
baseline and origin slice pass and only under a reviewed plan that does not
replace the baseline.

## Execution Phases

Phase 0 has executed and stopped at its declared blocker. No later phase is
active. A future revision may reopen Phase 0 only with the missing exact
transport/retained identity or an explicit owner decision selecting a new
baseline. Later serious training and GPU phases still require concise execution
notes with exact commands, versioned artifact roots, compute budgets, and stop
conditions.

### Phase 0: Fixed-Variant Baseline Freeze

1. Load the exact P88 JSON; verify its SHA-256, Git provenance, status, 36
   serialized cores, basis, rank, trainer, seeds, target id, measure and
   dependency closure. Record that it has no July 30 observation-hash binding;
   do not infer one.
2. Locate or implement only the minimal loader needed to reconstruct those
   exact cores as the existing `SquaredTTDensity`/`FixedTTSIRTTransport`.
3. Bind that transport to the existing T1 retained-object step without changing
   cores, frame, basis, density, correction, normalizer or event order.
4. Run the independent P90 author-formula comparator architecture on that exact
   P88 T1 branch. Do not cite the prior P90 T2 fixture as a P88 bridge pass.
5. Carry the resulting P88 T1 retained object into the existing T2
   previous-marginal target assembly and verify identity, axes, measure and
   callable compatibility. Do not fit or synthesize a T2 transport.
6. Implement and pass anti-drift guards 1-5. Record guards 6-13 as fail-closed
   admission requirements for their owning phases. Record source-relation
   classifications for every touched operation.
7. Write a Phase-0 baseline artifact describing exactly what is working and
   exactly which integration surfaces were newly connected.

Exit statuses:

- `PASS_FIXED_VARIANT_BASELINE_FROZEN`, only if exact P88 cores run through the
  fixed T1 scalar, match an independent same-branch author-formula replay, and
  produce a retained object accepted by the T2 target boundary; or
- `BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE`, with no replacement.

No training, UKF insertion, parameter extension, GPU run, or HMC occurs here.

### Phase 1: Active-Data Fixed Baseline

1. Keep the Phase-0 algorithm and callable graph frozen.
2. Bind the sealed observation tensor/hash and transition-before-observation
   event order.
3. Audit the fixed state/measure convention. Stop if it cannot define a coherent
   operational scalar.
4. Apply the same fixed-variant training algorithm to observation-specific T1
   and T2 targets. This is the first phase allowed to train T2 cores. First run
   an inherited-setting transfer diagnostic; if it
   fails proposal/fit gates, tune within the same training-base family using
   fresh calibration/validation/audit splits and L1 policy.
5. Record actual memory and forecast the horizon ladder.

Exit: active-data fixed T2 value artifact with parent identity, or a precise
target/measure/tuning/resource blocker. No estimator switch is allowed.

### Phase 2: Parameter-Conditioning Design And Zero Slice

1. Select a centered TensorFlow parameter chart and basis.
2. Add three parameter cores and constrained correction channels satisfying
   `C_t(0,z)=0`; do not mutate parent state cores.
3. Implement partial contraction: evaluate theta cores, integrate state cores.
4. Implement the fixed-theta conditional adapter over the retained-object
   recursion. It generates/evaluates only state coordinates.
5. Issue child identities binding all fixed parent hashes.
6. Declare FP64/FP32 tolerances before evaluation, then prove state density,
   normalizer, retained marginal and value equality at theta zero for T1/T2.
7. Add negative tests for joint theta/state normalization, theta-particle
   generation, caller-stamped parent identity and parent-core mutation.

Exit: exact algebraic zero slice and numerical origin-value regression pass
before any parameter correction training.

### Phase 3: Scope-Specific Parameter TT Training

1. Train only new parameter and reviewed coupling channels with the parent
   origin slice constrained.
2. Use TensorFlow batch-native target evaluation across theta and state; no
   scalar row-mapped training fallback.
3. Tune rank, degree, LR, L1, regularization, batch budget and stopping policy
   for this exact target/horizon scope with disjoint splits.
4. Require both target-density validation and downstream fixed-filter checks;
   heldout loss alone cannot promote a candidate.
5. Freeze a repository-issued tuning artifact before untouched evaluation.
6. Check truth plus all eight theta-domain corners for finite slices,
   normalizers and values.

Exit: trained child artifact that still passes every origin-slice invariant.

### Phase 4: Complete Manual Total Score

1. Reuse the existing SIR transition and observation score hooks.
2. Reuse P90 deterministic local derivative carry.
3. Implement parameter-core and state-normalizer derivatives.
4. Implement the carried previous-marginal derivative at T2 and beyond.
5. Resolve chart, affine, shift, defensive and conditional/proposal/transport
   derivative owners for the exact Phase-3 scalar.
6. Bind value and score to identical child/parent branches, cores, retained
   lineage, target and tuning artifact.
7. Compare against central FD and GradientTape only as diagnostics at T1/T2 and
   representative theta points.

Exit: every owner implemented or proved theta independent; manual runtime score
matches the same scalar diagnostics and contains no autodiff/FD.

### Phase 5: Streaming Horizon Ladder

Run the unchanged selected program at `T=1,2,5,10,20`, one time-step trainer/
transport at a time. Record origin value regression, score diagnostics, ranks,
core bytes, previous-marginal identities, normalizer increments, fit/proposal
diagnostics, wall time and peak memory.

Stop on target, derivative, identity, numerical or memory vetoes. A poor fit
triggers same-family tuning. It never authorizes APF, original estimation, or
retained-grid fallback.

### Phase 6: GPU/XLA Value And Score Artifact

After CPU reference gates pass, run the exact T20 program using TensorFlow FP32,
TF32, XLA, trusted GPU access and verified memory growth. Record the required
serious-run manifest. Do not run HMC.

Exit: finite T20 fixed-variant value, three-coordinate manual score, origin
regression, identities and bounded memory; otherwise a precise backend/resource
blocker.

### Phase 7: Same-Target Comparison And Closeout

Only after Zhao-Cui value/score validity, compare with GenUT, SGQF and UKF on
the identical target and observations. Report hard vetoes first and treat
continuous differences as descriptive unless uncertainty evidence supports a
ranking. Write the result, decision table, inference-status table, post-run red
team and reset memo.

HMC requires a later plan and is outside this program.

## Evidence Ledgers

### Baseline preservation

Tracks the P88 parent artifact, callable graph, state cores, basis/frame,
retained lineage, event order and operational fixed value. No downstream metric
can excuse failure here.

### Parameter-extension correctness

Tracks theta-conditioning semantics, origin slice, child identity, state-only
normalization and the complete same-scalar manual score. Training loss, ESS,
GPU speed and HMC behavior cannot rescue a failed correctness gate.

### Scale/backend

Tracks T20 streaming, memory forecasts/measurements, dtype, XLA, GPU device and
memory-growth provenance. It cannot promote mathematical evidence missing from
the first two ledgers.

## Stop And Repair Matrix

| Failure | Classification | Required action | Forbidden reaction |
|---|---|---|---|
| P88 cores cannot be reconstructed or wired | baseline reconstruction | stop and identify missing loader/metadata/interface | synthesize new cores or call P76/P88 inspiration sufficient |
| Fixed scalar has incoherent measure | target/measure | stop or obtain owner-approved target correction | hide mismatch with “approximate” language |
| Active observations degrade fit | transfer/tuning | tune same training-base family with fresh splits | switch estimator or original TT-cross/ALS |
| Origin slice differs | implementation/identity | repair conditional core embedding | refit/mutate parent slice or relax equality post hoc |
| Previous-marginal derivative missing | derivative ownership | implement on existing retained object | substitute APF or local score |
| Parameter ranks exceed memory | representation/resource | revise TT coupling/rank policy within fixed variant | dense theta grid or retained grid |
| Manual score fails diagnostic | correctness | localize missing owner | run HMC or report only finite value |
| GPU/XLA fails after CPU pass | backend | repair unchanged program within budget | change scalar or method |

## Skeptical Plan Audit

| Risk | Finding and correction |
|---|---|
| Wrong baseline | Earlier drafts incorrectly composed P76 UKF training with P88/source-route components. This plan starts from the exact serialized P88 artifact and proves its actual call graph before extension. |
| Proxy promotion | Rank/degree stability, fit loss, UKF moments, ESS, FD and T1/T2 are explicitly scoped; none establishes T20 value/score correctness. |
| Hidden integration gap | P88 is a T1 fit; retained-loop and P90 T2 fixtures are separate evidence. Phase 0 proves exact P88 T1 execution and T2-boundary compatibility, not nonexistent P88 T2 cores. |
| Estimator drift | Original TT-cross/ALS, source replica, APF and retained grid are forbidden, with static/runtime guards. |
| Source-claim drift | Every inherited or new operation has an exact `source_faithful`, `fixed_hmc_adaptation`, or `extension_or_invention` classification with paper/author anchors where applicable. |
| Parameter semantic drift | Theta is an external conditioning input; only state axes are integrated and theta particles are forbidden. |
| Defensive-density drift | The child extends the P88 TT amplitude while preserving `phi^2 + tau*lambda`; it does not silently discard the defensive density. |
| Event-order drift | Active `x0 -> x1 -> y1` timing is bound before active-data fitting. |
| Measure mismatch | Clipped push versus Gaussian-density semantics is a hard audit item, not softened into correctness. |
| Derivative mismatch | The plan uses a total derivative through retained state; previous marginal and relevant TTSIRT owners stay blocked until implemented for the exact scalar. |
| Memory blow-up | Parameter/state grids, histories and full Jacobians are prohibited; time and conditional operations stream under preallocation caps. |
| Non-answering execution | No serious training before baseline freeze, no GPU before total score, and no HMC in this plan. |

Audit verdict: `PASS_FOR_PHASE0_ONLY`.

The audit repaired five material flaws in the earlier direction: it had treated
separate P76/P77/P88/P90 components as one existing callable, treated UKF as
proved P88 baseline behavior, described the defensive density as a pure square,
left room for a local partial score to be called total, and treated a P88 T1
artifact plus a separate P90 T2 fixture as a P88 T1/T2 pipeline. Phase 0 now
proves exact P88 T1 execution and the T2 input boundary, then stops. Later
phases preserve `phi^2 + tau*lambda` and carry total derivatives through
retained state.

## Definition Of Done

The program is complete only when:

1. exact P88 cores are reconstructible, execute through the fixed T1 scalar,
   pass same-branch replay and produce a valid T2 previous-marginal input;
2. the fixed active-data route reaches T20 with the same estimator;
3. the parameter child preserves the parent slice at theta zero;
4. theta remains an external conditioning input with state-only normalization;
5. the manual total score differentiates the same operational scalar, including
   previous-marginal and every applicable normalizer/transport owner;
6. the horizon ladder streams within reviewed memory caps;
7. the T20 GPU/XLA artifact records value, score, identity and memory growth;
8. all nonclaims are explicit and no HMC is run.

Otherwise the closeout names the first unsatisfied baseline, target, tuning,
derivative, numerical or resource gate. Completion through a different
estimator is forbidden.
