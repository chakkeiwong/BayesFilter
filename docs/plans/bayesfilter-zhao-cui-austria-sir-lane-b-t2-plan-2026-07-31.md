# Zhao-Cui Austria SIR Lane-B T2 Fixed-Value Plan

Date: 2026-07-31

Status: `ACTIVE_EXECUTION`

## Entry Evidence

- T1 value: `PASS_NEW_FIXED_VARIANT_T1_VALUE_BASELINE`.
- B2 sampler: `PASS_B2_RETAINED_SAMPLER_ADMISSION`.
- B3 marginal boundary: `PASS_T2_PREVIOUS_MARGINAL_BOUNDARY`.
- T1 artifact identity:
  `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`.

No T2 core or value is inherited. T1 settings are warm starts only.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can a separately tuned T2 squared-TT object reproduce the finite target `p1_TT(z1) f(z2|z1) g(y2|z2)`, reload deterministically, and add the correct independently calibrated value increment? |
| Candidate | Existing batch-native TensorFlow training-base family, T2 order `[z2,z1]`, retained axes `0:18`, proposal `z1~q_grid`, `z2~f`, with exact `log(p1_TT/q_grid)` correction. |
| Expected failure | Proposal correction omitted; T1 settings transfer poorly; shape collapses; scale is inferred from loss; T2 marginal or normalizer disagrees; non-finite values; or resource cap breach. |
| Promotion criterion | Frozen validation selection, fresh T1/T2 reload, direct normalizer equality, untouched independently estimated increment agreement under the predeclared uncertainty rule, deterministic T1-to-T2 lineage, and memory compliance. |
| Promotion veto | Wrong event/axis order, different T1 identity, wrong proposal density, missing correction, calibration/validation/untouched reuse, caller-stamped identity, validation metric substituted for value, non-finite result, or memory breach. |
| Continuation veto | The target/measure is incoherent, B2/B3 evidence is invalid, or no compact finite T2 representation can pass same-scalar value checks within the campaign budget. |
| Repair trigger | Prepared-cloud, serialization, training, XLA, or selected-candidate failure under the unchanged target; or insufficient current candidate capacity. |
| Explanatory diagnostics | Validation log-shape RMS, correction ESS/range, training trace, rank, runtime, and memory. |
| Must not be concluded | No exact nonlinear likelihood, score, T20, HMC, production KR, posterior correctness, source-faithful assembled route, or superiority. |

## Mathematical Program

Let the admitted normalized T1 retained density be `p1_TT(z1)` and its finite
sampler law be `q_grid(z1)`. Draw

\[
 z_1\sim q_{grid},\qquad z_2\sim f(\cdot\mid z_1,\theta=0).
\]

The unnormalized T2 target is

\[
 \gamma_2(z_2,z_1)=p_{1,TT}(z_1)f(z_2\mid z_1)g(y_2\mid z_2).
\]

Relative to the proposal, its exact weight is

\[
 \log w_2=\log p_{1,TT}(z_1)-\log q_{grid}(z_1)
          +\log g(y_2\mid z_2).
\]

The transition cancels but remains in the target and proposal ledgers. The T2
increment is independently estimated as

\[
 \Delta L_2=\log E_{q_{grid}f}[w_2].
\]

The shift is fixed from the calibration cloud as `c2=-Delta L2_cal`, making
the calibrated shifted mass one. Cross-entropy establishes shape only. After
training, one core is rescaled so the shifted TT mass is exactly one. The
operational cumulative value is

\[
 L_2=L_1+(\log Z_2-c_2)=L_1-c_2.
\]

The physical-to-reference conversion uses the same uniform probability
measure and mandatory `36 log 2` factor as T1, now in `[z2,z1]` order.

For T2, evaluate the empirical cross-entropy measure directly in log space as
`softmax(log w2)`. The T1 factorization into a square-root target and an
integration weight is equivalent in exact arithmetic, but it is numerically
invalid for T2 tails because exponentiating the square-root target can
underflow. No sample may be clipped or dropped. The log-weight kernel changes
only the numerical factorization of the same normalized empirical measure.

## Source And Classification Ledger

| Operation | Classification | Anchor |
|---|---|---|
| T2 target `previous * transition * likelihood` | `source_faithful` operation | Zhao-Cui Algorithm 2(a), Eq. (15); author `models/full_sol.m:72-80,132-135` |
| Squared-TT reapproximation and right-block marginal | `source_faithful` operation | paper Algorithm 2(b-c); author `full_sol.m:101-124`, `@TTSIRT/marginalise.m:19-85` |
| T1 finite sampler and exact correction | `extension_or_invention` with source-supported proposal role | B2 result; paper Eq. (20)-(23), Algorithm 3 |
| Training-base optimizer, MC scale anchor, L1 tuning, serialization | `extension_or_invention` | BayesFilter fixed-variant baseline |
| Frozen clouds, ranks, schedules, identities | `fixed_hmc_adaptation` | freezes author randomness; no HMC authorization |

## Prepared Data And Separation

Generate once in one batch-native TensorFlow CPU process and serialize. This is
a reviewed exception to process-level multicore sharding: the B2 diagnostic
measured about 700 MiB per TensorFlow process, so multiple workers would
duplicate the full T1 TT/basis state and increase memory/contention without
changing the vectorized 64-column contraction kernel. Record worker count one,
chunk size, and CPU-hidden posture in every artifact.

| Role | Count | Reference seed | Transition seed | Use |
|---|---:|---:|---:|---|
| training/frame | 4,096 | 73801 | 73811 | fitting and frame only |
| validation | 8,192 | 73802 | 73812 | hard gates and deterministic selection |
| scale calibration | 12,288 | 73803 | 73813 | shift and post-fit scale only |
| untouched claim | 16,384 | 73804 | 73814 | generated only after selection freezes |

Prepared artifacts bind reference arrays, `z1`, `z2`, proposal, target,
transition, likelihood, correction, target identity, worker count, TensorFlow
version, and hashes. Sample generation is CPU work; GPU training begins only
from sealed prepared tensors.

## Frozen Pilot Arms

All arms use microbatch size 256, 96 full-cloud Adam updates, expansion factor 4,
covariance jitter `1e-5`, quantile fraction `0.01`, `tau=1e-8`, L2 `1e-8`,
and gradient cap 100.

| Arm | Rank | Order/elements | LR | L1 | Status |
|---|---:|---:|---:|---:|---|
| `t2_p01_r2_b3_lr3e4_l1_0` | 2 | 1/2 | `3e-4` | `0` | required comparator |
| `t2_p02_r2_b3_lr3e4_l1_1e8` | 2 | 1/2 | `3e-4` | `1e-8` | hypothesis |
| `t2_p03_r4_b3_lr3e4_l1_1e9` | 4 | 1/2 | `3e-4` | `1e-9` | warm start |
| `t2_p04_r4_b3_lr3e4_l1_1e8` | 4 | 1/2 | `3e-4` | `1e-8` | hypothesis |
| `t2_p05_r4_b5_lr3e4_l1_1e9` | 4 | 2/2 | `3e-4` | `1e-9` | T1 warm start |
| `t2_p06_r4_b5_lr1e4_l1_1e9` | 4 | 2/2 | `1e-4` | `1e-9` | lower-LR hypothesis |

An arm is viable only if training, validation, independent normalizer,
identity, proposal correction, XLA, and memory gates pass and validation
normalized-log-density RMS is at most `0.95` times the constant-density
baseline. Select minimum validation RMS with arm id as tie-breaker. This is a
deterministic validation rule, not statistically supported superiority.

## Untouched Value Gate

The selected artifact alone reads the untouched cloud. Let `se_cal` and
`se_audit` be delta-method log standard errors. Require

\[
 |(\log Z_2-c_2)-\widehat{\Delta L}_{2,audit}|
 \le 3\sqrt{se_{cal}^2+se_{audit}^2}+10^{-6}.
\]

Also require:

- exact fresh T1/T2 identity reload;
- direct TT mass versus serialized `log Z2` residual
  `<=1e-9*(1+abs(Delta L2))`;
- cumulative value equals `L1 + Delta L2_TT` under the same tolerance;
- deterministic retained lineage and source hashes;
- peak GPU allocation below 6 GiB with verified memory growth.

Exit: `PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE`.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| Training-base family | admitted T1 family; required baseline | optimizer fails on T2 shape | tiny eager/XLA parity and pilot trace |
| Six arms | T1 ladder; warm starts/hypotheses only | T2 needs different capacity | validation screen; failure triggers reviewed expansion |
| Counts | refreshed from non-campaign 64/256/512 ladder | poor shape or excessive MC error | require training/validation ESS >=100 and calibration ESS >=200 before pilots |
| Batch 256 | reduced from T1 for bounded T2 scope | noisy updates or wasted GPU | one-update parity and allocator peak |
| Exact B2 proposal | admitted finite law | correction omitted or tails unstable | algebraic correction equality and cloud ESS |
| MC absolute scale | same-scalar derivation | scale-correct weak shape passes value | separate validation shape veto |
| CPU generation / GPU training | repository device policy | cross-device tensor drift | hashes and shared TensorFlow dtype/version |

## Skeptical Pre-Execution Audit

| Risk | Finding/correction |
|---|---|
| Wrong baseline | Requires exact selected T1 and B2/B3 evidence; P88, APF, UKF, source replica, and retained-grid routes are excluded. |
| Proxy promotion | Training/validation shape can veto/select but cannot pass value. Untouched same-scalar increment is primary. |
| Hidden assumption | Proposal is `q_grid*f`; exact `p1/q_grid` correction is carried and the transition cancellation is checked rather than assumed. |
| Stale setting transfer | Every T1 setting is labelled warm start or hypothesis; T2 has its own six-arm scope. |
| Missing stop condition | Invalid target/measure/identity, no viable compact arm, untouched failure, and memory breach are explicit stops or repairs. |
| Unfair comparison | No GenUT/SGQF/UKF comparison occurs before T2 admission. |
| Environment mismatch | Prepared clouds are explicit CPU artifacts; serious training is GPU/XLA with verified memory growth. |
| Non-answering command | Prepared-data hashes answer lineage; pilots answer capacity; untouched estimate answers the same scalar. |

Audit verdict: `PASS_FOR_EXECUTION`. The plan fixes the T2 scale target and
proposal law before implementation. No T2 validation or untouched data have
been generated.

## Prepared-Count Refresh

A diagnostic-only ladder using seeds outside the campaign ledger found:

| Count | ESS | Log-SE | Log increment |
|---:|---:|---:|---:|
| 64 | `3.25` | `0.545` | `-35.723` |
| 256 | `17.70` | `0.230` | `-35.600` |
| 512 | `20.15` | `0.219` | `-35.488` |

The original 1,024/2,048 counts were therefore underpowered. The refreshed
4,096/4,096/8,192/16,384 ladder is a target-specific budget hypothesis based
on the observed information rate, not a claim-data adjustment. Expected ESS is
approximately 160/160/320/640. Training and validation clouds must each attain
ESS at least 100; calibration must attain ESS at least 200. Failure triggers a
proposal/count repair before pilots. Untouched seed 73804 remains unread.

The first 4,096-sample training preparation passed ESS (`120.10`) and memory
gates but exposed two finite tail weights below `-10^12`, with a minimum about
`-4.83e18`. The inherited square-root batch underflowed and failed its positive
target assertion. This is a numerical-factorization repair trigger. The
refreshed log-weight kernel above is required before validation preparation or
GPU pilots. It preserves every finite sample and the exact empirical
cross-entropy law. Refreshed audit verdict: `PASS_FOR_EXECUTION`.

The pre-pilot audit also found that applying `softmax` independently inside
each 256-row microbatch would change the empirical target measure. Freeze the
global training `logsumexp(log w2)`. For `M=N/B` equal-sized deterministic
microbatches, each update uses

\[
  -M\sum_{i\in batch}\exp(\log w_i-\operatorname{LSE}_{all})\log\rho_i
  +\log Z+R.
\]

A complete deterministic cycle evaluates all 16 microbatches at unchanged
parameters. Raw gradients, including the normalizer and regularization terms,
are averaged over that cycle, clipped once, and followed by exactly one Adam
update. This is one full-cloud update. Sequential microbatch Adam updates are
forbidden because they do not equal the gradient of the frozen full-cloud
objective. The repair must pass variable-by-variable full-objective versus
accumulated-microbatch gradient equality before any pilot. Because the T2
module source closure changes, all prepared roles must be regenerated under the
final closure; existing prepared artifacts remain diagnostic negative/repair
evidence and are not silently upgraded.

The first validation preparation then encountered an observation log density
of `-inf` for an extreme but finite proposal row. For a Gaussian target this is
the FP64 representation of an effectively zero density; dropping the row would
change the proposal measure and clipping the residual would change the target.
The refreshed finite-program rule accepts negative infinity only, binds an
explicit zero-density mask, assigns those rows exactly zero normalized weight
through log-sum-exp/softmax semantics, and rejects `NaN` or positive infinity.
Shape diagnostics mask zero-weight rows before multiplying by squared log
residuals, avoiding the undefined numerical product `0*inf`. This repair does
not establish a parameter score at those rows; the later score phase must prove
their zero contribution or introduce a separately derived wider stable
algebra. Refreshed audit verdict: `PASS_FOR_EXECUTION`.

The repaired 4,096-row validation artifact passed target, lineage, extended-
real, memory, and count gates but had ESS `96.26`, below the frozen `100`
threshold. Its increment `-35.1813` with log-SE `0.1007` is consistent with the
training estimate `-35.2061` with log-SE `0.0899`; the failure is insufficient
effective count, not target disagreement. Keep the threshold fixed and refresh
validation to 8,192 rows. Refresh calibration to 12,288 rows so its ESS-200
gate is not a borderline extrapolation from the observed validation rate. This
uses the same role-specific streams and remains inside the three-CPU-hour B4
cap. Untouched seed 73804 remains unread. Refreshed audit verdict:
`PASS_FOR_EXECUTION`.

The full-cloud accumulation implementation and cloned-trainer Adam parity test
pass. The prepared-cloud loader now fails closed unless its sealed hashes for
the T2 program, admitted sampler, preparation program, and this frozen plan all
equal the current files. This plan is frozen for B4 execution; later run facts
belong in a separate result note. All earlier prepared clouds are stale by this
gate and must be regenerated in fresh directories. Refreshed audit verdict:
`PASS_FOR_EXECUTION`.

## Budget And Stop Conditions

- CPU prepared-data generation: 90 minutes plus one localized repair;
- CPU engineering tests: four launches, 20 minutes total;
- six GPU pilot arms: ten minutes each;
- one selection process and one untouched claim, 45 minutes maximum;
- total B4 cap: four GPU-hours and three CPU-hours, within the remaining
  approved T1/T2 campaign cap;
- outputs under
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/attempt-NN/`;
- no HMC, parameter training, score, T20, or comparator campaign.
