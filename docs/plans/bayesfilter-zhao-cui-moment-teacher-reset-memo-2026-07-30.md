# Reset Memo: Zhao-Cui Moment Teacher

Date: 2026-07-30
Lane: Zhao-Cui squared-TT moment teacher only
Status: LGSSM, predator-prey, and actual SV pass; Austria TT tuning blocks all-model completion

## 2026-07-31 Restart Bootstrap

The canonical particle/OT/Contract-E-Chol moment-teacher integration and its
same-program analytical score are implemented. Do not restart from the old
"canonical particle integration is next" instruction below; that section is
historical context.

Current result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-result-2026-07-31.md`.

Actual-SV result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-actual-sv-campaign-result-2026-07-31.md`.

Current route remains FP32, TF32 disabled, trusted GPU, XLA, verified memory
growth. Particle weights own the likelihood. The TT scale/normalizer never
enters that scalar. Canonical reset identity remains `contract_e_chol_v1` with
total direct moment/weight and streaming-transport derivatives.

Verified current state:

- final focused deterministic suite: 52 passed;
- integrated GPU/XLA parity gate: pass;
- LGSSM `T=2,10,50`, six seeds per horizon: all hard validity gates pass;
- every LGSSM value/score error series has mixed signs; maximum absolute mean
  error is 2.299 MCSE;
- predator-prey `T=20`, `N=1024`: one-seed descriptive feasibility pass; and
- exact transformed SV `T=20`, `N=1024`, six claim seeds: all hard validity
  gates pass after scope-specific terminal-balance tuning; value and `log_beta`
  are descriptively closer to the dense reference, `z_gamma` is descriptively
  farther, and no paired gain is statistically supported; and
- latent-preclip Austria SIR `T=20`, `N=1024`: offline TT tuning hard veto,
  with claim seed not evaluated.

Terminal nonlinear artifacts:

- predator-prey pass:
  `docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/nonlinear_predator_prey_t20_attempt02_terminal/result.json`,
  SHA-256 `94610c09f5448b29fa5f1bc29c0ce05e7f9c6bdd4c1313e3296074aebd45b4d4`;
- Austria tuning veto:
  `docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/nonlinear_austria_sir_t20_attempt03_terminal/result.json`,
  SHA-256 `ad58416a9162a937f6f795ddddfff04e4d52818bd5047e9d6c90def5c91b181a`.

Terminal actual-SV artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/gpu_attempt02_repair_balance/result.json`,
SHA-256 `18eddf1f47adce1e2c210e7263d8b18aa26f31ace38132881283c2c24f88ddfa`.
The selected tuning artifact uses `balance_steps=20`; do not transfer that
setting to another model, horizon, particle count, or backend. KSC-SV remains
untested by the integrated moment-teacher route and is a different surrogate
target, not another name for this exact transformed-SV result.

The Austria blocker is the current recursive squared-TT representation, not
the particle/OT/reset implementation. Target-preserving tests of two charts,
ranks 1/2, basis sizes 2/3, 96--192 rows, one/two sweeps, two ridges, and
scale-consistent defensive weights remain invalid before particle execution.
Do not waive this by adding per-time log centering without rederiving the
physical defensive coefficient; that diagnostic changes the represented
mixture and was removed.

Next smallest justified work, only under a new plan:

1. derive a target-preserving normalized or log-domain TT recursion whose
   scale convention and defensive coefficient remain exact in FP32;
2. test it first on the Austria setup-only teacher with value/JVP and covariance
   validity gates; and
3. only then repeat offline tuning and the untouched Austria claim.

No HMC, posterior, ranking, default, leaderboard, or source-faithful Zhao-Cui
claim is established. Actual-SV feasibility is established, but improvement is
not. Existing empirical-target Contract E remains unchanged.

## Historical Bootstrap (Superseded By The Section Above)

## Restart Bootstrap: Do This First

Proceed with particle/OT/Contract-E integration using only route
`zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1`. Attempts 09
and 10 executed the full fused teacher on the trusted RTX 4080 SUPER with FP32,
TF32, XLA, verified memory growth, finite recursion, and no host callbacks. Both
deterministically failed the predeclared same-program FP32/FP64 relative-parity
veto: maximum absolute error passed at \(1.170\times10^{-3}\), but maximum
relative error was \(0.6341>0.005\). The worst element was a near-zero carried-
marginal tangent, \(4.885\times10^{-6}\) versus the FP64 value
\(2.989\times10^{-6}\).

Attempt 11 disabled TF32 as an explicitly diagnostic comparison arm while
retaining FP32, GPU, and XLA. It passed with maximum absolute error
\(5.531\times10^{-7}\) and maximum relative error \(0.001674\). This isolates
TF32 arithmetic as the active cause on the deterministic fixture. It was
diagnostic when executed; the later owner decision selected this execution
mode for the moment-teacher lane. Attempt 12 is the fresh selected-route
artifact.

TF32 is not selected for this lane. The 11.0% measured median time reduction
does not justify its observed parity failure and systematic score displacement.
Use trusted GPU, FP32 tensors, XLA, verified memory growth, and explicitly call
`tf.config.experimental.enable_tensor_float_32_execution(False)` before graph
execution. This is a route-specific reviewed exception, not a repository-wide
default change.

A downstream score/MCSE transfer diagnostic was also completed on the nearest
complete canonical Contract-E LGSSM score route (`T=2`, `N=1024`, 16 paired
seeds). It is not a moment-teacher score test because that complete finite
program is still unimplemented. Both TF32 and FP32-no-TF32 arms passed all
route-validity checks with identical prepared inputs. Under the predeclared
rule `abs(mean TF32 drift) / reference MCSE <= 0.1` for every score coordinate,
the transfer diagnostic failed: ratios were 0.159, 0.0995, 0.00753, 0.483, and
0.245 for `phi1`, `phi2`, `phi3`, `q_scale`, and `r_scale`. TF32 drift was below
one MCSE in every coordinate, but not an order of magnitude below MCSE. Do not
use generic MCSE scale to waive the current veto. Once the integrated moment-
teacher score exists, repeat the paired target-specific test at its own scope.

The follow-up transfer diagnostic at `N=4096` used eight paired seeds and the
required `K=2048`, `2 x 2` transport grid. The absolute displacements remained
nearly unchanged from `N=1024`, but the smaller reference MCSE increased the
worst ratio to 0.759 for `q_scale`. Four coordinates (`phi1`, `phi2`, `q_scale`,
and `r_scale`) were one-sided in all eight seeds with exact sign-test
`p=0.0078125` and mean displacement greater than twice the paired-difference
MCSE. `phi3` did not meet the systematic-displacement criterion. The proposed
0.5-MCSE practical screen therefore fails at `N=4096`; particle replication
does not average away the observed TF32 displacement.

Repeated `N=4096` performance timing found a median warm execution time of
3.588 seconds with TF32 versus 4.032 seconds without TF32: an 11.0% time
reduction and 12.4% throughput gain. This is faster but did not meet the
predeclared 20% condition for "a lot faster." The current TFP HMC wrapper uses
one `log_prob_and_grad` call to supply both target value and score, so MH
corrects leapfrog integration relative to that returned finite target; it does
not independently recompute a higher-precision acceptance energy that removes
TF32 value displacement. TF32 remains the owner-directed production target
direction, but this timing does not justify waiving the moment-teacher precision
gate.

Attempt 12 is the fresh selected-route artifact. It passes with maximum
absolute FP32/FP64 error `5.531e-7`, maximum relative error `0.001674`, finite
recursion, no host callbacks, XLA control flow, and verified memory growth. Its
result SHA-256 is
`80ab900a514f9703d878ba20d4500fee417c8dbb4da02fa6fce404988ec9cf88`.

## Current State

The algorithm and proofs are documented in
docs/chapters/ch32c_entropic_ot_sinkhorn.tex, section
sec:bf-eot-zhao-cui-moment-teacher. The composition is an
extension_or_invention, not a source-faithful Zhao-Cui filtering claim.

Implemented:

- bayesfilter/highdim/zhao_cui_moment_teacher.py
- bayesfilter/highdim/zhao_cui_moment_teacher_xla.py
- normalized paired-core observables, raw/affine moments, and manual JVPs;
- fixed square-root target and scale-consistent defensive-weight JVPs;
- sequential fixed-ALS value/JVP replay with current-design tangents and solve residuals;
- normalized carried-marginal value/JVP with both TT-core copies and quotient rule;
- non-frozen TT shape-target value/JVP through mean/covariance, Cholesky
  whitening, affine-form contractions, pair masks, and defensive scaling;
- reusable setup-static recursive teacher API with warm-start cores and
  carried-marginal tangents;
- named reference adapter from non-frozen TT targets into the existing
  Contract E higher-moment correction;
- generic dual-lane reference step composing a supplied particle increment,
  Sinkhorn OT, Contract E-Chol, and TT shape repair with centered-FD parity;
- frozen TT shape-target adapter;
- ordered co-skew and symmetric co-kurtosis masks;
- padded equal-rank graph-native XLA contraction;
- padded/masked graph-native fixed-ALS value/JVP with TensorFlow control flow;
- graph-native normalized marginal/JVP and two-step carried recursion;
- graph-native batched degree-four affine-moment automaton;
- graph-native TT mean/covariance, Cholesky-whitened skew/kurtosis, ordered
  co-skew, symmetric co-kurtosis, and all corresponding manual JVPs;
- fused graph-native TT time recursion emitting marginal and shape targets from
  the same fitted cores;
- explicit-target support in bayesfilter/highdim/higher_moment_contract_e.py; and
- focused tests and a source-bound GPU mechanics harness.

Not implemented:

- model-scale orchestration coupling particle adapters to the fused TT
  recursion through canonical streaming Contract E;
- the complete dual-lane particle/TT filtering step (particle likelihood,
  Sinkhorn OT, Contract E reset, and the TT lane in one graph);
- a total analytical score of that complete finite program; or
- nonlinear value/score experiments.

## Semantics That Must Not Drift

- Particle weights still determine the likelihood increment and Contract E
  mean/covariance targets.
- The TT normalizer must not replace the particle likelihood increment.
- Frozen target tangents are zero only for route
  zhao_cui_frozen_tt_shape_targets_diagnostic_v1; they are wrong relative to
  a recursively refitted teacher-score claim.
- Co-skew \(\mathbb E[z_i^2z_j]\) uses an ordered structural mask.
- Co-kurtosis \(\mathbb E[z_i^2z_j^2]\) uses a symmetric structural mask.
- Missing pair targets are masked out, never filled with a claimed zero.
- The empirical-target higher-moment Contract E path remains unchanged and no
  default was promoted.
- Lebesgue defensive marginals include the volume of integrated coordinates;
  reference-measure marginals are unchanged.

## Verified Evidence

- 26 focused moment-teacher/Contract E/ALS tests passed.
- 47 independent squared-TT/fixed-branch tests passed.
- 73 focused tests total for this lane.
- 9 additional graph-native candidate tests passed, including fused per-time
  recursion parity and concrete-graph inspection.
- GPU/TF32/XLA mechanics:
  docs/benchmarks/artifacts/zhao_cui_moment_teacher_20260730/attempt04/
- Result:
  docs/plans/bayesfilter-zhao-cui-moment-teacher-result-2026-07-30.md

The attempt 04 GPU artifact measures only the contraction primitive. It is not
the full-teacher FP32/TF32 gate requested above. It records source hashes
because the repository had concurrent uncommitted work.

The full-teacher GPU/XLA gate now has four executed results. Attempts 09 and 10
are reproducible TF32 failures under the declared relative-parity veto.
Attempt 11 is the passing diagnostic no-TF32 comparison, and attempt 12 is the
passing selected-route no-TF32 artifact. All four record XLA execution, memory
growth, graph control flow, absence of host callbacks, allocator data, exact
source hashes, and complete run manifests. Attempts 05--08 remain preserved as
historical pre-execution failures.

Transfer score/MCSE artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/attempt01/aggregate_v2/result.json`.
This supports the criterion design only; it does not test the selected
moment-teacher route's complete final score.

N=4096 transfer artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_attempt01/aggregate/result.json`.

N=4096 timing artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_timing_attempt01/aggregate/result.json`.

Selected-route mechanics artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_20260730/attempt12_gpu_xla_fp32_no_tf32_selected/result.json`.

## Next Smallest Correct Implementation

1. Put fitting, marginalization, moment contraction, particle likelihood, OT,
   Contract E reset, and moment correction into one graph-native per-step
   TensorFlow program. Python setup may construct immutable configuration, but
   no Python sample loop or NumPy may enter the XLA path.
   Do not wrap `tt_particle_contract_e_step_reference_jvp`: it is an eager
   mechanics adapter using `_restore_cloud_jvp_core`, not the canonical
   streaming route. Integrate through the repository-issued canonical
   streaming Contract E value/JVP factory with an exact scope-matched tuning
   artifact and preserve its route identity.
2. After selected-route integration and derivative parity pass, run LGSSM at \(T=2,10,50\), comparing value and analytical score with
   Kalman and with empirical-target Contract E. A failed derivative parity is a
   hard implementation veto.
3. Only after LGSSM passes, run one-seed \(N>1000\) feasibility on
   predator-prey and score-admissible Austria SIR. These are descriptive
   feasibility probes, not ranking evidence.

## Evidence Interpretation

- Claimed target: total analytical derivative of the same finite hybrid
  particle/TT program whose particle lane supplies the likelihood value.
- Quantity checked so far: FP64 finite-program TT teacher mechanics and manual
  JVP parity; full fused FP32/GPU/XLA execution with TF32 on and off.
- Verdict: GPU/XLA mechanics pass. The TF32 program is wrong relative to the
  declared derivative-parity gate on this fixture. FP32-no-TF32 passes and is
  selected as a route-specific exception. On a different complete canonical
  score route, TF32
  drift is below one MCSE but fails the stricter 0.1-MCSE rule in three of five
  coordinates. Canonical moment-teacher particle composition, multi-step value,
  and total score remain not checked.
- No default change, nonlinear improvement claim, leaderboard result, or
  source-faithful Zhao-Cui claim follows from the present evidence.

The skeptical execution audit no longer treats TF32 parity as a continuation
veto for this lane. The active veto is the absent canonical particle/teacher
composition. The exact baseline after integration is
empirical-target Contract E; Kalman is an oracle only for LGSSM. The parity
thresholds above are mechanics diagnostics and must not be promoted into model
accuracy criteria. Each later LEDH horizon, particle count, dtype, backend, and
chunk combination requires an exact scope-matched tuning artifact, and the
repository exact-divisor chunk policy remains binding.

## Shared Worktree Boundary

Other agents were modifying Austria SIR, UKF, SGQF, Zhao-Cui model routes, and
leaderboard artifacts concurrently. Do not revert or rewrite those files.
Continue only the files named in this memo unless the owner reassigns scope.
