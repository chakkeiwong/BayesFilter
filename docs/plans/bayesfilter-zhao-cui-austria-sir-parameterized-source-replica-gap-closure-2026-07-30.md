# Zhao-Cui Austria SIR Gap-Closure Plan

Date: 2026-07-30

Status: `SUPERSEDED_HISTORICAL_SOURCE_REPLICA_PATH_NOT_ACTIVE`

> Superseded on 2026-07-30 by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.
> This source-replica experiment is not the active parameter-extension path.
> Its T1 failure does not block extension of the exact P88 training-base fixed
> variant. P76/P77 UKF was a separate experiment, not proved baseline behavior.

## Research Intent

Question: can the fixed-parameter Zhao-Cui Austria SIR source pipeline be
reproduced on the sealed comparison observations before adding the three-variable
parameter value/score extension?

Target: the observed-data finite program on the latent pre-clipping state,
with event order `x0 -> transition -> y1 -> ... -> y20`, sealed observation
hash `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07`.
The author joint order is `(x_t, x_previous)` (paper Eq. 15 and
`full_sol.m:25,132`); the existing forward particle compiler uses
`(x_previous, x_t)`, so any axis reordering/conditional affine adapter is an
explicit `fixed_hmc_adaptation` or `extension_or_invention`, not silent source
parity.

Paper/source anchors: paper Algorithm 2 and conditional proposal at
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:680`
and `:830-924`; Austria settings at `:2249-2305`;
`third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:14-55`;
sequential reuse at `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-130`;
joint target assembly at `full_sol.m:132-136`; parent-conditioned inverse at
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100`.

## Claim Ledger

| Operation | Classification | Evidence required |
|---|---|---|
| 36D squared TTSIRT target and paired-core marginal | `source_faithful` | paper Algorithm 2, Proposition 2; author `TTSIRT` source |
| author target order `(x_t,x_previous)` | `source_faithful` | paper Eq. 15; `full_sol.m:25,132-136` |
| runtime reorder to `(x_previous,x_t)` for forward compiler | `fixed_hmc_adaptation`/`extension_or_invention` | must be bound in artifact identity |
| block-upper refactor of the full `computeL` covariance in author order | `fixed_hmc_adaptation` | covariance tie-out, zero lower-left block, source upper-conditional anchors |
| suffix-conditioned reverse numerical KR inversion | `extension_or_invention` implementing the source Eq. (20) dependency order | conditional density and roundtrip tests; no production-KR claim |
| `Lagrangep(4,8)` and `AlgebraicMapping(1)` basis/domain | `source_faithful` | author `mainscript.m:43-45`, P85 basis/source crosswalk |
| deterministic/frozen ranks, seeds, optimizer, or HMC-compatible schedule | `fixed_hmc_adaptation` | author route anchor plus manifest of the frozen operation |
| TensorFlow implementation of the author fit, sealed-data binding, or streaming serialization | `extension_or_invention` | must not be called source-faithful without a separate author-code proof |
| UKF moments | `fixed_hmc_adaptation` geometry only | no likelihood, proposal-truth, or score claim |

## Baseline Ladder And Gates

1. Original-author expectation: `d=0`, `m=18`, `n=9`, `T=20`, `N=5000`,
   squared TTSIRT, `Lagrangep(4,8)`, `AlgebraicMapping(1)`, max rank 40,
   init rank 20, kick rank 5, ALS limits 8/2. These are source baselines,
   not silently promoted local defaults.
2. Bootstrap fixed branch: mechanics and score recursion only; no proposal
   quality claim.
3. Source-replica fixed proposal: author-shaped 36D fit with sealed `y_t`,
   immutable serialized cores, parent-conditioned conditional proposal, and
   previous-marginal carry.
4. Parameterized value/score extension: attach the existing frozen-branch
   normalized-weight score recursion to exactly the fixed proposal branch.

Continuation gates are sequential: T1 source-replica ESS/roundtrip/value
validity, then T2 previous-marginal validity, then streaming T20 validity, then
parameter value/score. A failed fit or tuning candidate is a repair trigger,
not evidence against Zhao-Cui. Stop only for a wrong target/order, missing
parent conditioning or previous marginal, invalid density, nonfinite values,
memory breach, underdetermined fit, or corrupted artifact.

Primary promotion criterion: finite value and score, exact target identity,
conditional roundtrip `<=1e-4`, and minimum validation ESS fraction `>=0.5` at
the current horizon. T1 is a hard continuation gate, not a ranking claim.

Veto diagnostics: wrong observation hash, wrong event order, unconditional
proposal, prior substituted for the carried previous marginal, nonfinite or
negative density, KR denominator/monotonicity failure, memory forecast breach,
or a missing immutable core identity. Explanatory diagnostics include ESS
trajectory, log-weight spread, residuals, optimizer trace, and runtime.

Nonclaims: no exact likelihood, pseudo-marginal unbiasedness, posterior
correctness, HMC readiness, production-KR closure, statistical superiority,
GPU performance, or source-faithful assembled-route claim until separate gates
earn them.

## Memory And Execution Policy

Never retain full tensor-product grids, all time-step training clouds, or
particle histories beyond the current step and the serialized TT/core manifest.
The required working set is bounded by `batch * cdf_grid * (prefix + rank^2)`
and must be checked before KR inversion. T2/T20 use streaming step artifacts,
not a `[T,N,36]` retained tensor. CPU-hidden TensorFlow is used for focused
checks and T1 capacity diagnostics; GPU/XLA is deferred until the fixed branch
passes and must enable memory growth before device initialization.

## Defaults And Assumptions Audit

| Choice | Provenance | Failure mode | Early check | Status |
|---|---|---|---|---|
| `Lagrangep(4,8)`, algebraic scale 1 | author `mainscript.m:43-45` | sealed data may require another rung | T1 residual/ESS and basis manifest | baseline |
| rank 4/5 local fit | P86 training artifacts | optimizer/rank failure mistaken for math failure | finite trace, holdout residual, core activity | hypothesis |
| L1 tuning | owner Zhao-Cui policy | holdout leakage or unstable density | disjoint calibration/validation IDs | reviewed procedure |
| UKF frame | existing local scout | wrong joint covariance or proxy promotion | manifest says geometry-only; compare source frame | warm-start geometry |
| FP64 offline / FP32 online | repository backend policy | dtype drift or memory surprise | tensor dtype and branch manifest | reviewed default |
| finite-grid KR inversion | current transport substrate | approximation error or high memory | roundtrip and byte forecast | diagnostic adaptation |

## Skeptical Plan Audit

The superseded July 30 plan had the wrong first baseline (rank-one stochastic
SGD), treated a T1 mechanics smoke as proposal evidence, transferred stale
degree/rank settings, replaced the author adaptive fit with an unrelated
trainer, omitted the joint previous/current covariance, and had no fixed-route
gate before parameter score. This plan repairs those defects by requiring the
author-shaped fixed route first, classifying every extension, using disjoint
validation, and stopping before T2/T20/score on a T1 veto.

The first audit found a material execution defect: stopping merely because the
upper conditional adapter was absent did not close the gap named by this plan.
The revised plan therefore implements the smallest exact dependency repair:
refactor the same full affine covariance into a block-upper square root in
author order, condition on its suffix coordinates, and invert the current-state
coordinates in reverse order. This preserves the full covariance but changes
the author's particular lower-Cholesky square root, so it is a
`fixed_hmc_adaptation`, not a source-faithful `computeL` claim.

The revised audit passes with the following limits: the local TensorFlow fitter
is an extension unless independently shown equivalent to author TT-cross/ALS;
the finite-grid KR route is diagnostic rather than production; and the sealed
observations are not the P86 training-data scope. No command below can answer
those unearned claims.

## Execution Sequence And Artifacts

1. Add the repository-issued source-replica spec and identity checks.
2. Add the full-covariance block-upper author-order frame adapter and a
   memory-bounded suffix-conditioned reverse KR sampler/density evaluator.
3. Add focused tests for covariance preservation, forbidden cross-block drift,
   conditional correction, inverse/forward roundtrip, immutable source
   settings, and score-branch reuse.
4. Run CPU-hidden focused tests and a bounded T1 fit/proposal diagnostic under a
   fresh artifact directory. The T1 artifact must report same-frame holdout
   behavior, ESS, roundtrip error, and the KR working-set bound.
5. Continue to T2 and streaming T20 only if the T1 gate passes; otherwise write
   the repair result and stop. A failed tiny fit becomes a fit-repair trigger,
   not an adapter-missing blocker and not evidence against Zhao-Cui.
6. Run parameter value/score only after the fixed branch passes; write a result
   note and reset memo with command, environment, seed, wall time, memory,
   artifact paths, decision table, inference-status table, and nonclaims.

## Current Execution Result

The full-covariance block-upper author-order adapter and reverse suffix KR route
are implemented and tested. Attempt 6 passes finiteness, identity, memory,
conditional roundtrip (`1.51676346677454e-05`), and numerical-versus-exact TT
conditional consistency (`0.007612825749294672` maximum absolute log-density
difference). It fails the T1 proposal gate with ESS fraction
`0.12500000000392308 < 0.5`.

The continuation veto therefore fired. T2, T20, parameter score, GPU/XLA, and
HMC-facing work were not run. The remaining gap is the fit route: the local P86
Adam trainer is an `extension_or_invention` and did not reproduce the proposal
quality needed from the author's TT-cross/ALS pipeline. This is a repair trigger
for author-training reproduction or a reviewed target-specific protocol, not a
rejection of Zhao-Cui. The result and attempt ledger are in
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-result-2026-07-30.md`.
