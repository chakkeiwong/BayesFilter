# Zhao-Cui Austria SIR Observed-Data Score Historical Implementation Plan

Date: 2026-07-30
Status: `SUPERSEDED_HISTORICAL_APF_PATH_NOT_ACTIVE`
Upstream handoff: `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-implementation-handoff-2026-07-30.md`

> Superseded on 2026-07-30 by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.
> This file and its results remain historical evidence about the July 30
> frozen-proposal APF candidate. They do not govern the current extension of
> the exact P88 training-base fixed-variant artifact. P76/P77 UKF was a
> separate experiment, not proved baseline behavior.

## Decision

Implement the missing Austria SIR observed-data value and manual score as a
fixed, source-order particle program.  Begin with the coupled 36-dimensional
adjacent squared-TT/TTSIRT proposal used as a prototype by P76/P77.  Fit and
freeze one time step at a time.  Do not use the historical all-axes retained
grid.  A compartment/neighborhood factorization is a repair candidate only if
the coupled route fails a declared memory or proposal-quality gate.

This plan governed the now-superseded implementation campaign. The upstream
handoff is retained only as its historical mathematical handoff.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Can a fixed Zhao-Cui-derived proposal produce a finite, memory-bounded value and three-coordinate manual score for the exact Austria SIR `T=20` finite observed-data program? |
| Exact target | `J=9`, latent pre-clipping state `z_t` of dimension 18, infectious observations of dimension 9, `theta=(log_kappa_scale,log_nu_scale,log_observation_noise_scale)`, `theta_ref=(0,0,0)`, event order `z0 -> transition -> y1 -> ... -> transition -> y20`. |
| Observation identity | source FP64 SHA-256 `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07`; runtime FP32 SHA-256 `40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009`. |
| Candidate | Offline full 36D adjacent squared-TT densities and conditional TTSIRT maps, frozen at `theta_ref`, compiled into a source-order fixed-genealogy APF. |
| Baseline ladder | Direct tiny scalar; simple fixed Gaussian/bootstrap branch; P76/P77 adjacent trainer prototype; coupled TTSIRT candidate; existing same-target GenUT and SGQF rows. |
| Primary criterion | The untouched `N=1008,T=20` GPU/XLA run returns a finite scalar and three finite analytical score coordinates for the same program, with complete target/branch/program/tuning identities. |
| Hard vetoes | Wrong observation hash/event order; physical clipped-state Gaussian density; reduced/local-complete-data target; retained-grid fallback; omitted importance or derivative term; theta-dependent allegedly frozen branch; nonfinite result; same-scalar score mismatch; invalid proposal density/round trip; memory budget breach; missing target-specific tuning artifact; claim data used for tuning; missing GPU/XLA/memory-growth evidence. |
| Explanatory diagnostics | Held-out density metric, ESS, log-weight spread, score increments, central FD difference, transport round trip, CDF residual, runtime, and allocator peak. |
| Artifact | Versioned JSON/Markdown artifacts under `docs/benchmarks/artifacts/zhao_cui_austria_sir_observed_data_score_20260730/attemptNN/`; prior attempts are never overwritten. |
| Nonclaims | No exact nonlinear likelihood, pseudo-marginal unbiasedness, HMC convergence, posterior correctness, whole-route source faithfulness, statistical superiority, or production KR closure. |

The current numerical grid-CDF `FixedTTSIRTTransport` reports
`production_kr_closure=False`.  Therefore a successful run under this plan is
an executed fixed finite-program value/score research artifact, not HMC or
production readiness.  Production/HMC promotion remains blocked until the KR
closure gate is separately resolved.

## Mathematical Program

For frozen samples, ancestors, proposal densities, and auxiliary laws,

\[
\ell_0^i=\log p_\theta(z_0^i)-\log q_0(z_0^i),
\]

\[
\ell_t^i=\log W_{t-1}^{A_t^i}
+\log f_\theta(z_t^i\mid z_{t-1}^{A_t^i})
+\log g_\theta(y_t\mid z_t^i)
-\log a_{t-1}(A_t^i)-\log q_t(z_t^i\mid z_{t-1}^{A_t^i},y_t),
\]

\[
\Delta_t=\operatorname{logsumexp}_i\ell_t^i-\log N,
\qquad \widehat\ell(\theta)=\sum_{t=0}^{T}\Delta_t.
\]

With `D_{t-1}^j = grad_theta log W_{t-1}^j`, define

\[
M_t^i=D_{t-1}^{A_t^i}+\nabla_\theta\log f_\theta
+\nabla_\theta\log g_\theta,
\quad
s_t=\sum_iW_t^iM_t^i,
\quad
D_t^i=M_t^i-s_t.
\]

Then `grad_theta widehat_ell = sum_t s_t`.  Proposal and auxiliary score terms
are zero only because the complete online branch is issued before runtime and
is independent of runtime `theta`.

The filtering state is the continuous pre-clipping state:

\[
x_0=z_0,\qquad x_t=\operatorname{clip}_{S\geq0}(z_t),\ t\geq1,
\]

and `f_theta` is the ordinary Gaussian density of `z_t` conditional on the
previous physical state.  No Jacobian is introduced by this generative
representation.

## Source-Support Ledger

| Claim/operation | Technical source checked | Classification |
| --- | --- | --- |
| Positive squared-TT density plus defensive reference density | Zhao-Cui JMLR 2024 Section 3.1 Eq. (13), local full text lines 549-570 | `source_faithful` operation |
| Paired-core marginal and conditional KR construction | Proposition 2 and Eq. (14), local full text lines 593-655; author `@TTSIRT/marginalise.m:19-85` | `source_faithful` operation |
| Conditional inverse and proposal density | Paper Eq. (20)-(23), Algorithm 3; author `@TTSIRT/eval_irt_reference.m:16-71` and `eval_rt_jac_reference.m:17-113` | `source_faithful` operation when local tests tie out |
| Sequential reapproximation and proposal correction | Author `models/full_sol.m:21-43`, `:46-129`, `:132-136` | source anchor, not identical assembly |
| Fixed samples/ranks/seeds/schedules | Adaptation of the source operations for a deterministic HMC-facing program | `fixed_hmc_adaptation` |
| Latent pre-clipping representation | Project derivation preserving the source simulator law | `extension_or_invention` |
| Fixed cross-parameter APF and recursive score | Project derivation above; not present in inspected author route | `extension_or_invention` |

Local full text and the pinned companion code were inspected.  The existing P17
source ledger records no known retraction or version conflict, but no fresh
network metadata/retraction query is required for this implementation campaign.
Backward/forward snowballing is inherited from the project scholarship lane and
does not affect the narrow implementation claims above.  The major omission
risk is Cui-Dolgov 2022 for the original squared transport proof; this plan uses
Zhao-Cui's stated operation and local implementation behavior, and does not make
a new theorem claim requiring that proof.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Full 36D adjacent TT | Closest executed Austria prototype; hypothesis, not default | rank/fit or KR memory infeasible | one sealed-target `T=1` fit and byte estimate |
| UKF initialization | P76 scout guide only | block-diagonal frame misses cross-time coupling | compare held-out metric to initializer and inspect ESS |
| Degree 2/rank 4 | P77 warm start only | underfit or unstable tails | target-specific calibration ladder |
| L1 regularization | Required Zhao-Cui training policy | wrong weight masks or destabilizes fit | include explicit L1 grid with validation-only selection |
| Frozen `theta_ref=0` | fixed-HMC hypothesis | poor proposal away from reference theta | representative-theta ESS/score stability diagnostic |
| Numerical grid-CDF KR | existing diagnostic transport | inaccurate inverse or excessive `B*G*prefix` memory | round-trip/CDF tests and preallocation byte gate |
| Predictive auxiliary law | source-order APF hypothesis | normalization error or low ESS | direct normalization and bootstrap comparator |
| FP64 fit, FP32 online | existing training/production split | cast changes density/score materially | tiny FP64/FP32 parity before GPU claim |

No inherited rank, degree, learning rate, regularization, defensive mass, CDF
grid, chunk, or UKF setting was promoted without campaign-scope validation.

Execution added two numerical contracts that were missing from the initial
plan: inverse/CDF round-trip maximum error must be `<=1e-4`, matching prior
Zhao-Cui diagnostic plans, and the T1 downstream minimum ESS fraction must be
`>=0.5` before advancing to T2. The initial proposal is the exact Gaussian
prior represented by a constant reference density and a full-covariance
Gaussian-quantile coordinate map; an algebraic initial map is not an eligible
substitute because it confounds the adjacent proposal test with avoidable
initial importance-weight variance.

## Resource And Attempt Budget

| Resource | Limit |
| --- | --- |
| CPU implementation/smoke attempts | 12, each at most 10 minutes |
| GPU mechanics attempts before tuning | 3, each at most 10 minutes |
| Target-specific tuning attempts | 6 candidate fits total, each at most 45 minutes, only after Phases 0-3 pass |
| Untouched claim attempts | 2, each at most 30 minutes |
| FP64 training host-memory cap | 12 GiB peak |
| One ALS dense design | 2 GiB estimated bytes |
| One ALS normal matrix | 1 GiB estimated bytes |
| One KR inverse microbatch | 512 MiB estimated live tensor bytes |
| TensorFlow GPU allocation | verified memory growth; no whole-device preallocation |

Every failed serious attempt consumes the corresponding budget.  Localized
harness repairs may be retried within the budget when the scientific target and
method remain unchanged.  A new proposal factorization, larger particle count,
larger budget, package mutation, or production/HMC claim requires a revised
plan.

## Implementation Phases

### Phase 0: target and source closure

1. Add a repository-owned Austria target manifest/factory using the exact
   leaderboard observations and both hashes.
2. Bind `J=9,d=18,m=9,p=3,T=20,N=1008`, event order, latent-preclip measure,
   `theta_ref`, and route classification.
3. Fail closed for the reduced SIR, local-complete-data, wrong hash, `y0`-first,
   or retained-grid routes.

Exit: manifest and source/classification tests pass.

### Phase 1: FP32-native source-order value/score

1. Reuse the generic source-order evaluator in
   `zhao_cui_predator_prey_fixed_variant_tf.py`; do not fork its recursion.
2. Implement an FP32-native latent-preclip Austria model adapter with analytical
   initial, transition, and observation scores.  Use tensor time indices inside
   the compiled loop.
3. Seal an Austria-specific program factory and repository-issued identity.
4. Test `T=1/2` against an independent direct scalar, manual-score additivity,
   central FD and GradientTape diagnostics, FP64 reference parity, and replay.

Exit: eager FP64 reference and FP32 graph/XLA mechanics pass.

### Phase 2: frozen artifact and memory boundaries

1. Add immutable serialization/deserialization for trained TT cores, basis,
   coordinate maps, defensive density, normalizers, controls, source hashes,
   calibration/validation roles, and dependency closure.
2. Add a KR microbatch estimator/gate before `[batch,grid,prefix]` allocation.
3. Extend the source-order compiler to chunk inverse evaluation while preserving
   pointwise proposal density and branch identity.
4. Add a resumable staged runner.  Claim mode may consume only a matching
   repository-issued tuning artifact and may not train or refit.

Exit: artifact replay, mismatch rejection, and byte-gate tests pass.

### Phase 3: sealed-target `T=1/2` proposal bridge

1. Adapt the P76/P77 trainer to the exact sealed observations and
   `theta_ref=0`; retain UKF only as offline geometry.
2. Train the full coupled 36D adjacent density with separate calibration and
   validation seeds and explicit L1 candidates.
3. Freeze/export the selected candidate, compile chunked TTSIRT samples, and
   evaluate the source-order APF at `T=1`, then `T=2`.
4. Check density, normalization, support, proposal correction, inverse/forward
   round trip, ESS, score stability, and memory.

Exit: finite, replayable `T=2` branch with no resource or mathematical veto.

Execution result: the T1 bridge and complete proposal-corrected APF branch were
implemented, but the coupled candidate failed the downstream T1 quality gate.
The terminal structured smoke had initial ESS `8/8`, inverse/CDF round-trip
error `5.95e-8`, minimum ESS fraction `0.125`, and log-weight spread `94.8`.
Bounded degree/rank/training/L1 diagnostics did not repair the collapse, so the
campaign stops before T2. Phases 4-6 were not executed.

### Phase 4: sequential `T=20` fit and scope-specific tuning

For each `t=1,...,20`, build and freeze an approximation to

\[
\widehat p_{t-1}(z_{t-1})f_{\theta_{ref}}(z_t\mid z_{t-1})
g_{\theta_{ref}}(y_t\mid z_t).
\]

Only one trainer and one KR microbatch may be live at a time.  Retain frozen
cores/maps and the compact online branch, not training batches or full KR grids.
Tune degree, rank, learning rate, L1 weight, defensive mass, UKF controls, CDF
grid/chunk controls, and auxiliary law on calibration/validation data.  Freeze
the selection before untouched claim data are opened.

Exit: a repository-issued tuning artifact exactly matches the claim scope and
all twenty frozen proposal steps.

### Phase 5: untouched GPU/XLA claim

Run `T=20,N=1008` with FP32, TF32, XLA, trusted GPU access, and verified memory
growth.  Record value, score, per-time increments, ESS, branch/program/tuning
identities, current/peak allocator bytes, command, environment, commit, seeds,
wall time, and artifact paths.  Compare with GenUT/SGQF descriptively only.

Exit: `executed_value_score_research_candidate` or a precise terminal blocker.

### Phase 6: integration and closeout

Run focused and relevant high-dimensional regressions.  Write a new leaderboard
attempt, result note, run manifest, and reset memo.  Never overwrite the July 23
artifact.

## Skeptical Pre-Execution Audit

| Audit question | Finding and correction |
| --- | --- |
| Wrong baseline? | P76/P77 is an initializer/trainer prototype, not an end-to-end baseline.  The ladder begins with a direct scalar and simple fixed branch. |
| Proxy used for promotion? | Held-out density, UKF moments, FD, and `T=1/2` are diagnostics only.  Promotion requires the untouched `T=20,N=1008` finite program. |
| Missing stop conditions? | Mathematical, target, memory, tuning-identity, ESS/nonfinite, and backend vetoes are explicit above. |
| Unfair comparison? | GenUT/SGQF use the exact same observation tensor and target hash; reported differences remain descriptive without uncertainty support. |
| Hidden assumptions? | Rank/degree/L1/UKF/KR/frozen-theta assumptions are recorded and tested rather than inherited as defaults. |
| Stale context? | The exact July 23 target replaces the P76 seed-5901 observation.  The latent-preclip representation replaces the wrong clipped-state Gaussian density. |
| Environment mismatch? | FP64/CPU is reference only; the claim requires FP32/TF32/XLA/GPU and verified memory growth. |
| Non-answering commands? | No serious tuning starts before the direct scalar, adapter, artifact, and memory gates pass.  Claim mode cannot train. |
| Memory blow-up? | Full retained grids are forbidden; ALS/KR estimates fail before allocation; steps and microbatches stream under explicit byte caps. |
| Derivative mismatch? | The score is checked against FD of the exact same frozen scalar, and all theta-dependent proposal terms are forbidden or must be differentiated. |

Audit verdict: `PASS_AFTER_REVISIONS`.  The revisions are the FP32-native latent
adapter, immutable trainer-to-TTSIRT export, preallocation KR byte gate, explicit
mechanics-versus-production-KR claim boundary, and fixed attempt budgets.

Post-execution audit correction: the original plan omitted numeric round-trip
and T1 ESS thresholds and did not specify the exact initial proposal. Those
omissions allowed the coarse-grid `attempt08` to be misclassified as a pass.
That attempt is preserved as superseded evidence; the corrected contracts above
governed that campaign's terminal decision.

## Stop And Repair Logic

- A candidate fit failing held-out/ESS gates triggers fresh target-specific
  tuning within budget; it does not reject the research direction.
- A mathematical target, measure, score, proposal-density, or identity failure
  blocks progression until repaired and retested.
- Coupled 36D memory failure after the preallocation gates triggers a new
  reviewed compartment/neighborhood-block repair plan.  It does not permit the
  retained-grid route.
- Exhausting a campaign budget stops execution with a result artifact; limits
  are not silently enlarged.
- `production_kr_closure=False` blocks production/HMC promotion but does not
  prevent a correctly labeled finite-program research value/score artifact.

## Definition Of Done

The task is complete when the exact target has a frozen `T=20,N=1008` branch,
the finite value and manual three-coordinate score are emitted by the same
FP32/XLA program, all focused and relevant regressions pass, the GPU manifest
records memory growth and allocator peaks, and versioned result/reset artifacts
state the route classification and nonclaims.  Otherwise the closeout must name
the first unsatisfied mathematical, implementation, tuning, or resource gate.
