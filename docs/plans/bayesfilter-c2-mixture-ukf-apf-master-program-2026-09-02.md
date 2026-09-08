# BayesFilter C2 Mixture-UKF/APF Master Program

Date: 2026-09-02
Status: `PHASE8D_RECOVERY_RECORDED_CANDIDATE_UNPROMOTED`
Owner: chakwong
Scope: generic observation-informed proposals for the C2 diagnostic fixture
Last reviewed: 2026-09-08 (dedicated-workspace reset and focused integration
verification; the Phase 8D numerical result was not rerun in this reset)
Readiness: `PHASE8E_PRELAUNCH_AUDIT_READY`

This is the active master program for the new mixture-UKF direction. It does
not rewrite or promote any earlier C2 result. The older plans remain the
authorities for the campaigns that they executed; this document records their
results as baselines and governs the next, unimplemented experiment.

**Workspace reset (2026-09-08).** Active C2 work now lives only in
`/home/chakwong/BayesFilterZhaoCui` on branch
`zhao-cui-tt-regression-20260908`, based on GitHub main at `d2124d42`. The
recovery call chain was restored file-by-file from `21d5870f`; a focused
CPU-only integration suite passed (`99 passed`). See
`bayesfilter-zhao-cui-tt-regression-reset-memo-2026-09-08.md`. That mechanics
result does not reproduce Phase 8C/8D numerical evidence and is not proposal
promotion evidence.

## 1. Why a new master program is needed

The repository has several completed or halted C2 plans, but none governs the
proposed SPPF/UPF-style method as a whole:

| Record | Current status | Boundary |
| --- | --- | --- |
| [`../benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`](../benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex) | Historical failure analysis | Documents the n=4 failure and the actual GH9/TT program; not a current UKF result |
| [`../benchmarks/artifacts/c2_completion_20260824/attempt05/n4_diagnostic_handoff_memo_20260828.md`](../benchmarks/artifacts/c2_completion_20260824/attempt05/n4_diagnostic_handoff_memo_20260828.md) and [`../benchmarks/artifacts/c2_completion_20260824/attempt05/n4_diagnostic_t20_result_addendum_20260828.md`](../benchmarks/artifacts/c2_completion_20260824/attempt05/n4_diagnostic_t20_result_addendum_20260828.md) | Historical diagnostic handoff | Evidence and interpretation boundaries retained; no promotion evidence |
| [`bayesfilter-c2-actual-program-mathdevmcp-audit-20260901.md`](bayesfilter-c2-actual-program-mathdevmcp-audit-20260901.md) | Documentation audit complete with partial coverage | Clarifies that the existing attempt05 guide is GH9-based and that formalization flags are not proof failures |
| [`mixture-ukf-literature-audit-20260902.md`](mixture-ukf-literature-audit-20260902.md) | Literature audit complete | Establishes SPPF/UPF, Gaussian-sum UKF, and GMSPPF as method families; no implementation or C2 success claim |
| [`bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md`](bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md) | Executed bounded diagnostic | Fixed TT/Student DMIS had valid engineering checks, but a heuristic-dominance veto fired |
| [`bayesfilter-c2-coherent-tt-proposal-testing-plan-20260831.md`](bayesfilter-c2-coherent-tt-proposal-testing-plan-20260831.md) | Stage 0/1 complete; Stage 2 vetoed | Exact-factor ladder was tested; recursive map was not implemented |
| [`bayesfilter-c2-phase2-generic-dmis-contract-correction-execution-result-20260902.md`](bayesfilter-c2-phase2-generic-dmis-contract-correction-execution-result-20260902.md) | Precision veto | Complete-DMIS convention and tangent checks passed, but the plain-DMIS precision gate failed at captured steps; recursive stage was not run |

Before Phase 1, the generic C2 proposal call chain had only a global UKF scout
and a separate historical LEDH per-particle lifecycle; it did not run one UKF
per ancestor. Phase 1 now supplies a candidate per-ancestor K=1 endpoint and
the C2 adapter wires it into the shared frozen finite-program evaluator. The
current retained-TT and defensive channels remain separate comparators, and
the new method is still a hypothesis rather than a promoted feature.

### Coverage crosswalk

The earlier recommendations are all represented explicitly in this program;
the table also records the execution order rather than treating them as one
undifferentiated modification.

| Recommended element | Location in this program | Execution status |
| --- | --- | --- |
| Per-ancestor UKF or SPPF/UPF proposal | Sections 2--3; Phases 1--3 | implemented and smoke-checked; candidate only |
| APF lookahead and exact complete-mixture DMIS | Section 3; Phase 0 mechanics and Phase 2 | fixture, entry, and serious N=8192 wiring checked; larger ladder pending |
| Smooth data-dependent mixture/defensive weights | Section 3; Phase 4 | implemented and replayed; validity passes, efficiency/promotion veto recorded |
| Recursive predicted moments and lagged map `T_t` | Section 3; Phase 5 | generic callable and bounded fixture/probe passed; representation ladder now at Phase 5C |
| Hermite degree, RBF, and Hermite-plus-RBF tests | Section 3; Phases 5A--5C | Hermite, RBF, hybrid mechanics, and three-seed/two-step replication passed; no width promoted |
| Gaussian, Student, and Gaussian-plus-Student `r(u)` | Section 3; Phase 5D/reference-law arm | proposal Student tested; Student TT measure deferred |
| Held-out/shell/`Z_H`-vs-`Z_T`/Gram/rank/recursive diagnostics | Sections 5 and 7; Phases 5A--7 | required artifacts |
| Analytical-gradient contract | Section 3; all claim-bearing phases | frozen-law route first; adaptive-total route separate |

The earlier coherent plan remains the detailed historical specification for
its completed stages. This master supersedes its *future-stage ordering* for
the mixture-UKF campaign, while preserving the earlier result and review files.

## 2. Research intent ledger

### Main question

Can an observation-conditioned, per-ancestor mixture-UKF proposal provide the
localization of an auxiliary particle filter while retaining the exact C2
transition/observation target, a complete importance denominator, and an
analytical gradient of one declared frozen finite program?

### Candidate mechanisms

1. `K=1`: one UKF-conditioned Gaussian proposal per retained ancestor, with
   APF predictive lookahead.
2. `K=2`: a fixed symmetric local two-component proposal around the UKF
   posterior moments.
3. `K=4`: a fixed two-direction, four-component local proposal.
4. A small full-support Student-t defensive component mixed with the local
   proposal.
5. A lower-cost global Gaussian-sum/SPKF bank with deterministic mixture
   reduction, used as a GMSPPF-style cost arm.
6. A lagged recursive moment-derived coordinate map for the TT fitter, tested
   only after the proposal and integration gates are valid.
7. A smooth, fixed-size data-dependent gate for local components and the
   defensive tail, using softmax/sigmoid weights rather than hard selection.
8. A representation ladder consisting of Hermite degrees 6, 8, and 10, fixed
   separable RBF channels at UKF offsets, and a Hermite-plus-RBF hybrid.
9. A separately identified Gaussian, product-Student, or Gaussian-plus-
   Student reference measure for the TT fitter.

The K=2 and K=4 split constructions are explicitly
`extension_or_invention`; they are not presented as the published GMSPPF
algorithm. SPPF/UPF and Gaussian-sum filtering are the literature anchors.
For K=4, use the four sign combinations of two fixed directions and subtract
the sum of the two rank-one offset covariances from P before the split. With
equal weights, the sign cross-terms average to zero, so the mixture has mean
\(\mu\) and covariance \(P\) for any fixed directions whenever the common component
covariance \(P-\delta^2(d_1d_1^{\mathsf T}+d_2d_2^{\mathsf T})\) is SPD. Orthonormality
is not required for that moment identity; it is only a possible convention
for choosing or scaling the directions. A true K-component Gaussian-sum UKF
bank, in which each component is propagated and updated separately, is
reserved for the GMSPPF-style cost arm in Phase 6.

### Expected failure modes

- A raw signed C2 observation has zero population state/observation
  cross-covariance, so a raw-observation UKF can have zero gain.
- A transformed-observation UKF can match only low-order moments and still
  miss skewness, multimodality, or tails.
- The local mixture can be mathematically valid but too narrow, producing the
  same weight collapse as the retained TT.
- A selected component density can be used accidentally in the denominator
  instead of the complete mixture density.
- APF lookahead can be computed but its required w/a correction can be omitted.
- A recursive map can feed fitting error into the next step and amplify it.
- A frozen analytical gradient can omit derivatives of moments, Cholesky
  factors, or proposal densities.

### Promotion and interpretation boundary

No default or production promotion is sought in this program. A candidate may
be called *viable for a follow-up randomized-likelihood study* only if its
finite-program and proposal-law gates pass. A positive ESS contrast alone is
not a correctness or superiority claim.

### Source boundary

The local primary source inspected for this design is
`docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf`,
especially its SPPF and GMSPPF sections. The literature audit
supports per-particle sigma-point proposals and finite Gaussian-mixture/SPKF
banks, but the original UPF report and the full Alspach--Sorenson paper were
not separately obtained in this pass. Claims about those methods remain
bounded by the inspected thesis and cited metadata. The C2 transformed guide,
fixed local splits, frozen-gradient contract, and exact-DMIS wiring are
project extensions, not source-faithful claims.

LEDH is a related proposal family, but it is governed by the repository's
canonical LEDH-PFPF-OT and Contract-E policies. Pre-2026-08-21 LEDH/PFPF
results are historical and are excluded from this campaign's evidence. A
fresh C2 LEDH comparator, if desired, requires its own scope-specific tuning
and conformance subplan; this master does not silently revive the historical
route.

## 3. Evidence contract

### Target

For a normalized carried particle approximation at time t-1,

\[
  \gamma_t(j,x)
  = \bar w_{t-1,j}\,f_t(x\mid x_{t-1,j})\,g_t(y_t\mid x),
\]

where f_t and g_t are the exact transition and observation densities.
The carried particle approximation is part of the finite diagnostic target; it
is not silently replaced by a UKF density.

For each ancestor j, the UKF supplies a data-conditioned proposal

\[
  q_{t,j}^{(K)}(x)
  = \sum_{k=1}^{K}\pi_{t,j,k}
      \mathcal N(x;m_{t,j,k},P_{t,j,k}),
  \qquad \sum_k\pi_{t,j,k}=1.
\]

The APF lookahead probability is

\[
  a_{t,j}
  = \frac{\bar w_{t-1,j}\,\widehat\ell_{t,j}}
          {\sum_\ell \bar w_{t-1,\ell}\,\widehat\ell_{t,\ell}},
  \qquad
  \widehat\ell_{t,j}
  =\mathcal N\!\left(y_t;\widehat y_{t,j},S_{t,j}\right)
  \approx \int f_t(x\mid x_{t-1,j})g_t(y_t\mid x)\,dx,
\]

with the predictive likelihood computed by the UKF moment update. The
Gaussian quantity on the left is a proposal-selection approximation; it is
not substituted for the exact \(f_tg_t\) factors in the weight. If
an ancestor J is sampled from a_t and X is sampled from q_{t,J}^{(K)}, the exact conditional importance
weight is

\[
  \widetilde w
  = \frac{\bar w_{t-1,J}
          f_t(X\mid x_{t-1,J})g_t(y_t\mid X)}
         {a_{t,J}\,q_{t,J}^{(K)}(X)}.
\]

When a component label is sampled, the denominator is still the sum over all
components \(q_{t,J}^{(K)}(X)\), not only the selected component. If the
ancestor is marginalized rather than retained, both numerator and denominator
must be marginalized:

\[
  \gamma_t^{\mathrm{marg}}(x)
    =g_t(y_t\mid x)\sum_j \bar w_{t-1,j}f_t(x\mid x_{t-1,j}),
  \qquad
  q_t^{\mathrm{marg}}(x)=\sum_j a_{t,j}q_{t,j}^{(K)}(x),
  \qquad
  \widetilde w=\gamma_t^{\mathrm{marg}}(X)/q_t^{\mathrm{marg}}(X).
\]

Using the marginal denominator with the conditional numerator (or vice versa)
defines a different, incorrect finite target.
At the initial time, use the corresponding initial-state target and proposal
without an APF ancestor factor; this special case is tested separately.

### Finite importance identity

If \(q_{t,j}^{(K)}(x)>0\) wherever
\(\bar w_{t-1,j}f_t(x\mid x_{t-1,j})g_t(y_t\mid x)>0\), then for every
integrable test function \(\varphi\),

\[
\begin{aligned}
 &\mathbb E_{J\sim a_t,\;X\sim q_{t,J}^{(K)}}
       [\,\widetilde w\,\varphi(X)\,] \\
 &\quad=\sum_j\int
       a_{t,j}q_{t,j}^{(K)}(x)
       \frac{\bar w_{t-1,j}f_t(x\mid x_{t-1,j})g_t(y_t\mid x)}
            {a_{t,j}q_{t,j}^{(K)}(x)}
       \varphi(x)\,dx \\
 &\quad=\sum_j\int
       \bar w_{t-1,j}f_t(x\mid x_{t-1,j})g_t(y_t\mid x)
       \varphi(x)\,dx.
\end{aligned}
\]

The cancellation is the reason the approximate UKF lookahead may guide
ancestor selection without changing the exact finite target. Phase 0 tests
this identity numerically and, where supported by the local formalization
toolchain, as an elementary Lean lemma. The identity does not imply low
variance; proposal overlap and ESS remain empirical questions.

### UKF moment update

For each ancestor, a generic model adapter must expose batched transition and
observation evaluations. The UKF computes predicted state moments `mu-minus`
and `P-minus`, predicted observation moments `y-hat` and `S`, and cross-
covariance `C` from the sigma points before the update.

\[
  \mu_{t,j}=\mu^-_{t,j}
       +C_{t,j}S_{t,j}^{-1}(y_t-\widehat y_{t,j}),
  \qquad
  P_{t,j}=P^-_{t,j}-C_{t,j}S_{t,j}^{-1}C_{t,j}^{\mathsf T}.
\]

Every solve, covariance, and scale must be checked for finite values and
positive definiteness. The raw signed-observation zero-cross-covariance test
is a required negative control for C2; the transformed log-square observation
is a model adapter, not a hidden generic change to the target.

### Fixed local mixtures

The first extension uses an observation-informed unit direction d_{t,j}
and a fixed offset delta:

\[
  m_{t,j,\pm}=\mu_{t,j}\pm\delta d_{t,j},
  \qquad
  P_{t,j,\pm}=P_{t,j}-\delta^2d_{t,j}d_{t,j}^{\mathsf T}.
\]

It is admissible only when both component covariances are SPD. Equal weights
preserve the first two mixture moments. The four-component arm applies the
same construction in two fixed directions. Directions, offsets, component
counts, and topology are frozen before the claim run; no online pruning or
model-order change is allowed in the analytical-gradient route.

For K=2, the covariance calculation is
\[
  \frac12\sum_{\sigma\in\{-1,1\}}
  \left[P-\delta^2dd^{\mathsf T}
    +(\sigma\delta d)(\sigma\delta d)^{\mathsf T}\right]=P.
\]
For K=4, replace \(\sigma\delta d\) by
\(\sigma_1\delta d_1+\sigma_2\delta d_2\) and average over the four sign
pairs; the mixed \(d_1d_2^{\mathsf T}\) terms cancel. This identity is tested
before any C2 run, together with the SPD condition.

### Defensive tails

For a fixed 0 < epsilon < 1,

\[
  q_{t,j}^{\mathrm{def}}(x)
  =(1-\epsilon)q_{t,j}^{(K)}(x)+\epsilon r_{t,j}^{\mathrm{Student}}(x).
\]

The complete log density is evaluated by log-sum-exp. Student degrees of
freedom, scale convention, and epsilon are calibration hypotheses, not
defaults. The exact numerator remains f_t g_t.

### Smooth data-dependent gating

The proposal may respond continuously to the observation without using a
discrete argmax. For fixed smooth scores \(s_{t,j,k}\) (for example, negative
UKF innovation quadratic forms), temperature \(\tau>0\), and a floor
\(0\leq\rho<1/K\), define

\[
  \pi_{t,j,k}
    =\rho+(1-K\rho)
      \frac{\exp(s_{t,j,k}/\tau)}
           {\sum_{h=1}^{K}\exp(s_{t,j,h}/\tau)}.
\]

An observation-dependent defensive fraction can likewise be written

\[
  \epsilon_{t,j}
    =\epsilon_{\min}
      +(\epsilon_{\max}-\epsilon_{\min})
       \operatorname{sigmoid}(c_{t,j}),
  \qquad 0<\epsilon_{\min}\leq\epsilon_{\max}<1,
\]

where \(c_{t,j}\) is a declared smooth innovation or tail score. The actual
\(\pi_{t,j,k}\) and \(\epsilon_{t,j}\), not a selected component, enter the
complete proposal denominator. Hard clipping, argmax selection, online
pruning, and data-dependent model-order changes are excluded from the first
gradient-bearing route. In the frozen-law score contract, the realized
weights are snapshots held fixed at the reference point. In an adaptive-total
route, derivatives through the scores, softmax/sigmoid, and all resulting
proposal log densities are required.

### Recursive coordinate-map candidate

Only after the one-step proposal/integration gates pass, test the lagged
recursion:

1. propagate the retained approximation through the transition;
2. compute predicted mean, covariance, and relevant cross-covariances;
3. construct an SPD lower-Cholesky affine map T_t;
4. fit the likelihood-corrected density in the new coordinates;
5. pass only the frozen lagged moments and fitted object to the next step.

The recursive candidate must use an actual TensorFlow moment-contraction and
map callable. External GH9 moments may be a comparator, but cannot be renamed
as recursive UKF feedback. Hard EM, pruning, or fixed-point iteration is
outside the first gradient-bearing route.

For a carried mixture with normalized weights w_j, the predicted map inputs
are computed from the transition itself. If \(m_{t,j}^-\) and
\(Q_{t,j}\) are the conditional transition mean and covariance for ancestor
\(j\), then

\[
  m_t^-=\sum_j w_j m_{t,j}^-,
  \qquad
  P_t^-=\sum_j w_j\left[
       Q_{t,j}+
       (m_{t,j}^--m_t^-)(m_{t,j}^--m_t^-)^{\mathsf T}\right].
\]

For a continuous retained density, the same equations are evaluated by
integration or an explicitly declared quadrature rule. A single global
process-noise matrix is valid only when \(Q_{t,j}\) is constant. The map is
\(T_t(u)=m_t^-+L_tu\) with \(L_tL_t^{\mathsf T}=P_t^-\); the fitted
reference-coordinate target includes the Jacobian determinant of \(L_t\) and
the chosen reference density. These moments are lagged: the map used at \(t\)
is not re-solved from the fit produced at that same \(t\).

More explicitly, for a physical current-state target density
\(\gamma_t(x)\), the density ratio represented in reference coordinates is

\[
  h_t(u)
    =\frac{\gamma_t(T_t(u))\,|\det L_t|}{r_t(u)},
  \qquad
  s_t(u)=\sqrt{h_t(u)}.
\]

The fitted TT is compared with \(s_t\) under \(r_t(u)\,du\). Omitting either
\(|\det L_t|\) or \(r_t(u)\) fits a different finite target. If the target
retains an ancestor or component label, the same formula is applied to the
corresponding joint density before any declared marginalization.

### Basis and reference-density candidates

The representation ladder is tested separately from proposal quality. With a
Gaussian reference, test normalized Hermite degrees
\(d\in\{6,8,10\}\), fixed separable RBF channels centered at declared UKF
offsets over a finite width grid, and a Hermite-plus-RBF hybrid with exactly
one constant channel. For a coordinate \(u\), a separable RBF channel has the
form

\[
  \psi_c(u)=\prod_{r=1}^{D}
      \exp\!\left[-\frac{(u_r-c_r)^2}{2s_r^2}\right],
  \qquad s_r>0.
\]

The Hermite constant is already present; a duplicate constant is a singular
Gram direction. RBF cross-Gram terms are retained. Held-out \(L^2\), shell
residuals, direct \(Z_H-Z_T\), Gram conditioning, and realized TT rank are
reported before any recursive combination.

The evaluated functions are time-indexed, \(H_{t,b}(u)\): the coordinate map,
UKF offsets, and RBF centers may change with \(t\), while the selected family,
channel topology, and degree/width ladder are frozen for a claim branch. Thus
the program can use a changing \(H_{t,b}\) without silently changing the
finite-dimensional basis contract at each step.

The reference-density ladder is
\(r^{\mathrm G}(u)=\eta_D(u)\),
\(r^{\mathrm{St},\nu}(u)=\prod_r t_\nu(u_r)\), and a fixed
Gaussian--Student mixture. A Student reference is a new route, not a
parameter toggle: its mass matrix, marginal contractions, row-sampling law,
pulled-back target, normalizer, and score must be changed together. For
degree \(d\), the product-Student mass entries needed by polynomial
contractions are finite only when \(\nu>2d\). A Student proposal component may
therefore be tested before a Student TT reference is admissible.

### Gradient identity

There are two explicitly different score contracts. The first claim-bearing
route is the frozen-law finite program: observations, random inputs, ancestor
labels, component labels, sigma-point topology, selected hyperparameters,
proposal snapshots, maps, rows, and basis decisions are frozen at a reference
point, and the derivative is taken through the resulting declared scalar.
This is an analytical derivative of that frozen program, not of an adaptive
algorithm. An adaptive-total diagnostic may instead differentiate UKF moments,
ancestor probabilities, soft gates, Cholesky factors, proposal densities,
coordinate maps, and carried-density terms; it must include every such term.
For a fixed least-squares or ALS solve \(A c=b\), the coefficient tangent is
\(A\,\dot c=\dot b-\dot A\,c\); coefficients, mass matrices, and normalizer
contractions are either differentiated by this identity or explicitly frozen
and excluded from the claim. A partial derivative through only the exact
numerator cannot be labeled an analytical gradient of the adaptive algorithm.

## 4. Current call chain and planned endpoints

### Existing evidence

| Endpoint | What it currently does | Consequence |
| --- | --- | --- |
| [`bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py`](../../bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py#L643) | Compiles global retained-TT and parent-conditioned defensive branches; evaluates complete mixture densities for those branches (loop begins at line 643) | Reusable shared APF/DMIS authority, but not a per-ancestor UKF proposal |
| [`bayesfilter/highdim/ukf_scout.py`](../../bayesfilter/highdim/ukf_scout.py#L212) | Carries one global deterministic UKF scout path (lines 212--304) | Metadata/scouting only; not an SPPF/UPF implementation |
| [`experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py`](../../experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_alg1_ukf_tf.py#L314) | Contains a per-particle UKF prediction/update lifecycle (lines 314--445) | Historical/reference lifecycle; it is not wired into the C2 generic claim-bearing path |

### Planned endpoints (provisional until Phase 0 API freeze)

- one repository-owned, model-independent batched UKF conditional kernel;
- one APF compiler that returns ancestor probabilities, local mixture
  parameters, samples, and complete proposal log densities;
- one shared exact finite-program evaluator accepting the ancestor/base-mass
  row and the complete conditional or marginal denominator;
- one C2 adapter for the transformed observation, with the raw-observation
  zero-gain negative control;
- tests that prove the claim-bearing C2 endpoint calls the general kernel, not a
  C2-local reduced copy; and
- a diagnostic driver under
  `docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py` with fresh versioned
  output directories.

The endpoint names may change during Phase 0, but the call-chain requirement
does not: function existence without executable wiring is insufficient. The
wiring test must exercise the public C2 driver, resolve the repository-owned
batched UKF callable, and compare its output with a direct call on compatible
inputs. A source grep or an unused helper does not pass the call-chain audit.

## 5. Execution phases and gates

### Phase 0: governance, source, and parity preflight

Freeze the target identities, observation convention, particle count ladder,
dtype, seed map, API shapes, and proposal topology. The inherited serious row
ladder is \(N\in\{8192,16384,32768\}\), with twelve paired claim branches; the
smoke ladder is \(N\in\{256,1024\}\). A change to either ladder requires a plan
revision. Verify the retained snapshot fingerprint and preserve all prior
outputs. Build a small linear-Gaussian fixture with an exact Kalman conditional
proposal.

Required checks:

- linear-Gaussian UKF mean/covariance/innovation parity;
- exact importance identity and the APF w/a correction on the fixture;
- selected-component-versus-complete-mixture negative control;
- ancestor-label permutation invariance;
- C2 raw signed-observation zero-gain negative control;
- MathDevMCP audit of the displayed identities and a recorded Lean result for
  the elementary weight identity when the local toolchain supports it; and
- focused CPU tests, compile checks, `git diff --check`, and a call-chain
  inspection before any serious run.

Entry/exit: no GPU or long run before every check is finite and the API is
reviewed. Close Phase 0 with the phase-close table in Section 5, including the
repair disposition and the refreshed Phase 1 command. A tool limitation is
recorded as a gap, not silently called a proof; reviewer unavailability is not
itself a continuation veto, but a material review finding must be repaired or
resolved before the affected phase proceeds.

### Phase 1: generic per-ancestor UKF and K=1 APF

Implement the batched TensorFlow kernel and the exact conditional proposal.
The hot path uses `tf.function` with an explicit stable signature and XLA as
the default; it contains no pfor, `tf.vectorized_map`, sample-wise Python
loop, or NumPy numerical path. Covariance failures are fail-closed.

Compare on the linear fixture against the exact Kalman proposal and then run a
small C2 transformed-observation smoke. The old uniform route must reproduce
its scalar, score, normalized weights, and ESS within the existing float64
regression tolerance.

Gate: finite target and proposal values, complete denominator, finite
analytical score, central finite-difference agreement, eager/graph/XLA parity,
and a demonstrated per-ancestor response to changed observations.

**Phase 1 close (2026-09-03).** The generic kernel, C2 adapter, deterministic
frozen-input sampler, and focused call-chain tests are implemented. CPU smoke
attempts at (N=256) and (N=1024) passed all required checks; see
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase1-smoke-close-20260903.md`.
The K=1 minimum ESS was descriptively below bootstrap and transformed Student
at both rows, so it is retained as a candidate-efficiency repair trigger and
not promoted. The Student comparator emitted retracing warnings; this is an
open Phase 2 performance risk.

### Phase 2: K=1 C2 proposal comparison

Use the C2 transformed observation with APF lookahead and exact f_t g_t
weights. Compare against bootstrap conditional, transformed Student, Gaussian
hint, stationary Gaussian, and retained TT. Keep the PF reference as a gross
compatibility check only; it is not a finite-program oracle. The score route
must obey the frozen-program gradient identity above.

Gate: all engineering checks pass and the independent finite-target estimate is
finite and compatible. A low ESS is a repair trigger, not evidence of an
implementation bug by itself.

The first execution is a bounded **Phase 2 entry pilot**: one fresh retained-TT
fit, all six declared proposal families, one paired branch at (N=8192), and
full device/memory-growth/XLA provenance. This pilot measures the actual cost
of the serious call chain before any row or branch expansion. If it passes its
validity checks, the phase owner refreshes the remaining budget and may run up
to twelve paired branches at (N=8192), followed by (N=16384) and (N=32768)
only when the measured cost stays within the declared six-GPU-hour budget.
This staging repairs the original plan's unexamined assumption that
(3\times12\times6) full branches fit that budget.

**Phase 2 serious close (2026-09-03).** The twelve-branch `N=8192` expansion
completed with `72/72` records and no hard validity failure; see
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-serious-close-20260903.md`.
The K=1 ESS contrast was descriptively and consistently below bootstrap,
transformed Student, and Gaussian-hint comparators, so K=1 receives a
candidate-efficiency/promotion veto. The exact target, complete denominator,
APF correction, observation response, analytical score, and GPU provenance all
passed. This is a repair trigger for Phase 3, not a continuation veto. The
paired summary uses descriptive t intervals only; it does not rank methods or
establish posterior correctness.

### Phase 3: fixed-topology K=2 and K=4

Add the symmetric split arms with a predeclared dimensionless offset ladder.
Calibrate delta only on a disjoint calibration partition, freeze it, and
replay untouched claim branches. Record component separation, covariance
eigenvalue margins, conditional ESS, maximum weight, log-normalizer error, and
cost per effective sample.

Gate: no component collapse, no SPD violation, complete-mixture label
invariance, and a paired uncertainty interval for each proposed ESS contrast.
If K>1 loses to K=1 or a cheap adversary, reject that candidate only;
continue to the predeclared defensive-tail arm if the finite program remains
valid.

### Phase 4: smooth gating and Student defensive mixture

Test the softmax/sigmoid gates in Section 3 together with a small fixed
defensive fraction and a small degrees-of-freedom ladder. Calibration and
validation banks are independent; the final claim banks are never used to
select temperature, floors, epsilon, or degrees of freedom. Evaluate the full
mixture density at every sample and retain the exact target numerator.

Gate: support, normalization, sampling-density parity, finite score, explicit
observation sensitivity, and the same paired uncertainty contract as Phase 3.
A Student arm that merely masks a bad local proposal is reported as such.
Hard gating or a change in component count is a veto for the
gradient-bearing route.

### Phase 5: recursive lagged moment map

Run the bounded mechanics and integration study specified in
`docs/plans/c2-mixture-ukf-apf-phase5-recursive-map-20260904.md`. The linear
fixture is the first gate: the generic law-of-total-covariance and Cholesky
callable must agree with the independent Kalman recursion. A small C2 probe is
then allowed because Phase 4 validity passed; its ESS and cloud residuals are
descriptive and do not override the Phase 4 promotion veto.

Compare a static initial map with the lagged moment-derived map using the same
fixed bank, observations, and dtype. Record per-step map eigenvalue margins,
Cholesky diagonals, map condition, standardized-coordinate residuals, and
cumulative empirical error. Freeze maps and discrete decisions for any future
gradient-bearing fit; until a total derivative through map construction is
implemented, no score from this stage is claim-bearing.

**Phase 5 close (2026-09-04).** The generic moment/Cholesky callable passed
the independent linear Kalman oracle and the three-observation C2 wiring probe
on GPU/XLA (`phase5-recursive-map-attempt02`). Maximum linear mean,
covariance, and affine round-trip errors were `5.55e-17`, `1.11e-16`, and
`2.22e-16`; all C2 map rows were finite and SPD, and the model/fixture
transition and process-covariance checks were exact. The C2 ESS and
finite-bank residuals are descriptive only. See
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5-recursive-map-close-20260904.md`.

### Phase 5A: fitter and basis ladder

At the first divergent time, hold the proposal and map fixed and test the
representation arms in this order:

1. normalized Hermite degrees \(6,8,10\);
2. fixed separable RBF centers at declared UKF offsets over a finite width
   grid; and
3. a Hermite-plus-RBF hybrid with exactly one constant channel and a broad
   nonconstant tail channel.

Use one-factor-at-a-time calibration; do not run a Cartesian sweep. Freeze the
map, stabilization constant, fit and held-out rows, weights, rank, ALS order,
and stopping schedule before each claim replay. A candidate enters the
recursive horizon only after its held-out and direct-normalizer gates pass.

Gate: finite and normalized representation, held-out central and shell
residuals, direct \(Z_H\) versus \(Z_T\) agreement, Gram eigenvalue margin,
and recorded TT rank/ALS residual. Training RMS alone cannot pass this gate.

**Phase 5A entry refresh (2026-09-04).** The first bounded arm is specified
in [`c2-mixture-ukf-apf-phase5a-hermite-pilot-20260904.md`](c2-mixture-ukf-apf-phase5a-hermite-pilot-20260904.md).
It freezes the Phase 5 map and tests Hermite degrees 6, 8, and 10 on disjoint
training, holdout, and audit banks.  Its primary direct \(Z_T\) estimate
samples the normalized predictive mixture and averages the exact observation
likelihood.  A Gaussian-reference \(\gamma/\eta\) estimate is retained only
as a tail-variance diagnostic, since the initial smoke exposed substantial
reference-weight variability.  A valid but poor Hermite result opens the
RBF/hybrid arm; it does not veto the generic research direction.

### Phase 5B-RBF: fixed-map RBF representation

The fixed-map RBF pilot is governed by
[`c2-mixture-ukf-apf-phase5b-rbf-pilot-20260904.md`](c2-mixture-ukf-apf-phase5b-rbf-pilot-20260904.md).
Its paired GPU/XLA run passed the exact target/map, analytic mass and
integral identities, quadrature audit, finite/SPD, and fitter checks.  The
width records remain descriptive: width `1.5` had the smallest observed
holdout and shell residuals, while width `3.0` approached the conditioning
veto.  No width is promoted and no proposal-efficiency or recursive claim is
made.  The close note is preserved under
`phase5b-rbf-attempt03/phase5b-close-20260904.md`.

### Phase 5C: Hermite-plus-RBF hybrid

Open a bounded fixed-map hybrid diagnostic only after the RBF mechanics pass.
The executed pilot is specified in
[`c2-mixture-ukf-apf-phase5c-hybrid-pilot-20260904.md`](c2-mixture-ukf-apf-phase5c-hybrid-pilot-20260904.md)
and closes at
`phase5c-hybrid-attempt03/phase5c-close-20260904.md`.
Use one normalized Hermite degree, a fixed RBF offset grid, and exactly one
constant channel (the Hermite constant; the RBF block is nonconstant).  The
hybrid must provide analytic Hermite/RBF cross-Gram entries and cross-integrals
under the same standard-normal reference measure, with independent quadrature
checks before fitting.  Keep the C2 target, lagged map, rows, predictive
`Z_T`, rank, sweeps, ridge, and no-promotion interpretation unchanged.  A
valid hybrid opens replicated/recursive representation validation; a poor
hybrid is a candidate result and does not veto the generic method.

**Phase 5C execution close (2026-09-04).** The GPU/XLA hybrid pilot passed
the exact target/map, two-order full and cross quadrature checks, and the
executable endpoint-wiring check.  The degree-6 widths `0.75` and `1.5` were
finite and fit-valid; width `3.0` was retained as a candidate conditioning
failure (`2.57e16`).  The descriptive width comparison is not statistically
ranked, and the hybrid width-1.5 normalizer gap remains nonzero.  See
`phase5c-hybrid-attempt03/phase5c-close-20260904.md`.

### Phase 5C-R: replicated and recursive hybrid validation

Run a fresh bounded replication using the surviving hybrid arms, multiple
disjoint bank seeds, and a short multi-step horizon.  Keep the exact C2
target, lagged moment map, predictive-mixture `Z_T`, and no-promotion rule
unchanged.  Report paired uncertainty intervals, per-step shell residuals and
`Z_H` gaps, condition/eigenvalue margins, and cumulative recursive errors.
This phase can nominate an arm for the later integrated candidate; it cannot
promote a default from a small replication.

**Phase 5C-R execution close (2026-09-04).** The refreshed GPU/XLA attempt02
passed the exact target/map, independent quadrature, bank-separation,
fixed-signature, and hybrid endpoint checks for all `3 x 2 x 2 = 12` records
with the plan-matched `512/512/4096` row contract.  The focused CPU regression
reported `27 passed`; the scoped Lean certificate exited `0`; and MathDevMCP
returned the same value-level scope limitations documented in the artifact.
Attempt01 is retained as pre-refresh evidence because its launch plan declared
1,024 audit rows while the driver used 4,096; the attempt02 rerun is the
authoritative contract-matched record.  Widths `0.75` and `1.5` remain
candidate arms only: three seed clusters and two steps do not support a
ranking.  Continue to the integrated exact-DMIS diagnostic with both arms
recorded and width `1.5` used only as a warm-start nomination.

### Phase 5D: reference-density rewrite

Only if a Student or mixture proposal remains useful after Phases 2--5C,
implement the Student reference as a new route identity. The first candidate
is the product law \(r^{\mathrm{St},\nu}\); an elliptical Student law is a
separate non-product route. Rebuild the mass matrix, Gram/marginal
contractions, row-sampling law, pulled-back target, normalizer, and score
together. Require \(\nu>2d\) for a degree-\(d\) polynomial mass matrix, or
replace the polynomial contraction with a separately justified finite
construction.

Gate: mass and Gram parity against an independent reference, support and
normalization, held-out/shell error, direct normalizer, finite total or
explicitly frozen score, and no hidden Gaussian fallback.

### Phase 6: global GMSPPF-style cost arm

After local proposals are characterized, test a fixed-capacity global Gaussian
mixture and SPKF bank with deterministic EM/WEM reduction. This is a cost and
scalability comparison, not an exact-target shortcut. Any approximate
predictive mixture used by the published GMSPPF construction must not replace
the exact f_t g_t numerator or the complete proposal denominator in this
repository's target evaluator.

### Phase 7: integrated candidate, replicated decision run, and terminal audit

Combine only mechanisms that independently pass their earlier gates:
recursive map, selected basis/reference, smooth gate, and exact DMIS. Run at
most twelve paired branches on disjoint calibration/claim partitions. Report
raw branch values, paired bootstrap intervals, a sign test, all heuristic
comparisons, compile versus evaluation time, and the full run manifest.
Perform a terminal code call-chain audit, mathematical audit, and post-run
red-team note. The result can nominate a follow-up study; it cannot promote a
default without a separate review and evidence contract.

**Phase 7 execution close (2026-09-04).** Attempt02 completed with all `33/33`
exact proposal records and `24/24` recursive-control cells valid on the RTX
4080 SUPER with XLA and verified memory growth. The focused suite reported
`60 passed`, the scoped Lean file compiled, and MathDevMCP structurally matched
the complete-DMIS weight equation. Every UKF family nevertheless loses to a
cheap heuristic at at least one declared salient time, while retained TT has
minimum ESS only `4.01` to `5.47` out of `8192`. This is a candidate/promotion
failure, not a target, denominator, map, or continuation failure. See
`../benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase7-integrated-attempt02/phase7-close-20260904.md`.

The all-time table localizes the common UKF collapse to time 14, which was not
one of the four predeclared map checkpoints. The transformed guide maps one
near-zero observation to an extreme log-square innovation, although the exact
likelihood state score remains bounded and its curvature tends to zero. The
proposal is observation-responsive but responds incorrectly in that
transformation-tail regime. Future comparisons must report all times before
selecting event-based checkpoints.

### Phase 8: exact-likelihood local guide repair

Phase 8 is a new candidate-method plan, not a silent Phase 7 tuning repair.
Its active plan is
[`c2-exact-likelihood-laplace-mixture-apf-phase8-20260904.md`](c2-exact-likelihood-laplace-mixture-apf-phase8-20260904.md).
Replace the Gaussianized log-square measurement closure with a fixed-topology
local proposal whose location and covariance are constructed from the exact
likelihood's state score and curvature. Use fixed iteration and temperature
schedules, retain a full-support Student defensive component, and evaluate the
same exact target with the same complete conditional DMIS denominator. The
first analytical-score route freezes the resulting proposal snapshots; a
separate adaptive-total route would owe all proposal-construction derivatives.

Before any C2 decision run, require a generic derivative-provider interface,
linear-Gaussian exactness, a bounded near-zero-observation limit check,
non-log-concave fail-closed behavior, complete-density normalization and label
invariance, a known multimodal fixture, full-time heuristic comparisons, and
fixed-signature graph-factory reuse. A dedicated plan must set a new bounded
compute budget because the original campaign budget did not reserve a new
method family.

### Phase-close repair, refresh, and continuation protocol

Every phase closes before the next phase is launched. The close is a research
record, not an informal verbal decision, and it uses a fresh versioned output
directory. The phase owner must write a result note containing the command
actually run, source/commit and dirty-state identity, environment and device
settings, seeds, budget consumed and remaining, every required diagnostic, and
the following disposition for each failed check:

1. **Classify the failure.** Use one of `infrastructure_or_harness`,
   `implementation`, `numerical_validity`, `target_or_math`, `tuning`,
   `candidate_failure`, or `continuation_veto`. A low ESS, a heuristic
   dominance veto, a poor held-out residual, or a candidate losing to a cheap
   adversary is a candidate/promotion failure, not automatically a
   continuation veto.
2. **Repair what is local and repairable.** An infrastructure, wiring,
   serialization, or localized numerical repair may be made immediately when
   the target, data, method contract, hardware class, promotion criteria,
   vetoes, and total campaign budget are unchanged. Record the failed attempt,
   the smallest repair, and a focused regression before retrying. A numerical
   protection that changes the computed object (for example, damping, a ridge,
   or a trust cap) is a Class-C candidate: it receives its own calibration and
   non-harm check and is never silently inserted into the baseline.
3. **Refresh the next phase.** Before continuing, update the next phase's
   entry settings with the repaired API, selected frozen controls, unresolved
   risks, artifact paths, remaining budget, and exact command. Do not transfer
   claim-data tuning, relax a threshold after seeing a result, silently change
   the proposal denominator, or reuse a failed output directory. If a repair
   changes the target, measure, data partition, hardware class, method family,
   promotion rule, or budget, stop and issue a reviewed plan revision instead
   of calling it a repair.
4. **Continue when no real blocker remains.** If validity gates pass and no
   stated continuation veto fires, launch the refreshed next phase even when a
   candidate was rejected or a promotion criterion was missed. Preserve the
   rejected candidate as evidence and run the predeclared next candidate or
   repair. A reviewer timeout or a MathDevMCP/Lean tool limitation is recorded
   as a limitation; it is not by itself a continuation veto under the repository
   governance profile. A material mathematical, target, or implementation
   finding remains a veto until resolved.
5. **Stop only for a true continuation veto.** Stop the campaign (and ask for
   direction when the choice is material) for a target/measure mismatch,
   unsupported or nonfinite finite program, failed required exact identity or
   call-chain parity, corrupted or missing evidence, an unresolvable source
   contract, exhausted budget, or a change requiring new authority. The result
   note must say whether the veto invalidates the harness, implementation,
   target/math, or only the current candidate; it must name the smallest next
   discriminating check.

The close note uses this compact decision table for every phase:

| Field | Required entry |
| --- | --- |
| Phase and attempt | phase identifier, attempt directory, exact command |
| Result | pass, candidate/promotion veto, repair trigger, or continuation veto |
| Failure class | one of the seven classes above for every failed check |
| Repair | change made, why the scientific contract stayed fixed, focused regression result |
| Next-phase refresh | frozen settings, unresolved risks, entry gates, command, and remaining budget |
| Continuation | `CONTINUE_NO_REAL_BLOCKER` or the precise veto identifier |
| Evidence | manifest, raw records, result note, and checksums |

The next phase may not be marked ready until this table is complete. A failed
candidate is never silently promoted, and a failed candidate is never allowed
to veto the whole research direction when the finite program remains valid.
This protocol applies equally to smoke, serious, representation, reference-law,
and integrated phases; it is the mechanism that keeps the program moving while
preserving a reproducible account of every repair and decision.

The following matrix makes the same rule concrete for each planned phase. The
last column lists conditions that stop the campaign; a candidate losing its
promotion comparison is deliberately absent from that column.

| Phase | Repair trigger | Allowed repair and refreshed next action | True blocker |
| --- | --- | --- | --- |
| 0 preflight | fixture, shape, wiring, or provenance check fails | repair the fixture/adapter or test harness, rerun the focused check, and refresh the Phase 1 API; preserve the failed preflight | target or observation convention cannot be made identical, or required evidence is corrupted/missing |
| 1 UKF/K=1 | Kalman parity, covariance, derivative, or observation-sensitivity check fails | repair batched kernel shapes, sigma-point arithmetic, factorization, or call-chain wiring under the same contract; rerun parity before Phase 2 | no finite exact proposal/weight program or failed required call-chain parity after the repair budget |
| 2 serious K=1 | complete-DMIS precision or ESS/promotion screen fails | repair denominator/weight assembly or scope-specific tuning when validity fails; for low ESS or candidate loss, retain the result and refresh Phase 3 with the valid K=1 baseline | target numerator/denominator mismatch, nonfinite program, or exhausted repair/budget allowance |
| 3 K=2/K=4 | component collapse, SPD loss, or label-invariance failure | recalibrate the predeclared split on calibration data, reject only the invalid K arm, and refresh Phase 4 with the surviving arm(s) | no valid component topology or failed exact identity/call-chain checks |
| 4 gates/defense | gate is uniform/nearly hard, Student defense dilutes the bulk, or score parity fails | tune temperature/floor/tail only on calibration data, repair complete-density evaluation, and refresh Phase 5 with a valid fixed gate; otherwise carry K=1 forward | unsupported proposal, hidden fallback, or an unresolvable gradient/measure contract |
| 5 recursive map | map factorization or cumulative error fails | repair lagged moment assembly or map factorization and rerun the linear fixture; if recursion amplifies error, reject the map and refresh the current fixed-map representation phase with the unchanged target | nonfinite/unstable map with no valid frozen alternative, or target change required |
| 5A/5B-RBF basis | held-out/shell or direct-normalizer gate fails, or Gram is ill-conditioned | repair basis normalization/constant duplication or select the next predeclared degree/width arm on calibration data; retain the predictive-mixture (Z_T) comparator and refresh the next representation arm | no finite normalized representation or failed independent mass/normalizer check |
| 5C hybrid | analytic cross-Gram/integral parity, SPD, or fit validity fails | repair the coupled Hermite/RBF contractions and rerun the focused quadrature/Lean checks; if valid but poor, retain the result and refresh replicated/recursive validation with the best non-promoted arm | no finite normalized hybrid or failed independent mass/normalizer check |
| 5C-R hybrid replication | replicated arm loses uncertainty/recursive validity screen | repair bank separation or recursive wiring under the unchanged target and rerun; retain candidate-level losses and refresh the integrated phase with any surviving arm | no valid replicated arm, target/map mismatch, or missing uncertainty/records |
| 5D reference law | Student mass, row law, pullback, or score parity fails | repair the coupled Student route; if moments do not exist or consistency cannot be restored, reject Student TT and refresh Phase 6 with the Gaussian reference | inconsistent target measure or unsupported/nonfinite normalizer |
| 6 global mixture | reduction, cost, or scalability arm fails | repair deterministic reduction/harness and rerun within budget; a costly or weak candidate is retained as a negative result and Phase 7 uses the passing local arm | invalid exact-target evaluator, corrupted records, or exhausted budget |
| 7 integration | combined route loses a gate, heuristic, or replication criterion | repair wiring/manifest or run the predeclared surviving combination; report candidate rejection and close with a bounded follow-up, without changing the target | unresolved validity failure, missing terminal audit, or any new default/scientific claim requiring review |

## 6. Baseline ladder and heuristic-dominance gate

Every eligible arm uses the same observations, target, particle budget, seeds,
auxiliary convention, and claim partition:

1. exact linear-Gaussian Kalman conditional proposal (mechanics oracle);
2. bootstrap conditional proposal;
3. current transformed-observation Student proposal;
4. current Gaussian-hint proposal;
5. stationary Gaussian independence proposal;
6. retained TT proposal;
7. K=1 per-ancestor UKF/APF;
8. K=2 and K=4 local mixtures; and
9. the fixed Student-defensive variants.

The canonical, freshly tuned LEDH-PFPF-OT route may be added as a separately
identified comparator only after its own conformance artifact is available.
Historical LEDH outputs and any reduced lane are excluded from this ladder.

At a fixed proposal and map, the representation ladder additionally contains
Hermite \(d=6,8,10\), fixed separable RBF, and Hermite-plus-RBF arms. The
reference ladder contains Gaussian, product-Student, and Gaussian--Student
mixture measures only after their consistency gates pass. These representation
and reference arms are not silently ranked by proposal ESS; their primary
quantities are held-out error, direct normalizer agreement, and mass/Gram
validity.

The practical cheap adversary set is bootstrap conditional, transformed
Student, Gaussian-hint, and stationary Gaussian. Evaluate them conditionally
at t=3, t=4, the predeclared minimum-ESS time, the largest transformed
innovation time, and the full horizon. A complex candidate losing to a cheap
adversary in a salient situation is a promotion veto, even if an unconditional
mean looks favorable. Retained TT is a separate representation comparator and
remains in every proposal ladder, but it is not described as a cheap adversary.

## 7. Diagnostics and inference roles

### Hard vetoes

- target or observation-convention mismatch;
- incomplete conditional or marginal mixture denominator;
- nonnormalized ancestor/base-mass rows;
- nonfinite target, score, covariance, scale, or log density;
- failed linear-Gaussian parity, same-scalar finite difference, or
  eager/graph/XLA parity;
- failed plain complete-DMIS precision screen for a serious claim: the
  cross-scramble 95% half-width of the declared plain log-normalizer must be
  at most \(0.00125\) nats at every captured time and row count. This is the
  inherited C2 threshold; Phase 0 must confirm its scope before use, and it
  cannot be silently relaxed after seeing a result;
- invalid snapshot identity or call-chain wiring;
- unsupported samples or non-positive proposal scale;
- GPU memory-growth/XLA provenance failure on a serious GPU run; or
- exhausted declared budget.

### Promotion criterion for this diagnostic program

After hard gates pass, a candidate must have a predeclared paired 95%
uncertainty interval for

\[
  \log\!\left(\frac{\min_t ESS_{\mathrm{candidate},t}}
                       {\min_t ESS_{\mathrm{baseline},t}}\right)
\]

that excludes zero on the positive side, with at least 10 of 12 paired
contrasts positive, and must not lose to a cheap adversary in a salient
situation, and must pass the finite-target/reference-compatibility screen.
This only supports a bounded mechanism nomination.

### Explanatory diagnostics

Held-out and shell residuals, direct \(\log Z_H-\log Z_T\), Gram conditioning,
TT rank, component overlap, maximum normalized weight, per-time ESS, APF
lookahead variation, proposal sensitivity to the observation, map condition,
runtime, and compile cost are explanatory unless explicitly promoted above.
The plain-DMIS half-width is a validity/promotion gate, not a tuning target on
the claim branches. None of these quantities may be optimized after looking at
the claim branches.

### Nonclaims

This program does not establish exact pseudo-marginal likelihoods, exact
posterior inference, HMC readiness, universal proposal efficiency, a new
production default, source-faithful Zhao--Cui reproduction, or superiority on
models outside the declared scope. A frozen branch is a deterministic finite
program; a randomized-likelihood claim requires a separate unbiasedness and
posterior study.

## 8. Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| Finite carried target \(\gamma_t\) | existing C2 DMIS contract | silently substitutes a fitted density or wrong Jacobian | independent target assembly and coordinate-convention parity | required |
| Per-ancestor UKF | SPPF/UPF literature; primary hypothesis | expensive or inaccurate local moments | linear-Gaussian parity and observation-sensitivity test | candidate |
| APF lookahead | Pitt--Shephard framework plus UKF predictive moment | w/a omitted or lookahead uninformative | ancestor-law and weight-identity fixture | candidate |
| Complete conditional/marginal denominator | importance-sampling identity | selected-component or mixed numerator/denominator error | label permutation and conditional/marginal negative controls | required |
| K=2,4 symmetric split | project extension | duplicate/narrow components or SPD loss | eigenvalue margin and label permutation | extension |
| Transformed C2 observation | required because raw gain is zero | log-square closure misses shape | raw-zero negative control and shell diagnostics | model adapter |
| Student defense | prior tail hypothesis | dilutes bulk or still misses modes | independent pilot second moment and radial tails | candidate |
| Smooth softmax/sigmoid gate | differentiable data-dependent proposal hypothesis | temperature/floor can make the gate nearly uniform or overly sharp | per-ancestor weight variation, temperature sensitivity, and finite difference | candidate |
| Hermite/RBF/hybrid basis | earlier coherent C2 plan; representation hypothesis | higher degree or duplicate constants can make Gram ill-conditioned | held-out/shell residual, Gram eigenvalues, and cross-Gram parity | candidate |
| Student TT reference | earlier coherent C2 plan; tail-measure hypothesis | missing moments or inconsistent row law changes the represented program | \(\nu>2d\) check, independent mass-matrix integral, and normalization | deferred candidate |
| Fixed topology and frozen rows | analytical-gradient contract | differs from an adaptive algorithm | finite difference and manifest identity | required |
| Covariance jitter/ridge | zero alteration initially | factorization failure or silent target change | no-fire healthy regression; separate calibration if nonzero | Class-C candidate |
| Float64 correctness lane | existing C2 diagnostic convention | production TF32 behavior differs | separate future precision plan | diagnostic |
| Twelve paired branches | bounded uncertainty pilot | tail ranking remains noisy | interval width and raw branch table | pilot |

No entry in this table is a universal default. Any change to target,
conditioning, component family, topology, precision, or budget requires a plan
revision before the affected run.

## 9. Pre-mortem and repair rules

| How the run could mislead | Distinguishing check | Repair or interpretation |
| --- | --- | --- |
| UKF appears data-guided but all ancestors receive nearly the same proposal | per-ancestor mean/covariance and lookahead variation over changed observations | repair adapter or reject localization claim |
| Selected component density is used instead of q | label permutation and direct log-sum-exp recomposition | hard veto; repair shared denominator |
| ESS improves only because of the Student tail | Student-only and local-only arms at each salient time | report retained TT as unnecessary; do not call mixture superior |
| More components improve training RMS but not held-out target error | held-out central/tail residual and direct normalizer | reject representation claim |
| A smooth gate is differentiable but effectively uniform or nearly hard | per-ancestor entropy, temperature/floor sweep on calibration data, and finite-difference check | retain fixed gate only as a diagnostic or reject the localization claim |
| Recursive map lowers RMS while amplifying later error | per-time cumulative error and map margins | stop recursion arm; retain one-step result |
| Student reference appears to help because its mass matrix is truncated or falls back to Gaussian | independent mass-matrix integrals, moment-existence check, and route-identity assertion | hard veto the reference arm; retain Student only as a proposal |
| Gradient is a partial derivative through frozen moments | central finite difference of the exact declared program | relabel score or implement total derivative |
| Successful GPU command hides allocator or device mismatch | pre-initialization memory-growth and placement manifest | hard veto serious run |

Low ESS alone is a repair trigger when all validity gates pass. A failed
candidate does not reject the SPPF/UPF research direction.

## 10. Environment, budget, and artifact contract

### Environment

- TensorFlow/TFP is the implementation backend.
- Serious GPU runs use the `tftwogpu` environment, trusted GPU access,
  `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth before device
  initialization, float64 for the correctness ladder, and XLA enabled by
  default.
- CPU-only checks set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and
  are labeled diagnostic/reference runs.
- No pfor or implicit pfor derivative path is permitted without a separately
  recorded approval; use native TensorFlow loops or explicit batched kernels.

### Budget

- one Phase 0 preflight;
- at most three localized implementation/harness repairs;
- smoke ladders at N=256 and N=1024;
- one Phase 2 GPU entry pilot at (N=8192), one branch per family;
- one Phase 2 serious GPU expansion at (N=8192), twelve paired branches;
- one Phase 3 fixed-split GPU expansion at (N=8192), twelve paired branches,
  with a six-row disjoint calibration ladder and K=2/K=4 claim arms;
- one Phase 4 smooth-gate/Student-defensive GPU expansion at (N=8192), twelve
  paired branches, with a twelve-row disjoint `(nu, epsilon)` calibration
  ladder and K=1/K=2/K=4 defensive arms;
- expansion to at most twelve paired branches and the larger rows only after a
  measured-cost refresh;
- conditional basis/reference pilots are capped at eighteen one-factor arms;
  no Cartesian product of proposal, map, basis, and reference settings;
- six GPU-hours and four CPU-hours for this campaign; and
- a fresh output directory for every attempt under
  `docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/`.

### Command contract

Phase 0 must create and freeze the driver interface before implementation
claims are made. The required CPU preflight command is:

    CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python -m pytest -q \
      tests/highdim/test_c2_coherent_plan_math.py \
      tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py \
      tests/highdim/test_c2_phase2_generic_dmis_repair_wiring.py

After the driver and its Phase 0/1 tests exist, the frozen smoke, Phase 2 entry,
and bounded serious commands are:

    CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase phase0 --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase0

    CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase1-smoke \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase smoke --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase1-smoke-attempt03 \
      --rows 256,1024 --branches 1

    TF_FORCE_GPU_ALLOW_GROWTH=true \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase phase2-entry --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-entry-attempt01 \
      --rows 8192 --branches 1

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase serious --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-serious-n8192-attempt01 \
      --rows 8192 --branches 12

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase phase3 --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase3-fixed-split-n8192-attempt01 \
      --rows 8192 --branches 12

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase4-serious-attempt01 \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_20260902.py \
      --phase phase4 --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-smooth-defensive-n8192-attempt01 \
      --rows 8192 --branches 12

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase5-map \
    /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
    python docs/benchmarks/run_c2_mixture_ukf_apf_phase5_recursive_map_20260904.py \
      --output-root \
      docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5-recursive-map-attempt02 \
      --rows 128 --horizon 3

The first command is a CPU reference exception. The Phase 2 entry and serious
commands require trusted GPU access, pre-initialization memory-growth
verification, XLA by default, and fresh output directories. The serious command
selects physical GPU 1 with PCI-bus ordering because GPU 0 carries an unrelated
desktop workload. The first serious
implementation is deliberately limited to the measured `N=8192` row and twelve
paired branches; the Phase 3 command uses the same hardware and row cap, with
offset and defensive controls restricted to independent calibration banks.
The Phase 4 command adds a fixed Student defensive component and smooth
innovation gate while retaining the same exact target and score evaluator.
The Phase 5 command is a bounded generic map mechanics/integration probe; it
does not run the closed `N=8192` proposal ladder. Larger rows or a fitted TT
integration require a new cost/memory refresh and plan update.

### Required artifacts

Each serious attempt must preserve the plan hash, git commit and dirty-state
summary, exact command, environment, device/memory/XLA policy, model and data
identities, seeds, proposal/topology settings, source hashes, timing,
`result.json`, `result.md`, logs, and all raw branch records. Result notes must
include a decision table, inference-status table, and post-run red-team note.

## 11. Skeptical pre-execution audit

This refresh was audited before execution for the required failure modes:

- **Wrong baseline:** prior retained-TT, bootstrap, Gaussian-hint, Student, and
  stationary arms are retained; the exact Kalman fixture is a mechanics oracle,
  not a claim comparator for nonlinear C2.
- **Proxy promotion:** ESS, residuals, and pilot objectives are explicitly
  separated from hard correctness and promotion criteria.
- **Missing stops:** target, support, denominator, gradient, call-chain,
  device, and budget vetoes are explicit.
- **Unfair comparison:** shared observations, target, seeds, particle budget,
  auxiliary convention, and disjoint calibration/claim partitions are required.
- **Hidden defaults:** component count, split offset, tail parameters, gate
  temperature/floors, basis family, reference law, jitter, topology,
  precision, and branch count are listed with provenance and early diagnostics.
- **Mathematical split check:** the K=4 covariance identity is stated with the
  correct condition (SPD of the common covariance); orthonormality is not
  incorrectly required.
- **Measure consistency:** the reference-density section requires the mass
  matrix, row law, target pullback, normalizer, and score to change together.
- **Executable scope:** the command contract separates CPU reference checks
  from trusted GPU runs and requires fresh versioned output directories.
- **Stale context:** historical incompatible TT and pre-policy LEDH results are
  not reused as claim evidence.
- **Unanswered implementation claim:** the plan requires an executable
  call-chain/wiring test; no function-existence check is treated as sufficient.

Audit disposition before execution was
`PASS_FOR_BOUNDED_PHASE0_AND_IMPLEMENTATION_DESIGN`. The Phase 0/1 execution
record and the MathDevMCP/Lean record are now linked from the phase-close
artifacts. The historical document rigor scan remains partial and advisory.

**Phase 2 entry audit (2026-09-03, before launch).** The entry command is
bounded to one fresh fit, six declared families, one paired branch, and
`N=8192`; it cannot overwrite an existing root and it rejects a non-XLA
configuration. The exact target, complete conditional denominator, and shared
analytical-score evaluator are fixed across candidates. GPU memory growth is
configured and verified before the placement probe. Partial branch records are
written after each candidate, and setup/fit failures are preserved for repair.
The K=1 validity checks are the only entry promotion gate; ESS and
log-likelihood contrasts are explanatory/descriptive, and comparator failures
cannot be relabeled as K=1 failures. The pilot measures actual fit and branch
cost before any row/branch expansion, so the original six-GPU-hour assumption
is not silently used. This passed the skeptical audit for the bounded entry
pilot. The serious expansion was separately audited below before launch.

**Entry attempts 01--02 repair (2026-09-03).** Attempts 01 and 02 failed before
setup because this TensorFlow build initialized logical GPUs before the
repository helper could call `set_memory_growth`. Attempt 01 exposed the
environment-override ordering; attempt 02 showed that restoring the override
before importing the high-dimensional package was still too early. Both are
classified as the same localized infrastructure-ordering failure, not a
proposal or target result. The driver now defers the inherited environment
value and all high-dimensional algorithm imports until after the policy helper
has verified both GPUs. A fresh unchanged-contract retry is
`phase2-entry-attempt03`.

**Phase 2 entry close (2026-09-03).** The unchanged-contract retry passed all
required K=1 validity checks and completed all six candidate records on GPU;
the close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-entry-close-20260903.md`.
The measured fit and branch costs imply approximately 1.47 GPU-hours for twelve
`N=8192` branches, but that extrapolation is descriptive. The next action is the
bounded serious `N=8192`, twelve-branch run. Its promotion gate is validity and
complete record coverage; ESS, log likelihood, and timing remain descriptive.

**Serious expansion audit (2026-09-03, before launch).** The expansion has one
fresh output root, a fixed row (`N=8192`), twelve paired seeds, one retained-TT
fit shared by all branches, and the same six proposal families and exact frozen
finite-program evaluator as the entry pilot. It writes a partial record after
every candidate and a terminal raw branch file, so an interruption cannot be
mistaken for a complete comparison. The primary question is whether the
multi-branch run supplies stable descriptive uncertainty and an affordable
call-chain measurement; it is not a superiority or posterior-correctness test.
The hard vetoes are nonfinite values, failed exact/APF/score/parity checks,
unsupported GPU/XLA provenance, missing branch records, corrupted artifacts,
or an exhausted budget. A low ESS or a candidate losing the heuristic ladder is
an explanatory/promotion result and refreshes the next phase rather than
stopping the research direction. This passes the skeptical audit for the
bounded serious row; larger rows remain closed until its cost and memory record
are reviewed.

**Phase 3 pre-launch audit (2026-09-03).** The repair keeps the exact C2 target,
complete conditional proposal denominator, APF correction, frozen random-input
score program, row count, paired seeds, and GPU/XLA contract unchanged. The
only selected control is the largest member of the predeclared dimensionless
offset ladder `(0.20, 0.35, 0.50)`, chosen on a separate stateless
model-generated calibration bank (`N=64`, horizon 10); claim observations are
not used for that choice. The focused CPU regression reports `31 passed`, and
the calibration smoke passes both K=2 and K=4 at every ladder value, with the
selected `delta=0.50` retaining a minimum component eigenvalue above `0.78`
and moment/label/density errors below `7e-15`. The candidate-specific gates
are SPD, moment recomposition, complete-mixture label invariance, observation
sensitivity, exact/APF/score/parity checks, and GPU output. ESS and cost are
descriptive/promotion diagnostics only. Partial records are checkpointed after
each candidate, and a fresh output root is required. No wrong baseline,
unexamined default, proxy promotion, missing stop condition, or budget change
was found, so this bounded Phase 3 launch passes the skeptical audit.

Local document checks on 2026-09-03: Pandoc GFM parsing passed; display and
inline math delimiters were balanced (16/16 and 57/57); all linked source and
plan paths were present; and `git diff --check` passed. These are document
integrity checks, not a formal mathematical certificate.

**Phase 3 close (2026-09-03).** The fresh `N=8192`, twelve-branch fixed-split
run completed `96/96` records with unique branch IDs, finite values, GPU/XLA
placement, exact-DMIS/APF checks, analytical-score finite-difference parity,
and all K=2/K=4 SPD, moment, label-invariance, and observation-response gates
passing. The selected dimensionless Cholesky-column offset was `0.50`; the
independent six-row calibration ladder passed. The close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase3-fixed-split-close-20260903.md`.
The scalar split identities were independently checked by MathDevMCP and a
Lean `ring`/positivity file; those checks do not replace the matrix executable
tests. Mean minimum ESS was descriptively 449.26 (K=2) and 367.91 (K=4),
versus 416.35 (K=1), 1108.56 (bootstrap), and 1401.73 (Gaussian hint), while
retained TT was 3.87. Thus K=2/K=4 receive a candidate-efficiency/promotion
veto, not a validity or continuation veto. Repeated TensorFlow retracing
warnings are recorded as an infrastructure/performance repair trigger. The
run consumed about 0.87 wall GPU-hours; larger rows remain closed. With no
real blocker, Phase 4 is refreshed to test a fixed-size smooth gate and
full-support Student defensive mixture under the same exact target and
complete denominator.

**Phase 4 pre-launch audit (2026-09-03).** The implementation was checked
before the serious run. The exact C2 numerator, APF ancestor law, complete
proposal denominator, frozen analytical-score evaluator, particle count,
paired-branch seed policy, and GPU/XLA lane are unchanged from Phase 3. The
new route adds only a full-support Student component and a smooth sigmoid
fraction based on the UKF innovation quadratic. The independent CPU pilot
tested four `(nu, epsilon_min, epsilon_max)` configurations across K=1, K=2,
and K=4; all twelve rows passed finite, support, SPD, moment, label, density,
and nonuniform-gate checks. The predeclared selection rule chose
`nu5_eps05_20` (worst calibration minimum ESS 14.85). The Phase 4 driver
initially exposed a comparator-key wiring mismatch; the key was corrected and
the full focused regression (`39 passed`) was rerun before this audit. The
serious run is fixed at 132 records (eleven families times twelve branches),
with partial checkpoints and a fresh output root. ESS and timing remain
descriptive/promotion diagnostics; nonfinite values, target/denominator/score
mismatch, missing records, device or memory-policy failure, and budget
exhaustion remain hard vetoes. This audit passes for the bounded GPU launch;
the pilot's low ESS is a candidate-efficiency signal, not a continuation veto.

**Phase 4 repair close (2026-09-04).** The immutable first attempt reported
five absolute APF-identity failures. The fresh scale-aware replay completed
`36/36` defensive records on GPU/XLA; all replay checks and all 36
branch/program parity rows passed. The maximum normalized identity error was
`1.1102224395112269e-16` versus the declared `32 epsilon_64` bound
`7.105427357601002e-15`, and the independent final-weight recurrence error was
`2.9297083281690912e-15`. Parent source, fixture, and frozen-control identity
all matched. The close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-backward-error-repair-close-20260904.md`.

The defensive arms nevertheless failed the predeclared ESS/heuristic screen
(primary positive contrasts 5/12, 4/12, and 4/12 for K=1, K=2, and K=4). This
is a candidate/promotion veto, not a validity or continuation veto. The raw
replay and its hashes remain immutable; the renderer metadata issue is handled
by the close note rather than by rewriting evidence.

**Phase 5 pre-launch skeptical audit (2026-09-04).** The bounded recursive-map
plan passed review. Its independent linear Kalman recursion is the mechanics
baseline; ESS and coordinate residuals are explanatory only. The generic
moment/Cholesky callable has fixed shapes, explicit finite/SPD flags, no
NumPy/pfor path, and an eager/XLA parity test. The C2 probe uses exact model
transition and observation factors with a fixed `N=128` bank and three steps;
it cannot retune or promote the Phase 4 proposal. A map, target, call-chain,
artifact, or budget failure remains a continuation veto; a noisy finite cloud
is only a repair trigger. The exact command and one-hour GPU cap are recorded
in `docs/plans/c2-mixture-ukf-apf-phase5-recursive-map-20260904.md`.

**Phase 5 execution close (2026-09-04).** The fresh GPU/XLA attempt
`phase5-recursive-map-attempt02` passed the linear oracle, exact C2
model/fixture identity, finite/SPD map, and artifact checks. The close note,
formal MathDevMCP/Lean sidecar, and raw records are preserved under the
Phase 5 artifact root. The result opens Phase 5A; it does not rank the C2
cloud or alter the Phase 4 promotion veto.

**Phase 5A execution close (2026-09-04).** The Hermite degree-6/8/10 pilot
passed its finite target/map and fitter mechanics checks.  After the missing
C2/Hermite sources were restored from the preserved snapshot, a current-tree
CPU replay reproduced the recorded predictive normalizer and degree records
to roundoff.  The direct Gram normalizer and shell behavior were not a
monotone accuracy ladder, so no degree is promoted.  The provenance repair and
GPU/XLA result remain under the Phase 5A artifact root; the result opens the
RBF arm.

**Phase 5B-RBF execution close (2026-09-04).** The paired-seed GPU/XLA RBF
pilot passed analytic mass/integral, quadrature, target/map, finite/SPD, and
fitter gates for widths `0.75`, `1.5`, and `3.0`.  Width `1.5` is
descriptively favorable on this one bank, while width `3.0` is strongly
conditioned; neither is promoted.  See
`phase5b-rbf-attempt03/phase5b-close-20260904.md` for the decision and
inference tables.  The next bounded action is Phase 5C's analytic
Hermite-plus-RBF cross-Gram pilot.

**Phase 8A execution close (2026-09-04).** The generic exact-likelihood
Laplace kernel and C2 analytical score/curvature adapter passed 15 focused CPU
tests, the scoped Lean certificate, and the fresh attempt-03 GPU/XLA mechanics
smoke. Exact Gaussian recovery, the near-zero C2 limit, fail-closed curvature,
complete-density permutation checks, one-trace graph reuse, frozen-score finite
differences, and XLA parity all passed. The audit corrected the sign
convention: the implementation traces the nondecreasing log-posterior
\(\mathcal L=-\Phi\), while \(\Phi\) is the positive negative-log objective.
This closes mechanics only and opens Phase 8B calibration; no efficiency or
promotion claim is made. See
`../benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8a-mechanics-attempt03/phase8a-close-20260904.md`.

**Phase 8B execution close (2026-09-05).** The calibration selected the
`quarter_long` fixed schedule before ESS inspection. The admissible N=1024
comparison passed every exact-target, complete-denominator, APF identity,
finite-difference, and parity check. On the already-seen mechanism path, the
Laplace K=1 arm had minimum ESS `208.607746` and time-14 ESS `980.594322`,
versus `55.285247` for the transformed UKF arm. Attempts 01 and 02 were
localized harness repairs and are preserved as superseded evidence. The
contrast is descriptive and not promotion evidence.

**Phase 8C execution close (2026-09-05).** A TensorFlow-generated fresh C2
fixture (state seed `20260905`, observation seed `424242`) was evaluated with
three N=8192 branches for five families; the stale Phase 7 Gaussian-hint
snapshot was excluded. After repairing a branch-overwrite bug in the heuristic
table and aligning K=1 random-key offsets, all `15/15` records passed. The
candidate was nonworse than every listed heuristic at all `60` branch/time
pairs and had minimum ESS `3548.778`--`3633.416`, versus `156.148`--`293.105`
for UKF and `599.651`--`648.529` for bootstrap. This is fresh mechanism
evidence, still descriptive and candidate-only. See
`../benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase8c-paired-attempt02/phase8c-close-20260905.md`.

Phase 8D is opened for frozen-control replication on at least three additional
observation seeds. It must use the selected schedule without fresh ESS-based
retuning and must report uncertainty descriptively; no default or general
model claim is available from Phase 8C alone.

**Phase 8D entry (2026-09-05).** Phase 8D freezes `quarter_long` and opens
three additional disjoint observation seeds under
`c2-exact-likelihood-laplace-mixture-apf-phase8d-20260905.md`. The purpose is
replication and descriptive uncertainty, not ESS-based retuning or default
promotion. The stale Gaussian-hint snapshot remains excluded.

**Phase 8D recovery record (2026-09-07).** The later recovery plan records a
Phase 8D schedule-repair pass and opens Phase 8E after clean integration. The
2026-09-08 workspace reset completed that focused integration check, but did
not rerun the Phase 8D numerical experiment. The next action is therefore the
Phase 8E prelaunch audit, not another unreviewed Phase 8D launch.

## 12. Execution status and next action

Current status is `PHASE8D_RECOVERY_RECORDED_CANDIDATE_UNPROMOTED`.
The immediate readiness decision is `PHASE8E_PRELAUNCH_AUDIT_READY`.

Completed evidence supports these bounded statements:

- SPPF/UPF and GMSPPF are established method families relevant to the design.
- The existing fixed TT/Student DMIS implementation has finite-program
  engineering evidence but fails the current practical/promotion screens.
- The corrected generic-DMIS integration did not meet its precision gate; the
  later recursive-map mechanics stage is now tested independently, without
  changing that proposal verdict.
- The generic per-ancestor K=1 C2 adapter exists and passes the Phase 1 smoke
  validity gates.
- The frozen exact evaluator and analytical score agree with central finite
  differences and eager/non-JIT/XLA parity on the smoke branches.
- The early Phase 1 K=1 efficiency result was descriptively worse than its two
  cheap comparators; that result remains historical and does not describe the
  later exact-likelihood Laplace arm. The Phase 8B/8C Laplace contrasts are
  favorable but remain descriptive, with no statistical ranking supported.
- The Phase 3 K=2/K=4 fixed-topology mixtures are implemented and the serious
  `N=8192` run completed with `96/96` valid records. The independent offset
  calibration, matrix SPD, moment recomposition, complete-mixture label
  invariance, observation response, exact/APF/score/parity, and GPU checks all
  passed. The mixtures remain candidate-only because their ESS is below the
  cheap adversaries and their paired intervals do not establish a positive
  ranking against K=1.
- Smooth gating and the Student defensive mixture have now been replayed with
  valid finite-program checks; their efficiency/promotion veto is recorded in
  the Phase 4 close note. The recursive map mechanics and exact C2 wiring have
  passed; the Hermite, RBF, and hybrid representation mechanics plus the
  three-seed/two-step hybrid replication have passed their bounded diagnostics.
  The Student TT reference remains unexecuted.
  TensorFlow retracing warnings observed in Phase
  3 remain an open performance/lifecycle repair trigger, not a numerical
  validity failure.

- The smooth Student-defensive implementation passes the focused CPU and
  regression suites, and its independent calibration bank selects
  `nu5_eps05_20`; this is calibration evidence, not a claim of efficiency.
- The integrated Phase 7 run passed every declared exact-program, proposal-law,
  map, GPU/XLA, and artifact validity gate. No candidate is promoted. The K=1
  UKF arm is descriptively strong at several times but all UKF arms collapse at
  time 14, while the retained-TT proposal remains grossly inefficient.
- Phase 7 proves that the observation enters the UKF proposal and that the
  complete importance correction is implemented. It also identifies the
  remaining mathematical mismatch: the Gaussianized log-square guide
  overreacts to a near-zero observation even though the exact likelihood score
  remains bounded.

- Phase 8A proves only the generic exact-likelihood Laplace mechanics and the
  frozen C2 score boundary.
- Phase 8B selected `quarter_long` by calibration-first validity and showed a
  descriptive time-14 repair on the already-seen path.
- Phase 8C reproduced the mechanism on one fresh observation path with three
  N=8192 branches and no heuristic loss after the branch-aware comparison
  repair. It does not establish statistical superiority, posterior accuracy,
  recursive filtering error, or a default.

The next action is a skeptical prelaunch audit of
`c2-phase8e-statistical-replication-20260907.md` against the restored call chain
and current repository policy, followed by its smallest declared GPU/XLA smoke
if the audit passes. The frozen schedule must not be retuned on claim fixtures,
the exact complete-mixture weights must remain unchanged, and fixture-level
uncertainty must govern any statistical nomination. This remains a generic
candidate method, not a threshold adjustment or a model-specific
small-observation rule. The Student TT reference stays closed until its
coupled measure/basis route is implemented and audited.
