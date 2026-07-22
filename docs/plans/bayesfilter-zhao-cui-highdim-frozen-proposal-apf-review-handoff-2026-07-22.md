# Zhao--Cui High-Dimensional Architecture Review Handoff

Date: 2026-07-22

Status: `REVIEW_REQUEST_RECOMMENDATION_NOT_ADOPTED`

Intended recipient: the agent taking ownership of the Zhao--Cui lane

Required review output:

```text
docs/plans/bayesfilter-zhao-cui-highdim-frozen-proposal-apf-review-result-2026-07-22.md
```

## Assignment

Independently review, correct, and fully derive the architecture proposed in
this handoff. Audit both mathematical correctness and computational
feasibility. The review must cover the exact finite likelihood program, its
analytical score, HMC semantics, peak memory, total work, compilation behavior,
and expected scaling for a high-dimensional nonlinear structural model.

Use NAWM II as a concrete sizing and structural-analysis case, but do not claim
that BayesFilter currently implements or has tested NAWM. The repository has no
executable NAWM adapter. Extract any NAWM dimensions used in calculations from
the checked primary paper or an inspected model implementation, cite the exact
source location, and distinguish measured dimensions from hypothetical sizing
scenarios.

The recipient must not simply approve this recommendation. Try to falsify it.
If the proposal is wrong, incomplete, or inferior to a direct fixed-TTSIRT
filter, state that plainly and replace it with a fully derived alternative.

## Executive Recommendation To Audit

The recommended high-dimensional architecture is:

```text
offline fixed proposal compiler
    UKF/blockwise cubature geometry
    + actual adjacent-target density fitting
    + compact TT/TTSIRT or block-triangular proposal

online HMC likelihood evaluator
    fixed-randomness auxiliary particle filter
    + complete importance correction
    + streaming particle state
    + analytical recursive score of the same finite scalar
```

P76 should be treated as an unfinished offline proposal-compilation experiment,
not as a filtering likelihood. A direct Zhao--Cui fixed-TTSIRT filter remains a
serious competing architecture and must be evaluated fairly. The current
GenUT/Contract-E implementation remains another comparator, but dense all-pairs
optimal transport must not be a mandatory high-dimensional runtime operation.

This recommendation is an `extension_or_invention` relative to Zhao--Cui. It
must not be called source-faithful Zhao--Cui. Zhao--Cui supplies important
adjacent-density, squared-TT, KR-transport, retained-marginal, normalizer, and
proposal-correction operations. The proposed division into an offline frozen
proposal and an online importance-corrected HMC evaluator is a BayesFilter
architecture hypothesis.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Which architecture can deliver a target-consistent high-dimensional nonlinear filtering likelihood and analytical same-program score for HMC without a memory explosion? |
| Candidate mechanism | Frozen UKF-guided TT/TTSIRT proposal compiled offline, used inside a streaming importance-corrected auxiliary particle filter with a recursive analytical score. |
| Baseline 1 | Direct fixed-TTSIRT retained-density filtering with paired-core marginalization and a complete total derivative through previous marginals and transport. |
| Baseline 2 | Bootstrap or simple fixed-proposal particle filter with fixed randomness and an analytical same-program score. |
| Baseline 3 | Current GenUT/Contract-E particle reset, with dense and streaming-OT costs separated. |
| Expected failure modes | Weight collapse with dimension or horizon; TT-rank explosion; proposal support failure; incorrect score from omitted state/proposal dependence; singular structural transition; pseudo-marginal misuse; offline cost larger than HMC savings; GPU/XLA graph or memory failure. |
| Promotion criterion | A fully specified finite value program and matching analytical score; support and measure correctness; no omitted derivative terms; peak memory within a declared device budget; total work competitive under a declared HMC workload; small-model value/score certification; a dimension/horizon/particle/rank scaling forecast with checked assumptions. |
| Promotion veto | Wrong target; local complete-data score substituted for marginal score; adaptive or parameter-dependent proposal construction hidden inside HMC; finite differences or autodiff used as the admitted runtime score; dense tensor-product grids; mandatory dense \(N^2\) OT; unbounded TT ranks; unsupported NAWM dimensions; singular-transition terms written as ordinary Lebesgue densities without a valid measure treatment. |
| Continuation veto | The proposed finite scalar cannot be defined on the model support, or no tractable exact derivative of that scalar exists under any reviewed frozen-randomness formulation. Candidate rejection alone is not a research-direction rejection. |
| Repair triggers | Poor ESS triggers proposal/block/rank redesign; high memory triggers streaming or parameter blocking; singular dynamics trigger innovation-coordinate formulation; high offline amortization cost triggers simpler proposal; score variance triggers fixed control variates or independent-cloud averaging without changing the value target. |
| Explanatory diagnostics | ESS, weight entropy, maximum log-weight spread, TT heldout density loss, rank growth, proposal log-density cost, compile time, warmed time, peak allocator bytes, per-time score variance, and HMC acceptance. These are not correctness proofs. |
| Must not be concluded | No NAWM readiness, posterior correctness, HMC convergence, source-faithful Zhao--Cui status, default promotion, or superiority follows from this document. |

## Why The Existing Routes Are Insufficient

### Local complete-data SIR score

For a fixed latent path, the implemented local score is

\[
S_c(\theta;x,y)=\nabla_\theta\log p_\theta(x_{0:T},y_{1:T}).
\]

Parameter-only HMC requires

\[
\nabla_\theta\log p_\theta(y_{1:T}),
\]

or the derivative of a declared finite likelihood estimator. One conditioned
latent path is not that marginal score. The current P91 admission is scoped to
the local complete-data component and explicitly leaves previous-marginal and
fixed-TTSIRT proposal/transport derivatives open.

### Direct fixed-TTSIRT source route

The source-route mechanics contain retained-object carry and paired-core
marginal evaluation. They do not yet contain the complete analytical derivative
of the filtering scalar. Missing ownership includes the derivative through the
previous retained marginal and the fixed TTSIRT proposal/transport.

This route may ultimately be the correct architecture. The recipient must
derive and cost it, rather than rejecting it because the current implementation
is incomplete.

### P76

P76 tested UKF initialization and density training principally on the one-step,
36-dimensional Austria-SIR adjacent target \((x_1,x_0)\). Its substantive run
used degree 2, rank 4, 20 CPU batches of 128, and 2,560 fresh draws. It did not
run the full \(T=20\) filter, did not emit a filtering likelihood or marginal
score, did not test HMC, and did not evaluate a trained candidate under the
later corrected heldout density metric.

The useful P76 components are:

- UKF geometry and whitening;
- TT initialization;
- fresh actual-target minibatches;
- target/geometry bridge checks;
- density-space training and heldout-metric machinery.

These belong naturally in an offline proposal compiler.

### Current GenUT/Contract-E implementation

The current implementation materializes arrays with shapes including

\[
N\times N\times d,\qquad N\times N\times d\times p,
\qquad N\times N,\qquad N\times N\times p.
\]

In particular, the tangent difference tensor alone needs approximately

\[
bN^2dp\quad\text{bytes},
\]

where \(b=4\) for FP32 and \(b=8\) for FP64. For
\(N=5000,d=18,p=3\), this is about \(5.4\) GB in FP32 before the other dense
arrays are counted. Streaming Sinkhorn can bound peak memory, but exact
all-pairs transport still requires \(O(N^2d)\) work per transport iteration.

Gaussian GenUT is also not a universal positive-mass high-dimensional design:
the current specialization can produce negative weights above its supported
low-dimensional regime. It may remain useful blockwise or as geometry, not as
a universal full-state rule.

## Proposed Architecture

### Model and measure contract

Start from a nonlinear state-space model

\[
X_0\sim \mu_\theta,\qquad
X_t\mid X_{t-1}=x\sim M_{t,\theta}(x,dx'),\qquad
Y_t\mid X_t=x\sim G_{t,\theta}(x,dy).
\]

Do not assume that every transition has a nonsingular density with respect to
Lebesgue measure. For a model with transition density \(f_{t,\theta}\) and
observation density \(g_{t,\theta}\), the formulas below use those densities.
For singular or deterministic structural coordinates, the recipient must
derive an innovation-coordinate or mixed-measure version.

### Offline fixed proposal compiler

For a reference parameter \(\theta_\star\), or a predeclared set of anchor
parameters, construct at each time or reusable observation regime a proposal

\[
q_t^B(x_t\mid x_{t-1},y_t),
\]

where branch \(B\) binds all non-smooth or adaptive choices:

- coordinate ordering and structural blocks;
- UKF or blockwise-cubature centers and scale factors;
- basis family, domains, TT ranks, defensive mass, and regularization;
- fitting samples, random seeds, batches, optimizer, and stopping rule;
- transport/KR ordering and inverse/PDF implementations;
- any auxiliary ancestor proposal;
- support and invalid-state handling.

The fit must use the actual adjacent target or a documented proposal objective,
not UKF moments as likelihood truth. A useful conditional target is

\[
\pi_{t,\star}(x_{t-1},x_t)
\propto
\widehat\pi_{t-1,\star}(x_{t-1})
f_{t,\theta_\star}(x_t\mid x_{t-1})
g_{t,\theta_\star}(y_t\mid x_t).
\]

UKF or blockwise cubature should provide geometry and initialization. A
squared-TT/TTSIRT or block-triangular map should represent the proposal only if
it supplies:

1. sampling;
2. pointwise \(\log q_t^B\);
3. defensive full support on the model's stochastic subspace;
4. bounded and monitored rank;
5. a cost that can be amortized over the HMC workload.

The recipient must decide whether the proposal should be conditioned on
\(x_{t-1}\), use a joint adjacent map followed by a conditional KR extraction,
or use blockwise innovation coordinates. Derive all Jacobians and densities.

### Online fixed-randomness auxiliary particle filter

For particle \(i\), choose an ancestor using a frozen auxiliary distribution
\(a_{t-1}^{B,i}\). Using a fixed base random number \(u_t^i\), sample

\[
A_t^i\sim a_{t-1}^B,
\qquad
X_t^i=T_t^B(u_t^i;X_{t-1}^{A_t^i},y_t),
\]

so that \(X_t^i\) has proposal density
\(q_t^B(\cdot\mid X_{t-1}^{A_t^i},y_t)\). The unnormalized importance weight is

\[
\widetilde w_t^i(\theta)
=
\frac{
W_{t-1}^{A_t^i}(\theta)
f_{t,\theta}(X_t^i\mid X_{t-1}^{A_t^i})
g_{t,\theta}(y_t\mid X_t^i)
}{
a_{t-1}^{B,A_t^i}
q_t^B(X_t^i\mid X_{t-1}^{A_t^i},y_t)
}.
\]

With the particle states treated as fixed samples of the frozen proposal for
the finite program, define

\[
\widehat Z_t(\theta)=\sum_{i=1}^N\widetilde w_t^i(\theta),
\qquad
W_t^i(\theta)=\frac{\widetilde w_t^i(\theta)}{\widehat Z_t(\theta)}.
\]

The normalization convention must be checked carefully. Depending on whether
ancestor and proposal sampling are written as empirical mixtures or normalized
categorical laws, a fixed \(1/N\) factor may be present. It changes the value by
a parameter-independent constant but must still be defined consistently.

The finite likelihood scalar is

\[
\log\widehat L_B(\theta;u)
=
\sum_{t=1}^T\log\widehat Z_t(\theta),
\]

with an analogous initial-state term. The review must prove exactly which
probability measure this program estimates and under what conditions it is an
unbiased nonnegative likelihood estimator.

### Score derivation: state-independent frozen proposal case

The simplest score recursion applies only if the realized states, proposal
density, auxiliary probabilities, and all branch objects are held fixed with
respect to \(\theta\) during evaluation. Define

\[
G_{t-1}^j=\nabla_\theta\log W_{t-1}^j.
\]

Then

\[
H_t^i
=
G_{t-1}^{A_t^i}
+\nabla_\theta\log f_{t,\theta}
  (X_t^i\mid X_{t-1}^{A_t^i})
+\nabla_\theta\log g_{t,\theta}(y_t\mid X_t^i),
\]

and

\[
S_t
=\nabla_\theta\log\widehat Z_t
=\sum_i W_t^iH_t^i,
\qquad
G_t^i=H_t^i-S_t.
\]

Consequently,

\[
\nabla_\theta\log\widehat L_B(\theta;u)
=\sum_{t=1}^T S_t.
\]

The recipient must prove this recursion from the exact implemented
normalization convention and include the initial step.

### Critical score alternative: reparameterized state proposal

The simple recursion above is not automatically correct when

\[
X_t^i=T_t^B(u_t^i;X_{t-1}^{A_t^i},y_t)
\]

inherits parameter dependence through previous states, or when the proposal
itself is allowed to depend smoothly on \(\theta\). In that case the total
derivative includes state tangents

\[
D_t^i=\frac{dX_t^i}{d\theta}
\]

and possibly derivatives of \(\log q_t^B\), \(\log a_{t-1}^B\), and proposal
Jacobians. The receiving agent must choose and prove one of these contracts:

1. **Fully parameter-independent proposal program.** Generate the entire
   proposal genealogy from a frozen reference dynamics independent of
   \(\theta\), so all sampled states are constant during likelihood
   evaluation. Then the simple recursion is valid but importance-weight
   variance may grow rapidly away from \(\theta_\star\).
2. **Smooth reparameterized fixed proposal.** Freeze the branch while allowing
   smooth parameter dependence and propagate \(D_t^i\) plus the complete
   proposal-density derivative analytically. This is more efficient
   statistically but more expensive in memory and algebra.
3. **Blockwise or adjoint score.** Use an exact forward or reverse recursion
   that avoids storing \(Ndp\) state tangents. Derive the required state and
   parameter adjoints and show they differentiate the identical finite value
   program.

Do not omit this distinction. The earlier concise recommendation understated
it. A score that ignores parameter-dependent state paths is wrong relative to
the total derivative of that value program.

### Resampling and HMC smoothness

Discrete ancestor indices must be fixed during an HMC trajectory. The review
must define whether:

- ancestor uniforms are fixed and the categorical map is frozen at a reference
  parameter;
- an auxiliary categorical law is entirely parameter-independent;
- resampling is disabled within a trajectory;
- or another exactly specified fixed-branch construction is used.

Recomputing parameter-dependent resampling decisions during the HMC trajectory
creates discontinuities. Ignoring their dependence while changing the value
program is not an analytical total derivative.

### Pseudo-marginal HMC contract

If the estimator is unbiased and nonnegative, pseudo-marginal inference must
treat the random state \(u\) as part of the Markov state. Keep \(u\) fixed
within an HMC trajectory and refresh it through a valid independent or
correlated Markov update. Permanently fixing one \(u\) targets the deterministic
approximate posterior defined by \(\widehat L_B(\theta;u)\), not the exact
pseudo-marginal posterior.

The review must state which target is intended and derive detailed balance for
the chosen transition. It must also determine whether ordinary HMC with
discrete genealogy fixed conditionally is sufficient, or whether particle HMC,
pseudo-marginal HMC, or an alternative extended-state sampler is required.

## Memory And Performance Model Required From The Reviewer

Use the following symbols:

| Symbol | Meaning |
| --- | --- |
| \(T\) | time horizon |
| \(N\) | particle count |
| \(d_x\) | full state dimension |
| \(d_s\) | stochastic/innovation dimension |
| \(d_y\) | observation dimension |
| \(p\) | parameter dimension |
| \(p_b\) | score parameter-block size |
| \(B\) | structural block size |
| \(r\) | maximum TT rank |
| \(m\) | one-dimensional basis size |
| \(K\) | OT chunk extent, when applicable |
| \(I\) | Sinkhorn or optimizer iteration count |
| \(b\) | bytes per scalar |

### Required peak-memory derivations

Derive peak, persistent, and temporary memory separately for:

1. current dense GenUT/Contract-E forward and score;
2. streaming/chunked GenUT/Contract-E;
3. direct fixed-TTSIRT retained-density filtering;
4. proposed frozen-proposal APF with the simple score;
5. proposed APF with state tangents;
6. proposed APF with parameter blocking or an adjoint;
7. offline proposal training and TT construction.

At minimum, verify or correct these first-order estimates:

\[
M_{\mathrm{APF,simple}}
\approx
b\{c_xNd_x+c_wN+c_sNp_b+c_qd_xmr^2\},
\]

\[
M_{\mathrm{APF,tangent}}
\approx
b\{c_xNd_x+c_DNd_xp_b+c_sNp_b+c_qd_xmr^2\},
\]

\[
M_{\mathrm{dense\ OT\ tangent}}
=\Omega(bN^2d_xp_b),
\]

and a streaming-OT bound of the general form

\[
M_{\mathrm{stream\ OT}}
=O\{b(K^2d_xp_b+Nd_x+Np_b)\},
\]

while total exact all-pairs OT work remains at least quadratic in \(N\).

The constants \(c_x,c_w,c_s,c_q\) must be enumerated from actual arrays, not
left as unexplained placeholders in the final result.

For TT/TTSIRT, derive storage and operation counts from the actual core shapes,
including density evaluation, conditional CDF construction, inverse map,
normalizer, marginalization, fitting/training, and any derivative or adjoint.
Do not assume \(r\) is constant without a rank-growth diagnostic and failure
policy.

### Required work and amortization derivations

Report asymptotic operations and a hardware-relevant byte/FLOP estimate for:

- model transition and observation evaluation;
- proposal sampling and \(\log q\) evaluation;
- particle weighting and normalization;
- score recursion;
- resampling or auxiliary ancestor construction;
- TT fitting/training;
- KR conditional construction/inversion;
- dense and streaming OT;
- XLA compilation and warmed execution.

Separate:

\[
C_{\mathrm{total}}
=C_{\mathrm{offline}}
+H\,C_{\mathrm{online}},
\]

where \(H\) is the expected number of HMC likelihood/gradient evaluations.
Compute the break-even \(H\) at which the offline proposal compiler is cheaper
than each comparator. Include proposal rebuilding or anchor-refresh cost if the
HMC posterior region is too broad for one frozen proposal.

### NAWM II sizing case

The local primary paper is:

```text
/home/chakwong/DSGE/general/The new Area-wide model II an extended version of the ECB's micro-founded model for forecasting and policy analysis with a financial sector  Coenen(19).pdf
```

The paper reports 24 observed series and 24 distinct structural shocks in its
estimation discussion, but those counts are not by themselves the complete
state or parameter dimensions. Inspect the model equations, log-linear
appendix, estimation tables, and any official model code before fixing
\((d_x,d_s,d_y,p,T)\).

The final review must provide:

1. a primary-source dimension table with exact anchors;
2. a separate hypothetical stress table if some dimensions remain unavailable;
3. memory in GiB and time estimates for at least \(N\in\{1000,5000,10000\}\);
4. FP32/TF32 and FP64 cases;
5. parameter blocks such as \(p_b\in\{1,4,8,16\}\);
6. at least two TT rank/basis scenarios, including a pessimistic rank-growth
   case;
7. an RTX 4080 SUPER-class single-GPU capacity assessment and a CPU-host-memory
   fallback assessment;
8. whether stochastic dimension rather than full state dimension can control
   the particle proposal;
9. treatment of deterministic, algebraic, companion, and mixed-frequency
   state blocks;
10. expected HMC amortization count and proposal-refresh policy.

No numerical value may be labeled NAWM-derived unless its source was checked.

## Correctness Proof Obligations

The review result must contain proposition-and-proof derivations for at least:

1. **Finite scalar definition.** The exact random variables, branch objects,
   reference measures, normalization constants, and likelihood estimator.
2. **Importance-weight identity.** The Radon--Nikodym derivative for the
   proposal, including auxiliary ancestors and defensive mixtures.
3. **Unbiasedness or explicit non-unbiased target.** State precise assumptions.
4. **Recursive score identity.** Prove equality to the derivative of the exact
   finite scalar, including initialization, normalization, previous weights,
   state dependence, proposal terms, and any parameter blocking.
5. **Support validity.** The proposal dominates the model target on the
   stochastic support.
6. **Singular structural dynamics.** Give a valid innovation-coordinate or
   mixed-measure formulation for deterministic state components.
7. **Pseudo-marginal target.** State and prove the extended target and update
   invariance, or clearly choose a deterministic approximate posterior.
8. **Memory bounds.** Peak memory follows from the enumerated live tensors and
   buffers, not only Big-O notation.
9. **Work bounds.** Separate offline, per-time, per-particle, per-score-block,
   and compilation costs.
10. **TT error role.** Explain whether TT approximation affects correctness,
    estimator variance only, or both, and under exactly which importance
    correction.

## Architecture Comparison Required

The result must include a decision table comparing at least:

| Architecture | Value target | Score target | Peak memory | Total work | Main risk |
| --- | --- | --- | --- | --- | --- |
| Direct fixed-TTSIRT filter | Approximate retained-density likelihood | Full derivative through retained TT and transport | rank-dependent | rank/fitting-dependent | derivative and rank complexity |
| Frozen TT-proposal APF | Importance-filter likelihood estimator | Same finite estimator score | particle-linear unless state tangents dominate | particle-linear plus proposal cost | weight degeneracy and proposal refresh |
| Bootstrap fixed-randomness PF | Standard PF estimator | Same estimator score | particle-linear | particle-linear | severe high-dimensional degeneracy |
| GenUT/Contract-E dense OT | Current finite reset scalar | Same finite scalar JVP | quadratic in \(N\) and tangent size | quadratic in \(N\) per OT iteration | memory/work explosion |
| GenUT/Contract-E streaming OT | Same | Same | chunk-bounded | still quadratic exact all-pairs work | runtime explosion |
| Blockwise deterministic Gaussian filter | Gaussian approximate likelihood | Analytical recursion | block/dense covariance dependent | block/cubic factorization dependent | approximation bias |

The reviewer should add any better architecture identified during the audit.

## Minimal Validation Program To Propose

Do not execute a long campaign as part of this review. Propose the smallest
discriminating implementation ladder:

1. LGSSM exact-oracle value/score identity for the finite APF program.
2. Scalar exact-SV or another continuous nonlinear model with a dense reference.
3. Reduced continuous SIR with a dense or very-high-particle reference.
4. A singular-transition structural fixture in innovation coordinates.
5. Austria SIR \(d=18,T=20\), only after the target measure and comparator are
   correct.
6. A synthetic NAWM-shaped sparse/block structural model before any real NAWM
   claim.

For each rung, specify particle counts, ranks, seeds, score parameters, value
and score uncertainty, memory budget, wall-time budget, promotion criterion,
promotion veto, and continuation veto. Runtime scores must be analytical;
finite differences are diagnostic only.

## Required Source Audit

The Zhao--Cui lane has a binding paper-and-author-code anchor gate. At minimum,
inspect:

### Primary paper

```text
.local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf
```

Inspect Sections 1--5, state-space equations (1)--(9), Algorithms 1--5,
Proposition 2, Proposition 4, Theorems 7--8, Corollary 12, and Appendices A--B.
Do not rely only on the existing project summaries.

### Pinned author source

```text
third_party/audit/zhao_cui_tensor_ssm_p10/source
```

Pinned upstream commit:

```text
80034dccb99eb1d86284a1839b4a12067d13b9da
```

Inspect at least:

- `models/full_sol.m` for sequential sampling, proposal correction, retained
  marginal use, fitting, and normalizer updates;
- `models/computeL.m` for localization;
- `deep-tensor.dev/src/SIRT.m`;
- `deep-tensor.dev/src/@TTSIRT/marginalise.m`;
- `deep-tensor.dev/src/@TTSIRT/eval_rt_jac_reference.m`;
- `deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m`;
- `deep-tensor.dev/src/AbstractIRT.m`;
- `deep-tensor.dev/src/ApproxFun.m`;
- `eg3_sir/mainscript.m` for the author SIR configuration.

Exact paths may differ slightly inside the snapshot; locate and cite them.
Classify every proposed operation as `source_faithful`,
`fixed_hmc_adaptation`, or `extension_or_invention`.

### Local implementation and result anchors

Inspect at least:

- `bayesfilter/highdim/source_route.py`;
- `bayesfilter/highdim/fitting.py`;
- `bayesfilter/highdim/filtering.py`;
- `bayesfilter/highdim/ukf_initializer.py`;
- `bayesfilter/highdim/stochastic_density_training.py`;
- `bayesfilter/highdim/cubature_genut_filter.py`;
- `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`;
- `docs/plans/bayesfilter-highdim-zhao-cui-p83-phase2-transport-marginalization-design-result-2026-06-22.md`;
- `docs/plans/bayesfilter-highdim-zhao-cui-p83-phase4-analytical-derivative-audit-result-2026-06-22.md`;
- `docs/plans/bayesfilter-highdim-zhao-cui-p91-phase9-final-decision-result-2026-06-29.md`;
- `docs/plans/bayesfilter-highdim-zhao-cui-p76-phase6-bounded-minibatch-pilot-result-2026-06-18.md`;
- `docs/plans/bayesfilter-highdim-zhao-cui-p76-phase10-generated-corrected-metric-diagnostic-result-2026-06-19.md`;
- `docs/plans/bayesfilter-genut-sv-lgssm-math-code-audit-result-2026-07-22.md`.

### Literature expansion and ledgers

Because this review can change the research architecture, maintain these
ledgers in the result or companion files:

1. source-support ledger;
2. citation/venue metadata ledger;
3. backward-snowball ledger from Zhao--Cui and the key transport/PF sources;
4. forward-snowball ledger for important follow-ups, if metadata access is
   available;
5. claim-support ledger;
6. omitted-paper and reviewer-risk register.

At minimum consider literature on auxiliary particle filters, pseudo-marginal
MCMC/HMC, differentiable particle filters with fixed randomness, transport
particle filters, low-rank/factored OT, tensor-train conditional densities,
and high-dimensional particle degeneracy. Citation count or venue prestige is
coverage metadata, not correctness evidence. Record unavailable network
metadata as unavailable rather than inventing it.

## Handoff Evidence Ledger Snapshot

This compact ledger records what was checked while preparing the handoff. It
does not replace the receiving agent's full source audit.

### Source support

| Source | Classification | Local status and inspected material | Claim allowed in this handoff | Claim not allowed |
| --- | --- | --- | --- | --- |
| Zhao and Cui, *Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models*, JMLR 25 (2024) | `DIRECT_METHOD` | Local full text exists at the path above. Existing project source ledger records inspection of Sections 1--5, equations (1)--(9), Algorithms 1--5, Propositions 2 and 4, Theorems 7--8, Corollary 12, and Appendices A--B. | Zhao--Cui contains sequential TT posterior approximation, squared-TT, conditional KR, retained-marginal, proposal-correction, and error-analysis operations that the reviewer must inspect directly. | It does not establish the proposed frozen-proposal APF, HMC readiness, bounded ranks for NAWM, or BayesFilter correctness. |
| Pinned Zhao--Cui companion source, commit `80034dccb99eb1d86284a1839b4a12067d13b9da` | `IMPLEMENTATION_OR_SOFTWARE` | Manifest and every exact source path listed above were checked to exist. | Supports an author-code audit of actual solver structure and implementation operations. | It is not a mathematical oracle and cannot make a BayesFilter extension source-faithful. |
| Coenen, Karadi, Schmidt, and Warne, *The New Area-Wide Model II*, ECB Working Paper 2200, revised 2019 | `EMPIRICAL_EXAMPLE` / structural sizing source | Local 136-page full text inspected at the path above. Introduction and estimation discussion state 18 retained plus 6 added observables and 24 structural shocks. | Supports the checked facts that the estimation uses 24 observed series and describes 24 structural shocks, and that the model contains rich structural/deterministic blocks requiring source analysis. | These counts do not establish full state dimension, stochastic state dimension, parameter dimension, nonlinear runtime shape, or BayesFilter NAWM readiness. |
| Current BayesFilter P76/P83/P91/GenUT code and results | `IMPLEMENTATION_EVIDENCE` | Exact local paths are listed above and were inspected for the stated one-step scope, derivative blockers, and dense-array shapes. | Supports statements about what the current local programs compute and allocate. | Does not establish scientific correctness, general scaling, or NAWM performance. |

### Citation and venue metadata

| Source | Metadata checked | Status |
| --- | --- | --- |
| Zhao--Cui | JMLR 25 (2024), pages 1--51, from local bibliography and paper ledger | Venue metadata only; citation count not queried. |
| NAWM II | ECB Working Paper Series No. 2200, revised December 2019, from local PDF front matter | Working-paper metadata only; citation count not queried. |

No network/API metadata lookup was needed to write this handoff. Citation
counts, publisher-status updates, forward citations, and current venue metrics
are `not checked`, not zero.

### Backward and forward snowball status

| Ledger | Current status | Required next action |
| --- | --- | --- |
| Backward snowball | Existing project P1R ledgers identify direct TT filtering, TT sampling, deep inverse Rosenblatt transport, transport-map filtering, sparse-grid/cubature, and TT-rank sources. The auxiliary-PF and pseudo-marginal foundations needed by this new architecture were not fully re-audited for this handoff. | Inspect Zhao--Cui related work and the primary auxiliary-PF, pseudo-marginal, particle-HMC, and high-dimensional-degeneracy sources before adopting the architecture. |
| Forward snowball | `not checked`; no metadata query was run. | Query important follow-ups, corrections, replications, and recent citing works if network metadata is available. |

### Claim support

| Claim | Support class | Status |
| --- | --- | --- |
| Current dense GenUT score materializes \(N^2d_xp_b\)-scale tensors | `IMPLEMENTATION_EVIDENCE` | Supported by direct inspection of `cubature_genut_filter.py`. |
| P76 reached a 36D one-step SIR training pilot but not full filtering value/score | `IMPLEMENTATION_EVIDENCE` | Supported by the Phase 6 and Phase 10 artifacts. |
| Zhao--Cui supplies TT/KR filtering operations relevant to the proposal compiler | `PRIMARY_TECHNICAL_SUPPORT` pending direct recipient recheck | Supported by the existing source ledger; recipient must inspect the primary paper and author code directly for final use. |
| Frozen-proposal APF importance correction yields a valid likelihood estimator and score for all proposed structural cases | `PROJECT_DERIVATION` | Open proof obligation; this handoff deliberately does not claim it. |
| NAWM stochastic dimension can replace full state dimension in the cost model | `SOURCE_GAP_BLOCKER` | Unsupported until the NAWM equations and implementation partition are inspected. |

### Quarantine and omission risks

| Item | Status | Consequence |
| --- | --- | --- |
| Spantini et al. 2016, *Decomposable Transport Maps for Bayesian Filtering and Smoothing* | `RETRACTED_OR_QUARANTINED` in the existing project ledger following the user notice dated 2026-05-28 | Must not support this architecture. Use non-quarantined transport sources. |
| Auxiliary particle filter foundations | Important omission risk | Must be inspected before the APF weight and unbiasedness propositions are accepted. |
| Pseudo-marginal HMC and particle-HMC foundations | Critical omission risk | Must be inspected before any exact-posterior or detailed-balance claim. |
| High-dimensional particle degeneracy theory | Critical omission risk | Needed to assess whether particle-linear memory merely trades memory failure for exponential particle demand. |
| Singular-state/innovation-coordinate particle filtering | Critical omission risk | Needed for NAWM-like deterministic structural blocks. |
| Low-rank/factored/sliced OT and sparse transport | Engineering/scientific omission risk | Needed for a fair alternative if some transport reset remains online. |
| TT conditional-density rank and evaluation complexity | Critical omission risk | Needed to prevent an unsupported bounded-rank assumption. |

The hostile-review status is therefore `NOT_READY_FOR_ADOPTION`: local evidence
is sufficient to pose the architecture question and expose the main memory and
score risks, but not to certify the proposed answer.

## Default And Assumption Audit

The receiving agent must audit every material choice below:

| Choice | Current provenance | Status before review | Failure mode |
| --- | --- | --- | --- |
| UKF supplies proposal geometry | P70/P76 experiments | hypothesis | misleading geometry or invalid covariance |
| Actual adjacent target supplies training signal | Zhao--Cui/P70/P76 | recommended | target bridge or support mismatch |
| TT/TTSIRT represents the proposal | Zhao--Cui/P76 synthesis | hypothesis | rank explosion, expensive PDF/inverse |
| Proposal is frozen for HMC | fixed-branch policy | required principle | posterior region leaves proposal support |
| Importance correction restores target | standard IS identity, not yet derived here for all cases | proof obligation | missing ancestor/proposal/Jacobian term |
| Particle count \(N\) is feasible | no NAWM evidence | unknown | memory or degeneracy explosion |
| Parameter blocking preserves exact score | algebraic hypothesis | proof obligation | omitted cross-block state dependence |
| Innovation dimension controls stochastic cost | structural-model hypothesis | unknown | deterministic completion still dense/expensive |
| TT rank remains bounded | unsupported for NAWM | unknown | exponential effective cost |
| Offline cost amortizes over HMC | workload hypothesis | unknown | proposal training dominates total time |
| Single frozen anchor covers posterior | unsupported | unknown | severe importance-weight variance |
| FP32/TF32 is adequate | model-specific evidence only | unknown | weight and score cancellation |

## Skeptical Audit Of This Handoff

The plan survives review-request execution, not algorithm promotion, for these
reasons:

- The baseline ladder includes direct fixed-TTSIRT rather than comparing only
  against dense OT or a weak bootstrap filter.
- The proposed APF is explicitly classified as an extension, not Zhao--Cui
  source faithfulness.
- The score derivation identifies the key hidden assumption about frozen state
  paths and requires a corrected tangent/adjoint derivation when it fails.
- Singular structural transitions are a correctness gate, not treated as
  ordinary densities.
- Peak memory and total work are separate criteria; streaming does not make
  quadratic OT scalable.
- NAWM supplies a sizing and structural case only; the absence of a BayesFilter
  NAWM adapter is explicit.
- The requested artifact can reject the candidate without prematurely
  rejecting the research direction.

The main residual risk is literature omission around pseudo-marginal HMC and
high-dimensional particle degeneracy. The receiving agent must close or record
that source gap before recommending implementation.

## Required Result Structure

Write the result file at the exact path stated at the top. It must include:

1. direct verdict: `ADOPT`, `REVISE`, or `REJECT`;
2. corrected self-contained algorithm;
3. proposition-and-proof derivations;
4. operation classification against Zhao--Cui paper and source;
5. exact score/value target statement;
6. singular-transition and pseudo-marginal treatment;
7. peak-memory tensor ledger and performance ledger;
8. NAWM-derived and hypothetical sizing tables kept separate;
9. architecture comparison and decision table;
10. source-support, snowball, claim, and omission-risk ledgers;
11. smallest justified implementation and validation plan;
12. explicit nonclaims and unresolved blockers.

Do not edit Zhao--Cui runtime code, change defaults, or launch a serious
benchmark campaign during this review unless separately requested. The purpose
of this handoff is to obtain a rigorous independent architecture decision
before further implementation.
