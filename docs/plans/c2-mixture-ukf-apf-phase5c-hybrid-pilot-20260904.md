# C2 Mixture-UKF/APF Phase 5C: Hermite-plus-RBF Hybrid Pilot

Date: 2026-09-04  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Status: `EXECUTED_PASS_HYBRID_MECHANICS_RECURSIVE_READY`  
Scope: one-step fixed-map hybrid representation diagnostic on the C2 fixture  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Purpose and boundary

Phase 5A showed that increasing the degree of a normalized Hermite basis did
not produce a monotone shell or direct-Gram normalizer result.  Phase 5B
showed that fixed Gaussian RBF channels are mechanically valid, with a
moderate-width arm descriptively better on one paired bank but with severe
conditioning for the broadest arm.  This phase asks whether the two channel
families can be combined without changing the target or hiding a duplicated
constant direction.

This is a fixed-map representation diagnostic.  It does not test proposal ESS,
recursive filtering, posterior correctness, HMC readiness, or an analytical
total gradient.  The C2 model is a fixture only.  No degree, center, or width
is promoted by this phase.

## Exact finite target

Retain the Phase 5 lagged map and the exact one-step C2 target:

\[
  \gamma(x)=\sum_j w_j f(x\mid x_j)g(y_1\mid x),\qquad x=m+Lu.
\]

With \(\eta_d(u)\) the standard-normal probability density, fit

\[
  h_\star(u)=\gamma(m+Lu)|\det L|/\eta_d(u),
  \qquad s_\star(u)=\sqrt{h_\star(u)},
\]

on the same disjoint train, holdout, shell, and audit banks as Phases 5A and
5B.  The fitted Gram contraction is

\[
  Z_H=\int h(u)^2\,d\mu(u),
\]

and the independent predictive-mixture comparator remains

\[
  Z_T=\mathbb E_{J\sim w,\,X\sim f(\cdot\mid x_J)}[g(y_1\mid X)].
\]

The Gaussian-reference \(\gamma/\eta_d\) estimate is explanatory only; it
cannot replace \(Z_T\).

## Hybrid basis derivation

Let \(\operatorname{He}_k\) denote the probabilists' Hermite polynomial and
\[
  \psi_k(u)=\operatorname{He}_k(u)/\sqrt{k!},\qquad k=0,\ldots,d,
\]
so that \(\{\psi_k\}\) is orthonormal under \(\eta_1\).  For fixed centers
\(c_i\) and positive widths \(s_i\), define nonconstant RBF channels
\[
  \phi_i(u)=\exp\left[-\frac{(u-c_i)^2}{2s_i^2}\right],
  \qquad i=1,\ldots,m.
\]
The hybrid channel order is
\[
  b(u)=(\psi_0(u),\ldots,\psi_d(u),\phi_1(u),\ldots,\phi_m(u))^\mathsf T.
\]
There is exactly one constant channel, \(\psi_0=1\); the RBF block must be
constructed with `include_constant=False`.

### Hermite block

Orthogonality gives
\[
  \int \psi_k(u)\psi_\ell(u)\,d\mu(u)=\delta_{k\ell},
  \qquad
  \int \psi_k(u)\,d\mu(u)=\mathbf 1_{\{k=0\}}.
\]

The derivative identity used by the future gradient path is
\[
  \psi_0'(u)=0,\qquad \psi_k'(u)=\sqrt{k}\,\psi_{k-1}(u),\quad k\ge1.
\]

### RBF block

For \(a_i=s_i^{-2}\), the constant-RBF integral is
\[
 I_i=\int\phi_i(u)d\mu(u)
 = (1+a_i)^{-1/2}
   \exp\left[-\frac{c_i^2a_i}{2(1+a_i)}\right].
\]
For two RBF channels put
\[
 A_{ij}=1+a_i+a_j,\qquad b_{ij}=c_i a_i+c_j a_j.
\]
Completing the square in \(\phi_i\phi_j\eta_1\) gives
\[
 M_{ij}=A_{ij}^{-1/2}\exp\left[
 -\frac12(c_i^2a_i+c_j^2a_j)+\frac{b_{ij}^2}{2A_{ij}}\right].
\]

### Hermite/RBF cross block

Multiplying one RBF by the standard-normal density and completing the square
gives the exact tilted-normal factorization
\[
 \phi_i(u)\eta_1(u)=I_i\,\mathcal N(u;\mu_i,v_i),
 \qquad
 \mu_i=\frac{c_i a_i}{1+a_i}=\frac{c_i}{1+s_i^2},
 \qquad
 v_i=\frac{1}{1+a_i}=\frac{s_i^2}{1+s_i^2}.
\]
Define \(q_{k,i}=\mathbb E_{U\sim\mathcal N(\mu_i,v_i)}
[\operatorname{He}_k(U)]\).  The generating function
\[
 \mathbb E[e^{tU-t^2/2}]
 =\exp\left(\mu_i t+\frac{v_i-1}{2}t^2\right)
\]
implies the recurrence
\[
 q_{0,i}=1,\qquad q_{1,i}=\mu_i,\qquad
 q_{k+1,i}=\mu_iq_{k,i}+k(v_i-1)q_{k-1,i}.
\]
Therefore the cross-Gram entry is
\[
 C_{k,i}=\int\psi_k(u)\phi_i(u)d\mu(u)
       =I_i\,q_{k,i}/\sqrt{k!}.
\]
The complete one-dimensional mass matrix and integral vector are
\[
 G=\begin{bmatrix}I_{d+1}&C\\C^\mathsf T&M_{\rm rbf}\end{bmatrix},
 \qquad
 \iota=\begin{bmatrix}1&0&\cdots&0&I_1&\cdots&I_m\end{bmatrix}^\mathsf T.
\]
The implementation must return the symmetrized floating-point matrix and
must refuse measures other than `MassMeasure.REFERENCE_MEASURE`; changing the
measure would define a different finite program.

The derivative channels are concatenated as
\[
 \phi_i'(u)=-(u-c_i)\phi_i(u)/s_i^2,
 \qquad b'(u)=(\psi_0',\ldots,\psi_d',\phi_1',\ldots,\phi_m').
\]

## Frozen arms and protocol

Use Hermite degree \(d=6\), centers \((-2,-1,0,1,2)\), and the three
predeclared RBF widths from Phase 5B.  Testing all three widths avoids
choosing a hybrid width from the holdout result.

| Arm | Hermite degree | centers | width | RBF constant | role |
| --- | ---: | --- | ---: | :---: | --- |
| `hybrid_d6_w075` | 6 | `(-2,-1,0,1,2)` | 0.75 | no | narrow locality hypothesis |
| `hybrid_d6_w150` | 6 | `(-2,-1,0,1,2)` | 1.50 | no | moderate-overlap hypothesis |
| `hybrid_d6_w300` | 6 | `(-2,-1,0,1,2)` | 3.00 | no | broad-tail/conditioning hypothesis |

Keep the Phase 5A/5B paired seed `(20260904, 701)`, map seed `(20260904,
501)`, row counts, TT rank 2, two ALS sweeps, ridge `1e-8`, shell radius 2,
and exact target/map calls unchanged.  Use a new output root and do not tune
on the holdout or audit banks.

## Evidence contract

| Field | Declaration |
| --- | --- |
| Question | does the hybrid reduce fixed-map representation error while preserving exact contractions? |
| Comparator | independent predictive-mixture `Z_T` and exact C2 transition/observation calls |
| Hard validity gate | cross-Gram/integral quadrature parity for every arm; at least one arm with finite symmetric SPD mass, finite fit, condition below `1e14`, and exact target/map checks |
| Representation diagnostics | held-out central/shell RMS, `log Z_H-log Z_T`, mass and fitted-Gram condition, rank, audit residual |
| Promotion criterion | none in this pilot; all arm differences descriptive |
| Hard vetoes | source/target mismatch, cross-contraction error, no surviving finite/non-SPD-valid arm, invalid global fit, missing/corrupt records; an individual condition-vetoed arm is a candidate failure |
| Explanatory only | predictive standard error, reference-tail estimate, runtime, rank and residual comparisons |
| Nonclaims | no proposal ESS, recursive filtering, statistical ranking, default, posterior, HMC, or gradient claim |
| Artifact | fresh `phase5c-hybrid-attempt01/` with manifest, raw records, result, close note, MathDevMCP and Lean sidecars |

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| degree 6 | first Hermite arm retained from Phase 5A | underfit or poor tails | held-out/shell and normalizer records | frozen diagnostic |
| centers and widths | Phase 5B predeclared offset/width ladder | collinearity or mode miss | eigenvalues and quadrature parity | hypotheses |
| one constant only | exact duplicate-constant repair | singular mass if violated | basis dimension and smallest eigenvalue | required invariant |
| recurrence for cross block | generating-function derivation above | index/normalization error | two-order independent quadrature | analytic candidate |
| rank, sweeps, ridge | paired Phase 5A/5B setting | underfit or ridge bias | fit/holdout/condition | frozen diagnostic |
| Gaussian reference | existing map/target contract | measure mismatch | exact target and mass refusal | reviewed fixed baseline |

The ridge is a Class-C numerical choice.  It stays frozen for comparison and
cannot be adjusted after observing these results.  A future ridge calibration
needs a separate non-harm plan.

## Required checks before fitting

1. Run focused hybrid basis tests: analytic mass, integral, and cross block
   versus independent order-80 and order-100 standard-normal quadrature;
   derivative finite differences; constant uniqueness; SPD; invalid-measure
   refusal; and `ProductBasis` integration.
2. Run MathDevMCP structural audits for the tilted-normal parameters,
   recurrence, cross block, RBF block, and derivative.  Preserve raw statuses
   and do not call structural matches semantic proofs.
3. Compile a scoped Lean file proving the recurrence's first cases, symmetry
   of the RBF completed-square exponent, and the block/integral identities.
   Lean is an algebraic sidecar, not a floating-point or fitter proof.
4. Re-run the Phase 5A/5B focused regressions and current-tree Phase 5A replay
   before the GPU attempt.

## Skeptical plan audit

Disposition before execution: `PASS_FOR_BOUNDED_PHASE5C_HYBRID_DIAGNOSTIC`.

The target and map are inherited unchanged, the predictive normalizer is
independent of the fitted representation, and all three widths are fixed
before seeing holdout values.  The cross block is derived from a tilted normal
and will be checked independently before any ALS call.  The constant channel
is present exactly once, and a condition veto exposes nearly dependent hybrid
directions.  A valid but weak hybrid is a candidate failure that opens
replicated/recursive validation or a different representation; it does not
invalidate the exact evaluator.  A target/measure mismatch, failed analytic
identity, nonfinite program, corrupted artifact, or exhausted budget is a
continuation veto.

### Pre-mortem

| Misleading or failed outcome | Distinguishing check | Action |
| --- | --- | --- |
| Cross block looks correct only at low quadrature order | order-80 and order-100 independent quadrature agree with analytic values | repair formula/tests before fitting |
| Hybrid improves residual by duplicating the constant | assert RBF block has no constant and inspect mass eigenvalues | invalidate arm and repair basis construction |
| Broad arm wins through a nearly singular mass | pre-fit eigenvalue/condition and fitted condition | retain as candidate failure; do not promote |
| `Z_H` agrees through shared rows | predictive bank hash is disjoint from fit banks | invalidate artifact and rerun |
| One width appears best from finite-bank noise | one-factor ladder, paired bank, no ranking claim | open replicated plan only |
| Current source differs from recorded Phase 5 runs | current-tree replay and source hashes | resolve provenance before interpretation |

## Budget, commands, and close rules

Budget: one focused CPU suite and one GPU/XLA attempt, each at most one hour;
localized harness repairs may retry in a fresh directory under the same target,
data, method, vetoes, and budget.  No Student reference-law or large proposal
ladder is opened by this phase.

The CPU command, GPU command, source hashes, and actual elapsed time are
written into the result manifest.  GPU runs must set
`TF_FORCE_GPU_ALLOW_GROWTH=true`, configure growth before logical-device
initialization, and record device/XLA settings.  CPU-only checks must set
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and identify themselves as
reference diagnostics.

At close, write the required decision and inference-status tables.  If the
global checks pass and at least one arm survives, report all three arms
descriptively, classify any condition-vetoed arm as a candidate failure, and
refresh the next phase for replicated/recursive validation; no arm is
promoted.  If a local cross-contraction or harness check fails, preserve the
attempt, make the smallest repair, rerun the focused check, and continue only
if the exact target/map contract remains unchanged.  If no finite normalized
hybrid survives or an independent mass identity cannot be established, stop
with the precise continuation veto.  Record MathDevMCP and Lean limitations
explicitly.

## Execution close (2026-09-04)

The authoritative GPU/XLA run is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5c-hybrid-attempt03/`.
It passed the exact target/map, two-order full/cross quadrature checks,
finite/SPD and fitter checks for two of three arms, and the executable hybrid
endpoint wiring check.  The `w=3.0` arm is retained as a candidate-level
conditioning failure (`mass condition 2.57e16`); it is not a continuation
veto because the `w=0.75` and `w=1.5` arms remain valid.  The final focused
CPU suite reported `27 passed, 2 warnings`.

The first smoke, GPU metadata retry, and final wiring retry are preserved as
`attempt01`, `attempt02`, and `attempt03`; none overwrites prior evidence.  The
launch-time plan snapshot is `attempt03/plan-at-launch.md` with SHA-256
`b305fe3cfd1999ba38e92b1c97c3f75cffc570d1fd428df1aaa391b13deea8ac`.  This
close section is appended after execution, so the current plan hash is
intentionally different; the snapshot and manifest preserve the exact
launch contract.

The phase result is a mechanics pass and a candidate-level negative result for
the broad arm, not an improvement or default claim.  The next phase is a new
replicated/recursive validation plan using the surviving hybrid arms, the
unchanged exact C2 target/map, disjoint banks, and a predeclared uncertainty
analysis.  Student reference-law and proposal-ESS work remains deferred.
