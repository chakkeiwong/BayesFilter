# Zhao-Cui Austria SIR Measure-Bridge Plan

Date: 2026-08-01

Status: `BLOCKED_RATIO_BRIDGE_PARENT_TAILS`

Parent result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-sample-growth-result-2026-08-01.md`.

## Research question

Can the admitted fixed Zhao-Cui T1 parent be used as a base measure for a
parameter-conditioned density whose origin normalized derivative equals the
exact Austria observed-data score, without changing the finite value target or
retaining an unbounded history/grid?

## Candidate bridges

### Bridge A: exact target-weighted parent base

Define the base measure on the latent T1 joint coordinates by

\[
 d\mu_0(z_0,z_1)=
 p_0(z_0)f_0(z_1\mid z_0)g_0(y_1\mid z_1)\,dz_0dz_1/Z_1(0).
\]

The new child must represent a density relative to `mu_0`, not the parent's
36-dimensional uniform reference measure.  Its normalized score at the origin
then has the physical score identity by construction.  This is an
`extension_or_invention` unless the source route is explicitly mapped to this
measure.

### Bridge B: exact Radon-Nikodym correction of the fixed parent

Retain the parent's reference coordinates and define

\[
 \rho_\theta(r)=\rho_0(r)\,R_\theta(r),\qquad
 R_\theta(r)=\frac{\pi_\theta(z(r),y_1)}{\pi_0(z(r),y_1)}.
\]

The finite value is computed by integrating `rho_theta` under the original
reference measure.  The implementation must include the parent-to-physical
Jacobian and prove that the origin value and score reduce to the exact
target-weighted integrals.  If the parent does not have support covering the
target, fail closed; do not add a defensive floor and call the result exact.

### Bridge C: conditional KR proposal

Use a parameter-conditioned upper KR proposal for `z1 | z0, y1` and apply the
exact correction `f_theta g_theta / q_theta`.  This follows the structure of
Zhao-Cui Algorithm 3 and paper Eqs. (20)-(23), but the finite density remains
an extension until the proposal and its Jacobian are source-mapped.  It must
be evaluated as a proposal-correction route, not substituted into the fixed
parent density silently.

## Selection rule

Implement no child until one bridge passes all derivation gates:

1. The measure identity is written in project notation and maps every density,
   Jacobian, and normalizer term.
2. At `theta=0`, the finite value equals the admitted parent value and the
   finite total derivative equals the conditional-reference authority score.
3. The same finite program has an eager/autodiff parity check at origin and two
   symmetric nonzero theta rows.
4. The bridge uses only batched TensorFlow operations, stores no time history,
   and has a pre-run memory bound below 6 GiB.
5. Independent seeds and sample-growth checks pass before any optimizer or
   child fitting.

If A, B, and C all fail the identity, stop the Zhao-Cui score effort and report
that the admitted fixed parent is value-only for this Austria target.  Do not
hide the failure by changing the score target, adding a gauge, or calling a
proxy score correct.

## Skeptical audit

| Risk | Audit result |
|---|---|
| Wrong target | Exact latent pre-clipping T1 observed-data target is frozen. |
| Wrong measure | Candidate measures and Radon-Nikodym factors are explicit. |
| Proxy score | Conditional-reference score is an authority; parent point score is not substituted. |
| Source drift | Zhao-Cui Algorithm 3/Eqs. (20)-(23) and author `full_sol.m`/`marginalise.m` anchors are required. |
| Memory blow-up | No retained tensor-product transition grid, time history, or full TT rank sum. |
| Premature HMC | HMC remains forbidden until value and total score are admitted. |

Audit verdict: `PASS_FOR_DERIVATION_ONLY`.  The next work item is a written
bridge derivation and a tiny scalar/one-dimensional prototype, not a GPU
campaign or density optimizer.

## Derivation outcome

The derivation exposes a continuation blocker for the current parent-preserving
child family.  Let `q_0` denote the admitted fixed TT density in its reference
measure and let `pi_theta` denote the exact normalized Austria T1 observed-data
density in the same coordinates.  Any family that claims exact target equality
for all parameters must satisfy

\[
 q_\theta=\pi_\theta\quad\Longrightarrow\quad q_0=\pi_0.
\]

The admitted parent is a fitted finite TT approximation; its source artifact
does not contain a proof or pointwise identity `q_0=pi_0`.  The failed rank
ladder additionally rejected its off-origin mass/shape representation.  Thus
the origin parent slice cannot be combined with the exact physical score by
continuity alone.

For a generic smooth parent-preserving family
`q_theta=q_0+sum_k theta_k h_k+o(theta)`, the normalized origin derivative is

\[
 \partial_k\log Z_q(0)=E_{q_0}[h_k/q_0]-E_{q_0}[h_k/q_0],
\]

or, for a normalized density family, an expectation under `q_0`.  The exact
observed-data score is an expectation under `pi_0`; these are equal only with a
proved measure identity or a fully specified Radon-Nikodym correction.  The
current centered residual and core-affine families provide neither.

Bridge B remains mathematically possible only as a new target definition:
evaluate the exact ratio `pi_theta/q_0` pointwise, prove support and Jacobian
terms, and integrate it under `q_0`.  That route is not the current finite TT
child, is not source-faithful Zhao-Cui, and requires a new reviewed plan and
implementation.  It must not be silently substituted.

Decision: exact physical-likelihood equality is blocked, but Bridge B defines a
distinct parent-preserving finite Zhao-Cui extension:

\[
 L_{\rm ZC}(\theta)=L_{\rm parent}(0)+
 \log E_{q_0}[\pi_\theta/\pi_0].
\]

The user's standing instruction to continue the finite-program value/score
work authorizes a bounded diagnostic of this explicitly named target.  It must
be compared against the physical authority and may not be called exact physical
likelihood.  No rank/optimizer arm, horizon recursion, comparator, or HMC is
opened by this definition.
