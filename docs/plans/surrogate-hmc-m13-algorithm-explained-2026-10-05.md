# How the M13 LEDH–PFPF–OT algorithm works, and what marginal weighting changes

The October 4 proposal changes the importance correction between the LEDH
flow and the OT reset. It retains the UKF covariance lifecycle, the local flow
equations, Contract E reset, and the existing higher-moment correction with
its trust controls. However, those later operations receive new weights.
Their resulting particles change, and subsequent observations therefore also
have different flow trajectories. This is a proposed method extension; it has
not been implemented or shown to repair M13.

This explanation follows the current M13 consumer and the finite operations
it calls. The research goal remains testing surrogate HMC. The filter is the
component that supplies its analytical proposal force. The current M13
Metropolis potential comes from a separate frozen trajectory-importance
calculation. Neither calculation is automatically the exact integrated model
likelihood.

**1. The model and the information carried between observations.**

The state is position and velocity,

\[
x_t=(p_{x,t},p_{y,t},v_{x,t},v_{y,t})^\top\in\mathbb R^4,
\qquad x_t=A x_{t-1}+\eta_t,
\qquad \eta_t\sim N(0,Q_\theta),
\]

where the constant-velocity matrix uses a time increment of 0.1. The two
observations are range and bearing, with observation map

\[
h(x)=\bigl(\sqrt{p_x^2+p_y^2},\operatorname{atan2}(p_y,p_x)\bigr)^\top.
\]

The implementation uses its declared radius regularization and a wrapped
Gaussian observation density for the bearing. Write that density as
\(g_\theta(x)=p_\theta(y_t\mid x)\). The two inferred parameters are log
noise-standard-deviation multipliers:

\[
Q_\theta=e^{2\theta_Q}\operatorname{diag}(.0015,.0015,.0008,.0008),\qquad
R_\theta=e^{2\theta_R}\operatorname{diag}(.12^2,.04^2).
\]

At the start of an observation, particle \(a_i\) carries a normalized weight
\(\omega_i\), a local UKF covariance \(P_i\), and parameter derivatives of
the quantities that depend on \(\theta\). After the usual reset,
\(\omega_i=1/N\). Initial particles represent \(N(m_0,P_0)\), and each local
UKF covariance also starts at \(P_0\). Initial/noise/reset designs and numerical
controls are frozen for the likelihood evaluation.

There are three distinct covariance quantities: process noise \(Q_\theta\),
the local UKF covariance \(P_i\), and the empirical covariance of the whole
particle cloud. They have different roles. In particular, \(P_i\) is not the
conditional transition covariance in an importance density.

**2. Predict a local Gaussian and draw a transition child.**

For this linear state transition, the UKF prediction is

\[
m_i=Aa_i,\qquad P_i^-=A P_i A^\top+Q_\theta.
\]

The actual child, using the fixed innovation \(\epsilon_i\), is

\[
U_i=m_i+L_Q\epsilon_i,\qquad L_Q L_Q^\top=Q_\theta.
\]

Its transition density is \(f_i(u)=N(u;m_i,Q_\theta)\). The conditional draw
uses \(Q_\theta\); the next stage uses \(P_i^-\) to construct its flow. For a
general nonlinear state model the shared UKF computes its prediction by sigma
points, while the model's transition callback still defines the actual draw
and transition density. The marginal proposal here is restricted to the
full-rank Gaussian transition used by M13.

**3. Move the child toward the observation with LEDH.**

An auxiliary particle starts at the zero-noise prediction \(\bar x_{i,0}=m_i\).
At artificial time \(\lambda\), linearize the observation about this auxiliary
particle:

\[
H_i=Dh(\bar x_i),\qquad e_i=h(\bar x_i)-H_i\bar x_i.
\]

Use \(\mathcal A_i\) for the flow matrix, to distinguish it from the physical
transition matrix \(A\). The implemented coefficients are

\[
\mathcal A_i=-\tfrac12 P_i^-H_i^\top
 (\lambda H_iP_i^-H_i^\top+R_\theta)^{-1}H_i,
\]

\[
b_i=(I+2\lambda\mathcal A_i)
 \left[(I+\lambda\mathcal A_i)P_i^-H_i^\top R_\theta^{-1}(y_t-e_i)
       +\mathcal A_i m_i\right].
\]

For each of the frozen Euler substeps, with \(\Delta\lambda=1/K\) and
\(\lambda=k/K\), update both the actual and auxiliary particles using these
same coefficients:

\[
x_i\leftarrow(I+\Delta\lambda\mathcal A_i)x_i+\Delta\lambda b_i,
\qquad
\bar x_i\leftarrow(I+\Delta\lambda\mathcal A_i)\bar x_i+\Delta\lambda b_i.
\]

Conditional on the incoming cloud, the auxiliary path does not depend on this
step's process innovation. Consequently the full map of \(U_i\) is affine,
even though the observation is nonlinear:

\[
X_i=F_i(U_i)=B_iU_i+c_i.
\]

The current code accumulates \(\log|\det B_i|\) and its derivative as sums
over the substeps. Nonsingular step matrices and valid covariance solves are
required. The proposal needs to retain the full \(B_i,c_i\) as well. The
current returned determinant alone cannot evaluate another component's
density at \(X_i\).

**4. Correct the flow proposal and accumulate the likelihood.**

For invertible \(B_i\), the density of the moved child is

\[
q_i(x)=\frac{f_i(B_i^{-1}(x-c_i))}{|\det B_i|}.
\]

The existing ancestor-conditional correction is

\[
V_i^{\rm old}=\omega_i\,g_\theta(X_i)\frac{f_i(X_i)}{q_i(X_i)}.
\]

At its own sample, \(F_i^{-1}(X_i)=U_i\), so this is precisely the current
log-weight computation

\[
\log V_i^{\rm old}=\log\omega_i+\log g_\theta(X_i)+\log f_i(X_i)
                      +\log|\det B_i|-\log f_i(U_i).
\]

The transition correction and the determinant are essential: a useful-looking
flowed cloud does not itself establish a correct likelihood. Normalize with

\[
\widehat Z_t=\sum_i V_i,\qquad
\ell_N\leftarrow\ell_N+\log\widehat Z_t,\qquad
w_i=V_i/\widehat Z_t.
\]

This happens **before** OT and higher-moment correction. Those later stages
cannot retroactively change this observation's recorded likelihood increment.

The conditional correction is mathematically valid under the stated sampling
and invertibility assumptions. Its failure in the inspected M13 calculation
is severe finite-design importance-integration error, not an established
missing term in this formula.

**5. Update each UKF covariance.**

The UKF transforms sigma points from \((m_i,P_i^-)\) through \(h\), obtaining
an observation covariance \(S_i\) including \(R_\theta\), and state/observation
cross-covariance \(C_i^{xy}\). It forms

\[
K_i=C_i^{xy}S_i^{-1},\qquad
P_i^+=P_i^--K_iS_iK_i^\top.
\]

The call also computes a posterior mean, but this consumer discards that mean:
the particle state is still the flowed child \(X_i\). The updated covariance
is carried through the following transport to construct future flows.

**6. Replace the weighted cloud with an equal-weight cloud using OT and Contract E.**

Finite entropic Sinkhorn iterations construct a coupling whose intended row
mass is \(1/N\) and column mass is \(w_j\). The cost is the squared distance
between particle locations, with the implemented cost normalization. Because
balancing is finite, the code explicitly normalizes each realized row:

\[
T_{ij}=\frac{\Pi_{ij}}{\sum_k\Pi_{ik}},\qquad
b_i^{\rm OT}=\sum_jT_{ij}X_j,\qquad
P_i^{\rm carry}=\sum_jT_{ij}P_j^+.
\]

Simple averaging tends to shrink the cloud. Contract E computes the weighted
source mean and covariance

\[
\mu=\sum_iw_iX_i,\qquad
C=\sum_iw_i(X_i-\mu)(X_i-\mu)^\top.
\]

Let \(C_b\) be the equal-weight covariance of \(b_i^{\rm OT}\), \(z_i\) the
fixed residual design, and \(\rho\) the declared ridge. The finite reset is

\[
G=\operatorname{chol}\{\operatorname{sym}(C-C_b)+\rho I\},\quad
r_i=b_i^{\rm OT}+Gz_i,
\]

\[
L_C=\operatorname{chol}(C+\rho I),\quad
L_r=\operatorname{chol}(C_r+\rho I),\quad
a_i^{E}=\mu+L_CL_r^{-1}(r_i-\bar r).
\]

Here \(C_r\) is the unregularized empirical covariance of \(r_i\). The code
checks the required factorizations. With zero ridge and nonsingular matrices,
the final affine map restores mean and covariance exactly in exact arithmetic.
With the implemented ridge, its covariance is
\(L_CL_r^{-1}C_rL_r^{-\top}L_C^\top\), so exact equality to \(C\) is not a
universal algebraic identity. Moment residuals, rather than the name of the
reset, establish achieved numerical agreement.

**7. Correct higher moments with the GenUT shape iterations and trust controls.**

In this consumer the GenUT stage is a finite equal-weight cloud correction.
It computes its targets from the weighted source \((X_i,w_i)\). After
whitening that source, the targets include

\[
s_j=\sum_iw_i z_{ij}^3,\quad k_j=\sum_iw_i z_{ij}^4,\quad
s_{jk}=\sum_iw_i z_{ij}^2z_{ik},\quad
k_{jk}=\sum_iw_i z_{ij}^2z_{ik}^2\quad(j\ne k).
\]

Whiten the reset cloud to coordinates \(u_i\). For a coordinate with current
third/fourth moments \(m_3,m_4\), the diagonal correction directions are

\[
d_3(u)=u^2-1-m_3u,\qquad d_4(u)=u^3-m_3-m_4u.
\]

They remove first-order mean/scale changes in ideal standardized coordinates.
The residual is \(r=(s-m_3,k-m_4)^\top\); the two-by-two Jacobian \(J\) of
those moments along \(d_3,d_4\) has entries such as
\(J_{33}=\operatorname{mean}(3u^2d_3)\) and
\(J_{43}=\operatorname{mean}(4u^3d_3)\). With column scaling \(S\),
\(\widetilde J=JS^{-1}\), damping \(\lambda_{\rm LM}\), and strength \(\eta\),
the implemented LM coefficient is

\[
c=\eta S^{-1}(\widetilde J^\top\widetilde J+
 \lambda_{\rm LM}I)^{-1}\widetilde J^\top r.
\]

It proposes \(\delta_i=c_3d_3(u_i)+c_4d_4(u_i)\). A smooth per-particle
coordinate-RMS trust cap uses

\[
\delta_i\leftarrow
 \frac{\delta_i}{\sqrt{1+\operatorname{RMS}(\delta_i)^2/\tau^2}}.
\]

The updated cloud is centered and whitened again. Separate pairwise
iterations use co-skewness/co-kurtosis residuals, project away affine moment
changes, normalize their directions, apply their own per-particle RMS cap,
multiply by pairwise strength, and restandardize. They are finite correction
steps, not a proof of exact simultaneous higher-moment matching.

The final standardized-coordinate cap in the inspected configuration is

\[
u\leftarrow\frac{u}{[1+(u/c_{\rm coord})^p]^{1/p}},\qquad p=8.
\]

The code then recolors to the weighted source and performs affine restoration.
That restoration uses its own recorded relative covariance floors. The cap
therefore bounds its own intermediate coordinates; it does not by itself
establish the same bound on the finally restored cloud.

The saved M13 configuration has two diagonal iterations of strength 0.2,
LM damping \(10^{-6}\), diagonal trust radius 100, two pairwise iterations of
strength 0.02, pairwise RMS cap 2000, and standardized-coordinate cap 16000.
These are the saved configuration's controls, not universal recommendations.
They operate **after likelihood weighting**, not as a cap on the earlier LEDH
proposal's displacement. The next observation receives the corrected
equal-weight states and \(P_i^{\rm carry}\); the shape correction does not
recompute those carried UKF matrices.

**8. Differentiate the same finite computation analytically.**

For each parameter direction, the implementation propagates derivatives of
particles, UKF covariances, process draws, observation linearizations, flow
matrices, determinants, weights, transport, moment restoration, and shape
correction. With a dot denoting this total directional derivative,

\[
\dot\ell_N=\sum_t\sum_i w_{t,i}\,\dot{\log V_{t,i}}.
\]

The incoming derivatives at the next observation include all preceding
reset/shape effects. Differentiating only the immediate observation density,
or treating previous particles or \(P_i\) as constants, would compute a
different quantity. Agreement with finite differences checks the derivative
of this finite filter; it does not establish agreement with the model's
integrated likelihood score.

**9. The proposed replacement of step 4.**

Define the whole incoming predictive density and whole proposal density:

\[
p_N(x)=\sum_j\omega_j f_j(x),\qquad
q_N(x)=\sum_j\omega_j q_j(x).
\]

Keep the same per-component draws and flow construction, but replace the
weight by

\[
\boxed{V_i^{\rm mar}=\omega_i g_\theta(X_i)\frac{p_N(X_i)}{q_N(X_i)}}.
\]

Both numerator and denominator change. The leading \(\omega_i\) is appropriate
to one draw per component with the displayed weighted sum; independent random
selection of component labels would require a different averaging convention.

Conditional on the incoming cloud, under randomized component draws,

\[
E\left[\sum_iV_i^{\rm mar}\right]
=\sum_i\omega_i\int q_i(x)g_\theta(x)\frac{p_N(x)}{q_N(x)}dx
=\int g_\theta(x)p_N(x)dx.
\]

The old correction integrates to that same finite-mixture normalizer. The
proposal changes the importance estimator, not the intended conditional
integral. For a fixed design, the two numerical estimates generally differ.
This identity does not prove unbiasedness of the full deterministic-reset
filter or equality to the exact model likelihood.

The required new work is to accumulate \(B_i,c_i\) and their derivatives,
evaluate all component densities at every \(X_i\), and analytically
differentiate their log sums. Start with \(B_i=I,c_i=0\). A flow substep with
\(M_i=I+\Delta\lambda\mathcal A_i\) updates
\(B_i\leftarrow M_iB_i\) and
\(c_i\leftarrow M_ic_i+\Delta\lambda b_i\), together with their product-rule
derivatives. Equivalently, each proposal component is the Gaussian
\(q_i=N(B_im_i+c_i,B_iQ_\theta B_i^\top)\). For example, with
\(r^f_j=\omega_jf_j/p_N\), \(r^q_j=\omega_jq_j/q_N\),

\[
\dot{\log V_i^{\rm mar}}=
\dot{\log\omega_i}+\dot{\log g_\theta(X_i)}
+\sum_jr^f_j(X_i)(\dot{\log\omega_j}+\dot{\log f_j(X_i)})
-\sum_jr^q_j(X_i)(\dot{\log\omega_j}+\dot{\log q_j(X_i)}).
\]

The dot includes movement of \(X_i\) and every ancestor. For
\(v_{ji}=B_j^{-1}(X_i-c_j)\),

\[
\dot v_{ji}=B_j^{-1}(\dot X_i-\dot c_j-\dot B_jv_{ji}),\qquad
\dot{\log q_j(X_i)}=\dot{\log f_j(v_{ji})}
                     -\operatorname{tr}(B_j^{-1}\dot B_j).
\]

This adds an all-pairs density calculation, of order \(N^2\) component/sample
pairs. Exact tiling can bound memory; it does not remove quadratic work.

| Part of an observation | Direct change in the proposal |
|---|---|
| State/observation model, physical \(Q,R\) | None |
| UKF prediction and update equations | None |
| LEDH coefficients, substeps, child generation | None for the same incoming cloud |
| Flow bookkeeping | Retain affine maps and their analytical derivatives |
| Importance correction | Replace component ratios with the full mixture ratio |
| Log likelihood and its score | Recompute from the new weights and total derivatives |
| OT, Contract E, GenUT, caps | Same operators and controls, different weighted inputs |
| Following observations | Different incoming particles/covariances can change their trajectories |
| LGSSM and KSC legacy mode | Remains the existing execution; requires consumer regressions when implemented |
| HMC Metropolis potential | Remains separately declared; changing a force does not change it automatically |

If all mixture components coincide, the old and new weights coincide. If flow
is disabled, \(q_i=f_i\), so both reduce to bootstrap weights
\(\omega_i g_\theta(X_i)\). These are useful implementation checks.

Marginalization removes dependence of the density ratio on an arbitrary
generating label. A particle near another ancestor's predicted state is judged
using that ancestor too, with the proposal denominator changed consistently.
It cannot create samples in an entirely unvisited region. The random-label
Rao–Blackwell variance theorem does not establish universal variance reduction
for this one-per-ancestor fixed design; the proposal contains an explicit
stratified counterexample. No empirical repair result is available yet.

The proposal also contains a later contingency if the flowed mixture still
has inadequate coverage: mix each \(q_i\) with its original transition
\(f_i\), using a fixed weight \(\alpha\). Then the actual child-generation
law must change to \(r_i=\alpha f_i+(1-\alpha)q_i\), and the denominator
becomes \(r_N=\alpha p_N+(1-\alpha)q_N\). Changing that denominator without
sampling the corresponding mixture would be wrong. No \(\alpha\) has been
selected; this is separate from the primary proposal that keeps child
generation unchanged. The full-rank derivation also does not cover structural
AR(2) dynamics without a separate derivation on their innovation-space measure.

**10. Why this change addresses the observed problem.**

At the saved low-process-noise probe, the first-observation quadrature log
likelihood is -0.13791267; the same-particle bootstrap gives -0.13013616;
current LEDH gives -2.22116858 with particle ESS 6.66 out of 4096. Its
pre-observation normalization estimate is 0.00344047 although the corresponding
conditional integral is one. The local UKF/process covariance scale mismatch
and large flow displacements provide a mechanism for this integration failure.
These are descriptive results for the saved designs, not a replicated ranking.

The recorded likelihood error exists before the first OT/GenUT operation.
Matching moments of that already inaccurate weighted cloud cannot repair it.
The proposed marginal weighting is therefore a targeted candidate to test,
not an established solution. It preserves the old LGSSM/KSC route by choosing
an explicit separate mode; it supplies no theorem that switching those models
to the new weights cannot worsen their finite calculations.

In the actual M13 HMC wiring, the proposal gradient of potential is the
negative LEDH likelihood score plus the prior contribution. A separately frozen
importance likelihood supplies the Metropolis potential. The LEDH force is
therefore a surrogate for the endpoint gradient even though its own analytical
derivative can be correct. The marginal extension aims to improve that force's
relation to the integrated model score. Its effect on HMC must still be tested
with the frozen endpoint and appropriate tuning and posterior diagnostics.

Source checks for this explanation used HEAD
`a55c6be7a16e9401242ee3449248fa347eb58d90` plus the existing dirty working tree.
Only documentation was changed; no new numerical, GPU, or HMC experiment was
run. The current call chain was inspected in source, not newly certified by an
executed wiring test. The prior localization artifact records executable
full-consumer/trace equality at eight points for its saved source snapshot.

- [M13 model and consumer](../../bayesfilter/highdim/ledh_range_bearing_target_tf.py),
  lines 67–115; batch adapter to scalar analytical consumer,
  [lines 344–367](../../bayesfilter/highdim/ledh_canonical_batch_fused_tf.py).
- [Full observation iteration](../../bayesfilter/highdim/ledh_canonical_score_tf.py),
  lines 432–832; LEDH substep equations, lines 1171–1338.
- [Contract E and covariance carry](../../bayesfilter/highdim/ledh_unified_reset_tf.py),
  lines 105–317; shared adapter in `ledh_canonical_reset_score_tf.py`.
- [Higher moments and caps](../../bayesfilter/highdim/higher_moment_contract_e.py),
  lines 211–361, 620–999, 1100–1120, and 1540–1595;
  [scaled LM and RMS cap](../../bayesfilter/highdim/genut_shape_lm_tf.py),
  lines 17–65 and 137–185.
- [Actual M13 HMC force/endpoint roles](../../bayesfilter/testing/surrogate_hmc_m13_public_tuning.py),
  lines 1–5 and 77–79; implementation in
  `surrogate_hmc_m13_four_chain_cost.py`, lines 28–58,
  `surrogate_hmc_m13_endpoint_mechanics.py`, lines 109–125, and current retained
  report fields in `surrogate_hmc_m13_retained.py`, lines 65–67.
- [Saved configuration](artifacts/surrogate-hmc-m13-score-before-after-20261004-01/inputs.json),
  lines 2–18; [measured localization](artifacts/surrogate-hmc-m13-likelihood-gap-localization-20261004-01/result.md).
- [October 4 proposal and proofs](surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex).
