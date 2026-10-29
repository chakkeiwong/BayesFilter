# Retained shared-Gaussian score: mathematical and call-chain audit

This audits the implementation change retained in the current computation
campaign, not canonical LEDH admission. The comparison target is the same
finite covariance-proposal filter as the preserved extension-start source.

For evaluation point x_i(theta), component mean m_j(theta), shared covariance
C(theta), precision P=C^{-1}, and r_ij=x_i-m_j, the component log-density
derivative in a parameter direction is

    dot(ell_ij) = -(dot(x_i)-dot(m_j))' P r_ij
                 + (1/2) r_ij' P dot(C) P r_ij
                 - (1/2) tr(P dot(C)).

For the uniform mixture f_i=N^{-1} sum_j exp(ell_ij),

    dot(log f_i) = sum_j w_ij dot(ell_ij),
    w_ij = exp(ell_ij) / sum_k exp(ell_ik).

No additional dot(w) term belongs in this first derivative: the weights arise
from differentiating log(sum(exp(ell))). Differentiating that resulting score
again would require additional terms; this optimization does not compute a
Hessian.

The implementation centers both points and their tangents at the same
parameter-dependent origin a=x_0. Set z=x-a, m=m_original-a,
dot(z)=dot(x)-dot(a), dot(m)=dot(m_original)-dot(a), H=P' dot(C) P.
Writing E for the conditional weighted component average, expansion gives

    score_i = -dot(z_i)' P z_i
              + dot(z_i)' P E[m]
              + (P z_i)' E[dot(m)] - E[(P m)' dot(m)]
              + (1/2) z_i' H z_i - z_i' H E[m]
              + (1/2) E[m' H m] - (1/2) tr(P dot(C)).

This is exactly the expression in shared_weighted_score. The common origin
and its derivative cancel from residuals. A single weighted feature matrix
contains m, all dot(m), (P m)'dot(m), and m'Hm. It replaces materializing
all component scores with shape [parameter,batch,row,column]. The pair weights
still exist; the change does not remove the quadratic particle-pair workload.

Column tiles form partial mixture sums. Their value/tangent pairs are merged
by analytical Jet logsumexp, preserving their relative masses. Covariances
must truly be common across components in this route; per-component covariance
routes retain their general implementation. Factorization/guard semantics are
unchanged from the baseline. Algebraic equality here concerns a healthy fixed
branch in real arithmetic; actual FP32 reordering is checked separately.

Inspected call chain: actual HMC force binding -> make_batched_score_filter ->
shared covariance mixture dispatch -> shared_gaussian_mixture ->
shared_weighted_score. Observation and particle-tile loops are TensorFlow
while loops; parameter directions remain a tensor axis in the same recursion.

Executable evidence:
- tests/highdim/test_covariance_proposal_shared_gaussian.py checks the complete
  weighted expression against an independently expanded dense formula for
  dimensions 1, 3, 18; FP32/FP64; offsets 0/10000, under XLA.
- mixture-paired-*/ preserves same-input full-horizon baseline comparisons for
  all five models, nominal and off-centre. These passed the declared one-percent
  arithmetic and no-new-invalid-row screens.
- consumer-paired-tests-01.log verifies actual tiny consumer parity, distinct
  baseline/candidate factory wiring and restoration after trace failure.
- Full-size paired actual-consumer timing/parity remains to be run under
  consumer-paired-contract.md; CPU wiring alone does not establish this.

Decision: no missing first-derivative term was found in the retained algebra.
The executable full-size consumer check remains required. This audit does not
establish model-oracle score accuracy, posterior convergence, a statistically
supported speed ranking or canonical LEDH admission.
