# MathDevMCP Audit of the C2 Coherent Testing Plan

Date: 2026-08-31

This note records bounded checks made with the repository-local MathDevMCP CLI.
The checks are scoped evidence for individual identities or visible code terms;
they are not a proof of the whole plan or of the numerical algorithm.

## Commands and results

### 1. Importance-ratio cancellation

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      check-proof-obligation 'q*(gamma_fin/q)' 'gamma_fin' \
      --assumption 'q != 0' --backend sympy

Result: status=equivalent. SymPy simplified the difference to zero. This
certifies only the encoded scalar cancellation under the nonzero-denominator
assumption. The measure-theoretic requirements (measurability, integrability,
normalization, and support) remain explicit assumptions in the plan.

### 2. Defensive-floor scale bookkeeping

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      check-proof-obligation 'exp(c)*(h_norm2 + tau*z0)' \
      'exp(c)*h_norm2 + exp(c)*tau*z0' --backend sympy

Result: status=equivalent. This checks the distributive scale identity used
to distinguish the scaled TT fit from its physical normalizer.

### 3. Finite-sum analytical score

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      check-proof-obligation \
      '(b1*dg1 + b2*dg2)/(b1*g1 + b2*g2)' \
      '(b1*g1/(b1*g1+b2*g2))*(dg1/g1) + (b2*g2/(b1*g1+b2*g2))*(dg2/g2)' \
      --assumption 'g1 != 0' --assumption 'g2 != 0' \
      --assumption 'b1*g1+b2*g2 != 0' --backend sympy

Result: status=equivalent. This is the two-sample algebraic instance of
\(\nabla\log\widehat Z=\sum_i\bar w_i\nabla\log\gamma_i\). The full vector and
recursive cases still require the explicit frozen/pathwise contract in the
main plan.

### 3a. Initial affine pullback

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      check-proof-obligation 'gamma0*detL/r * r/detL' 'gamma0' \
      --assumption 'r != 0' --assumption 'detL != 0' --backend sympy

Result: status=equivalent. This checks the scalar cancellation in the explicit
\(t=0\) pullback \(F_0=\gamma_0(m_0+L_0u_0)|\det L_0|/r_0(u_0)\)
when it is integrated against \(r_0(u_0)\,du_0\). The change-of-variables
regularity and positivity assumptions remain the responsibility of Stage 0.

### 3b. Student covariance-matching scale

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      check-proof-obligation 'nu/(nu-2) * ((nu-2)/nu) * P' 'P' \
      --assumption 'nu != 0' --assumption 'nu != 2' --backend sympy

Result: status=equivalent. This verifies the algebraic covariance-matching
choice \(S=(\nu-2)P^-/\nu\) given the standard multivariate-Student covariance
formula \(\nu S/(\nu-2)\). The probabilistic requirements \(\nu>2\) and
positive-definite \(P^-\) are still explicit Stage 0 conditions.

### 4. APF initial log weight

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      audit-math-to-code \
      'log_unnormalized = initial_log_base_mass + initial_log_density + observation_log_density - initial_log_proposal_density' \
      bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py

Result: status=structural_match, with no missing terms. The visible assignment
is at zhao_cui_frozen_proposal_apf_tf.py:821-826.

### 5. APF transition log weight

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      audit-math-to-code \
      'log_unnormalized = transition_log_base_mass + selected_previous_log_weights + transition_log_density + observation_log_density - selected_auxiliary_log_probability - transition_log_proposal_density' \
      bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py

Result: status=structural_match, with no missing terms. The visible assignment
is at zhao_cui_frozen_proposal_apf_tf.py:901-908. This supports the plan's
requirement that ancestor weights, auxiliary correction, transition density,
observation density, and the complete bound proposal density all enter the
finite APF program.

### 6. Retained quadratic-form floor

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      audit-math-to-code \
      'sum_sq = sum(square(v_prev)) + floor_values' \
      bayesfilter/highdim/squared_tt_engine_gaussian_tf.py

Result: status=scope_limited_match. The terms sum_sq, square, v_prev, and
floor_values are visible at squared_tt_engine_gaussian_tf.py:417-431; the
conservative matcher does not recognize the TensorFlow reduction as the literal
token sum. This is diagnostic parser scope, not evidence that the code omits
the reduction.

### 7. Recursive-map call-chain check

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/mathdevmcp \
      audit-kalman-recursion \
      bayesfilter/highdim/squared_tt_engine_gaussian_tf.py \
      --required-operation mean --required-operation covariance

Result: status=mismatch, with mean and covariance absent from the recognized
required-operation set. The source does validate hint shapes and performs
Cholesky/prediction-related operations, but the call chain currently consumes
externally supplied initial_moment_hint and predictive_moment_hint at
lines 328-369; it does not derive those moments from the retained quadratic
form. This is exactly the Stage 2 implementation gap, not a reason to claim
that Stage 2 has already been tested.

## Inconclusive routes

The CLI's conservative scalar grammar could not encode the full identities
containing integral, expectation operators, or matrix products. Attempts to
route those expressions returned not_encodable, unknown, or
human_review_required; they did not refute the mathematics. In particular:

- the measure-theoretic identity
  \(\mathbb E_q[\gamma/q]=\int\gamma\) is justified in the plan by direct
  substitution and the stated support/integrability assumptions, while the
  CLI certifies only the local algebraic cancellation;
- the linear-Gaussian covariance recursion is retained as a manual derivation
  with explicit independence and finite-moment assumptions; and
- the assumption audit for the moment recursion returned proposal_ready but
  did not establish global minimality or proof closure.

The generated-test route also produced symbolic-identity,
finite-difference, shape, and expected-failure diagnostics for
q*(gamma_fin/q) = gamma_fin. These are test specifications, not certificates.

## Audit conclusion

The MathDevMCP evidence supports the plan's key boundaries:

1. the local importance quotient and scale algebra are correct under stated
   domain assumptions;
2. the current APF evaluator contains the required finite log-weight terms;
3. the current Gaussian TT engine does not yet implement the proposed
   density-derived recursive moment map; and
4. full integral, matrix, tail, and recursive-error claims still require the
   fixture tests and numerical diagnostics specified in the main plan.
