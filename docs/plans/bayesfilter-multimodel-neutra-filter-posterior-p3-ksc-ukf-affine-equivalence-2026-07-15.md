# P3 KSC Affine Principal-Square-Root Equivalence

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Scope: scalar component update inside the declared seven-component KSC
mixture-moment-collapse filter. This derivation does not claim exact latent-state
filtering or exact non-Gaussian SV likelihood.

## Claimed Target

For component `j` at time `t`, conditional on the previous collapsed Gaussian
state, the declared filter uses

```text
h_t = gamma h_(t-1) + eta_t,       eta_t ~ N(0, 1)
z_t = h_t + 2 log(beta) + m_j + e_j,  e_j ~ N(0, v_j)
```

At time zero the predicted state is the stationary Gaussian
`N(0, 1/(1-gamma^2))`; there is no additional process increment. For positive
time the prediction is

```text
a = gamma m
P = gamma^2 C + 1.
```

The principal-square-root UKF applies symmetric sigma points to affine
transition and observation functions. Such points reproduce Gaussian first and
second moments exactly. Therefore its component predictive observation law is

```text
q_j = z_t - a - 2 log(beta) - m_j
S_j = P + v_j
log L_j = -0.5 [log(2 pi) + log(S_j) + q_j^2 / S_j]
K_j = P / S_j
m_j^+ = a + K_j q_j
C_j^+ = P v_j / S_j.
```

These are the quantities computed directly by
`bayesfilter/testing/ksc_ukf_neutra_target_tf.py`. They are equal to the scalar
affine component quantities computed by the existing principal-square-root
score wrapper, subject to the same stationary initialization, transform,
mixture tensors, and absence of an active numerical floor.

## Mixture Collapse

The filter is not an exact mixture-history recursion. It normalizes the seven
component weights at each time,

```text
alpha_j = softmax(log w_j + log L_j),
```

then collapses them to one Gaussian using

```text
m^+ = sum_j alpha_j m_j^+
C^+ = sum_j alpha_j [C_j^+ + (m_j^+ - m^+)^2].
```

This is the same declared moment-collapse approximation as the reference
wrapper. Equality of component updates does not make this collapse exact for
the KSC state posterior.

## Total Derivative

Let `D` denote the two source coordinates. The graph route propagates

```text
D a, D P,
D q_j = -D a - D[2 log(beta)],
D S_j = D P,
D log L_j = -(q_j/S_j) D q_j
              + 0.5 (q_j^2/S_j^2 - 1/S_j) D S_j,
D alpha_j = alpha_j [D log L_j - sum_k alpha_k D log L_k],
```

and differentiates the component posterior means, variances, collapsed mean,
and collapsed second moment. The collapsed variance derivative is
`D C^+ = D E[h^2] - 2 m^+ D m^+`. The likelihood score is the sum of the
per-time mixture log-normalizer scores. The posterior score additionally
includes the physical Uniform-box prior and complete two-probit chart Jacobian.

## Required Checks

- T=1 and T=2 likelihood value parity against the existing value wrapper;
- T=1 and T=2 source-coordinate score parity against the existing
  principal-square-root analytical score after applying the exact chart chain
  rule;
- centered-FD agreement for the graph posterior;
- strictly positive state/component/innovation variances and normalized
  component weights;
- independent dense KSC-mixture filter comparison under the P3 subplan.

Passing the first four checks establishes engineering correctness for the same
finite UKF/moment-collapse program. Only the last gate addresses whether that
approximate filter is admissible for this campaign posterior.
