# Zhao-Cui Austria SIR Conditional-Reference Repair Plan

Date: 2026-08-01

Status: `AUDITED_EXECUTING_SAMPLE_GROWTH`

Parent reset:
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-density-t1-reset-memo-2026-08-01.md`.

## Research intent

Determine whether the failed static-frame child is failing because its
parameter-conditioned representation is inadequate, rather than because the
Austria T1 target or score definition is wrong.  Construct an independent
reference in conditional innovation coordinates and compare its finite value
and total score with the existing scalar model.

## Mathematical target

Draw a proposal cloud once at the origin `theta_ref=0`:

\[
 z_0\sim p_0(z_0),\qquad z_1\sim f_0(z_1\mid z_0).
\]

where `m_theta` and `L_theta` are the exact pre-clipping transition mean and
Cholesky factor used by the latent Austria model.  For the first observation,
the finite value program at any `theta` is the importance estimate
estimate

\[
 \widehat Z_1(\theta)=N^{-1}\sum_i\exp\{
 \log p_\theta(z_{0i})+\log f_\theta(z_{1i}\mid z_{0i})+
 \log g_\theta(y_1\mid z_{1i})-\log p_0(z_{0i})-\log f_0(z_{1i}\mid z_{0i})\}.
\]

`theta_ref=0` makes the origin ratio one for every row, while the denominator is
constant under differentiation.

The total score is the derivative of this same finite value program:

\[
 \nabla_\theta\log \widehat Z_1(\theta)=
 \frac{E[w_\theta(z_0,z_1)
 (\nabla_\theta\log p_\theta(z_0)+
 \nabla_\theta\log f_\theta(z_1\mid z_0)+
 \nabla_\theta\log g_\theta(y_1\mid z_1))]}{Z_1(\theta)}.
\]

The origin innovations are only a way to generate the fixed proposal cloud.
Their Jacobian is not added to the physical score.  The score is the weighted
sum of the model's analytical initial, transition, and observation scores; the
proposal denominator is constant.  This is an independent reference authority,
not a source-faithful Zhao-Cui route.

## Source and classification boundary

The squared-TT density and marginalization operations remain grounded in
Zhao-Cui Eqs. (9)-(12), Eq. (15), Algorithm 2, Proposition 2, author
`third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-135`,
and `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-85`.
The innovation-coordinate authority, common-random-number evaluation, and any
parent-conditioned residual compiler are `extension_or_invention` until a
separate derivation and source mapping are recorded.

## Evidence contract

| Item | Requirement |
|---|---|
| Exact target | `p_theta(z0) f_theta(z1|z0) g_theta(y1|z1)` for the sealed Austria observation `y1`. |
| Primary value check | Innovation estimate and scalar reference agree within `3*MCSE` for each declared theta row. |
| Primary score check | Analytical complete-data score ratio and autodiff derivative of the same finite innovation program agree within `3*MCSE + 1e-8`. |
| Seed check | Two independent stateless seed pairs on disjoint batches give paired standardized residuals no larger than `3`. |
| Backend check | Eager and XLA outputs differ by at most `3e-10` for value and score. |
| Finite/ESS veto | All values, scores, weights, and MCSEs finite; ESS at least half the sample count. |
| Resource veto | TensorFlow allocator peak below 6 GiB; no retained time history, tensor-product grid, or sample-wise target loop. |
| Nonclaims | Passing does not admit a Zhao-Cui child, prove full-horizon score recursion, establish proposal quality, or authorize HMC. |

## Bounded execution

1. Implement the batch-native innovation cloud and value/score estimator in a
   new module.  Use only TensorFlow/TFP runtime operations and the existing
   latent pre-clipping model.  Add an exact finite-program `GradientTape`
   derivative check at `theta=0` and at two symmetric nonzero rows.
2. Add focused CPU tests for shape, source observation identity, analytical
   score parity, common-random-number reproducibility, and fail-closed finite
   checks.  Run the tests with `CUDA_VISIBLE_DEVICES=-1`.
3. Run a bounded CPU authority with two seeds and `N=8192` per seed.  Record a
   versioned JSON result and manifest; no claim data is consumed.
4. If the authority passes, implement only a mechanics-level parent-conditioned
   residual interface whose coordinates are `(z0, epsilon)` and whose exact
   target log-density is evaluated by the model.  Prove zero-slice value and
   score parity before any optimizer is considered.
5. Stop at the first failed target/score/backend/resource gate.  Do not train a
   child or extend the horizon until this plan is refreshed with the result.

### Sample-growth refresh

The two-seed `N=8192` authority passed, but its first two score coordinates
remain uncertainty-limited.  Run exactly two fresh origin clouds at
`N=65536`, seeds `92003` and `92004`, with the same frozen proposal law,
analytical score, autodiff parity, and artifact schema.  This is an authority
precision diagnostic, not a new target or promotion criterion.  Require ESS at
least `N/2`, finite values/MCSEs, analytical/autodiff parity, and paired seed
differences within `3` combined MCSEs.  Cap the run at 20 minutes and 6 GiB;
write a fresh directory and preserve the earlier result unchanged.

Skeptical audit: the target, proposal law, score definition, dtype, and source
observation are unchanged; only sample count and fresh seeds change.  A pass
reduces uncertainty but cannot admit a child.  A failure blocks the
parent-conditioned representation and returns to target/authority repair.

## Skeptical audit

| Risk | Audit result |
|---|---|
| Wrong baseline | Uses the sealed Austria latent pre-clipping target and the admitted T1 parent only as a comparator; no APF/UKF/retained-grid authority. |
| Proxy promotion | The reference value and total score are the primary gates; training loss and proposal ESS are explanatory or veto-only. |
| Hidden Jacobian term | The proposal cloud is frozen at `theta_ref=0`; its denominator is constant and no innovation Jacobian enters the score. |
| MCSE underestimation | Two independent seeds and ESS/MCSE gates are mandatory. |
| Memory blow-up | The cloud is batched and streamed; no full history or tensor-product retained grid is permitted. |
| Source drift | Paper and author-code anchors are explicit; the new route is not called source-faithful. |
| Premature escalation | No density fitting, horizon recursion, comparator, or HMC is allowed before the reference gates pass. |

Audit verdict: `PASS_FOR_BOUNDED_MECHANICS`.  The plan answers the current
representation blocker with the smallest independent diagnostic and has an
explicit stop condition.

## Artifact

`docs/plans/artifacts/zhao-cui-austria-sir-conditional-reference-t1-20260801/`
