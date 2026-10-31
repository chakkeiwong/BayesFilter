# Structural model coverage and canonical-consumer prerequisite

The master question is surrogate-HMC parameter inference using the full
canonical analytical DCTR proposal field. This P0 audit identifies the
structural model laws and tests one prerequisite in the actual shared
consumer call chain. It performs no tuning, reference integration or sampling.

The AR2 fixture in `bayesfilter/testing/structural_fixtures.py:14` has
F=[[.45,-.15],[1,0]], Q=diag(.3²,0), H=[1,0], R=.2² and initial N(0,I).
Its second coordinate is exactly the previous first coordinate. The master
noise-scale inference task multiplies .3 and .2 by exp(lambda_Q) and
exp(lambda_R), with independent Normal(0,log(2)²) priors. This is a proposed
parameter task, not an existing posterior. Exact Kalman likelihood is a
possible B endpoint and reference integrand even though Q is singular;
posterior quadrature, time-order/data partition choices and tuning remain open.

The five-parameter nonlinear structural posterior is a separate obligation.
`bayesfilter/testing/structural_ukf_neutra_target_design_tf.py:34` supplies
independent uniform bounds (.05,.05,.05,.02,.02) to (.98,1.25,.98,1,1),
probit coordinates (rho,sigma,phi,gamma,R), and initial covariance
diag(.04,.09). Its simulator at line125 observes x0 first, then uses
m'=rho*m+sigma*epsilon, k'=phi*k+gamma*(m')² and y=m+k+sqrt(R)*eta.
The second state depends on the same innovation through the new first state;
an additive full-rank Gaussian transition would change this law. The existing
UKF posterior is an approximation comparator, not the model-posterior oracle.
The full T100 task and the quadratic-completion fixture under
`experiments/dpf_implementation/tf_tfp/fixtures/structural_ar1_quadratic_tf.py`
remain separate; the latter's precise law has not been audited here.

The inspected canonical path is
`ledh_program_fixtures_tf.BoundedLEDHFixtureTarget.physical` ->
`ledh_canonical_batch_fused_tf.canonical_batch_fused_value_score` ->
`ledh_canonical_score_tf.canonical_value_and_analytical_score` ->
`safe_cholesky(model.process_covariance)`. The final engine requires a
positive-definite process covariance and records a factorization failure when
it is singular. Merely finding a structural callback elsewhere does not prove
that this path supports deterministic completion. The executable diagnostic
calls the same batch endpoint with the AR2 F,Q,H,R and full Contract E/GenUT/
dual-cap controls, then checks propagation of the original failure status.
It also tests a strictly positive covariance at the factorization boundary as
a no-fire control; that control is not an alternative AR2 model candidate.

Success for this diagnostic means the singular-process limitation is exposed
as invalid value and force with the factorization code, rather than silently
accepted or repaired. Unexpected acceptance, missing failure telemetry or a
crash is an audit failure. Neither outcome establishes posterior correctness.
Do not add a ridge to the structural transition or reuse an unrelated linear
scope's tuning to make this model appear covered. Supporting a lower-dimensional
innovation law requires a general canonical implementation and separate checks.

Use the tftwogpu Python environment with CUDA_VISIBLE_DEVICES=-1, two CPU
threads and XLA, N=K=8, one transition and two parameter rows. This is a tiny
CPU diagnostic exception, not the GPU production route. Reserve60 CPU-worker
seconds under P0, with a90-second wall timeout, for
`python -m pytest -q -s tests/test_surrogate_hmc_structural_scope.py`.
Preserve source hashes, command, environment, log and incremental accounting
in `surrogate-hmc-all-models-20260922-structural-scope-audit-01`.

Skeptical audit before execution: exact Kalman feasibility is not DCTR adapter
eligibility; the original process covariance is singular, not numerically
ill-conditioned positive definite; the nonlinear law is not additive Gaussian.
This check cannot count as surrogate-HMC inference, and all four posterior
comparisons remain NOT_RUN. The next action depends on the implementation
finding, not force accuracy or agreement with an endpoint gradient.
