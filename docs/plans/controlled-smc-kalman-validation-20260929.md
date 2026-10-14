# Controlled-SMC exact Kalman control

This closes the first tranche's analytic consumer check, not its general-model
or empirical-replication requirements. The owner authorized the master execution.

Question: does one backward quadratic fit through the actual scalar controlled
SMC endpoint recover the optimal twist for a linear Gaussian model? The exact
comparator is an independent scalar Kalman recursion using Python math. A second
independent recursion supplies the optimal policy, rather than comparing only
two calls to the same filter implementation.

Use x0~N(.2,1.3), xt|x(t-1)~N(.8*x(t-1),.35), yt|xt~N(xt,.4), and
observations [-.7,.1,1.4,-.4,.8,1.1,-.3,.2]. These are deliberately constructed
identity-test settings, not a recovered paper experiment. N=64 gives an
overdetermined three-coefficient fit; 8 dates exercise the backward recursion.
Seed 20260929 and stateless streams 71–76 are fixed diagnostic choices, not
selected on likelihood outcomes. FP64 and TF32 off isolate algebra from reduced
precision. GPU, memory growth, stable tf.function signatures and XLA are required.

For psi_next(x)=exp(-a*x^2-b*x-c), integrate under N(alpha*x,q).
Writing D=1+2*q*a and k=q/D, the negative log of this integral has coefficients
(alpha^2*a/D, alpha*b/D, c+.5*log(D)-.5*k*b^2). Add the observation coefficients
(1/(2*r), -y/r, y^2/(2*r)+.5*log(2*pi*r)). Starting from the last observation
gives psi_t=g_t M_(t+1)(psi_(t+1)), the optimal recursion in Heng et al.,
section 4.1, Proposition 1 and equation (19), specialized to state potentials.
With the initial normalizing integral stored separately, as in this consumer,
every remaining incremental weight is one; that initial integral equals the
Kalman likelihood. The paper includes the initial integral in its G0 instead.

Promotion criterion for this identity only: absolute plus relative tolerance
1e-8 for all policy coefficients, exact Kalman log likelihood, zero incremental
log weights and ESS=N, across two fresh evaluation streams. The unfitted
bootstrap run is a finite mechanics comparator; its Monte Carlo error is
explanatory, never a ranking or tuning target. Invalid precision, nonfinite
results, deficient regressions, a fired source precision projection, or a
failed identity veto correctness. One trace per signature is an engineering
criterion; runtime and memory are explanatory. A failed identity triggers
localization. Wrong target, corrupt oracle or exhausted budget stops the run.

The scalar projection remains the declared source-comparison setting; this
Gaussian fixture must not activate it. It cannot validate the projection's
behavior at an invalid precision boundary or promote the fixed .001 margin.
Likewise a well-conditioned Gaussian fit says nothing about general nonlinear
regression stability. No statistical superiority, general multidimensional
filter, HMC, default-readiness or whole-paper replication follows from a pass.

Exact command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B
scripts/run_controlled_smc_kdm.py run --phase c2_kalman

Cap: 120 GPU seconds, charged to the unchanged 7200-second tranche and inherited
campaign balance. Save source snapshots, seed/configuration, both reference
recursions, all checks, device/memory/tracing diagnostics, wall time and manifest
under the next versioned attempt. No external reference environment is needed.

Skeptical pre-run review: PASS for this limited deterministic identity. This
would be misleading if a sample mean were treated as an exact posterior moment,
if the Kalman model used a different initial date, or if exact-twist normalization
were inferred from ESS alone. The checks instead use the same x0 convention,
an independent normalized likelihood, explicit coefficients and every incremental
weight; no finite-particle moment or performance claim is made.
