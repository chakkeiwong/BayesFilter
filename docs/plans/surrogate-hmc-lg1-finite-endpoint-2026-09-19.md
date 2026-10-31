# LG1 finite DCTR endpoint qualification

This preparation answers whether an explicitly defined full-DCTR finite
posterior can support the master program's C/D, D/R and C/R comparisons. B/R
continues independently. It does not change the model: stationary LG1, T10,
Q=1, R=.5, uniform phi on (-.95,.95), the attempt-4 claim observations, and the
same selected complete N1024 DCTR computation and frozen random inputs.

The existing FP32 unconstrained evaluator is not a total numerical endpoint:
at extreme finite z, sigmoid rounds to a prior boundary. Its status correctly
calls this a numerical failure. Replacing that failure by zero density would
silently restrict the posterior. No such restriction is allowed here.

Define a new, named finite endpoint before sampling:
`lg1_dctr_physical_fp32_chart_fp64_v1`. For physical phi, round to FP32 and,
only if this rounds onto a prior endpoint, use the adjacent interior FP32 value.
Evaluate the unchanged full canonical DCTR physical log density at that value.
For finite z, compute phi=-.95+1.9*sigmoid(z) in FP64 and add the stable FP64
log-Jacobian to the promoted physical density. The physical routine includes
the constant uniform prior density already; do not add it again. Its FP32
roundoff in this constant cancels on normalization. The proposed force remains
the already selected, explicitly imperfect position-only field.

This defines a different finite arithmetic program from promoting the entire
old FP32 unconstrained scalar. Record that distinction in identities and
results. It preserves the complete DCTR likelihood algorithm and statistical
model; it does not confer exact-gradient status on the force. The adjacent-value
rule is a numerics-altering endpoint protection, not a support exclusion, and
must emit a boundary-rounding flag. It is a candidate until the checks below
pass. The float32/float64 bounds must be checked to bracket the same intended
prior; endpoints differ by representational rounding only.

Before a C or D inference launch, perform a dedicated bounded qualification:

* CPU wiring checks substitute an instrumented full physical evaluator, verify
  the prior/Jacobian exactly once, preserve value-failure status independently
  of force status, and exercise finite extreme logits and true physical support.
* GPU checks use the actual full route. At interior physical inputs the
  likelihood value and status must be identical to the original physical
  routine. On nominal logits (-3,-1,1,3), compare new and old endpoint values;
  require absolute change <=1e-3*(1+abs(old value)). This threshold is a
  conservative non-harm screen for FP32 coordinate-rounding changes, not a
  posterior-accuracy margin or a tuning target. Record actual differences.
* At logits (-100,-20,20,100), require finite endpoint values, original DCTR
  value-valid status and explicit rounding flags. Force validity is recorded
  separately and does not become endpoint validity. An underlying DCTR value
  failure remains a failure and cannot be hidden by the coordinate protection.
* Check actual endpoint replay, row permutation/independence, and that the
  proposal force still resolves to the same frozen full implementation.

The continuation vetoes are a wrong prior/Jacobian, hidden endpoint failure,
changing random inputs, row dependence, failed non-harm screen or missing
flags. A veto prevents this endpoint from supporting C/D/R. Force mismatch is
explanatory only. Successful qualification permits a separate measured
quadrature/tuning/sampling plan; it establishes no posterior accuracy itself.

Budget: at most one 300-second GPU qualification job plus focused CPU tests,
charged to the existing LG1 allocation. No inference attempt is consumed by
these mechanics checks. Use a new `lg1-finite-endpoint-*` artifact directory,
preserve exact source hashes, controls and reference bindings, all measured
values/statuses, memory policy, command, time and a machine-readable verdict.
The remaining quadrature and C schedule must be forecast before launch.

Skeptical audit: the main risk is silently changing the finite target while
claiming the old scalar. The new identity and explicit formula prevent that
claim. Healthy physical evaluations are unchanged; the new rounding rule is
exposed and independently exercised. The acceptance endpoint is independent
of proposal-force validity. Neither a good pointwise tie-out nor success of B
establishes C/D or D/R, and the old endpoint is retained as the comparator.
