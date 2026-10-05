# KSC reset-mechanism diagnostic addendum

This explanatory extension remains inside the discrepancy plan's 14,400-second
allocation, original aggregate cap, elapsed deadline, and retry allowance. It
does not change the filter or select settings. Run after the planned particle
and numerical-control phases; stop if the existing budget is insufficient.

The d=1 residual design is the repeated pair (+1,-1). The shared reset uses a
kernel exp(-cost/(scale*epsilon)), row-normalizes its transport, injects this
design through a covariance-gap factor, and restores the target covariance.
As epsilon tends to infinity, identical transport rows give identical
barycenters. With this residual design, the restored cloud then has only two
locations, up to the subsequent higher-moment correction. This code-derived
limit motivates a check at the actual finite epsilon; it does not prove that
the observed discrepancy has this cause.

Use the existing canonical executor's trace, all four routes, the two selected
T120 datasets, N1008, and their original design seeds. The only arm is the
frozen baseline. Record weighted child and reset-cloud mean, variance,
skewness and kurtosis; the within-parity-group variance divided by the total
variance measures concentration around the repeated design's two groups.

For every t<120, integrate the next observation exactly conditional on each
particle. With Q=1 and component variance v_k, the kernel is

    K_theta(x) = sum_k w_k Normal(y_(t+1); phi*x + 2*b + m_k, 1+v_k).

Compute A=sum_j posterior_weight_j*K(child_j) and
B=sum_i outgoing_weight_i*K(reset_i). Report log(B)-log(A), and its total
directional derivatives for both parameters, including particle and weight
dependence. This isolates the immediate effect of the actual reset plus
higher-moment correction on the next predictive likelihood functional. Both
clouds already contain previous approximation error; these local changes must
not be summed and described as a decomposition of the full likelihood error.
Also report the actual next finite-program increment minus log(B), which
describes the ensuing proposal/particle-integration discrepancy conditional on
the reset cloud. Its tangent is diagnostic only, not another score estimator.

Validity: require normal-endpoint value/score parity within 1e-8; check this
diagnostic's analytic Gaussian-convolution value and tangent on a small
CPU-only independent calculation before launching. Preserve all per-time
measurements, source hashes, command, hardware settings and wall time.
Nonfinite measurements or failed parity invalidate this diagnostic and trigger
local repair; they do not silently invalidate completed filter evidence.

Self-review: the comparison uses the same trace and exact seven-component
kernel on both sides, so initial inputs and proposal noise cannot explain a
within-step difference. It measures a local mechanism, not its entire global
causal contribution. Fixed baseline controls remain untuned diagnostic
choices; no numerical protection or model is altered. CPU checks and endpoint
parity address implementation risk; only two retrospective datasets limit
generalization. Proceed with bounded diagnostic implementation and checks.
