# Range–bearing likelihood scope audit

The all-model surrogate-HMC program needs a reference for the law that actually
generates each scope's data. This audit checks the angular observation law in
`experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py`; it does
not run inference or introduce a replacement DCTR implementation.

The simulator (lines206–235) adds diagonal Gaussian observation noise and wraps
the bearing into [-pi,pi). The density callback (lines160–197) wraps the
residual and evaluates one Gaussian density. If the angular residual is d and
the bearing standard deviation is s, the simulator's density on the circle is

    p(d | s) = sum over integer k of Normal(d + 2*pi*k; 0, s²).

Each term accounts for a different preimage of the wrapping map, whose
Jacobian is one. Integrating the sum over one circle partitions the real line,
so its integral is one. The callback keeps only the nearest term. Its integral
is erf(pi/(sqrt(2)*s)), which depends on s and is less than one. Renormalizing
that single term would define a truncated angular law, still different from
the simulator's wrapped law. This matters for the proposed noise-scale
posterior, even when aliases are negligible at the fixture's s=.04.

Pre-run skeptical audit: compare the actual callback with an independently
constructed finite alias sum, including a refinement check. Use s=.04 and
s=pi: the former checks the original narrow-noise regime, the latter exposes
the missing mass without relying on underflow. Both are deterministic
capability probes, not draws from a posterior or new filtering results.
Keep the radial Gaussian factor identical and divide it out explicitly.
The covariance stays positive definite. The alias and midpoint counts below
are diagnostic hypotheses; check their refinements before interpreting them.

The primary criterion is whether the callback integrates to the wrapped law's
unit mass. Numerical-refinement failure vetoes the diagnostic. For s=pi,
require the callback integral to agree with erf(1/sqrt(2)) within1e-7 and the
alias integral to agree with one within1e-10. Compare 4 versus8 aliases on
each side, and 4096 versus8192 midpoint angles. Narrow-noise agreement is a
no-fire control, not general correctness evidence. No ranking, posterior
equivalence, original-data discrepancy, default change or surrogate-HMC
rejection may follow from this check. Existing historical results are not used.

Reserve60 CPU-worker seconds and90 wall seconds under P0 and the unchanged
54,000 CPU campaign cap. Run exactly the focused diagnostic tests once, with
GPU devices intentionally hidden, TensorFlow/TFP, stable graph signatures,
XLA, and two threads. A localized test/harness defect may be repaired within
that same reservation. Stop for refinement failure, non-finite output or
exhausted reservation. Seeds are inapplicable to deterministic quadrature.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q -s
tests/test_surrogate_hmc_range_bearing_scope.py`, with
`CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_PRELOAD_CUSTOM_OP=0
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2`.
Preserve the exact command, checkout, source snapshot, CPU choice, accounting,
diagnostic numbers and complete log under
`docs/plans/artifacts/surrogate-hmc-all-models-20260922-range-bearing-scope-audit-01`.
Record the result in `surrogate-hmc-range-bearing-scope-result-2026-09-22.md`.

The next action after a confirmed mismatch is to require a normalized,
refinement-qualified wrapped observation likelihood for this scope's reference
and declared model endpoint. Proposal forces may remain imperfect; agreement
with the endpoint gradient is not a surrogate-HMC requirement. This audit
must not turn an observation-law finding into a force-accuracy veto.
