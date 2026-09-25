# KDM auxiliary native repair result

The proposed KDM native repair was not installed. Source audit confirms that
`bayesfilter/highdim/ledh_younis_kdm_tf.py::canonical_linear_gaussian_kdm_auxiliary`
still owns a Python time loop, performs eager `.numpy()` model-validity checks,
and creates its atom and KDM TensorFlow kernels inside that loop. Its reusable
mixture kernels already default to XLA, but that does not compile the enclosing
auxiliary.

The first candidate was removed before commit after its compatibility worker
failed the existing auxiliary validity gate. The restored baseline worker 03912
fails the same assertion (`result["valid"]` is false), so the failure cannot be
attributed to the candidate. No runtime bytes, numerical method, tolerances,
validity semantics, or external source were changed. The preserved baseline
failure is the correct evidence that this gap needs a separate, source-grounded
repair before any native candidate can be compared.

The next valid repair must add a private owner/factory around the existing
analytical score's TensorArray trace, prebuild the two KDM kernels once per
owner, and use a `tf.while_loop` for accumulation. It must compare the complete
public step records, canonical no-feedback fields, changed operands, invalid
model behavior, finite differences, graph/XLA host-callback absence and costs.
The public wrapper may materialize records and raise completed model mismatch
errors at the host boundary; the numerical body may not use `.numpy()`. This
gap remains open and diagnostic-only; it does not authorize a canonical LEDH
rebuild, claim-bearing KDM route, HMC, or scientific promotion.

The 03911 candidate and 03912 restored-baseline logs are preserved under the
campaign raw root. The campaign classifies the baseline worker as explanatory,
so its expected failure cannot be mistaken for a passing repair.
