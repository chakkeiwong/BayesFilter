# Reference compatibility addendum: qualification-design preflight

This adds one inspected difference to the unchanged
[finite-filter source audit](bayesfilter-mams-reference-source-audit-2026-10-12.md).
It applies to target source commit `e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`.
The original reference archives and previous compatibility reports are preserved.
The earlier audit note is not edited, because existing reports bind its bytes.

The first `reference-consumer-03` attempt stopped before numerical work in
2.224093 wall seconds: the diagnostic reader rejected an unreviewed change to
`bayesfilter/inference/hmc_acceptance_statistics.py`. This is the intended
source guard, not an integrand discrepancy or sampler failure.

The reviewed change appends only
`replicated_qualification_attainability` at line 271. The archived digest is
`c0a4499ff35d244e0eb6e1b08f5476c692a070ce4d013cf7a1403e4c7cf81dcd`; the new digest is
`81bd3bb8fcc14d932a494d1d90cf01688f2f3618d6c5bf660821c2030c67eb69`.
Removing that one function from the parsed new module gives an AST exactly
equal to the old module, including imports, constants and every previous
function body. The new function has no decorator or module-level invocation.
It calls the existing acceptance statistic on synthetic extreme scores and
does not accept or call a model adapter, filter, observation array, coordinate
chart or quantity transformation.

The call chain in `run_modern_hmc_frozen_full_model.worker` invokes this
preflight before `full_model_binding`; the consumer wiring regression verifies
that an impossible budget stops before that binding is constructed. Target
preparation still uses `price_mams_adaptation_full_models.prepare`,
`modern_prepared_nonlinear.prepared_binding`, the actual factory selected by
`modern_nonlinear.make_nonlinear_comparison_target`, and the direct target
value/score delegation described in the original audit. None of those finite
filter or chart calculations changed in this commit. The new declaration
cannot alter the reference integrand by import, and the unchanged target
evaluation does not call it. Changes to candidate inventory/reporting and
supervisor bookkeeping are outside that mathematical evaluation.

The compatibility reader pins exactly the two inspected hashes. It does not
accept arbitrary later changes to the statistics module. A regression checks
the actual additive function, equality of all earlier AST, acceptance of the
reviewed source pair, and rejection after a further executable change.

Skeptical audit: the exclusion is supported by an inspected additive diff and
the call chain, not by a few parity probes or a filename exemption. Retain the
original value/score/quantity tolerances and reference-rung reaggregation.
Repeat the four-target numerical audit from a committed diagnostic driver,
with GPUs hidden and a fresh `reference-consumer-04` output, under the existing
420-second local allocation. A failed source, data, chart, native status,
parity or reference-refinement check vetoes the affected posterior comparison.
The independent qualification-only GPU deployment does not consume this
reference and may continue. Compatibility is not established until the fresh
audit succeeds; no sampler ranking, exact nonlinear truth or release follows.
