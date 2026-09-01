# Generic DMIS and TT Control-Variate Repair: Execution Result

Date: 2026-09-01
Plan: `docs/plans/bayesfilter-generic-dmis-control-variate-repair-plan-20260901.md`
Plan review: `docs/plans/bayesfilter-generic-dmis-control-variate-repair-plan-review-20260901.md`
Status: completed for the bounded candidate/diagnostic scope

## Question and target boundary

The question was whether a model-independent complete deterministic-mixture
importance program, with an optional known-integral squared-TT control variate,
can evaluate the finite carried target while retaining an analytical derivative
of the same frozen finite program. The target is the finite carried-density
integral with exact transition and observation factors. This execution does
not claim the true C2 marginal likelihood, universal finite variance, exact
posterior inference, HMC readiness, or a production default.

## Skeptical audit

The plan passed the pre-execution skeptical audit in the linked review. The
audit checked the target and baseline identity, deterministic-bank versus iid
semantics, support and positivity vetoes, total-derivative scope, default and
assumption provenance, bounded retries, and the absence of a C2-specific
runtime fork. During implementation, a Class-B fail-closed edge case was
found: a zero proposal density with zero control residual could produce a NaN
in the directional path. The inverse density is now masked and the tangent
route explicitly reports invalid support unless every frozen row has positive
mixture density. The regression test and all reruns below include that repair.

## Changes executed

- The LaTeX document now contains proposition-proof statements for complete
  mixture identity, deterministic-bank masses, the squared-TT control
  variate, defensive-tail bound, frozen total directional derivative, affine
  recursive maps, recursive moments, error propagation, basis/reference
  boundaries, and the staged testing program.
- `bayesfilter/highdim/frozen_dmis_control_variate_tf.py` implements the
  model-independent TensorFlow value and explicit tangent kernels. It uses all
  mixture components in the denominator, accepts declared row masses, exposes
  support/finite/positivity diagnostics, and provides fixed-shape XLA factories.
- `tests/highdim/test_frozen_dmis_control_variate_tf.py` contains analytic
  finite-bank, Student, control-variate, tangent, compiled-parity, and
  fail-closed support tests.
- `docs/benchmarks/run_generic_dmis_c2_compatibility_smoke_20260901.py`
  exercises the generic kernel with a retained Hermite proposal and a fixed
  Student component against one exact C2 transition-observation factor.
- `docs/plans/lean/generic_dmis_control_variate_20260901.lean` gives the
  direct Lean companion proof.

## Evidence contract

The primary correctness checks were (i) an independently assembled finite
mixture denominator and closed-form finite weighted sum, and (ii) central
finite differences of the identical frozen value program. A nonfinite value,
unsupported required row, nonpositive normalizer, incomplete denominator, or
failed tangent check was a hard veto. ESS, residual second moment, shell
diagnostics, and C2 proposal comparisons remain explanatory or nomination
diagnostics.

## Commands and results

| Check | Command/result | Interpretation |
| --- | --- | --- |
| LaTeX pass 1 | `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=docs/benchmarks/artifacts/c2_completion_20260824/attempt05 docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex` | Exit 0; 38-page PDF produced. |
| LaTeX pass 2 | the same full command | Exit 0; cross-references settled. |
| LaTeX log scan | counts for `LaTeX Error`, `Undefined`, `Emergency stop`, `Fatal error`, `Overfull \\hbox`, `multiply defined` | All counts 0. Underfull boxes and hyperref math-string warnings remain nonfatal. |
| Render inspection | `pdftoppm` pages 27--38 plus `view_image` inspection | Equations, proposition proofs, tables, and the MathDev provenance section are legible; no clipping or overlap observed. |
| Python compile | `python -m py_compile` on module, tests, and C2 smoke | Exit 0. |
| Generic TensorFlow tests | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_frozen_dmis_control_variate_tf.py` | `11 passed, 2 warnings` in 5.96 s. |
| Existing C2 integration tests | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py` | `9 passed, 2 warnings` in 9.36 s. |
| Retained-proposal regressions | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_c2_gaussian_hermite_proposal_tf.py` | `6 passed, 2 warnings` in 10.07 s. |
| Default XLA factory smoke | CPU-hidden Python check calling both compiled factories with default `jit_compile=True` | XLA compiled successfully; value `3.550112243350956`, valid `True`; tangent `0.29843853533796244`, valid `True`. |
| Lean | `lake env lean /home/chakwong/BayesFilter/docs/plans/lean/generic_dmis_control_variate_20260901.lean` from `/home/chakwong/lean_sandbox` | Exit 0, no warnings or placeholders. |

All TensorFlow commands above were deliberate CPU-only diagnostics with
`CUDA_VISIBLE_DEVICES=-1`; they are not GPU production evidence.

## Exact MathDevMCP invocation

The nominal report used:

```text
mathdevmcp audit-math-document-rigor docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex --max-labels 0 --validation-backend sympy --validation-backend lean --backend-env mathdevmcp-backends --report-profile forensic --response-mode detailed --output-md docs/plans/bayesfilter-generic-repair-mathdevmcp-final-20260901.md --output-json docs/plans/bayesfilter-generic-repair-mathdevmcp-final-20260901.json --artifact-root docs/plans/mathdevmcp-final-rerun-artifacts-20260901
```

The five focus invocations used the same flags with the corresponding
`--focus-label` list recorded in each batch JSON report. The special-label
invocation supplied `--focus-label eq:actual-transition-retention
--focus-label eq:actual-feedback-loop` and wrote
`docs/plans/mathdevmcp-final-special-labels-20260901.{md,json}`.

## MathDevMCP audit

The nominal command was run with `--max-labels 0`, SymPy and Lean validation,
the `mathdevmcp-backends` environment, and the forensic profile. It completed
with exit 0 and source digest
`536681032a2758687816667d7e5b5f9ade3d8a30c796afca261889694d20c517`.
The parsed inventory contains 174 equation rows, 122 labeled equation rows,
zero duplicate labels, and zero missing references. The CLI selected 30 rows
under its internal response budget, reported `partial_coverage`, 10 gaps, and
3 proposals, and reported no algebraic counterexample. Its gaps are primarily
formalization or dimension/invertibility obligations; they are not proof of a
mathematical failure.

Five refreshed, disjoint focus batches cover 24, 24, 25, 25, and 24 labels.
Their union is exactly 122 labels, equal to MathDevMCP's parsed inventory, and
the nominal report and those five batch reports (six reports total) carry the
digest above. The separate special-label report carries the same digest. Two raw `eq:` labels
(`eq:actual-transition-retention` and `eq:actual-feedback-loop`) are attached
to aligned schematic blocks that the parser does not classify as equation
rows. The focused special-label run selected zero rows; those two blocks were
manually inspected. Thus the honest coverage statement is: every parsed
labeled equation row was sent through a bounded MathDevMCP focus audit, while
the two parser-unclassified schematic labels were manually reviewed. This is
a structured rigor audit, not a proof certificate for continuous or stochastic
claims.

Reports:

- `docs/plans/bayesfilter-generic-repair-mathdevmcp-final-20260901.md`
- `docs/plans/mathdevmcp-final-batch01-20260901.md`
- `docs/plans/mathdevmcp-final-batch02-20260901.md`
- `docs/plans/mathdevmcp-final-batch03-20260901.md`
- `docs/plans/mathdevmcp-final-batch04-20260901.md`
- `docs/plans/mathdevmcp-final-batch05-20260901.md`
- `docs/plans/mathdevmcp-final-special-labels-20260901.md`

The refreshed commands supplied
`--artifact-root docs/plans/mathdevmcp-final-rerun-artifacts-20260901`. In the
CLI's detailed mode the evidence is embedded in the listed JSON/Markdown
reports, and no separate artifact directory was materialized. Pre-layout
reports are preserved with the `-pre-layout` suffix for provenance.

The reports' repeated `needs_formalization` and invertibility entries are
explanatory routing findings. The document now states the relevant SPD,
positive-Cholesky-diagonal, dimension, support, and UKF innovation conditions
in its own text; MathDevMCP does not turn those contextual repairs into a
certificate.

## Lean boundary

The Lean file directly proves, over a finite index type and a field:

1. complete-mixture cancellation;
2. the known-integral finite control-variate identity;
3. the explicit tangent quotient identity; and
4. positive denominator implies nonzero denominator.

This is a clean finite algebra proof. It deliberately does not formalize
measure-theoretic measurability/integrability, stochastic unbiasedness,
TensorFlow/XLA execution, or finite-sample variance. Those boundaries are
stated in both the Lean comments and the LaTeX document.

## C2 compatibility smoke

The final smoke was written to a fresh `attempt02` directory after the
fail-closed tangent repair. It used 64 frozen rows (equal retained-Hermite and
product Student banks), exact C2 transition times observation at a fixed
parent, and the complete two-component denominator. It passed all validity and
recomposition checks:

- normalizer: `0.4699604414743356`;
- log normalizer: `-0.7551067548961616`;
- target-weight ESS fraction: `0.7877474564322884`;
- maximum tangent finite-difference error: `5.338063324700215e-11`.

The artifact is
`docs/benchmarks/artifacts/c2_generic_dmis_compatibility_20260901/attempt02/`.
The earlier `attempt01` is preserved as a prior execution record; it was not
overwritten. These numbers demonstrate compatibility of the generic kernel
with the C2 factor APIs only. They are not a C2 marginal-likelihood estimate,
an efficiency ranking, or evidence for a production route.

## Decision and inference status

| Decision | Primary criterion | Veto status | Decision |
| --- | --- | --- | --- |
| Generic finite DMIS algebra | Independent denominator and finite weighted identity | None fired | Passed. |
| Known-integral control variate | Finite positive result and residual diagnostic on analytic fixture | None fired | Passed as a candidate mechanism. |
| Explicit analytical tangent | Same-program central finite difference | None fired after support repair | Passed for the frozen finite program. |
| TensorFlow/XLA implementation | Eager/compiled parity and default-XLA smoke | None fired | Passed as a candidate/diagnostic kernel. |
| C2 wiring compatibility | Exact one-step factors, complete denominator, finite tangent | None fired | Passed as compatibility diagnostic only. |
| Proposal efficiency or variance reduction | ESS/second-moment comparisons | Not a correctness gate | Unresolved; no promotion claim. |
| Recursive high-dimensional filter repair | Full horizon, target-specific and GPU evidence | Not run in this bounded stage | Unresolved; no default claim. |

| Inference class | Supported statement |
| --- | --- |
| Hard veto screen | The listed finite-bank, support, tangent, compilation, and smoke screens passed. |
| Statistically supported ranking | None; no multi-seed uncertainty analysis was run for this candidate kernel. |
| Descriptive-only differences | The C2 ESS and normalizer values are descriptive compatibility diagnostics. |
| Default readiness | Not established; the module remains `candidate_diagnostic_only`. |
| Next evidence needed | Scope-specific proposal ladder, independent finite-target integration at the earliest divergent time, recursive moment-map validation, and paired multi-seed uncertainty analysis. |

## Post-run red-team note

The strongest alternative explanation is that the small C2 smoke is too benign:
it has one fixed parent, one time step, and frozen proposal logs, so it cannot
expose recursive state error, poor TT tails, or adaptive-map derivative terms.
The conclusion would be overturned by an independently integrated finite target
that disagrees with the complete-DMIS value, or by a multi-step run with a
required-support/tangent failure. The weakest evidence is proposal efficiency:
64 rows and one fixture do not support a variance or ranking claim.

## Hashes and provenance

- TeX source SHA-256: `536681032a2758687816667d7e5b5f9ade3d8a30c796afca261889694d20c517`
- PDF SHA-256: `a6f6230892551caf6bd8163187ac16c0c04f3ab1dc09a1387cd0e539b18f4747`
- Generic module SHA-256: `e863b42195686629a920fc4a41ba2f82ddf674c46f03439c37895b6ae2b5edc6`
- Test SHA-256: `fb463edf3fa2145b3cd069c7f4c9f2fa0a42bfc4281d755751cbafbcba15aa8a`
- C2 smoke script SHA-256: `e617c3852dc99b1280c01112575cb6dc0017d35e5979fb43039f47f9f5ea8b57`
- Lean SHA-256: `4a0951e4c5dd8c7a9e1c981e22a0a9aa7f782c7f25f88be82afe9126bea4d799`
- Final C2 `attempt02/result.json` SHA-256: `1e5ef5f601d141d8edd432717481d1bf604e7b8a5fb90cb3b024eea3cfd39022`
- Git commit recorded by the smoke: `7bee7a660f19f542470af0f2a19f6cf4070f2fd7`
