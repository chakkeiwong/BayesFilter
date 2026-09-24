# Disturbance-coordinate score: document and audit result

Date: 2026-09-12. The LaTeX derivation is delivered. **A full MathDevMCP
audit pass has not been achieved.** The tool reports partial coverage and
three identities requiring manual formalization. They are not reported as
refuted, but unverified obligations cannot be counted as proofs.

## Documents

- [Standalone proposal, 9 pages](../../../papers/ledh_younis_kdm_score/disturbance_score_proposal.pdf), with [LaTeX source](../../../papers/ledh_younis_kdm_score/disturbance_score_proposal.tex).
- [Integrated manuscript, 43 pages](../../../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf), section 7.2, pages 21--28; [LaTeX source](../../../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex).

The standalone body is extracted from the integrated source, not maintained
as a second derivation. It contains one explicit regularity assumption, nine
propositions with proofs, and one corollary. The proof order is: the model on
fixed disturbance coordinates; total state and score derivatives including
the initial law; the Fisher identity; sequential importance correction;
conditional APF correction and full particle-measure induction; normalization
bias; exactly centered control variates and their optimal fixed coefficient;
a tractable reference control; conditional integration; and an exact Gaussian
example with parameter-dependent state support.

The claimed target is the observed-data likelihood derivative of the declared
degenerate model. The construction computes an unbiased **unnormalized**
likelihood/derivative pair under the stated hypotheses. Its normalized ratio
is generally biased at finite particle count. A derivative of a particular
finite LEDH program is a separate quantity. These distinctions are proved or
demonstrated explicitly, not hidden under a surrogate-score label.

The control-variate construction has a known centering constant from a
tractable reference model. It includes the reference-to-target density ratio.
Variance reduction requires the stated moments and a suitable coefficient;
neither a Gaussian reference nor empirical centering automatically supplies
that guarantee. Genealogical variance over long horizons and DSGE regime
nonsmoothness remain unresolved.

## MathDevMCP evidence

Final equation audit: [round 4](mathdevmcp-rigor-round4.md), with the detailed
JSON alongside it. It binds the integrated TeX SHA-256
`6feeafff29e5f810b7032b9c8915efd1a5f7feb32fb661cfc01d302f63c7ebf0`.
All 30 new equation labels were selected; the full manuscript has 109 labeled
equation targets. Selection is not proof. The reported status is
`partial_coverage`, with three gaps and five issues resolved by existing
context. The remaining gaps are:

| Equation label | Meaning | Reported limitation |
|---|---|---|
| `eq:disturbance-importance-pair` | Likelihood and derivative importance estimators | Manual formalization required |
| `eq:disturbance-analytical-score-recursion` | Accumulating the complete disturbance score along genealogy | Manual formalization required |
| `eq:disturbance-control-variate` | Centered unnormalized derivative estimator | Manual formalization required |

Separate source-bound label audits accepted the equation source and returned
zero mismatches, with the obligations still unverified because the stochastic
notation exceeded the bounded algebra backend. Those separate audits precede
a citation-anchor-only TeX edit; round 4 binds the final TeX. Absence of a
mismatch is not a pass. The report's coverage-level diagnostic-abstention
counter is zero even though its gap rows use `diagnostic_abstention`; the
gap ledger and source-label results, not that counter alone, determine this
interpretation.

The separate [derivation-tree audit](mathdevmcp-derivation-round2.md) extracted
16 of the 30 requested labels and omitted 14 multi-relation targets. It also
explicitly disabled document promotion because its backend evidence lacked
the required Phase 01 binding (`legacy_unbound_document_evidence` and
`document_repair_publication_quarantined`). This is a tool evidence limitation,
not permission to mark the document accepted or a mathematical refutation.

Five direct MathDevMCP/SymPy algebra checks returned `equivalent`: Gaussian
posterior-score equality, quotient-rule algebra, the control-variate variance
square, particle normalization cancellation, and ratio-bias arithmetic. Two
initial function-notation encodings returned `inconclusive`; all seven calls
are preserved in `mathdevmcp-symbolic-checks.json`. No Lean certificate or
complete machine proof was produced.

Round 1 incorrectly focused on proposition labels and selected no equation
targets. Round 2 selected all 30 and identified six gaps. Round 3 is stored
under the historical name `mathdevmcp-rigor-final.*`; it is superseded by
round 4. Four full audit rounds were used. Repeated parser submissions cannot
substitute for formalizing the probability and integration arguments.

## Checked repairs and exact references

The revision made the fixed-support and domination hypotheses explicit,
included initial-law dependence, corrected matrix orientations, supplied the
complete PF induction, and stated the exact posterior-proposal exception to
one-draw ratio bias. It distinguished a direct Fisher estimator from AD
corrections to a random program, restricted the Rao--Blackwell comparison to
the coupled estimators it actually proves, and included all density ratios
and moment assumptions required by the reference control.

An independent SymPy derivation starts from the Gaussian observation
covariance, differentiates its exact likelihood, and agrees with both the
displayed score and the conditional disturbance expectation.
`symbolic-verification.json` records PASS. Standard-library rational
arithmetic checks three moving-support parameter points and the missing
initial-law derivative. An exact enumeration of a two-particle, two-step
filter covers all 64 outcomes: expected likelihood is `33/128`, expected
unnormalized derivative is `1/32`, and their ratio is the exact score `4/33`.
The expected normalized genealogical score is instead `-19771/443520`.
`verification.json` and `enumerated-particle-filter.json` preserve the full
results. These are independent mathematical diagnostics, not a new
TensorFlow/GPU implementation or evidence of empirical superiority.

Both PDFs build without LaTeX warnings, undefined references, or box
overflow. The added integrated pages 21--28 and all nine standalone pages
were visually inspected. A final bibliographic correction uses DOI metadata
for Murray et al.: publication year 2013, volume 1(1), pages 494--521; the
local author manuscript is dated 2018. This changes the bibliography, not
the audited TeX equations. The corrected citation and reference pages were
rebuilt and rechecked. No baseline equation label or citation was removed.
Human readability remains for the user's review.

## Decision and limits

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Deliver the derived manuscript | Nine propositions, proofs, assumptions, and worked example supplied; PDF build passes | No failure in the exact checks | Paper proofs are not fully machine verified | Review the displayed derivations and assumptions | No implementation, runtime, or DSGE performance claim |
| Withhold full MathDevMCP acceptance | Not met: partial coverage and three unverified identities | Missing formalization and document-promotion support block an audit-pass claim | Integration and conditional-expectation obligations need a suitable formal backend | Formalize those obligations and establish source-bound proof coverage | No Lean certificate or whole-document PASS |
| Retain the proposal as mathematically specified | Exact unnormalized identities proved under the stated assumptions | Finite-N ratio-bias counterexample blocks an unbiased normalized-score claim | Variance growth, reference quality, regime boundaries | A separately planned model-specific variance diagnostic can test these questions | No default change, exact HMC force, or general variance guarantee |

| Inference status | Finding |
|---|---|
| Hard veto screen | Ratio unbiasedness is refuted in general; a full audit-pass claim is unsupported |
| Statistically supported ranking | Not applicable: no stochastic comparison campaign ran |
| Descriptive-only differences | No method ranking is inferred from the toy checks |
| Default-readiness | Not evaluated |
| Next evidence needed | Formalization for the audit; a model-specific implementation and variance study for practical claims |

The strongest remaining alternative explanation for future poor performance
is genealogical degeneracy or an unstable reference-density ratio, despite
correct expectations. A counterexample satisfying the written hypotheses
would overturn a proposition and require repair. The weakest current
verification is formal coverage of the stochastic induction and integration
arguments; numerical agreement on a finite example cannot close that gap.

Reproduction and hashes: `run-manifest.json`, `document-manifest.json`,
`final-build-commands.json`, and [source ledger](source-ledger.md).
