# UKF covariance proposal: LaTeX and MathDevMCP audit

Date: 2026-10-08. This completes the requested self-contained mathematical
document and audit. It does not implement or promote a new filtering runtime.

[Read the PDF](../papers/ledh_covariance_proposal_20261008.pdf) or
[open the LaTeX source](../papers/ledh_covariance_proposal_20261008.tex).
The note develops the variance distinction from an AR(1) example, gives a
specified UKF/common proposal and conditional affine flow, derives their
marginal importance correction, specifies the actual-moment Cholesky reset,
and gives a bounded optional correction of selected third/fourth moments.
It includes total analytical derivative recurrences and three complete
pseudocode routines. Seven primary references ground the borrowed mechanisms;
the combined algorithm is explicitly a proposal.

Integration update, 2026-10-08: this proposal is now Chapter 27 of
[the monograph](../main.pdf), with a
[shared mathematical source](../chapters/ledh_covariance_proposal_body.tex)
used by both PDFs. The [integration result](ledh-covariance-monograph-integration-results-20261008.md)
records exact preservation of the audited body after reversible label, citation,
macro, and layout changes. The audit findings and limitations below are unchanged;
integration did not rerun or extend the machine proof audit.

## What the audit established

MathDevMCP processed all 49 labelled equations in the initial source in
13 disjoint batches. Its reporting interface rejected both an all-label
actionable request and an all-label forensic request with the same four-label
limit; splitting the requested coverage preserved every label. Each batch
correctly reports partial coverage individually. Their union covers the
49 initial labels, not every unlabelled display or prose claim.

The initial machine ledger emitted 38 gaps: 24 diagnostic abstentions and
14 proposed assumption clarifications. Two other source-role issues were
classified as resolved by existing context. Nine labels received no issue;
that is not a proof verdict.

Eleven of the proposed clarifications concerned invertibility. The manuscript
already assumed positive definite covariances, invertible flow maps, or proved
the skew-matrix Cayley inverse valid. These conditions have now also been
placed immediately beside the affected calculations. Three proposed repairs
introduced a Neumann series and a spectral-radius condition for an unrelated
matrix Omega. No such matrix or series occurs in those derivations. Those
suggestions were rejected; inserting them would not justify the actual
covariance or affine-map inverses.

A focused second audit processed the revised UKF gain, Cholesky reset, Cayley
map, and Cayley derivative. It still emitted four flags: one unrelated Neumann
proposal and three requests for invertibility conditions already present in
the adjacent prose and proofs. These machine flags remain visible in the raw
report. Their manual disposition is contextual false positives; they are
not silently converted into machine-proved results.

Eleven bounded symbolic attempts were made through MathDevMCP's SymPy route.
Nine returned proved. Two expressions using a derivative function returned
not_encodable with "'Symbol' object is not callable"; the expanded algebraic
forms were then checked successfully. The later checks validate those supplied
algebraic forms, not an independently parsed symbolic differentiation step.
The Cayley examples are for a parameterized centred three-particle matrix;
the general-dimensional argument is the explicit proof in the LaTeX.

There is no full-document formal proof certificate. The tool's unresolved
source-role/encoding obligations, the manual mathematical review, and the
bounded backend equalities are separate evidence.

## Disposition of the substantive checks

| Question | Checked reasoning or correction | Status |
| --- | --- | --- |
| Which covariance belongs to an ancestor? | Total covariance separates each conditional C_j from the between-ancestor term. The AR(1) calculation illustrates why replacing the conditional covariance by Q does not replace the whole cloud's covariance by Q. | Derived in the note; scalar identity checked by MathDevMCP. |
| Where does UKF covariance enter? | A_G = L_plus L_minus^{-1} maps predictive moments to UKF-updated moments when those supplied moments are exact. A concrete scaled sigma-point rule is now supplied. | Matrix identity derived; moment approximation is explicitly a proposal guide. |
| Are the proposal densities correct? | Maps are fixed before drawing each integration input, with independent pilots when needed. The actual affine determinant and every ancestor/branch appear in q. Frozen label probabilities are used by both sampler and denominator. | Change-of-variables derivation checked manually; no implemented call-chain claim. |
| Is the likelihood counted twice? | The increment is the mean of g p/q. The UKF normalizer is not multiplied into it. | Conditional one-step identity proved in the note; no unbiased full-likelihood claim. |
| What does the reset preserve? | The mean and covariance of the actual weighted particles, using actual column normalization and actual barycentre covariance after finite Sinkhorn iterations. | Matrix proof in the note; scalar reset identity checked symbolically. |
| Can all constraints always be met? | A nonnegative two-particle counterexample shows exact weighted covariance and support can be incompatible. Positive definite factors are required; adding a ridge would change the target. | Explicit mathematical limitation, not repaired by tuning. |
| Why can kurtosis change? | Ensemble-column orthogonal mixing fixes mean/covariance but changes higher moments. Two four-particle examples have variance one and fourth moments one and two. | Constructive explanation; no attainability claim under an arbitrary cap. |
| Can the higher-moment fit explode? | A bounded skew parameterization gives a nonsingular Cayley map and global standardized and coordinate displacement bounds. These bound only the extra rotation. | General proof in the note; centred 3-particle identities checked symbolically. |
| Can the optimizer fail? | The zero-start gradient method can stall at a symmetric cloud. Its stationarity is derived; finite steps need not reduce the objective. | Open numerical concern stated plainly; optional stage unpromoted. |
| Is the reported score a total derivative? | Source weights/moments, every mixture component, Cholesky factors, finite Sinkhorn iterations, map coefficients and finite optimizer updates all have recurrences. | Mathematical specification checked; runtime implementation and finite-difference parity not run. |
| Does the score equal the true model score? | It differentiates the fixed-design finite likelihood computation. Agreement with the true likelihood/score needs separate reference comparisons. | Not established by these identities or this audit. |

## Automated suggestions rejected or clarified

The 14 initial proposals concern:
UKF gain; global map and covariance; conditional bridge moments; mapped density;
Cholesky reset; Cayley map; dual caps; moment objective; inverse tangent;
component-density tangent; UKF tangent; global-map tangent; and Cayley tangent.

The gain, mapped-density and moment-objective suggestions incorrectly invoked
a Neumann series. Their actual sufficient conditions are respectively
S_obs positive definite, the component affine matrix invertible, and P_w
positive definite so its Cholesky factor is invertible. The remaining proposals
requested invertibility that follows from the displayed assumptions or the
Cayley norm proof. The revision made these facts local and explicit.

No guard was removed to obtain an automated pass. No full-filter accuracy,
cross-model non-regression, higher-moment convergence, unbiasedness, or HMC
claim follows from the absence of a generated counterexample.

## Build and rendering

The standalone source builds with pdflatex from the existing TeX installation.
latexmk is unavailable on this machine; no package or environment was installed.
An initial double-subscript notation error was fixed. Two algorithm-box
indentation overflows were corrected after visual inspection. The final log
has no undefined citations/references, overfull boxes, or underfull boxes.
All equations and pseudocode remain in the main document.

The final typesetting change after the focused audit adds only two noindent
commands before full-width algorithm boxes. The audited revision is preserved,
and the manifest verifies that removing precisely those two commands restores
the audited bytes. The mathematical source is unchanged by that last fix.

Commands:

    pdflatex -interaction=batchmode -halt-on-error \
      -output-directory=docs/plans/artifacts/ledh-covariance-latex-audit-20261008-01/build \
      docs/papers/ledh_covariance_proposal_20261008.tex

The build was repeated to settle cross-references. The build directory retains
the final full log, extracted text, and rendered page images. Rendered pages
were inspected for equation layout, covariance explanations, citations and
pseudocode continuity. Human assessment of the teaching narrative remains
pending; compilation and model review do not establish human readability.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Deliver the self-contained LaTeX/PDF and audit | Complete derivations, three algorithms, compiled document, preserved MathDevMCP evidence | No unresolved build defect or mathematical counterexample found; machine formalization remains incomplete | Numerical usefulness, tail behaviour, optimizer stagnation, support/rank failures, and implementation fidelity | Review the mathematical proposal; if implementation is requested, start with the simpler conditional-covariance repair and protected value/score comparisons | No default change, runtime correctness certificate, empirical improvement or HMC readiness |

No stochastic method comparison was run, no candidate ranking is supported,
and no new defaults were selected. The canonical runtime and pre-existing
diagnostics were preserved.

## Evidence locations

All raw outputs and source snapshots are under
[the versioned audit directory](artifacts/ledh-covariance-latex-audit-20261008-01/).

- Initial labelled-equation audits: rigor-01.json through rigor-13.json, with
  adjacent compact Markdown and detailed agent-reports directories.
- Focused revision: revision.json and revision.md.
- Symbolic checks: ar1_symbolic.json and symbolic-*.json.
- Machine-readable coverage, per-label status, and checks: audit-summary.json.
- Source hashes, environment, commands and build verification: manifest.json.
- Protected source snapshots: audited-initial.tex and audited-revision.tex.
- Existing source survey and paper/code ledger:
  [literature proposal](ledh-covariance-followup-survey-and-proposal-20261007.md).

The strongest alternative explanation for favourable future metrics would be
a proposal or reference mismatch rather than a successful covariance repair.
A failure of density normalization, total derivative parity, support, or
held-out reference agreement would overturn a correctness/performance claim.
This documentation audit makes no such empirical claim.

## Subsequent beta-selection addition

The later Section 5 addition (monograph Section 27.5) supplies eight new
equations and Algorithm 0; the original 49 equations and three algorithms
remain unchanged. The old audit counts above describe the earlier stage, not
full coverage of this addition. See
[the separate selection results](ledh-proposal-beta-calibration-results-20261008.md)
for the focused MathDevMCP extraction limits and independent symbolic checks.
