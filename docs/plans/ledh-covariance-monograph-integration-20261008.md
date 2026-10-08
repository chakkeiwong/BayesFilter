# Integrate the audited covariance proposal into the monograph

Date: 2026-10-08. User instruction: integrate the self-contained proposal into
`docs/main.tex` and compile the monograph.

The reader knows Bayesian filtering and matrix calculus and should be able to
reconstruct the three covariance roles, marginal importance correction, reset,
bounded higher-moment correction, and total analytical derivatives in one
chapter. Preserve the audited argument, all equations, algorithms, assumptions,
source attributions, and limitations. This is document integration, not a new
filter implementation, numerical experiment, or change to canonical policy.

## Implementation and verification

1. Preserve the current standalone source/PDF and relevant monograph sources
   with SHA-256 records under
   `docs/plans/artifacts/ledh-covariance-monograph-integration-20261008-01/`.
2. Extract the full audited body into one shared LaTeX input. Include it through
   a new chapter immediately after the existing custom-gradient/reset chapter;
   keep the standalone wrapper usable. Namespace labels, scope local macros and
   formatting, and use the monograph bibliography for both documents.
3. Retain the existing scientific text exactly apart from mechanically checked
   label/citation substitutions. Add only a short chapter introduction connecting
   the proposal to the existing monograph and identifying its candidate status.
4. Build `main.tex` and the standalone wrapper with pdflatex/BibTeX until
   references stabilize. Check the full input chain, bibliography, all 49
   equation labels, both other labels, and all three algorithms. Inspect the
   rendered chapter, including the diagram and algorithm boxes.
5. Preserve logs and a concise result/checkpoint. Report pre-existing document
   warnings separately from defects introduced by this integration.

Budget: up to six pdflatex passes per document plus BibTeX and local layout
repairs, with a 45-minute initial build/debug budget. No GPU or scientific run
is involved. Stop and investigate missing substantive content, source drift,
unresolved new references/citations, fatal build errors, or unreadable new
material. Routine dependency/layout fixes remain within this task.

## Skeptical audit before edits

The comparator is the exact final standalone source already audited with
MathDevMCP, not an earlier draft. A successful compile alone cannot establish
mathematical preservation: verify the shared body reverses exactly to the
protected original after label/citation normalization. A shared body prevents
independent copies from drifting. The monograph uses a book class, chapter-based
theorems, and a central BibTeX database; importing an article preamble or a local
`thebibliography` would be wrong. Local macro/formatting scope and unique labels
are required. Existing bibliography entries must be matched by source identity,
not similar titles. The proposal remains a candidate; neither compilation nor
prior bounded symbolic checks promotes it or changes canonical runtime claims.
Audit passes with these controls. Human readability review remains pending.

Current stage: complete. Both documents compile; all equations, algorithms,
references, source input chains, and rendered chapter pages have been checked.
See [the integration result](ledh-covariance-monograph-integration-results-20261008.md)
and `artifacts/ledh-covariance-monograph-integration-20261008-01/verification.json`.
Five monograph and six standalone pdflatex passes were used, within the compiler
pass limit. Total elapsed time through PDF delivery was 128.7 minutes, exceeding
the initial 45-minute estimate. Routine macro/layout repairs, source verification,
and rendered-document checks continued within the authorized documentation scope;
no scientific run or expanded compute campaign was launched. Next action: deliver
the integrated PDF; no implementation is in progress.
