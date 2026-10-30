# iAPF source recovery and correction, 27 September 2026

> Source update: the Bristol publisher PDF has now been downloaded and verified.
> Local copy: [.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf).
> The cover identifies it as the version of record; Section 5.1 says
> "a regularized version" of Eq. (15). Previous statements that this PDF
> was unavailable describe the earlier retrieval attempts and are superseded.
> Cover and fitting passage inspected; full version reconciliation remains open.

## Finding and scope

The accepted manuscript used by the earlier fitting studies is not the final
published text. Publisher-indexed Section 5.1, page 1640, specifies
"a regularized version" of Equation (15); those words are absent in the local
accepted manuscript. Direct publisher and Bristol PDF retrieval returned HTTP
403, and the CORE mirror returned HTTP 522. The published PDF has therefore
NOT been locally recovered or read in full; the indexed wording is corroborating
evidence, not a substitute for a complete source.

The first author's August 2017 thesis has now been downloaded and inspected.
Section 7.3, printed pages 53–55 (PDF pages 61–63), explicitly derives the
vanishing-density/vanishing-amplitude degeneracy. It introduces a general
Tikhonov penalty, but states that all Part III state-space examples used zero
explicit penalty and R optim with BFGS. Local optimization supplies the intended
implicit regularization. L-BFGS-B box constraints are suggested for problematic
cases; an explicit penalty is used for the separate diffusion examples in Part IV.
Consequently the final article's wording does NOT establish that an explicit
penalty was used for the linear-Gaussian experiment.

Section 8.2.1, printed page 67, confirms the same linear-Gaussian experiment:
d=5,10,20,40,80; T=100; alpha=.42; 1,000 replicates; iAPF N0=1,000,
BPF N=10,000, fully adapted APF N=5,000. Section 7.1 gives k=5 and tau=.5.
Chapter 6, Section 7.1–7.3, Section 8.2 and the introductory discussion of
Chapter 11 were inspected for the current implementation question; this is not
a claim to have audited all 149 PDF pages.

## What changes

The earlier root-cause derivation remains correct for the unregularized
objective and the saved failed trajectories. The thesis independently discusses
that degeneracy. It is not evidence that the authors overlooked the problem,
that no useful local optimum exists, or that their filter cannot work.
The optimizer family and the explicit-penalty choice for Part III are now
recovered. Claims that all original optimizer choices are unknown are stale.

The original code, starting values, covariance/amplitude parameterization,
objective scaling, optimizer controls and positive-floor formula remain
unrecovered. Profiling lambda gives the same minimum over lambda, but changes
the finite BFGS path compared with joint optimization. The previous BFGS
counterfactual used profiled lambda, standardized coordinates, fixed scaling
and reltol=1e-14. It did not exhaust the source-described BFGS procedure.

We will test these remaining choices as reconstruction hypotheses, not silently
identify any successful candidate with the author's implementation. Explicit
bounds are a source-suggested repair family, but the bounds themselves are a
local choice and change the constrained optimization problem.

## Sources and local copies

- Published article DOI: https://doi.org/10.1080/01621459.2016.1222291
- Publisher full text (indexed Section 5.1):
  https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1222291
- Thesis landing: https://wrap.warwick.ac.uk/id/eprint/96047/
- Thesis PDF:
  https://wrap.warwick.ac.uk/id/eprint/96047/1/WRAP_Theses_%20Guarniero%20_2017.pdf
- Local thesis: .localresources/papers/guarniero-2017-iapf-thesis.pdf and .txt.
  Extracted text lines 2230–2330 support the fitting account; lines 2650–2725
  support the experiment identity. Preserve source hashes in the run manifest.
- Accepted manuscript:
  .localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf
  and .txt, Section 5.1.
- Author publication listing: https://awllee.github.io/publications.html,
  item 23 links the article and arXiv; no iAPF source-code link was displayed.

ResearchAssistant is not exposed among this session's tools. Primary repositories,
web access, local PDF storage and pdftotext were used. No author was contacted.
