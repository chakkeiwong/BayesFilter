# Modern HMC chapter: mathematical, source and document audit

Completed 7 October 2026. This note records the review of
[the LaTeX chapter](../chapters/ch26g_modern_hmc_methods.tex), its incorporation
in [the official monograph](../main.tex), and the requested MathDevMCP audit.
This is a document result, not a sampler benchmark or a change to defaults.

The chapter develops randomized-length HMC, ChEES/SNAPER, MALT/MEADS,
MAMS/LAPS, whitening and Pathfinder, parallel Gaussian filtering, conditional
particle methods, and posterior precision. Recommendations distinguish
marginal parameter inference from latent-path inference and the actual
MacroFinance and BGS target/force conventions. The standalone wrapper inputs
the same chapter as the monograph; there is no second tuning guide.

## Deliverables and scope

- [Standalone chapter PDF](../plans/artifacts/modern-hmc-monograph-2026-10-07/r1/chapter-build/modern_hmc_survey.pdf):
  39 pages including contents and bibliography.
- [Complete monograph PDF](../plans/artifacts/modern-hmc-monograph-2026-10-07/r1/book-build/main.pdf):
  the new material is Chapter 54, printed pages 574–605.
- [Chapter wrapper](../modern_hmc_survey.tex) and
  [bibliography](../references.bib).
- [Plan](../plans/bayesfilter-modern-hmc-monograph-chapter-2026-10-07.md) and
  [run manifest](../plans/artifacts/modern-hmc-monograph-2026-10-07/r1/run-manifest.json).

There are 13 sections, 113 labeled equation groups, three comparison/test
tables, an analytic resonance figure, and 26 cited works. An aligned group can
contain several numbered equations. This is a detailed targeted survey, not
an exhaustive systematic review of every MCMC development in the decade.

The full book was rebuilt from the shared working tree, which already
contained extensive unrelated work. The saved baseline confirms that this
task added the chapter input and a contents-number spacing correction to
main.tex, appended distinct bibliography entries, and left preamble.tex
unchanged. Other chapters were compiled but not mathematically audited here.

## MathDevMCP: actual results

The installed MathDevMCP CLI/API ran from the tfgpu Python environment with
CUDA devices intentionally hidden. No sampler or accelerator was initialized.

| Check | Result | Meaning |
| --- | --- | --- |
| Initial high-level document audit | Completed, selecting 30 labels | Partial coverage, not a chapter proof |
| High-level audit with a larger explicit label budget | Failed in report assembly with KeyError for evidence_refs | Reporting failure, preserved rather than represented as a pass |
| Final public-API audit of every label | 113/113 inspected; 95 unverified, 18 inconclusive | Complete label coverage, no automatic theorem certification |
| Focused scalar algebra and finite-dimensional reductions | 34/34 equivalent | Checks identities under recorded domains |
| False negative control, 1+1=3 | Mismatch | Checker rejects this incorrect identity |
| Source binding | Final audits and snapshot match the delivered chapter hash | Evidence refers to the revised source |

Final evidence:
[label-audits.json](../plans/artifacts/modern-hmc-monograph-2026-10-07/r1/mathdev/final-r3/label-audits.json)
and
[symbolic-obligations.json](../plans/artifacts/modern-hmc-monograph-2026-10-07/r1/mathdev/final-r3/symbolic-obligations.json).
The source snapshot and runnable scripts are retained beside the reports.
Earlier evidence was preserved.

The CLI's zero label limit did not mean unlimited in the installed version:
it became None, which the called function treated as its default of 30.
An explicit larger limit exposed the separate report-assembly bug. The
fallback calls MathDevMCP's public equation locator, index builder, and
audit_derivation_v2_for_label directly. MathDevMCP itself was not modified.

The final audit extracts 171 obligations: 131 request manual formalization,
27 report source-label limitations, 10 report missing assumptions, one a
parser limit, one an unavailable encoding/backend, and one a missing shape.
These are tool statuses, not 171 mathematical errors. For example, its parser
can split a product's lower limit at the equals sign and lose secondary
labels inside an align environment. Its source-label status therefore does
not contradict the separate successful location of all 113 literal labels.

The assumption and shape flags received explicit manual disposition:

| Flagged expressions | Review |
| --- | --- |
| SNAPER endpoint derivative; OU dynamics | Fixed SPD mass and smooth potential are declared in the foundation; the derivative holds start, center and principal direction fixed and concerns an exact smooth trajectory |
| Microcanonical force | Differentiability, dimension greater than one, sphere state and nonexplosive-flow qualification are stated; no general ergodicity is inferred |
| Sphere-refresh inner product | Both directions lie on the same sphere in R^d; density is relative to surface measure and includes its Gaussian normalization |
| Trace of Q S Q S | Q and S are the displayed 2-by-2 energy-error and covariance matrices |
| Particle normalizer inverse | Z_T is a positive finite scalar, not a matrix inverse |
| Batch-means outer product | The observable is explicitly R^p-valued; the result is p-by-p and the number of batches exceeds one |
| Lugsail remainder b^(-1) | b is a positive batch length, not an unspecified matrix |
| Inverse-normal rank transform | Phi is a scalar CDF with argument strictly between zero and one |

Focused checks cover sphere norm, projection dynamics, inverse projection
and Jacobian; OU symmetry; MALT local work; leapfrog determinant and
energy-error matrix; unadjusted Gaussian variance and bias/error conversion;
derivatives and restricted convexity; LAPS contraction optimization;
equipartition counterexamples; funnel cancellation; spectral convexity and
Gaussian correlations; Schur expansion; Kalman covariance differential;
GRAD products and weights; lugsail bias; and nested-chain conditional variance.
Matrix and probabilistic arguments still require the manual derivations.
Domains were recorded and reviewed, not formally discharged by a proof assistant.

## Mathematical and source review

The [source manifest](../../.localresources/hmc-methods-review-20261007/source-manifest.json)
records versions, inspected technical sections, hashes and pinned author
revisions. Four foundations were added: Neal's HMC chapter, original NUTS,
Flegal–Jones batch means, and Andrieu–Doucet–Holenstein particle MCMC.
Previously archived rank-diagnostic, fixed-width and calibration sources
remain available.

| Topic | Checked argument and scope |
| --- | --- |
| Transformed targets | Total score includes both maps and log-Jacobian terms; frozen coefficients do not remove differentiation through the map |
| Ordinary HMC | Energy and volume explain exact-flow invariance; a separate involution proof supplies MH correction for the stated measure |
| Position-only force | Reversible shears can be corrected against the actual endpoint potential without an exact-gradient assertion; score identities do not inherit this result |
| Length adaptation | Gaussian resonance separates mean and variance mixing; ChEES normalization matches the paper; the SNAPER objective is not an ESS guarantee |
| MALT/MEADS | OU ratios cancel refreshment work; deterministic local errors remain; folded adaptation conditions on complementary chains |
| MAMS | Sphere divergence yields stationarity; the exact force subflow has log-Jacobian minus (d-1) log D; acceptance work has the corresponding positive term |
| Unadjusted bias | Gaussian variance/error identity belongs to the specified refreshed, one-step stationary chain; Jensen's bound has a restricted convexity range |
| LAPS | Contraction uses a genuine distance and uniform assumptions; its practical statistic is not such a distance; mean, dependence and fourth-moment counterexamples are given |
| Geometry/Pathfinder | Mass convention, total Jacobian, residual funnel curvature, inverse-BFGS approximation and augmented importance ratio are distinguished |
| Parallel filtering | Convolution is associative; likelihood constants are retained; correlated noise and masks require equivalent factors |
| Particle methods | Extended target and selected-path marginal are derived; local GRAD weights retain the parent-dependent Gaussian factor |
| Precision/calibration | Moment and dependence conditions matter; nested replication cannot exclude common bias; SBC requires joint exchangeability and treatment of ties/dependence |

Repairs before delivery included explicit matrix/domain assumptions, the
sphere-refresh normalization, reconciliation of the ChEES factor of two,
and clarification that GRAD's conditional proposal differs from its
single-auxiliary marginal. Foundational citations identify relevant sections
and equations. The SBC statement explicitly requires joint exchangeability,
with randomized ties.

Three proceedings metadata fields had picked up arXiv identifiers from their
reference lists; those fields were corrected without changing source bytes.
A mistaken initial Neal retrieval remains honestly named as the Beskos
optimal-tuning paper and is excluded from the chapter's authorities.

The complete joint marginal weight algebra of Particle-mGRAD is not
reproduced. The chapter gives the invariant conditional-particle construction,
Gaussian proposal interpolation and complete local auxiliary correction; it
explicitly directs readers to the paper for marginalization across particles.
General ergodicity and optimal-scaling theorems are likewise not newly proved
for arbitrary economic models. These limits appear in the chapter itself.

LAPS paper/source differences, MAMS settings and benchmark counts, installed
SNAPER helper behavior, and JAX/FunMC versus TensorFlow availability are tied
to inspected snapshots. Those snapshots are not asserted to be the revisions
used for the published benchmarks.

## Builds and rendered review

Both documents use latexmk/pdfTeX and BibTeX, with separate build directories.
All citations and cross-references resolve and no duplicate-label warning
remains. A stale main.bbl was detected and regenerated against the current
bibliography before the final build.

All 39 standalone pages were rendered and inspected as page sheets, with
selected pages also inspected at larger scale. Book inspection included the
title, contents, chapter opening, representative derivations/tables, chapter
ending and bibliography. This found and repaired crowded contents numbers
such as 54.10.1, overly long running headings and a small prose overflow.
Rendering evidence is under rendered-final and rendered-delivery.

The full book retains layout warnings in unrelated chapters. Its rebuild
does not certify their mathematics or prose. Reader feedback on this chapter
is pending; compilation, symbolic checks and self-review cannot certify a
human voice.

## Decision and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Deliver chapter and book for reading | Source-grounded derivations and PDFs complete | No unrepaired material target/sign/Jacobian error identified within the review; references resolve | Automatic proof coverage limited; reader review pending | Read/review, then design a separately budgeted implementation study | Formal certification, global convergence or default readiness |
| Retain proposed investigation order | Applicability and source availability support a candidate sequence | Exact-score and target restrictions explicit | Consumer batch scaling and equal-accuracy runtime unmeasured | Profile state-space targets and compare valid frozen kernels | Local or universal superiority of SNAPER, MAMS or LAPS |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No remaining material documentation error identified within scope; no sampler run |
| Statistically supported ranking | None from this work |
| Descriptive-only differences | Quoted author gradient counts and ratios retain their settings and exclusions |
| Default-readiness | Not assessed; no numerical default changed |
| Next evidence needed | Repeated state-space comparisons with uncertainty, matched targets/maps and total costs |

The strongest alternative explanation for apparent promise is a mismatch
between benchmarks and consumers: a large latent-state target and a small
parameter target with costly QZ/UKF/Kalman work have different costs. A fair
local comparison could reverse the proposed investigation order. The weakest
part of the performance argument is absent consumer batch-scaling evidence.

This is a completed mathematical/source review with bounded symbolic
assistance. It is not a formal proof certificate or evidence that the
proposed samplers have already been implemented and validated.
