# Modern HMC literature chapter and mathematical audit

The user requests a detailed, self-contained survey in the official LaTeX
monograph, full derivations, good citations, a MathDevMCP audit, a chapter PDF
and the rebuilt complete monograph. The chapter will use the reviewed papers
and pinned source snapshots in `.localresources/hmc-methods-review-20261007/`.
It will be a new input in the existing HMC part, with one numerical notation
and a chapter-only wrapper consuming exactly that input. The existing book and
tuning reference retain their roles; this is not a new operational tuning guide.

Scientific intent: explain which mathematical mechanisms could lower the cost
of accurate posterior estimation for structural state-space models, what they
preserve, what they assume, and which improvements are plausible for the actual
TensorFlow consumers. Comparator: current corrected fixed-metric/transport HMC
and correctly configured NUTS on the same declared posterior. No local speed
ranking, backend migration, sampler implementation or default change follows.

The chapter will derive the transformed target and total score; Hamiltonian
invariance and general deterministic MH; Gaussian resonance and ChEES/SNAPER
objectives; OU refreshment and the MALT path correction; microcanonical sphere
dynamics, their stationary density, exact force subflow and Jacobian; LAPS's
equipartition identity, explicit counterexamples, and bias scheduling; L-BFGS
initialization, Gaussian message composition and the conditional-particle
construction; and Monte Carlo variance, lugsail and short-chain diagnostics.
Source-specific assumptions and the scope of author benchmarks accompany the
derivations. Specialist alternatives receive enough mathematics to identify
their requirements without claiming to reproduce every published ergodicity
theorem.

Evidence contract: mathematical identities require derivations in the chapter
and source anchors for imported results. MathDevMCP will run its document
audit and focused symbolic obligations, with non-symbolic assumptions reviewed
manually. A parser result, algebraic identity check or successful PDF build
does not establish a general convergence theorem. Material sign, measure,
Jacobian, target or citation errors require repair before delivery. Diagnostic
examples cannot establish local speedup. Numerical symbolic substitutions are
deterministic checks, not statistical experiments.

Defaults/assumptions: use the existing book class, preamble and plainnat
bibliography (inherited document conventions); use dimension d>1 for the
microcanonical formula (derived requirement); use smooth positive targets and
vanishing integration-by-parts boundary terms where stated (mathematical
assumptions, not facts about DSGE boundaries). Paper thresholds, chain counts
and acceptance targets are attributed author settings, never new defaults.
Model examples use analytic parameters chosen to expose a mechanism, not to
support a performance comparison. No GPU use. Bound local audit and compilation
to an initial one-hour CPU/wall working allowance; stop any individual hung
symbolic job and report unsupported obligations rather than expanding compute.

Skeptical audit before editing: a survey can mislead by conflating exact
invariance with finite-time convergence, gradients per chain with total work,
the UKF target with an exact likelihood, a position-only force with its score,
latent-path sampling with marginalized parameter inference, or paper algorithms
with changed current source. The outlined derivations and separate consumer
discussion address each risk. The existing main.tex and references.bib contain
unrelated edits; preserve their current contents and add only the chapter input
and new distinct bibliography entries. No theorem will be inferred solely from
MathDevMCP's parser. This plan passes the skeptical audit.

Execution: save a bounded source baseline; write the chapter and references;
audit and repair mathematics; compile the standalone chapter and complete book
in a fresh artifact directory; inspect the rendered PDFs, citations, references
and layout; record actual audit status and remaining limits. Use the installed
local MathDevMCP CLI because this session does not expose its MCP tools.
Author sources are read only. No existing HMC campaign is changed.

Output root: `docs/plans/artifacts/modern-hmc-monograph-2026-10-07/r1/`.
Primary source: `docs/chapters/ch26g_modern_hmc_methods.tex`.
The final note will link both PDFs and all audit evidence. Reader feedback is
pending; mathematical and build checks do not certify human prose quality.

At the first complete draft, about 54 minutes of elapsed writing time had
been used. Continue routine symbolic/document checks and PDF assembly with a
further bounded allowance of two hours, at most two concurrent local jobs,
and no individual symbolic job exceeding 15 minutes. This is document work,
not an added sampler campaign. No GPU or scientific default change is involved.
The first source scan found no control characters, unbalanced environments,
duplicate labels or missing bibliography keys.

Continuation at 06:38 UTC: the chapter and book have already compiled. Finish
the four foundational source checks, audit the final revised source, rebuild
and inspect the PDFs, and record the audit disposition. Allow up to one further
hour of routine local document work, with no sampler or GPU run. Calendar
elapsed time includes session inactivity and is not a CPU-use measurement.
The skeptical check remains satisfied: no performance ranking or default
change is being inferred from symbolic checks or document compilation.

Completed on 7 October 2026. The chapter has 13 sections, 113 labeled equation
groups and 26 cited works. The standalone PDF has 39 pages; the full monograph
has 691 pages, with the new Chapter 54 on printed pages 574–605. Both build with
all citations and cross-references resolved. Rendered inspection repaired a
contents-spacing defect in main.tex in addition to the originally planned
chapter input; this is a layout adjustment, with no change to other chapter
text or the shared preamble.

MathDevMCP's final source-bound API audit covered all 113 labels but reported
95 unverified and 18 inconclusive. The 34 focused symbolic identities all
returned equivalent, and a deliberately false control returned mismatch.
The high-level report's partial coverage and subsequent report-assembly error
are preserved. Manual mathematical/source review and repairs are recorded in
docs/reviews/bayesfilter-modern-hmc-monograph-audit-2026-10-07.md.
This is a completed document delivery with explicit proof-tool limitations;
no new sampler performance or convergence claim follows. Reader feedback
remains pending. No sampler campaign, GPU workload, commit or push was run.
