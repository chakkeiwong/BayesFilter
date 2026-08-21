# Case study: making an internal technical survey readable by humans

Audience: the humanvoice agent building tooling for human-facing
documentation. This note records what went wrong, in order, while turning an
internal AI-authored survey (the ZLB / discontinuous-HMC survey in
`docs/surveys/zlb_discontinuous_hmc/`) into a document a human could read,
what fixed each problem, and which fixes are automatable versus editorial.
Every problem below was hit for real; nothing is hypothetical. Artifacts to
inspect alongside this note: the final LaTeX manuscript and PDF in that
directory, the style contract `latex_manuscript_style_contract.md`, the
addendum trail at the end of `hostile_review.md` (addenda I--V), and the
governing policy `~/workspace/claudecodex/policies/
global-scientific-coding-agent-policy.md` ("Reader-Facing Scientific
Prose"), plus `humanizer-ai-writing-patterns.md` installed next to it.

## The starting point

The source was a 2,300-line internal Markdown survey: mathematically strong,
audited, with claim ledgers, and unreadable by an outside human. The owner's
successive complaints, each after reading real output, define the problem
sequence:

1. "The generated LaTeX is unreadable. The math is not math." (naive
   format conversion)
2. "This reads like an AI governance document, not a survey. It does not
   teach, does not derive, does not explain. Statements like 'derived from
   the code inspected on 19 August 2026 (two_currency_double_zlb_math.py,
   ...)' mean nothing to the reader." (register failure)
3. "Still not quite publishable. Did you follow the writing policy?"
   (mechanical AI tells: dashes, template monotony)
4. "Still over-defensive. Making a nonclaim instead of explaining. Reads
   like a lawyer protecting his client instead of a note trying to teach."
   (defensive register)
5. "Lacks a set of models to test algorithm correctness." (missing
   content a real reader needs, hidden until the prose stopped being the
   problem)

The order matters: each layer of badness masked the next. Nobody can notice
defensive register while the page is full of file paths.

## Problem 1: format conversion is not document production

What happened: `pandoc markdown -> latex` produced double section numbering
(the source had literal "## 13. Title" headings and `--number-sections` was
also on), a TOC labeled 0.1/0.2, `longtable`+minipage table wreckage, and
overflowing `snake_case` identifiers. Two rounds of flag-fiddling improved
it; the owner still rejected it, correctly, because the *content* was
internal.

Lesson for tooling: a converter can only ever expose the source's register.
If the source is an internal artifact, perfect conversion yields a perfectly
typeset internal artifact. Budget for a rewrite decision up front: ask
"who is the reader?" before "what is the output format?". The useful
mechanical pieces we kept: strip literal heading numbers and let LaTeX
number (cross-references must then use `\ref`, never hardcoded numbers);
convert pipe tables to booktabs `tabularx` with inline math preserved and
long code names made breakable; keep display math flowing through untouched
(`tex_math_single_backslash`).

## Problem 2: internal register (the biggest one)

Symptoms in the source: absolute file paths and module names as evidence
("derived from `two_currency_double_zlb_math.py`"), inspection dates as
scoping ("as coded on 2026-08-19"), governance vocabulary ("claim-bearing",
"evidence contract", "veto", "promotion gate", "fixture", "target_id"),
checklists of project obligations, and work-package numbering. All of it is
*correct* for an internal audit trail and *meaningless* to a reader.

Fix: a written style contract (preserved as
`latex_manuscript_style_contract.md`) given verbatim to every rewriting
agent. Its load-bearing rules:

- Name the audience explicitly (grad student in econometrics; knows Bayes
  and Kalman filters; knows NOTHING about any codebase) and forbid the
  internal vocabulary by list, with replacements ("declared target" -> "the
  model actually being estimated").
- Applications become self-contained case studies: the model is specified
  from scratch in the text with its constants presented as "the working
  example uses ...". Code identifiers become math names (mf_c1_k40_hardmax
  -> $\mathcal{C}_1$, defined at first use) with the mapping recorded in the
  internal README so the ledgers stay linked.
- Teach before analyzing: every section opens with why the problem exists;
  derivations carry connecting steps; results are cited to papers, project
  deductions are derived in the text's own notation.
- Keep two documents. The internal Markdown remains the record that audit
  ledgers anchor to; the LaTeX is the human-facing primary. Do not try to
  make one document serve both masters -- that is what produced the original
  hybrid.

Mechanics that made a multi-agent rewrite safe: a global label map (old
section numbers -> LaTeX labels, old equation tags -> `eq:` labels) fixed in
the contract before fan-out, so six agents writing in parallel produced
fragments whose cross-references all resolve at assembly. Per-agent drop
reports ("I deliberately dropped/weakened X") let the assembler audit
against the source instead of diffing blind.

## Problem 3: mechanical AI tells survive good intentions

After the rewrite, the owner still balked. Diagnostics (run them, do not
guess) found: 133 em dashes in 46 pages; 20 of 55 sections opening with
"The ..."; every section following the identical
motivation->math->forward-link shape; five "we now turn to" transitions.
Root cause of the monotony: *our own style contract mandated the uniform
shape*. A template that guarantees a floor also imposes a ceiling.

Fixes and tooling implications:

- Measure first. Cheap greps produce actionable numbers: dash count,
  section-opener first-word distribution, roadmap-phrase count, stock-word
  list hits. These made the owner's "still not publishable" concrete and
  checkable after repair.
- The policy hierarchy matters. We installed the humanizer pattern list
  (Wikipedia "Signs of AI writing", from github.com/blader/humanizer) into
  `claudecodex/policies/` with a precedence header: it is a DIAGNOSTIC,
  subordinate to the scientific-prose policy. Order of authority:
  correctness > source fidelity > domain meaning > reader comprehension >
  template regularity. Without that header, a pattern list becomes a new
  template and you trade one uniformity for another (e.g. its flat em-dash
  ban must not strip dashes from math or ranges).
- One editor, not six. Voice unification and rhythm variation is a
  whole-document pass by a single agent; parallel writers reintroduce seams.
- Repairs are per-sentence, not per-pattern: each dash became a comma,
  colon, parenthesis, or a restructured sentence by local judgment; a
  mechanical substitution would have produced new damage.

## Problem 4: defensive register ("lawyer, not teacher")

The subtlest layer, and invisible until the others were gone. Symptoms:
negative definitions ("X cannot by itself do Z") with no mechanism; bare
prohibitions ("must not be silently promoted"); moralizing tics ("honest"
x7, "silently" x4); defensive section headings ("Why this is not a
nonlinear ZLB filter"); apology sentences answering objections nobody
raised. The internal audit culture (nonclaims, boundaries, vetoes) leaks
into prose as liability-avoidance.

The editorial test that worked, from the owner's framing: **does this
sentence give the reader a reason, or only protect the author?** Repairs by
shape:

- Negative definition -> positive mechanism. "A fixed-branch HMC update
  cannot by itself cross the support boundary" became: within one region it
  explores efficiently; crossing requires moving the continuous state,
  because a proposal that changes only the regime label has zero target
  density under eq. (5a). The prohibition becomes a consequence the reader
  can re-derive.
- Prohibition -> consequence. Explain what goes wrong; let the command
  become advice. Theorem validity conditions are NOT lawyering -- keep them
  as stated conditions.
- Moralizing -> substantive property. "honest about its Monte Carlo
  variance" -> "unbiased, with quantified Monte Carlo variance".
- Keep the limitations a reasonable reader needs (where they would
  otherwise draw a stronger wrong inference). The policy's phrasing:
  positive result first, then the qualification and why it matters. Roughly
  half the ~65 flagged sentences were rewritten; the rest were substantive
  mathematics and kept.

## Problem 5: readable but incomplete

Once prose stopped being the obstacle, the owner immediately found a
content gap (no named set of standard verification models). Expect this:
readability audits surface content audits. The fix was a "ladder of
standard test models" organized by where the independent answer comes from
(closed form / exact enumeration / published reference implementation /
model-agnostic harness), which is the property that makes a test a test.
Tooling hint: "what would a reader try to DO with this document, and is
that action supported?" is a reviewable question a doc tool can ask.

## Invariants that held through every pass

These are the guardrails that let aggressive prose editing proceed safely,
and they are all scriptable:

1. Protected baseline. Before each pass, snapshot; after, verify equation
   bodies byte-identical (109/109 every time), and label/eqref/section/
   reference-item counts unchanged. "Readability improvement is not
   established by shortening" -- every removal must be explained.
2. Compile gate: N-pass pdflatex, zero errors, zero unresolved references,
   overfull-box budget.
3. Rendered-page inspection, not just text greps: convert pages to images
   and LOOK at the title/TOC, a math-dense page, and every table. pdftotext
   lies about layout.
4. Leak sweep on the RENDERED text (not the source) for paths, project
   names, dates, and banned vocabulary.
5. Provenance: every pass appended a dated addendum to the review record
   stating what changed, what was measured before/after, and what was
   deliberately dropped.
6. Self-review cannot certify a human voice (policy clause, and true). The
   human read is the acceptance gate; everything above only clears the
   ground so human feedback lands on substance. Correspondingly: targeted
   repairs against a specific human reaction converged far faster than
   another global pass -- optimize the loop, not the pass.

## What I would build, if I were you

- A register linter with a project-configurable banned-vocabulary map and
  leak patterns (paths, dates, identifiers), run on rendered text.
- A rhythm profiler: dash density, opener distribution, section-shape
  clustering, transition-phrase counts; report numbers, never auto-fix.
- A defensive-register finder: the pattern list from Problem 4 plus the
  "reason or protection?" prompt for an editing model; output candidate
  sentences with their surrounding mechanism so the editor can convert
  prohibition to consequence.
- A baseline differ specialized for LaTeX/Markdown: equation-body identity,
  label/citation census, with an explain-every-removal report.
- The two-document convention as a first-class concept: internal record +
  human manuscript, with an explicit identifier map between them.
- A policy-precedence mechanism, so pattern lists (humanizer and whatever
  you add) are always subordinate diagnostics, never templates.

One number for scale: source 2,326 lines of Markdown; final manuscript 49
PDF pages, 55 sections, 109 equations, 55 references; five editorial passes
end to end, each triggered by one round of human feedback.
