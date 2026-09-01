# Style contract for the human-facing ZLB survey (LaTeX)

You are rewriting one part of a technical survey. The source material is
`/home/ubuntu/workspace/BayesFilter/docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.md`
(markdown). Your job: keep ALL the mathematics and its correctness, but
rewrite the prose so it teaches. The current text reads like an internal
governance document; the rewrite must read like a good survey article.

## Audience and voice

- Reader: a graduate student or researcher in econometrics / computational
  statistics. Comfortable with Bayes, Kalman filters, and basic MCMC. Knows
  NOTHING about any codebase, repository, project, or team.
- Voice: "we". Explanatory. Every section opens with 1--3 sentences of
  motivation (what problem this solves, why the naive approach fails), then
  develops the material, then closes by connecting forward.
- Derive, don't decree. Where the source asserts a result tersely, add the
  one or two connecting steps a reader needs. Never remove a derivation.
- Do not invent new mathematical claims. If the source hedges (e.g. a paper
  was cited from metadata only), keep the scholarly honesty but phrase it
  naturally: "we cite X for orientation; we were unable to obtain the full
  text" — one clause, not a ledger row.

## Hard prohibitions

- NO file paths, file names, module names, fixture names, directory names.
- NO repository or project names (BayesFilter, MacroFinance, dsge_hmc, dz5,
  BGS, Dynare file paths). Dynare/OccBin as *software* may be named as
  published tools.
- NO inspection dates ("as coded on 19 August 2026") — instead "in the
  working example below" or "in the implementation we studied".
- NO governance vocabulary: "claim-bearing", "evidence contract",
  "non-claims", "veto", "promotion gate", "campaign", "fixture", "target_id",
  "declared target" (say "the model actually being estimated"), "owner",
  "migration debt", "scope match".
- NO checklists of project obligations. Convert genuinely useful conditions
  into prose ("this construction is valid provided (i)..., (ii)...").
- Do not address the reader as an implementer of a specific system. Practical
  advice is fine; internal work-package numbering is not.

## Mathematical conventions

- Equations: use \begin{equation}\label{eq:K} ... \end{equation} where K is
  the OLD tag from the markdown source (e.g. the source's \tag{60b} becomes
  \label{eq:60b}). DROP the \tag{} itself — LaTeX numbers automatically.
  Multi-line displays: \begin{aligned}...\end{aligned} inside equation.
- Every textual reference "(K)" to an equation becomes \eqref{eq:K}.
  NEVER hardcode an equation number in prose.
- Every reference "Section N" or "Sec. N.M" becomes \ref{sec:...} using the
  global label map below. NEVER hardcode a section number.
- Inline math: $...$. Text: ASCII only; use ---, ``quotes''. No unicode
  dashes/quotes/math symbols in your output.
- \(\vartheta\) stays \vartheta; keep the source's notation everywhere.

## Global section label map (old survey numbering -> LaTeX label)

1 -> sec:intro          2 -> sec:model         2.1 -> sec:geometries
3 -> sec:linear         3.1 -> sec:linear-tworegime
3.2 -> sec:linear-kalman  3.3 -> sec:linear-limits
4 -> sec:particle       4.1 -> sec:pf-generic  4.2 -> sec:copf
4.3 -> sec:ukf          4.3.1 -> sec:ukf-branchloss
4.3.2 -> sec:ukf-truncmoments  4.3.3 -> sec:ukf-censored
4.3.4 -> sec:ukf-branchwise    4.3.5 -> sec:ukf-imm
4.3.6 -> sec:ukf-proposal
5 -> sec:hmc            5.1 -> sec:kink-validity
6 -> sec:event          6.1 -> sec:event-calc  6.2 -> sec:event-zlb
7 -> sec:dhmc           7.1 -> sec:dhmc-embed  7.2 -> sec:dhmc-laplace
8 -> sec:mixed          9 -> sec:relaxations
10 -> sec:pmcmc         10.1 -> sec:pm-target  10.2 -> sec:pmhmc
10.3 -> sec:joint-hmc   10.4 -> sec:pgas       10.5 -> sec:dpf
11 -> sec:synthesis     12 -> sec:targets
12.1 -> sec:target-det  12.2 -> sec:target-stoch 12.3 -> sec:target-multi
13 -> sec:shadow        13.1 -> sec:shadow-model 13.2 -> sec:shadow-lit
13.3 -> sec:shadow-ladder 13.4 -> sec:shadow-filter
13.5 -> sec:shadow-softplus 13.6 -> sec:shadow-ident
13.7 -> sec:shadow-routes
14 -> sec:nk            14.1 -> sec:nk-solvers 14.2 -> sec:nk-mult
14.3 -> sec:nk-sunspot  14.4 -> sec:nk-baselines 14.5 -> sec:nk-design
15 -> sec:solver        16 -> sec:guidance     17 -> sec:conclusion

## Application chapters (special rules)

- The old Section 13 model becomes a self-contained CASE STUDY: "a
  two-country shadow-rate term-structure model". Specify the model from
  scratch in the text (8-dimensional Gaussian VAR(1) state: two dynamic
  Nelson--Siegel factor triples plus two currency-basis factors; bounded
  instantaneous forward curves; yields as maturity averages via a 40-node
  Gauss--Legendre rule; one covered-interest-parity FX row; diagonal Gaussian
  measurement noise; softplus bound with per-country sharpness constants
  alpha_d = 1.5e-3, alpha_f = 1.0e-3, bounds 0 and -0.005, decays 0.65 and
  0.45). Present these constants as "the working example uses...". Explain
  WHY each modeling piece is there (why bound forwards rather than yields,
  why quadrature, why softplus) before analyzing it.
- The old Section 14 becomes CASE STUDY II: "a small New Keynesian model
  with an occasionally binding policy rate". All the LCP/multiplicity/
  sunspot material stays, fully derived. The internal package-status
  material (old Sec. 14.6, the BGS anchors, the contract checklist as a
  checklist) is DROPPED; convert the genuinely general content of old 14.5
  into a short prose subsection "designing an estimation exercise"
  (label sec:nk-design) about what must be specified before estimation is
  well-posed (bound, expectations, terminal condition, solution operator,
  uniqueness domain, selection law, treatment of nonexistence).

## Table style

Use booktabs inside tabularx:
\begin{table}[htbp]\small
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}XX@{}}
\toprule ... \bottomrule
\end{tabularx}
\caption{...}\label{tab:...}
\end{table}
Give every table a real caption. Long identifiers like the three model
variants: use short math-style names (e.g. $\mathcal{S}_1$ smooth softplus
model, $\mathcal{C}_1$ finite hard-max model, $\mathcal{C}_0$
continuous-maturity hard model) instead of snake_case code identifiers, and
define them at first use.

## Citations

Keep author-year citations as plain text ("Guerrieri and Iacoviello (2015)")
exactly as in the source; the assembled document carries a formatted
reference list. Do not add or remove citations.

## Output format

Write a single .tex FRAGMENT to the output path you were given:
- starts with \section{...}\label{sec:...} (or your assigned sections),
- contains only body LaTeX (no preamble, no \begin{document}),
- compiles under: article + amsmath + amssymb + booktabs + tabularx +
  hyperref (labels/refs to OTHER sections/equations outside your fragment
  are fine — they resolve at assembly).
Your final message: one line confirming the path written and any source
statement you deliberately dropped or weakened (so the assembler can check).
