"""Document assembly only: preserve protected equations and replace stale prose."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEST = ROOT / 'docs/papers/ledh_younis_kdm_score'
base = (HERE / 'baseline/ledh_younis_kdm_score.tex').read_text()

def span(start, end):
    return base[base.index(start):base.index(end)]

def fragment(name):
    return (HERE / (name + '.tex')).read_text() + '\n\n'

def replace_once(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new, 1)

preamble = base[:base.index(r'\begin{abstract}')]
preamble = replace_once(preamble, r'\usepackage{array}',
    '\\usepackage{array}\n\\usepackage{microtype}\n\\usepackage{xurl}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\setlength{\\emergencystretch}{1em}')
preamble = replace_once(preamble, r'\date{10 September 2026}', r'\date{11 September 2026}')
preamble = replace_once(preamble,
    'Younis Kernel Mixtures and the Score of the\\\\\nLEDH--OT--GenUT Dual-Cap Trust-Region Filter',
    'Younis Kernel Mixtures and Model-Score Estimation\\\\\nFrom the LEDH--OT--GenUT Filter to Corrected Mixtures')

opening = fragment('opening')
opening = replace_once(opening, r'\end{abstract}',
    '\\end{abstract}\n\\clearpage\n\\tableofcontents\n\\clearpage')
atom = span(r'\section{The executed atom program}', r'\section{What the Younis construction supplies}')
atom = replace_once(atom, r'\section{The executed atom program}',
    '\\section{The executed atom program}\n\\label{sec:atom}')
atom = replace_once(atom,
    'The present Phase~4A\nand Phase~4B LGSSM fixtures have one parameter.  No result below establishes a\nmulti-parameter Phase~4B assembly call site.',
    'The Phase~4A and Phase~4B score-quality fixtures have one parameter.\nThe later adapter audit exercises coordinate and mixed directions for five\nmodels through the two-step endpoints. Those checks cover the derivative\ncomponents, but do not establish a scalable batch implementation of the full\nscore vector.')
atom = replace_once(atom,
    'The single-cloud analytical score authority inspected for this note is:',
    'The research snapshot exposes the following analytical function and\nits legacy row-mapped batch wrapper:')

younis = span(r'\section{What the Younis construction supplies}', r'\section{Why positive bandwidth is a target change}')
younis = replace_once(younis, r'\section{What the Younis construction supplies}',
    '\\section{What the Younis construction supplies}\n\\label{sec:younis}\n\n'
    'The mixture resampling and fixed-proposal importance gradient below follow\n'
    '\\citet[Section~4, equations (14)--(15)]{younis2023}. We distinguish that\n'
    'construction from its later integration with BayesFilter resets and scores.\n')
bandwidth = span(r'\section{Why positive bandwidth is a target change}', r'\section{Legitimate ways KDM can enter}')
oldprop = bandwidth[bandwidth.index(r'\begin{proposition}'):bandwidth.index(r'\end{proof}') + len(r'\end{proof}')]
newprop = r'''\begin{proposition}
Suppose the kernel weights are nonnegative and sum to one, and each kernel
is centered with finite covariance $B_i$. If $w_i>0$ and $B_i\ne0$ for at
least one component, the atom and kernel measures differ. Their parameter
derivatives therefore require separate justification; a positive-bandwidth
calculation is not, by its construction, the derivative of the atom program.
\end{proposition}

\begin{proof}
Take $\phi(z)=\|z\|^2$. Centering eliminates the cross term in
$\|x_i+u\|^2$, so the difference between the two expectations is
$\sum_i w_i\operatorname{tr}(B_i)>0$. Unequal parameterized expectations
can nevertheless have equal derivatives at particular points, or even differ
by a constant. Thus this measure argument proves a change of functional, not
a universal pointwise inequality of scores. Equality of the scores would need
an additional argument for the actual parameterized functional.
\end{proof}

A limiting theorem as bandwidth tends to zero would concern convergence;
it would not make two finite-bandwidth programs exactly equal.'''
bandwidth = replace_once(bandwidth, oldprop, newprop)

experiments = span(r'\subsection{Phase 4A:', r'\section{Degenerate DSGE support}')
experiments = experiments.replace('the chart density in Section~6.', 'the chart density in Section~\\ref{sec:support}.')
matrix = '\\subsection{The matrix Gaussian score comparison}\n\n' + span('The first score-quality campaign used a two-state model.', 'The comparator ladder is:')
matrix += r'''
The bootstrap entry differentiates the finite program along its realized,
fixed ancestor labels. It checks that program's derivative; it is not the
ancestor-averaged Fisher-score baseline developed in Section~\ref{sec:fisher}.
The canonical numerical controls were frozen campaign comparators and were
not independently optimized for each scope. The results therefore compare
the tested configurations. Since both $N$ and $T$ change between the two
scopes, the contrast does not isolate horizon dependence or particle-number
dependence. The reported subgroup intervals are pointwise, not simultaneous.

The raw-IWSG log-value shift relative to the atom program averaged
$-0.02449$ and $-0.02377$ in the two scopes, with mean squared shifts
$0.07284$ and $0.04023$. Small changes in a scalar do not imply small changes
in its derivative. These descriptive value differences therefore do not
rescue the failed score comparisons.

'''
proposal = span(r'\subsection{A KDM proposal with an exact model-target ratio}', r'\subsection{Phase 4A:')
proposal = proposal[proposal.index('\n\n')+2:]
proposal = replace_once(proposal,
    'The KDM proposal must be inserted before a continuous transition, or the route\nmust define a new measure and a new finite target explicitly.',
    'A density-corrected KDM proposal can instead sample a continuous predictive\nupdate, where the physical transition supplies the target density. Inserting\nit directly at an atomic reset requires a different measure and an explicit\nnew target; the predictive-density argument does not correct that reset.')

support = span(r'\section{Degenerate DSGE support}', r'\section{Bounded validation program}')
support = replace_once(support, r'\section{Degenerate DSGE support}',
    '\\section{Structural support and deferred extensions}\n\\label{sec:support}')

implementation = span(r'\section{Code boundary and next step}', r'\bibliographystyle{plainnat}')
implementation = replace_once(implementation, r'\section{Code boundary and next step}',
    '\\section{Implementation state and remaining gaps}\n\\label{sec:implementation}')
implementation = implementation.replace('Section~2, with', 'Section~\\ref{sec:atom}, with')
implementation = replace_once(implementation, 'The current canonical endpoints are the two functions named in',
    'The research snapshot and the main checkout must be distinguished. The\nreported numerical results belong to research commit \\texttt{804616e3}; the\nmain checkout inspected for this revision is \\texttt{5cc59cfa}. The analytical\nfunction and legacy batch wrapper are the two functions named in')
implementation = replace_once(implementation,
    'Further DSGE\nimplementation is deferred: the immediate task is to finish testing the\nexisting repository models.  This requires checking every parameter component\nof each model adapter, the complete filter that consumes it, and then\nmodel-score error under fresh model-specific calibration.',
    'Further DSGE\nimplementation is deferred. The five-adapter checks below establish a useful\nstarting point for current model support. Before those models support any\nnew score-quality conclusion, their initial laws and precise observation and\ntransition targets must be resolved and the complete consuming filter tested\nunder fresh model-specific calibration.')
implementation = replace_once(implementation,
    'The reduced registered fused batch lane is also\nexcluded: its current body does not execute the\nContract--E/\\GenUT{}/dual-cap reset.  No HMC promotion or canonical default\nchange is authorized.  A positive result at finite bandwidth would support a\nnew, explicitly named regularized program; it would not retroactively make its\nderivative the score of \\eqref{eq:atom-program}.',
    r'''The batch implementation has changed since the earlier limitation was
written. In the inspected main checkout,
\path{ledh_canonical_batch_fused_tf.py} now calls the single-cloud analytical
executor and forwards Contract--E, higher-moment, pairwise-cap, and
coordinate-cap settings. It uses nested \texttt{tf.map\_fn} calls over rows
and parameter directions. This is a row-mapped scalar implementation; despite
the filename and its description, it does not satisfy the repository's
requirement for a batch-native NeuTra training target. The older
\path{ledh_canonical_batch_tf.py} wrapper also maps rows and does not itself
forward a Contract--E reset selection, so its default invocation remains the
diagnostic reset-free slice.

The shared main-checkout score executor now reaches
\texttt{batched\_higher\_moment\_shape\_jvp} in
\path{ledh_unified_correction_tf.py} through a batch-axis adapter. This source
change means the old experiment hashes cannot certify the current shared
numerics. Source wiring was inspected for this revision; numerical parity of
that later dependency chain was not rerun. The integrated observation and
raw-IWSG source files match the research snapshot, but their shared dependency
has changed. Implementation reuse is therefore distinct from renewed
numerical certification.

The new marginal proposal, ancestor-averaged Fisher recursion, and selective
component gradient in Sections~\ref{sec:model-mixtures}--\ref{sec:hybrid}
are derived proposals. They are not integrated, tested runtime candidates in
either snapshot. The existing anchored \texttt{MODEL-IS} kernel is a
one-step diagnostic and supplies only part of that future implementation.
A useful model-score estimator could differ from the derivative of
\eqref{eq:atom-program}; its value would be established against the underlying
model score and through its own implementation checks. The present results
establish neither HMC readiness nor a change to the canonical default.''')

implementation = replace_once(implementation,
    'now implements \\eqref{eq:affine-chart-state}',
    'implements \\eqref{eq:affine-chart-state}')
phase_two_start = implementation.index('The bounded Phase~1 test suite')
phase_two_end = implementation.index('\\end{itemize}', phase_two_start) + len('\\end{itemize}')
implementation = (implementation[:phase_two_start] + '\\begin{samepage}\n'
    + implementation[phase_two_start:phase_two_end] + '\n\\end{samepage}'
    + implementation[phase_two_end:])

checks = span(r'\section{Bounded validation program}', 'The first score-quality campaign used a two-state model.')
checks = replace_once(checks, r'\section{Bounded validation program}',
    r'\subsection{Checks on the existing finite-program endpoints}')
checks = replace_once(checks,
    'The first diagnostic is a TensorFlow reconstruction of the Younis mixture\nalgebra with an analytical tangent.',
    'The existing TensorFlow diagnostics reconstruct the Younis mixture\nalgebra with analytical tangents.')
checks = replace_once(checks,
    'The\nclaim-bearing call chain must resolve from the actual canonical endpoint to\nthis evaluator; a standalone function is not evidence of integration.',
    'Any integrated claim requires the consuming endpoint to call the evaluator\nwith compatible shapes, dtypes, and derivatives. The proposed model-score\nroutes may have their own endpoints; a standalone density helper does not\nestablish integration with any complete filter.')
checks = replace_once(checks, 'The following gates are required:',
    'The finite-program checks have the following distinct purposes:')
checks = checks.replace('Any route claiming linear cost has an explicit no-$N^2$ memory and\n        work check.',
    'Any route claiming linear cost needs a scaling check with its stated\n        assumptions; an all-pairs $N^2$ operation is incompatible with that\n        claim.')

contents = (preamble + opening + atom + younis + bandwidth + fragment('history')
    + experiments + matrix + fragment('pivot') + fragment('model-mixtures')
    + proposal + fragment('fisher') + fragment('hybrid-controls')
    + fragment('alternatives') + support + implementation + checks
    + fragment('next-study') + fragment('provenance')
    + '\\clearpage\n\\bibliographystyle{plainnat}\n\\bibliography{ledh_younis_kdm_score}\n\n\\end{document}\n')
DEST.mkdir(parents=True, exist_ok=True)
(DEST / 'ledh_younis_kdm_score.tex').write_text(contents)
(DEST / 'ledh_younis_kdm_score.bib').write_text(
    (HERE / 'baseline/ledh_younis_kdm_score.bib').read_text() + '\n'
    + (HERE / 'additions.bib').read_text())
print(f'Assembled {len(contents.splitlines())} lines in {DEST}')
