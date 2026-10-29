# Active execution specification — repair tranche 1

Authorized by the owner's instruction to refresh, review and execute the master.
This specification implements R0/R1/R2 of the whole-repair plan. R3 proceeds
independently. Source identity and empirical replication remain separate.

## R0 iAPF reconciliation (28 September 2026 Hong Kong)

Inspected the publisher version's Section 2, Propositions 1–3, Sections 3.2–3.4,
Algorithms 3–5, Section 5.1, the Section 5.2 relative-variance experiment,
Section 6 and Appendix A, alongside the accepted version and thesis Section 7.3.
The publisher copy is DOI 10.1080/01621459.2016.1222291, JASA 112(520), 1636–1647.
The publisher PDF SHA-256 is
4a38ecd6c84bf1e591ca8ed05e443dacbc777c1f781a8225e24140fd5b9f1f1e.
The final paper adds 'a regularized version' before Eq. (15), but supplies no
complete optimization recipe or formula for the positive constant c.
The thesis explicitly uses R optim/BFGS and no explicit penalty in Part III;
it explains the escaping-density degeneracy and local optimization as implicit
regularization. These statements support a reconstruction, not recovery of the
final article's original executable or every regularization choice.
Public source searches have not recovered that executable. Web search in this
turn failed; no exhaustive absence claim or erratum claim is made.

Recovered: Gaussian-plus-positive-constant twist, diagonal covariance fit,
Eq. (15) density least squares with free amplitude, backward targets,
ESS resampling kappa=.5, k=5, tau=.5, N0=1000, k+1 likelihood history,
fresh final filtering pass, d=5/10/20/40/80, T=100, alpha=.42.
Unknown: initializer, exact parameter coordinates, scaling, finite optimizer
controls, final-version regularizer, floor formula, original data and seeds.
Paper Algorithm 4's early doubling indexes are undefined until the first full
window; use l>=k, record this reconstruction choice.

The independent reference's call chain was read:
iapf_iterate -> injected backward fitter -> iapf_backward_log_target ->
iapf_log_integral; iterate -> iapf_apf -> iapf_proposal/draw_proposal ->
positive normalized mixture -> weight cancellation -> fresh final pass.
Runtime wiring and identities are tested in controls, not certified by prose.

## Frozen finite-fit recipes (diagnostic reconstructions)

Both recipes use joint Eq. (15) Gaussian-density least squares, summing squared
residuals. Mean increments are scaled by initial standard deviations; variance
uses log coordinates; amplitude uses a linear coordinate around its initial
profiled positive value. Derivatives are analytic. The log-quadratic fit is an
initializer only, with its existing weighted-moment fallback explicitly logged.
It is not the fitted objective or a substitute after optimizer failure.
R optim BFGS: maxit=100, reltol=sqrt(machine epsilon), no explicit penalty,
no box bounds. Invalid evaluations fail; no clipped covariance or hidden ridge.
Optimizer code 1 (finite iteration limit) is retained and reported.

Recipe A: literal unscaled density residuals (raw SUM Eq. 15).
Recipe B: divide density residuals by the initial maximum density, fixed once.
They answer the unresolved finite-optimizer scaling choice. Neither is described
as recovered author settings. Freeze A before B; no selection by pilot rankings.
An objective-equivalent scaling is not a finite-trajectory equivalence.

Floor hypothesis shared by both:
c = delta/(1-delta) N(m;m,Q+V), delta=1/N.
This gives original-transition mixture probability >=1/N at every parent,
because the convolved Gaussian is maximized at m; at the peak it corresponds to
one expected original-transition draw per particle population. This is a
scale-aware safety rationale, NOT the author's recovered c. It changes the twist
and proposal, not the likelihood target when all terms are included.
Controls compare delta=0,1/N^2,1/N analytically for normalization/cancellation
and record the altered mixture weight. No floor value is promoted from this
pilot. Zero explicit penalty follows the thesis Part III, but its collapse
risk remains open; full density-norm and shape diagnostics are preserved.

## Evidence contract and budget

Question: can these declared finite fits complete Algorithm 4 and produce valid
terminal likelihood estimates, without our extra stationarity gate?
Comparator ladder: exact Kalman likelihood; constant-twist BPF N=10000;
fully adapted APF N=5000; one-step observation twist N=1000; exact-twist oracle.
Source allocations are intentionally different. Runtime includes all adaptation.
Two fresh fixed data seeds 9281001/9281002, d=5,T=100. Four independently seeded
full adaptation runs per dataset/recipe, at most 16. Dedicated adaptation/final
R L'Ecuyer substreams; comparator streams are separate. Seeds and histories saved.
At most 20 adaptation passes and 16000 particles are safety/work limits:
hitting either means INCOMPLETE, never paper convergence. A per-run time cap
may similarly leave unfinished runs; never condition summaries on successes.

Controls: <=300 CPU seconds; pilot plus local repairs <=900 CPU seconds.
Whole CPU remainder initially 1590.902954 seconds, reconciled in opening-budget.
First repair tranche <=12 numerical attempts; no GPU phase until its own
source/consumer contract is ready. No new paid compute or package mutation.

Validity vetoes: failed identity/gradient/wiring control, nonfinite state,
invalid covariance/amplitude, corrupt or missing required output. Stationarity,
shape, density escape, resampling, runtime and likelihood ratios are reported
diagnostics; they cannot silently become stopping vetoes. A collapsed fit blocks
a useful-fit promotion claim, not observation of its otherwise valid filter.
Candidate failure triggers the next documented diagnostic; independent R3 continues.
Pilot completion is the primary engineering criterion. Statistical ranking,
unbiasedness certification, paper-table reproduction, default readiness, GPU
port and HMC readiness are not established by this small diagnostic.

Conditional descriptions use fixed early 1–33 / middle 34–66 / late 67–100
prefix groups and low/high Kalman innovation split at chi-square(d) median.
These are falsification/explanatory views, not recipe selection targets.
Uncertainty at four replications is insufficient for ranking; retain all ratios
including failures. Proper confirmatory contrasts remain R5 work.

## Skeptical pre-run review

Checked wrong baseline risk: exact likelihood is primary, source comparator
allocations are explicit. Raw/scaled finite BFGS are hypotheses. No optimizer
stationarity or pilot mean is a promotion criterion. Prior four failure cases
are not fresh data. CPU R is explicitly an independent reference, not the TF
default. Caps, timeout and attempt limits prevent unbounded execution. Floor
and no-penalty choices have source/rationale, failure mechanisms and checks.
This diagnostic can discriminate solver-first obstruction from complete-filter
failure; it cannot close scientific replication. Verdict: PASS TO EXECUTE
bounded controls and pilot, after the master harness regression passes.
MDPF Appendix B, conditional baseline evaluation and GPU-port phases still need
their exact commands/specifications; their absence is reported, not hidden.
