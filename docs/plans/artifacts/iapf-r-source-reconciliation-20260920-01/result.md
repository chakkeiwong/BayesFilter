# Source reconciliation and equation-15 obstruction

Completed 2026-09-21 (Hong Kong; the deterministic run began 2026-09-20 UTC).
The written unrestricted fitting minimization is ill-posed for some valid
fitting clouds. The current R implementation computes that objective correctly;
changing optimizer tolerances cannot repair its global minimization problem.
The author's actual local optimization procedure remains unknown. These are
different conclusions: the first is proved below, and the second limits exact
replication. Neither establishes that the author's experiments failed.

## Checked sources

The [author's publication list](https://awllee.github.io/publications.html)
links the [paper](https://doi.org/10.1080/01621459.2016.1222291) and
[arXiv manuscript](https://arxiv.org/abs/1511.06286). Both arXiv source archives
were recovered and preserved. Version 2, `sources/arxiv-extracted/iapf_arxiv.tex`
lines 655–686, and version 1, `sources/arxiv-v1-extracted/iapf_arxiv.tex`
lines 639–660, give the same least-squares objective and specify a positive
floor without giving its formula. Neither supplies optimizer settings or
parameter bounds. Algorithm 4 specifies the fresh final estimate; its early
particle-doubling convention remains incomplete for indices before the first
full history window. The archives contain manuscript and figure sources, not
an implementation of the filter.

The Warwick deposit supplies an accepted manuscript. Publisher full-text and
supplement requests returned HTTP 403. Crossref, OpenAlex, the authors' pages
and bounded GitHub searches did not recover verified original-author code.
This is a bounded retrieval result, not proof that no such code exists. The
final published technical text was not recovered in this audit; equivalence
to the accepted/arXiv version is not independently certified. Forward-citation
metadata were inspected for corrections and implementation leads, but the
citing papers' technical methods were not audited. The literature coverage is
therefore explicitly narrow. Request details and file hashes are preserved in
`source-ledger.json`.

The previously pinned public R file
`.localresources/code/sempreteamo-iapf-a8811439/iapf.R`, lines 217–235,
profiles the amplitude and minimizes a differently scaled residual. Its
original-author provenance remains unverified. It cannot settle what optimizer
the paper used. Our optional `log_quadratic` fit is also a different objective.

## Why the unrestricted objective can fail

At fixed fitting locations, let the positive target values be the vector
\(y\), and let \(p(m,V)\) contain Gaussian density values. Equation 15 is
\[
 L(m,V,\lambda)=\|p(m,V)-\lambda y\|^2.
\]
For fixed Gaussian parameters, differentiation with respect to the amplitude
gives \(\lambda_*=(p^T y)/(y^T y)>0\). Substitution gives
\[
 L_*(m,V)=p^T p-\frac{(p^T y)^2}{y^T y}.
\]
The R reference computes the mean squared residual, \(L_*/N\), with a
declared fixed density scale. The diagnostic sets that scale to one. The
fixed factor \(1/N\) changes the reported loss units, not the minimizers or
the following obstruction; the numbers below use those R mean-loss units.
Fix the mean and a positive definite \(V_0\), and put \(V=s^2 V_0\).
At any finite set of locations, all Gaussian densities tend to zero as
\(s\to\infty\), so \(L_*\to0\), whatever the target shape. This follows
from density scaling, without floating-point underflow. If no Gaussian in the
fitting family is proportional to the target values, every finite fit has
positive loss and the infimum zero has no finite minimizer.

The diagnostic constructs exactly such a case. The target is a correlated
bivariate Gaussian with unit marginal variances and correlation 0.6, sampled
on a 5-by-5 grid. Its log-density contrast at the four points
\((0,0),(1,0),(0,1),(1,1)\) is \(0.6/(1-0.6^2)=0.9375\).
Every diagonal Gaussian has zero mixed contrast, even with a free mean and
overall amplitude. Consequently no finite diagonal fit is proportional to
these target values.

On the tested scale path, the actual implementation and independent direct
arithmetic agree. Absolute loss falls from 0.000605314 to
\(3.78092\times10^{-26}\), a factor of \(6.24621\times10^{-23}\).
Meanwhile the scale-free residual rises from 0.120831 to 0.611389, matching
its derived diffuse-density limit. All quantities remain finite, and an
exactly representable diagonal control has loss \(4.14\times10^{-35}\).
See `attempt01-objective/results/scale-path.csv` and `checks.csv`.

This proves an obstruction to interpreting the written global argmin as a
well-defined fitting prescription on every valid cloud. A local optimizer
with a particular initialization or constraints may still produce useful
fits. The diagnostic does not recover those missing author choices, prove
failure of every fit, or contradict the importance-correction identities.

## Decision and continuation

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Written unrestricted minimization has a counterexample | Analytic derivation and executed objective agree | No numerical-invalidity veto | Behavior of the unpublished numerical procedure | Preserve equation-15 route and explicit gap | Not evidence that the authors' runs failed |
| Exact numerical paper replication is blocked | Optimizer, constraints, floor and early controller are not fully specified | Source identity remains unresolved | Whether further author material supplies these choices | Record missing inputs; do independent authorized work | Do not relabel a different objective as equation 15 |
| Continue optional R reference validation | The frozen alternative already has d40/d80 evidence | No continuation veto from this deterministic test | Its same-setting d5/d10/d20 checks remain incomplete | Execute the frozen small-dimension completion plan | No original-paper, TensorFlow, LEDH, KDM or HMC promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Counterexample vetoes a general finite-global-minimizer claim |
| Statistically supported ranking | N/A: deterministic identity test |
| Descriptive-only differences | Optimizer performance and authors' numerical results are not tested here |
| Default readiness | No defaults changed |
| Next evidence needed | Verified author implementation/settings for exact numerical replication |

Red-team conclusion: an unstated bounded parameter domain or local-solver
convention could explain successful published calculations. It would not
invalidate the counterexample to the unrestricted formula, but it would change
the executable procedure to be replicated. No such convention was invented.

The one deterministic launch took 0.214906683 worker seconds, with exit 0 and
all checks passing. Its manifest preserves command, source snapshots, commit,
CPU-only environment and hashes. No numerical core was edited. The request
cap was reached: 19 archived HTTP requests and one failed web-search batch
(three queries, upstream HTTP 502). The failed search was repaired by bounded
direct primary-source retrieval; no author was contacted.

The remaining 479.519126226 worker seconds transfer to the next plan, together
with the previously unused 853.973386368 seconds from positive-floor
validation. Total available: 1333.492512594 summed worker seconds. Old
manifests remain unchanged and these allowances must not be spent twice.
