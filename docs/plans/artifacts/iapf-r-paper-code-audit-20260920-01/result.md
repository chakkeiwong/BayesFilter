# R iAPF paper/code audit and fitting diagnosis

2026-09-20. The proposal and importance correction match the inspected paper
identities. The demonstrated remaining failure is Gaussian fitting: a tiny
relative residual on the training particles can accompany a poor approximation
away from those particles. Increasing the optimizer allowance is insufficient.
The reviewed [audit plan](../../iapf-r-paper-code-audit-2026-09-20.md) was executed.

## Paper and actual call chain

Source: Guarniero, Johansen and Lee, *The iterated auxiliary particle filter*,
accepted manuscript corresponding to JASA (2017), DOI10.1080/01621459.2016.1222291.
Local PDF: `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`,
SHA256 `41d751f9698c17550477563ed2bde3ca89ca9918d868ab496b44cb6f7304c975`.
The technical construction, Algorithms3--5, Section5.1/5.2, Proposition3 and its
appendix argument were inspected. The public `sempreteamo/iapf` R comparator was
also inspected; its original-author provenance remains unverified.

The executable consumer trace is Python driver → captured R study runner →
`iapf_iterate` → `iapf_apf` and `iapf_fit_backward` → proposal/integral/fit.
Tests exercise this trace, including a deliberately failed fit and reconstruction
of its backward target from saved controller state.

| Paper requirement | Checked implementation and verdict |
| --- | --- |
| Equations5--6, Proposition1 | Gaussian-plus-constant proposal, initial normalizer and incremental potentials are correct for the stated positive twists |
| Algorithm3 | Backward target is the current observation likelihood times the integral of the already fitted next twist; failure inputs now preserve that dependence |
| Algorithm4 | Zero-based stopping, particle adaptation and fresh final estimate checked; earliest doubling remains a documented source ambiguity |
| Algorithm5 | Both ESS resampling branches retain the appropriate weights and likelihood normalizers |
| Equation15 | Literal default retains the written objective; acceptance of nonzero solver codes and objective underflow repaired |
| Equation16 / Section5.1 | Diagonal Gaussian plus positive floor; exact floor and solver settings are reconstructions, not recovered author settings |
| Section5.2 | Published linear model and baseline particle counts; new data and reduced repetitions, not full replication |

Three engineering gaps are closed. The strict fitter now rejects nonconvergence
and objective underflow. Failure RDS files now preserve next-time twist, model,
time, controller history and settings. The experiment driver now executes its
captured source tree; a test changes the live source and verifies that captured
behavior executes. Previous manifests are not retroactively upgraded.

## Why the fitting objective gives misleading reassurance

For fitted density values p and positive backward targets y, the optional
relative loss is `R=1-(p'y)^2/((p'p)(y'y))`. Define
`q_i=y_i^2/sum(y^2)` and `r_i=p_i/y_i`. Then exactly

`R = 1 - E_q[r]^2/E_q[r^2] = Var_q(r)/E_q[r^2]`.

Thus agreement is weighted by squared target values. In the two captured d20
failures, `1/sum(q_i^2)` is only1.272 and1.409 out of1000. A diagonal Gaussian
has40 shape parameters. In the limiting single-point example R is zero for
every positive fitted value and cannot identify the Gaussian shape. The
effective-size calculation alone does not prove a rank theorem; it motivates
independent validation. That validation now shows actual off-cloud failure.

Both saved failures replayed with exactly identical inputs and fitted parameters.
For each case, six independent APF clouds and six exact-smoothing clouds of1000
points were evaluated. The fitted function's error rises substantially away
from the training cloud. These sample ranges are descriptive only.

The linear model also permits an exact check. With the next fitted twist frozen,
the actual backward target `y(x)=g_t(x) integral f(x,z) psi_(t+1)(z) dz` is a
two-Gaussian mixture up to scale. Under the exact smoothing Gaussian nu, compute

`R_nu = 1 - (integral p*y dnu)^2 / ((integral p^2 dnu)(integral y^2 dnu))`.

All terms have closed Gaussian integrals, checked against independent 1D
quadrature. The comparator is the frozen Algorithm3 target, not the optimal
future likelihood. The smoothing law supplies the validation measure only.

| Fixed d20 case | Failed optimized fit | Original initializer | Diagonal target-moment Gaussian | Constant |
| --- | ---: | ---: | ---: | ---: |
| L-BFGS-B, time92 | .472332 | .175331 | .018480 | .968190 |
| nlminb, time68 | .261061 | .250284 | .017046 | .947228 |

Both optimizations worsened this continuous error while reducing their training
residuals to about7.3e-8 and3.0e-10. The diagonal family can represent a much
better approximation in these cases, so family restriction alone is not the
explanation. Adding the recorded floor changes these fixed-function errors by
less than5e-13; that does not rule out its effect on recursive targets or earlier
particle clouds. The moment fits are model-specific diagnostic oracles and are
not substituted into the runtime fitter.

Proposition3 connects twist error under smoothing distributions to asymptotic
variance for its stated particle system. It does not certify this adaptive
implementation from a training residual, nor supply a quantitative variance
prediction from the diagnostic above. The evidence rejects the current fits;
it does not reject iAPF as an importance-sampling construction.

## Decision and remaining uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Engineering repairs complete | Call-chain and failure replay tests pass | No observed identity violation | Unspecified author settings | Preserve captured-source execution | Original author reproduction |
| Current d20 relative fits rejected | Exact off-cloud errors poor despite tiny training residual | Solver failures and worse-than-initializer error | Generality across datasets and fitting times | Test an explicit log-quadratic objective | Failure of all iAPF or KDM implementations |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Nonconverged d20 fits remain rejected |
| Statistically supported ranking | None for full filtering in this audit; the integral comparisons are deterministic |
| Descriptive-only differences | Fresh-cloud error ranges and target effective sizes |
| Default-readiness | Not established; equation15 default retained |
| Next evidence needed | Fresh complete filtering runs for the separately labeled repair |

Red-team review: a small training loss could still be useful on other datasets;
these two failures do not prove universal failure. Nevertheless, the exact
integrals eliminate Monte Carlo variation as an explanation of the displayed
off-cloud errors. An improved fit that passes independent likelihood checks
would overturn rejection of the fitting approach. The source ambiguities and
absence of original-author code remain the weakest basis for paper replication.

Execution evidence: `summary.json`, `mechanics01-exact-risk/manifest.json`,
and `../iapf-r-reference-gap-repair-20260920-01/attempt10-code-audit/manifest.json`.
The replay worker used32.132448 seconds; exact integration used.273188 seconds.
At audit close, total repair workers used1323.329427/1550 seconds and mechanics
were charged50/120 seconds. All work is CPU-only independent R reference work,
with CUDA hidden and one BLAS/OpenMP thread. The owner requested continuation;
the [log-fit repair plan](../../iapf-r-log-fit-repair-2026-09-20.md) records the next
bounded experiment and its separate method identity.
