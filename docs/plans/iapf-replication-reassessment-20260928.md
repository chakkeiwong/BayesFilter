# Why the iAPF / MDPF replication has stalled

> Source update: the Bristol publisher PDF has now been downloaded and verified.
> Local copy: [.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf).
> The cover identifies it as the version of record; Section 5.1 says
> "a regularized version" of Eq. (15). Previous statements that this PDF
> was unavailable describe the earlier retrieval attempts and are superseded.
> Cover and fitting passage inspected; full version reconciliation remains open.

Date: 28 September 2026. Bounded source, mathematics, code and saved-result reassessment; no new numerical experiment, GPU job, default change or external contact. Both scientific replications remain incomplete.

## Verdict

We made a substantive methodological mistake: progress toward reproducing the iAPF experiments became conditional on certifying our reconstructed nonlinear fitter to a tolerance the paper does not specify. That certification was useful for a stationary-solution claim, but did not answer whether finite local fitting reproduces the paper's filtering behavior. Consequential numerical choices also remain unrecovered.

The unregularized objective has a real degeneracy. The author's thesis acknowledges it and reports local BFGS as implicit regularization in the state-space examples. We have not established a flaw in the core twisted-filter likelihood identity or a failure of the original implementation. We do not have the original iAPF executable or its complete fitting recipe.

MDPF is different: one author-code training trial and untouched evaluation completed. Its remaining gaps concern comparisons, objectives, repeated trials and the eligible local implementation.

## Sources and the mathematical issue

The local accepted iAPF manuscript specifies backward construction in Algorithm 3, density least squares in Equation (15), a Gaussian-plus-positive-constant twist in Equation (16), and diagonal covariance in Section 5.1. The [thesis, Section 7.3, pp. 53–55](../../.localresources/papers/guarniero-2017-iapf-thesis.txt:2253) specifies R optim/BFGS and zero explicit penalty for the Part III examples. It explicitly discusses density/amplitude escape and local optimization as implicit regularization. Its explicit Tikhonov penalty belongs to Part IV diffusion examples.

The initializer, covariance coordinates, derivative method, scaling, stopping controls and floor formula remain unrecovered. The complete final publisher PDF remains unavailable: these source versions must not be silently conflated. See [source recovery](artifacts/iapf-thesis-reconstruction-20260927-01/source-recovery.md). Fresh web searches/open requests for sources, code and corrections failed with HTTP 502/503 during this reassessment. There is no fresh exhaustive search, citation-count assessment, or absence-of-errata conclusion.

On a fixed cloud, write \(p_i=\mathcal N(x_i;m,V)\) and \(y_i>0\) for the backward targets. Equation (15) minimizes

\[
L(m,V,\lambda)=\sum_i(p_i-\lambda y_i)^2.
\]

Differentiating in amplitude and substituting gives

\[
\lambda_*=\frac{p^Ty}{y^Ty},\qquad
L_*=\|p\|^2\left[1-\frac{(p^Ty)^2}{\|p\|^2\|y\|^2}\right].
\]

The bracket measures shape disagreement on this cloud. The density norm can approach zero as the mean escapes the finite cloud at fixed covariance, or covariance expands without bound. Amplitude then also tends to zero. The global infimum is zero even without a finite matching Gaussian; unless a finite proportional fit exists, that infimum is not attained. A generally well-posed global argmin interpretation is wrong. Local nontrivial solutions can still exist: the derivation does not show that every local optimization fails.

Consequently, driving the objective lower can damage the twist. Initializer, coordinates and finite stopping are scientifically consequential when local optimization supplies practical regularization. Profiling amplitude preserves the minimized objective but does not preserve a joint BFGS trajectory. Positive constant objective scaling preserves minimizers but can change trajectories and stopping. These are reconstruction hypotheses, not proof of original execution.

The exact filter identity has a different requirement. [Proposition 1](../../.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt:261) proves cancellation for bounded, continuous, positive twists, without requiring least-squares stationarity. Correct sampling, integrals and importance weights remain essential. A poor admissible twist may be inefficient without changing that exact likelihood identity. This does not certify arbitrary adaptive stopping, numerical implementations or every variance assumption. Source Algorithm 4 and the inspected R iapf_iterate endpoint use a fresh final filter after adaptation. The [exact-twist/Kalman control](artifacts/iapf-scale-aware-reconstruction-20260926-01/result.md) agrees to 1.42e-14 in log likelihood, supporting the tested mechanics, not complete correctness.

## Our mistakes and the executable evidence

1. **Essential source recovery came late.** The R reference header explicitly says it contains no copied public iAPF code. Its profiled amplitude, log-variance coordinates, initializer and floor are local choices. We recovered the author's BFGS specification on 27 September after other solver investigations. The partial iAPF code was a weaker starting point than original author code. MDPF does have genuine author code.
2. **A solver criterion controlled the filtering question.** The executed diagnostic requires relative gradient <=1e-6 and a curvature check. All four saved cases must pass before fresh sweeps; every subsequent time fit must pass too. This supports an approximate-stationary-solution claim, not a paper-specified filter criterion. It should not universally veto a separately planned finite-optimizer diagnostic. Earlier plans acknowledged trajectory differences; the remaining defect is the dependency imposed on downstream work.
3. **We repeatedly studied selected failures.** The recent 96 BFGS fits are variants on four time-98 cases: two datasets crossed with two locally chosen floors. They are not 96 independent paper replications. Targets inherit our preceding backward fits. These runs diagnose our reconstruction; they do not estimate the original method's failure rate.
4. **Completion language hid different problems.** The master dispatches implemented diagnostic phases, not a fully implemented replication suite. Checksum audits establish provenance, not scientific replication. Conversely, calling both replications incomplete obscures the completed MDPF trial. Its supervisor disappeared while its worker continued: a local engineering failure, not evidence against the method.

The frozen call chain is checked: under [attempt01/sources](artifacts/iapf-thesis-reconstruction-20260927-01/attempt01/sources/diagnose_iapf_thesis_reconstruction.R), lines 56–64 compute stationarity, 155–159 require all four cases, and 177–190 gate fresh sweeps. The neighboring reference_iapf_paper.R header establishes independent status; its iapf_iterate calls fitting and a fresh final filter. The master dispatches these frozen diagnostic phases. This is source inspection of the executed call chain plus saved executable evidence; no new parity test was run.

## Saved results rule out a simple tolerance-only explanation

In the joint-amplitude, log-quadratic-start, scaled-objective, default-BFGS arm, the two floor cases on seed 9262001 have relative gradients 6.54e-5 and 7.53e-5 and descriptive shape errors 7.83e-6 and 9.08e-6. They miss our stationary-solution threshold but motivate a distinct finite-filtering question. On seed 9262002, the same arm has gradients 9.99 and 15.41, shape errors 0.760 and 0.792, and substantial density collapse. Loosening the threshold would not repair those outcomes. These are selected-case diagnostics, not statistical performance comparisons; exact rows are in [fits.csv](artifacts/iapf-thesis-reconstruction-20260927-01/attempt01/fits.csv).

Conversely, all eight repaired box endpoints pass projected stationarity but collapse, reach nine or ten coordinate walls, and have worse shape than their simple starts. Stationarity is not sufficient for useful fitting. No saved candidate is retrospectively promoted. See the [completed result](artifacts/iapf-thesis-reconstruction-20260927-01/result.md).

These failures occurred in R/FP64 CPU diagnostics without TensorFlow, XLA, GPU kernels or a NumPy/Python numerical implementation. Compiler work cannot remove this objective degeneracy. Separate GPU/XLA issues may exist; they do not explain these saved failures.

The [MDPF trial](../../.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/source-trial-result-20260925.md) completed 137 pretraining and 621 full epochs, with 326,823 passing gradient checks, 8.589 training hours and an untouched 5,000-trajectory evaluation. This is one NLL-trained trial, not the MSE-refined comparison or full repeated study. Conditional heuristics, objective/checkpoint reconciliation, Appendix B, comparators, uncertainty and TF/TFP GPU parity remain open.

## Corrected next step

Time-bound further recovery of the original R script and missing fitting settings. If unavailable, explicitly label the next implementation a reconstruction and freeze its choices before execution. Exact numerical reproduction remains a source-recovery question; a successful reconstruction could support a separately scoped scientific replication.

Predeclare one small complete finite-local-BFGS pilot on fresh data under the original density objective. Do not silently add penalties or change the floor. Keep gradient/curvature diagnostics for solver claims. Keep finite valid covariance, valid sampling/weights and complete artifacts as validity requirements; record collapse and fitting damage. Evaluate the actual likelihood computation against exact Kalman and conditional bootstrap/fully adapted filter baselines, with predeclared replication counts, uncertainty, compute/attempt limits and stopping rules. Existing controls need not be rerun wholesale without a relevant change.

This is an ordering correction, not an executable experiment plan or a retroactive change to old criteria. Concrete settings, seeds, metrics, thresholds, budget and commands must be recorded and skeptically audited before the pilot. No new automatic launch is added. MDPF's saved-trial comparisons are independent of iAPF and should proceed under their own contract.

## Decision and inference record

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Correct replication strategy | Source and executed gate establish a mismatch with the filtering question | Preserve prior failures; no retrospective passes | Original finite recipe | Rewrite the pilot contract | Existing fits qualify |
| Retain objective-degeneracy finding | Derivation and thesis agree | Collapsed endpoints remain rejected as useful-fit evidence | How original choices avoided escape | Bounded source recovery and finite-fit pilot | Core theorem or original experiments are false |
| Separate MDPF progress | One source trial/evaluation completed | Comparative evidence missing; original worker exit unknown | Learned quality and objective identity | Conditional saved-trial comparisons | Full replication or superiority |

| Inference status | Assessment |
| --- | --- |
| Hard veto screen | Saved collapse rejects affected endpoints, not all iAPF implementations; MDPF comparative quality unevaluated |
| Statistically supported ranking | None from this reassessment |
| Descriptive-only differences | Selected shapes/gradients, runtimes and one MDPF trial's metrics |
| Default readiness | Not established; no numerical default changed |
| Next evidence | Fresh complete filtering pilot with oracle/comparator uncertainty; separately scoped larger replications |

Post-review alternative explanation: an unexposed code error or substantial mismatch with the original recipe may remain. Finite derivative and exact-twist controls cannot exclude all bugs. Recovering an original implementation that reproduces the published experiment would resolve much uncertainty; failure of a genuinely matched original implementation would strengthen criticism of the paper. We do not yet have that evidence.

## Provenance and budget

Checkout /home/chakwong/BayesFilter, branch surrogate-hmc, base commit 89065bc6354801cb368d5e163ba49fd9c4372d10. Relevant files are uncommitted; unrelated dirty work is preserved. Inspected local primary sources, frozen R call paths, selected CSV fields and existing result records; generated no new stochastic evidence. Campaign CPU accounting stays 209.097046/1800 seconds (1590.902954 remaining); previously recorded GPU balance stays about 36.567 hours. These ledger balances are not total elapsed human/agent effort or current hardware-usage measurements.
