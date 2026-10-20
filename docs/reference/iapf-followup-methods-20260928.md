# iAPF follow-up methods: sources and implications

28 September 2026. Focused literature audit following the owner's question about related and later methods. This is a reference review, not a numerical experiment or a change to either replication target.

**Controlled SMC is the most useful immediate reference.** It supplies a different, explicit backward fitting procedure and author code. Annealed controlled SMC and a January 2026 forward-only method explain why fitting can fail even when the twisted-filter equations are correct. SIXO supplies a further learned-twist alternative, with a substantially different training procedure.

## Sources checked

| Method | Technical material inspected | Practical relevance | Limits |
|---|---|---|---|
| Heng, Bishop, Deligiannidis and Doucet, *Controlled Sequential Monte Carlo*, Annals of Statistics 48(5), 2904–2929 (2020), [DOI](https://doi.org/10.1214/19-AOS1914), [arXiv](https://arxiv.org/abs/1708.08396) | arXiv v3 (2019), §§4.2–4.3, Algorithms 2–3, regression/error discussion in §5; selected author MATLAB route | Fits negative log twists. A quadratic basis gives a linear least-squares coefficient problem; later passes refine the policy multiplicatively. This avoids the particular nonlinear Gaussian-plus-floor fitting problem in our reconstruction. | Different objective, family and update from the original iAPF reconstruction. It cannot close original-iAPF replication by substitution. Stability results have assumptions. |
| Fulop, Heng and Li, *Efficient Likelihood-based Estimation via Annealing for Dynamic Structural Macrofinance Models*, [arXiv:2201.01094](https://arxiv.org/abs/2201.01094) (2022 posting) | v1, manuscript dated December 2021; §§3.3–3.5, Algorithm 3, theorem assumptions and Appendix D.5 | Anneals the likelihood and progressively refines the policy. Addresses training particles lying in poor regions of the smoother; Appendix D.5 describes damping to maintain positive-definite Gaussian proposal precision. | This is a changed algorithm, not an iAPF numerical-port repair. Latest revision was not inspected; see version note below. |
| Procope-Mamert, Chopin, Delattre and Kon Kam King, *A forward-only scheme for online learning of proposal distributions in particle filters*, [arXiv:2601.16089](https://arxiv.org/abs/2601.16089) (January 2026) | v1, §§4.2 and 5.1–5.2, Appendices B, D and E | Directly discusses backward fitting instability and poor cloud coverage. Uses forward learning, weighted log-quadratic regression, precision constraints and tempering of degenerate training weights. | Preprint. Its reported comparisons concern controlled SMC, not our exact original-iAPF reconstruction. The forward scheme is not uniformly superior; Appendix E includes an easier setting favorable to controlled SMC. |
| Lawson, Raventós, Warrington and Linderman, *SIXO: Smoothing Inference with Twisted Objectives*, [NeurIPS 2022 paper](https://papers.neurips.cc/paper_files/paper/2022/file/fddc79681b2df2734c01444f9bc2a17e-Paper-Conference.pdf) | §§3.2–3.3, Proposition 1 and relevant training appendix material; repository README and selected training references | Learns twists by density-ratio classification and alternates with model/proposal learning. Useful as another way to learn future-observation information. | Much larger change than repairing our finite fitter. The supplied implementation uses JAX. Neither that backend nor its learning procedure is an admitted BayesFilter runtime by this review. Its gradient treatment does not establish KDM/IWSG correctness. |

## What the author code establishes

The [controlled-SMC repository](https://github.com/jeremyhengjm/controlledSMC) is cached at `.localresources/code/heng-controlled-smc`, commit `2696f6d611aae172d10102cb7bfbfd314bb6b88f`. In `StateSpaceModels/NeuroScience/cSMC_Resample.m`, lines 184–193 construct the backward regression and solve normal equations. Lines 196–208 project quadratic coefficients when the inspected scalar or diagonal approximation would violate the precision constraint; lines 218–232 accumulate coefficient updates and form the proposal covariance. The caller at lines 249–260 initializes a bootstrap policy, fits, then runs the controlled filter.

This provides concrete algorithmic choices, not a universally safe numerical template. Normal equations can amplify conditioning problems. The local `0.001` precision margin is an author implementation choice, not a justified BayesFilter default. The inspected full-quadratic branch does not use the same componentwise projection; do not generalize the scalar/diagonal safeguard to every route. A new TensorFlow implementation would need a stable solve, a derived precision condition, a declared policy for rank deficiency, and executable checks of its actual consumer call chain.

The [SIXO repository](https://github.com/lindermanlab/sixo) is cached at `.localresources/code/lawson-sixo`, commit `51e0ac255995f9cb9201eb49dd4eca81a3f3440e`. Its README identifies JAX dependencies. Neither downloaded repository was executed; numerical equivalence and call-chain parity are **not checked**.

## Code availability follow-up, 28 September 2026

Two of the four papers have confirmed author repositories containing end-to-end experiment workflows: controlled SMC and SIXO. This is a source-availability finding, not a successful execution or reproduction verdict.

Controlled SMC contains MATLAB demos, the particle-filter and backward-fitting implementations, systematic resampling, likelihood-estimation scripts, model definitions, and example data. The inspected neuroscience entry point calls cSMC_Resample, whose controller calls ApproxDP and RunSMC; the required Data_NeuroScience.mat is present. The Gaussian example has both exact LQG and learned-policy demos. The code uses MATLAB functions such as mvnrnd and x2fx; runtime/toolbox compatibility is not checked. No explicit reuse license was found in the inspected repository. It is the closest algorithmic reference; permission for direct copying or translation needs clarification.

SIXO contains the SMC engine, twist objectives, models, training and evaluation scripts, a bundled forex dataset, sweep launchers, and unit/training tests. The diffusion entry point reaches train_diffusion, the chosen twist training step, and snax.train_alternating; its SIXO loss reaches bounds.sixo and smc.smc. It carries an MIT license, with a notice-preservation condition in LICENSE. This is the clearer direct-code-reuse option. Its JAX, Equinox, Optax and Git-hosted snax dependencies still need environment testing, and a BayesFilter runtime would require a TensorFlow port under current policy. Tests were inspected, not executed; their presence is not a passing test result.

The source call chains above were read, but executable wiring/parity is **not checked**. A further bounded search of author pages, paper text and repository search results did not confirm public author code for the annealed or forward-only papers. This records an unresolved availability question, not proof of absence.

Recommendation: use controlled SMC as the immediate method/reference benchmark, clarifying its license before incorporating source. SIXO is available for permissively licensed reuse but changes the fitting/training problem more substantially. Neither repository supplies the missing original-iAPF recipe or completes the separate KDM/IWSG replication.

## How this changes the diagnosis

Our R4d evidence concerns finite optimizer sensitivity: very small target perturbations can move the fitted coefficients, and the paper-horizon R-versus-TensorFlow comparison still fails its frozen criterion. Later literature does not explain that particular discrepancy by itself.

It adds a second mechanism worth distinguishing: a correct regression implementation may fit the wrong regions because its current particles poorly cover the smoothing distribution, and the errors can propagate backward. The annealed and forward-only papers address this mechanism explicitly. A third possibility is inadequate approximation capacity: a single log-quadratic twist can remain poor for nonlinear or multimodal future likelihoods even with a stable fit and adequate cloud coverage. This last statement is a structural limitation of the approximation family, not a finding established for our current dataset.

The publications therefore support testing fitting geometry, coverage and approximation capacity separately. They neither establish that the original iAPF article is wrong nor establish that our implementation is correct. Exact Kalman comparisons remain especially useful in the linear-Gaussian case because they separate numerical implementation from nonlinear approximation error.

## Recommended next comparison

Keep original iAPF replication and a controlled-SMC comparison as separately named tasks. Start with the author-defined log-quadratic controlled-SMC fit on the same linear-Gaussian model, using the exact Kalman likelihood and the existing bootstrap and fully adapted filters. Establish the fit and proposal-normalizer algebra on fixed clouds before comparing complete algorithms on fresh streams. Evaluate likelihood error, likelihood-estimator variability, and total adaptation-plus-filtering cost; do not promote a method using regression loss alone.

Before execution, write the paired downstream evidence contract already required by the checkpoint: calibration/claim separation, matched data and budgets, seeds and uncertainty, finite/positive-definite proposal vetoes, and regime-specific checks. Distinguish likelihood estimates from log-likelihood estimates when assessing bias. Conditioning, coverage, coefficient differences and regression residuals explain failures; they do not independently establish a correct likelihood estimator.

Then test annealing only if poor coverage or backward error propagation is supported. Use the forward-only scheme as a later discriminating alternative; use SIXO only if the question requires learned, richer twists. This priority is an engineering recommendation based on proximity and available code, not a statistically supported ranking of the methods. It creates no claim that any of them has repaired iAPF or MDPF.

## Search and version limits

The search followed direct citations and method references around iAPF, controlled SMC, annealed controlled SMC, forward-only proposal learning and SIXO. It is a bounded repair-oriented review, not a complete survey or citation-impact ranking. Citation counts were not collected. No claim is made that all errata or corrections have been ruled out.

Junye Li's [author page](https://junye-li.github.io/) lists a January 2026 revision titled *Annealed Controlled Sequential Monte Carlo with Applications to Dynamic Economic Models*, with a revise-and-resubmit status at JASA when checked. The linked newer manuscript could not be retrieved; technical conclusions here concern the locally saved 2022 arXiv v1 only. Author code for the annealed and forward-only papers was not located in this bounded search; that does not establish that no code exists.

Older related methods identified through the checked papers include Whiteley and Lee's *Twisted particle filters* (2014), Richard and Zhang's efficient importance sampling (2007), and Scharth and Kohn's particle efficient importance sampling (2016). They deserve follow-up if the fitting-family question remains open; they were not fully technically re-audited in this review. Broader control, Schrödinger-bridge and neural-twist literature was not searched comprehensively.

All four inspected PDFs and their extracted text are stored under `.localresources/papers/`. The accompanying [structured source ledger](iapf-followup-methods-20260928.json) records exact files, hashes, versions, citation relationships, claims and omission risks. No TensorFlow/GPU command or numerical experiment was run in this review; the campaign compute balance is unchanged.

