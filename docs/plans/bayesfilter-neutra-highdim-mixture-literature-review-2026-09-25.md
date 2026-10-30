# Kernel mixtures, importance-weight collapse, and modern learned samplers

Literature review, 2026-09-25. This answers whether the kernel-mixture proposal
discussed with the owner remains sensible as dimension increases, and what
developments since approximately 2011 change the recommendation. No experiment,
training, sampling, implementation change, or default promotion is performed.

The review separates three questions: representing a density, constructing
reliable importance weights, and discovering materially important regions.
Success at one does not establish success at the other two. The current q20/T30
UKF approximate posterior has **four sampled parameters**; its state dimension
and neural-network parameter count are different quantities. See the
[current training result](bayesfilter-neutra-training-evaluation-results-2026-09-24.md).

Skeptical audit: an unrestricted KDE is not a demonstrated scalable solution;
a modern paper's favorable ELBO is not coverage evidence; a structured
high-dimensional example is not evidence for arbitrary high-dimensional
multimodality. Comparisons below describe published mechanisms and their
boundaries, not a new ranking. Selected primary papers extend through 2025;
this is not an exhaustive review of all 2026 work. Numerical choices in cited
experiments are reported descriptions, not proposed BayesFilter defaults.

## Why the original proposal needs qualification

For a conventional second-order KDE under standard smoothness and integrability
conditions, the leading mean integrated squared error has the form

\[
\operatorname{MISE}(h)\approx C_1h^4+\frac{C_2}{Nh^d}.
\]

Balancing the terms gives \(h\propto N^{-1/(d+4)}\) and
\(\operatorname{MISE}\propto N^{-4/(d+4)}\). These are derived rates, not a
runtime estimate for our target. Delyon and Portier's Section 4.1 explicitly
distinguishes this density-estimation curse from fixed-dimensional central
limit results for estimated integrals. A root-N asymptotic rate for an integral
does not bound the finite-sample cost uniformly in dimension.

There is also a weight-collapse problem independent of KDE. Let normalized
densities p and r satisfy the importance-sampling support condition, with
\(W=p/r\). Then

\[
\mathbb E_r W=1,\qquad
\mathbb E_rW^2=\int p^2/r=1+\chi^2(p\Vert r).
\]

For product densities \(p=\prod_jp_j\), \(r=\prod_jr_j\), Fubini gives

\[
1+\chi^2(p\Vert r)=\prod_{j=1}^d[1+\chi^2(p_j\Vert r_j)].
\]

Thus even fixed modest per-coordinate mismatch can multiply into exponential
weight variability. When the second moment is finite, the empirical importance
ESS divided by N converges to the reciprocal of this product as N grows. This
identity does not make observed ESS a detector of unsampled modes.
Chatterjee and Diaconis give a complementary sample-complexity result:
when log density ratios concentrate under the target, the relevant importance
sample-size scale is approximately exp(KL(p||r)), with explicit fluctuation
terms and distinctions between ordinary and self-normalized estimators
(Theorems 1.1 and 1.2). It is not a universal exact sample-count formula.

A fitted finite mixture has parametric structure and is not identical to KDE
with a kernel at every observation. It need not follow the unrestricted KDE
rate, but full covariance estimation, the number of components needed to
represent curved geometry, and importance-weight reliability remain issues.
Enumerating every mode can itself be impractical when the mode count grows
exponentially with dimension.

## Relevant developments

| Work | Mechanism inspected | What it addresses | Boundary for this project |
| --- | --- | --- | --- |
| Hoogerheide, Opschoor and van Dijk, MitISEM (2012; local 2011 working paper) | Sections 2 and 5 and Appendix A: importance-weighted EM for Student-t mixtures; component addition; marginal/conditional block extension | A direct continuation of the proposed mixture plus forward-KL idea, including financial/econometric applications | Requires reliable weighted exploration; conditional structure is an assumption, and a fitted mixture does not certify unknown-mode coverage |
| Delyon and Portier (2021); Korba and Portier (2022) | Safe KDE mixture, heavy-tailed component, subsampling, and regularized weights linked to mirror descent; theorems and bandwidth conditions | Stabilizes adaptation and supplies convergence analysis for modern kernel AIS | Density fitting retains dimension-dependent rates. Power weights target an intermediate distribution during adaptation; they must not silently become final posterior weights |
| Elvira et al., GRAMIS (2023) | Section 3/Table 2: gradient and curvature adaptation of proposals, repulsion among centers, full-mixture importance weights; Section 4 examples | Moves mixture centers using target geometry and discourages every component converging to one region | Repulsion needs calibration. The reported 50-dimensional banana example has strong structure; the paper itself notes that structure when interpreting dimensional scaling |
| Arenz et al., GMMVI (2022/2023) | Natural-gradient component and weight updates, trust regions, component adaptation; Sections 2–4 and Appendix A | An evaluable mixture density with more efficient fitting than an unrestricted KDE; examples up to a few hundred dimensions | Optimizes reverse KL/ELBO, not the proposed forward KL. The paper warns that ELBO-tuned settings can give worse distributional discrepancy |
| AFT (2021), CRAFT (Matthews et al., 2022) | CRAFT Eq. 5, Algorithms 1–3 and supplement: repeated SMC passes, local flow transports, weights, resampling, and MCMC refinement | Breaks a difficult direct transport/weighting problem into annealed transitions; CRAFT replenishes training particles across passes | Staged reverse-KL objectives are not whole-target forward KL. Particle depletion, insufficient mutation, and training costs remain; published successes do not establish our coverage |
| Midgley et al., FAB (ICLR 2023) | Section 3/Algorithm 1, Appendices A–C and E: alpha=2 divergence, AIS bootstrapping, corrected prioritized replay | Directly trains an evaluable flow to reduce importance-weight second moments without requiring pre-existing target samples | Objective differs from forward KL. Dimensional scaling analysis assumes factorization and perfect intermediate transitions. AIS exploration and flow capacity remain material |
| Chen et al., SCLD (ICLR 2025) | Section 2, Algorithms 1–4, Proposition 2.3 and appendices: learned diffusion transport, sequential importance weights, resampling, subtrajectory log-variance loss, optional MCMC refinements | A newer learned sampler combining diffusion flexibility and SMC correction, with explicit attention to training-gradient variance | Produces a weighted path sampler, not automatically a tractable terminal mixture/flow density. Discretization and target-evaluation cost remain; benchmark coverage evidence is target-specific |

FAB is particularly relevant to the current learned-transport question. For a
normalized target p its objective, up to constants, is

\[
L_2(\phi)=\int p(x)^2/q_\phi(x)\,dx=1+\chi^2(p\Vert q_\phi).
\]

An unknown normalizer only multiplies the objective by a parameter-independent
constant. FAB uses AIS directed at a density proportional to
\(p^2/q_\phi\), emphasizes regions where the current flow underrepresents
the target, and applies the paper's stop-gradient and replay corrections.
The practical self-normalized training estimator is not an exactly unbiased
gradient of the original integral at finite sample size. Replacing its target
with p gives a different method/objective and must be labeled accordingly.

Appendix C's favorable dimension-scaling argument uses factorized p and q,
separate coordinate parameters, and perfect independent intermediate MCMC
transitions. It studies a normalized gradient variance, not a general
polynomial-time guarantee. The 32-dimensional Many-Well example has 65,536
modes from a product construction; that structured example illustrates why
explicit mode enumeration is unnecessary, but not general exhaustive
discovery. Its architecture and numerical settings are not imported here.

Another kernel literature is **Stein variational gradient descent** (Liu and
Wang, 2016): score attraction and kernel repulsion evolve a particle ensemble.
The particles are not automatically a normalized, evaluable proposal density
for p/r weighting. Projected SVGD (Chen and Ghattas, 2020) uses a
likelihood-informed low-dimensional subspace; its projection-error control
depends on neglected sensitivity eigenvalues. The assumption that the
posterior differs from the prior mainly in that subspace must be checked.
This is a way to exploit structure, not an unrestricted cure for dimension.

## What independent evaluations change

Blessing et al., *Beyond ELBOs* (ICML 2024), evaluate 11 methods across synthetic
and Bayesian targets; Sections 7–8 and Appendices D–F are directly relevant.
Their Gaussian/Student-mixture tests vary dimension over 2, 50 and 200.
Several methods that covered modes in two dimensions lost coverage at the
higher dimensions. GMMVI and FAB often obtained favorable ELBOs while still
showing mode collapse; diffusion methods sometimes covered more modes while
having other approximation or cost problems. These are findings under the
paper's architectures, tuning, metrics and budgets, not a universal ranking.

SCLD's 2025 evaluation includes a 40-component Gaussian mixture in 50
dimensions, Student mixtures and multimodal robot targets. Its reported
sample-distance/coverage results motivate investigation. Its 1,600-dimensional
LGCP case is chiefly evaluated by ELBO and must not be described as a
1,600-dimensional unknown-mode-discovery demonstration. Appendix A.6.4 shows
that settings selected for a sample-distance metric and for log-normalizer
accuracy can differ. No single scalar diagnostic certifies every goal.

## Implication for BayesFilter

### Owner clarification: FAB trains the canonical IAF

The owner's follow-up asks for FAB to help train NeuTra while retaining IAF.
These roles are compatible: IAF specifies the transport family, FAB specifies
how its parameters are trained, and NeuTra uses the trained, frozen transport
for sampling the original posterior. The proposed research direction is
therefore **FAB training of the canonical IAF for subsequent NeuTra**, not
replacement of the IAF architecture or replacement of posterior inference by
the FAB training sampler. This clarification selects the intended role; it
does not establish an implemented or validated FAB trainer.

For the paper's alpha=2 objective, define

\[
J(\phi)=\int\widetilde\pi(\theta)^2/q_\phi(\theta)\,d\theta,
\qquad
g_\phi(\theta)=\frac{\widetilde\pi(\theta)^2}
 {q_\phi(\theta)J(\phi)}.
\]

Assuming J is finite and differentiation under the integral is justified,
\(\nabla_\phi\log J=-\mathbb E_{g_\phi}\nabla_\phi\log q_\phi\).
At each outer iteration, freeze the current q for AIS, generate weighted
samples targeting g, and differentiate the paper's weighted log-density loss
with samples and weights detached. The self-normalized Monte Carlo gradient
is generally biased at finite sample size. Refreshing AIS and applying the
paper's replay corrections are necessary as q changes: fitting one fixed g
to completion would not implement the full adaptive FAB procedure.

The training target g is deliberately different from the posterior pi. It
emphasizes regions underrepresented by q. The eventual NeuTra target remains
\(\widetilde\pi(T_\phi(z))|\det DT_\phi(z)|\), with a frozen map. Neither g
nor a fitted surrogate posterior may silently replace it. This is a change
from the current reverse-KL training objective to the published alpha=2
objective, not a claim that the existing Vaitl/Roeder gradient estimator can
be reused unchanged. AIS targeting pi and weighted forward-KL training would
be a separately identified variant.

Source anchors: Midgley et al. Section 3, Eqs. 4–7 and Algorithm 1; Appendix A
and B.1; author `fab/core.py::fab_alpha_div_inner`, `fab_alpha_div`, and
`set_ais_target`. Canonical local compatibility was checked in the preserved
worktree at commit `6ccfebc027a36906d109775a474dc42a2f49634e`, whose configured
transport supplies forward sampling, inverse and log_prob, and whose weighted
trainer accepts the configured transport with dtype-aware weights/clipping.
The shared main checkout at `de80aaff5` contains older numerical files, so its
code must not substitute for that canonical snapshot in a future integration.
Only source was read; no TensorFlow process was initialized for this check.

The missing integration is a paper-grounded FAB controller with the AIS
training target, correct weighting, corrected replay, checkpointed sampler and
training state, calibration and coverage-sensitive evaluation. Existing weighted
likelihood and AIS helpers alone do not constitute that controller. Arbitrary-
point log_prob requires the IAF's sequential coordinate inverse inside the
TensorFlow graph; its cost and derivatives need measurement. The finite
alpha=2 moment and tail coverage must also be checked; full support alone does
not imply J is finite. FAB cannot amplify an undiscovered region until its
sampling mechanism reaches it, and global density coverage does not by itself
establish useful transformed HMC geometry. Independent coverage checks and
the existing 1,000-point whitening diagnostic serve different purposes.

The earlier suggestion is narrowed: an adaptive kernel or Student-t mixture
is a reasonable **candidate initializer** for the current four-dimensional
target, not a demonstrated general high-dimensional solution. Sensible next
work would select one published construction and first establish whether it
can generate reliable weighted, globally representative training data within
our target-evaluation budget. Mode-search discoveries can initialize it,
without treating exhaustive mode enumeration as a prerequisite.

If a direct proposal exhibits weight collapse, annealed sampling with
learned transport is the supported repair direction. CRAFT informs how to
produce progressively transported weighted populations. FAB is the most
direct paper to audit for changing the training of a single evaluable flow,
but changes the objective from forward KL to alpha=2 divergence. Applying
that training algorithm to the canonical IAF preserves our architecture while
using a different flow family from the FAB experiments inspected. This local
combination requires source and numerical validation; it is not already
implemented or validated merely because both use flows. SCLD is a more substantial new
sampler project and is not nominated as an immediate replacement here.

No HMC mass-adaptation change, new architecture default, or broad comparison
campaign follows from this review. The canonical IAF remains the architecture
authority. A fixed mixture proposal would leave the target-weighted
forward-KL objective intact; a different objective or sampler needs an explicit
plan and evidence before use. Current convergence and posterior readiness
remain unresolved.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Narrow the KDE recommendation | Mathematical and modern literature support the scaling concern | No local candidate run | Whether a small mixture suffices for this four-dimensional geometry | Choose a published proposal-construction method and define coverage-sensitive validation | All mixtures fail in all high dimensions |
| Investigate annealed learned sampling when direct weights fail | Mechanisms and conditional theory inspected | No new implementation validated | Target cost, exploration and canonical-IAF compatibility | Bounded source audit and experiment plan if execution is requested | FAB/CRAFT/SCLD solves q20 or universally dominates |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No new candidate tested; published numerical/mode-collapse failures remain scoped to those studies |
| Statistically supported ranking | No cross-paper or local ranking established |
| Descriptive-only differences | Published dimensions, runtimes and benchmark observations are not transferred into local performance predictions |
| Default readiness | No numerical/architecture/sampler default changed |
| Next evidence needed | Current-target weight stability, replicated regional masses, independent exploration and downstream validation |

The strongest alternative explanation for our current failure is still
objective/exploration mismatch in four dimensions rather than the
high-dimensional KDE curse. A reliable simple mixture on the current target
would make a more complex sampler unnecessary for constructing this training
bank. Conversely, unstable independent regional-mass estimates would invalidate
promotion even if flow training loss and local whitening improve.

## Sources and inspection boundary

Local PDFs and extracted text are in
`.localresources/neutra-highdim-literature-20260925/`. The accompanying
`sources.json` records download URLs and checksums. Original-author FAB
`fab/core.py` was inspected at `fab_alpha_div_inner`, `fab_alpha_div` and
`set_ais_target`; original-author CRAFT `craft.py` was inspected at
`inner_step_craft`, the training loop and its parameter update ordering. These
are bounded implementation tie-outs, not full source-faithfulness audits.
The other methods are paper-level assessments; no claim of a faithful local
implementation is made.

Primary sources:

- [Chatterjee and Diaconis, The sample size required in importance sampling](https://arxiv.org/abs/1511.01437), 2018.
- [Hoogerheide, Opschoor and van Dijk, MitISEM working paper](https://www.princeton.edu/~erp/erp%20seminar%20pdfs/papersFall2011/Hoogerheide%20Opschoor%20%20Van%20Dijk-%282011%29.pdf), precursor to the 2012 Journal of Econometrics article.
- [Delyon and Portier, Safe and adaptive importance sampling: a mixture approach](https://arxiv.org/abs/1903.08507), 2021.
- [Korba and Portier, Adaptive Importance Sampling meets Mirror Descent: a Bias-variance Tradeoff](https://proceedings.mlr.press/v151/korba22a.html), 2022.
- [Elvira et al., Gradient-based Adaptive Importance Samplers](https://arxiv.org/abs/2210.10785), 2023.
- [Arenz et al., A Unified Perspective on Natural Gradient Variational Inference with Gaussian Mixture Models](https://arxiv.org/abs/2209.11533), 2022/2023.
- [Matthews et al., Continual Repeated Annealed Flow Transport Monte Carlo](https://proceedings.mlr.press/v162/matthews22a.html), 2022.
- [Midgley et al., Flow Annealed Importance Sampling Bootstrap](https://arxiv.org/abs/2208.01893), ICLR 2023.
- [Blessing et al., Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423), ICML 2024.
- [Chen et al., Sequential Controlled Langevin Diffusions](https://arxiv.org/abs/2412.07081), ICLR 2025.
- [Liu and Wang, Stein Variational Gradient Descent](https://papers.neurips.cc/paper_files/paper/2016/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf), 2016.
- [Chen and Ghattas, Projected Stein Variational Gradient Descent](https://proceedings.neurips.cc/paper/2020/file/14faf969228fc18fcd4fcf59437b0c97-Paper.pdf), 2020.

The local copy of Wu et al., *Annealing Flow Generative Models Towards Sampling
High-Dimensional and Multi-Modal Distributions* (ICML 2025), is a supplementary
reading lead. Its staged continuous-flow objective was inspected; its theory
and implementation were not audited sufficiently to ground a recommendation.
