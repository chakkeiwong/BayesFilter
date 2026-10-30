# Literature precedents for mode discovery and flow initialization

Question: how closely does mode discovery, approximate example-based flow
initialization and subsequent reverse-KL refinement match published methods?
This is a paper/source review, not a sampling or training experiment. It adds
no numerical default and consumes no research campaign allowance.

Scope audit: distinguish discovered mode locations from posterior masses;
biased local examples from equilibrium samples; ideal KL formulas from the
loss actually implemented; standalone pure reverse-KL refinement from mixed
training; and corrected fixed-map sampling from arbitrary adaptive sampling.
These distinctions determine whether the proposed sequence has a faithful
precedent. Read technical algorithms, relevant appendices and author source
where located. Do not describe the complete local combination as a verbatim
implementation of one paper.

## Closest published procedures

### Pompe, Holmes and Latuszynski (2020): discover modes before sampling

*A Framework for Adaptive MCMC Targeting Multimodal Distributions*, Annals
of Statistics 48(5), 2930–2952. https://arxiv.org/abs/1812.02609

Section 2.2 and Algorithm 3 give a concrete preparation phase: dispersed
uniform or prior starts, parallel BFGS minimization of negative log density,
curvature-based merging of duplicate endpoints, inverse-Hessian initial
scales and further within-region covariance estimation. Supplementary
Section 11 requires first- and second-order optimality checks to exclude
saddles. This directly supports the proposed discovery phase. It is not
neural training and does not make local covariance into an exact posterior
approximation.

Equations (2.1)–(2.2) define an augmented target
pi_tilde(x,i)=pi(x) w_i Q_i(x)/sum_j w_j Q_j(x). Summing over i returns pi(x)
for any eligible positive allocation weights. The guessed allocations are
not asserted to be posterior masses. Local and inter-region corrected moves
perform the inference. Finite multistart discovery is not exhaustive; the
paper's asymptotic argument requires positive start probability in the
relevant attraction basins. Its adaptive ergodicity results have further
conditions. The author algorithm and supplement were inspected; a separate
author-code implementation was not located or audited in this focused review.

### Noe, Olsson, Kohler and Wu (2019): example initialization and energy training

*Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with
Deep Learning*, Science 365, eaaw1147.
https://doi.org/10.1126/science.aaw1147 ; https://arxiv.org/abs/1812.01729

Main equations (1)–(2) distinguish energy-based reverse KL from maximum
likelihood on example configurations. Examples can come from separate short
simulations; the particle-dimer discussion explicitly uses disconnected
open/closed-state simulations. The examples need not encode equilibrium
region weights. Methods equation (9) combines ML, energy and optional
reaction-coordinate losses. The Methods maximum-likelihood derivation
explicitly substitutes a sample distribution rho for the unknown target.

Supplementary Figure S2 is directly relevant to the proposed handoff. It
compares a common ML pretraining stage followed by different objectives.
In that double-well example, energy-only refinement tends to collapse to
one metastable state, while combined ML and energy training avoids that
collapse. This is a reported example, not a theorem or q20 result. It argues
against assuming that turning off the example term must be beneficial.
It does not justify transferring the paper's numerical coefficients.

The original code archive is DOI 10.5281/zenodo.3242635. Its
`notebooks/Fig2_DoubleWell.ipynb`, code cells 35 and 39 (zero-based JSON
indices), call `train_ML` followed by `train_flexible` with both ML and KL
terms enabled. The archived `deep_boltzmann/networks/invertible.py` has
`train_ML` at line 225, `log_KL_x` at line 462, and `train_flexible` at
line 581; lines 602–607 and 640–652 assemble the separate losses and weights.
The source uses TensorFlow/Keras and coupling-flow examples, not the current
canonical BayesFilter IAF. These source calls were inspected, not executed.

Important implementation difference: `log_KL_x` applies `linlogcut` to the
energy and exposes an `explore` multiplier on the log Jacobian. Thus its
regularized training scalar is not automatically the unmodified reverse KL
for an arbitrary target. A local implementation must explicitly identify
any such change rather than copying it under an exact-target claim. The
paper also performs importance reweighting of flow output. Neither mixed
training nor a low loss makes the generated density equal to the target.

### Gabrie, Rotskoff and Vanden-Eijnden (2022): learn while correcting mode weights

*Adaptive Monte Carlo Augmented with Normalizing Flows*, PNAS 119(10),
e2109420119. https://arxiv.org/abs/2105.12603 ;
https://doi.org/10.1073/pnas.2109420119

Sections III.B and IV.A–C, Algorithm 1, and Appendix A.3 closely match the
initialization problem. Initialize walkers in relevant metastable regions;
they need not be drawn from equilibrium. Local moves supply initially
useful within-region examples, forward training improves a flow, and flow
proposals with Metropolis correction enable inter-region travel. The sample
weights between regions can then equilibrate as training continues.

The finite-stage loss is -E_{rho_t} log q, where rho_t is the current chain
distribution, not automatically p. For a fixed learned independent proposal
q, the acceptance probability is min(1, gamma(y)q(x)/(gamma(x)q(y))). This
correction needs the unnormalized target but no known mode masses. The
paper's adaptive/continuous-time analysis must not be confused with an
automatic finite-step correctness theorem for every adaptation schedule.
The composition of invariant local/global kernels remains invariant; their
composition is not generally reversible merely because both kernels are.

Appendix G.1 / Figure 5 tests the initialization issue: local-only chains
cannot correct the initial mode proportions; initializing only one mode
misses the other in their example; initializing both with the combined
sampler produces the desired behavior. Section IV.C explicitly says not
to expect uninitialized metastable basins to be discovered. Appendix I also
reports loss of a less readily learned basin through finite-population
depletion, motivating continued local exploration or separate maps there.
That alternative is not silently adopted as our canonical architecture.

The paragraph following Algorithm 1 and Appendix A discuss combining forward
and reverse objectives after the map becomes useful. Inspected author code
`flonaco/training.py` (stored as `gabrie-training-author.py`) initializes
from supplied examples/mode locations around lines 102–127 and implements
the mixed loss around lines 239–241. Its option named `js` adds forward and
reverse losses; that name does not make the result the usual Jensen–Shannon
divergence. Only these mechanisms were checked; the entire optimizer and
all branches were not validated. The sampler source and paper include both
adjusted and unadjusted local choices; ULA must not be silently treated as an
exact finite-step MCMC kernel in a local adaptation.

### Elvira, Chouzenoux, Akyildiz and Martino (2023): maintain diverse proposals

*Gradient-based Adaptive Importance Samplers*, Journal of the Franklin
Institute. https://arxiv.org/abs/2210.10785

GRAMIS Section 2.2, Section 3 and Table 2 support the multiple-proposal
alternative. The method adapts locations with derivatives and repulsion,
uses curvature for scales, draws a fixed number from each proposal and
weights samples with the full deterministic-mixture denominator. The
repulsion addresses the tendency of multiple optimizers/proposals to merge
into the same attractive region. Experiments and the no-repulsion ablation
show the mechanism on their selected targets, not guaranteed discovery or
performance for q20. The proposed frozen local-mixture initialization shares
the weighting construction; it does not implement the full GRAMIS algorithm.
An author repository was not identified by the bounded source search here;
the paper is the inspected authority for these statements.

## Consequences for the local design

The ingredients are established. The complete combination of JAMS-like
discovery, corrected proposal clouds, the canonical IAF, and a particular
forward/reverse schedule is a local composition that needs its own test.
It must not be described as the original NeuTra paper's training recipe or
a reproduction of the Boltzmann Generator architecture.

A precise posterior sample is not needed before an initialization attempt.
If local examples follow rho_0=sum_j b_j p(.|A_j) with approximate b_j,
maximum likelihood fits rho_0, not p=sum_j a_j p(.|A_j). This may be a useful
initializer. Known-density importance proposals correct the mismatch through
weights; the Gabrie procedure instead improves it through target-corrected
global moves and refreshed training samples. These are distinct mechanisms.

The published Figure S2 comparison changes the recommendation: preserve
pure reverse-KL refinement as one explicit hypothesis and include a mixed
example/energy refinement from the same initialized checkpoint. Do not
silently assume that the forward coefficient should become zero. No fixed
coefficient, stage duration, batch count or energy clipping is inherited.

For an ideal target-based forward term, a mixture
lambda KL(p||q)+(1-lambda)KL(q||p) has the same unrestricted minimizer p.
For a fixed biased example law rho, lambda KL(rho||q)+(1-lambda)KL(q||p)
generally does not. Retaining a fixed biased example term is therefore an
explicit initialization/regularization choice, not exact posterior training.
Refreshing examples through corrected sampling or ending with appropriately
assessed true-target refinement addresses a different objective. Final
Metropolis-corrected inference still uses the original posterior.

| Decision | Evidence | Limitation | Next justified action |
|---|---|---|---|
| Keep mode discovery first | JAMS Algorithm 3 and supplement | No finite exhaustive guarantee | Test target-valid multistart/local-proposal construction |
| Permit coarse, region-representative initialization | BG equations (1)–(2); Gabrie IV.C/A.3 | Coarse samples define an approximate training distribution | Assess canonical IAF initialization without demanding a converged archive |
| Reconsider an automatic pure-RKL handoff | BG Figure S2 and archived notebook calls | Published target differs from q20 | Compare pure and mixed refinement with independent coverage checks |
| Preserve correction and architecture boundaries | Weight/MH identities; inspected source options | No local sampler or training result | Keep source-mapped formulas, actual target and canonical core explicit |

No BayesFilter stochastic ranking, new default or successful trained map is
established. Strongest unresolved explanation for failure is missing regions
or capacity/optimization mismatch; correct weighting alone cannot exclude
either. Exact source copying also cannot establish target suitability.

Source copies and hashes are preserved in
`.localresources/neutra-mode-initialization-literature-20260929/`, together
with the previously saved JAMS, GRAMIS and flonaco paper/source paths listed
in its `source-checksums.json`. The BG GitHub discovery initially used a
stale repository name; the actual author repository is
https://github.com/noegroup/paper_boltzmann_generators. Source inspection uses
the paper's versioned Zenodo archive. No downloaded code was run or installed.

## Follow-up: author-source adoption

The user's subsequent instruction selects Gabrié's concurrent sampling and
forward-training procedure as the first implementation path. The complete
`flonaco` package source, licence and selected examples are now saved at
`.localresources/flonaco-author-20260929/`, pinned to
`6b9286b4e58194aa65373200d7bfacde06a2d180`. All 17 acquired files match the
pinned Git tree. The MIT licence permits the intended port with attribution.

See `bayesfilter-gabrie-flonaco-source-adoption-2026-09-29.md` for exact
sampler/trainer anchors, the MALA versus ULA distinction and required
equivalence checks. The active warm-start design now prioritizes that source
procedure with the canonical IAF. The static importance-cloud and subsequent
pure/mixed refinement ideas above remain separately identified hypotheses.
Source acquisition does not establish a completed TensorFlow port or a
successful trained map. The earlier literature checksum file records its
original document snapshot; it is not a hash of these follow-up edits.
