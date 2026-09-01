# Source-Support Ledger

Audit date: 2026-08-18. `OpenAlex: not retracted` means the exact OpenAlex work
record had `is_retracted=false` on this date; it is a dated index check, not an
independent proof that no correction exists. Local PDFs were inspected as full
text, not only through metadata or abstracts.

## Core discontinuous and mixed HMC sources

### Nishimura, Dunson, and Lu (2020)

- **Title / identifier:** *Discontinuous Hamiltonian Monte Carlo for Discrete
  Parameters and Discontinuous Likelihoods*; DOI
  `10.1093/biomet/asz083`; *Biometrika* 107(2), 365--380.
- **Class / status:** `FOUNDATIONAL`; published journal article; OpenAlex
  W2795604964, not retracted. The index uses online year 2019; the inspected
  issue and citation year are 2020.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/discrete/Discontinuous Hamiltonian Monte Carlo for discrete parameters and discontinuous likelihoods Nishimura(20).pdf`.
- **Inspected anchors:** Sec. 2.1, embedding in eq. (2); Sec. 3.1, Laplace
  momentum and coordinate update in Algorithm 1; Proposition 1 and Corollary 1
  on reversibility and volume preservation; Sec. 3.2 on randomized order and
  step size; supplement on alternative kinetic energies.
- **Allowed support:** ordinal embedding, coordinatewise Laplace update,
  reflection rule, and stated invariance conditions. **Forbidden:** claiming
  that the paper supplies a nonlinear ZLB solver or a full-posterior boundary
  oracle.

### Afshar and Domke (2015)

- **Title / identifier:** *Reflection, Refraction, and Hamiltonian Monte Carlo*;
  NeurIPS 28, 3007--3015; ANU handle `1885/103828`.
- **Class / status:** `FOUNDATIONAL`; archival proceedings paper; OpenAlex
  W2185585545, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/discrete/Reflection, refraction, and hamiltonian monte carlo Afashar(15).pdf`.
- **Inspected anchors:** Sec. 2 energy argument; Algorithm 1 event-aware
  leapfrog; Sec. 5, Lemmas 1--2 and Theorem 1 on affine-boundary volume
  preservation; experiments only as method diagnostics.
- **Allowed support:** first-boundary detection and Gaussian-momentum
  reflection/refraction for affine piecewise-smooth targets. **Forbidden:**
  extrapolating the affine proof to an implicit multi-root equilibrium surface.

### Zhou (2020)

- **Title / identifier:** *Mixed Hamiltonian Monte Carlo for Mixed Discrete and
  Continuous Variables*; NeurIPS 2020, volume 33; arXiv `1909.04852`.
- **Class / status:** `DIRECT_METHOD`; archival proceedings version inspected;
  OpenAlex W3034642882 represents the preprint, not retracted. OpenAlex
  undercounts or duplicates proceedings metadata, so its count is not used to
  characterize impact.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/discrete/Mixed Hamiltonian Monte Carlo for mixed discrete and continuous variables Zhou(20).pdf`.
- **Inspected anchors:** Sec. 2.2 eq. (1), torus clocks, proposal-corrected
  energy change and event rule; Sec. 2.3 Theorem 1 and proof sketch; Sec. 2.4
  Laplace implementation; supplementary detailed-balance construction.
- **Allowed support:** mixed-support clock dynamics and the exact proposal ratio.
  **Forbidden:** applying a regime-only move to deterministic support (5a), or
  treating a conditional Gibbs draw inside HMC as automatically valid.

### Torgander, Magnusson, and Wallin (2024)

- **Title / identifier:** *Hamiltonian Monte Carlo with Categorical Parameters
  Using the Concrete Distribution*; 6th AABI Symposium workshop, 1--12.
- **Class / status:** `COMPETITOR`; explicitly non-archival workshop paper; no
  stable DOI located; no exact OpenAlex record located in the bounded search.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/discrete/Hamiltonian Monte Carlo with categorical parameters using the Concrete distribution Torgander(24).pdf`.
- **Inspected anchors:** Sec. 2 Concrete-mixture construction; Appendix A
  Gumbel-softmax map; Appendix B transformed density and theoretical results;
  Secs. 3--4 experiments and limitations.
- **Allowed support:** a continuous-relaxation comparator. **Forbidden:** exact
  categorical, hard-ZLB, or posterior-equivalence claims.

### Chaari et al. (2016)

- **Title / identifier:** *A Hamiltonian Monte Carlo Method for Non-Smooth
  Energy Sampling*; DOI `10.1109/TSP.2016.2585120`; *IEEE TSP* 64(21),
  5585--5594.
- **Class / status:** `COMPETITOR`; published journal article; OpenAlex
  W2231895817, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/constrained/A Hamiltonian Monte Carlo Method for Non-Smooth Energy Sampling Chaari(16).pdf`.
- **Inspected anchors:** Sec. II-A HMC; Sec. III proximity-based leapfrog and
  volume-preservation discussion; Appendix A proximity calculation.
- **Allowed support:** convex nonsmooth/proximal HMC. **Forbidden:** crossing a
  density jump or choosing a ZLB equilibrium branch.

## Occasionally binding constraint and DSGE sources

### Guerrieri and Iacoviello (2015)

- **Title / identifier:** *OccBin: A Toolkit for Solving Dynamic Models with
  Occasionally Binding Constraints Easily*; DOI
  `10.1016/j.jmoneco.2014.08.005`; *JME* 70, 22--38.
- **Class / status:** `FOUNDATIONAL`; published journal article; OpenAlex
  W2120422722, not retracted. The local 2014 working-paper text was checked
  against published title, DOI, and pagination metadata.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/DSGE/QE/OccBin A toolkit for solving dynamic models with occasionally binding constraints easily Guerrieri(14).pdf`.
- **Inspected anchors:** Sec. 2 solution algorithm, especially backward solution,
  forward simulation, and inequality verification; Sec. 5 ZLB model; Appendices
  A--B nonlinear comparison.
- **Allowed support:** piecewise-linear regime-sequence solution and its limits.
  **Forbidden:** exact nonlinear filtering or global uniqueness.

### Giovannini, Pfeiffer, and Ratto (2021)

- **Title / identifier:** *Efficient and Robust Inference of Models with
  Occasionally Binding Constraints*; JRC Working Paper 2021/3; handle
  `10419/249365`.
- **Class / status:** `DIRECT_METHOD`; working paper; OpenAlex W3160175912, not
  retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/DSGE/QE/Efficient and robust inference of models with occasionally binding constraints Giovannini(21).pdf`.
- **Inspected anchors:** Sec. 2, equations (5)--(13) and algorithm outline;
  Secs. 3--5 applications; Sec. 6 Dynare implementation.
- **Allowed support:** PKF guess-and-verify recursion and reported empirical
  behavior. **Forbidden:** exactness outside the conditional piecewise-linear
  Gaussian model.

### Aruoba et al. (2021)

- **Title / identifier:** *Piecewise-Linear Approximations and Filtering for
  DSGE Models with Occasionally Binding Constraints*; DOI
  `10.21799/frbp.wp.2020.13`; FRB Philadelphia Working Paper 20-13.
- **Class / status:** `DIRECT_METHOD`; working-paper version dated 2021; OpenAlex
  W4285695907 exact metadata request was rate-limited after an earlier successful
  metadata lookup, so retraction status is `not available` rather than inferred.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/DSGE/QE/Piecewise-Linear Approximations and Filtering for DSGE Models with Occasionally Binding Constraints Aruoba(21).pdf`.
- **Inspected anchors:** Sec. 4 canonical PLC form; Sec. 6 Algorithm 1 and
  Proposition 1, including regime-specific truncated Gaussian terms; online
  appendix algorithms.
- **Allowed support:** COPF proposal and likelihood for the paper's PLC model.
  **Forbidden:** a general nonlinear transition or an HMC score.

### Childers et al. (2022)

- **Title / identifier:** *Differentiable State-Space Models and Hamiltonian
  Monte Carlo Estimation*; NBER Working Paper 30573; DOI `10.3386/w30573`.
- **Class / status:** `DIRECT_METHOD`; non-peer-reviewed working paper; OpenAlex
  W4307182406, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/DSGE/Machine Learning/Differentiable State-Space Models and Hamiltonian Monte Carlo Estimation Childers(22).pdf`.
- **Inspected anchors:** Sec. 2 joint likelihood; Sec. 3 HMC; Sec. 4 implicit
  differentiation; Sec. 5 implementation; Appendices C, F, and G.
- **Allowed support:** joint latent-state/shock HMC for differentiable model
  programs. **Forbidden:** hard-branch differentiability or ZLB event handling.

## Particle and differentiable-filter sources

### Andrieu, Doucet, and Holenstein (2010)

- **Identifier/status:** *Particle Markov Chain Monte Carlo Methods*; DOI
  `10.1111/j.1467-9868.2009.00736.x`; *JRSS B* 72(3), 269--342;
  `FOUNDATIONAL`; OpenAlex W1501586228, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/Particle Markov chain Monte Carlo methods Andrieu(10).pdf`.
- **Inspected anchors:** Sec. 2 particle likelihood; Sec. 4 extended PMCMC
  construction; Appendix A conditional SMC; proofs in Appendix B.
- **Allowed support:** nonnegative unbiased likelihood and exact extended-target
  PMMH/particle Gibbs. **Forbidden:** finite-particle log-likelihood unbiasedness.

### Alenlöv, Doucet, and Lindsten (2021)

- **Identifier/status:** *Pseudo-Marginal Hamiltonian Monte Carlo*; *JMLR* 22,
  article 141, 1--45; `DIRECT_METHOD`; OpenAlex has split preprint/published
  records, all located records not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/hamiltonian MC/Pseudo-Marginal Hamiltonian Monte Carlo Alenlov(21).pdf`.
- **Inspected anchors:** Sec. 2 extended target and Algorithm 1; Proposition 3
  stationarity; Sec. 3 Propositions 4--6 convergence to ideal HMC; appendices on
  integrator and assumptions.
- **Allowed support:** exact PM-HMC under its differentiable extended estimator
  assumptions. **Forbidden:** differentiability of ordinary discrete resampling.

### Lindsten, Jordan, and Schön (2014)

- **Identifier/status:** *Particle Gibbs with Ancestor Sampling*; *JMLR* 15,
  2145--2184; `DIRECT_METHOD`; OpenAlex W2140051159, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/nonlinear SSM/particle/Smoothing/Particle Gibbs with Ancestor Sampling Lindsten(14).pdf`.
- **Inspected anchors:** Sec. 3 Algorithm 2; Sec. 4 Theorem 1 and partial-collapse
  proof; Sec. 5 Algorithm 3 and Markov ancestor weights; Theorems 2--3.
- **Allowed support:** PGAS invariance for latent paths. **Forbidden:** an
  independent update of a deterministic regime label.

### Poyiadjis, Doucet, and Singh (2011)

- **Identifier/status:** *Particle Approximations of the Score and Observed
  Information Matrix in State Space Models with Application to Parameter
  Estimation*; DOI `10.1093/biomet/asq062`; *Biometrika* 98(1), 65--80;
  `DIRECT_METHOD`; OpenAlex W2010522529, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/nonlinear SSM/particle/score estimation/Particle approximations of the score and observed infomration matrix in state space models with application to parameter estimation Poyiadjis(11).pdf`.
- **Inspected anchors:** Secs. 2--3 score and observed-information estimators;
  propositions comparing path-space and forward recursion; numerical study for
  variance behavior.
- **Allowed support:** particle score estimator structure and asymptotics.
  **Forbidden:** treating a finite-particle score as the exact marginal score.

### Ścibior and Wood (2021)

- **Identifier/status:** *Differentiable Particle Filtering without Modifying
  the Forward Pass*; arXiv `2106.10314v2`; `DIRECT_METHOD`; preliminary work;
  OpenAlex W4287116413/W3174237283 duplicates, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/nonlinear SSM/particle/differentiable/Differentiable Particle Filtering without Modifying the Forward Pass Scibior(21).pdf`.
- **Inspected anchors:** Algorithm 1; Secs. 2--4 stop-gradient calculus and score
  estimator; Theorem 1 in supplement; Sec. 7 limitations.
- **Allowed support:** unchanged forward PF plus corrected AD estimator.
  **Forbidden:** exact finite-particle HMC force or peer-reviewed status.

### Corenflos et al. (2021)

- **Identifier/status:** *Differentiable Particle Filtering via
  Entropy-Regularized Optimal Transport*; ICML 2021, PMLR 139, 2100--2111;
  arXiv `2102.07850v3`; `DIRECT_METHOD`; OpenAlex duplicate records are not
  retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/nonlinear SSM/particle/differentiable/Differentiable Particle Filtering via Entropy-Regularized Optimal Transport Corenflos(21).pdf`.
- **Inspected anchors:** Algorithms 1 and 3; Secs. 3--4 transport resampling and
  Propositions 4.1--4.3; Appendices A--D proofs; Sec. 6 limits.
- **Allowed support:** differentiable regularized transport resampling and its
  stated convergence regime. **Forbidden:** equality with ordinary resampling
  at finite \(M\) and fixed regularization.

## Constrained, truncated, and mixture sigma-point sources (added 2026-08-19)

These sources support Section 4.3. Audit date for this block: 2026-08-19.
Four papers were inspected as local full text; five are pay-walled and are
cited from verified OpenAlex/Crossref metadata as family orientation only.
Every equation in Section 4.3 is a project derivation checked numerically by
`derivation_check_ukf_section_20260819.py`; no equation is imported from a
blocked source.

### Kandepu, Imsland, and Foss (2008)

- **Title / identifier:** *Constrained State Estimation Using the Unscented
  Kalman Filter*; DOI `10.1109/MED.2008.4602001`; 16th Mediterranean Conference
  on Control and Automation, Ajaccio, 1453--1458.
- **Class / status:** `DIRECT_METHOD`; published proceedings paper; OpenAlex
  W2098630962, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/constrained and degenerate/Constrained state estimation using the unscented Kalman filter Kandepu(08).pdf`.
- **Inspected anchors:** Sec. II UKF algorithm, eqs. (2)--(13); Sec. III
  constraint handling by projecting transformed sigma points onto the feasible
  region, Fig. 2; comparison against EKF "clipping".
- **Allowed support:** sigma-point projection as a constrained-UKF mechanism and
  its approximate character. **Forbidden:** claiming projection represents
  branch-conditional densities or an endogenous regime change.

### Teixeira, Tôrres, Aguirre, and Bernstein (2010)

- **Title / identifier:** *On Unscented Kalman Filtering with State Interval
  Constraints*; DOI `10.1016/j.jprocont.2009.10.007`; *Journal of Process
  Control* 20(1), 45--57.
- **Class / status:** `DIRECT_METHOD` and comparison authority; published
  journal article; OpenAlex W1999345491, not retracted. The index year 2009
  reflects online publication; the issue is January 2010.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/constrained and degenerate/On unscented Kalman filtering with state interval constraints Teixeira(10).pdf`.
- **Inspected anchors:** Sec. 2 problem formulation (2.1)--(2.3); Sec. 3 UT/UKF
  recursions (3.1)--(3.16); Sec. 4 interval-constraint statement (4.1); Sec. 5
  review of forecast/assimilation constraint enforcement, including the
  interval-constrained unscented transform (5.1)--(5.6) and the truncation
  procedure (5.2.4); Table 1 taxonomy of eight constrained-UKF variants; Sec. 2
  statement that the motivating examples have multimodal densities.
- **Allowed support:** the projection/constraint/truncation taxonomy and the
  explicitly approximate status of every variant. **Forbidden:** optimality or
  exact-filter claims and any endogenous-branch statement.

### Zhang, Zhang, Xie, and Yang (2020)

- **Title / identifier:** *Constrained Multiple Model UK Filter*; DOI
  `10.1109/ICSP48669.2020.9320991`; 15th IEEE International Conference on
  Signal Processing (ICSP), Beijing, 48--51.
- **Class / status:** `DIRECT_METHOD`; published proceedings correspondence;
  OpenAlex W3127769421, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/constrained and degenerate/Constrained multiple model UK filter Zhang(20).pdf`.
- **Inspected anchors:** Sec. II Markov model-jump formulation, eqs. (1)--(4);
  Sec. III truncated measurement-noise support (5), model-conditioned feasible
  region (6), constrained likelihood (7), importance density and weights
  (9)--(11), objective (12); Table 1 algorithm summary.
- **Allowed support:** constrained multiple-model UKF structure for genuinely
  stochastic Markov switching. **Forbidden:** applying its mode-transition
  matrix to a deterministic threshold.

### Wang, Wang, He, and Sun (2020)

- **Title / identifier:** *Iterative Truncated Unscented Particle Filter*; DOI
  `10.3390/info11040214`; *Information* 11(4), 214.
- **Class / status:** `DIRECT_METHOD`; published MDPI journal article; OpenAlex
  W3016809523, not retracted.
- **Local full text:**
  `/home/ubuntu/google-drive-papers/bayesian and SSM/constrained and degenerate/Iterative Truncated Unscented Particle Filter Wang(20).pdf`.
- **Inspected anchors:** Sec. 3.1.1 iterated Newton/MAP unscented update, eqs.
  (16)--(24); Sec. 3.1.2 truncated pdf and its Gaussian moment approximation,
  eqs. (25)--(31); Sec. 3.2 Algorithm 1 and importance weight eq. (36).
- **Allowed support:** truncated-UKF proposals inside a particle filter with
  importance reweighting, and the fact that their weight numerator keeps a
  Gaussian predictive approximation. **Forbidden:** treating their filter as an
  exactly weighted likelihood estimator or as ZLB-specific.

### García-Fernández, Morelande, and Grajal (2012a)

- **Title / identifier:** *Truncated Unscented Kalman Filtering*; DOI
  `10.1109/TSP.2012.2193393`; *IEEE Transactions on Signal Processing* 60(7),
  3372--3386.
- **Class / status:** `DIRECT_METHOD` with full text `SOURCE_BLOCKED`;
  published journal article; OpenAlex W2042845763, not retracted; Semantic
  Scholar reports closed access; no local copy.
- **Allowed support:** existence and naming of the truncated-UKF architecture
  as family orientation. **Forbidden:** any equation-, theorem-, or
  performance-level claim from the paper; Section 4.3 derives its own truncated
  construction.

### García-Fernández, Morelande, and Grajal (2012b)

- **Title / identifier:** *Mixture Truncated Unscented Kalman Filtering*; 15th
  International Conference on Information Fusion (FUSION), Singapore, 479--486;
  no DOI located.
- **Class / status:** `DIRECT_METHOD` with full text `SOURCE_BLOCKED`;
  OpenAlex duplicate records W1571674711 / W3149050165, not retracted; no local
  copy.
- **Allowed support:** existence and naming of the mixture-truncated variant.
  **Forbidden:** any technical claim beyond the title-level architecture.

### Gokce and Kuzuoglu (2015)

- **Title / identifier:** *Unscented Kalman Filter-Aided Gaussian Sum Filter*;
  DOI `10.1049/iet-rsn.2014.0088`; *IET Radar, Sonar & Navigation* 9(5),
  589--599.
- **Class / status:** `DIRECT_METHOD` with full text `SOURCE_BLOCKED`;
  published journal article; OpenAlex W2050970093; Crossref confirms the DOI
  metadata; not retracted; no local copy.
- **Allowed support:** existence of a UKF-based Gaussian-sum filter as family
  orientation. **Forbidden:** any technical claim from the paper body.

### Alspach and Sorenson (1972)

- **Title / identifier:** *Nonlinear Bayesian Estimation Using Gaussian Sum
  Approximations*; DOI `10.1109/TAC.1972.1100034`; *IEEE Transactions on
  Automatic Control* 17(4), 439--448.
- **Class / status:** `FOUNDATIONAL` with full text `SOURCE_BLOCKED`; OpenAlex
  W2099867508, not retracted; no local copy.
- **Allowed support:** historical attribution of Gaussian-sum filtering.
  **Forbidden:** theorem-level claims; the survey's mixture recursion is a
  project derivation.

### Blom and Bar-Shalom (1988)

- **Title / identifier:** *The Interacting Multiple Model Algorithm for Systems
  with Markovian Switching Coefficients*; DOI `10.1109/9.1299`; *IEEE
  Transactions on Automatic Control* 33(8), 780--783.
- **Class / status:** `FOUNDATIONAL` with full text `SOURCE_BLOCKED`; OpenAlex
  W2117397690, not retracted; no local copy.
- **Allowed support:** historical attribution of the IMM algorithm. The survey's
  mixing equations (49)--(51) are derived independently and checked by
  enumeration. **Forbidden:** paper-specific algorithmic or optimality claims.

## Shadow-rate term structure and censored-measurement sources (added 2026-08-19)

These sources support Sections 13 and 5.1. Audit date: 2026-08-19. Unless
marked otherwise, the listed local PDF was inspected as full text via
extracted text and page images of the cited sections.

### Krippner (2012)

- **Identifier/status:** *Modifying Gaussian Term Structure Models When
  Interest Rates Are Near the Zero Lower Bound*; Reserve Bank of New Zealand
  Discussion Paper DP2012/02; `FOUNDATIONAL`; OpenAlex holds only
  SSRN/RePEc duplicate records without a canonical DOI; none retracted.
- **Local full text:** `.../finance/yield curve/ZLB/Modifying Gaussian term
  structure models when interest rates are near the zero lower bound
  Krippner(12).pdf`.
- **Inspected anchors:** Secs. 1--2: Black's currency-option observation;
  intractability of direct Black-GATSMs (Gorovoi--Linetsky); the ZLB-GATSM
  construction bounding the *forward* curve with a closed-form option effect
  and obtaining yields by elementary numerical integration.
- **Allowed support:** the bounded-forward-then-integrate architecture that
  MacroFinance instantiates. **Forbidden:** arbitrage-exactness claims for the
  MacroFinance reduced form.

### Priebsch (2013)

- **Identifier/status:** *Computing Arbitrage-Free Yields in Multi-Factor
  Gaussian Shadow-Rate Term Structure Models*; FEDS 2013-63; DOI
  `10.17016/feds.2013.63`; `DIRECT_METHOD`; OpenAlex W27412800, not
  retracted; a later Quarterly Journal of Finance version exists and was not
  separately audited.
- **Local full text:** `.../finance/yield curve/ZLB/Computing arbitrage-free
  yields in multi-factor Gaussian shadow-rate term structure models
  Prebsch(13).pdf`.
- **Inspected anchors:** title page and abstract; method identified as
  cumulant-based approximation. Cited for family orientation only.
- **Allowed support:** existence of the cumulant approximation route.
  **Forbidden:** accuracy claims not checked here.

### Kim and Priebsch (2013)

- **Identifier/status:** *Estimation of Multi-Factor Shadow-Rate Term
  Structure Models*; preliminary draft, 9 October 2013, Federal Reserve
  Board; `DIRECT_METHOD`; no exact indexed OpenAlex record located in the
  bounded search.
- **Local full text:** `.../finance/yield curve/ZLB/Estimation of multi-factor
  shadow-rate term structure models Kim(13).pdf`.
- **Inspected anchors:** Sec. 4 estimation: observation equation (8), the
  extended-filter linearization (9) and its instability discussion, the
  unscented-transform alternative and cost comparison; statement that earlier
  zero-bound studies used the extended filter.
- **Allowed support:** the domain precedent for UKF-based QML estimation of
  shadow-rate models and the recorded EKF-instability motivation.
  **Forbidden:** treating UKF QML as exact likelihood inference.

### Wu and Xia (2016)

- **Identifier/status:** *Measuring the Macroeconomic Impact of Monetary
  Policy at the Zero Lower Bound*; DOI `10.1111/jmcb.12300`; *JMCB* 48(2--3),
  253--291; `DIRECT_METHOD`; OpenAlex W3125912523, 1,854 citations, not
  retracted.
- **Local full text:** `.../finance/yield curve/ZLB/Measuring the
  macroeconomic impact of monetary policy at the zero lower bound Wu(16).pdf`
  (a 2015 copy also present).
- **Inspected anchors:** Sec. 2.3, eqs. (6)--(8): closed-form forward
  observation with \(g(z)=z\Phi(z)+\phi(z)\); extended-Kalman estimation and
  the monotonicity remark; GATSM comparison eq. (9).
- **Allowed support:** the closed-form censored-forward observation and EKF
  estimation practice. **Forbidden:** shadow-rate policy-measure claims.

### Christensen and Rudebusch (2015)

- **Identifier/status:** *Estimating Shadow-Rate Term Structure Models with
  Near-Zero Yields*; DOI `10.1093/jjfinec/nbu010`; *Journal of Financial
  Econometrics* 13(2), 226--259; `DIRECT_METHOD`; OpenAlex W2126788221 (year
  2014 reflects online publication; OpenAlex's author disambiguation
  misnames the first author, who is Jens H. E. Christensen on the inspected
  paper), not retracted.
- **Local full text:** `.../finance/yield curve/ZLB/Estimating shadow-rate
  term structure models with near-zero yields Christensen(15).pdf`.
- **Inspected anchors:** Appendix B: extended Kalman filter estimation on the
  discretized Ornstein--Uhlenbeck transition; introduction's note that
  earlier estimation was two-factor EKF; sensitivity of shadow-rate paths
  noted in the text.
- **Allowed support:** EKF practice for arbitrage-free shadow-rate NS models
  and the transition form matching survey eq. (75). **Forbidden:** empirical
  U.S. conclusions.

### Bauer and Rudebusch (2016)

- **Identifier/status:** *Monetary Policy Expectations at the Zero Lower
  Bound*; DOI `10.1111/jmcb.12338`; *JMCB* 48(7), 1439--1465;
  `EMPIRICAL_EXAMPLE` with one method finding used; OpenAlex W3123152602, not
  retracted. The local copy is the 2015 working-paper text; the published
  pagination is cited from verified metadata.
- **Local full text:** `.../finance/yield curve/ZLB/Monetary policy
  expectations at the zero lower bound Bauer(15).pdf`.
- **Inspected anchors:** the passage documenting that shadow-short-rate
  estimates are highly sensitive to the assumed lower bound and model
  specification (their Sec. discussion of shadow-rate usefulness and fn. 6).
- **Allowed support:** the sensitivity finding used in Sec. 13.6.
  **Forbidden:** policy-expectation conclusions.

### Lemke and Vladu (2017)

- **Identifier/status:** *Below the Zero Lower Bound: A Shadow-Rate Term
  Structure Model for the Euro Area*; ECB Working Paper 1991; ISBN
  978-92-899-2710-9; `EMPIRICAL_EXAMPLE`; OpenAlex holds SSRN duplicates, not
  retracted.
- **Local full text:** `.../finance/yield curve/ZLB/Below the zero lower
  bound A shadow-rate term structure model for the euro area Lemke(17).pdf`.
- **Inspected anchors:** estimation statements: maximum likelihood via the
  extended Kalman filter; time-varying/estimated lower bound treatment.
- **Allowed support:** the estimated, shifting lower-bound precedent used in
  Sec. 13.3. **Forbidden:** euro-area empirical conclusions.

### Opschoor and van der Wel (2024)

- **Identifier/status:** *A Smooth Shadow-Rate Dynamic Nelson-Siegel Model for
  Yields at the Zero Lower Bound*; DOI `10.1080/07350015.2024.2365779`;
  *JBES* 43(2), 298--311; `DIRECT_METHOD` and closest published counterpart
  of the MacroFinance design; OpenAlex W4399562250, not retracted; open
  access.
- **Local full text:** `.../finance/yield curve/discrete time/A Smooth
  Shadow-Rate Dynamic Nelson-Siegel Model for Yields at the Zero Lower Bound
  Opschoor(24).pdf`.
- **Inspected anchors:** Sec. 2.1 DNS with AR(1) factors (their eq. (2));
  Sec. 2.2 yield-level bound (3), smooth approximations (4)--(7) including
  the softplus, sharpness scaling \(\gamma f(x/\gamma)\), and Fig. 1; the
  economic argument for smoothing (gradual departures from the bound); the
  two-step NLS/OLS estimation and the noted EKF state-space alternative.
- **Allowed support:** the yield-level smooth shadow-rate DNS construction,
  the smoothing-function menu, and the structural reading of the sharpness
  constant used in Sec. 13.5. **Forbidden:** forecast-performance claims and
  any transfer of their yield-level bound to MacroFinance's forward-level
  bound without the (80) distinction.

### Geng, Liu, Ma, and Yi (2021)

- **Identifier/status:** *Multi-Sensor Filtering Fusion Meets Censored
  Measurements Under a Constrained Network Environment*; DOI
  `10.1080/00207721.2021.2005178`; *Int. J. Systems Science* 52(16),
  3410--3436; `SURVEY_OR_TUTORIAL`; OpenAlex W3215453560, not retracted.
- **Local full text:** `.../bayesian and SSM/constrained and degenerate/
  Multi-sensor filtering fusion meets censored measurements under a
  constrained network environment advances challenges and prospects
  Geng(21).pdf`.
- **Inspected anchors:** the Tobit-type-1 censoring model definition (their
  eq. (14) region) and the Tobit Kalman filter review passage.
- **Allowed support:** the noise-ordering distinction of Sec. 13.4 (censoring
  before versus after noise). **Forbidden:** theorem-level claims from
  surveyed papers without primary inspection.

### Metadata-only sources for Sections 5.1 and 13

- **Black (1995).** *Interest Rates as Options*; DOI
  `10.1111/j.1540-6261.1995.tb05182.x`; *Journal of Finance* 50(5),
  1371--1376; `FOUNDATIONAL`; OpenAlex W2144290449, not retracted; full text
  not in the local corpus. Historical attribution of the shadow-rate option
  observation only; the construction used here is Krippner's inspected
  description.
- **Allik, Miller, Piovoso, and Zurakowski (2016).** *The Tobit Kalman
  Filter*; DOI `10.1109/TCST.2015.2432155`; *IEEE TCST* 24(1), 365--371;
  `DIRECT_METHOD` with full text `SOURCE_BLOCKED`; OpenAlex W2218617828, not
  retracted. Named as the Tobit-filter reference; its content is used only
  through the inspected Geng et al. survey description.
- **Pakman and Paninski (2014).** *Exact Hamiltonian Monte Carlo for
  Truncated Multivariate Gaussians*; DOI `10.1080/10618600.2013.788448`;
  *JCGS* 23(2), 518--542; `DIRECT_METHOD` with full text `SOURCE_BLOCKED`;
  OpenAlex W2085893390, not retracted. Cited for the existence of exact
  piecewise-quadratic HMC; reclassified from the earlier omit-with-reason
  status because Secs. 5.1 and 13.4 now name it as an implementation
  candidate; full-text audit is required before implementing it.

## Occasionally-binding-constraint solution and estimation sources (added 2026-08-19)

### Holden (2023)

- **Identifier/status:** *Existence and Uniqueness of Solutions to Dynamic
  Models with Occasionally Binding Constraints*; DOI `10.1162/rest_a_01122`;
  *Review of Economics and Statistics* 105(6), 1481--1499; `FOUNDATIONAL`
  for Sec. 14; OpenAlex W2340435193 (online year 2021), not retracted. The
  local copy is the author's accepted version; earlier ledger rows recorded
  this work under its 2022 working-paper year.
- **Local full text:** `.../DSGE/QE/Existence and uniqueness of solutions to
  dynamic models with occasionally binding constraints Holden(22).pdf`.
- **Inspected anchors:** abstract and Sec. 1; Sec. 2.1 knife-edge example;
  Sec. 2.2 lagged-rule example with its LCP (their eqs. (1)--(2)), Figs.
  1--2, the \(\pi_0=-r/A^2\) threshold discussion, and the pervasive-
  multiplicity remark; Sec. 3 LCP equivalence and Definition 2; Sec. 4
  P-matrix uniqueness statements.
- **Allowed support:** the LCP representation (93), the P-matrix uniqueness
  criterion, and the worked example reproduced with full derivation as survey
  eqs. (94)--(96). **Forbidden:** policy conclusions (price-level targeting)
  not re-derived here.

### Holden (2016)

- **Identifier/status:** *Computation of Solutions to Dynamic Models with
  Occasionally Binding Constraints*; unpublished working paper with the
  DynareOBC implementation; `DIRECT_METHOD`; OpenAlex holds RePEc records
  without canonical DOI; not retracted.
- **Local full text:** `.../DSGE/QE/Computation of solutions to dynamic models
  with occasionally binding constraints Holden(16).pdf`.
- **Inspected anchors:** introduction: imposition of the bound via anticipated
  news shocks; DynareOBC lineage notes.
- **Allowed support:** the news-shock representation named in Sec. 14.1.
  **Forbidden:** solver performance claims.

### Holden (2017)

- **Identifier/status:** *Tractable Estimation and Smoothing of Highly
  Nonlinear Dynamic State-Space Models*; unpublished working paper;
  `COMPETITOR`; no canonical indexed record located; not retracted so far as
  the bounded search shows.
- **Local full text:** `.../DSGE/QE/Tractable estimation and smoothing of
  highly nonlinear dynamic state-space models Holden(17).pdf`.
- **Inspected anchors:** abstract: extended skew-t augmented-state cubature
  Kalman filter with dynamic state-space reduction, third/fourth-moment
  tracking, OBC-model application.
- **Allowed support:** the existence of a skew-aware sigma-point competitor
  named in Sec. 14.4. **Forbidden:** accuracy claims without full method
  audit.

### Boehl (2022)

- **Identifier/status:** *Efficient Solution and Computation of Models with
  Occasionally Binding Constraints*; DOI `10.1016/j.jedc.2022.104523`; *JEDC*
  143, 104523; `DIRECT_METHOD`; OpenAlex W4295129637, not retracted.
- **Local full text:** `.../DSGE/QE/Efficient Solution and Computation of
  Models with Occasionally Binding Constraints Boehl(22).pdf`.
- **Inspected anchors:** abstract: closed-form trajectory given guessed spell
  durations; speed comparison claim recorded as the paper's own claim.
- **Allowed support:** the duration-search solution operator named in
  Sec. 14.1. **Forbidden:** relative-speed claims as established fact.

### Boehl and Strobel (2023)

- **Identifier/status:** *Estimation of DSGE Models with the Effective Lower
  Bound*; DOI `10.1016/j.jedc.2023.104784`; *JEDC* 158, 104784;
  `COMPETITOR`; OpenAlex W4388486195, not retracted.
- **Local full text:** `.../DSGE/QE/Estimation of DSGE Models with the
  Effective Lower Bound Boehl(23).pdf`.
- **Inspected anchors:** abstract and Sec. 1: ensemble Kalman filter with
  shifting-based updates, combination with the piecewise-linear solver and an
  ensemble MCMC sampler; the inversion-filter comparison; the ensemble
  Rauch--Tung--Striebel smoother remark.
- **Allowed support:** the EnKF estimation baseline of Sec. 14.4 and its
  Gaussian-closure character. **Forbidden:** accuracy conclusions from their
  artificial-data study without re-derivation.

### Lubik and Schorfheide (2003)

- **Identifier/status:** *Computing Sunspot Equilibria in Linear Rational
  Expectations Models*; DOI `10.1016/S0165-1889(02)00153-7`; *JEDC* 28(2),
  273--285; `FOUNDATIONAL` for Sec. 14.3; OpenAlex W2045998004, not
  retracted.
- **Local full text:** `.../DSGE/solving rational expectation/indeterminate
  equilibria/Computing sunspot equilibria in linear rational expectations
  models Lubik(03).pdf` (technical appendix also local).
- **Inspected anchors:** abstract and Sec. 1: under indeterminacy the
  endogenous forecast errors are not uniquely determined and sunspot shocks
  enter through them with estimable effects.
- **Allowed support:** the sunspot-completion machinery named in Sec. 14.3.
  **Forbidden:** transfer of the linear construction to the piecewise model
  without new derivation.

### van der Merwe, Doucet, de Freitas, and Wan (2000)

- **Identifier/status:** *The Unscented Particle Filter*; NIPS 13, 584--590;
  `FOUNDATIONAL` for UKF-proposal particle filtering; OpenAlex W2124156864,
  1,419 citations, not retracted. The inspected copy is the technical-report
  layout and lists de Freitas second; the proceedings index lists Doucet
  second, and the reference list follows the proceedings order.
- **Local full text:** `.../bayesian and SSM/nonlinear SSM/particle/The
  unscented particle filter Merwe(00).pdf`.
- **Inspected anchors:** title/abstract and construction: UKF-generated
  Gaussian proposals inside a particle filter.
- **Allowed support:** lineage attribution for Secs. 4.3.6 and 13.4.
  **Forbidden:** performance claims.

### Aruoba, Cuba-Borda, and Schorfheide (2018)

- **Identifier/status:** *Macroeconomic Dynamics Near the ZLB: A Tale of Two
  Countries*; DOI `10.1093/restud/rdx027`; *Review of Economic Studies*
  85(1), 87--118; `DIRECT_METHOD` for the sunspot-completion program; full
  text `SOURCE_BLOCKED` (not in the local corpus); OpenAlex W4296974713, not
  retracted.
- **Allowed support:** named, metadata-verified instance of a Markov-sunspot
  ZLB model estimated by particle filtering, as used in Sec. 14.3; the
  characterization is corroborated by Holden's (2023) inspected description
  of that literature. **Forbidden:** any claim requiring their model details
  or empirical results; full-text audit is required before the dsge_hmc
  benchmark copies their regime structure.

## Implementation evidence

- Dynare OccBin model inspected at
  `/home/ubuntu/workspace/DynareMCP/docs/AIpostdoc/literature/bgs_foundation_corpus_2026_05_28/03_code_and_replication_sources/dynare_occbin_example.mod`.
  Regime-tagged equations and constraint timing support implementation behavior.
- Dynare PKF source inspected at `/usr/lib/dynare/matlab/+occbin/kalman_update_algo_3.m`
  and `kalman_update_algo_1.m`. The files support update order and
  `regime_history` iteration, not mathematical validity beyond the papers.
- MacroFinance softplus implementation inspected at
  `/home/ubuntu/workspace/MacroFinance/two_currency_double_zlb_math.py`. It is
  classified as an approximation baseline.
- MacroFinance contract inspection (2026-08-19) for survey Sec. 13:
  `two_currency_double_zlb_math.py` (quadrature, softplus map, DNS forward),
  `two_currency_double_zlb_contract.py` (state/parameter names, FX identity),
  `two_currency_double_zlb_target.py` (VAR(1) transition construction,
  SR-UKF route, variant charts), `two_currency_double_zlb_fixtures.py`
  (per-country decay/bound/alpha constants, binding-fraction tracking), and
  the `dz5` identification/alpha-counterfactual/NeuTra modules. These support
  implementation-behavior statements only.
- MacroFinance solver-branch evidence for survey Sec. 15:
  `ccma_g_v7_pade_frechet_diagnostic.py`, whose module docstring records the
  fixed-scaling Pad\'e(13) differentiation policy and the branch-discontinuity
  caveat at norm thresholds.
- `/home/ubuntu/workspace/dsge_hmc` was checked again on 2026-08-19 and still
  does not exist at that path. **Correction (release revision 2026-08-20):**
  the `dsge_hmc` package exists at `/home/ubuntu/workspace/python/src/dsge_hmc`;
  verified anchors are `models/bgs_restricted_surface_generated.py:225,228`
  (notional rule `rn`; placeholder residual `r = rn`),
  `models/bgs_restricted_surface_tf_coefficients.py:17` (rows tagged
  `OBC_ZLB_NO_RUN_GUARD`), and the master program
  `/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-master-program-v2-2026-07-09.md`
  (no active OBC/ZLB logic; evidence contract excludes OBC/ZLB estimation).
  Survey Secs. 14.1--14.4 are pedagogical; Secs. 14.5--14.6 are the
  source-anchored BGS contract.
- Pericoli and Taboga (2018), Banca d'Italia Temi di discussione 1189,
  inspected as local full text on 2026-08-20 from the recovered
  ResearchAssistant corpus: framework Sec. 2 (SRTSM eq. (6)); posterior and
  data augmentation Secs. 4--5 (blockwise random-walk Metropolis over
  parameters and the full latent path, no moment-closure filter); the
  "nearly exact" neural-network pricing surrogate Introduction p. 6 and
  Sec. 9 (sub-basis-point test error). Supports the survey Sec. 13.2
  frontier narrowing. The recovered PDF is the 2018 working paper; the later
  journal version was not verifiable from the local corpus.
