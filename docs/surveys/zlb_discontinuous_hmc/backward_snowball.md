# Backward-Snowball Ledger

Method: related-work, introduction, method-comparison, and reference sections of
the inspected seed papers were read. The table records references that could
materially change the nonlinear-ZLB project; peripheral application citations
were not copied mechanically.

| Seed | Referenced work | Class | Action | Reason |
|---|---|---|---|---|
| Afshar--Domke; Nishimura et al. | Duane et al. (1987), *Hybrid Monte Carlo* | FOUNDATIONAL | context only | Historical origin of HMC; Neal supplies the implementation-level exposition used here. |
| Afshar--Domke; Nishimura et al.; Zhou | Neal (2011), *MCMC Using Hamiltonian Dynamics* | FOUNDATIONAL | cite | Standard Hamiltonian, leapfrog, reversibility, volume preservation, and Metropolis correction. |
| Afshar--Domke; Zhou | Pakman and Paninski, exact/auxiliary-variable HMC for truncated and binary distributions | COMPETITOR | omit with reason | Important constrained/binary predecessor, but it does not solve an endogenous nonlinear ZLB boundary; retained as reviewer risk. |
| Nishimura et al. | Livingstone et al. (2019), kinetic-energy choice in HMC | BACKGROUND | omit | Supports kinetic-energy design generally; not needed to derive the paper-specific Laplace update. |
| Zhou | discontinuous HMC (Nishimura et al.) | DIRECT_METHOD | cite | Direct predecessor and comparator. |
| Zhou | reflection/refraction HMC (Afshar--Domke) | DIRECT_METHOD | cite | Direct event-dynamics predecessor. |
| Zhou | probabilistic-path HMC for phylogenetic trees | COMPETITOR | omit with reason | Specialized tree topology kernel; no direct state-space or ZLB transfer. |
| Torgander et al. | Maddison, Mnih, and Teh (2017), Concrete distribution | FOUNDATIONAL | considered, not separately cited | The survey cites the HMC paper's instantiated construction; a future relaxation implementation should cite the original Concrete paper too. |
| Torgander et al. | Jang, Gu, and Poole (2017), Gumbel-softmax | FOUNDATIONAL | considered, not separately cited | Same scope as preceding row; no exact hard-regime result. |
| Chaari et al. | Moreau proximity operator and proximal MCMC literature | BACKGROUND | omit | Convex nonsmooth machinery is a comparator, not the central ZLB target. |
| Guerrieri--Iacoviello | Christiano and Fisher (2000), occasionally binding constraints | FOUNDATIONAL | omit with reason | Earlier solution context; OccBin is the direct benchmark implemented in Dynare. |
| Guerrieri--Iacoviello | Judd (1992) and projection/global solution methods | COMPETITOR | omit with reason | Global solution methods matter for the economic solution, but the survey concerns inference kernels after a solution map is specified. |
| Guerrieri--Iacoviello | Fernández-Villaverde et al., *Nonlinear Adventures at the Zero Lower Bound* | EMPIRICAL_EXAMPLE | retain as omission risk | Demonstrates economically meaningful nonlinear ZLB effects; technical inference method is not the direct target of this survey. |
| Guerrieri--Iacoviello | Holden and Paetz, OBC solution by anticipated shocks | COMPETITOR | retain as omission risk | Alternative piecewise-linear OBC solver; should enter a later solver-comparison survey. |
| Giovannini et al. | Guerrieri and Iacoviello (2015) | FOUNDATIONAL | cite | PKF calls an OccBin-style regime solver. |
| Giovannini et al. | Cuba-Borda et al., likelihood evaluation with OBCs | DIRECT_METHOD | inspect next | Direct inference comparator; full technical text was not separately audited within this bounded survey. |
| Giovannini et al. | Mavroeidis, ZLB and estimation accuracy | EMPIRICAL_EXAMPLE | omit with reason | Motivates inference bias but does not define the selected kernel. |
| Aruoba et al. | Andrieu, Doucet, and Holenstein (2010) | FOUNDATIONAL | cite | Exact particle-MCMC identity. |
| Aruoba et al. | Pitt and Shephard (1999), auxiliary particle filter | FOUNDATIONAL | considered | Proposal design background already derived in project notation. |
| Aruoba et al. | Guerrieri and Iacoviello (2015) | FOUNDATIONAL | cite | Main PLC/OccBin benchmark. |
| Aruoba et al. | Gust et al., U.S. macroeconomic dynamics at the ZLB | EMPIRICAL_EXAMPLE | omit with reason | Application motivation, not algorithm support. |
| Andrieu et al. | Andrieu and Roberts (2009), pseudo-marginal approach | FOUNDATIONAL | considered | The PMCMC paper gives the state-space extended construction used here. |
| Andrieu et al.; Lindsten et al. | Del Moral, Doucet, and Jasra, SMC samplers | FOUNDATIONAL | context only | General Feynman--Kac/SMC theory; not needed for the bounded algorithm derivation. |
| Andrieu et al.; Lindsten et al. | Doucet and Johansen, particle-filter tutorial | SURVEY_OR_TUTORIAL | omit as theorem support | Useful orientation but primary PMCMC/PGAS papers support the claims. |
| Lindsten et al. | Whiteley; Lindsten and Schön, particle Gibbs with backward simulation | COMPETITOR | omit with reason | Direct smoothing competitor; PGAS was selected because it gives the single-forward-sweep ancestor rule. |
| Alenlöv et al. | Andrieu and Roberts (2009); Andrieu et al. (2010) | FOUNDATIONAL | cite PMCMC paper | Establishes pseudo-marginal extended-target background. |
| Alenlöv et al. | correlated pseudo-marginal methods | COMPETITOR | omit | Useful for likelihood-noise correlation but not an HMC discontinuity solution. |
| Ścibior--Wood; Corenflos et al. | Poyiadjis, Doucet, and Singh (2011) | FOUNDATIONAL | cite | Direct score-estimator authority. |
| Ścibior--Wood | Le et al.; Maddison et al.; Naesseth et al., variational SMC gradients | COMPETITOR | omit with reason | Optimize variational objectives and often stop resampling gradients; not exact posterior HMC. |
| Corenflos et al. | ensemble-transform and optimal-transport particle filters | FOUNDATIONAL | considered | Supplies transport-resampling lineage; the ICML paper's own propositions are the checked source. |
| Teixeira et al. (2026-08-19 seed) | Simon and coauthors, Kalman truncation via pdf reshaping (their refs. [24,25]) | DIRECT_METHOD | omit with reason | Linear-KF ancestor of the truncated-UKF variants; Section 4.3 derives the one-sided truncated moments directly. |
| Teixeira et al. (2026-08-19 seed) | Vachhani et al. and Kolås et al., constrained recursive/UKF estimators | COMPETITOR | omit with reason | Alternative constrained-update designs already represented by the paper's Table 1 taxonomy. |
| Wang et al. (2026-08-19 seed) | van der Merwe, Doucet, de Freitas, and Wan, unscented particle filter | FOUNDATIONAL | considered | The survey derives the proposal weights (52)--(53) from importance sampling; UPF adds no branch structure. |
| Wang et al. (2026-08-19 seed) | García-Fernández et al., truncated UKF | DIRECT_METHOD | cite (metadata) | Confirms the truncated-UKF lineage used for family orientation. |
| Zhang et al. (2026-08-19 seed) | Blom and Bar-Shalom (1988), interacting multiple model | FOUNDATIONAL | cite (metadata) | IMM lineage for the stochastic-regime construction in Sec. 4.3.5. |
| Alspach--Sorenson (metadata only) | Sorenson and Alspach (1971), recursive Gaussian-sum estimation | FOUNDATIONAL | omit with reason | Predecessor of the 1972 paper; both pay-walled; historical attribution only. |
| Opschoor--van der Wel (2026-08-19 seed) | Krippner (2013 JMCB letter); Wu--Xia (2016); Christensen--Rudebusch (2015, 2016); Kim--Singleton (2012); Gorovoi--Linetsky (2004) | DIRECT_METHOD / FOUNDATIONAL | Krippner/Wu--Xia/C-R 2015 cited and inspected; others omit with reason | The shadow-rate approximation lineage; the 2013 letter, 2016 C-R paper, and Kim--Singleton remain uninspected omission risks. |
| Opschoor--van der Wel (seed) | Swanson and Williams (2014), yield sensitivity near the bound; Grisse (2023), lower-bound uncertainty | EMPIRICAL_EXAMPLE | omit with reason | Empirical motivation for smooth transition; not needed for the derivations. |
| Kim--Priebsch (seed) | Ichiue and Ueno (2007); Kim and Singleton (2012); Christensen and Rudebusch (2013 draft) | DIRECT_METHOD | omit with reason | Named as prior EKF users; superseded for this survey's purpose by the inspected 2015 C-R paper. |
| Kim--Priebsch (seed) | Julier, Uhlmann, and Durrant-Whyte (1995); Julier and Uhlmann (1996) | FOUNDATIONAL | context only | UKF origin; the survey derives its sigma-point usage independently. |
| Krippner (seed) | Black (1995); Gorovoi and Linetsky (2004); Bomfim (2003); Ueno, Baba, and Sakurai (2006) | FOUNDATIONAL / BACKGROUND | Black cited (metadata); rest omit with reason | Black-GATSM tractability history; not needed beyond orientation. |
| Holden 2023 (seed) | Benhabib, Schmitt-Grohe, and Uribe (2001a,b); Schmitt-Grohe and Uribe (2012); Mertens and Ravn (2014); Aruoba, Cuba-Borda, and Schorfheide (2018); Cochrane (2011) | COMPETITOR / DIRECT_METHOD | ACS cited (metadata); rest omit with reason | Steady-state-multiplicity and sunspot literature contrasted with within-steady-state multiplicity; ACS is the estimation-relevant instance. |
| Boehl--Strobel (seed) | Evensen (1994, 2009), ensemble Kalman filter; Herbst and Schorfheide (2016, 2019), adapted/tempered particle filters; Raanes (2016), ensemble RTS smoother | FOUNDATIONAL / COMPETITOR | omit with reason | EnKF origin and PF competitors; named in Sec. 14.4 through the inspected paper only. |
| Lubik--Schorfheide (seed) | Sims (2000/2002) Gensys; Blanchard and Kahn (1980) | FOUNDATIONAL | context only | Linear RE solution machinery presupposed by every solver in Sec. 14.1. |

## Unresolved backward-snowball work

The strongest uninspected direct candidates are the Cuba-Borda likelihood paper
and the Holden/Paetz OBC solver family. They can affect baseline breadth, but
neither removes the support distinction derived in (5a). They are therefore
recorded as expansion work rather than silently treated as covered.
