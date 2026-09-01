# Omitted-Paper and Reviewer-Risk Register

`Acceptable` means omission does not invalidate the narrow algorithmic survey.
It does not mean the paper is unimportant. `Expansion required` marks work that
must be inspected before a later implementation or economic claim crosses the
stated boundary.

| Candidate or family | Why plausible | Reason omitted now | Reviewer risk | Status |
|---|---|---|---|---|
| Holden (2016), computation with occasionally binding constraints | Major alternative OBC solver | **Resolved 2026-08-19:** local full text inspected; news-shock representation cited in survey Sec. 14.1 | Closed at survey level; solver-engine choice still open | Audited for the survey; engine comparison remains implementation work |
| Holden (2022/2023), existence and uniqueness for OBC models | Directly bears on target uniqueness | **Resolved 2026-08-19:** local full text inspected; LCP/P-matrix results and worked example carried into survey Secs. 14.1--14.2 | Closed for the survey's uniqueness treatment | Audited; the published RESTAT 105(6) version is the cited record |
| Boehl (2022), efficient OBC solution/computation | Recent solver alternative | **Partially resolved 2026-08-19:** local full text present, abstract-level solution operator inspected and cited in Sec. 14.1 | Method internals not audited | Deeper audit required before choosing the dsge_hmc solution engine |
| Cuba-Borda, Guerrieri, Iacoviello, and Zhong, likelihood evaluation with OBCs (inversion-filter line) | Direct inference comparator cited by PKF/Aruoba and by Boehl--Strobel | **Partially resolved 2026-08-20:** metadata-cited in survey Sec. 14.4 and the reference list as a named baseline arm | Primary-source inspection still required | Expansion required before empirical baseline freeze |
| Fernández-Villaverde et al., *Nonlinear Adventures at the ZLB* | Foundational nonlinear economic effects | Economic solution/application rather than sampler derivation | Readers may expect it in a broad ZLB survey | Acceptable for this algorithm-focused version; add in a model-economics companion survey |
| Gust et al. and other estimated ZLB DSGE applications | Empirical benchmark lineage | Do not derive a general discontinuous HMC kernel | Omission could make the survey seem detached from macro evidence | Acceptable now; required for application design |
| Pakman and Paninski exact/truncated/binary HMC papers | Direct constrained/discrete HMC predecessors | Afshar and Nishimura provide the checked event/embedding methods used | A computation reviewer may request fuller genealogy | Acceptable with explicit mention in backward ledger |
| Probabilistic-path HMC for phylogenetic trees | Mixed topology HMC competitor cited by Zhou | Specialized state space and proposal mechanism | Low for ZLB, moderate for mixed-HMC history | Acceptable |
| Bouncy particle, Zig-Zag, and other piecewise-deterministic samplers | Can cross nonsmooth/discrete structures | Different nonreversible process family; no inspected ZLB likelihood treatment | Could supply an alternative to event HMC | Expansion required only if HMC event handling fails |
| Reversible-jump and involutive MCMC | Natural joint state/regime transformations | Broad trans-dimensional framework beyond current fixed-dimension ZLB path | Relevant if regimes change dimension or require deterministic transforms | Expansion required if the chosen solution representation changes dimension |
| Particle Gibbs with backward simulation | Direct PGAS competitor | PGAS selected for a one-forward-sweep derivation | Could mix differently under severe degeneracy | Acceptable for first implementation; include in performance comparison later |
| SMC2 and particle learning methods | Parameter/state alternatives to PMMH | Survey focuses on HMC and exact particle-MCMC authority | May be preferable for online/sequential applications | Acceptable for offline posterior goal |
| Nemeth et al. linear-cost particle score estimators | Follow-up to Poyiadjis | Poyiadjis and Ścibior suffice to expose score-vs-exact-force issue | Could materially improve gradient cost | Expansion required before choosing a score estimator |
| Chen et al. overview of differentiable particle filters | Recent survey/tutorial | Secondary source cannot support theorem claims | Useful for coverage and newer methods | Acceptable; consult during implementation search refresh |
| New differentiable resampling papers after 2021 | Active method family | Bounded public search and local seeds did not establish exhaustive 2022--2026 coverage | Recent omissions are the largest literature-completeness risk | Expansion required before implementation default selection |
| Piecewise-smooth HMC beyond affine boundaries | Directly relevant to implicit ZLB events | No primary paper with a checked general nonlinear-boundary proof was established | The affine Afshar proof cannot be silently generalized | Expansion required before event-HMC implementation |
| Markov-switching DSGE filtering and mixture Kalman filters | Relevant when regime probabilities are stochastic | The user's primary case is endogenous ZLB complementarity | Important if the project deliberately adopts a stochastic regime law | Expansion required for target (73), not target (71) |
| Simon's constrained-Kalman survey and the pdf-truncation KF line | Linear ancestors of every truncated-UKF variant in Sec. 4.3 | Section 4.3 derives the truncated moments directly; the ancestors add history, not new ZLB structure | A filtering reviewer may expect the canonical constrained-estimation survey citation | Acceptable now; add to an implementation-phase methods appendix |
| van der Merwe, Doucet, de Freitas, and Wan, unscented particle filter | Standard UKF-proposal-in-PF reference behind Sec. 4.3.6 | Weights (52)--(53) are derived directly from importance sampling | Low; Wang et al. (2020) is the inspected representative | Acceptable |
| Sorenson and Alspach (1971) recursive Gaussian-sum papers | Historical predecessor of the cited 1972 paper | The metadata-cited 1972 paper suffices for attribution | Low | Acceptable |
| Adaptive Gaussian-splitting filters for nonlinear boundaries | Sec. 4.3.2 names component splitting for a curved switching surface | No primary splitting paper was inspected in this bounded pass | Moderate: the splitting recommendation currently rests on project reasoning only | Expansion required before implementing a splitting-based filter |
| Kim and Singleton (2012); Ichiue and Ueno (2007); Gorovoi and Linetsky (2004) | Named shadow-rate predecessors in inspected papers | **Updated 2026-08-21:** Gorovoi--Linetsky is now metadata-cited in the survey's test-model ladder as the closed-form analytic anchor (Tier 3); the other two remain omitted | Low-moderate for a term-structure referee | Gorovoi--Linetsky cited (metadata); full-text audit before any Japan/two-factor empirical comparison; others acceptable |
| Krippner (2013 JMCB letter) and the Krippner (2015) book | Published successors of the inspected DP2012/02 | The discussion-paper construction suffices for the Sec. 13 contract | A referee may prefer the published citation | Expansion at publication-facing revision |
| Christensen and Rudebusch (2016) and later shadow-rate refinements | Post-2015 estimation refinements | Bounded pass stopped at the 2015 JFEC paper | Moderate: newer estimation practice may differ | Expansion before empirical benchmark freeze |
| Swanson and Williams (2014) yield-sensitivity evidence | Empirical basis for the smooth-transition reading of alpha | Cited only through the inspected Opschoor--van der Wel discussion | Low | Acceptable |
| Post-2016 Tobit-filter extensions (multivariate, dynamic censoring) | Would refine the Sec. 13.4 noise-ordering discussion | Only the Geng et al. survey and Allik et al. metadata are in scope | Low for the contract, moderate for an implementation | Expansion if an exactly observed policy rate is added |
| Genz-type orthant algorithms and Botev-style truncated-normal samplers | Needed to implement cell probabilities and cell draws in (84)--(87) | **Partially resolved 2026-08-20:** metadata-cited in the survey (Genz--Bretz 2009; Botev 2017) as explicit admission conditions on the Sec. 13.7(v) cell-adapted route | Moderate at implementation time; inspection required before any exactness claim | Expansion required at Phase 2 implementation, with tested tolerances |
| Herbst and Schorfheide particle-filter book and tempered/adapted PF papers | Standard DSGE filtering practice named by Boehl--Strobel | Cited only through the inspected comparison | Moderate for dsge_hmc baselines | Expansion at dsge_hmc Phase 1--2 |
| Evensen (1994, 2009) EnKF origins; Raanes (2016) ensemble smoother | Origin of the Sec. 14.4 EnKF baseline | Named through the inspected Boehl--Strobel text | Low | Acceptable |
| Sims Gensys and Blanchard--Kahn linear RE solution papers | Presupposed by every Sec. 14.1 solver | Standard machinery outside the survey's derivation obligations | Low | Acceptable |

## Highest risks

Updated 2026-08-19. The Holden existence/uniqueness gap is closed at survey
level by the full-text audit behind Sections 14.1--14.2. The remaining
material gaps are: the inversion-filter/Cuba-Borda likelihood line and the
Aruoba--Cuba-Borda--Schorfheide model internals (metadata-cited only) on the
DSGE side; post-2015 shadow-rate estimation refinements on the term-structure
side; recent general-boundary event-HMC theory; and the computational
references (orthant algorithms, truncated-normal samplers) that Phase 2
implementation of (84)--(87) will need. The survey therefore still stops
short of selecting a nonlinear equilibrium solver, asserting that the affine
reflection/refraction proof extends to the MacroFinance boundary, or copying
any uninspected model's regime structure.

## Release-revision additions (2026-08-20)

| Candidate or family | Why plausible | Reason omitted/added now | Reviewer risk | Status |
|---|---|---|---|---|
| Pericoli and Taboga (2018 WP 1189), nearly exact Bayesian estimation of non-linear no-arbitrage term-structure models | Closure-free Bayesian shadow-rate comparator | **Resolved 2026-08-20:** local full text inspected (recovered PDF is the 2018 Banca d'Italia working paper); cited in survey Secs. 13.2 and reference list; the Sec. 13.2 frontier claim was narrowed accordingly | The later journal version could not be verified from the local corpus and is not cited | Audited at survey scope; journal-version metadata check at publication-facing revision |
| Holden and Paetz (2012), efficient simulation with inequality constraints | Original anticipated-shock route behind Holden (2016) | **Partially resolved 2026-08-20:** metadata-cited in survey Sec. 14.1 | Low after citation | Acceptable; inspect if the anticipated-shock engine is chosen |
| Deligiannidis, Doucet, and Pitt (2018), correlated pseudo-marginal | PMMH variance/mixing comparator | **Partially resolved 2026-08-20:** metadata-cited in survey Secs. 10.2 and 13.7(iii), explicitly not a PM-HMC differentiability fix | Low after citation | Expansion at implementation if PMMH variance binds |
| Nemeth, Fearnhead, and Mihaylova, linear-cost particle score | Score-estimator cost/variance alternative | Not newly cited; remains as above | Could materially improve gradient cost | Expansion required before choosing a score estimator (unchanged) |

The 2026-08-19 audit's remaining literature-table rows (Gust et al.;
Fernandez-Villaverde et al.; Atkinson--Richter--Throckmorton;
Keen--Richter--Throckmorton; Adjemian--Juillard extended path;
Dahlin--Lindsten--Schon; particle Gibbs with backward simulation;
Kim--Singleton/Ichiue--Ueno/Gorovoi--Linetsky lineage; published Krippner and
post-2015 refinements; reversible-jump/involutive MCMC and PDMP samplers;
2022--2026 differentiable-PF work) are implementation-phase or
publication-facing gates and retain their existing rows and statuses above.
