# Phase P2 result: source-grounding and literature synthesis

## Date

2026-05-15

## Governing plan

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-plan-2026-05-14.md`

## Scope and safety state

Branch at phase start:

- `main`

Worktree classification:

- Unrelated student-baseline files remain dirty and were not edited.
- This phase created only this DPF reviewer-grade result artifact.

Local checks run:

- Extracted DPF chapter citation usage from:
  - `docs/chapters/ch19_particle_filters.tex`
  - `docs/chapters/ch19b_dpf_literature_survey.tex`
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`
- Verified that every DPF citation key used by those chapters has an entry in
  `docs/references.bib`.
- Queried ResearchAssistant in read-only/offline mode.

ResearchAssistant status:

- Workspace: `/home/ubuntu/python/ResearchAssistant`.
- Mode: read-only.
- Offline mode: enabled.
- Review list: empty.
- Searches for SMC, EDH/LEDH/PF-PF, differentiable resampling/OT/Sinkhorn, and
  HMC/pseudo-marginal sources returned no local paper summaries.
- Parser tool matrix reports `pdftotext` available, but this phase did not
  ingest sources because write/fetch workflows were not authorized and the
  current MCP mode is read-only.

Implication:

- The current revision may use the bibliography and existing chapter/source-map
  spine, but it must not claim ResearchAssistant-reviewed source support for
  the DPF literature.  Primary-source roles below are local bibliographic roles
  pending future paper-level review.

## Source-role table

| Key | Source family | Local availability | What it may support in this revision | What it must not be used to claim |
|---|---|---|---|---|
| `gordon1993novel` | Bootstrap particle filter | Bibliography entry; cited in SMC/debug chapters; no RA record | Historical bootstrap/SIR particle-filter origin; nonlinear/non-Gaussian particle filtering motivation. | Does not prove BayesFilter DPF correctness, HMC suitability, or banking validity. |
| `doucet2001sequential` | SMC foundations | Bibliography entry; cited in SMC/PF-PF/debug chapters; no RA record | Sequential importance sampling, resampling, particle approximations, likelihood-estimator background. | Does not by itself validate differentiable relaxations or learned transport. |
| `chopin2020introduction` | Modern SMC text | Bibliography entry; cited in PF-PF/debug chapters; no RA record | General SMC notation, proposal/weight interpretation, degeneracy diagnostics. | Does not establish new BayesFilter-specific DPF claims. |
| `bengtsson2008curse` | High-dimensional degeneracy | Bibliography entry; cited in SMC/debug chapters; no RA record | Weight collapse/high-dimensional particle-filter warning. | Does not imply every structural model is impossible or every ESS drop is invalidity. |
| `snyder2008obstacles` | High-dimensional degeneracy | Bibliography entry; cited in SMC/debug chapters; no RA record | Obstacles to high-dimensional particle filtering; dimensionality warning. | Does not justify DPF correctness; only motivates caution and diagnostics. |
| `daumhuang2008` | Particle flow/EDH | Bibliography entry; cited in debugging crosswalk; no RA record | Particle-flow motivation and EDH-style flow construction. | Must not be used to say raw EDH is exact for nonlinear structural models or HMC-ready. |
| `li2017particle` | Invertible particle flow / PF-PF | Bibliography entry; cited in PF-PF/debug chapters; no RA record | Invertible flow, proposal correction, Jacobian/log-det accounting for particle-flow particle filters. | Does not remove finite-particle variance, flow-discretization error, or HMC target checks. |
| `hu2021particle` | Particle-flow applications | Bibliography entry; cited in debugging crosswalk; no RA record | Particle-flow implementation/application warnings, high-dimensional flow context. | Does not validate BayesFilter banking models or learned/HMC variants. |
| `zhumurphyjonschkowski2020` | Differentiable resampling | Bibliography entry; cited in resampling/debug chapters; no RA record | Soft/differentiable resampling motivation; pathwise differentiability problem. | Does not prove preservation of categorical resampling law or exact posterior target. |
| `corenflos2021differentiable` | Differentiable PF via EOT | Bibliography entry; cited in resampling/learned/debug chapters; prior literature gate accepted it for differentiable PF frontier only. | Entropy-regularized OT resampling as differentiable approximate PF; teacher map for learned OT. | Does not certify generic pseudo-marginal HMC, arbitrary differentiable resampling, banking deployment, or learned-map posterior preservation. |
| `villani2003topics` | OT foundations | Bibliography entry; cited in resampling/debug chapters; no RA record | Coupling formulation, transport cost, unregularized OT object. | Does not justify entropy regularization, finite Sinkhorn errors, or HMC target claims. |
| `reich2013nonparametric` | Ensemble transform / OT filtering relation | Bibliography entry; cited in resampling chapter; no RA record | Transport/equal-weighting perspective for particle or ensemble transforms. | Does not make the transport projection equal to categorical resampling. |
| `cuturi2013sinkhorn` | Entropic OT / Sinkhorn | Bibliography entry; cited in resampling/learned/debug chapters; no RA record | Entropic regularization, Sinkhorn scaling motivation, computational OT route. | Does not by itself justify small-epsilon stability, finite-iteration accuracy, or posterior invariance. |
| `peyre2019computational` | Computational OT | Bibliography entry; cited in resampling/learned/debug chapters; no RA record | OT/EOT background, numerical formulation, computational perspective. | Does not replace local derivation of BayesFilter EOT symbols or solver residuals. |
| `schmitzer2019stabilized` | Stabilized Sinkhorn | Bibliography entry; cited in resampling/debug chapters; no RA record | Log-domain/stabilized scaling warning for small epsilon or numerical underflow. | Does not validate a specific implementation unless residuals and stability are tested. |
| `zaheer2017deep` | DeepSets | Bibliography entry; cited in learned/debug chapters; no RA record | Permutation-invariant/equivariant set-processing architecture family. | Does not imply target correctness, posterior preservation, or out-of-distribution generalization. |
| `lee2019set` | Set Transformer | Bibliography entry; cited in learned/debug chapters; no RA record | Attention-based set architecture option and symmetry-aware learned maps. | Does not turn a learned map into an exact OT solver or validated filter. |
| `neal2011mcmc` | HMC foundations | Bibliography entry; cited in HMC/debug chapters; prior literature gate accepted HMC foundations through Betancourt but not this record specifically. | Hamiltonian dynamics, scalar target/potential, gradient-based transition logic, Metropolis correction context. | Does not validate any BayesFilter likelihood construction or DPF target. |
| `betancourt2017conceptual` | HMC conceptual foundations | Bibliography entry; cited in debugging crosswalk; prior literature gate accepted for HMC foundations. | Target geometry, Hamiltonian lift, energy diagnostics, HMC conceptual discipline. | Does not validate DPF likelihood implementation or banking suitability. |
| `andrieu2009pseudo` | Pseudo-marginal MCMC | Bibliography entry; cited in HMC/debug chapters; no RA record | Distinction between exact posterior inference with unbiased likelihood estimators and ordinary likelihood evaluation. | Must not be conflated with differentiable surrogate-gradient HMC. |
| `andrieu2010particle` | Particle MCMC / PMCMC | Bibliography entry; cited in SMC/HMC/debug chapters; no RA record | Particle likelihood estimator role in PMCMC; formal particle-system context. | Does not justify smooth HMC through a randomized or relaxed estimator without target analysis. |
| `greydanus2019hamiltonian` | Hamiltonian neural networks | Bibliography entry; cited in HMC/debug chapters; no RA record | Surrogate/HNN line as a comparison for dynamics or geometry acceleration. | Does not support replacing the target likelihood or ignoring accept/reject target consistency. |

## Source-gap register

| Gap | Affected claim family | Current disposition |
|---|---|---|
| No local RA records for DPF sources | All source-facing claims | Treat as bibliography-spine only; P4-P9 must derive locally and cite only for provenance. |
| EDH/LEDH primary details not locally reviewed through RA | Particle-flow derivations | P4 must be self-contained and conservative; no raw-flow exactness outside special cases. |
| PF-PF source not locally reviewed through RA | Proposal correction and log-det claims | P5 must derive change-of-variables and log-det formulas locally. |
| Differentiable resampling source not locally reviewed through RA | Soft resampling and bias | P6 must derive soft-resampling mean/bias statements locally. |
| OT/Sinkhorn sources not locally reviewed through RA | EOT scaling and numerical warnings | P6 must include local derivation and numerical caveats; cite sources for provenance only. |
| Learned OT has no source proving BayesFilter posterior preservation | Learned transport and banking claims | P7 must frame learned OT as a layered surrogate with evidence requirements. |
| HMC/pseudo-marginal sources not fully reviewed through RA in this workspace | HMC target status | P8 must derive the target contract locally; use prior literature-gate note only for broad HMC foundations. |
| DSGE/MacroFinance DPF-HMC validation absent | Banking suitability | P8/P13 must record validation as missing; no bank-facing validity claim. |

## Source-to-claim map

| Claim ledger IDs | Permitted source support | Required local derivation or downgrade |
|---|---|---|
| P1-C01--P1-C06 | `gordon1993novel`, `doucet2001sequential`, `chopin2020introduction`, `bengtsson2008curse`, `snyder2008obstacles`, `andrieu2010particle` | P3 must derive recursion, SIS ratios, and likelihood-estimator status locally. |
| P1-C07--P1-C12 | `daumhuang2008`, `li2017particle`, `hu2021particle` | P4 must derive homotopy, continuity equation, EDH/LEDH formulas, exact special case, and stiffness limits. |
| P1-C13--P1-C16 | `li2017particle`, `doucet2001sequential`, `chopin2020introduction` | P5 must derive target/proposal density, change of variables, corrected weights, and log-det ODE locally. |
| P1-C17--P1-C20 | `zhumurphyjonschkowski2020`, `corenflos2021differentiable`, `villani2003topics`, `reich2013nonparametric`, `cuturi2013sinkhorn`, `peyre2019computational`, `schmitzer2019stabilized` | P6 must derive categorical discontinuity, soft bias, OT/EOT formulation, Sinkhorn scaling, and solver residual status. |
| P1-C21--P1-C22 | `corenflos2021differentiable`, `cuturi2013sinkhorn`, `peyre2019computational`, `zaheer2017deep`, `lee2019set` | P7 must state teacher/student object, training distribution, residual hierarchy, and "does not prove" limits. |
| P1-C23--P1-C28 | `neal2011mcmc`, `betancourt2017conceptual`, `andrieu2009pseudo`, `andrieu2010particle`, `greydanus2019hamiltonian` | P8 must derive target contract and downgrade banking/HMC claims to evidence-gated status. |
| P1-C29--P1-C30 | All above sources through revised chapters | P9 must make diagnostics equation-indexed; source citations cannot be validation claims. |

## Literature synthesis insertion plan

### `ch19_particle_filters.tex`

Insert or expand a source paragraph after the bootstrap/SIR definition:

- Gordon et al. provide the bootstrap-filter origin.
- Doucet et al. and Chopin/Papaspiliopoulos provide the SMC proposal/weight
  framework.
- Andrieu et al. explain why unbiased likelihood estimators matter for particle
  MCMC.
- Bengtsson et al. and Snyder et al. support high-dimensional degeneracy
  warnings.

Required local limit:

- These sources do not make a realized particle path differentiable and do not
  validate DPF-HMC.

### `ch19b_dpf_literature_survey.tex`

Insert or expand a source paragraph before the EDH derivation:

- Daum-Huang motivates the EDH particle-flow construction.
- Li-Coates is relevant because later PF-PF treats the flow as an invertible
  proposal map.
- Hu/van Leeuwen supports implementation and high-dimensional particle-flow
  caution.

Required local limit:

- Raw particle flow is not a corrected posterior sampler outside the explicitly
  derived special cases.

### `ch19c_dpf_implementation_literature.tex`

Insert or expand a source paragraph near the PF-PF target/proposal definitions:

- Li-Coates supports the proposal-corrected/invertible particle-flow viewpoint.
- SMC texts support the general importance-ratio logic.

Required local limit:

- Proposal correction restores the proposal-to-target density ratio for the
  stated proposal; it does not remove finite-particle or numerical error.

### `ch32_diff_resampling_neural_ot.tex`

Insert or expand source paragraphs:

- Zhu/Murphy/Jonschkowski for the differentiable-resampling problem and soft
  resampling.
- Villani and Reich for transport/coupling and ensemble-transform perspective.
- Cuturi and Peyre/Cuturi for entropic OT and Sinkhorn computation.
- Schmitzer for stabilized scaling warnings.
- Corenflos et al. for differentiable PF via EOT.

Required local limit:

- OT, EOT, and finite Sinkhorn are different mathematical objects from
  categorical resampling.

### `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

Insert or expand source paragraphs:

- Corenflos et al. for the EOT teacher map.
- Cuturi/Peyre for the computational OT teacher.
- Zaheer and Lee et al. for set-map architectures.

Required local limit:

- Architecture symmetry is necessary for a learned set map but not sufficient
  for posterior or HMC target correctness.

### `ch19e_dpf_hmc_target_suitability.tex`

Insert or expand source paragraphs:

- Neal and Betancourt for HMC target/gradient discipline.
- Andrieu-Roberts and Andrieu-Doucet-Holenstein for pseudo-marginal/PMCMC
  distinctions.
- Greydanus et al. only as a comparison to surrogate dynamics, not as a
  likelihood-target replacement.

Required local limit:

- Pseudo-marginal MCMC is not differentiable-surrogate HMC; learned/surrogate
  dynamics are not target correction unless the true target is used in the
  correction.

### `ch19f_dpf_debugging_crosswalk.tex`

The crosswalk should cite source families only after equation-local diagnostics:

- SMC sources for ESS/log-weight diagnostics.
- Particle-flow/PF-PF sources for flow and correction diagnostics.
- OT/Sinkhorn sources for marginal residuals and numerical stabilization.
- HMC sources for value-gradient target diagnostics.

Required local limit:

- A diagnostic validates only the equation or failure mode it tests.

## Claims blocked by source gaps

The following claims must remain blocked or weakened unless future source
review or experiments provide stronger evidence.

1. Any statement that EDH/LEDH is exact for generic nonlinear filtering.
2. Any statement that PF-PF is HMC-ready or production-ready for nonlinear
   structural models.
3. Any statement that soft resampling preserves the categorical resampling law.
4. Any statement that EOT/Sinkhorn targets the original categorical-resampling
   posterior without qualification.
5. Any statement that finite Sinkhorn is the same object as exact EOT.
6. Any statement that learned OT preserves the teacher posterior or original
   posterior because the map is permutation-equivariant.
7. Any statement that low map residual implies low posterior error.
8. Any statement that DPF-HMC for nonlinear DSGE or MacroFinance models has
   been validated in this repository.
9. Any bank-facing recommendation stronger than "research candidate requiring
   target-status, numerical, posterior-comparison, and governance evidence."

## P2 audit result

Audit rules checked:

- Every DPF citation key in the current chapter block exists in
  `docs/references.bib`.
- Major source families have role notes and limitations.
- ResearchAssistant unavailability as source evidence is recorded.
- HMC and banking claims are not supported by analogy; they are marked as
  evidence-gated.
- Source gaps are explicit and mapped to later derivation obligations.

Veto diagnostics:

- No P2 veto fired.
- Caution: because source support is bibliography-spine rather than
  ResearchAssistant-reviewed, P4-P9 must keep derivations self-contained and
  must not strengthen claims solely by citation.

Exit decision:

- P2 passes for controlled rewrite execution.
- Superseded next-phase note: this result originally pointed to P4 EDH/LEDH
  derivation expansion.  The governing master was later tightened: P3 is now
  the next baseline gate, existing P4/P5 work is provisional, and P6 is blocked
  until P3 plus P4/P5 reconciliation pass.
