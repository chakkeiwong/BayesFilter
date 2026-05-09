# Phase M1 result: DPF literature survey and source-grounding audit

## Date

2026-05-09

## Purpose

This note records the first literature-grounding pass for the DPF monograph
rebuild.  The aim is not to finish the survey, but to determine whether the
source base is broad enough to support the next architectural phase and to make
explicit which mathematical obligations must govern the eventual exposition.

## Sources inspected in this pass

### BayesFilter/CIP local sources

- `~/python/latex-papers/CIP_monograph/chapters/ch26_differentiable_pf.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch20_hmc.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch21_advanced_hmc.tex`

### Student comparison sources

- `experiments/student_dpf_baselines/vendor/2026MLCOE/full_report.pdf`
- `experiments/student_dpf_baselines/vendor/2026MLCOE/README.md`
- `experiments/student_dpf_baselines/vendor/advanced_particle_filter/README.md`
- `experiments/student_dpf_baselines/vendor/advanced_particle_filter/docs/amortized_ot_operator.md`
- `experiments/student_dpf_baselines/vendor/advanced_particle_filter/notebooks/README.md`

### Tooling lane

- ResearchAssistant workspace initialized under `/tmp/ra-bayesfilter-dpf-rebuild`
- ResearchAssistant doctor run in offline mode
- MathDevMCP was not yet exercised on individual equations in this phase; this
  phase defines where that bounded audit work is needed later.

## Tool readiness result

### ResearchAssistant

Workspace initialization and doctor status passed in offline mode.

Observed status:
- core local lifecycle: ok
- metadata-only ingest: ok
- structured source inspection: ok
- PDF text ingest: warnings only, using `pdftotext`
- parser benchmark smoke: warnings only

Interpretation:
- ResearchAssistant is usable for local source-grounding, source inspection, and
  claim-audit organization.
- Extraction quality from PDFs should be treated cautiously, especially for
  mathematical documents.
- For the student PDF report, the parser result had low confidence and should be
  treated only as a pointer.  Human reading remains necessary.

### Student comparison inventory

Located local artifacts for both student tracks.  This is important because the
master program assumed they might still need to be fetched.

- `2026MLCOE`:
  - `full_report.pdf`
  - earlier part-1 report PDF
  - `README.md`
- `advanced_particle_filter`:
  - `README.md`
  - `requirements.txt`
  - notebook index
  - classical-filter, HMC-with-differentiable-resampling, and neural-amortized-OT notebooks
  - `docs/amortized_ot_operator.md`

Interpretation:
- The student comparison lane is now materially available, not merely planned.
- The advanced repo provides stronger implementation-facing evidence on OT,
  amortization, and HMC diagnostics than the MLCOE report.
- The MLCOE report provides a broader literature review arc and wider method
  spread, including particle flow, PF-PF, kernel PFF, and stochastic PF.

## Table 0: citation-key and source-identity audit

| Claim family | Primary paper or source | Alternate versions noted locally | Preferred citation-key action | Notes |
| --- | --- | --- | --- | --- |
| Bootstrap PF / SMC | Gordon et al. (1993); Doucet et al. (2001) | student report shorthand keys only | keep BayesFilter canonical bibliography keys | already familiar, but must verify exact key names in `docs/references.bib` |
| PMCMC / pseudo-marginal | Andrieu et al. (2010) | student report and README mention PMMH / particle MCMC | keep primary published key | needed for exact-target comparison, not only implementation context |
| EDH | Daum and Huang series | CIP uses multiple Daum--Huang references; student report also cites variants | one canonical BayesFilter key per actual paper | must distinguish original flow papers from later exact-flow/stiffness papers |
| LEDH / PF-PF | Li and Coates (2017) and related local references | student report and advanced repo both rely on Li & Coates examples | audit actual paper identity and final preferred key | load-bearing for proposal-correction discussion |
| Kernel / stochastic particle flow | Hu & van Leeuwen; Dai & Daum | student report emphasizes both | verify preferred primary keys | needed mainly for comparison and stress/stiffness context |
| Soft resampling | Zhu, Murphy, Jonschkowski | CIP and student sources both mention soft resampling | verify preferred primary key in BayesFilter bibliography | load-bearing for bias/differentiability trade-off |
| OT differentiable PF | Corenflos et al. (2021) | advanced repo and CIP both build on it | verify preferred primary key and publication version | central for transport-resampling discussion |
| Learning-based DPF | Jonschkowski et al.; Karkus et al. | MLCOE report groups these under differentiable/learning-based filtering | keep as separate learning-objective literature lane | useful but must not be conflated with HMC-safe likelihoods |
| Learned/amortized OT | repo-local architecture note plus external literature trail | advanced repo cites its own engineering note and recent OT-learning papers | treat local note as comparison source, not bibliography authority | requires literature trail if promoted into monograph theorem-level prose |

## Table 1: method-family map

| Method family | Core sources inspected in this pass | Mathematical object | Exact / unbiased / approximate / surrogate status | Implementation implications | HMC relevance |
| --- | --- | --- | --- | --- | --- |
| Bootstrap particle filter | CIP ch26; MLCOE report | empirical filtering measure and likelihood estimator | unbiased likelihood estimator under standard assumptions | particles, weights, ESS, resampling | relevant as exact-target comparison baseline for PMCMC, not pathwise differentiable by default |
| EDH | CIP ch26 | homotopy-defined deterministic flow under Gaussian closure | exact in linear-Gaussian recovery, approximate more generally | ODE integration, stiffness, Gaussian-closure assumptions | useful benchmark and possible proposal mechanism; not generic HMC-safe likelihood on its own |
| LEDH | CIP ch27; MLCOE report | local particle-wise linearized flow | improved local approximation, not generic exactness | local Jacobians, per-particle drift, stronger numerical burden | promising but still approximate; target status depends on correction |
| PF-PF / importance-corrected flow | CIP ch27; MLCOE report | proposal correction with change-of-variables weight | proposal-corrected particle method, not merely transport approximation | Jacobian/log-det machinery, variance diagnostics | first strong rung for HMC-related value-side development |
| Soft resampling | CIP ch26/ch32; advanced repo notebooks index | smooth resampling relaxation | biased differentiable surrogate | tuning parameter, bias analysis, pathwise gradient | useful for learning/surrogate HMC experiments, not automatically exact-target HMC |
| OT resampling | CIP ch26/ch32; advanced repo OT notes | entropic OT barycentric transport | relaxed differentiable transport, not exact multinomial resampling | Sinkhorn solver, regularization, convergence diagnostics | strong candidate surrogate/approximate target; needs explicit interpretation |
| Learned / amortized OT | advanced repo architecture note | neural approximation to OT barycentric map | learned surrogate to a relaxed transport map | training distribution, approximation residual, extrapolation risk | promising acceleration layer, but adds another approximation between value and target |
| Learning-based differentiable PF | MLCOE report, CIP note on variational directions | end-to-end differentiable training objective | often surrogate / lower-bound / training objective rather than exact likelihood estimator | autodiff-friendly objective, learned model components | relevant for learning, but HMC interpretation requires extra care |

## Table 2: student-coverage comparison

| Topic | Primary literature / CIP coverage seen here | Student coverage | Gap / disagreement | Action |
| --- | --- | --- | --- | --- |
| Classical PF / SMC basics | strong in CIP ch26 | strong in MLCOE report | no major disagreement in this pass | keep as baseline exposition |
| EDH derivation | strong in CIP ch26 | MLCOE report gives a parallel derivation | likely close, but needs equation-level comparison later | compare derivations mathematically in M3 |
| LEDH and PF-PF | strong in CIP ch27 | strong in MLCOE report and advanced repo classical track | student work may lean more quickly into implementation comparison than target-status discussion | preserve math comparison; add stricter target-status language |
| Kernel PFF / marginal collapse | limited in CIP DPF spine | strong in MLCOE report | student coverage is broader than current BayesFilter/CIP emphasis | consider whether a comparison subsection is needed in architecture phase |
| Stochastic particle flow / stiffness | present in CIP and stronger in MLCOE survey | strong in MLCOE | aligned on stiffness importance | keep as comparison/stress context |
| Soft resampling | present in CIP ch26/ch32 | present indirectly in advanced repo HMC thread | no major disagreement yet | rigorous bias discussion required in M4 |
| OT resampling / Sinkhorn | strong in CIP ch32 | strong in advanced repo | advanced repo adds better implementation detail and sensitivity context | use for implementation stress, not theorem authority |
| Amortized / learned OT | only partially in CIP endpoint framing | strong in advanced repo architecture note | student source covers more implementation detail than current CIP source | require separate surrogate-status treatment in M4/M5 |
| HMC with DPF | limited and forward-looking in CIP | stronger experimentally in advanced repo | likely disagreement in strength of conclusions, not necessarily in formulas | explicit rung-by-rung HMC-target analysis required in M5 |

## Table 3: unresolved or load-bearing claims

| Claim | Source support in this pass | Needed audit | Status |
| --- | --- | --- | --- |
| EDH linear-Gaussian recovery | CIP source present | compare derivation and exact statement | source-backed, equation audit pending |
| PF-PF weight correction | CIP ch27 and MLCOE coverage present | bounded derivation audit of change-of-variables formula | source-backed, equation audit pending |
| Soft-resampling bias statement | CIP ch32 present | bounded derivation audit; compare sign/bias claims | source-backed, equation audit pending |
| OT barycentric map interpretation | CIP ch32 and advanced OT note present | source-vs-engineering comparison audit | source-backed, interpretation audit pending |
| Learned OT posterior shift | advanced repo note states it explicitly | needs primary-literature and implementation-status treatment | comparison-backed, literature extension pending |
| Differentiability implies HMC suitability | no source in this pass supports that statement | must be analyzed and likely rejected in general | unresolved, expected negative result |
| Nonlinear DSGE / MacroFinance suitability of DPF ladder | no single source closes this | synthesize in M5 using target-status analysis | unresolved, central program question |

## Table 4: theorem / equation obligation register

| Theorem or equation family | Derive fully | Cite and adapt | Compare variants | Human review required |
| --- | --- | --- | --- | --- |
| bootstrap PF likelihood estimator and status | no | yes | yes | no |
| EDH homotopy and continuity-equation setup | yes | yes | yes | likely yes |
| EDH affine flow under Gaussian closure | yes | yes | yes | likely yes |
| EDH linear-Gaussian recovery statement | yes | yes | yes | no |
| PF-PF importance-weight correction | yes | yes | yes | no |
| Jacobian / log-determinant evolution under flow | yes | yes | yes | no |
| soft-resampling bias statement | yes | yes | yes | likely yes |
| entropic OT primal / Sinkhorn / barycentric map | yes for core formulation | yes | yes | likely yes |
| learned/amortized OT as approximation layer | no theorem-level derivation expected | yes | yes | yes |
| HMC target interpretation at each rung | yes | yes | yes | yes |

## Tests run in this phase

- ResearchAssistant workspace init: passed
- ResearchAssistant doctor: passed in offline mode with expected PDF-tool warnings
- Local student artifact inventory: passed
- Literature-name extraction from key CIP DPF chapters: passed
- Student full-report PDF local extraction via `Read`: partial but sufficient for orientation

## Interpretation

This pass supports three important conclusions.

### 1. The literature base is broad enough to proceed

The source pool is now clearly broad enough to justify an architecture phase.
CIP covers the core DPF, PF-PF, resampling, and HMC-interface material.  The
student materials extend coverage on kernel/stochastic PF, implementation
sensitivity, OT engineering, amortized OT, and DPF-HMC experiments.

### 2. The student lane adds real coverage, not merely examples

The student sources are mathematically and experimentally useful in two specific
ways:
- the MLCOE report broadens the literature sweep, especially on kernel and
  stochastic particle flow;
- the advanced repo is much richer on implementation-facing OT/HMC details than
  the current BayesFilter or CIP DPF draft.

This means the final monograph should not ignore them entirely.  But they still
belong as comparison and coverage inputs, not as the monograph's voice.

### 3. The most important unresolved question remains the HMC target question

No source inspected in this pass justifies collapsing the DPF ladder into a
single general statement of HMC correctness.  The key unresolved issue is still
method-rung-specific target interpretation: exact, unbiased, approximate,
relaxed, or learned surrogate.

## Audit

The literature survey phase has done enough to justify the next architectural
phase.

What is now well established:
- the final DPF monograph treatment will likely require more than three chapters;
- the chapter structure should separate particle-flow theory, resampling/OT
  theory, and HMC-target assessment;
- there is enough source support to design that structure responsibly.

What still remains unresolved:
- exact theorem/derivation ownership for several load-bearing formulas;
- the final citation-key canonicalization in `docs/references.bib`;
- the precise rung-by-rung HMC claims for nonlinear DSGE and MacroFinance.

## Next phase justified?

Yes.

Phase M2 remains justified because the literature map is now strong enough to
support a deliberate mathematical chapter architecture rather than continuing
with ad hoc rewriting.
