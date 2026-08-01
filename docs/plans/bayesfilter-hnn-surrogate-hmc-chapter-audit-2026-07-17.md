# HNN surrogate-HMC chapter source and mathematics audit

Date: 2026-07-17

## Decision

The chapter is admissible for the BayesFilter monograph after compilation and
terminal mathematics/readability review. Its exactness theorem is a project
derivation for a narrower construction than the seed paper: a frozen
position-only force, symmetric kick-drift-kick proposal, momentum flip, and
true joint-Hamiltonian endpoint correction.

The seed paper supports the research motivation, algorithm description, and
reported experiments. It does not support the chapter's corrected discrete-map
theorem. The paper's appeal to Liouville volume preservation and empirical
forward/reverse paths is insufficient to prove momentum-flip reversibility and
unit-Jacobian detailed balance for its executed general momentum-dependent
L-HNN numerical map.

## Source-Support Ledger

| Source | Class | Local artifact | Technical anchors inspected | Allowed support | Forbidden support |
|---|---|---|---|---|---|
| Dhulipala, Che, and Shields (2022), arXiv:2208.06120v2 | direct method | `.localresources/papers/dhulipala-che-shields-2022-latent-hnn.pdf` and `.txt`; PDF SHA-256 `31ffe49672361fb6bad5c6ce45fd1d84575921b0884fd39e901c0314ab88b573` | Equations 2.1-2.12, 3.1-3.5; Algorithms 3.1-3.3, 4.1-4.2, 8.1-8.2; Sections 3.2-3.3, 4.2, experiments, conclusion | paper algorithm, intended endpoint Hamiltonian test, online fallback, reported gradient-count/ESS evidence, stated limitations | proof that the executed discrete general L-HNN map is reversible or volume-preserving; filtering-posterior validity; BayesFilter readiness |
| Greydanus, Dzamba, and Yosinski (2019) | foundational | bibliography/source already used by monograph | HNN scalar-Hamiltonian/vector-field construction as summarized and cited by seed paper | historical HNN background | corrected MCMC invariance theorem |
| Neal (2011); Betancourt (2017) | foundational | bibliography entries | standard phase-space HMC, reversibility, volume preservation, MH correction | standard HMC background | validity of an unexamined neural map |
| Andrieu and Roberts (2009); Andrieu, Doucet, and Holenstein (2010) | foundational/competitor boundary | bibliography entries and existing particle-filter chapters | pseudo-marginal extended-target logic | extended-state marginalization boundary | naive noisy-gradient or fresh-noise HMC |
| BIhNNs official repository | implementation/software | `.localresources/software/BIhNNs-c44c9cb9/` | `hnn.py`, `utils.py`, `hnn_hmc.py`, `hnn_nuts_online.py` at commit `c44c9cb9` | executed input/output structure, synchronized update, endpoint energy test, online fallback implementation | mathematical correctness oracle, production readiness |
| Earlier `~/python` DSGE-HMC implementation and artifacts | implementation evidence | read-only external workspace paths cited in chapter | `surrogate.py`, `_advanced_hmc.py`, delayed-acceptance contract tests, `sgu_surrogate_hmc.npz`, `sgu_surrogate_dahmc.npz` | historical implementation behavior and evidence gaps | successful DSGE HNN-HMC validation |

No retraction, withdrawal, or expression-of-concern notice was found in the
inspected arXiv artifact or official repository. The official repository is
archived and explicitly warns that it is no longer maintained. Publication
status beyond arXiv v2 was not established and is not claimed.

## Citation And Venue Metadata Ledger

| Source | Publication metadata | Citation count | Venue/rank use | Caveat |
|---|---|---:|---|---|
| arXiv:2208.06120v2 | arXiv version 2, 24 October 2022 | not queried | none | Citation count and venue rank are unnecessary for the mathematical decision and were not used as truth evidence. |
| Hamiltonian Neural Networks | NeurIPS 2019 | not queried | none | Foundational context only. |
| Pseudo-marginal sources | Annals of Statistics 2009; JRSS B 2010 | not queried | none | Used only for established extended-target context already developed elsewhere in the monograph. |

## Backward-Snowball Ledger

| Seed-paper reference family | Classification | Action |
|---|---|---|
| Standard HMC and NUTS sources | foundational | Already represented by Neal, Hoffman-Gelman, and Betancourt entries. |
| Greydanus et al. HNN | foundational | Retained as the HNN origin. |
| Neural-gradient/neural-transition HMC predecessors discussed in Section 1.1 | adjacent competitors | Mentioned only generically; not needed to prove the chapter's narrow construction. Add only if a future chapter claims literature completeness or comparative priority. |
| Symplectic neural networks and Symplectic ODE-Net | adjacent architecture | Recorded as an omission risk below; outside the proof because BayesFilter deliberately uses an explicit shear composition. |
| Roberts and Rosenthal MCMC reference cited by the seed paper | foundational context | Not used to transfer validity to the seed paper's unproved discrete map. |

## Forward-Snowball Ledger

No external citation-index query was run. Network use was limited to the
user-specified seed paper and its official code. The official repository now
includes a SympNet lane and cites the related arXiv:2209.09349 work; this is a
known follow-up/adjacent source but is not needed for the theorem proved here.
Forward-citation coverage remains `not checked`, so the chapter does not claim
to be a complete survey of learned HMC integrators.

## Claim-Support Ledger

| Chapter claim | Support class | Support |
|---|---|---|
| The paper uses a learned Hamiltonian in HMC/NUTS and a true endpoint Hamiltonian test. | primary technical support | Seed paper Sections 3.2-3.3, Algorithm 3.3, Equation 3.5. |
| The paper uses online error monitoring and true-gradient fallback. | primary technical support | Seed paper Section 4.2, Algorithms 4.1-4.2. |
| Frozen position-only kick-drift-kick plus true joint-energy MH preserves the deterministic target. | project derivation | Chapter Lemmas on shear volume and momentum-flip reversibility plus deterministic MH proposition. |
| Endpoint kinetic energy is required in general. | project derivation | Explicit phase-coordinate swap counterexample. |
| Liouville volume preservation does not imply momentum-flip reversibility. | project derivation | Explicit Hamiltonian `H(u,p)=u^2/2+p^2/2+cp` counterexample. |
| Stage-two delayed-acceptance correction cannot replace missing stage one. | project derivation | Three-state deterministic-cycle counterexample. |
| Deterministic UKF/SGQF correction targets the filter-defined posterior, not automatically the exact latent-model posterior. | project derivation | Equality-up-to-constant posterior argument. |
| Pseudo-marginal exactness is an extended-state result. | primary support plus project derivation | Andrieu-Roberts/PMCMC sources; marginalization proof restated in chapter. |
| Earlier SGU work does not establish success. | implementation evidence | Stored artifact fields and local code inspection. |

## Omitted-Paper And Reviewer-Risk Register

| Risk | Status | Reason/action |
|---|---|---|
| Later exactness analyses or corrections of L-HNN HMC may exist. | open, nonblocking | Forward snowball not performed. The chapter avoids claiming novelty or literature completeness and proves its own narrower result. |
| Symplectic neural proposal literature is not surveyed fully. | accepted scope limit | The chapter does not rank neural integrator families. A future comparative chapter should audit SympNets and learned involutive MCMC explicitly. |
| Delayed-acceptance literature beyond the direct balance derivation is not surveyed. | accepted scope limit | The theorem is proved self-contained. Add classical delayed-acceptance citations if historical priority is discussed later. |
| Pseudo-marginal Hamiltonian methods are not surveyed. | accepted scope limit | The chapter states only that a complete extended-state kernel is required; it does not claim no such methods exist. |

## Skeptical Plan Audit

| Risk | Resolution before drafting |
|---|---|
| Wrong baseline | Matched ladder includes tuned ordinary HMC, NeuTra/affine true-gradient HMC, zero-residual force, learned residual, and fallback. |
| Proxy promotion | Value RMSE, force cosine, and training loss are explanatory only; corrected-chain validity and amortized ESS/sec are the decision metrics. |
| Missing stop condition | Nonfinite endpoint, reversibility/Jacobian test failure, target-identity mismatch, fresh stochastic endpoint noise, or convergence veto blocks promotion. |
| Unfair cost comparison | Training, compilation, endpoint values, gradient calls, force calls, fallback, and wall time are charged. |
| Hidden assumption | Position-only frozen force, symmetric fixed-step map, endpoint momentum, deterministic scalar, and fixed retained kernel are stated explicitly. |
| Stale context | Earlier `~/python` artifacts are classified as scaffolding/failure evidence, not current success. |
| Environment mismatch | No experiment is launched by this documentation task. TensorFlow/XLA/GPU policy is stated for a future implementation. |
| Artifact does not answer question | Chapter proves invariance and target boundary; it does not claim empirical speedup. |

## What Is Not Concluded

- No BayesFilter HNN implementation has been added.
- No corrected HNN-HMC experiment has been run.
- No speedup, convergence, posterior-recovery, or default-readiness claim is made.
- The chapter is not a complete literature survey of learned HMC proposals.
- Corrected HMC for a deterministic approximate-filter scalar is not exact for
  the latent state-space model unless the likelihoods agree up to a constant.

## Terminal Review

Local mathematics and readability audit:

- passed the deterministic involution/Jacobian proof;
- passed the kick-drift-kick shear and momentum-flip proof;
- passed the kinetic-energy counterexample;
- passed the delayed-acceptance balance derivation and missing-stage-one
  counterexample;
- repaired the fixed-randomness proof with an explicit positive unbiased
  estimator whose realized likelihood ratio is nonconstant;
- repaired pseudo-marginal notation by introducing a dominating measure and
  an auxiliary density, including its conditional energy contribution;
- distinguished mathematical out-of-support rejection from undefined
  numerical execution;
- made state-dependent fallback restrictions explicit; and
- checked the rendered chapter for self-contained flow and terminology.

Independent Claude Opus max-effort read-only review of exactly
`docs/chapters/ch26c_hnn_surrogate_hmc.tex` returned `VERDICT: AGREE` and found
no material proposition or proof error. It requested four exposition
repairs, all applied:

1. density-level pseudo-marginal notation under a dominating measure;
2. the cited paper's actual position update displayed before the integrator
   mismatch conclusion;
3. schematic executed logic for the historical missing-stage-one defect; and
4. an exact-flow existence qualifier in the momentum-reversal proposition.

Terminal decision: mathematically and pedagogically suitable for the
BayesFilter monograph, subject to the explicit nonclaims above. This is a
theory and design chapter, not implementation or empirical promotion evidence.

## Final Document Verification

The complete monograph was rebuilt after the terminal typography pass with:

```bash
cd /home/chakwong/BayesFilter/docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The command exited successfully and produced `docs/main.pdf` with 436 pages.
The final log contains no overfull box attributed to
`ch26c_hnn_surrogate_hmc.tex`, and the chapter's citation and cross-reference
targets resolve. The compact filtering-boundary table has underfull cell
warnings, which are expected line-spacing diagnostics rather than clipped
content. The full monograph retains four duplicate labels and eleven unresolved
citations in unrelated pre-existing chapters; those warnings are outside this
chapter's scope and were not used to weaken its verification standard.

Focused source checks also passed:

```bash
git diff --check -- docs/main.tex \
  docs/chapters/ch26c_hnn_surrogate_hmc.tex \
  docs/chapters/ch19e_dpf_hmc_target_suitability.tex \
  docs/references.bib \
  docs/plans/bayesfilter-hnn-surrogate-hmc-chapter-audit-2026-07-17.md
```

The chapter source is ASCII-clean. Final review found no material mathematical,
source-support, target-boundary, or readability defect.
