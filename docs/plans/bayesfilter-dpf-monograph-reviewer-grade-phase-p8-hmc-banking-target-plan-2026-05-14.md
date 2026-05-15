# Phase P8 plan: HMC target correctness and banking suitability expansion

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, P3, P4, P5, P6, and passed P7 result.
- P2 established bibliography-spine support only for HMC, pseudo-marginal, and
  surrogate-source families; do not claim ResearchAssistant-reviewed support
  unless a later artifact records it.
- Allowed write set: this target chapter and the P8 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Make the HMC and banking-suitability chapter credible to reviewers trained in
Hamiltonian dynamics, MCMC, numerical analysis, physics, or scientific
computing.  The chapter must distinguish value-gradient consistency from exact
posterior correctness and must avoid premature banking claims.

## Required implementation instructions

1. Add a notation/object inventory:
   - parameter;
   - scalar log target;
   - potential energy;
   - momentum;
   - Hamiltonian;
   - numerical integrator;
   - Metropolis correction;
   - gradient supplied to integrator;
   - value used for accept/reject;
   - target-status category.
2. Expand HMC correctness contract:
   - exact target HMC;
   - value-gradient consistency;
   - integrator error and correction;
   - what happens when value and gradient differ.
3. Separate method families:
   - exact likelihood HMC;
   - pseudo-marginal MCMC;
   - noisy-gradient methods;
   - delayed-acceptance or surrogate-corrected methods;
   - relaxed-target HMC;
   - learned-surrogate HMC.
4. Rewrite rung-by-rung analysis:
   - EDH;
   - PF-PF;
   - soft resampling;
   - OT/EOT;
   - learned OT.
   For each rung, state scalar value, gradient path, target status, assumptions,
   diagnostics, and missing validation.
5. Replace risky phrasing:
   - avoid `HMC-ready`;
   - avoid `production-ready`;
   - replace `first serious candidate` with a precise statement such as
     `first rung with an explicit proposal-correction interpretation, pending
     finite-particle and numerical validation`.
6. Expand nonlinear DSGE limitations:
   - determinacy;
   - pruning;
   - invalid parameter regions;
   - structural timing;
   - latent-state semantics;
   - local Jacobian fragility.
7. Expand MacroFinance limitations:
   - latent dimension;
   - long panels;
   - missing or mixed-frequency data;
   - volatility/noise curvature;
   - compiled repeatability;
   - model-risk governance.
8. Add evidence table:
   - exploratory use;
   - credible research use;
   - bank-facing research claim;
   - production or governance claim.

## Required source use

- HMC foundations.
- Pseudo-marginal MCMC.
- Surrogate/noisy-gradient MCMC if claims are made.
- DPF chapter claims from P4-P7.
- DSGE/MacroFinance structural-filtering chapters for model-specific limits.
- Because P2 found no local ResearchAssistant summaries, source use is
  bibliography-spine provenance unless reviewed source evidence is added later.
  Banking and HMC claims must remain evidence-gated unless supported by local
  derivation and recorded validation evidence.

## Mathematical audit rules

- Correct HMC target statements must name the scalar target.
- Pseudo-marginal likelihood-estimator logic must not be confused with smooth
  surrogate-gradient logic.
- Banking suitability must be evidence-gated.
- Do not let tables carry the whole target argument.

## Required local tests/checks

- Search for `HMC-ready`, `production`, `validated`, `serious`, `suitable`,
  `correct`, and `exact`.
- Confirm every target-status word has assumptions and limitations nearby.
- Use MathDevMCP or manual proof obligations for:
  - value-gradient consistency statement;
  - rung target-status table;
  - pseudo-marginal versus surrogate distinction.
- Create a P8 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.

## Veto diagnostics

The phase fails if:

- method labels substitute for scalar target definitions;
- PF-PF is oversold;
- relaxed or learned DPF is implied to target the original posterior;
- banking claims exceed evidence;
- pseudo-marginal and differentiable-surrogate logic remain blurred.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build status is omitted.

## Exit gate

The chapter is ready only if a skeptical HMC-literate reviewer can identify the
target sampled by each rung and the evidence still missing for banking use.
