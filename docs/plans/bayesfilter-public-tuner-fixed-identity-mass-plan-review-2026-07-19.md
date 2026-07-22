# Plan Review: Public Fixed-Identity Mass Policy

Plan reviewed: `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-plan-2026-07-19.md`

Date: 2026-07-19

## Review scope

This review checks the plan's research question, baseline, mass semantics,
public API boundary, phase handoffs, convergence ownership, default behavior,
test coverage, repair procedure, and stop conditions. It does not authorize
implementation or a serious GPU run.

## Findings

| Area | Finding | Disposition |
| --- | --- | --- |
| Research question | Correctly asks whether the public `tune_hmc_kernel` can own fixed identity mass without another tuner or convergence implementation. | Pass |
| Baseline | Correctly retains `windowed_adaptive` as the default and requires comparison with the existing public route. | Pass |
| Mass semantics | The plan correctly identifies that passing `initial_covariance=I` is insufficient because later phases can replace the mass. The exact meaning of `PrecomputedMassArtifact.covariance` must still be verified from the native HMC construction before coding. | Required Phase 0 check |
| API propagation | The plan names `HMCKernelTuningConfig.mass_policy`, but implementation must also thread the policy through the internal `HMCTuneVerifyRepairLoopConfig` and every phase handoff. | Required Phase 0/1 check |
| Phase ownership | The plan keeps geometry, native step/trajectory tuning, repair, sequential verification, and retained sampling under existing BayesFilter owners. | Pass |
| R-hat stopping rule | Correctly rejects the campaign-private fixed 1,000-draw hard veto and requires the native sequential verifier to own chunked checks through its cap. | Pass |
| NeuTra migration | Correctly requires the active runner to call only public `tune_hmc_kernel`; the existing specialized tuner is preserved as historical evidence during migration. | Pass |
| Default regression | Explicitly requires default payload and adaptive behavior tests. | Pass |
| Evidence discipline | Separates acceptance/tuning, warm-up, retained convergence, and truth-tail evidence and states nonclaims. | Pass |
| Repair and retry | Keeps localized repairs within the unchanged scientific contract and requires fresh attempt roots and records. | Pass |
| Stop conditions | Includes target, numerical, artifact, budget, and scientific-contract vetoes. | Pass |
| Review coverage | Claude health probe returned `CLAUDE_PROBE_OK`; the bounded packet-read and substantive review returned no output. This is recorded as reviewer unavailability, not as agreement. | Limitation recorded |

## Skeptical audit verdict

`PASS_WITH_REQUIRED_PHASE0_CLARIFICATION`

The plan is suitable for implementation after Phase 0 resolves the mass
convention and confirms the complete policy propagation surface. It does not
justify executing the old private tuner again, changing the repository default,
or launching a serious run before those checks and focused tests pass.

## Required first implementation checks

1. Inspect the native `PrecomputedMassArtifact` construction and TFP kernel
   call to document whether covariance is consumed directly as the mass or as
   a preconditioning covariance/factor.
2. Add and validate the policy in both public and internal tuning config
   payloads, with the default serialized behavior unchanged.
3. Enumerate every phase that can replace or reset mass and prove that
   `fixed_identity` bypasses those updates while retaining step-size,
   leapfrog, acceptance, health, and sequential R-hat/ESS mechanics.
4. Add the default-regression, identity-lineage, zero-update, mutation-veto,
   and no-private-route tests before any serious execution.

## Nonclaims

This review does not establish that fixed identity mass is statistically
efficient, that NeuTra is scientifically valid, or that the public tuner will
pass on LGSSM. Those require the planned implementation and evidence run.
