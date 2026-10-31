# M13 repair proposal and mathematical audit

Date: 2026-10-04. Question: can we remove the diagnosed ancestor-conditional
importance-weight collapse without degrading the existing LGSSM and KSC-SV
value/analytical-score implementations? The purpose remains testing surrogate
HMC: an improved force is a means to efficient sampling of the declared endpoint
target, not a substitute for that endpoint or for posterior diagnostics.

This stage writes and audits a proposal. It does not change the production
algorithm, its defaults, the likelihood used at Metropolis acceptance, or launch
HMC. Baseline is the current dirty checkout at
`a55c6be7a16e9401242ee3449248fa347eb58d90`, including the synced score fixes,
and the completed October 4 M13 localization result.

## Evidence contract and research intent

- Mechanism: marginalize the ancestor label in the importance correction while
  retaining the canonical UKF/LEDH map; examine a prior mixture component as an
  optional safeguard. Both are explicitly proposed extensions, not claims about
  the unchanged Li--Coates Algorithm 1.
- Expected failure: the transported particle is evaluated against one narrow
  conditional process density, although its flow uses the broader predictive
  covariance. Correct density ratios then have extreme finite-design variance.
- Primary criterion for this stage: a complete derivation of the proposed
  finite likelihood and its total analytical derivative, explicit theorem
  assumptions, identified limits of the variance theorem, and an implementable
  non-regression protocol for the original consumers.
- Veto: incorrect density normalization, missing total-derivative terms,
  unsupported transfer of random-sampling theorems to fixed designs, or a claim
  that conditional one-step results prove full-filter likelihood unbiasedness.
- Repair trigger: any failed identity or MathDevMCP counterexample revises the
  proposal before handoff. Tool abstention is recorded, never relabeled proof.
- Explanatory evidence: prior localization ESS, normalization, displacement and
  covariance ratios. These do not certify the new method.
- Nonconclusions: no new default, empirical accuracy improvement, completed
  non-regression test, full-horizon exact score, or surrogate-HMC convergence.

Cheap adversaries for future implementation are the same-design bootstrap
filter, current canonical LEDH, and the independent quadrature/reference.
Compare them separately at low Q, posterior center and tails, and high Q;
preserve all eight diagnosed points as development data, with fresh heldout
points and randomizations required for any later promotion claim.

## Assumptions and skeptical audit

The marginal predictive density is a finite mixture conditional on the previous
cloud, not the exact filtering distribution. The current M13 flow is affine
conditional on each ancestor because the auxiliary path does not depend on the
current process innovation. These facts must be checked through the consumer
call chain. Marginalization costs quadratic density work; tiling reduces memory,
not arithmetic. A defensive proportion, if retained, is an untuned hypothesis
and must be fixed offline and differentiated correctly in the finite program.

Variance reduction by Rao--Blackwellization applies to the matched random-label
experiment, not automatically to the existing one-child-per-ancestor fixed
design or its log-likelihood score. Original LGSSM/KSC scopes must retain their
unchanged route and frozen inputs until direct regression evidence permits an
explicit migration. Equality by dispatch is a proposed preservation property,
not evidence that unimplemented code already has it.

Audit verdict before work: proceed with source inspection and scoped proofs.
Reject a silent replacement of UKF covariance by Q, uncorrected weight clipping,
and a universal finite-sample non-deterioration claim. Those do not follow from
the diagnosed failure and would change the scientific object without proof.

## Budget, environment, and outputs

Local reading, primary-source retrieval, symbolic MathDevMCP checks and document
builds only. At most 120 seconds of local algebra/test execution, excluding
tool-service latency and document builds; no GPU experiment or HMC allocation.
No production source edit is planned. Existing 1,575.90-second diagnostic
balance remains reserved. Preserve logs and tool responses under
`docs/plans/artifacts/surrogate-hmc-m13-marginal-repair-proposal-20261004-01/`.
Write a mathematical proposal, audit/result note, source-support ledger and a
one-path Claude review handoff. Record exact checked equalities separately from
structural document checks and manual proofs. Stop this stage when these
deliverables are complete, or report a material unresolved mathematical issue.
