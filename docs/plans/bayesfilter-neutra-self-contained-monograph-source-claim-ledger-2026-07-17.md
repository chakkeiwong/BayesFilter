# NeuTra Monograph Source And Claim Ledger

Date: 2026-07-17

Status: `FROZEN_AT_COMPLETION`

## Source Support

| Source | Class | Technical material inspected | Allowed support | Boundary |
| --- | --- | --- | --- | --- |
| Hoffman et al. (2019), local ICML source and PDF | `DIRECT_METHOD` | Sections 2.1--2.3, Equation 4 IAF, Steps 1--3, Section 3.1 geometry, experiment setup and results | Historical method attribution, IAF convention, variational-to-HMC construction, induced metric motivation | Does not support BayesFilter implementation or BayesFilter numerical results. |
| Papamakarios et al. (2021) | `SURVEY_OR_TUTORIAL` | Existing monograph citation only; not needed for any theorem-level claim in this expansion | General normalizing-flow context | No core derivation depends on this survey. |
| `bayesfilter/inference/neutra_training.py` | `IMPLEMENTATION_OR_SOFTWARE` | Config, MADE masks, bounded scale, stage reversal, fixed affine composition, reverse-KL training and checkpointing | Exact BayesFilter transport and training semantics | Supports implementation behavior, not statistical validity. |
| `bayesfilter/inference/neutra_hmc.py` | `IMPLEMENTATION_OR_SOFTWARE` | Fixed HMC construction, health telemetry, sequential warm-up and retained collection | Exact BayesFilter sampling protocol | Native TFP divergence flags are not exposed on this route. |
| Retrospective truth-tail result and JSON | `IMPLEMENTATION_EVIDENCE` | All four eligible historical archives, hashes, diagnostics and 33 parameter rows | Common truth-tail results for LGSSM, predator--prey UKF/SGQF, and SIR SGQF | One frozen dataset per configuration; UKF/SGQF are approximate likelihoods. |
| Structural owner-adjudicated result and JSON | `IMPLEMENTATION_EVIDENCE` | Training, tuning, warm-up, retained diagnostics and five parameter rows | Qualified structural UKF result | Original bulk-ESS 1000 gate missed; revised 900 threshold was post-result and is disclosed. |
| Cross-repository model evidence ledger and cited result notes | `IMPLEMENTATION_EVIDENCE` | Count rule and the eight additional historical configurations | Breadth of learned-NeuTra transformed-HMC experience | Historical protocols are heterogeneous and are not retroactive truth-tail passes. |

## Claim Support

| Claim | Support class | Anchor |
| --- | --- | --- |
| NeuTra samples an exact Jacobian-corrected pullback target for any frozen bijection | `PROJECT_DERIVATION` | Chapter change-of-variables and pushforward-invariance derivations. |
| An imperfect variational fit affects efficiency, not the stationary target, when HMC and Metropolis correction are correct | `PROJECT_DERIVATION` | Chapter invariance and HMC acceptance derivations. |
| The BayesFilter stage has a triangular Jacobian with log determinant equal to the sum of bounded log scales | `PROJECT_DERIVATION` plus `IMPLEMENTATION_EVIDENCE` | Chapter dense-IAF derivation and `TrainableDenseAutoregressiveIAF.forward_and_logdet`. |
| BayesFilter training minimizes reverse KL using batched reparameterized samples | `PROJECT_DERIVATION` plus `IMPLEMENTATION_EVIDENCE` | Chapter reverse-KL derivation and `train_plain_dense_iaf`. |
| Five common configurations placed all 38 truths inside empirical 95% intervals | `IMPLEMENTATION_EVIDENCE` | Retrospective result plus structural owner-adjudicated result. |
| NeuTra reached transformed HMC in ten families and thirteen configurations | `IMPLEMENTATION_EVIDENCE` | Cross-repository ledger plus structural campaign close. |

## Snowball And Omission Audit

Backward snowballing from the NeuTra paper identifies transport-map MCMC,
Riemannian HMC, adaptive mass-matrix HMC, learned transition kernels, and
variational/MCMC hybrids as neighboring lines. The chapter discusses the
conceptual comparisons needed to motivate NeuTra but does not attempt a full
survey; Chapters 21, 22, and 26 cover HMC and transport context elsewhere.

Forward citation metadata and venue rankings were not queried because the task
does not depend on live popularity metadata and the primary method source is
already local. No retraction, withdrawal, or erratum notice is present in the
local source record. The bibliography mismatch between the verified ICML 2019
paper and the previous JMLR-style entry is corrected during this expansion.

The main omission risk is a comprehensive comparison with all learned MCMC and
transport-map methods. That is outside this chapter's algorithm-and-evidence
scope and no superiority claim is made.

## Terminal Audit

- The active dense-IAF implementation was checked against the documented masks,
  bounded log scale, three-stage composition, reverse permutations, fixed
  affine lift, reverse-KL loss, stateless batched sampling, manual Adam state,
  TensorFlow control flow, XLA, and GPU requirement.
- The common table was parsed and compared with the retrospective and structural
  JSON archives: 38 rows, 190 numeric fields, 38 interval-inclusion flags, and
  five configuration minima passed at the table's printed precision.
- The cross-repository breadth count is compositional: the frozen pre-structural
  ledger has nine families and twelve configurations; the structural UKF result
  adds one family and one configuration, yielding ten and thirteen.
- Bounded Claude reviews agreed on core mathematics, the repaired experimental
  section, and the final explicit HMC/IAF/SGQF proofs.  The one material review
  finding, ambiguous structural parameter ordering, was corrected before the
  agreeing rerun.
- The complete monograph built to `docs/main.pdf`; the NeuTra chapter produced
  no local LaTeX warning.  Remaining global duplicate-label and undefined-
  citation warnings originate outside this chapter and were not modified.
