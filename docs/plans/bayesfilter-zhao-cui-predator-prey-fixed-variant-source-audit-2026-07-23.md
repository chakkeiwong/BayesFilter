# Zhao-Cui Predator-Prey Fixed-Variant Source Audit

Date: 2026-07-23
Scope: source grounding for the BayesFilter predator-prey fixed-branch implementation
Decision: `PASS_SOURCE_GROUNDING_FOR_EXTENSION_IMPLEMENTATION`

## Boundary

The implementation may reuse the paper's squared-TT defensive density,
paired-core marginalization, conditional KR sampling, and importance-density
correction. The assembled source-order fixed-branch likelihood and analytical
score are not in Zhao and Cui. The assembled route is therefore
`extension_or_invention`; freezing its randomness and numerical settings is a
`fixed_hmc_adaptation` of individually cited source operations.

This audit does not support a claim that the assembled route is source-faithful,
an exact likelihood, an unbiased pseudo-marginal estimator, HMC-ready, or
statistically superior.

## Source-Support Ledger

| Source | Class | Local artifact | Status | Inspected technical anchors | Allowed support | Forbidden support |
| --- | --- | --- | --- | --- | --- | --- |
| Zhao and Cui, *Tensor-train methods for sequential state and parameter learning in state-space models*, JMLR 25 (2024) | `DIRECT_METHOD` | `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt` | Published JMLR text; no local retraction, withdrawal, or erratum notice found; external status not queried | Equations 1-5 and event order, lines 45-118; Eq. 13, lines 539-573; Proposition 2 and KR conditionals, lines 592-670; Algorithm 3, lines 890-924; recursive/preconditioned construction, lines 1656-1718; predator-prey experiment, lines 2412-2522 | Source event order, squared-TT defensive construction, paired-core marginals, conditional KR map, importance correction, predator-prey experiment settings | Fixed-branch likelihood-score recursion, finite-grid inverse implementation, HMC correctness, exact likelihood, broad high-dimensional success |
| Zhao-Cui pinned MATLAB repository snapshot | `IMPLEMENTATION_OR_SOFTWARE` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/` | Pinned local source; license files present | `eg4_predatorprey/mainscript.m:12-17,45-79`; `models/ssmodel.m:34-58`; `models/full_sol.m:21-43,46-135`; `models/pre_sol.m:16-31`; `models/computeL.m`; extracted `models/pp/{setup,transition,st_process,ob_process,like,predator_step}.mlx`; `deep-tensor.dev/src/SIRT.m:51-85`; `@TTSIRT/TTSIRT.m:117-175`; `@TTSIRT/marginalise.m:19-85`; `@TTSIRT/eval_cirt_reference.m:43-100` | Author implementation order, distributions, adaptive fitting/sampling, correction density, conditional transport semantics | Mathematical validity by itself, BayesFilter fixed-HMC assembly, runtime score |

The `.mlx` files were inspected through their embedded `matlab/document.xml`,
not through binary terminal output. The archive's `predator_step.mlx` contains a
fourth-stage expression that differs from the standard RK4 expression used by
the paper description and sealed BayesFilter target. This audit does not
silently resolve that archive ambiguity into a source-faithfulness claim: the
implementation binds the already sealed BayesFilter RK4 target and records the
assembled route as an extension.

## Citation And Venue Metadata Ledger

| Source | Venue/year | Citation count | Venue metric | Access date | Caveat |
| --- | --- | --- | --- | --- | --- |
| Zhao and Cui | JMLR 25, 2024 | not available | not available | 2026-07-23 | Network metadata was unnecessary for the implementation gate and was not queried. Counts and rankings would be visibility signals only. |

## Backward-Snowball Ledger

The scoped audit inspected the paper's introduction, method comparisons, and
conclusion. The following directly adjacent works were considered but are not
needed to define the fixed predator-prey program:

| Work/group | Classification | Action | Reason |
| --- | --- | --- | --- |
| Cui et al. (2023), stationary SIRT/function approximation | `FOUNDATIONAL` | omit from implementation claim | Zhao-Cui's inspected equations and pinned code directly define the operations used here. |
| Chopin et al. (2013); Crisan and Miguez (2018), SMC integration | `COMPETITOR` / `BACKGROUND` | omit from implementation claim | The paper presents integration as future work; it cannot support this fixed APF assembly. |
| Gerber and Chopin (2015), sequential quasi-Monte Carlo | `COMPETITOR` | omit from implementation claim | Relevant to structured points, not to the analytical score derived here. |
| Knothe-Rosenblatt and transport-map references cited by Zhao-Cui | `FOUNDATIONAL` | covered through checked Zhao-Cui construction for this narrow implementation | No new theorem or optimal-transport claim is made. |

## Forward-Snowball Ledger

No public metadata index was queried. Recent citing works, replications, and
corrections are `not available` for this scoped implementation audit. This is a
coverage limitation for a literature survey, but it does not leave the exact
pinned paper/code operations used here undefined.

## Claim-Support Ledger

| Claim | Support class | Anchor or derivation |
| --- | --- | --- |
| The observation sequence starts at `y1`, after transition from `x0` | `PRIMARY_TECHNICAL_SUPPORT` and `IMPLEMENTATION_EVIDENCE` | Paper equations 1-3; `models/ssmodel.m:34-42` |
| Squaring a TT approximation with a positive defensive term defines the proposal density used by SIRT | `PRIMARY_TECHNICAL_SUPPORT` | Paper Eq. 13; `deep-tensor.dev/src/SIRT.m:51-85` |
| Paired-core contractions provide prefix marginals used by conditional KR sampling | `PRIMARY_TECHNICAL_SUPPORT` and `IMPLEMENTATION_EVIDENCE` | Paper Proposition 2; `@TTSIRT/marginalise.m:19-85`; `@TTSIRT/eval_cirt_reference.m:43-100` |
| Proposal approximation error is corrected by evaluated proposal density | `PRIMARY_TECHNICAL_SUPPORT` and `IMPLEMENTATION_EVIDENCE` | Paper Algorithm 3; `models/full_sol.m:33-42` |
| The fixed finite value is `c0 + sum(c_t)` and its score follows the normalized-weight derivative recursion | `PROJECT_DERIVATION` | Handoff mathematical contract; implementation tests compare the exact scalar with central finite differences |
| The complete fixed predator-prey route is Zhao-Cui source-faithful | `SOURCE_GAP_BLOCKER` | The author route adapts/reapproximates and jointly learns parameters/states; it does not contain this frozen finite value/score assembly |

## Omitted-Paper And Reviewer-Risk Register

| Risk | Disposition |
| --- | --- |
| A later correction or replication changes interpretation of Zhao-Cui | External forward search not performed; blocks a broad literature claim, not this pinned-source implementation. |
| Another transport/APF paper is a better comparator | Out of scope for implementing the user-selected method; same-target SGQF and GenUT remain descriptive engineering comparators only. |
| Author source and paper differ in a low-level RK4 expression | Explicitly recorded above; the sealed BayesFilter target controls and the route remains an extension. |

## Hostile Review

The source gate passes for implementing the classified extension because every
source-backed constituent operation has both a paper and author-code anchor,
while the new finite scalar and score are labelled as a project derivation.
The audit would fail if the assembled route were called source-faithful or if
its proposal correction did not use the density of the actual generated
branch.

