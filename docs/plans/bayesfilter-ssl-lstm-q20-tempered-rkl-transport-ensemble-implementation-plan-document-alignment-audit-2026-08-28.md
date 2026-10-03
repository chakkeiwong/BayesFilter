# Document-alignment audit: tempered reverse-KL transport ensemble plan

Date: 2026-08-28  
Status: `PASS_PHASES_0_TO_7_PHASE8_C1_BUDGET_EXHAUSTED_GRAPH_LIMIT`

Mathematical document:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`.

Implementation plan:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`.

## Verdict

The plan implements the corrected proposal rather than the withdrawn
high-dimensional replay recommendation. Every exactness theorem has a matching
implementation veto or fixture; every discovery limitation remains an
empirical gate or nonconclusion. The plan does not equate variational weights,
chart frequencies, or initialization counts with posterior mode masses.

Two design defects were found and repaired during the initial internal review:

1. Pure continuation from a fully optimized `beta=0` reference can erase
   component diversity. The note and plan now require a positive-temperature
   restart/branching ablation.
2. The first plan wording treated every invalid HMC proposal as a campaign
   veto. It now distinguishes exactly handled proposal rejection from invalid
   current/retained states or asymmetric failure handling, which are hard
   vetoes.

Claude subsequently returned `AGREE` on the mathematics and `REVISE` on the
implementation plan. The valid plan findings have now been repaired. The
finding-by-finding disposition is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-claude-review-adjudication-2026-08-28.md`.
No unresolved document-plan contradiction remains. The absence of selected
numeric settings and a serious compute budget is deliberate and blocks only
the claim-bearing campaign.

## Claude review repairs

| Finding | Resolution in the active plan |
|---|---|
| Intermediate bridge properness | Phase 0 requires a source-bound proof: the q=20 finite Gaussian-innovation likelihood is positive and uniformly bounded because the covariance weights are nonnegative and the fixed observation variance is positive. The normalized Gaussian law has `Z_beta <= max(1,M)`; the unnormalized runtime kernel used by the value program has `bar Z_beta <= A_prior max(1,M)`. A finite beta grid is retained only as a numerical stress screen. |
| Blind-start invalidity deadlock | A fixed pre-optimizer latent Gaussian bank and finite reference-affine/scale initialization ladder replace fresh-batch retry. Invalid training rows are never replacement-resampled; the pre-update state is preserved and the update fails. |
| Fixed-gamma semantics | Phase 6 now checks `pi K = pi` exactly for two fixed gamma choices and checks the state-dependent two-state failure exactly; the controller rejects state-dependent gamma structurally. |
| Joint-arm cost gate | Phase 8A derives an admissible `K`, batch, update-count, memory, and wall-time envelope from Phase 1 measurements and the optional arm's allocated budget. It distinguishes `K^2 B` transport cross-density work from `K B` target evaluations. |
| Search versus confirmation | Phase 8 is split into calibration/search and candidate freeze; it consumes no untouched confirmation random stream. Phase 9 alone consumes confirmation streams. |
| Learned-map numerical reliability | Phase 4 implements, and Phase 8 applies, self- and cross-component inverse, log-determinant, score, and conditioning screens before tuning. |
| Undefined mode/travel criteria | The gate now uses the existing positive/negative observation-weight half-spaces as declared regions, with MCSE-aware cold/hot/start-stratified protocols and explicit nonclaims about basin identity. |
| Replica-exchange comparator mismatch | Phase 5 requires the physical and charted routes to share the same proper bridge, endpoint identities, status/cache semantics, and swap implementation. |

## Proposition crosswalk

| Mathematical result | Plan obligation | Verification |
|---|---|---|
| Reverse-KL Gaussian identity | Phases 0 and 2 use fresh Gaussian batches and exact target values; particle samples are forbidden from the primary objective. | Analytic Gaussian objective and gradient fixtures; batch/XLA receipt. |
| Product mismatch and missing-region results | Replay is removed as the high-dimensional foundation; independence MH is optional. | Evidence-role audit and optional high-dimensional mismatch diagnostics. |
| Categorical mixture and averaged-map counterexample | Phase 1 implements categorical sampling and log-sum-exp density, never map averaging. | Analytic affine mixture and averaged-map counterexample. |
| Mixture reverse-KL identity | Phase 2 enumerates outer components and all cross-component densities. | Analytic loss/gradient fixtures and `O(K^2 B)` shape/cost receipt. |
| Separated-region weight decomposition | Alpha is labeled variational; Phase 6 includes unequal local approximation errors. | Derived biased-weight fixture; no mode-mass promotion. |
| Conditional multi-start coverage | Phase 3 uses blind independent seeds and preserves lineage identities. | Seed collision and duplicate-lineage report; no exhaustive-coverage claim. |
| Finite-query non-identification | The plan has no finite exhaustive-discovery gate. | Explicit nonconclusion in research ledger and Phase 10. |
| Proper bridge and unchanged cold objective | Phase 0 exposes the Gaussian-prior/likelihood decomposition and exact endpoints, and proves every q=20 intermediate law is proper from a bounded positive likelihood. | Source-bound covariance-weight/observation-variance receipt; `beta=0`, arbitrary beta, and `beta=1` value/score/status parity; finite stress draws are numerical only. |
| Frozen chart-kernel invariance | Phase 4 binds each frozen chart to an exact transformed target and active tuner. | Analytic fixtures plus learned-map self/cross inverse, Jacobian-cancellation, score, and conditioning screens before q=20 tuning. |
| Fixed kernel-mixture invariance and state-dependent counterexample | Gamma is fixed and state independent; uncorrected state-dependent selection is rejected. | Exact `pi K = pi` checks for fixed uniform/nonuniform gamma, configuration rejection, and exact two-state counterexample. |
| Replica swap detailed balance | Phase 5 uses the complete bridge cross-density ratio. | Direct product-ratio and forward/reverse swap fixtures. |
| Exact cold marginal | Phases 5 and 9 integrate the product kernel into the canonical sequential controller and expose only beta-one retained states. | Route ledger, continuation archive, cold-stream, and modern posterior diagnostics. |
| Exactness does not imply discovery | Phases 8 and 9 require predeclared replica travel, positive/negative declared-region transitions, hot-level region forgetting, initialization forgetting, and retained diagnostics. | MCSE-aware promotion gates distinct from implementation exactness; no formal basin or exhaustive-mode claim. |
| Independence MH correction | Separate optional arm only. | Density-ratio fixture and high-dimensional acceptance diagnostic. |

## Algorithm crosswalk

Algorithm C in the note maps to Phases 0--3 and 8. The plan adds the required
target-specific capacity/optimizer search, disjoint held-out selection, positive-
temperature branching ablation, GPU batching policy, and explicit no-invalid-row
filtering. These additions do not change the reverse-KL population objective.

Algorithm D maps to Phases 4--6 and 9. The plan adds repository constraints that
the theorem alone cannot supply: one tuning artifact per beta/chart identity,
active capability checks, route-ledger classification, canonical sequential
warmup and retained sampling, and symmetric invalid-path handling. These are
implementation conditions for the theorem's invariant-kernel assumptions.

## Skeptical audit

| Required question | Finding |
|---|---|
| Wrong baseline | Repaired. The ladder includes physical HMC, single cold NeuTra, matched physical replica exchange, single-chart tempering, cold ensemble, plain proposed, and enhanced proposed arms. |
| Proxy used as promotion | Repaired. Loss, latent moments, component distance, acceptance, swap rate, and alpha are explanatory. Exactness, retained diagnostics, travel, transitions, and downstream agreement have separate roles. |
| Missing stop conditions | Repaired. Target/bridge proof failure, exhausted blind-initialization repair, theorem-fixture contradiction after focused repair, unusable inverse/log determinant after bounded repair, and budget exhaustion are continuation vetoes. Candidate region locking is not silently upgraded to a direction veto. |
| Unfair comparison | Repaired. Arms share target/data/coordinates/status and the physical/charted replica routes share one bridge/swap implementation. Accounting separates target calls, transport cross-densities, cache recombinations, compile time, memory, and wall time. |
| Hidden assumptions | Recorded. Component count, ladder, branching, architecture, optimizer, batch, ESS/MCSE target, and attempt cap remain hypotheses. |
| Stale context | Repaired. The q=20 parameter target is `R^4`; the 60-dimensional filter state is internal. High-dimensional claims require Phase 10. |
| Environment mismatch | Repaired. TensorFlow/TFP, batch-native static shapes, XLA default, trusted GPU, and pre-initialization memory growth are hard gates. |
| Artifact answers the question | Repaired. Each phase has an exit artifact tied to engineering, numerical, sampler, or scientific evidence; a smoke cannot promote the method. |
| Existing code overclaimed | Repaired. The single-map tuner and diagnostic pure-power replica-exchange module are explicitly insufficient for multi-chart authority. |
| Particle circularity reintroduced | No. Primary training and validation use unconditional Gaussian draws and exact target calls. Invalid rows cannot be replacement-resampled until a batch passes. Replay remains historical/optional. |

## Source and code boundary

The source-derived pieces are narrowly identified: Hoffman et al. Section 2.2
equations (2)--(3) and Section 2.3 for reverse-KL NeuTra; Hukushima and Nemoto
Section II equations (2.1)--(2.7) for the product ensemble and swap; and Parno
and Marzouk Section 3.1, especially equation (21), for fixed transport-map MCMC.
The categorical mixture, separated-error weight formula, finite-query result,
and fixed multi-chart kernel composition are project derivations.

The current code was inspected rather than assumed. It contains the q=20
batch-native posterior, single-map matched reverse-KL trainer, invertible IAF,
single-map fixed-transport tuner, canonical sequential policy, and a diagnostic
pure-power replica-exchange implementation. It does not contain the proposed
proper-bridge multi-chart training and sampling route. The plan therefore calls
for new implementation instead of relabeling existing diagnostics.

## Readiness boundary

Phases 0--6 implementation and analytic/mechanics fixtures are complete. The
automatic trusted idle-GPU probes returned `no_idle_policy_permitted_gpu`, but
the user explicitly authorized a bounded shared GPU0 exception. Attempt 7
exposed and repaired an XLA GPU string-metadata gather. Fresh Attempt 8 then
passed the GPU/XLA mechanics smoke with memory growth; its manifest is
`phase7-mechanics-smoke/attempt-08-gpu0-shared/run_manifest.json` and records a
67,494,144-byte allocator peak. The mathematical note is independently accepted
by Claude; its plan findings were adjudicated and repaired. The fresh Phase 8
campaign subplan now supplies the bounded compute budget and freezes component,
ladder, architecture, batch, optional-arm feasibility, ESS/MCSE,
declared-region travel, checkpoint, and attempt-cap hypotheses. Its C0
compatibility and immutable-checkpoint gate has passed. Serious q=20 training
now uses the repository-default GPU launcher; two historical C1 approval
requests ended in approval-service HTTP 503 before evaluation and therefore say
nothing about GPU or candidate validity. Those denials remain preserved. The
launcher supplies one GPU, pre-import memory growth, fresh outputs, and runtime
provenance without an idle-probe dependency; no indirect GPU path is admissible.

## Execution alignment (2026-08-29)

The Phase 7 GPU0 manifest is the immutable launch snapshot at
`phase7-mechanics-smoke/attempt-08-gpu0-shared/`. It records finite endpoint
values/scores on GPU0, two components, a static batch greater than one, XLA and
TF32, memory growth, learned-map reliability v2, a finite fixed-chart
transition, and proper-replica health. Those facts close implementation
mechanics only. The current plan's
normalized-versus-unnormalized notation was repaired after that launch; the
manifest's original plan hash is retained so the evidence is not silently
rewritten. No result from this execution supports whitening, mode discovery,
posterior convergence, high-dimensional scaling, statistical ranking, or
default readiness.

The 2026-08-29 refresh preserves those boundaries and adds an exact pullback
density/score Gaussianization diagnostic. Unlike moments of the generated base
Gaussian, this diagnostic is not tautological, but it remains local to regions
reached by each chart. C0 passed 9 full complexity-target tests, 38 full
predictive tests, 39 focused route tests, and a 13-test post-repair checkpoint
suite. The C1 harness binds every checkpoint to target/data, backend, dtype,
XLA, seed derivation, and validation-bank identities and reconstructs a fresh
object before continuation.

The first C1 launch (`attempt-02-default-gpu`) reached the repository GPU
boundary and completed only the beta-0 chart-0 checkpoint before the measured
1,800-second timeout (`exit 124`). The bounded localization (`attempt-03`)
completed both B=8 target calls but timed out at B=256 beta=0. The final
small-bank feasibility retry (`attempt-04`) completed both B=8 charts and B=32
chart 0, then timed out before the complete B=32 receipt at 2,700 seconds.
Timeout records and immutable partial checkpoints are preserved beside each
attempt. The plan is therefore aligned with a real C1 budget/graph-limit
blocker; it does not claim a candidate failure or promote any partial run.
