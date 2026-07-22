# Complete High-Dimensional Leaderboard Master Program

Date: 2026-07-11

Status: `DRAFT_REVIEW_REQUIRED`

## Material Review Question

Does this master program define a consistent, feasible, fail-closed route to a
24-cell numeric leaderboard without promoting stale, scoped, proxy, or
source-ungrounded evidence?

## Objective

Produce a numerically complete high-dimensional value-and-score leaderboard
with four algorithms on six main observed-data rows, while preserving the
parameterized-SIR complete-data component as a sidecar.

The 24 main cells are the Cartesian product of:

- `fixed_sgqf`;
- `ukf`;
- `zhao_cui_scalar_or_multistate`;
- `ledh_pfpf_ot`;

and:

- `benchmark_lgssm_exact_oracle_m3_T50`;
- `zhao_cui_sv_actual_nongaussian_T1000`;
- `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`;
- `zhao_cui_spatial_sir_austria_j9_T20`;
- `zhao_cui_predator_prey_T20`;
- `zhao_cui_generalized_sv_synthetic_from_estimated_values`.

`zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` remains a scoped
sidecar. It is not a seventh four-way main row.

## Research Intent Ledger

| Field | Intent |
| --- | --- |
| Main question | Can every main-row/algorithm cell emit an admitted total log likelihood and its admitted total score for the same row target, with current-source LEDH evidence and source-anchored Zhao-Cui evidence? |
| Mechanisms | Current compact LEDH score routes; target-aligned SGQF and UKF score routes; fixed-variant Zhao-Cui source routes; fail-closed artifact assembly. |
| Expected failure modes | Stale target or source identity, prefix/single-seed promotion, missing analytical score, source-unfaithful Zhao-Cui extension, GPU/XLA failure, FD failure, memory failure, incomplete evidence, or stale leaderboard builder inputs. |
| Promotion criterion | All 24 main cells contain row-matched admitted total log likelihoods and their total scores; all six LEDH cells use current-source five-seed GPU/XLA/TF32 paired aggregates; `--require-complete` and integrity tests pass. |
| Promotion vetoes | Target substitution, wrong parameter order, historical/manual LEDH route, autodiff diagnostic promoted as an admitted score, scoped sidecar promoted to a main row, retained-grid Zhao-Cui route promoted to production, incomplete LEDH seed set, nontrusted GPU evidence, or failed artifact validation. |
| Continuation vetoes | Corrupt or unavailable target data; missing paper/author-source anchors for a required Zhao-Cui route; overlapping dirty-work conflict; human authority boundary; trusted infrastructure unavailable; five nonconvergent review rounds; or the active wall-clock limit. |
| Repair triggers | Fixable test/compile failure, numerical mismatch, missing artifact field, bounded evaluator implementation gap supported by the frozen target/source, or stale builder schema. |
| Explanatory only | Runtime, memory below budget, compile time, single-prefix metrics, per-seed descriptive differences, MCSE without a ranking design, and tiny fixtures. |
| Must not be concluded | Statistical or runtime superiority, HMC readiness, posterior correctness, calibrated confidence coverage, broad scientific validity, or production-default changes. |

## Binding Boundaries

1. A status matrix with blocked cells is truthful but not numerically complete.
2. Each score is the total derivative of the exact same value scalar reported
   in its cell. A partial, stopped, diagnostic-autodiff, or different-target
   derivative cannot be relabeled.
3. LEDH production evidence uses TensorFlow/TFP, GPU, XLA JIT, float32, TF32,
   seeds `81120..81124`, and the current owner FD-only rule
   `max_j(r_j) <= 0.05*sqrt(p)`.
4. The repaired Sinkhorn update-count helper changes current source identity.
   Historical LEDH score shards may explain but cannot close current admission.
5. Zhao-Cui implementation or approval requires paper/math and local author
   source anchors. Production work uses the fixed-variant source route. The
   generic retained-grid route is diagnostic only.
6. The parameterized-SIR component row is never promoted to the full
   observed-data fixed-SIR row.
7. Runtime cross-ranking remains disabled unless all arms are rerun under a
   separately reviewed synchronized timing design.
8. Historical artifacts are immutable. Every new run uses a new dated path.

## Cell Scalar And Aggregation Contract

The admitted cell value is the total observed-data log likelihood
`L(theta)`. The admitted score is `dL(theta)/dtheta` in the frozen parameter
order. `average_log_likelihood = L(theta) / T` may be shown as a derived display
field, but it is not the scalar paired with the unscaled total score.

For LEDH seed `s`, preserve the paired record `(L_s, dL_s/dtheta)` from the
same fixed-randomness value/score route. The released LEDH cell is

- `L_bar = arithmetic_mean_s L_s`; and
- `score_bar = arithmetic_mean_s dL_s/dtheta`;

over exactly seeds `81120..81124`. Linearity makes `score_bar` the derivative
of `L_bar`. Medians, independently aggregated values/scores, historical
forward values, or mixing different seed sets are forbidden.

Across the five shards, the canonical target signature, code-source hashes,
source-value/target artifact hash, parameter order and theta, transport and
numerical configuration, TensorFlow/XLA/TF32 settings, and command-manifest
schema must be identical. Only the declared execution seed, seed-derived
particles/noise/resampling tensors, output path, timestamps, timings, and
runtime memory may differ. The aggregate validator must compare these fields
directly and reject mixed-source or mixed-configuration seeds even when every
seed number is present.

Finite difference is validation only. It is never the admitted score. For
every seed and every coordinate `j`, recompute

`r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)`

and require `max_j(r_j) <= 0.05 * sqrt(p)`. All five seed-level policies must
pass individually; an aggregate pass cannot hide a failed seed. The 5% choice
mirrors the conventional 95% threshold only. It is not a computed confidence
interval or coverage claim.

Every seed/coordinate FD record must include finite minus/plus parameter
values, a nonzero realized denominator `plus_theta_j - minus_theta_j`, finite
minus/plus total-objective scalars, and the exact prepared-input fingerprint,
target signature, value-route id, execution seed, source/config hashes, and
numerical settings shared with the admitted score shard. The realized endpoint
parameters must be distinct in the active dtype. The validator recomputes the
objective numerator, realized denominator, and `FD_j` from those recorded
endpoint scalars and parameters; it does not trust a serialized FD value or
pass flag. Collapsed endpoints, different fixed randomness, a different value
route, or a different target/configuration are hard vetoes even if `score_j`
and the stored FD are both zero or near zero.

## Canonical Target Signatures

Phase 0 freezes the declared identities below. Phase 1 must materialize a
canonical row-target artifact for each row before any production run. That
artifact must hash the exact observation tensor, initial-state/time convention,
target-density and normalization semantics, evaluation theta and coordinate
order, and the code/config sources that generate them. All four algorithms in
a row must bind the same target signature. Algorithm-specific randomness,
particle count, quadrature settings, ranks, and transport settings belong in a
separate execution signature.

| Row | T | p | Evaluation coordinates | Data identity |
| --- | ---: | ---: | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | 50 | 5 | `phi1,phi2,phi3,q_scale,r_scale = 0.72,0.55,0.35,0.35,0.45` | deterministic dataset seed `81100`; exact observation hash required in Phase 1 |
| `zhao_cui_sv_actual_nongaussian_T1000` | 1000 | 2 | `gamma_unconstrained,log_beta = 0.2533471031357997,-0.916290731874155` | deterministic dataset seed `81101`; exact observation hash required in Phase 1 |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | 1000 | 2 | same coordinates and theta as actual-SV, but KSC target density | deterministic dataset seed `81101`; exact observation and mixture-constant hashes required in Phase 1 |
| `zhao_cui_spatial_sir_austria_j9_T20` | 20 | 3 | `log_kappa_scale,log_nu_scale,log_obs_noise_scale = 0,0,0` | fixed Austria j9 observations; exact observation hash required in Phase 1 |
| `zhao_cui_predator_prey_T20` | 20 | 6 | `r,K,a,s,u,v = 0.6,114,25,0.3,0.5,0.5` | deterministic dataset seed `81104`; exact observation hash required in Phase 1 |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | 1008 | 3 | `gamma_unconstrained,log_tau,mu = 1.0824113944610982,-2.076793740349318,0` | deterministic dataset seed `81105`; exact observation hash and transition-before-t0 convention required in Phase 1 |

The actual-SV and KSC rows remain different targets even though they share
data and evaluation theta.

## Candidate Re-Admission

The nine July 3 non-LEDH cells are frozen candidates, not admitted cells. Hash
identity proves which bytes were inspected; it does not prove correctness.
Phase 4 owns re-admission of these nine candidates. Before release, every
candidate must pass a per-cell ledger covering target
signature, total-value semantics, parameter transform/order, same-scalar
value-score pairing, derivative provenance, implementation/config hashes, and
finite value/score shape. Zhao-Cui candidates additionally require the
row-specific source-anchor ledger below. A failed candidate becomes a repair
task; it is never silently copied into the release.

## Final Release Dependency Manifest

Phase 8 must generate a final release manifest after all evaluator phases have
finished. It defines, for every row and algorithm:

- canonical target-signature hash;
- admitted cell-artifact hash;
- direct implementation, model, transform, derivative, configuration, and
  data dependencies with SHA-256 hashes;
- shared algorithm-source hashes and allowed row-specific source/config hashes;
- execution provenance and same-scalar status;
- source-anchor ledger hash for every Zhao-Cui cell.

The manifest builds a dependency graph from files/configs to cells. Any change
after a cell's last admission invalidates every transitively dependent cell.
Any computation-relevant target, data, model, transform, derivative,
implementation, or configuration hash change requires recomputing the affected
cell. Documentary re-admission without recomputation is allowed only for an
explicitly classified non-computational metadata change whose non-effect is
demonstrated and reviewed. Merely retaining an old implementation hash is not
sufficient.

Dependency completeness is itself a gate. Phase 8 must reconcile a reviewed
static import/dependency closure with runtime import/load records, generated
configuration inputs, dynamically loaded model/data paths, and every file
opened while producing or validating each cell. Unresolved dynamic loads,
unknown dependencies, missing transitive hashes, or disagreement between
declared and observed dependencies veto release. Phase 8 checks all 24 cells,
including the nine Phase 4 candidates, against this final manifest. After the
last Phase 8 or Phase 9 code/config/document patch, rerun dependency closure,
all affected computations, the completeness validator, and final artifact
hashing before sealing release. Phase 9 refuses any unbound, stale, or
incompatibly shared dependency.

## Zhao-Cui Route Identity

The display id `zhao_cui_scalar_or_multistate` does not define an
implementation. Before code or admission, every row must record a route id,
frozen configuration and source hash, paper section/equation anchors, local
author-source file/line anchors, and one classification:

- `source_faithful`; or
- `fixed_hmc_adaptation` that freezes the cited author route without changing
  it.

`extension_or_invention` cannot close a Zhao-Cui cell without explicit human
approval. The generic all-axes retained-grid route remains diagnostic only.
No Zhao-Cui phase may execute until its row-specific anchor ledger passes the
project source-anchor gate.

Before Phase 2 begins, Phase 1 must also create a six-row Zhao-Cui
anchor-availability and route-feasibility ledger. For each main row it records
the candidate fixed-variant route, paper/math location, local author-source
path availability, preliminary classification, and whether a source-grounded
route appears feasible without extension/invention. Missing or contradictory
anchors, an unavailable author source, or a route that already requires an
unapproved invention is an early continuation veto. This is an availability
screen, not final source-faithfulness approval; detailed line anchors,
configuration freeze, implementation review, and admission remain in Phases
4-7.

## Phase Index

| Phase | Name | Subplan | Required result |
| ---: | --- | --- | --- |
| 0 | Target, baseline, and authority freeze | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-result-2026-07-11.md` |
| 1 | Six-row LEDH schema-v4 harness | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-result-2026-07-11.md` |
| 2 | Full-time LEDH seed-81120 ladders | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase2-ledh-fulltime-seed81120-result-2026-07-11.md` |
| 3 | LEDH five-seed aggregation | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase3-ledh-five-seed-admission-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase3-ledh-five-seed-admission-result-2026-07-11.md` |
| 4 | Nine legacy non-LEDH candidate re-admissions | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase4-legacy-nonledh-readmission-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase4-legacy-nonledh-readmission-result-2026-07-11.md` |
| 5 | Fixed-SIR non-LEDH cells | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase5-fixed-sir-nonledh-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase5-fixed-sir-nonledh-result-2026-07-11.md` |
| 6 | Predator-prey non-LEDH cells | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase6-predator-prey-nonledh-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase6-predator-prey-nonledh-result-2026-07-11.md` |
| 7 | Generalized-SV non-LEDH cells | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase7-generalized-sv-nonledh-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase7-generalized-sv-nonledh-result-2026-07-11.md` |
| 8 | Final dependency manifest, generic builder, and completeness gate | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase8-builder-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase8-builder-result-2026-07-11.md` |
| 9 | Final release and closeout | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase9-release-subplan-2026-07-11.md` | `docs/plans/bayesfilter-complete-highdim-leaderboard-phase9-release-result-2026-07-11.md` |

Only Phase 0 and the next phase draft are created at launch. Each later subplan
is drafted or refreshed at the preceding handoff after its actual entry
evidence exists.

## Program Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the six-row by four-algorithm main matrix be completed without changing targets or promoting diagnostic evidence? |
| Exact baseline | SHA-frozen July 3 non-LEDH candidates; six July 7 LEDH artifacts used only as frozen target/shape and historical forward-value evidence; current score/FD and transport source hashes frozen by Phase 0. No July artifact supplies a released LEDH cell. |
| Primary criterion | Exactly 24 admitted main cells, six current-source paired LEDH five-seed aggregates, sidecar separated, generic builder fail-closed completeness pass, integrity tests, release result, and final review. |
| Veto diagnostics | Binding boundaries above, plus any source/artifact hash mismatch or unsupported score provenance. |
| Explanatory diagnostics | Runtime, memory, prefix results, tiny tests, and descriptive stochastic variation. |
| What will not be concluded | Ranking, superiority, HMC/posterior correctness, confidence coverage, default promotion, or source-faithfulness without checked anchors. |
| Preserved artifacts | Phase subplans/results, reviews, ledgers, manifests, logs, row artifacts, final JSON/Markdown, and isolated-run export bundle. |

## Skeptical Plan Audit

| Risk | Finding and control |
| --- | --- |
| Wrong baseline | July 6 LEDH leaderboard embeds superseded score artifacts. It is historical only. The Phase 0 generator reads every non-LEDH cell directly from the SHA-frozen July 3 baseline and records that source per cell; only LEDH historical status is read from July 6. |
| Proxy promotion | Prefix, singleton-seed, score-only, memory-only, smoke, and tiny results cannot close cells. |
| Missing stop conditions | Every subplan separates promotion veto, continuation veto, and repair trigger. Candidate failure does not stop unrelated rows. |
| Unfair comparison | The program fills target-matched cells; it does not runtime-rank frozen and fresh arms. |
| Hidden assumptions | Phase 0 freezes row scope, parameters, seeds, source hashes, missing-cell matrix, and sidecar status; Phase 1 must add byte-level canonical target signatures before runtime. |
| Stale context | Current source includes the repaired manual Sinkhorn update count and endpoint-rich FD schema. Older score claims are not reused. |
| Environment mismatch | Production LEDH runs require trusted GPU/XLA/TF32 evidence. CPU-hidden checks are engineering/reference only. |
| Artifact insufficiency | Every serious run requires structured terminal artifacts, run manifests, canonical target and execution signatures, paired values/scores, hashes, and immutable paths. |
| Detached execution | The visible runbook cannot authorize detachment. A separate isolated-supervisor plan controls launch, export, and merge-back. |
| Unbounded research | The overnight run has an eight-hour cap. It may stop with a direct blocker rather than fabricate a complete result. |

Audit decision: `REVISE_THEN_PASS_WITH_PHASE_GATES`. The first review found
missing target-signature, re-admission, per-seed FD, aggregation, and Zhao-Cui
route contracts. The controls above repair those design gaps. No later phase
may execute before its own reviewed subplan and entry conditions exist.

## Execution Policy And Eight-Hour Budget

New TensorFlow algorithmic benchmark paths default to trusted GPU/XLA as
required by repository policy. A CPU-only analytical/reference check must be
explicitly reviewed and cannot support a GPU/default-readiness claim.

The detached eight-hour cap is a continuation veto, not a promise that all
phases fit. Soft allocation is: Phase 1 `60m`; Phase 2 `100m`; Phase 3 `120m`;
Phase 4 `45m`; Phases 5-7 combined `105m`; Phase 8 `25m`; Phase 9 `25m`.
Every per-seed and per-cell terminal artifact is an immutable checkpoint. A
phase approaching its budget writes a resumable result and handoff before the
outer timeout. Timeout is reported as `STATUS_COMPLETE_WITH_BLOCKERS`, not
infrastructure failure and not numeric completion.

## Final Completeness Validator

`--require-complete` must reject anything other than exactly 24 unique main
row/algorithm keys. Every cell must have a finite total scalar, a finite score
vector of exactly `p` coordinates in frozen order, the canonical target
signature, admitted provenance, implementation/config hashes that equal its
final dependency-manifest entries, no dependency invalidation, and same-scalar
status. LEDH cells additionally require the exact five paired seeds, five
individual FD passes reconstructed from valid noncollapsed endpoint records,
trusted GPU/XLA/float32/TF32 provenance, and current source hashes. Zhao-Cui cells additionally require admitted source anchors and
route classification. The validator must reject duplicate/missing keys,
nonfinite values, wrong dimensions/order/signatures, historical or diagnostic
provenance, unbound dependency closure, and any sidecar contamination.

## Review And Repair Loop

Codex is supervisor and executor. Claude is read-only reviewer only. Material
plans, implementations, results, source-faithfulness decisions, and final
release use the smallest-path review protocol and end with `VERDICT: AGREE` or
`VERDICT: REVISE`.

For one material blocker, allow at most five rounds:

1. record and classify the finding;
2. patch only actionable issues;
3. rerun focused local checks;
4. rerun the narrowest review;
5. stop for human direction if no convergence occurs.

Claude agreement is advisory and never overrides local gates or human
boundaries. A trusted Claude health failure may trigger a fresh read-only Codex
substitute review, which must be labeled weaker evidence.

## Program Stop Conditions

- A required row target, data artifact, paper anchor, or author source is
  missing or contradictory.
- A proposed Zhao-Cui production route is `extension_or_invention` without
  explicit human approval.
- Continuing requires changing thresholds, targets, seeds, coordinates,
  defaults, public APIs, funding, credentials, packages, or product scope.
- A dirty-work overlap cannot be isolated safely.
- Trusted GPU or review infrastructure remains unavailable after the reviewed
  bounded recovery procedure.
- The same blocker fails to converge after five review rounds.
- The active visible or detached wall-clock limit expires.

## Completion Classification

- `NUMERICALLY_COMPLETE`: all 24 admitted main cells and all release gates pass.
- `STATUS_COMPLETE_WITH_BLOCKERS`: every cell has a direct executed or blocked
  status, but at least one numeric cell remains absent.
- `BLOCKED_INVALID_EVIDENCE`: a target, source, harness, or artifact invariant
  is invalid.

Only the first classification permits the phrase "complete numeric
leaderboard".
