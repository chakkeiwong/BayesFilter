# SGQF Whole High-Dimensional Leaderboard Repair Master Program

Date: 2026-07-22

Status: `AUDITED_READY_FOR_PHASE0_TARGET_RESET; NUMERICAL_PROMOTION_BLOCKED`

## Purpose And Governing Scope

Make the SGQF side of the canonical high-dimensional leaderboard honest and
operational for every row emitted by
`docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py`.

"Working" is row-contract aware. It does not mean forcing every cell to
`executed_value_score`. It means that every in-scope SGQF cell is in exactly one
of these terminal states:

1. `executed_value_score` when the row has free parameters and a same-scalar
   manual analytical score is mathematically defined and validated;
2. `executed_value_only` when the canonical row has no free parameter;
3. `not_applicable` when the row is explicitly a scoped component for another
   algorithm rather than a cross-algorithm filtering comparison; or
4. `blocked` only after a concrete implementation, numerical, source, or
   evidence veto is preserved with the smallest next repair.

The primary implementation objective is to eliminate avoidable SGQF blockers.
The program does not make Zhao--Cui or UKF repair a prerequisite for an honest
SGQF-complete column.

This program operationally supersedes
`docs/plans/bayesfilter-sgqf-highdim-leaderboard-completion-master-program-2026-07-01.md`.
The July 1 artifacts remain historical evidence. Their mandatory review chain,
launch gates, and July 1 artifact authority are retired by the current
`AGENTS.md` proportional-governance policy.

## Skeptical Audit Of The July 22 Predator-Prey Handoff

Audited artifact:

`docs/plans/bayesfilter-predator-prey-sgqf-leaderboard-repair-agent-handoff-2026-07-22.md`

### Verdict

`NOT_SENSIBLE_AS_WRITTEN; LOCAL_TARGET_REPAIR_ONLY; SOURCE_SCOPE_RESET_REQUIRED`

The central diagnosis is correct only for the current BayesFilter seed-81104
target. That dataset observes the initial state at `y0` and transitions only
for `y1` onward, so the generic transition-first SGQF value `-171.368581` is
wrong relative to that local target. The corrected recurrence in
`bayesfilter/testing/predator_prey_sgqf_neutra_target_tf.py` assimilates `y0`
once, starts the recurrence at index 1, and differentiates the same finite
likelihood scalar, producing about `-103.13789` at the truth point.

However, the row is named `zhao_cui_predator_prey_T20` and is emitted in a
`highdim_source_scope` lane. The checked Zhao--Cui author code does something
different: `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m`
allocates `X(:,1)=x0`, then for `t=1:T` generates `X(:,t+1)` with
`st_process` and `Y(:,t)` from that transitioned state. Thus the source program
has 20 transition-then-observe steps and no observation of `x0`. The handoff's
T=1 zero-transition/zero-score gate is wrong relative to that declared source
target. The current `-103.13789` route remains valid implementation evidence for
the local initial-observation-first target, but it cannot be promoted as a
source-faithful Zhao--Cui leaderboard cell.

### Audit Findings

| Severity | Finding | Required correction |
| --- | --- | --- |
| Critical | The proposed initial-observation-first route is wrong relative to the checked Zhao--Cui author-code target, which performs `X(:,t+1)=st_process(...)` then `Y(:,t)=ob_process(...,X(:,t+1))` for every `t=1:T`. | Block source-scope promotion, preserve the local route under an explicit amended-target identity, and rebuild the source-scope dataset/evaluator as transition-then-observe. |
| Critical | The same time-index drift affects current source-named SV and fixed-SIR dataset generators: they call BayesFilter simulators that emit `y0` while the shared Zhao--Cui `ssmodel.complete` emits `y1:T` after transitions. | Audit every source-named row before preserving a baseline; regenerate or relabel affected targets and rerun all same-row algorithms on one frozen dataset. |
| Critical | The handoff repairs only `zhao_cui_predator_prey_T20`; it cannot make the whole SGQF leaderboard work while fixed SIR and generalized SV remain blocked. | Use the row-by-row program below and treat predator-prey as one bounded phase. |
| Critical | The current readiness schema requires all three algorithms to emit `executed_value_score`, even for the fixed-parameter SIR row where no score exists and for a scoped Zhao-Cui-only component row. Completion is unattainable without false claims. | Replace the single `full_three_way_ready` rule with applicability-aware value and score completeness. |
| High | The handoff treats the July 1 artifact as the current authority although a July 3 pair exists and the live runner now has a seventh parameterized-SIR scoped row. | Freeze the live runner, current worktree, and latest valid artifact pair at Phase 0; record historical artifacts by date rather than declaring July 1 current. |
| High | The handoff asks for an identity binding target and data while also asking for `T=1` and `T=2` tests. The canonical factory should reject noncanonical data, so tiny-horizon mechanics tests need a separate non-admitted test constructor. | Separate the sealed canonical factory from an internal mechanics kernel/test fixture. Never weaken canonical data validation for tests. |
| High | The current corrected endpoint consumes six-probit coordinates and obtains a static Python batch size with `int(values.shape[0])`. A physical-coordinate leaderboard wrapper and supported batch-shape contract do not yet exist. | Make physical-coordinate value/manual-score and fixed/dynamic batch support explicit implementation and test gates. |
| High | The handoff does not separate the current in-progress GenUT edits from SGQF ownership, although the shared leaderboard and two tests are dirty. | Capture a pre-edit diff/hash manifest and patch only SGQF-owned blocks; fail and rebase the phase if shared anchors change unexpectedly. |
| High | The same-target level-5 comparison is necessary but is not an independent correctness authority because levels share the recurrence and derivative algebra. | Use exact affine/T=1/T=2 checks, independent same-scalar FD, value-only versus score-scalar parity, PF value diagnostics, and source-grounded timing checks in addition to the level ladder. |
| Medium | The proposed `0.25` value and `0.5` score-coordinate gates are inherited screening limits and are orders of magnitude looser than the observed level-2/5 gaps. | Predeclare two gates: regression limits derived from fresh CPU/GPU reproducibility and broader viability limits. Do not silently use the broad viability screen as numerical agreement evidence. |
| Medium | "PF interval expanded by a practical margin" is vulnerable to post-hoc acceptance, and agreement at the truth point does not validate a likelihood surface. | Freeze the exact interval rule before execution, use all frozen audit points where PF artifacts exist, and classify PF as value-only external diagnostic rather than promotion oracle. |
| Medium | GPU/XLA readiness is mixed into scientific promotion. GPU placement and memory growth are engineering gates, not evidence that the scalar is scientifically correct. | Maintain separate engineering, numerical, and scientific ledgers. |
| Medium | Source-line closure is useful provenance but does not make Python source immutable; ordinary Git and SHA-256 are sufficient under current policy. | Use repository-issued identity plus Git commit/diff hash and dependency hashes. Do not build elaborate custom immutability machinery. |
| Medium | The existing runtime status hard-codes `innovation_condition_estimate = 1`, so that field is not a real conditioning diagnostic. | Either compute a meaningful estimate or remove it from promotion criteria and label it unavailable. |
| Medium | The handoff demands source-faithful SGQF claims but the local source cache lacks Jia--Xin--Cheng (2012) and Singh et al. (2018) full text. | Fetch and inspect the primary SGQF technical sources before making source-faithfulness claims; implementation correctness may proceed from project derivation and tests. |

### What The Audit Confirms

- For the local seed-81104 initial-observation-first target, `y0` assimilation
  is parameter-independent and its physical likelihood score contribution is
  zero.
- The manual recurrence includes state and direct parameter dependence through
  RK4, Cholesky derivatives, predicted and observation moments, innovation
  covariance, cross covariance, gain, filtered mean, and all three covariance
  update derivative terms.
- The likelihood-only endpoint is separate from prior and chart-Jacobian terms.
- The checked Zhao--Cui paper fixes the predator-prey parameters, noises, RK4
  step, prior box, and T=20, while the checked author code fixes the unambiguous
  transition-then-observe indexing used by the executable source program.
- The shared author-code indexing also applies to the paper SV and SIR
  experiments; current BayesFilter `simulate(final_time=T-1)` fixtures instead
  include the initial observation and only `T-1` transitions.
- The checked July 15 result and target-identity hashes match their handoff
  values.
- The checked July 22 GenUT/PF diagnostic artifact hash matches its handoff
  value.
- Historical evidence supports preserving and testing the local amended route,
  but cannot unblock a source-scope leaderboard cell.

## Research Intent Ledger

| Field | Program definition |
| --- | --- |
| Main question | Can every applicable canonical high-dimensional SGQF row execute the exact declared finite SGQF value program, with a manual same-scalar score where the row has free parameters, without changing row targets? |
| Candidate mechanisms | Source-target audit and reset; existing SGQF routes only after same-target requalification; model-specific transition-then-observe predator-prey and fixed-SIR routes; a native raw-observation generalized-SV SGQF Gaussian-projection route. |
| Expected failure modes | Wrong time order, cross-row dataset reuse, posterior/likelihood mixing, observation-model mismatch, signed-weight covariance loss, insufficient generalized-SV Gaussian closure, static-shape/XLA failure, or stale shared-file integration. |
| Promotion criterion | Each applicable row passes its row contract, same-scalar/value gate, numerical validity gate, route identity gate, and required CPU/GPU-XLA checks. |
| Promotion veto | Wrong target/data/time order; non-manual score admitted; score for a no-free-parameter row; posterior terms in a likelihood cell; nonfinite or invalid covariance; caller-stamped identity; or a comparator from another target treated as authority. |
| Continuation veto | The target or dataset cannot be reconstructed; required primary data are unavailable; the proposed SGQF computation is mathematically undefined; the campaign budget is exhausted; or fixing the row would require changing its scientific target. |
| Repair trigger | A candidate route or gate fails while the target, data, math, and harness remain valid. Continue to the smallest planned repair or preserve a precise blocker. |
| Explanatory diagnostics | Runtime, point count, PF/UKF/TT gaps, GPU speed, Cholesky eigenvalue margins, and historical HMC results. |
| Must not be concluded | Exact nonlinear likelihood, exact posterior, SGQF superiority, universal sparse-grid convergence, production/default readiness, HMC readiness, or Zhao--Cui correctness. |

## Canonical Row Taxonomy And Terminal Contract

The Phase 0 inventory must generate this table from the live runner and then
fail if the runner contains an unclassified row.

| Row | Scope | Free theta | Required SGQF terminal state | Planned route |
| --- | --- | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | main filtering | yes | preserve `executed_value_score` | existing direct affine fixed-SGQF |
| `zhao_cui_sv_actual_nongaussian_T1000` | source-named main filtering, declared Gaussian-closure lane | yes | requalify after source-time-order audit; then `executed_value_score` | existing route on a newly frozen source-consistent dataset, or relabeled amended row |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | source-named main filtering, KSC surrogate | yes | requalify after source-time-order audit; then `executed_value_score` | existing independent-panel route on the same audited SV timing convention |
| `zhao_cui_spatial_sir_austria_j9_T20` | main filtering, paper fixed parameters | no | `executed_value_only`; score `not_applicable_no_free_theta` | new fixed-parameter level-2 axis-cloud SIR SGQF |
| `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` | scoped local complete-data Zhao--Cui component | not applicable to SGQF comparison | SGQF `not_applicable`, excluded from SGQF column denominator | no SGQF route; preserve scope |
| `zhao_cui_predator_prey_T20` | source-named main filtering | yes, physical `(r,K,a,s,u,v)` | `executed_value_score` | new transition-then-observe T20 level-2 SGQF manual score; local initial-observation-first route stays separate |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | source-route scalar `svmodels` prior-mean amendment | yes, active `(gamma,tau,mu)` coordinates | `executed_value_score` if the source-route SGQF evaluator is viable; otherwise evidence-backed blocker | new scalar raw-observation SGQF with manual score; native two-state reference is separate |

The fixed SIR route must not reuse
`bayesfilter/testing/sir_filter_neutra_target_design_tf.py` by name alone. That
module targets a BayesFilter three-log-scale extension, seed 81120, and a
posterior adapter. The canonical fixed-paper row has fixed `kappa=0.1`,
`nu=18`, no free theta, and the live row dataset identity. Shared mechanics may
be extracted only after an explicit target/data comparison.

## Applicability-Aware Leaderboard Schema

Replace the current all-score definition with these independent fields:

| Field | Meaning |
| --- | --- |
| `algorithm_applicability` | `applicable`, `not_applicable_scoped_row`, or `blocked_pending_evidence` |
| `required_result_kind` | `value_score`, `value_only_no_free_theta`, or `not_applicable` |
| `value_complete` | Every applicable algorithm required for this row has an admitted value. |
| `score_complete` | Every applicable algorithm with a free-theta score requirement has an admitted manual/analytical score. |
| `comparison_ready` | Required applicable cells meet their declared result kind; not-applicable cells do not count against readiness. |
| `sgqf_column_complete` | Every SGQF-applicable row meets its SGQF result kind, independent of Zhao--Cui/UKF blockers. |

Keep `full_three_way_ready` as a deprecated compatibility field for one release
if downstream consumers require it, but do not use it as the SGQF completion
criterion.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the live runner produce a complete, applicability-aware SGQF column from repository-owned TensorFlow/TFP routes without stale/caller-stamped identity? |
| Exact baselines | Affine LGSSM/Kalman equality; source-code transition-count checks; source-route T=1 one-transition checks; local amended-route analytic initial-observation checks; frozen dataset/hash/time-order contracts. |
| Deterministic approximation comparator | Higher SGQF levels or denser quadrature on the identical row scalar. This tests refinement, not absolute correctness. |
| External value diagnostics | Refined bootstrap PF where available, native dense generalized-SV reference on feasible prefixes, and UKF same-target diagnostics. They are not score oracles. |
| Score criterion | Manual recursive score equals central FD of the independent value-only endpoint at predeclared interior points and step sizes; endpoint scalar equality also passes. |
| Hard vetoes | Target/data/timing mismatch; invalid identity; nonfinite scalar/score; covariance or status failure; score-coordinate mismatch; runtime autodiff/FD score; CPU fallback in a GPU claim; source-row substitution; or a value-only/no-theta row forced to emit a score. |
| Explanatory only | Runtime, PF point gap, UKF gap, TT gap, sparse point count, and historical HMC/NeuTra artifacts. |
| Artifacts | One versioned campaign root with inventory, per-row result/manifest, raw diagnostic rows, hashes, and a final JSON/Markdown leaderboard pair. |

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- | --- |
| TensorFlow/TFP, float64 | repository policy and existing SGQF routes | reviewed baseline | device kernel drift or high cost | CPU/GPU-XLA parity and allocator manifest |
| XLA enabled for claim routes | repository policy | engineering requirement | unsupported dynamic shape or silent CPU fallback | tiny compiled device/shape smoke before full row |
| Sparse level 2 for two-state predator-prey | July 15 fresh-hash historical ladder | strong warm start | shared bias or source drift | reproduce levels 2/3/5 on current route |
| Level-2 axis cloud for 18-state fixed SIR | existing SIR-SGQF design | hypothesis, not current row default | negative center weight or high interaction order gives misleading closure | affine/moment checks, level/dense diagnostic if feasible, PF/UKF value comparison |
| Native generalized-SV Gaussian projection | project SGQF definition | hypothesis | non-Gaussian raw observation makes one-Gaussian closure inaccurate | T=1/T=2 dense-reference ladder before full T |
| Predator-prey physical score | leaderboard row contract | required | source-probit score or posterior terms leak into row | direct physical FD and analytic chart-chain tests |
| Fixed SIR value-only | Zhao--Cui Section 6.3 fixes parameters | required target fact | artificial parameter extension changes the row | target/data manifest comparison |
| Existing three admitted SGQF rows | live runner and prior artifacts | preserved baselines, not exempt from smoke | current source drift breaks old status | focused regression before new row work |
| `highdim_source_scope` label | live leaderboard and June 11 source contract | disputed until reset | local initial-observation datasets are presented as author-source rows | compare paper, author code, generator, hashes, and every algorithm's time loop |
| Numerical regularization | existing routes currently use direct Cholesky/status | no new default authorized | jitter changes scalar/derivative branch | fail closed first; any repair requires explicit branch identity and same-scalar retest |

## Phase Program

### Phase 0: Source-Target Reset, Live Inventory, Coexistence, And Schema Repair

1. Freeze Git commit, dirty paths, hashes/diffs for shared leaderboard/test files,
   Python/conda/TensorFlow/TFP versions, live row list, latest valid leaderboard
   pair, and all row/data identities.
2. For every source-named row, create a source-target crosswalk covering paper
   notation, author-code state/observation indexing, local data generator,
   state/observation counts, seed, hashes, parameter coordinates, and every
   algorithm's likelihood loop. Inspect the checked author code, not only old
   project plans.
3. Classify any mismatch directly. For affected SV, SIR, and predator-prey
   rows, choose the action determined by the existing lane label: keep the
   source-named row and regenerate transition-then-observe data, while
   preserving old artifacts under explicit
   `bayesfilter_initial_observation_first_amended_target` status. Renaming the
   source row or changing its target is a human project-direction boundary.
4. Recompute target/data identities and mark all prior values, scores, PF/UKF/
   TT comparators, GenUT results, and SGQF level ladders from the old data as
   historical amended-target evidence. They cannot validate the reset row.
5. Classify every row with `scope`, `free_theta`, `algorithm_applicability`, and
   `required_result_kind`.
6. Implement applicability-aware readiness and tests before any row is
   promoted.
7. Record that GenUT code, values, scores, artifacts, and tuning are out of
   scope. Integrate only against stable shared-file anchors.
8. Preserve old artifacts without completing retired July 1 review ceremony.

Gate: the source-target crosswalk has no unexplained indexing mismatch; affected
source rows have fresh identities and explicit historical/amended-target
classification; the row taxonomy is exhaustive; no score is required for fixed
SIR; and the scoped parameterized-SIR component cannot make the SGQF column
incomplete.

### Phase 1: Requalify Existing SGQF Baselines

Preserve affine LGSSM after focused value/score/same-route regressions. For
actual SV and KSC SV, first run the audited time-order/data reset from Phase 0,
then execute every compared algorithm on the same reset dataset. Verify each
emitted row states the exact target or surrogate scope it computes. Do not
copy old initial-observation-first values into the source-consistent row.

Gate: LGSSM preserves its status; actual/KSC SV obtain fresh source-consistent
identity and value/score evidence or are demoted to explicit historical amended
targets. A failure triggers localized repair before using a row as an anchor.

### Phase 2: Predator-Prey Dual-Target Separation And Source-Row Admission

1. Preserve the corrected initial-observation-first likelihood/manual-score
   kernel for the existing NeuTra/GenUT target. Move reusable mechanics to a
   model-specific non-testing TensorFlow module and keep the testing adapter as
   a thin compatibility wrapper. Its route identity must say
   `bayesfilter_initial_observation_first_20_observations_19_transitions`; it is
   not the source-scope leaderboard route.
2. Build the source-row value/manual-score program for 20
   transition-then-observe steps. It begins from the Gaussian initial prior,
   propagates before the first observation, and performs exactly 20 transitions
   and 20 observation updates. Reuse derivative mechanics, not the wrong loop
   boundary.
3. Freeze new source-consistent observations and reference artifacts under a
   new target identity. Match the author-code random-number law exactly only if
   cross-language source replay is a declared goal; otherwise classify the
   TensorFlow seed as a source-model synthetic replication, not the author-code
   `rng(1)` dataset.
4. Provide a sealed canonical factory for the reset T20 observations and level 2.
   It issues, rather than accepts, route identity. Bind row id, observation and
   state hashes, seed, timing, physical parameter order and bounds, cloud
   manifest/hash, dtype, backend, XLA policy, value/score callable closure, and
   Git/source hashes.
5. Provide physical likelihood-only value and physical manual-score endpoints.
   Convert physical theta to probit source coordinates analytically and apply
   `d ell/d theta = (d ell/d z)/(d theta/d z)` only for strict interior points.
   Boundary points fail with a clear domain status.
6. Keep internal mechanics constructors for T=1/T=2 tests. They must be marked
   non-admitted and cannot issue the canonical route id.
7. Support a declared leading batch shape. Remove or explicitly constrain the
   static Python batch-size assumption and test the chosen contract under XLA.
8. For the source route, add T=1 exactly-one-transition, T=2 exactly-two-
   transition, T=20 exactly-20-transition, value/score scalar equality,
   likelihood versus posterior separation, level ladder, FD, permutation,
   no-runtime-autodiff, covariance/status, identity rejection, CPU, and trusted
   GPU/XLA tests. Keep the old analytic-`y0` and T=1-zero-transition tests only
   for the explicitly named local amended route.
9. Use a runtime sentinel around the complete public score call graph, not only
   an AST scan of one function.

Fresh numerical gates must be frozen before the run. At minimum:

- regression gate: the local amended route reproduces the checked historical
  values/scores; the source route has a new, independently frozen regression
  baseline and must not be compared numerically to `-103.13789`;
- viability gate: level 2 remains within the predeclared broad level-5 limits;
- independent derivative gate: physical-coordinate FD passes at the truth and
  at least three frozen interior audit points over a step-size ladder;
- external value diagnostic: run a fresh transition-then-observe refined PF
  on the reset observations. The old PF interval is recorded as wrong-target
  history and is forbidden as a promotion gate;
- GPU claim gate: verified memory growth, physical/logical GPU, XLA, output
  placement, allocator current/peak bytes, and CPU/GPU numerical parity.

Gate: emit `executed_value_score` only when all target, value, score, identity,
and engineering gates pass.

### Phase 3: Fixed-Parameter Austria SIR SGQF Value Route

1. Freeze the paper/source row: Section 6.3 fixes `kappa_j=0.1` and `nu_j=18`,
   uses state `(S1,I1,...,S9,I9)`, RK4 step `0.005` over observation spacing
   `0.02`, process covariance `I18`, observes infectious coordinates with
   covariance `100 I9`, and uses T=20 after transitions.
2. Replace the current seed-81103 initial-observation-first fixture for this
   source row with an audited 20-transition/20-observation fixture, issue fresh
   hashes, and invalidate old same-row numeric evidence. Do not reuse the
   seed-81120 three-log-scale NeuTra route or its posterior identity.
3. Extract only proven reusable SIR mechanics if useful; build a sealed
   fixed-parameter likelihood-only SGQF route with the canonical row identity.
4. Validate moment/cloud construction, transition-then-observe timing, T=1 one
   transition, deterministic replay, covariance/status, and absence of a score.
5. Compare the finite value to the same-row UKF and, if affordable, a bounded
   bootstrap-PF value diagnostic with uncertainty. Treat differences as
   descriptive unless an uncertainty-supported criterion is predeclared.
6. Test GPU/XLA and memory growth for the 37-point, 18-dimensional level-2 axis
   cloud. Do not import retained-grid/all-pairs transition requirements from a
   different algorithm: SGQF propagates a fixed small cloud through the RK4 map.

Gate: emit `executed_value_only`, `score_status=not_applicable_no_free_theta`,
and make the row comparison-ready under a value-only contract.

### Phase 4: Source-Route Generalized-SV SGQF Route

This is the highest scientific-risk phase and must begin with the smallest
diagnostic.

1. Freeze the exact source/amended classification for
   `GeneralizedSVPriorMeanSSM`, seed 81105, scalar `svmodels` raw-observation
   equation, active `(gamma,tau,mu)` coordinates, horizon, time order, and data
   hashes. The prior-mean convention is an explicit source-row amendment
   recorded by the June 11/29 contracts, not an unqualified reproduction of a
   posterior estimate. Keep actual transformed SV and KSC surrogate SV
   separate.
2. Do not use `NativeGeneralizedSVSSM` as the source-row comparator: it is a
   distinct two-state project fixture with five parameters and its own dense
   raw-y oracle. It may remain a separate explanatory diagnostic only. The
   source-row comparator must be a scalar `svmodels` value program with the same
   transition/observation timing, or a source-consistent independent reference
   that is explicitly labeled as such.
3. Define the SGQF scalar explicitly for the scalar source route: Gaussian
   projection of the one-dimensional transition followed by sparse-grid
   evaluation of the raw-observation mean/variance, innovation increment, and
   filtered moments. This is an approximate SGQF likelihood, not an exact
   source posterior or TT reproduction.
4. Derive the complete manual score, including parameter dependence in the
   state transition, state-dependent observation variance, factors, gain, and
   filtered moments. A partial derivative or autodiff score is wrong for the
   claim.
5. Implement T=1 and T=2 first. Compare SGQF value to the independent
   scalar-source value program over a quadrature-order refinement ladder, and
   compare the manual score to FD of the SGQF value-only scalar. Do not use a
   two-state native dense score as source-row truth.
6. Proceed to the full frozen horizon only if the tiny-horizon scalar is finite,
   stable under SGQF level refinement, manual FD passes, and the approximation
   gap is within a predeclared viability threshold. A poor candidate result is
   a repair trigger, not evidence that the harness or research direction is
   invalid.
7. If Gaussian closure is descriptively inadequate, try only predeclared SGQF
   refinements that preserve the same scalar source route. Do not substitute a
   two-state native fixture, transformed residuals, or the KSC mixture row.

Gate: emit `executed_value_score` only for the explicitly named scalar
source-route raw-y SGQF approximation with valid manual same-scalar score.
Otherwise preserve a scientifically classified blocker and the smallest repair
result.

### Phase 5: Integration, Isolated Row Runner, And Final Regeneration

1. Add a row-selective SGQF runner so validation does not require executing
   every slow UKF/Zhao--Cui cell. The full final runner may consume frozen,
   identity-validated per-cell artifacts where the schema permits; it must not
   silently use stale July 1 values.
2. Wire only admitted row factories into the canonical runner.
3. Emit canonical value/score route identity, result kind, score coordinate,
   manual provenance, cloud level, data/target identity, CPU/GPU/XLA claim
   status, diagnostics, and nonclaims.
4. Run focused completeness, analytical-score, target-alignment, and artifact
   schema tests, then one bounded full regeneration.
5. Inspect JSON and Markdown for row counts, duplicates, nulls, stale blocker
   phrases, contradictory scopes, false `full_three_way_ready`, and identical
   JSON/Markdown interpretation.

Gate: `sgqf_column_complete=true`; every SGQF-applicable row meets its required
result kind; no not-applicable row counts as blocked; and every remaining
blocker is a real recorded veto rather than missing wiring.

### Phase 6: Terminal Review And Reset Memo

Perform one independent terminal review of the final plan/results, code anchors,
and artifacts. Fix material scientific or engineering findings within the
remaining budget. Write a reset memo and final result note with separate
engineering, numerical, and scientific decisions plus the required stochastic
inference-status table.

## Test Matrix

| Gate | LGSSM | Actual SV | KSC SV | Fixed SIR | Predator-prey | Generalized SV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen target/data identity | preserve | reset/audit | reset/audit | reset required | reset required | amendment audit required |
| Exact/tiny-horizon oracle | affine exact | source-time-order check | source-time-order check | T=1 one transition | source T=1 one transition; local route exact y0 separately | T=1/T=2 dense reference |
| Same scalar value/score | required | required | required | N/A no theta | required | required |
| Manual score versus FD | required | required | required | N/A no theta | physical coordinates | full native-raw-y SGQF scalar |
| Level/refinement ladder | existing | existing | existing | level/axis diagnostic | levels 2/3/5 | SGQF level plus dense-order ladder |
| CPU reference | required | required | required | required | required | required |
| Trusted GPU/XLA | smoke/regression | smoke/regression | smoke/regression | required | required | required if promoted |
| Runtime autodiff sentinel | score route | score route | score route | N/A | required | required |
| Repository-issued identity | required | required | required | required | required | required |
| Leaderboard result kind | value+score | value+score | value+score | value only | value+score | value+score or blocker |

## Campaign Budget And Stop Conditions

Authorization is the user's request to create and execute this program. No
additional local launch token is required.

| Item | Budget |
| --- | --- |
| Total serious campaign attempts | 10 fresh versioned attempts across all rows |
| Predator-prey claim attempts | at most 2 after routine smokes |
| Fixed-SIR claim attempts | at most 2 after routine smokes |
| Generalized-SV claim attempts | at most 4, including refinement repairs |
| Final full leaderboard regenerations | at most 2 |
| Aggregate trusted GPU wall time | 8 hours |
| Aggregate CPU diagnostic wall time | 8 hours; any single command capped at 2 hours |
| Output root | `docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attemptNN/` |

Routine tests, compile checks, and sub-minute smokes do not consume a serious
attempt. A failed serious attempt consumes budget and is preserved. Localized
harness/infrastructure repairs may be retried without renewed approval while
target, data, method, gates, hardware class, and total budget remain fixed.

Stop the whole campaign only if a continuation veto fires or the total budget
is exhausted. A failed generalized-SV candidate does not stop fixed-SIR or
predator-prey integration. A promotion veto blocks that cell, records the
repair, and allows independent phases to continue.

## Pre-Mortem

| How the run could mislead | Cheap discriminator |
| --- | --- |
| A BayesFilter amended target is labeled Zhao--Cui source scope. | Trace paper, author-code `ssmodel.complete`, generator, and algorithm loop in one crosswalk. |
| A route emits a plausible value for the wrong observation order. | T=1/T=2 transition-count tests plus frozen wrong-order sentinel hashes. |
| Level 2 and level 5 agree because they share the same bug. | Exact affine/tiny-horizon checks, independent value-only FD, and external value diagnostics. |
| Source-coordinate score is mislabeled physical. | Direct physical-coordinate FD at asymmetric interior points. |
| Posterior prior/Jacobian terms leak into a likelihood row. | Explicit likelihood/posterior decomposition and value-difference test. |
| SIR NeuTra evidence is wired to the fixed-paper row despite different data/parameters. | Compare seed, observation hash, parameter dimension, target kind, and time order before import. |
| Generalized SV silently becomes KSC or transformed residual. | Assert raw-observation callable identity and row-specific nonclaims. |
| GPU claim is actually CPU placement or full-memory reservation. | Verified memory policy, logical devices, output devices, compiler evidence, allocator bytes. |
| Shared GenUT edits are overwritten. | Pre-edit diff manifest and anchor check immediately before each shared-file patch. |
| A scoped not-applicable row prevents completion forever. | Applicability-aware schema test independent of algorithm count. |
| Full regeneration hangs on unrelated slow algorithms. | Row-selective SGQF artifacts first, bounded full regeneration last. |

## Run Manifest Requirements

Every serious attempt records:

- Git commit and a hash of relevant dirty diffs;
- exact command and environment/conda environment;
- Python, TensorFlow, TFP, CUDA, driver, and XLA status;
- CPU/GPU status, device placement, memory-growth verification, allocator
  current/peak bytes, and logical-device limit if used;
- row, target, data, time order, parameter coordinates, dtype, cloud and route
  identities;
- random seeds and comparator artifact hashes;
- wall time, exit/failure classification, prior attempt, repair, and remaining
  budget;
- plan, raw output, result, and final artifact paths.

CPU-only commands must set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and
record that devices were intentionally hidden. Every GPU/CUDA command must run
with trusted/escalated permissions and configure memory growth before device
initialization.

## Literature And Claim-Support Ledgers

Network metadata lookup is not needed to decide this implementation program.
Citation counts and venue ranks are therefore recorded as not checked rather
than invented. Fetching missing full text is a Phase 0/1 source task before any
source-faithfulness claim.

### Source-Support Ledger

| Source | Class | Local full text | Checked anchors | Allowed support | Gap/quarantine |
| --- | --- | --- | --- | --- | --- |
| Zhao and Cui (2024), *Tensor-train methods for sequential state and parameter learning in state-space models*, plus local author-code mirror | direct competitor/source-row definition and implementation source | `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`/`.txt`; `third_party/audit/zhao_cui_tensor_ssm_p10/source/` | Sections 1--5; Algorithms 1--5; Theorem 8; Sections 6.2--6.4; Eq. 37--38; `models/ssmodel.m::complete`; PP/SIR setup and process functions | Zhao--Cui model context, fixed SIR parameters/timing, predator-prey parameters/noise/RK4/prior, TT scope, and executable transition-then-observe indexing | No retraction/erratum check was performed in this local-only audit; author code supports source behavior but is not a mathematical oracle. |
| Jia, Xin, and Cheng (2012), *Sparse-Grid Quadrature Nonlinear Filtering* | foundational/direct SGQF | missing locally; bibliography metadata only | not checked from primary full text | none until fetched; current chapter anchors are unverified in this audit | `SOURCE_GAP_BLOCKER` for paper-faithful SGQF formula/theorem claims; no retraction/erratum check. |
| Singh et al. (2018), *Adaptive Sparse-Grid Gauss-Hermite Filter* | competitor/extension | missing locally; bibliography metadata only | not checked | none required for fixed-grid runtime correctness | omission/source gap for adaptive-grid positioning; no retraction/withdrawal check. |
| BayesFilter Chapters 34, 35b, 37, 38 | project derivation/context | local TeX/PDF | fixed-cloud recursion, same-scalar gradient, validation ladder, source map | project derivation and implementation specification, subject to code/tests | Not a substitute for unchecked primary-paper claims. |
| BayesFilter code/tests/artifacts cited in this plan | implementation evidence | local | live functions, tests, hashes, result JSON | implementation behavior and historical numerical evidence only | Cannot prove source-faithfulness or exact nonlinear correctness. |

### Citation/Venue Metadata Ledger

| Source | Venue/year | Citation count | Venue metric | Access date/caveat |
| --- | --- | --- | --- | --- |
| Zhao--Cui | JMLR 25, 2024 | not available | not checked | 2026-07-22; metadata not needed for implementation decision |
| Jia--Xin--Cheng | Automatica 48(2), 2012, DOI in `docs/references.bib` | not available | not checked | 2026-07-22; full text missing locally |
| Singh et al. | arXiv:1803.09272, 2018 | not available | not checked | 2026-07-22; publication/version status not checked |

### Backward-Snowball Ledger

The Zhao--Cui related-work and method sections were inspected. Relevant
families considered are Kalman/ensemble filters, SMC and resampling, transport
maps/KR rearrangements, TT decomposition and TT-cross, particle MCMC/SMC2,
and Gaussian-filter parameter estimation. They are context or competitor
families, not primary SGQF construction support. Jia (2012) and Singh (2018)
remain the direct SGQF sources requiring local full-text inspection.

### Forward-Snowball Ledger

Not performed: no approved/available citation metadata source was needed for
this implementation-planning task. Before publication-grade novelty or survey
claims, query citing works for Jia (2012), Singh (2018), and Zhao--Cui (2024),
including corrections, replications, and recent direct competitors.

### Claim-Support Ledger

| Claim | Support class | Anchor |
| --- | --- | --- |
| Fixed SIR has no inferred parameter in the paper experiment. | `PRIMARY_TECHNICAL_SUPPORT` | Zhao--Cui Section 6.3, lines corresponding to Eq. 37 and fixed `kappa`, `nu`. |
| Predator-prey uses six physical parameters, RK4, `4I` noises, and T=20. | `PRIMARY_TECHNICAL_SUPPORT` | Zhao--Cui Section 6.4, Eq. 38 and experiment settings. |
| Zhao--Cui executable source rows transition before every observation. | `PRIMARY_IMPLEMENTATION_SUPPORT` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:34` and model-specific `st_process`/`ob_process`. |
| The local seed-81104 target observes `y0` before the first transition. | `IMPLEMENTATION_EVIDENCE` | `PredatorPreySSM.simulate`, frozen hash artifacts, and corrected PP-SGQF tests. This is a local indexing convention and must not be attributed to the paper without qualification. |
| The corrected local PP score differentiates the same finite SGQF scalar. | `PROJECT_DERIVATION` plus `IMPLEMENTATION_EVIDENCE` | checked recurrence and required FD/value-only tests. |
| Level 2 is adequate for the current PP route. | historical `IMPLEMENTATION_EVIDENCE`, fresh evidence required | July 15 ladder; not a universal or current default until rerun. |
| SGQF sparse-grid formula is source-faithful to Jia (2012). | `SOURCE_GAP_BLOCKER` | primary full text not present locally during this audit. |

### Omitted-Paper And Reviewer-Risk Register

| Risk | Current decision | Next action |
| --- | --- | --- |
| Jia (2012) primary SGQF paper missing from local cache | material for source-faithfulness, not a blocker to project-derivation tests | fetch/store published or accepted full text; inspect Sections 2--4, Eq. 15--35, Algorithm 1, Theorems/Propositions 3.1--3.2 |
| Singh (2018) version/publication status unknown | not needed for fixed-grid implementation, material to adaptive-grid positioning | fetch arXiv/published versions and inspect Section 3; check withdrawal/errata/version differences |
| Official Jia/SGQF implementation not audited | implementation reviewer risk | locate official-author code if available and compare cloud/index/merge conventions without treating it as an oracle |
| Recent SGQF extensions not forward-snowballed | publication/survey risk only | perform dated forward search before broad novelty or literature-completeness claims |

## Required Final Artifacts

- `docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-result-2026-07-22.md`
- `docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-reset-memo-2026-07-22.md`
- versioned per-row JSON results and run manifests under the campaign root;
- a fresh dated high-dimensional leaderboard JSON/Markdown pair;
- a machine-readable row applicability/required-result-kind inventory; and
- source-support updates for any newly fetched primary papers.

The final result must include a decision table, inference-status table,
engineering/numerical/scientific ledgers, per-row result table, same-scalar FD
table, route identities, commands/environment, strongest alternative
explanation, overturning evidence, remaining gaps, and explicit nonclaims.

## Terminal Success Definition

The program succeeds when all of the following are true:

1. every source-named row has a checked paper/author-code/local-generator time-
   order crosswalk, and no amended-target artifact is promoted into a
   source-scope cell;
2. the schema no longer demands nonexistent scores or comparisons from
   not-applicable rows;
3. LGSSM remains valid and the actual/KSC SV rows are requalified on audited
   target identities;
4. fixed SIR emits a source-consistent transition-then-observe value-only SGQF
   cell with no fabricated score;
5. predator-prey emits a source-consistent 20-transition physical
   likelihood/manual-score cell, while the initial-observation-first route is
   preserved under a separate amended-target identity;
6. generalized SV emits a valid explicitly classified raw-y SGQF value/manual-score
   approximation after the bounded repair ladder;
7. every admitted route has repository-issued identity and fresh CPU/GPU-XLA
   evidence appropriate to its claim;
8. the final JSON/Markdown pair is internally consistent and records
   `sgqf_column_complete=true` only when every SGQF-applicable row has its
   required result kind; and
9. no GenUT code, result, tuning, or artifact was changed by this program.

If generalized SV exhausts its bounded repair ladder, the overall artifact may
close as `SGQF_COLUMN_PARTIAL_WITH_SCIENTIFIC_BLOCKER`, but it must not call the
whole SGQF leaderboard complete.
