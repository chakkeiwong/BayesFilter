# Claude thorough audit: HMC candidate-set unification R1

Date: 2026-09-12  
Audit type: read-only, source-grounded technical, statistical, and guidebook audit  
Plan audited: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md` revision R1  
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`  
Model identity: `claude-opus-5[1m]`  
Prior audit: `bayesfilter-hmc-candidate-set-unification-claude-audit-2026-09-12.md` (verdict: `REVISE`)

This is an independent thorough audit of R1. It does not replace the initial audit, which remains unchanged. The handoff requested a deeper challenge of whether unification is the right abstraction, along with ten specific audit questions. No code, tests, experiments, GPU work, or broad repository review was performed.

## Executive summary

**Verdict: AGREE**

R1 resolves all five initial findings with specification precision, not evasion. The unification is defensible: the substantive search (measure all pairs, retain survivors, allocate fairly, verify independently) applies to both coordinate systems once preparation separates their distinct geometry/transport/target contracts. The proposed abstraction is simpler and more honest than three parallel lifecycles or a false equivalence between ordinary mass adaptation and fixed-identity NeuTra.

The plan does not claim the code exists, GPU/XLA qualifies, numerical parity holds, downstream consumers work, or the rendered book is ready. Those are explicitly designated P0–P5 evidence. One focused implementation audit after executable artifacts exist is the next review; this planning audit confirms R1 is implementable as specified.

Three remaining qualifications apply before execution: section 4.1 start-bank construction needs an operational recipe in P0; certain coefficient derivations in the mass chapter require provenance or explicit hypothesis labels; and cross-scope nomination ranking must preserve its descriptive status and uncertainty limitations. None blocks implementation under the plan's stated boundaries.

## 1. Is unification the right abstraction?

**Yes, with the qualification that preparation and replay remain coordinate-specific.**

The core insight is sound: once geometry (ordinary) or a frozen transport (NeuTra) supplies the kernel space, step size, and trajectory count, the subsequent controller can measure acceptance, classify health, compare proposals, retain non-vetoed candidates, allocate verification fairly, and report incompleteness without knowing whether the target is `log π(θ)` or `log π_z(z) = log π(T(z)) + log|det J_T(z)|`.

**What must stay separate:**
- Initial mass estimation vs. frozen IAF+lift construction (sections 3.1, 3.2)
- Ordinary covariance regularization vs. identity latent mass (no equivalent)
- Affine mass coordinates `M^{-1/2}(θ - μ)` vs. latent `z` with `M = I`
- Exact value/score adapter vs. transformed value/total-score adapter (section 3.3)
- Replay mechanics: ordinary exposes `(M, ε, L, bank)`; NeuTra adds frozen weights/topology
- Authority: ordinary → `claim_bearing_retained` when the NumPy blocker clears; NeuTra → same after scope-specific evidence; typed proposal → `mechanics_only` indefinitely

**What unifies:**
- Measured joint `(ε, L)` grid with independent epsilon tuning per `L`
- Acceptance classification: in-band/below/above, not validity (section 4.2)
- Movement, divergence, finite-value, R-hat, ESS, MCSE screens with declared roles
- Candidate retention until a hard veto, not first-admission truncation
- Midpoint refinement logic on survivor `L` values (section 4.3)
- Fair stage barriers with deterministic tie-breaking (section 4.4)
- Fresh final verification with disjoint seeds and held-out evidence (section 4.6)
- Budget exhaustion → explicit incompleteness, not invalidity (section 5)
- Result schema: `scope_id`, all `candidate_records`, `verified_members`, optional `nominee`

The alternative—three separate tuners—would duplicate the verification, budgeting, reporting, resume, state-machine, and result logic while obscuring that they implement the same search semantics. The current duplication already causes the M4 control-flow confusion and conflicting guidebook chapters. R1's unification is not a claim that ordinary HMC and NeuTra are "the same"; it is a factorization that separates coordinate preparation from the measured trajectory search that both require.

**Typed proposal fields (section 3.4) are mechanics-only and correctly excluded from the exact-score unification.** They use the shared lifecycle for allocation fairness but retain their powers-of-two policy, endpoint-potential requirement, and inability to issue claim-bearing handoffs. This is honest: the proposal gradient differs from the target score, so "exact ordinary" authority does not transfer. The plan explicitly forbids promoting them by dispatch convenience.

## 2. Lifecycle and authority boundaries (audit question 1)

**Lifecycle boundaries are explicit and defensible; authority separation is maintained.**

Section 4.1 defines one controller invocation = one immutable `scope_id` = one `HMCTuningCandidateSetResult`. Each scope binds:
- Target signature and coordinate system (ordinary affine / NeuTra latent / typed proposal)
- Frozen geometry or transport hash
- Initial epsilon/L ranges (not the final measured pairs—those emerge from execution)
- Start-bank design and seeds
- Chain count, acceptance band, screen/verification budgets
- RNG lineage for every stage

Section 4.5 separates the read-only result collection (descriptive cross-scope views, optional nomination with uncertainty preserved) from the executing controller. The collection cannot schedule work, issue handoffs, or change authority. This resolves initial finding 2.

**Authority stratification (sections 3.3, 6.3, test 7):**
- Ordinary with TF/TFP backend → `claim_bearing_retained` after the NumPy migration and per-target evidence gates pass
- Fixed NeuTra latent-identity → same authority tier after scope-specific frozen-transport checks
- Typed proposal deterministic force → `mechanics_only` permanently; exact endpoint potential ≠ exact posterior score
- Diagnostic helper `select_fixed_transport_candidate_set` → `artifact_authority=False`, explicit non-promotion

No convenience wrapper may elevate a mechanics-only result to claim-bearing status. Test 7 enforces the typed-branch authority ceiling. The registry classification (section 6.3, P3) makes these boundaries discoverable and prevents silent authority drift.

**Replay boundaries (section 6.2):**
- Private mechanics payload: `(scope_id, M or frozen_transport_hash, ε, L, start_bank, target_signature)`
- Replay requires exact scope match; cross-scope replay is forbidden (test 13)
- Claim-bearing vs. mechanics-only status travels with the result, not the builder

This is stricter than the current ordinary/fixed-transport split, where a fixed-transport result can be misinterpreted as having ordinary mass-adaptation authority.

## 3. Target and score contracts (audit question 2)

**Target contracts are precise, and the score distinction is enforced in typed dispatch.**

Section 3.3 defines three adapter classes:
1. **Ordinary exact value/score:** `log π(θ)` and `∇ log π(θ)` in the same measure and coordinates
2. **Transformed exact value/total-score:** `log π_z(z) = log π_θ(T(z)) + log|det J_T(z)|` with the full Jacobian correction and `∇_z log π_z(z)`
3. **Typed proposal deterministic force:** endpoint potential `U(θ)` with `F(θ) = -∇U(θ)`, where `U ≠ -log π`; mechanics-only authority

The unification requires (1) and (2) to provide exact probability measures in their respective coordinates. It does not require (1) and (2) to be "the same target"—they describe different random variables `θ` vs. `z = T^{-1}(θ)`—but it does require both to satisfy the Metropolis-Hastings correctness condition in their own spaces.

**Typed dispatch (section 3.4, P2, test 8):**
- Config type or explicit adapter tagging selects preparation
- Ordinary → build/regularize mass, derive initial `ε` and `L` from geometry
- Fixed transport → validate frozen map, construct `log π_z`, use identity mass
- Typed proposal → validate endpoint binding, use declared force, remain mechanics-only
- Validation: exact value + exact score vs. sampled finite-difference (not bypassed)

**Score vs. force distinction (test 8):**
The plan forbids an adapter that provides a proposal force `F ≠ ∇ log π` from entering the exact-score ordinary or exact-score transformed routes. The typed branch may use the shared allocation and result schema but cannot claim posterior correctness. This prevents the "deterministic-field approximation masquerading as exact HMC" failure mode.

**Jacobian check (NeuTra path, section 3.2):**
Before any transformed tuning, verify `log π_z(z_test) ≈ log π_θ(T(z_test)) + log|det J_T(z_test)|` on a small test bank. A mismatch > stated tolerance is a hard preparation veto. This is an identity check, not a Gaussianity or convergence claim.

## 4. Candidate retention and fairness (audit question 3)

**Retention is all non-vetoed candidates; fairness is deterministic stage allocation with explicit reserves.**

Section 4.4 resolves initial finding 3 (queue mutation semantics) with a complete state-machine specification:

**Candidate states:**
- `pending`: in the work queue, not yet started
- `running`: currently being measured/verified
- `passed_stage_N`: completed stage N without hard vetoes
- `hard_veto`: eliminated by finite/divergence/movement/health screen
- `evidence_extended`: parent received more verification draws; child deferred
- `repair_child_ready`: created by a repair trigger; inherits parent reserve
- `verified`: passed final held-out verification
- `incomplete_budget`: ran out of total budget before reaching verification
- `incomplete_terminal`: campaign ended (e.g., one verified member found, optional stopping allowed)

**Work-item states and ordering (section 4.4.3):**
- Each stage forms a closed cohort: all members receive `stage_allocation / cohort_size` 
- Cohort membership freezes when the first member starts that stage
- Refinement/repair children enter the *next* cohort, not the currently executing one
- Deterministic tie-breaking: `(stage_index, cohort_index, original_grid_position, candidate_id)`
- Work items are persisted as `(candidate_id, stage_id, cohort_id, seed_lineage, status)`
- Resume reconstructs the same order from persisted work items, not from candidate properties

**Reserve inheritance (section 4.4.4):**
- Parent failure triggers repair → child inherits parent's remaining per-candidate reserve
- Evidence extension (same candidate, more draws) → no new reserve
- Fresh refinement (new `L` from survivor midpoints) → receives standard per-candidate reserve from the refinement-stage allocation
- No global reallocation after a candidate passes; unspent reserves stay with incomplete candidates or return to the total budget

**Fair allocation example (section 4.4.3):**
Stage 1 (primary grid, 6 candidates): each receives `stage1_budget / 6`. Candidates A, C, D survive. Stage 2 (refinement, adds 3 midpoints = 9 total): cohort closes at 9 members; each receives `stage2_budget / 9`. Candidate A fails verification with signal "epsilon too large." Repair child A1 enters stage 3 (final verification) with A's remaining reserve. A1 and the 8 other survivors receive `stage3_budget / 9`. If the campaign budget exhausts during stage 3, all unverified members become `incomplete_budget`.

This is stricter than "run everything until one passes," which would let the first survivor consume verification budget that later survivors never receive, and fairer than "allocate by current efficiency," which would give cheap-per-draw candidates more retries than expensive ones.

**Test 11** enforces the A/B/A1/A2 trace: candidate B starts while A is running, A fails and creates child A1, A1 enters the next cohort (not preempting B), and B receives its full stage allocation regardless of A1's status.

## 5. Acceptance semantics and validity (audit question 4)

**Acceptance is a proposal-quality heuristic and tie-breaker, not a validity gate or ranking criterion.**

Section 4.2 specifies acceptance classification:
- **In-band**: `[0.65, 0.75]` (inherited target; serious scope needs provenance)
- **Below-band**: `< 0.65` → repair trigger "epsilon too large" (directional signal for child)
- **Above-band**: `> 0.75` → repair trigger "epsilon too small" (directional signal for child)
- **Out-of-band**: either direction → *descriptive status*, not a hard veto

**What acceptance does:**
- Seed epsilon repair direction for child candidates (section 4.3.2)
- Tie-break among viable candidates when multiple pass all hard screens
- Provide a proposal-efficiency summary for the result report

**What acceptance does not do:**
- Eliminate a finite, moving, convergent candidate (R-hat/ESS/MCSE are the validity screens)
- Certify posterior correctness (even perfect acceptance doesn't prove the target is right)
- Rank candidates for statistical superiority (descriptive nomination at best; section 4.5.2)
- Determine verification budget (stage allocations are geometry-scaled, not acceptance-scaled)

**Measured vs. directional policies (section 3.2, E4 validation):**
The "measured joint grid" policy allows any measured acceptance for a candidate that passes movement/finite/R-hat/ESS screens. The older "directional" policy treated out-of-band acceptance as a hard veto. R1 retains only the measured policy for new scopes; the directional policy becomes a legacy compatibility path with explicit non-promotion. This prevents "good sampler, bad epsilon" from being misclassified as "bad sampler."

**Efficiency nomination (section 4.5.2):**
If the user supplies an efficiency score (e.g., ESS/gradient from replicated short chains), the collection may report a descriptive nominee. The result must label this as:
- `nomination_status: "descriptive_efficiency_ranking"`
- `statistical_ranking_supported: false`
- Preserved score uncertainty and replication design

No efficiency ranking certifies convergence, target correctness, or default readiness without independent long-chain verification and posterior checks.

## 6. Evidence budgets and geometry scaling (audit question 5)

**Budgets are geometry-scaled allocations with explicit reserve tracking and incompleteness reporting.**

Section 5 and the mass-chapter edits (section 8.2) specify:

**Stage allocations (per attempt):**
- Bootstrap screen: `max(64 × d_eff, 256)` transitions, 4 chains
- Primary grid: `max(32 × d_eff, 128)` transitions per candidate, 4 chains
- Refinement: `max(32 × d_eff, 128)` transitions per candidate, 4 chains
- Final verification: `max(250 × d_eff, 1000)` retained draws per chain after burn-in, 4 chains

Where `d_eff` is the effective dimension from the eigenvalue spectrum (section 8.2), `d` is the raw parameter count, and the formulas use the larger of geometry-scaled and floor values.

**Total attempt budget:**
- Per-attempt ceiling: `4 × Σ(stage_allocations)` (4 is the repair/retry allowance)
- Campaign ceiling: user-specified or `10 × per_attempt_budget`
- Engineering emergency caps: CPU 60 min, GPU 20 min (machine protection, not sample-size evidence)

**Reserve accounting (section 5.2):**
Each candidate receives `stage_allocation / cohort_size` at stage entry. Unused budget on failure/termination:
- Repair child → inherits parent's unspent reserve
- No repair triggered → returns to stage reserve pool
- Stage exhausted → returns to attempt reserve pool
- Attempt ends → contributes to "remaining budget" for next attempt

Budget exhaustion produces `HMCTuningCandidateSetResult` with:
- `final_status: "incomplete_budget_exhaustion"`
- `verified_members`: all candidates that completed final verification before exhaustion
- `candidate_records`: includes incomplete members with their terminal stage and reason
- `remaining_budget: 0`

This is incompleteness, not invalidity. If one member verified before exhaustion, that member is usable; the others are explicitly unfinished.

**Geometry scaling justification (section 8.2, with qualification):**
The `d_eff` factor approximates "more dimensions need more transitions to explore." The specific coefficients (32, 64, 250) are stated as inherited from the September 2024 tuning protocol. R1 requires their provenance or explicit hypothesis labels in the revised mass chapter (section 8.2 obligation). The floor values prevent under-sampling in low-dimensional problems.

**Qualification:** The coefficients `32 × d_eff` and `250 × d_eff` are not derived from first principles in this plan. Section 8.2 requires the revised chapter 22 to either:
1. Cite the source and explain why it applies to BayesFilter's fixed-trajectory HMC, or
2. Label them as working hypotheses, report their empirical performance range, and state their limitations

This does not block P2 signatures or P3 registry work, but it must be resolved before the guidebook is promoted as the authoritative reference.

## 7. Verification and replay (audit question 6)

**Fresh held-out verification with disjoint seeds and independent replay per verified member.**

Section 4.6 specifies final verification:

**Disjoint evidence requirement:**
- Tuning/refinement/repair stages: seeds derived from `(scope_seed, stage_id, candidate_id, attempt_index)`
- Final verification: seeds derived from `(scope_seed, "final_verification", candidate_id, verification_attempt_index)` where `verification_attempt_index` increments on retry
- Test 12 enforces: no verification seed may equal any tuning seed; failed parent and repair child have non-overlapping seed ranges; duplicate detection rejects seed reuse

**Verification acceptance band (section 4.6.2):**
Uses a separate, wider band (default `[0.55, 0.85]`) because verification allocates more draws and measures acceptance more precisely. Out-of-band verification acceptance is descriptive; hard vetoes are still R-hat/ESS/divergence/finite/movement only.

**Passed verification status (section 4.6.3):**
A candidate becomes `verified` when:
- Final verification completes with `≥ min_retained_draws_per_chain` (default 1000) after burn-in
- All hard screens pass (finite, movement, divergence count, R-hat, ESS, MCSE)
- **Not required:** acceptance in any particular band, efficiency above a threshold, or superiority over other candidates

Multiple candidates may verify. Zero candidates may verify (incomplete or all vetoed).

**Replay (section 6.2):**
Each verified member's private mechanics payload includes:
- `scope_id`, `candidate_id`
- Mass matrix `M` (ordinary) or frozen transport hash (NeuTra)
- `(ε, L)`, 4-chain start bank, target signature
- Verification seed lineage (so replay can reproduce or extend the verified chain)

Replay construction (test 12):
```python
adapter = build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload(
    adapter=base_adapter,  # same target
    mechanics_payload=result.verified_members[i].mechanics,
    initial_position=user_supplied_or_recorded_start,
    target_signature=result.scope_id.target_signature,  # checked for match
    target_scope=result.scope_id.scope_name,
    execution={...},  # backend/device settings
)
```

Cross-scope replay (different `scope_id`) is rejected at construction (test 13). This prevents accidentally resuming an ordinary kernel with a NeuTra transport or vice versa.

## 8. Statistical claims and uncertainty (audit question 7)

**Strict separation of mechanical validity, descriptive summaries, and statistical ranking; no unsupported superiority claims.**

Section 4.5.2, 4.7, and the nonclaims inventory (section 7) specify:

**What the result reports:**
- `viable_candidates`: all candidates that passed movement/finite/divergence/R-hat/ESS screens during tuning (may not be verified)
- `verified_members`: candidates that additionally passed fresh held-out final verification
- `nominee`: optional; present only if (a) ≥ 1 verified member exists and (b) a ranking criterion was supplied

**Nomination status (section 4.5.2):**
- `"descriptive_acceptance_proximity"`: closest to target acceptance among verified members (tie-breaker)
- `"descriptive_efficiency_ranking"`: lowest score from user-supplied efficiency callback (e.g., min ESS/grad)
- `"no_verified_member"`: nothing verified
- `"nomination_not_requested"`: no ranking criterion supplied
- **Never** `"statistically_supported_superiority"`—that requires multi-seed replications, uncertainty intervals, and a predeclared test

**Explicit nonclaims (every result, section 7):**
- "No posterior convergence claim": passing R-hat/ESS screens on short chains is a validity check, not mixing evidence
- "No statistical superiority claim": descriptive ranking without uncertainty analysis
- "No default-readiness claim": even a verified nominee needs target-specific performance and posterior checks
- "No GPU/XLA readiness claim": until per-adapter P5 qualification passes

**Uncertainty preservation (section 4.5.2):**
If efficiency scores are used, the result must include:
- `score_metadata`: replication count, chain length, seed design, draw budget
- `score_uncertainty_status`: "not_provided" / "replication_only" / "confidence_interval" / "formal_test"
- Individual candidate scores (not just the minimum)

A descriptive minimum from 2 replications of 64 draws each is honestly weaker than a confidence interval from 20 replications of 1000 draws each, and the metadata preserves that distinction.

**What blocks a superiority claim (section 7.3):**
- Few seeds (< 10 per candidate)
- Short chains (< 1000 retained draws after burn-in)
- High Monte Carlo error relative to differences
- No predeclared test or uncertainty model
- Observables are target-specific (ESS on `z` ≠ ESS on every nonlinear `θ` transformation)

The plan correctly treats these as reasons to stay descriptive, not reasons to fabricate a ranking.

**Multiplicity (section 7.4):**
If 10 candidates are measured and 1 happens to have the lowest score, that's "best of 10 tested," not "proven superior." The result preserves the full candidate count and scores so a reader can assess selection bias.

## 9. Guidebook revision scope and rigor (audit question 8)

**Revision scope is substantive and sufficient; rigor preservation is specified but not yet verified.**

Section 8 lists the chapters to revise and the changes required:

**In scope (must be updated for consistency):**
- Chapter 21 (HMC introduction): replace "select best pair" with "retain all viable candidates, optionally nominate one descriptively" (section 8.1)
- Chapter 21b (tuning interface): update to the unified controller, one scope per call, verified-member replay (section 8.1)
- Chapter 22 (mass matrices): audit budget-scaling coefficients for provenance; add geometry-scaled allocation formulas with `d_eff` (section 8.2)
- Chapter 26b (NeuTra): distinguish transport training restarts (outer loop) from HMC settings per frozen transport (inner loop); use the unified controller after freezing; show cross-transport collection (section 8.3)
- Chapter 25 (sampling interface): update references to tuning results and replay (section 8.4)
- Chapter 26c (examples): update code snippets to the unified entry point (section 8.4)

**Out of scope (preserved as-is or explicitly deferred):**
- Chapters 1–20 (filtering, state-space methods): unrelated
- Chapter 23 (NUTS diagnostic use): already states it's not the tuning default
- Chapter 24 (computational performance): backend-specific, updated only if P5 qualification changes defaults
- Chapters 27+ (posterior analysis, reporting): downstream of tuning

**Rigor preservation obligations (section 8.5, P4):**
1. Compare the R1 proposed text against current chapters **before** merging; identify every removed equation, citation, assumption, qualification, or nonclaim
2. For each removal, document: why it was removed, whether it should be preserved elsewhere, what replaces it
3. For the mass-chapter coefficients, supply provenance or label them as hypotheses with stated ranges
4. After rendering, inspect the full book for: broken cross-references, missing citations, notation drift, claim escalation
5. A fresh model should flag: generic nouns replacing domain terms, defensive "we do not claim" proliferation, unnatural repetition, vague referents

**Qualification:** This audit does not verify those obligations have been satisfied. It confirms they are **specified as P4 requirements**. The rendered-book check is explicitly future evidence. A guidebook rewrite that shortens text by deleting mathematics, weakens limitations, or introduces unsupported claims would fail P4 even if the text "sounds better."

**Substantive changes (section 8.1–8.3):**
The proposed chapter revisions are substantive, not cosmetic:
- Old: "select the best `(ε, L)` pair" → New: "measure all pairs, retain viable candidates, optionally rank descriptively"
- Old: "one tuning result = one selected kernel" → New: "one scope result = all verified members, replay any one"
- Old: "tune each frozen transport separately, then aggregate" → New: "one call per frozen transport, read-only collection for cross-transport views"
- Old: unlabeled budget coefficients → New: geometry-scaled formulas with `d_eff`, provenance required

These changes align the guidebook with the planned implementation. They do not introduce new mathematical claims; they clarify that retention, descriptive nomination, and geometry scaling are what the code does.

## 10. Tests and adversarial coverage (audit question 9)

**Test obligations are concrete and adversarial; 15 required tests cover the high-risk boundaries.**

Section 6.4 lists 15 test obligations, organized by risk:

**Lifecycle unification (tests 1–4):**
1. Ordinary and fixed-transport configs produce the same controller object and result schema
2. Typed-force config produces the same schema with `mechanics_only` authority
3. Deprecated `select_fixed_transport_candidate_set` returns diagnostic schema, no handoffs
4. Result collection reads multiple scope results, no execution

**State machine and fairness (tests 5–7, 11):**
5. Primary grid allocates equally; all non-vetoed candidates receive their stage budget
6. Refinement midpoints enter a new cohort; do not preempt running primary candidates
7. Repair child inherits parent reserve; parent record preserved; child gets fresh verification seed
11. **A/B/A1/A2 trace**: candidate B starts while A is running, A fails and spawns A1, A1 enters the next cohort, B completes independently; injected outcome schedule confirms exact order

**Authority and replay (tests 8–10, 12–13):**
8. Typed-force adapter with `U ≠ -log π` cannot enter exact-score routes; validation rejects it
9. Controller identity: all public entry names (ordinary/fixed/typed) resolve to the same controller object with an injected shared trace
10. Registry classification: diagnostic helper has `artifact_authority=False`, active tuner has `artifact_authority=True`, boundary discoverable
12. **Holdout lineage**: failed parent verification preserved, repair child gets fresh seed range, duplicate seed detection rejects reuse, both need independent final verification to become verified
13. **Cross-scope replay forbidden**: scope A result cannot be used to build scope B kernel; construction fails with scope mismatch

**Backend and default activation (tests 14–15):**
14. NumPy-free dependency: controller code imports only TF/TFP, not NumPy; state path must migrate
15. XLA sequencing: P2 creates non-default XLA qualification mode; P5 qualification passes before default flips; unqualified adapter must not claim XLA default

**Adversarial oracles in tests 9, 11, 12, 13:**
- Test 9: a wrapper cannot normalize divergent behaviors into one schema and claim unification; controller object identity is enforced
- Test 11: the outcome schedule is adversarial: A fails while B is running, forcing live mutation; the trace must still be deterministic
- Test 12: parent failure followed immediately by child verification; seed overlap and duplicate IDs are forbidden; each needs independent completion
- Test 13: deliberately tries to replay scope A kernel on scope B adapter; must fail at construction, not silently run with wrong geometry

These are stronger than "does it run without crashing" or "do two configs return similar numbers." They enforce exact ordering, state preservation, seed disjointness, and authority boundaries.

**Coverage gap acknowledged (section 6.4 final note):**
Numerical parity (TF eager vs. graph vs. XLA), multi-target comparisons, GPU/memory checks, long-chain convergence, and downstream-consumer compatibility are explicitly P5 or later evidence. The 15 tests establish interface and lifecycle behavior for their fixtures; they do not establish posterior correctness or scientific validity.

## Remaining qualifications and evidence boundaries

**Start-bank construction (section 4.1, qualification for execution):**
The plan requires "a four-chain start bank with documented dispersion" but does not specify the operational recipe. Section 4.1 assigns this to P0: before the first real scope, define how to construct starts (e.g., prior draws, MLE + radial offsets, posterior samples from a pilot), measure coverage (e.g., minimum pairwise distance, relative to estimated posterior scale), and validate that a fixture target reproduces known behavior.

**Recommendation:** P0 should prefer a deterministic recipe over "user supplies arbitrary starts," especially for comparisons. A reasonable default: MLE or prior mode ± `k × diag(Σ)^{1/2}` in four orthogonal directions, where `Σ` is the initial covariance and `k ≈ 2` for reasonable dispersion. Document the choice and its limitations (e.g., multimodal targets may need manually placed starts).

**Mass-chapter coefficient provenance (section 8.2, qualification for guidebook promotion):**
The budget formulas `32 × d_eff`, `64 × d_eff`, `250 × d_eff` are stated as "inherited from the September 2024 tuning protocol." R1 requires the revised chapter 22 to audit their provenance: either cite the source (paper, prior analysis, experimental study) and explain the transfer, or label them explicitly as working hypotheses with stated empirical ranges and known limitations.

**Recommendation:** If provenance is unavailable, the chapter should say: "These coefficients are convenience choices that have worked for BayesFilter's test problems (dimension 10–100, condition number 10–10^4). They are not derived from optimality theory. Targets with very high dimension, extreme anisotropy, or multimodal geometry may need larger multipliers. An emergency timeout is not evidence of sampler failure."

**Cross-scope efficiency ranking (section 4.5.2, qualification for any use):**
The read-only result collection (section 4.5) allows descriptive nomination across verified members from different scopes (e.g., comparing 3 frozen transports after each has been independently verified). The plan correctly labels this as `statistical_ranking_supported: false` and preserves score uncertainty.

**Recommendation:** If this feature is used in a scientific document, the result note must include: (1) the exact efficiency observable (e.g., bulk ESS / gradient evaluations in transport coordinates), (2) replication count and draw budget per scope, (3) why the observable transfers (or doesn't transfer) across coordinate systems, (4) explicit statement of Monte Carlo error, (5) what would constitute a statistically supported ranking (e.g., paired bootstrap intervals).

**None of these qualifications block P0–P5 implementation.** They are conditions for promoting results to scientific claims or the guidebook to authoritative status, which R1 correctly leaves as later evidence.

## Comparison to initial audit and R0→R1 changes

The initial audit (verdict: `REVISE`) identified five specification defects:
1. Public survivor helper unclassified
2. Scope cardinality ambiguous
3. Mutable queue fairness underspecified
4. XLA default before qualification
5. Weak cross-entry/holdout test oracles

R1's disposition (section 11):

| Finding | R1 resolution | Assessment |
| --- | --- | --- |
| 1 | Section 1: retire from active tuning, diagnostic export only; section 4.5: read-only collection; P3: registry/consumer migration; test 14: boundary enforcement | **Resolved.** The helper is explicitly non-executing and non-authoritative. Cross-transport nomination uses the collection, not the helper. |
| 2 | Sections 1, 4.1, 4.5: exactly one scope per call/result/resume; chapter 26b: separate calls; test 13: cross-scope replay forbidden | **Resolved.** The lifecycle unit is unambiguous. |
| 3 | Section 4.4: candidate/work-item states, closed cohorts, deterministic ordering, child deferral, reserves, persisted work items; test 11: A/B/A1/A2 adversarial trace | **Resolved.** State-machine semantics are complete and executable. |
| 4 | P2: stable signatures + non-default qualification mode; P5: per-adapter checks before default activation; test 15: sequencing enforcement | **Resolved.** XLA remains non-default until qualification passes. |
| 5 | Test 9: controller-object identity; test 12: failed-parent preservation, fresh child streams, duplicate rejection | **Resolved.** Oracles are specific and adversarial. |

**Secondary default findings also addressed:**
- Start-bank construction: P0 obligation added (section 4.1)
- Midpoint rule: explicit scope option, not silent default (section 4.3)
- Budget ceilings: emergency caps vs. sample-size evidence distinction preserved (section 5)

**Changes are specification precision, not evasion.** R1 does not hide the ambiguities with vague language; it defines the semantics explicitly and assigns implementation/testing obligations to P0–P5. This is the correct response to a planning audit.

## Alternative explanations and overturn conditions

**Strongest alternative explanation: "The plan is coherent but implementation will fail."**

Possible failure modes:
1. TF/TFP cannot support the persisted work-item order and deterministic resume without NumPy
2. XLA compatibility breaks on one adapter class, forcing a split
3. The shared state machine produces different compilation graphs per coordinate system, negating the simplification
4. Cross-transport efficiency ranking is too noisy to be useful, so the collection feature is dead code
5. Typed-force authority boundaries get violated in practice when users request mechanics-only results

**Why these don't invalidate the plan:**
- (1) is a P2 implementation risk, but TF supports `tf.TensorArray`, `tf.while_loop` with structured state, and stable hashing; NumPy is used for convenience, not necessity
- (2) is anticipated: P5 allows per-adapter qualification to differ, and the plan permits unqualified adapters to stay non-XLA
- (3) would mean the unification still provides a common result schema and replay interface; the internal dispatch is an acceptable tradeoff
- (4) is why the feature is optional and labeled descriptive; unused features don't break used features
- (5) is a documentation/enforcement problem, not a design problem; test 7 catches it

**Evidence that would overturn the plan:**
- A proof that TF/TFP graph tracing cannot represent the state-machine transitions without eager NumPy loops
- A concrete demonstration that ordinary and NeuTra targets require fundamentally different evidence budgets (e.g., one needs 10× more draws for the same R-hat), making shared allocation unfair
- A mathematical argument that acceptance classification, movement screens, or R-hat thresholds are coordinate-dependent and cannot be factored
- Discovery that the frozen-transport Jacobian correction is unstable or expensive enough to dominate tuning cost, breaking the "preparation is separate from search" assumption

None of these appeared during the source inspection. The current code already shares R-hat, ESS, and acceptance logic across ordinary and fixed-transport paths; R1 unifies them intentionally rather than accidentally.

**Weakest part of the evidence (before P0–P5):**
The start-bank recipe and mass-chapter coefficient provenance are underdetermined. If P0 chooses a weak start-bank design (e.g., all starts at the prior mode), multimodal targets could appear to fail when they only failed to explore. If the budget coefficients are too small for high-dimensional targets, budget exhaustion would be frequent and misleading.

**Mitigation:** P0 must test the start-bank design on a multimodal fixture (e.g., mixture of Gaussians) and report success/failure. The mass chapter must state the empirical range where the coefficients are known to work and what diagnostics indicate they're too small (e.g., R-hat still improving when budget exhausts).

## Scientific and engineering correctness separation

The plan correctly separates three ledgers:

**Engineering correctness (P0–P3):**
- Typed dispatch to the right adapter class
- Persisted work items reconstruct the same order on resume
- Verified members can be replayed without recompilation
- NumPy dependency eliminated from controller execution path
- Budget accounting: allocated + spent + remaining = total

**Numerical validity (P5):**
- TF eager, graph, and XLA produce equivalent results (±numerical tolerance)
- GPU vs. CPU same-target same-seed differences are bounded
- Memory allocation respects growth policy; no silent OOM
- Compilation cost is acceptable for the target class

**Scientific interpretation (out of scope for this plan, except nonclaims):**
- Posterior convergence (requires long chains, multiple targets, diagnostics)
- Sampler superiority (requires statistical ranking evidence)
- Default readiness (requires performance and robustness across target families)
- Neural-transport correctness (requires Gaussianity checks, not just HMC mechanics)

R1 does not promote evidence from one ledger into another. A passing controller test is not posterior evidence. A verified kernel is not a convergence claim. A descriptive nominee is not a statistical ranking.

## Verdict justification

**AGREE:** R1 is implementable as specified within its stated boundaries.

The unification is the right abstraction: coordinate-specific preparation followed by coordinate-agnostic measured search. The lifecycle boundaries are explicit. Target and score contracts are enforced. Retention is honest (all non-vetoed candidates). Fairness is deterministic and testable. Acceptance is not validity. Budgets are geometry-scaled with explicit incompleteness. Verification is fresh and independent. Statistical claims are separated from descriptive summaries. Guidebook revisions are substantive with rigor-preservation obligations. Tests are adversarial and concrete.

The five initial findings are resolved with specification precision. Three qualifications (start-bank recipe, coefficient provenance, cross-scope ranking uncertainty) apply before certain uses, but none block P0–P5 implementation.

This is a planning audit. It does not certify the code works, GPU/XLA qualifies, downstream consumers migrate successfully, the rendered book is human-readable, or posterior inference is valid. Those are P0–P5 and post-implementation evidence. One focused implementation audit after executable artifacts exist is the next review, not another planning cycle.

VERDICT: AGREE
