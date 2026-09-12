# Claude audit: HMC candidate-set unification and guidebook rewrite

Date: 2026-09-12  
Audit type: bounded, read-only technical and scientific plan audit  
Plan audited: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`  
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`  
Model identity: `claude-opus-5[1m]`

The audit inspected only the handoff, the full plan, and the bounded source excerpts named by them. No commands, tests, HMC, GPU initialization, build, package operation, or repository-wide review was performed. The checkout contained unrelated dirty work identified in the handoff; none was changed by the audit.

## Material findings

### 1. Confirmed defect — the public all-survivor helper has no explicit migration disposition

**Plan sections:** 1, 4.5, P0–P4, and 7.  
**Inspected sources:** `bayesfilter/inference/fixed_transport_candidate_selection.py:121–216` (`select_fixed_transport_candidate_set`); `docs/chapters/ch21b_hmc_tuning_interfaces.tex:446–507`; plan E5.

The plan requires one recommended entry point and says that no compatibility name may own a second selection procedure. The existing package-reachable helper nevertheless executes its own validation rungs, defines `viable_candidates`, and may choose `selected_candidate` using a caller score. Chapter 21b currently recommends tuning each frozen transport independently and then invoking this helper. The plan correctly observes that this is not an all-pairs tuner, but P3 never gives this known public helper a mandatory final classification.

**Counterexample:** `tune_fixed_transport_hmc_kernel` can be converted into the new set-valued controller while a consumer continues to tune each transport separately and call `select_fixed_transport_candidate_set`. That consumer still has an independently scheduled validation-and-nomination procedure with different candidate semantics. The migration would expose one schema while retaining two active lifecycle owners.

**Smallest required revision:** name `select_fixed_transport_candidate_set` explicitly in P0, P3, P4, the registry/export migration, and P5 tests. Choose one disposition before implementation: (a) a compatibility reader/adapter that only aggregates already-issued candidate-set results and cannot execute independent tuning, verification, or authority; or (b) a diagnostic/legacy helper with explicit non-authority and a replacement path. If cross-transport nomination remains useful, define it as a separate descriptive aggregation over verified scope results, not as another kernel-tuning lifecycle.

### 2. Confirmed implementability gap — the controller's scope cardinality is ambiguous

**Plan sections:** 1, 4.1, 4.4, 4.5, P1–P3, and 7.  
**Inspected sources:** `bayesfilter/inference/hmc_tuning_dispatch.py:29–91`; `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:842–1055`; `bayesfilter/inference/tuning_contract.py:939–1068`; `docs/chapters/ch26b_neutra_transport_hmc.tex:343–367`.

The current public calls prepare one ordinary geometry or one frozen transport. The plan sometimes describes one result containing scope identities and a fair queue over “every current survivor,” while the guidebook rewrite requires the hierarchy frozen transport → geometry scope → multiple `(epsilon,L)` kernels. It does not decide whether `HMCTuningCandidateSetResult` is exactly one frozen scope or whether one controller schedules several incompatible scopes.

**Counterexample:** two frozen transports can produce the same numeric `(epsilon,L)` but have different latent coordinates, start banks, transport hashes, and final states. A global barrier can accidentally compare or resume them as one cohort; a per-call implementation can instead return two independent results but leave campaign completion, cross-transport nomination, and budget accounting undefined.

**Smallest required revision:** define the lifecycle unit normatively. The simplest design is one `HMCTuningCandidateSetResult` per immutable `scope_id`; candidate IDs include that scope; no chain state or verification evidence crosses scopes. A separate collection may report several scope results and descriptive cross-scope nomination without tuning authority. If a multi-scope controller is intended instead, specify independent per-scope queues, geometry, state, RNG, reserves, completion, and replay, plus the exact global fairness rule. Align the public signatures, result schema, resume schema, helper disposition, and chapter 26b example with that choice.

### 3. Unsupported state-machine assumption — fairness is stated but not yet executable under queue mutation

**Plan sections:** 4.2–4.5, P1, and verification cases 2, 3, 5–7.  
**Inspected sources:** `bayesfilter/inference/hmc_kernel_tuning.py:25148–25465`; `bayesfilter/inference/hmc_tuning_state.py:13–107`; `tests/test_hmc_kernel_tuning_outer_loop.py:3782–3879,4168–4219`.

The plan replaces first-admission execution with a stage barrier or “equivalent persisted fair queue,” but it does not define barrier membership when measurement, refinement, evidence extension, or repair creates a candidate while other candidates are at different stages. The current state model is a single-kernel transition graph, and the current queue fixes order and seeds before execution. P1 therefore cannot be implemented unambiguously from the proposed semantics alone.

**Failure mode:** candidate A passes an early rung and creates a repair child A1 while candidate B is still awaiting that rung. Depending on implementation, A1 may consume B's reserved work, wait forever behind a barrier whose membership has changed, or advance before B. Resume can also produce a different order if pending work is reconstructed rather than persisted.

**Smallest required revision:** add a compact transition specification for candidate status and work-item status. Define when a cohort closes, where refinement/repair children enter, whether evidence extension retains candidate identity, how reserves move (or do not move) from parent to child, deterministic tie-breaking, and how terminal candidate versus terminal campaign states are computed. Persist ordered work-item IDs and stage/cohort IDs rather than reconstructing order on resume. Add an adversarial trace in which repair and refinement insert work during a partially completed barrier.

### 4. Confirmed sequencing defect — P2 makes XLA the default before the required qualification

**Plan sections:** P2, P5, default audit, and section 8.  
**Inspected sources:** plan lines 256–263 and 367–414; `docs/reference/hmc-tuning-interface.md:313–341,560–568`; `bayesfilter/inference/hmc_kernel_tuning.py:20–38`; `bayesfilter/inference/hmc_tuning_state.py:1–10`.

P2 completion currently requires stable signatures and “default XLA,” while P5 is where compatibility, numerical equivalence, memory, compilation, and device checks occur. The repository policy permits enabling `jit_compile=True` only after those checks pass. The current ordinary path also imports NumPy directly in both the executor and state/repair dependency, so a shared facade cannot itself resolve backend eligibility.

**Counterexample:** P2 can satisfy its written completion condition by setting the new controller's default to XLA before the nonlinear transformed adapter or ordinary state path has passed compatibility and equivalence checks. A later P5 failure would mean an unqualified default had already been introduced.

**Smallest required revision:** make P2 implement stable TensorFlow signatures and an explicit non-default XLA qualification mode. P5 must inspect the active call chain and pass compatibility, numerical-equivalence, peak-memory, compile-cost, and steady-state checks for each supported adapter class before any default flips. If a class fails, retain an explicit reviewed non-XLA exception or leave it non-admitting; do not claim one default across unqualified adapters. Require dependency-closure checks to include state, selection, artifact construction, and replay, not merely numerical kernels.

### 5. Missing discriminating evidence — P5 needs cross-entry and holdout-lineage tests with stronger oracles

**Plan sections:** evidence contract, P3, and verification cases 1–10.  
**Inspected sources:** `tests/test_hmc_kernel_tuning_outer_loop.py:3782–3879,4168–4250`; `tests/test_fixed_transport_candidate_selection.py:27–65`; `tests/test_hmc_tuning_policy_replay_authority.py:29–146`; `docs/reference/hmc-tuning-interface.md:570–694`.

The proposed tests are substantially better than the current first-admission and post-tuning survivor tests. Two concrete adversarial oracles remain unspecified:

1. “All supported entry names use the same controller” can pass if wrappers call identically named but separate controllers or normalize divergent traces into one schema.
2. The plan requires a fresh stream after failed final verification, but no test explicitly proves that a repair child cannot reuse the failed holdout, inherit its verified status, or charge the same draw range twice.

**Smallest required revision:** require an injected deterministic outcome schedule to produce the same controller work-item trace through every active wrapper, with a wiring assertion on the shared controller identity. Add a holdout-lineage test that records stream IDs and disjoint draw ranges across parent failure and child verification, rejects duplicate evidence IDs, preserves the failed parent, and prevents either record from becoming verified without its own completed final rung. Add a multi-scope replay test after finding 2 is resolved.

## Unsupported assumptions and missing evidence

- **Target and transformed-score preservation:** the plan states the correct target contracts, but equivalence of extracted adapters and transitions is future P2/P5 evidence, not checked by this audit.
- **Typed proposal-field integration:** registry evidence confirms that the current typed branch owns a powers-of-two search and mechanics-only authority. The plan correctly forbids promoting its force to an exact score, but whether it can use the common lifecycle without capability loss is not checked until the active call chain and parity tests exist.
- **Resume completeness:** binding source, scope, RNG lineage, states, draw ranges, settings, and remaining work is specified at a high level. Crash-safe serialization and exact continuation behavior are not checked; finding 3 requires a normative work-item representation first.
- **Downstream consumers:** P0 names a bounded audit and P3 requires concrete migration findings, but no downstream source was inspected in this review. The plan correctly leaves downstream adoption owner-controlled.
- **Literature support:** no new optimality claim is made. The proposed manuscript rewrite must inspect the cited methods, diagnostics literature, and official implementations before introducing or retaining stronger mathematical claims.
- **Rendered guidebook:** explicitly future work. Neither compilation nor human-reader acceptance has been checked.

## Evidence coverage

| Required check | Status | Inspected evidence and audit boundary |
| --- | --- | --- |
| Handoff and complete plan | Inspected | Handoff in full; plan sections 1–10 in full. |
| E1 ordinary public procedure | Inspected | `docs/reference/hmc-tuning-interface.md:128–170`; confirms complete grids but two-start/first-pass completion. |
| E2 ordinary selector | Inspected | `_select_joint_l_epsilon_candidate`, `hmc_kernel_tuning.py:22211–22290`; confirms acceptance-distance priority in both branches. |
| E3 ordinary direct queue | Inspected | `_run_phase7_direct_candidate_queue`, `hmc_kernel_tuning.py:25148–25465`; confirms quota and `first_admission`. |
| E4 fixed-transport tuner | Inspected | `fixed_transport_hmc_tuning_tf.py:79–106,842–1094,1425–1592,1992–2075`; confirms separate selector, held-out verification, measured-policy acceptance semantics, and single selected kernel. |
| E5 all-survivor helper and chapter | Inspected | `fixed_transport_candidate_selection.py:121–219`; chapter 21b lines 446–507; confirms post-tuning validation and optional nomination, not pair tuning. |
| E6 registry and dispatch | Inspected | `tuning_contract.py:939–1232`; `hmc_tuning_dispatch.py:29–100`; confirms two active tuners and a typed mechanics branch with its own powers-of-two trajectory policy. |
| E7 ordinary policy | Inspected | `hmc_ordinary_selection_policy.py:1–105`; confirms fixed grid, bound 25, and one midpoint barrier. |
| E8 earlier HMC and mass chapters | Inspected | Chapter 21 lines 55–143 and chapter 22 lines 36–101; confirms the conflicting best-pair narrative and ungrounded coefficients requiring audit. |
| E9 NeuTra chapter | Inspected | Chapter 26b lines 316–367; confirms transport-restart selection is currently separate from within-transport pair retention. |
| E10 current tests | Inspected | Outer-loop tests lines 3760–3900 and 4160–4250; candidate-selection tests lines 1–80; confirms contradictory current contracts. |
| Public artifact/replay contracts | Inspected | Interface reference lines 570–694 and replay-authority tests lines 1–146; confirms private geometry, replay roles, and current NumPy blocker. |
| Known ordinary NumPy dependency | Partially inspected | `hmc_tuning_state.py:1–260` and `hmc_kernel_tuning.py:1–180` show active NumPy imports. Full called dependency closure is intentionally P0 work and was not checked. |
| M4 motivating JSON | Not checked | The control-flow diagnosis was undisputed from source and tests; no inference relies on the external artifact. |
| Downstream MacroFinance/dsge_hmc consumers | Not checked | Correctly assigned to bounded P0 inventory; this audit makes no compatibility claim about them. |
| Chapter 25, chapter 26c, examples, generated tables, rendered PDF | Not checked | Their inclusion in P4 is appropriate; current content and rendered fidelity remain future migration evidence. |

## Disposition by area

| Area | Disposition | Reason |
| --- | --- | --- |
| Procedure | Revise | Core search semantics are defensible, but the public helper disposition and scope cardinality must be explicit. |
| Mathematics | Sound enough to implement after revision | Exact target/score, Jacobian-corrected transformed target, affine mass coordinates, identity-mass NeuTra, and mechanics-only proposal fields remain distinct. Numerical parity is future evidence. |
| Statistics | Sound | Acceptance is not validity or ranking; movement, divergences, R-hat, ESS, MCSE, observables, optional stopping, multiplicity, and descriptive nomination have appropriately distinct roles. |
| Candidate lifecycle | Revise | Retention and local repair are correct in intent; dynamic cohort/barrier and child insertion semantics are underspecified. |
| Budgets and resume | Revise narrowly | Total ceilings and under-budgeted status are honest; persisted work ordering and reserve transfer need exact semantics. |
| Backend | Revise | Dependency closure is planned correctly, but default XLA is sequenced before qualification. |
| Migration and replay | Revise | Explicit member IDs and scope-bound private handoffs are correct; helper migration and multi-scope replay must be specified. |
| Guidebook | Scope is substantively complete; implementation not checked | The listed chapters cover the known conflict and preserve math/citations, but helper/scope semantics must first be resolved and rendered review remains future work. |

## Remaining numerical defaults

| Choice | Provenance/limits sufficient for this plan? | Required treatment before claim-bearing execution |
| --- | --- | --- |
| Primary `L=(3,5,9,13,18,25)` and former cap 25 | Yes as an inherited starting hypothesis, not as a universal grid. | A serious scope must justify coverage, predeclare bounded expansion, and report boundary limitations. |
| Acceptance target 0.70 and band `[0.65,0.75]` | Yes only as inherited proposal/efficiency heuristics. | They must not eliminate a finite moving candidate, certify validity, or rank candidates. Preserve acceptance uncertainty and measure every proposed pair. |
| Four-chain bank | Partly. The plan labels it inherited and denies universality, but “dispersed” is not operationally defined. | P0 must define target-specific start-bank construction/provenance and a cheap mode/coverage diagnostic; a fixture may retain four chains without a scientific sufficiency claim. |
| R-hat 1.01, ESS 400/200, MCSE/SD 0.10 | Yes as identified existing screens whose transfer is forbidden. | Each serious validation design must name observables, evidence budget, threshold provenance, and extension behavior. Do not copy helper values into common defaults. |
| Fixed-transport 4+16 and related short screens | Yes as mechanics-only convenience fixtures. | They cannot support mixing, posterior, ranking, or default claims. |
| One midpoint-refinement barrier | Partly: clearly labeled a proposed initial hypothesis with bounded cost, but not yet a settled common default. | Treat it as explicit scope configuration or test it through coverage diagnostics; define how new refinement candidates enter the fair queue. |
| Equal per-chain transitions within a stage | Yes as a declared fairness convention, not cost equality. | Record gradients and wall time; reserve whole-cohort and final-verification work. Define dynamic cohort semantics per finding 3. |
| Mass freeze and latent identity mass | Yes as coordinate/kernel identity contracts. | Any geometry change creates a new scope and forces epsilon/L requalification; no ordinary adaptation may enter fixed-transport NeuTra silently. |
| CPU 60 minutes, GPU 20 minutes, four launches, five minutes each | Yes as convenience-chosen engineering ceilings with an explicit derivation for the per-launch cap. | Do not interpret them as sample-size evidence. Stop as under-budgeted if an engineering question cannot be answered. |
| TensorFlow/TFP, GPU, XLA, memory growth | Backend/GPU/memory-growth provenance is settled by repository policy; XLA readiness is not. | Enforce on-demand allocation and trusted GPU checks. Qualify XLA per adapter before default activation, as required by finding 4. |
| Future seeds, dimensions, transitions, and tolerances | Correctly left unset. | Derive or inherit them from inspected fixtures/error arguments in the P0 execution note; do not infer them from cost ceilings. |

## Required revision and next action

Revise the plan before implementation. The smallest adequate edit is:

1. classify `select_fixed_transport_candidate_set` and its exports, registry record, documentation, examples, and consumers;
2. declare whether a result/controller owns exactly one immutable scope or multiple scopes, and define any cross-scope collection separately;
3. add candidate/work-item transition semantics for cohort closure, child insertion, reserves, deterministic ordering, completion, and resume;
4. move XLA default activation after per-adapter P5 qualification while retaining stable signatures in P2; and
5. strengthen P5 with cross-wrapper trace identity, failed-holdout/repair-child lineage, duplicate-evidence rejection, and scope-isolated replay tests.

After those edits, the next justified action is a bounded re-read of the changed plan sections, not implementation or an experiment. There is no scientific or operational reason for another broad review chain. Material blockers to implementation are the four specification defects above; missing GPU, numerical, downstream, rendered-book, and literature evidence are expected later-phase checks rather than reasons to expand this planning audit.

The plan otherwise provides a scientifically defensible direction: it separates target correctness from proposal mechanics, preserves frozen geometry, measures rather than assumes epsilon behavior, retains non-vetoed alternatives, distinguishes candidate and campaign failure, reserves fresh verification, prevents acceptance-only validity claims, and keeps posterior estimation separate. Those strengths do not resolve the remaining interface and state-machine ambiguities.

VERDICT: REVISE