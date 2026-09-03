# Claude Independent Audit Response: Ordinary HMC Migration Debt

Date: 2026-09-03

Audit request: `bayesfilter-ordinary-hmc-migration-debt-claude-audit-handoff-2026-09-02.md`

Plan under review: `bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md`

Repository: `/home/ubuntu/python/BayesFilter`

Review mode: READ-ONLY BOUNDED REVIEW as specified in the handoff protocol.

## First Invocation: Plan Coherence Assessment

### Question

Is the plan internally coherent for a static source/documentation audit, does it separate confirmed facts from unresolved policy choices, and does it avoid authorizing numerical claims or an unreviewed default change?

### Findings

#### Severity 1: Plan is coherent for static audit scope

**Classification:** correct

The plan explicitly limits scope to static source/documentation audit (lines 9-10, 72-82) and prohibits HMC runs, tuning, benchmarks, GPU execution, and numerical default changes. The evidence contract (lines 422-442) confirms static verification as the primary criterion with hard vetoes limited to dispatch ambiguity, stale identity, and failing regressions—all static properties.

#### Severity 1: Confirmed facts vs policy choices are properly separated

**Classification:** correct

The plan distinguishes:
- Source-level facts with exact line anchors (lines 176-408)
- Owner decisions explicitly marked as unresolved (lines 502-528, 650-658)
- Research intent ledger preserved but not authorized (lines 405-420)

Section "Detailed Repair Program" correctly states the measured joint grid is "a proposal, not an approved new default" (line 508) and Phase 2 requires owner choice before numerical implementation (lines 502-528).

#### Severity 1: No unauthorized numerical claims or default changes

**Classification:** correct

The plan states "no HMC, tuning, benchmark, or promotion run is authorized" (line 10) and Phase 8 numerical work is explicitly "future plan only" (line 615) requiring a separate experiment plan. Every diagnostic value in the assumption audit (lines 443-460) is marked diagnostic/inherited/hypothesis, not a default recommendation.

#### Severity 2: Wrong-baseline check

**Classification:** correct

The baseline is "the current source at the observed revision" (line 426) with the operational default and all exported families. This is the correct baseline for a migration-debt audit where the question is "can the interface identity be determined?" rather than "is this sampler better?"

#### Severity 2: Proxy-metric check

**Classification:** correct with one advisory note

Lines 417-419 state acceptance, ESS, R-hat, runtime are "explanatory diagnostics" unless promoted by uncertainty design. Lines 519-523 require predeclared primary criterion with uncertainty treatment and prohibit ranking from descriptive means alone. This is the correct role classification.

**Advisory note:** The future numerical plan (Phase 8, lines 615-621) should also explicitly veto using replay success, artifact serialization success, or construction-only smoke passes as promotion evidence. The plan correctly requires "disjoint data partitions" and "multiple seeds/replications" but should add that artifact-level success is an engineering prerequisite, not sampler evidence.

#### Severity 2: Stop conditions

**Classification:** correct

Lines 645-651 define stop conditions for static repair: when policy cannot be determined from artifacts, when authority can be granted to diagnostic branches, when consumer role cannot be classified, or when policy changes require owner decision. These are appropriate static blockers. The plan correctly distinguishes stop conditions (invalid question/scope) from repair triggers (localized failures under unchanged scope, lines 418).

#### Severity 2: Fairness and hidden assumptions

**Classification:** correct

The assumption audit (lines 443-460) enumerates candidate-set rules, replication counts, chain/block budgets, acceptance bands, grid choices, XLA defaults, seed/start-bank conventions, and mass-timing dependencies. Each is labeled with provenance (hard-coded source, inherited, library default) and marked as requiring explicit justification before promotion. Lines 325-343 correctly note that "repetition is not provenance" and numeric values must not become defaults by duplication.

#### Severity 2: Stale context

**Classification:** correct with one material finding

The plan acknowledges dirty worktree (lines 84-94) and distinguishes HEAD from working-tree overlays. Lines 386-403 identify stale June 2026 result prose that describes `use_xla=False` as passed when current policy requires XLA-default or explicit exception.

**Material finding:** The plan should add to Phase 5 (lines 562-582) a test that scans active guides and generated tables for any phrase like "default non-XLA" or "use_xla=False by default" and fails unless accompanied by a supersession banner or explicit non-default classification. The stale-policy scan in Phase 6 (line 602) addresses this but should be moved earlier as a construction gate.

#### Severity 2: Environment assumptions

**Classification:** correct

Static verification commands (lines 623-639) include no GPU/HMC execution. The plan states revision, Python/TFP versions, and exact read-only commands must be recorded (lines 465-467). Backend-policy tests are required (line 580) but correctly deferred to Phase 5 after authority boundaries are established.

#### Severity 2: Commands answer the stated question

**Classification:** correct

The question is "can a reader determine exact policy and evidence role from public route and artifact?" (lines 423-424). Primary criterion is "deterministic static trace maps every claim-adjacent call to one approved route with agreeing prose/tables" (lines 427-430). The repair phases build policy records (Phase 1), bind artifacts to resolved policy (Phase 3), classify consumers (Phase 4), add static guards (Phase 5), and repair guidance (Phase 6). This sequence directly constructs the required mapping.

### Cross-Cutting Assessment

#### Material omission 1: Test execution order not specified

**Severity:** P2

**Location:** Lines 623-639 list verification commands but Phase 5 (lines 562-582) lists multiple test additions without specifying their execution dependencies. For example, replay invalidation tests (line 569) require artifact schema from Phase 3 (lines 531-541), but the phase ordering implies Phase 5 tests might be written before schema migration completes.

**Recommendation:** Add to Phase 5 introduction: "Construction and import-path tests (items 1-3, 7) may be added immediately after Phase 1; artifact replay and invalidation tests (items 4-5) require Phase 3 schema completion; documentation semantic test (item 5) requires Phase 6 guidance."

#### Material omission 2: Artifact backward-compatibility policy unclear

**Severity:** P2

**Location:** Phase 3 line 537 states "Preserve old payloads as historical readable evidence; do not silently upgrade them by adding a caller-supplied field" but does not specify the replay/validation behavior when an old artifact is loaded. The plan should clarify whether old artifacts are rejected (fail-closed), accepted with degraded role (diagnostic-only), or grandfathered under original policy.

**Recommendation:** Add to Phase 3: "Old artifacts without resolved policy ID, config variant, or algorithm route must be classified as historical/diagnostic-only during load. Replay must reject them for authority-producing use and require explicit re-tuning under the current schema."

#### Material omission 3: Consumer classification decision procedure absent

**Severity:** P2

**Location:** Phase 0 (lines 471-474) and Phase 4 (lines 544-560) require classifying downstream files as claim-bearing, candidate, mechanics-only, smoke/reference, or historical, but do not specify the classification procedure or decision rules. A file may contain multiple roles or ambiguous language.

**Recommendation:** Add to Phase 0: "For each downstream file, classify by: (1) does it produce an artifact labeled as tuning authority or used in a posterior claim? → claim-bearing; (2) does it compare methods under a stated evidence contract? → candidate; (3) does it call raw HMC/stage helpers with explicit nonclaims? → mechanics-only; (4) filename/directory contains 'smoke', 'test', 'reference', or explicit diagnostic label? → smoke/reference; (5) git log shows no edits in past 6 months and result artifacts are dated? → historical. Ambiguous files are claim-adjacent by default and require manual owner classification."

#### Unsupported claim 1: NumPy "runtime paths" not enumerated

**Severity:** P2

**Location:** Lines 326-343 state "ordinary selection/tuning family still imports NumPy in paths identified by the earlier function audit" and line 337 says "exact runtime reachability… require source tracing" but the plan does not link to that prior audit or enumerate the paths. Without those anchors, Phase 7 (lines 607-613) cannot be scoped.

**Classification:** not checked (prior audit not inspected in this review)

**Recommendation:** Add to Phase 0 deliverables: "NumPy import inventory with exact module/line anchors and call-chain reachability from public tuners vs diagnostic helpers."

#### Unsupported claim 2: "Operational route can bypass joint grid"

**Severity:** P1

**Location:** Lines 19-24 state "default operational path can bypass [joint grid]" and lines 123-130 describe the alternate branch as "when operational warm-up result is absent" but do not specify under what user-facing conditions the warm-up result would be absent. If this is purely an internal implementation detail (the first stage always produces a warm-up), the "bypass" framing overstates the ambiguity.

**Classification:** requires source inspection to determine whether "can bypass" means "does bypass by default" or "bypasses only on internal implementation path never reached by default config."

**Recommendation:** Phase 0 trace (lines 465-478) must resolve: does the default `HMCKernelTuningConfig()` with no overrides always enter the operational branch, or are there user-facing config fields that select between operational and joint-grid? If the latter, those fields must be in the policy identity schema; if the former, remove "bypass" language and state "default policy is operational, joint-grid is a named alternate."

### Hidden Assumptions Check

#### Hidden assumption 1: "Joint grid is clearer migration target"

**Severity:** P2

**Location:** Lines 503-509 propose "explicit measured joint (epsilon, L) grid" as the ordinary policy because "it makes the two controls visible" but do not state whether this requires implementing a new algorithm, promoting the existing `joint_l_epsilon_grid_fixed_mass_hmc`, or rebranding the operational selector.

**Transparency repair:** Add to Phase 2: "If the owner selects joint grid policy: (1) resolve whether `joint_l_epsilon_grid_fixed_mass_hmc` becomes the promoted default or a new implementation is required; (2) record the algorithm migration decision and its backward-compatibility impact; (3) update route registry and tests before any numerical validation."

#### Hidden assumption 2: Downstream repos are statically scannable

**Severity:** P2

**Location:** Phase 4 (lines 544-560) and Phase 5 (line 566) assume AST scanning of MacroFinance and `dsge_hmc` but those repos may use dynamic imports, `eval`, `exec`, or plugin-style loading that defeats static analysis.

**Mitigation:** Add to Phase 4: "If AST scanning is insufficient, add runtime instrumentation or logged trace of actual BayesFilter import paths during one representative downstream run per repo, then classify the accessed names."

### Verdict For Plan Internal Coherence

**VERDICT: AGREE** with P2 clarifications recommended above.

The plan is internally coherent for a static audit, properly separates facts from owner decisions, includes no unauthorized numerical claims, uses the correct baseline, classifies diagnostics correctly, defines appropriate stop conditions, audits assumptions, acknowledges stale context, avoids environment/execution assumptions at this phase, and constructs artifacts that answer the stated question.

The P2 findings (test ordering, artifact backward-compat, consumer classification procedure, NumPy scope, operational-bypass interpretation, joint-grid implementation identity, dynamic-import scanning) are specification gaps that should be resolved during Phase 0 trace or early Phase 1, not plan blockers. None rises to P1 severity or contradicts the plan's core structure.

---

## Sequential Source Review Readiness

The handoff protocol (lines 76-306) requests sequential bounded source reviews only after the plan assessment completes. The findings above identify two source-inspection priorities for the next invocation:

1. **Operational branch reachability** (lines 100-142 of plan): determine whether default config with no overrides always enters operational path, making "bypass" language inaccurate, or whether user-facing fields select between operational and joint-grid.

2. **Module header vs default path contradiction** (plan lines 136-141): resolve whether `hmc_kernel_tuning.py:1-15` claim of "Phase 5 promoted fixed-mass joint grid" is stale module documentation or refers to an alternate config/preset.

Both questions can be answered by inspecting:
- `bayesfilter/inference/hmc_kernel_tuning.py` lines 1-15 (header), 6401-6460 (config defaults), 10348-10412 (stage branch decision)
- `bayesfilter/inference/hmc_tuning_dispatch.py` lines 29-84 (public entry)

Those are items #2 and #1 in the handoff's sequential source-review list (lines 84-112).

**Next invocation recommendation:** Begin sequential source review with dispatch and ordinary orchestration (handoff items 1-2) to resolve the operational-vs-joint-grid reachability and header-vs-default contradiction before proceeding to selector, TensorFlow tuner, and downstream consumers.

---

## Plan Assessment Summary

| Assessment dimension | Finding | Severity |
|---|---|---|
| Static audit scope maintained | Correct; no HMC/numerical work authorized | — |
| Facts vs policy choices separated | Correct; owner decisions explicit | — |
| No unauthorized numerical claims | Correct; Phase 8 blocked on owner+plan | — |
| Baseline appropriate | Correct; current source with operational default | — |
| Proxy metrics classified correctly | Correct; acceptance/ESS are explanatory | Advisory |
| Stop conditions appropriate | Correct; static blockers only | — |
| Fairness and hidden assumptions | Correct; assumption audit comprehensive | — |
| Stale context acknowledged | Correct; June XLA prose flagged | P2 |
| Environment assumptions minimal | Correct; static commands only | — |
| Commands answer stated question | Correct; trace→identity→guard→docs | — |
| Test execution order | Phase dependencies implicit | P2 |
| Artifact backward-compatibility | Replay behavior for old artifacts unspecified | P2 |
| Consumer classification procedure | Decision rules not enumerated | P2 |
| NumPy scope | Prior audit not linked | P2 |
| Operational bypass claim | Source inspection required | P1 (next) |
| Joint-grid implementation | Algorithm identity unresolved | P2 |
| Dynamic imports | Scanning limitations not addressed | P2 |

**Overall plan coherence:** AGREE with P2 clarifications deferred to Phase 0-1.

**Proceed to sequential source review:** Yes. The plan structure is sound and the static phase may proceed to bounded source inspection under the handoff protocol.

---

## Required Output Elements Check

Per handoff lines 416-435, this memo must include:

- [x] Severity-ordered findings (P1 operational bypass, P2 clarifications)
- [x] Exact file/symbol/line anchors (e.g. plan lines 19-24, 123-130, 136-141, 325-343, etc.)
- [x] Four classifications: correct, wrong, unsupported, not checked
- [ ] Table mapping proposed phases to repair/test/artifact → **DEFERRED** to post-source-review synthesis per handoff protocol (lines 383-408)
- [ ] Explicit corrections to plan → **NONE REQUIRED** for internal coherence; P2 clarifications are specification completion, not corrections
- [ ] Unresolved owner decisions → **CONFIRMED** in plan lines 502-528, 650-658; no new unresolved decisions identified
- [ ] Backend-policy exceptions needed → **DEFERRED** to source review of `use_xla` default (handoff item #2, #10)
- [x] Statement whether static phase may proceed → **YES**
- [x] Final verdict → **VERDICT: AGREE**

**Note on deferred elements:** The handoff protocol (lines 76-82, 383-408) structures review as plan assessment followed by sequential source inspection followed by cross-path synthesis. The phase-repair table and backend-exception determination require source inspection (handoff items #1-36) before synthesis. This memo completes the first invocation (plan internal coherence); those elements belong in the final synthesis memo after bounded source reviews complete.

---

VERDICT: AGREE

---

## Comparison With Self-Review Record

The self-review at `bayesfilter-ordinary-hmc-migration-debt-plan-review-2026-09-02.md` was read after the independent plan assessment above completed. The self-review verdict is `PASS_FOR_STATIC_AUDIT; REVISE_BEFORE_NUMERICAL_EXECUTION` (line 234), which agrees with this audit's finding that the plan is coherent for static work and properly blocks numerical execution.

### Overlapping Findings

Both reviews identify:
- Baseline risk from capability vs executable default confusion (self-review §1, this audit "Unsupported claim 2")
- Proxy-metric classification as explanatory (self-review §2, this audit "Severity 2: Proxy-metric check")
- Hidden defaults requiring policy records (self-review §5, this audit "Fairness and hidden assumptions")
- Stale context and naming conflicts (self-review §6, this audit "Severity 2: Stale context")
- `use_xla=False` as policy mismatch (self-review §9, this audit "Material omission 1")
- Duplicate public definition creating import-path ambiguity (self-review §10, this audit noted in deferred backend review)
- Artifact schema needs resolved policy exposure (self-review §8, this audit "Material omission 2")

### Complementary Findings

**Self-review adds:**
- Preset role authority gap (§5, lines 103-105): `standard` vs `serious` not interchangeable
- Legacy algorithm reachability through active route (implicit in §1, lines 37-39)
- Replay signature enforcement requirement (§4, lines 85-87)
- Top-level result must expose resolved branch (§8, lines 131-136)

**This audit adds:**
- Test execution ordering across phases (Material omission 1)
- Artifact backward-compatibility policy for old payloads (Material omission 2)
- Consumer classification decision procedure (Material omission 3)
- NumPy scope enumeration gap (Unsupported claim 1)
- Dynamic import scanning limitations (Hidden assumption 2)
- Stale-policy scan should be construction gate (Severity 2: Stale context)

### Disagreements

None. The two reviews converge on the same core finding: the plan correctly structures a static audit that identifies authority/implementation mismatches without authorizing numerical work or default changes. Both flag the XLA-default and preset-role questions as open repairs requiring Phase 1 resolution.

### Integration

The self-review's open-repair items (lines 217-224) map directly to this audit's P2 clarifications and the handoff's sequential source-review questions. Specifically:

| Self-review open item | Handoff source review item | This audit finding |
|---|---|---|
| Preset authority | Items #2, #3 (config, selector) | Deferred to source review |
| Stale docs/registry reconciliation | Item #12 (tuning_contract) | Unsupported claim 2 |
| Backend default policy | Item #2 (use_xla default) | Material omission 1 → P1 next |
| Duplicate public definition | Item #2 (compatibility delegate) | Noted for item #2 review |
| External consumer scope | Items #27-36 (downstream files) | Material omission 3 |
| Top-level resolved-policy field | Item #2, #15 (kernel_tuning, artifacts) | Material omission 2 |

The self-review's verdict and this audit's verdict both approve proceeding to sequential bounded source inspection, which is the next step in the handoff protocol.

---

## Summary And Next Steps

### Current Status

The plan `bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md` is **internally coherent for static audit scope** and properly separates source facts from owner decisions. It may proceed to Phase 0 source trace and consumer classification under the bounded read-only protocol.

### Severity-Ordered Action Items

**P1 (resolve before Phase 1 implementation):**
1. Source inspection of dispatch and ordinary orchestration (handoff items #1-2) to resolve operational-branch reachability and module-header contradiction
2. Determine whether `use_xla=False` has an explicit exception record or requires repair to XLA-default

**P2 (resolve during Phase 0-1):**
1. Add test execution ordering guidance to Phase 5
2. Define artifact backward-compatibility policy for old payloads
3. Write consumer classification decision procedure for Phase 0
4. Enumerate NumPy imports and runtime reachability for Phase 7 scope
5. Clarify joint-grid implementation identity (new algorithm vs promoted existing)
6. Address dynamic-import scanning limitations if AST fails
7. Move stale-policy scan earlier as construction gate

### Required Owner Decisions (unchanged from plan)

- Ordinary canonical policy selection (measured joint grid vs per-L adaptation vs reviewed dynamic trajectory)
- Public API structure (split names vs typed dispatcher with fail-closed mechanics branches)
- Primary efficiency/selection criterion for future numerical validation
- Campaign budget for Phase 8 numerical work

### Recommended Next Invocation

Per handoff protocol (lines 84-112), begin sequential bounded source review with:

**Handoff item #1:** `bayesfilter/inference/hmc_tuning_dispatch.py`

**Question:** Does one public name dispatch materially different config variants and evidence contracts? Can a mechanics-only TensorFlow result be mistaken for ordinary artifact authority, and are forbidden combinations rejected?

This resolves the P1 dispatch-ambiguity finding and prepares for item #2 (ordinary orchestration), which addresses the P1 XLA-default and module-header questions.

---

## Document Control

| Field | Value |
|---|---|
| Review mode | READ-ONLY BOUNDED REVIEW (plan coherence only) |
| Plan revision | 54201f5cd925ed15036bad8156606b812d53b045 (observed, with dirty worktree) |
| Scope | Internal plan coherence, not source code verification |
| Source files inspected | None (plan-only review per handoff first invocation) |
| Commands executed | None |
| Verdict | AGREE (proceed to sequential source review) |
| Next invocation | Handoff item #1: dispatch source inspection |
| Authoring agent | Claude Opus 5 (1M context) |
| Review date | 2026-09-03 |

