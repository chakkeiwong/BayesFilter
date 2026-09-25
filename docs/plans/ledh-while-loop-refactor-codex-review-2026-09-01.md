# Codex Review: LEDH While-Loop Refactor Program

**Review date:** 2026-09-01
**Program:** ledh-while-loop-refactor-2026-08-30
**Reviewer:** Codex
**Readiness:** APPROVE_WITH_CORRECTIONS

## Review basis

The requested handoff path does not exist in the current checkout. I reviewed
the exact handoff and its referenced material in
/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild, at
commit 22042a1e1c4661462211d67b82b729b221488d51. The handoff is also present
in archived commit 2478d3da; the target worktree was clean at review time.
The current checkout contains unrelated dirty changes, which were not touched.

The review covered the master program, the Phase 0 through Phase 4 subplans,
the earlier audit memo, the delivery-status note, the wrapper script, the
implementation and test call chain, and the files named as supporting
diagnostics. The checks were documentary and static: path existence, line
reference, signature and call-site inspection, shell syntax validation, and
environment/provenance inspection. No pytest, benchmark, GPU command, or
refactor was run, so this note establishes no numerical result.

## Executive verdict

The proposed engineering direction is viable in principle: replacing the two
Python-unrolled loops with bounded TensorFlow while loops, then measuring the
multi-direction tangent representation, is a coherent refactor question. The
program is not executable as written, however. The wrapper invokes five files
that are absent from the target worktree, the Phase 0 coverage configuration is
placed where coverage.py will not discover it, the baseline counts and API
shape contract are internally inconsistent, and several phase transitions
still require an unplanned choice or plan change.

This is not a structural rejection of the refactor. It is an
APPROVE_WITH_CORRECTIONS decision: repair the execution surface, freeze a
reproducible baseline and evidence contract, and resolve the R7/R8/R9
contradictions before Phase 0 starts. The four corrections already listed in
the handoff remain pending in the reviewed documents; the register below adds
further findings.

## R1-R10 compliance

| Requirement | Status | Reason |
|---|---|---|
| R1: Phase 0 audit, coverage, and contract before refactor | DEFECT | Phase 0 is scheduled first, but its counts are wrong, coverage is not actually configured, and its acceptance criterion does not define a required coverage threshold or adequate test matrix. |
| R2: Master program with all details and phases | AT_RISK | The master is detailed, but it cites absent executable files, mixes incompatible baselines, and contains a fused score-shape contradiction. |
| R3: Subplan for every phase | AT_RISK | Five subplans exist, but their preconditions and branch rules disagree (notably Phase 2 rejection, Phase 3 optional work, and Phase 4 deliverables outside the allowlist). |
| R4: Background sufficient for a fresh agent | DEFECT | The narrative depends on diagnostics and a surrogate driver that are not in the target worktree; the cited measurements cannot be reproduced from the stated checkout. |
| R5: Required tool/command allowlist | DEFECT | The wrapper exists and parses, but it omits required files and writes, its target path is hard-coded, and the proposed settings JSON is not a verified Claude settings format. |
| R6: Upfront approvals for end-to-end execution | AT_RISK | A1-A5 enumerate decisions, but they do not authorize all actual boundaries and add a phase-boundary approval that conflicts with the no-click execution requirement and current proportional governance. |
| R7 (hard): no mid-execution choice questions | DEFECT | Missing files, package installation, GPU entries, owner approval after Phase 0, and unresolved branch outcomes can force a question after execution has begun. |
| R8 (hard): no mid-execution plan changes | DEFECT | "Refresh the next subplan if new information appears" permits scope changes; Phase 2 has an unresolved reject/continue branch and Phase 3 has optional work versus all-phase completion. |
| R9: repair and refresh after every phase | DEFECT | Repair scopes exist, but stop criteria, retry limits, and the meaning of a skipped optional item are not consistent enough to produce a deterministic next phase. |
| R10: Codex audit memo | SATISFIED | The handoff is a complete review request and this note supplies the requested compliance, defect, risk, hard-constraint, and readiness sections. |

## Acknowledged corrections still pending

These are the four errors explicitly identified by the handoff. They are
confirmed here as unresolved document defects, not presented as new findings.

| ID | Evidence | Required correction |
|---|---|---|
| K1 | Phase 0 counts five fused substitutions and includes the authority call at line 111; the actual test-side substitutions are at lines 119, 136, 143, and 161. | Freeze the four-call fused inventory and leave the authority call as flow_substeps. |
| K2 | Phase 0 lists one non-fused substitution; the actual test-side calls are at lines 85, 104, 108, 125, and 130, while line 80 is the authority call. | Freeze the five-call non-fused inventory plus one kernel forwarding fix. |
| K3 | The master and Phase 0 state score [B,P], while the current fused function returns one directional score [B] and Phase 2 proposes [B,K]. | Define one canonical rank-2/rank-3 API and update every contract, test, and consumer. |
| K4 | The original device-memory success criterion cannot be measured by the CPU-only wrapper. | Keep host RSS, graph nodes, and GraphDef bytes as the current criterion; record GPU memory as a separately authorized open item. |

## Defect register

### Blocking

| ID | Location | Finding | Required fix |
|---|---|---|---|
| D1 | Master references; wrapper lines 82-99 | diagnose_graph_size_20260830.py, diagnose_eval_time_20260830.py, diagnose_direction_cost_scaling_20260830.py, step1_true_surrogate_force.py, and the correction note are absent from the target worktree. The wrapper modes therefore fail before producing evidence. They exist only in the separate section-3-6-sidecar-force worktree. | Add the exact files to the canonical-rebuild branch, or replace every reference with a tracked target path. Verify the files at the pinned commit before execution; do not silently borrow sidecar files. |
| D2 | Master Allowlist; Phase 4 deliverables | Phase 4 must edit the surrogate driver, add a retracing test, write program/reset/result notes, and rerun diagnostics, but those paths are not in the allowlist. The wrapper itself is also not listed as an editable target, and generated coverage output can overwrite prior evidence. | List exact permitted paths and writes, including the wrapper, driver, tests, phase results, and a fresh versioned artifact directory. Use real settings syntax and keep broad wildcards out. |

### Major

| ID | Location | Finding | Required fix |
|---|---|---|---|
| D3 | Phase 0 Coverage Tooling, lines 151-180 | The plan adds [coverage:run] and [coverage:report] sections to pytest.ini. coverage.py does not search pytest.ini for those sections; the inspected search list contains .coveragerc, pyproject.toml, setup.cfg, and tox.ini. A successful pytest run could therefore report the wrong configuration or no configuration. | Put coverage settings in .coveragerc, pyproject.toml, or setup.cfg, or pass an explicit --cov-config. Pin and record pytest-cov/coverage versions and fail closed if the plugin is unavailable. |
| D4 | Master lines 196-198; Phase 0 lines 183-200 and 330-346 | The claimed 106 tests with 21 failures, 83 passes, and 8 errors is arithmetically inconsistent (83+21+8=112). The current parity files contain 3 fused tests and 4 non-fused tests, not 3 and 3. Expected 89/15/8 counts are consequently not a reliable baseline. | Run one pinned collection command, save machine-readable node IDs and outcomes, and derive every expected count from that manifest. Treat the seven parity tests and the separate capability-surface test explicitly. |
| D5 | Phase 0 Repair Plan and Deliverables | The stated API-drift inventory would modify an authority call and misses four non-fused calls. Applying it literally can break the authority path while leaving batch calls broken. | Replace line-based prose with an exact call-site inventory (file, line at review, callee, intended keyword), then verify authority and batch paths separately. |
| D6 | Master Refactor Contract line 258; Phase 0 line 357; Phase 2 lines 255-256 | The public score shape is contradictory. The current fused lane returns [B], the non-fused lane returns [B,P], and the proposed rank-3 direction input returns [B,K]. Existing consumers are not shown at the boundary where this changes. | State rank-2 and rank-3 signatures, promotion/squeeze rules, and diagnostics shapes in one normative section. Add compatibility tests for every claim-bearing consumer. |
| D7 | Phase 0 coverage acceptance lines 321-326; parity gates lines 269-272 | "Target >=85%" is mentioned but not a pass/fail condition. The measured scope is only two small parity files and does not exercise invalid rows, NaN handling, optional density, all model callbacks, horizon/substep edges, or the proposed K path. Manual inspection is not a coverage metric. | Define line and branch metrics, a hard threshold or justified baseline rule, and a contract matrix with positive and negative tests before refactoring. Do not continue on a tooling fallback described as coverage. |
| D8 | Master implementation contract lines 262-267; current kernel and tests | The contract requires a stable input_signature and XLA-compatible execution, but the current kernel is not a compiled entry point and the tests use tf.function without input_signature or jit_compile=True. The wrapper has no XLA mode. | Add a repository-owned compiled wrapper with explicit static signatures, a bounded retracing test, and an explicit XLA compatibility smoke (or mark the CPU graph path as a documented exception with no XLA claim). |
| D9 | Master graph diagnosis lines 85-111; stop criteria 380-381; Phase 4 table | Baselines mix one-call B=1 measurements, six-call gradient measurements, and a batched comparison. The text reports both 281.6 s and 101.8 s trace figures without defining which is normative. The graph-size sidecar still uses time.time, and RSS is sampled cumulatively in one process. These artifacts cannot support a 2x or 20x gate. | Define one fixture, call count, shape, clock, warm-up procedure, peak-memory measure, and isolated subprocess protocol. Store the baseline output before changing code and use the same protocol after each phase. |
| D10 | Master success criterion 6; Phase 4 lines 30-49 and 73-83 | "Driver runs without modification" conflicts with the explicit Phase 4 driver update. Acceptance-rate agreement within 5% is a sampler comparison, lacks seeds/replicates/uncertainty, and conflicts with the stated no-sampler-claim scope. | Replace this with a deterministic finite-value/finite-gradient integration smoke and a retracing assertion. Defer acceptance and posterior claims to the separate validation program. |
| D11 | Master lines 128-134 and 379; Phase 1/4 verification | XLA incompatibility is a mandatory stop condition and the contract says the current operations are XLA-compatible, but no executable XLA command or artifact is supplied. | Add a bounded XLA compile/equivalence check under the declared hardware and record its status, or explicitly remove XLA from this program's claims and make it a prerequisite for the next program. |
| D12 | Master non-functional contract line 276; Phase 4 objective; target call chain | The assertion that surrogate-force, leaderboard, and NeuTra callers work is not backed by a consumer inventory or wiring test. In the target checkout, direct searches find parity tests and the NeuTra adapter but no tracked surrogate driver or leaderboard endpoint. The KSC target adapter also imports NumPy at runtime (line 502), which conflicts with the diagnostic-only NumPy rule if it is claim-bearing. | Enumerate every claim-bearing endpoint, trace each to the general implementation, add a wiring/parity test, and classify or remove the KSC runtime NumPy dependency before making a runtime claim. |
| D13 | Phase 2 design lines 125-154 and 193-206 | The proposed K loop is a Python unroll inside the TensorFlow loop body. No bound, shape validation, or direction-rank guard prevents a caller from producing a new graph proportional to an arbitrary K. The planned tests do not use explicit signatures. | Validate K and direction shapes at configuration time, declare a supported K maximum, test K=1, 2, 5, P, and invalid K, and measure graph growth under the exact compiled signature. |
| D14 | Master repair policy lines 391-416; Phase 1-4 repair sections | "Numerical discrepancies within tolerance" is listed as repairable while a parity failure is also a stop condition; Phase 2 says a failed premise is a stop but Phase 3 applies anyway; Phase 3 says all items are skippable while program completion requires all phases; Phase 4 allows reducing smoke size without a predeclared bound. | Add a phase decision table with named vetoes, bounded repair attempts, deterministic branch outcomes, and a definition of completion when optional work is skipped. A repair may update factual notes, not scientific scope, comparator, threshold, or budget. |
| D15 | Master timeline/risk sections; all result deliverables | There is no total attempt budget, wall-time budget, or unique output root. The coverage HTML and diagnostic outputs can overwrite evidence, and the repair language permits repeated retries. | Declare the maximum attempts and compute/wall-time budget, create a fresh phase/attempt directory, and require a manifest with commit, command, environment, seeds, CPU/GPU status, wall time, and artifact hashes. |
| D16 | Phase 0 lines 217-232 and wrapper ignores | Missing fixtures and collection-error files are treated as out of scope, but the canonical command still presents a partial result as a clean baseline. A future refactor can alter shared imports while the ignored errors remain invisible. | Record an expected-failure manifest with exact node IDs and collection errors. Add a minimal import/call-chain check for shared modules, and state that the result is not a repository-wide regression guarantee. |
| D17 | Delivery-status note; master dependencies | The delivery note says all diagnostics and plans were delivered, but the target branch lacks the five supporting files. It also records an environment ambiguity: the status says the base interpreter was selected while the current target check resolves Python from tftwogpu. | Correct the status note and record sys.executable, conda environment, TensorFlow/TFP versions, branch, and commit in every run manifest. |
| D18 | Master allowlist lines 318-338; wrapper lines 21-23 | The master lists GPU pytest as approved-but-escalated while the active wrapper forcibly hides GPUs and the program says GPU validation is deferred. A GPU command would require a new permission boundary in the middle of execution. | Remove the inactive GPU command from the autonomous path and record GPU validation as a separately authorized step. Keep the CPU-only artifact explicit. |
| D19 | Phase 1 preconditions and Phase 4 baseline table | Phase 1 uses a B=1, horizon=50 baseline, while the hard memory stop and final criterion use B=6, N=252. No B=6 baseline is supplied, so the denominator for the stop condition is undefined. | Measure and freeze the exact final-shape baseline, including peak host memory and graph bytes, before Phase 1. |
| D20 | Phase 0 installation step | pip install pytest-cov is an environment/network mutation with no version pin or offline failure path. Treating an unavailable plugin as a manual-inspection fallback would silently weaken R1. | Authorize the package boundary before execution, pin a compatible version or use an existing environment, and stop Phase 0 if a quantitative coverage baseline cannot be produced. |

## Hard-constraint analysis

### R7: no mid-execution choice questions

R7 is not satisfied by naming an allowlist alone. The following interruption
points are visible in the program:

| Point | Why a question can arise | Status and correction |
|---|---|---|
| Before Phase 0 | A1-A5 ask for five separate YES/NO/REVISE answers, and Phase 0 also requires owner approval before Phase 1. | Replace this with one explicit upfront campaign authorization covering the fixed plan, target worktree, bounded repairs, and required package boundary; retain a stop report for true scientific vetoes. |
| Coverage installation | pytest-cov is not installed and installation changes the environment. | Resolve authorization and version before launch; do not discover the missing permission after Phase 0 has started. |
| Missing diagnostic/driver files | The wrapper reaches a path error, leaving an agent to ask whether to copy from the sidecar or change the plan. | Track the files or remove the modes before launch. |
| Allowlist mismatch | Phase 4 edits files not listed in the declared write scope. | List exact paths and generated artifact roots before launch. |
| GPU entry | The master calls GPU testing approved but requiring escalation, while the wrapper is CPU-only. | Remove it from autonomous execution or obtain separate approval before the campaign begins. |
| Phase 2 result | The plan permits retaining the swept form if the multi-direction premise fails, but does not define whether this is an automatic branch or a user decision. | Predeclare the branch and its next phase, result wording, and budget consumption. |
| Phase 3 optional work | The agent may skip several items, but completion still says all required work was done. | Define "skipped with reason" as a completed branch, or make the item mandatory with a stop condition. |
| Stop conditions | The distinction between a tolerable discrepancy, a repairable failure, and a mandatory stop is not operationalized. | Add numeric thresholds, attempt limits, and a decision table before execution. |
| Final promotion | Phase 4 requires owner acceptance, despite the stated no-click execution and the program's claim that it does not establish promotion evidence. | Make final owner review a post-campaign boundary, not an unannounced phase transition. |

The hard constraint is therefore **DEFECT**. The intent is clear, but the
current program can reach a question without a predeclared answer.

### R8: no mid-execution plan changes

R8 is also **DEFECT**. The master explicitly says to update the next subplan if
new information appears (lines 403-408), which is a plan change unless limited
to factual result notes. The following cases need a fixed rule:

- Phase 0's corrected call counts change the work and expected result, but the
  documents still carry the old counts.
- Phase 2 has both "stop if the premise fails" and "Phase 3 applies either
  way." That is a valid scientific branch only if it is frozen before running.
- Phase 3 says each item is skippable and the phase cannot block, while the
  master requires all phases and all technical criteria.
- Phase 4 says to update the driver, but the master success criterion says the
  driver must run without modification.
- A repair may alter an input signature, comparator, metric, or threshold under
  the current wording; those are scientific-plan changes, not localized repairs.

The repair rule should permit only implementation repairs that preserve the
target, comparator, criteria, vetoes, hardware class, privacy boundary, and
campaign budget. A factual refresh of the next note is acceptable; changing
the method or accepting a new metric requires a stop and a new authorized
plan.

### R9 interaction

R9 can be made compliant, but is not yet. Each phase has a nominal repair
section, yet none defines a maximum number of attempts or a machine-readable
repair log. "Proceed after repair" is unsafe when the same defect can recur.
The corrected procedure should record failure classification, repair, focused
regression, wall time, remaining budget, and the deterministic next branch.

## Evidence contract audit

The documents contain useful criteria, but not one complete pre-run evidence
contract. Before execution, record the following in the master and each phase
manifest:

| Field | Required content |
|---|---|
| Question | Does bounded while-loop execution preserve the current value/analytical score and reduce graph-construction cost at the declared target shape? |
| Comparator | The unmodified fused lane at the same commit, fixture, dtype, shape, call count, and execution mode; the single-cloud authority is the parity authority, not a performance comparator. |
| Primary criteria | Value/score parity at rtol 5e-4 plus graph/GraphDef/trace metrics from an isolated, repeatable protocol. |
| Vetoes | Non-finite output, authority mismatch, API or call-chain break, missing required artifact, undefined baseline, failed compiled-signature/XLA check when claimed, or a declared memory/performance threshold. |
| Explanatory diagnostics | Warm timing, node counts by K, RSS, optional branch coverage, and per-test differences. These cannot certify sampler or posterior quality. |
| Nonclaims | No posterior correctness, convergence, acceptance-rate adequacy, tuning, GPU throughput/memory, statistical superiority, or source-faithfulness claim. |
| Artifact | A fresh versioned directory containing JSON results, Markdown interpretation, run manifest, exact command, environment, seed, commit, and hashes. |

Until this contract is written and the baseline is captured, a successful
command would not answer the stated refactor question.

## Risk register

These are residual risks rather than confirmed defects. They remain after the
required corrections and should be tested in the stated order.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TensorFlow shape invariants or captured tensors are wrong in a nested loop. | Medium | High numerical drift or a trace-time failure. | Add intermediate-state parity at each outer step on a tiny fixture and test both optional observation-density branches. |
| Python K unrolling recreates the graph explosion for large K. | High | The claimed graph reduction disappears. | Enforce a configuration-time K bound and measure K=1, 2, 5, and P under one compiled signature. |
| Static signatures retrace when a Python control or shape changes. | Medium | HMC cost grows despite a successful smoke. | Assert concrete-function count over repeated identical calls and record all pinned dimensions. |
| Floating-point reassociation is larger than the declared tolerance. | Medium | A parity failure may be misclassified as an implementation bug or hidden by tolerance widening. | Compare value and score separately, use fixed float64 fixtures, and preserve the single-cloud authority as the reference. |
| User-supplied model callbacks accept rank-2 only. | Medium | A rank-3 tangent crosses the callback boundary and changes the mathematical program. | Test rank promotion/squeeze at the boundary and retain a K-looped internal call when required. |
| CPU timing does not predict the intended GPU route. | High | Host-side improvement is incorrectly promoted to production performance. | Label all current numbers CPU diagnostic evidence and schedule a separate escalated GPU experiment. |
| Ignored collection errors mask a shared import regression. | Medium | A refactor appears clean while the wider suite is unavailable. | Preserve the expected-error manifest and run explicit shared-import/wiring checks. |
| Environment drift changes TensorFlow graph construction or timings. | Medium | Baselines become incomparable. | Pin interpreter and package versions in every manifest and use a fresh output directory per attempt. |
| KSC runtime NumPy constants violate the backend policy in a claim-bearing call. | Medium | The advertised TensorFlow-only path is false. | Replace with TensorFlow constants or classify the adapter as diagnostic-only before admission. |
| Repeated repairs consume unbounded time or overwrite evidence. | Medium | Campaign loses reproducibility and budget control. | Set attempt and wall-time caps, retain every failed attempt, and stop at budget exhaustion. |

## Required corrections before execution

1. Track the five missing diagnostics/driver files in the target branch, or
   remove their references and rewrite the wrapper.
2. Correct K1-K4 and the additional test-count arithmetic; capture a fresh
   node-ID baseline for the actual seven parity tests.
3. Resolve the fused/non-fused score-shape contract and add consumer wiring
   tests.
4. Move coverage configuration to a discoverable file, define a hard metric,
   expand the contract test matrix, and stop if quantitative coverage is
   unavailable.
5. Freeze one measurement protocol and a B=6 baseline, with isolated
   subprocesses and consistent clocks.
6. Add the explicit compiled input signature, retracing check, and an XLA
   smoke or a clearly documented non-XLA exception.
7. Rewrite the allowlist as exact file/command permissions, including phase
   result and artifact writes, and resolve package-install authorization before
   launch.
8. Replace A1-A5 and phase-boundary prompts with one upfront campaign
   authorization plus predeclared deterministic branches; reserve owner review
   for the actual final boundary.
9. Make repair attempts bounded and classify every failure as repairable,
   promotion-veto, continuation-veto, or explanatory.
10. Remove the acceptance-rate criterion from this refactor and correct the
    delivery-status/provenance note.

## Final readiness decision

**APPROVE_WITH_CORRECTIONS.** The while-loop refactor remains a reasonable
engineering experiment, but no phase should start from the reviewed snapshot.
The missing execution files, invalid coverage setup, undefined baselines, API
contradictions, and unresolved no-click branches are material. Once the
corrections above are committed on the canonical-rebuild branch and Phase 0
produces a manifest-backed baseline, the program can proceed under one fixed
campaign authorization. This review does not approve the refactored kernel,
an HMC sampler, or any scientific/default promotion.
