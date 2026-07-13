# BayesFilter Kalman QR Batched XLA Repair Master Program

Date: 2026-07-11
Status: `SUPERSEDED_2026_07_13_BY_ACADEMIC_RISK_TIER_WORKFLOW`
Supervisor/executor: Codex in the current conversation
Reviewer: Claude Opus was explicitly approved by the user but remains policy-blocked; fresh bounded Codex read-only substitute review is active and is weaker than Claude review

Supersession note, 2026-07-13: this file is a historical program and evidence
index. Its mandatory per-phase subplans, hash freezes, snapshots, and repeated
review loops no longer govern execution. The active lane plan is
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-plan-2026-07-13.md`
under the risk-tier workflow in `AGENTS.md`. Completed artifacts remain valid
for their stated narrow evidence; no prior memory/performance promotion is
implied.

## Objective

Repair the Kalman QR score benchmark so that it compares true-batched analytical
and reverse-mode autodiff score paths under equivalent TensorFlow/XLA semantics,
localizes compilation failures, records trustworthy timing/provenance, and only
then runs a bounded CPU/GPU comparison ladder.

This program continues the completed dtype and batched-score engineering work in
`docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`.
It supersedes only that program's invalid Phase 7/7B benchmark route. It does not
revert or reopen the completed dtype work.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | At fixed model/data and equivalent batch semantics, do true-batched analytical and reverse-mode Kalman QR score paths compile under XLA, preserve numerical parity, and what are their synchronized warm runtimes? |
| Candidate mechanisms | True batched fixture tensor algebra, parameter-axis analytical derivative vectorization, corrected true-batched autodiff, method-isolated compilation, and separated trace/compile/runtime/reporting measurement. |
| Exact baseline | The Phase 0 commit, dirty-tree manifest, and per-file SHA-256 fingerprint, plus the failed 2026-07-09 overnight artifacts identified by `docs/plans/bayesfilter-kalman-qr-batched-xla-reset-memo-2026-07-10.md`. Scalar analytical and scalar autodiff rows are correctness references only at small batch. |
| Expected failure modes | XLA graph/codegen remains too large after vectorization; GPU XLA rejects QR reverse-mode layouts; analytical forward sensitivities are genuinely slower at large `P`; parity regresses; artifact resume logic reuses stale failures; CPU thread requests are mistaken for core pinning. |
| Primary promotion criterion | Common harness, fixture, math, comparator, and measurement gates in Phases 0-5 pass. CPU and GPU compile/runtime outcomes are lane-local: the final comparison ladder is launchable on each lane only after that lane has a fair pair of method-isolated XLA arms with finite outputs, parity, and complete provenance. |
| Promotion veto | Compile crash/timeout, non-finite timed output, parity failure, stale artifact reuse, method mismatch, unrecorded device/JIT/dtype/source identity, or timing that includes unreported full-output serialization. |
| Continuation veto | Broken common harness/math/fixture validity, corrupted or missing required artifact, a common correctness check remains broken after phase-local repair, boundary crossing requires new human authority, or the same material review blocker fails to converge after five rounds. A backend-specific compile failure is lane-local and does not automatically veto the other lane. |
| Repair trigger | Any promotion veto, GraphDef/HLO growth inconsistent with the intended vectorized structure, corrected batched autodiff failing row-independence, or supervisor status semantics misreporting failures as complete. |
| Explanatory diagnostics | GraphDef nodes/bytes, optional HLO bytes, trace time, first executable call, synchronized warm calls, host materialization time, peak RSS/device memory where available, requested thread settings, CPU affinity, and per-method failure stage. |
| Must not conclude | No universal speed superiority, statistical ranking without uncertainty, production/default readiness, HMC readiness, posterior correctness, broad scientific validity, or physical-core scaling from requested TensorFlow thread counts alone. |

## Phase Index

| Phase | Name | Subplan | Required result |
| --- | --- | --- | --- |
| 0 | Contract, baseline, and source fingerprint | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase0-contract-baseline-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase0-contract-baseline-result-2026-07-11.md` |
| 1 | Harness failure isolation and artifact integrity | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase1-harness-integrity-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase1-harness-integrity-result-2026-07-11.md` |
| 2 | True-batched fixture tensor algebra | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-result-2026-07-11.md` |
| 3 | Analytical parameter-axis vectorization | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-result-2026-07-11.md` |
| 4 | Correct true-batched autodiff comparator | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-result-2026-07-11.md` |
| 5 | Compile/runtime measurement separation | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-result-2026-07-11.md` |
| 6 | GPU-hidden CPU trace and XLA gates | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-result-2026-07-11.md` |
| 7 | Trusted GPU XLA method-isolated gates | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase7-gpu-xla-gates-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase7-gpu-xla-gates-result-2026-07-11.md` |
| 8 | Gate-approved comparison ladder | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase8-comparison-ladder-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase8-comparison-ladder-result-2026-07-11.md` |
| 9 | Closeout and reset memo | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase9-closeout-subplan-2026-07-11.md` | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase9-closeout-result-2026-07-11.md` |

## Program Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the benchmark produce method-isolated, resumable, source-fingerprinted CPU/GPU XLA evidence for equivalent true-batched score paths? |
| Comparator | True-batched analytical score versus true-batched reverse-mode autodiff. Scalar row paths are small-batch correctness references, never the large-batch promotion comparator. |
| Primary pass criterion | Common gates in Phases 0-5 pass. Phases 6-7 classify CPU/GPU outcomes independently. Phase 8 may run only on a lane with a fair pair of viable methods; included rows pass finite/dtype/shape/parity/provenance gates and preserve paired timing replications. |
| Hard vetoes | Compile crash/timeout, invalid JSON/artifact, stale reuse, missing method result, non-finite output, parity failure, incorrect device/JIT/dtype, or review/authority boundary violation. |
| Explanatory only | Graph/HLO sizes, compile time, warm runtime, memory, tail metrics, and observed ratios until uncertainty analysis supports a ranking. |
| Statistical evidence | A ranking requires an uncertainty-bearing, predeclared paired procedure. Before Phase 8 data collection, its refreshed subplan must freeze the fresh-process sampling unit, pairing key, balanced/randomized method order, cache policy, minimum usable replication count, estimand, interval/test, and ranking rule. A paired point estimate alone cannot support ranking. |
| Not concluded | HMC/posterior/scientific/default/production readiness or universal method superiority. |
| Preserving artifacts | Master/runbook/ledger, per-phase subplans/results, bounded reviews, source hashes, JSON/Markdown/log outputs, required run manifests, and final reset memo. |

## Default And Assumption Audit

| Choice | Provenance | Classification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| `T=120`, dimensions `10/20/30`, `P=50/150`, `B=1/4/16` | Original user-requested grid recorded in reset memo | Inherited target grid | Grid launched before harness validity | Phases 0-7 gate Phase 8 |
| CPU requested threads `1/4/16` | Original requested grid | Requested runtime setting, not physical-core pin | Report calls it core scaling | Record affinity and label accurately |
| CPU row timeout 300 seconds in smoke gates | Convenience cap replacing the failed 3600-second compile wait | Hypothesis/convenience | Viable compile killed too early | Phase 6 starts smallest cases and records stage; revise only prospectively |
| Claude probe timeout 90 seconds; material review 180 seconds | Review-gate guide plus first-call latency caution | Inherited operational default | Live reviewer mislabeled dead | Probe status and prompt redesign protocol |
| At most five review rounds per material blocker | User directive | Binding stop rule | Infinite review churn | Ledger round counter |
| XLA JIT on | BayesFilter owner policy | Binding default | Debug non-JIT evidence promoted | All benchmark artifacts record `jit_compile`; non-JIT only localization |
| GPU `float32` first, then `float64` | Existing grid and cheapest production-target preflight | Reviewed sequence | FP32 failure blocks analytical-only evidence due to comparator coupling | Method-isolated Phase 7 gates |
| Timing replications | Determined prospectively in Phase 8 from Phase 6/7 runtime feasibility | Not yet fixed | Arbitrary count wastes time or overstates uncertainty | Phase 8 subplan must be refreshed before launch |
| Float32 parity tolerances: value/score absolute `5e-3` | Current harness `_dtype_tolerances`; inherited from the failed benchmark | Inherited hypothesis, not yet revalidated | Too loose hides regression | Phase 0 records source; Phase 4 validates against scalar references; Phase 8 freezes prospectively |
| Float64 parity tolerances: value `1e-8`, score `1e-5` | Current harness `_dtype_tolerances`; inherited from the failed benchmark | Inherited hypothesis, not yet revalidated | Asymmetric or too loose/tight | Phase 0 records source; Phase 4 validates against scalar references; Phase 8 freezes prospectively |

## Baseline Ladder

1. Scalar analytical and scalar autodiff at `B=1`: correctness references.
2. True-batched analytical and corrected true-batched autodiff: primary method comparison.
3. Existing unvectorized graph-size measurements: historical failure baseline only.
4. GPU-hidden CPU XLA: compile/reference evidence, not production-target evidence.
5. Trusted GPU/XLA: target-device engineering evidence.

## Skeptical Plan Audit

- Wrong baseline: repaired by making equivalent true-batched paths primary and retaining scalar rows only for correctness.
- Proxy promotion: graph/HLO size and compile survival are explicit diagnostics, not runtime superiority criteria.
- Missing stop conditions: every phase declares engineering vetoes, continuation vetoes, repair triggers, and handoff gates.
- Unfair comparison: batch construction, synchronization, and reporting semantics must be common before timing.
- Hidden assumptions: timeouts, tolerances, replication counts, TF32, affinity, and dtype are recorded with provenance.
- Stale context: Phase 0 fingerprints current dirty source and marks 2026-07-09 artifacts historical/non-resumable.
- Environment mismatch: GPU devices are hidden before TensorFlow import for deliberate CPU execution; GPU evidence uses the managed-session trust basis or explicit trusted permission.
- Artifact adequacy: method isolation and stage-specific status ensure a failure artifact answers where execution stopped.
- Misleading pass: compiling alone cannot pass a runtime comparison; parity alone cannot establish speed; one repeat cannot rank.
- Misleading fail: one candidate or backend failure does not invalidate the Kalman math or the repair direction unless a declared continuation veto fires.

Audit status: `PASSED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND_3_FOR_PHASE_0_ONLY`.
Later execution is gated by the dedicated subplans.

The user explicitly approved bounded repository disclosure to Claude. The
trusted execution layer nevertheless rejected the review gate before its probe
because external disclosure remains policy-blocked. No Claude prompt or plan
content was sent. This is not Claude liveness evidence. Fresh bounded Codex
substitute review is therefore active and must be labeled weaker than Claude
review. Phase 0 has not started.

## Lane-Local Routing

- Common harness, fixture, score-target, parity, or artifact invalidity blocks
  both CPU and GPU lanes until repaired.
- A GPU-hidden CPU XLA compiler/codegen timeout or failure blocks only the CPU
  lane under the declared resource cap. It is not proof that compilation is
  impossible and does not prevent the smallest trusted GPU gate when common
  correctness checks pass.
- A GPU-specific XLA failure blocks only the affected GPU arm and does not erase
  valid CPU reference evidence.
- Phase 8 requires a fair analytical/autodiff pair on the same lane. If no lane
  has such a pair, write a negative/blocker result rather than forcing a ranking.

## Timed Callable And Pairing Boundary

- Untimed setup: deterministic observations, base tensors, derivative bases,
  parameter batch, function construction, tracing, XLA compilation, and warmup.
- Timed analytical callable: parameter batch to batched model tensors,
  analytical derivative-input tensors, likelihood, and score, followed by
  device synchronization. Nothing computed from the parameter batch for the
  score may be precomputed outside this boundary.
- Timed autodiff callable: the same parameter batch to the same batched model
  tensors and likelihood, plus tape/reverse-mode score, followed by device
  synchronization.
- Untimed reporting: full output transfer, `.numpy()`, Python/JSON conversion,
  parity calculations, and artifact writing. Materialize once per checked
  output outside warm timing.
- Fixture/data/source hashes must match within a paired comparison. Each paired
  replication uses fresh processes under the Phase 8 cache policy, and method
  order follows the prospectively frozen balanced/randomized schedule.
- Scaling fixtures must be nested across `P` and `B` where mathematically
  possible: one fixed observation sequence and base model per dimension/dtype;
  `P=50` derivative directions and parameter values are an exact prefix/subset
  of `P=150`; batch rows for `B=1/4` are selected from the same predeclared
  `B=16` proposal cloud rather than regenerated with batch-dependent offsets.
  Artifacts record cross-arm base-model/observation hashes, derivative-prefix
  checks, and proposal-row identities. If an axis necessarily changes the
  computed target, comparisons are restricted to within-cell analytical versus
  autodiff results and must not attribute cross-cell timing changes solely to
  `P` or `B`.
- Before timing promotion, a row-independence test must establish that the
  vector likelihood Jacobian is block diagonal with respect to parameter rows,
  and that gradient-of-vector semantics returns the intended `[B,P]` scores.

## Required Run Manifest

Every serious CPU/GPU run records: git commit, dirty tracked/untracked manifest
or digest, hashes of touched source/tests/harness, exact command, interpreter
and conda environment, Python/TensorFlow/TFP/XLA versions, CPU/GPU identity and
visibility, requested/effective affinity and thread settings, dtype, JIT, TF32,
XLA flags, fixture/data hash, random seeds or deterministic declaration, wall
time, output/log paths, plan/result paths, and for GPU the exact trust basis
`owner_designated_managed_session_visible_gpu_trusted` when applicable.
The manifest also records base-model and observation hashes, derivative-basis
hash/prefix status, proposal-cloud hash, and selected proposal-row identities.

## Phase-Local Implementation Repair Loop

For any fixable local check or execution failure:

1. Localize the smallest failing case and failing stage.
2. Classify it as common harness/artifact, common math/fixture, CPU backend,
   GPU backend, or current candidate/method failure.
3. State whether it is a promotion veto, lane-local veto, repair trigger, or
   true continuation veto.
4. Patch only within the current phase write set.
5. Rerun the smallest focused checks, then the phase gate if focused checks pass.
6. Refresh the phase result and next-subplan implications.
7. Continue while an in-scope repair can answer the phase question; stop only
   on a declared continuation veto or new human authority boundary.

The five-round limit applies only to the same material review blocker. It does
not turn an ordinary fixable implementation failure into a stop condition.

## Authoritative Phase Handoff

No next phase launches until the current phase has:

1. run all required local checks;
2. written its phase result/close record;
3. refreshed the next dedicated subplan with actual inherited evidence, exact
   commands, artifacts, and numeric provenance;
4. written a bounded review artifact and ledger entry; and
5. converged on review for consistency, correctness, feasibility, artifact
   coverage, and boundary safety.

## Review And Repair Protocol

Material plan/result review uses
`/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh` with `--model opus
--effort max`. Bundles name the smallest exact path that answers the gate.

1. Run the review gate with its tiny probe.
2. If the probe returns `OK` but material review times out or has no verdict,
   Claude is alive: shrink/redesign the prompt and retry.
3. If the probe establishes transport failure or repeated probe timeout, replace
   that review with a fresh bounded Codex read-only review and label it weaker.
4. On `REVISE`, patch the same artifact visibly and rerun focused local checks.
5. Stop after five rounds for the same material blocker and write a blocker result.

Claude is advisory only and cannot authorize runtime, model-file, funding,
product, release, default-policy, GPU-trust, or scientific-claim boundaries.

## Program Stop Conditions

- New package/network/credential/model-file/default-policy authority is required.
- Work would overwrite unrelated dirty changes rather than integrating surgically.
- Required artifact or provenance cannot be preserved.
- Local correctness or parity remains broken after the bounded repair loop.
- Claude/Codex review does not converge after five rounds for the same blocker.
- A long or comparison run is proposed before its current subplan survives skeptical audit and review.
