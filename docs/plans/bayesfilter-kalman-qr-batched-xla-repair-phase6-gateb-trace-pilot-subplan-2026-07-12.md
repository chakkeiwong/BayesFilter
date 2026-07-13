# Phase 6 Gate B Trace Census And Pilot Subplan

Date: 2026-07-12

Status: `R1_RUNTIME_COMMON_INVALIDITY_REPAIR_SUBPLAN_ACTIVE_NO_FRESH_TARGET_AUTHORITY`

Parent Phase 6 subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md`.

Parent SHA-256:
`b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b`.

Runtime repair subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-runtime-harness-repair-subplan-2026-07-12.md`.

The first authorized Gate B launch stopped after one internally successful trace
child because its embedded interpreter token was normalized from the reviewed
`.../bin/python` spelling to `.../bin/python3.13`. The invalid-evidence fallback
then exposed a second schema defect: a truthful zero process return code could
not be represented as `failed:invalid_child_evidence`, and exception recovery
closed the durable running record as `interrupted:supervisor_recovery`. This is
common harness invalidity, not method or CPU-XLA evidence. The original proposal,
attestation, trace, child, journal, budget state, and lease are immutable. No
resume or fresh target launch is authorized by this parent subplan; the dedicated
repair subplan governs preservation, repair, review, and prospective `r2`
reauthorization.

Gate A result:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatea-conformance-repair-result-2026-07-11.md`.

## Review Trail

- Gate A result review round 1: Claude Opus max effort, exact
  `VERDICT: AGREE`, reviewed result SHA-256
  `f16907003e5c98c28b8a0ee1cd9c4f228dc2d7e88e61c9946529ab811377f2c6`.
- Gate B subplan round 1: `VERDICT: REVISE` on subplan SHA-256
  `4403929e2f58d9027b88c21f8840e265a14666a3a7311eb7a0a833723e137bb3`;
  the five findings and visible repairs are preserved in the round-1 review
  record under `docs/reviews/`.
- Gate B subplan round 2: Claude Opus max effort, exact `VERDICT: AGREE` on
  repaired subplan SHA-256
  `bd449a78fb19c06e90da00892e814eecfa62623c6bcf6f08f2befca29813c332`.
  A healthy probe followed by a stalled broad prompt required a narrower
  authorization-invariant prompt; the probe carries no review authority.

The status/review-trail refresh above changes governance text only. It does not
change the reviewed objective, roster, commands, budget, predicate, artifacts,
evidence contract, forbidden actions, handoff, or stop conditions. A final
bounded consistency check of this refreshed exact path is required before
proposal construction.

## Phase Objective

Under one reviewed 3045-second monotonic Gate B authority, first produce the
complete 36-record target trace census at
`dimension in {10,20,30}`, `P in {50,150}`, `B in {1,4,16}`, `T=120`,
`float32`, two primary methods, GPU hidden, and one requested CPU thread. Run
the two method records that constitute the single paired
`dimension=10,P=50,B=1` CPU-XLA pilot cell only if the final trace evaluator
establishes common structural validity with zero rejected typed GraphDef
differences.

This phase answers an engineering localization question. It does not rank
methods, certify CPU production viability, or establish GPU, HMC, posterior,
default, production, or scientific validity.

## Entry Conditions Inherited From Gate A

All conditions are conjunctive:

- Gate A local checks pass exactly as recorded in the Gate A result;
- bounded Claude Opus max-effort review of the Gate A result returns exact
  `VERDICT: AGREE` within five material rounds;
- the three protected algorithm hashes and parent Phase 6 subplan hash still
  match the Gate A result;
- the opening hash ledger remains a regular non-symlink file at SHA-256
  `9261e0c560ede29dc6893e0ffe3769cd762b38f3dd651af6dfcfa2f90dce1911`;
- its known scope remains two headers plus 144 entries, with 36 historical JSON
  counterparts omitted and no complete-inventory claim;
- no prior Gate B target artifact, budget state, or active supervisor conflicts
  with the exact new authority; any present bytes must be validated and resumed
  or produce a blocker, never deleted or silently overwritten;
- this subplan receives a separate bounded Claude Opus max-effort read-only
  `VERDICT: AGREE` before proposal construction;
- Codex remains supervisor and executor; Claude is read-only reviewer only.

## Required Artifacts

### Preflight And Authority

- exact `--help` capture and check record in the Gate B result;
- import-discovery artifact:
  `/tmp/kalman_qr_phase6_cpu_xla/import_discovery.json`;
- immutable proposal:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json`;
- proposal-specific review record:
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round1-2026-07-11.md`;
  if material review repair is required, subsequent preserved records use the
  exact pattern
  `docs/reviews/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-budget-review-round<N>-2026-07-11.md`
  for `N=2..5`, and the attestation binds only the final agreeing record;
- detached attestation:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json`;
- budget state/lease under
  `/tmp/kalman_qr_phase6_cpu_xla/budget_state/gate_b-<authority_id>.json` and
  its `.lease` file.

### Runtime Evidence

- trace census:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json`;
- pilot ledger:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json`;
- losslessly embedded bounded child artifact, sidecar, journal, dependency
  manifests, process capture, and GraphDef evidence in those ledgers;
- Gate B result:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-trace-pilot-result-2026-07-12.md`;
- Gate B blocker result, only if a declared stop condition fires:
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-trace-pilot-blocker-result-2026-07-12.md`;
- refreshed Gate C scalar/remaining subplan, written before any Gate C action;
- bounded Claude logs under `.claude_reviews/`.

The `2026-07-11` proposal, review, attestation, trace, and pilot filenames are
inherited verbatim from the immutable parent Phase 6 artifact and command
contract. Their suffix is a lane identifier, not a claim about creation time;
freshness is determined only by exact path/digest/authority validation. The
Gate B result/blocker uses the current close-record date `2026-07-12`.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Do both repaired true-batched methods have common-valid target trace structure, and if so, what honest outcomes occur for the smallest reviewed CPU-XLA pilot cell? |
| Candidate/mechanism | Repaired batched Kalman QR analytical and autodiff score paths under exact v4 measurement and Phase 6 trace contracts. |
| Expected failure mode | Structural graph specialization/unrolling, provenance/evidence invalidity, or CPU-XLA compile/codegen/runtime failure under the reviewed cap. |
| Promotion criterion | Trace census is final and common-valid; pilot viability is only a passed screen when both exact pilot records pass all validity gates. |
| Promotion veto | Common invalidity, rejected typed GraphDef diff, non-finite/dtype/shape/parity failure, stale/corrupt artifact, wrong provenance, or authority violation. |
| Continuation veto | Invalid harness/authority/evidence, corrupted artifact, missing required diagnostics, unsafe process ownership, or exhausted/ambiguous budget state. A valid trace rejection prunes pilot and closes Gate B; a valid pilot candidate failure does not invalidate the research direction. |
| Repair trigger | Any common structural/provenance/evidence defect. Write a blocker/result and repair prospectively before any new target launch. |
| Explanatory diagnostics | GraphDef nodes/bytes, trace duration, first/warm duration, child wall time, and error tails. |
| Must not conclude | No superiority, CPU production target, GPU readiness, HMC/posterior/default/production/scientific validity, or universal impossibility from a capped failure. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering/scientific question | Exact question in the research intent ledger above. |
| Exact baseline/comparator | Historical retained static-unrolling/CPU-XLA failure evidence named in the parent Phase 6 subplan, plus Phase 4/5 artifacts bound by the proposal. The two primary methods use the same target identities and validity contract. |
| Primary pass/fail criterion | Complete final trace ledger and evaluator; zero common-invalidity or rejected typed-diff records. Only then may the exact two-record pilot run. |
| Promotion veto diagnostics | Common invalidity, structural rejection, numerical/dtype/shape/parity failure, stale/corrupt evidence, wrong source/runtime/schedule/provenance, or boundary violation. |
| Continuation veto diagnostics | Invalid authority, ledger, process ownership, required artifact, or missing diagnostics; budget ambiguity; unsafe cleanup. |
| Explanatory only | Graph sizes/counts/times, single-run first/warm durations, return-code/error-tail summaries after validity classification. |
| Not concluded even if passed | No method ranking, speed superiority, CPU/GPU scalability, HMC/posterior/default/production/scientific claim. |
| Preserved result | Trace/pilot ledgers, proposal/review/attestation/budget artifacts, Gate B result, run manifest, decision table, inference-status table, and next subplan. |

## Skeptical Pre-Execution Audit

Status: `PASS_FOR_NO_TARGET_PREFLIGHT_ONLY_RUNTIME_PENDING_REVIEW_AND_ATTESTATION`.

- Wrong baseline: avoided by binding the historical failure artifact class and
  exact Phase 4/5 evidence, not a weak or newly invented comparator.
- Proxy promotion: trace stability is a prerequisite for pilot runtime, not a
  numerical or performance promotion. Pilot survival is a screen, not proof of
  superiority or production readiness.
- Missing stop conditions: invalid authority/evidence/process ownership and
  common structural invalidity are explicit continuation vetoes; valid
  candidate failure is separated from research-direction rejection.
- Unfair comparison: both primary methods use exact matched target identities,
  fresh sequential children, one requested CPU thread, the same v4 measurement
  contract, and method-local outcomes.
- Hidden assumptions: proposal binds source/runtime/config/fixture/schedule,
  exact commands, import closure, 60-second child execution, 70-second child
  lifecycle, 160-second paired-cell cap, and 3045-second immutable gate budget.
- Stale context: all relevant hashes must be rechecked immediately before
  proposal construction and again before runtime attestation/launch.
- Environment mismatch: deliberate CPU-only target execution sets
  `CUDA_VISIBLE_DEVICES=-1`, one requested TensorFlow/OMP thread, CPU device,
  and XLA on only for the pilot. This is an explicit Phase 6 diagnostic lane,
  not the repository default production target.
- Artifact fitness: trace GraphDefs, transition ledgers, child bytes, process
  evidence, and strict evaluators directly answer the gate. Timing summaries do
  not substitute for validity.

Runtime audit remains pending until the exact proposal, proposal-specific
review record, detached attestation, clean preflight, and no-worker checks all
pass. Do not launch before recording that final audit as `PASS`.

## Preflight And Review Sequence

1. Recheck protected hashes, opening-ledger parse/digest, scoped whitespace,
   compile, focused tests, and exact no-worker/no-stale-artifact state.
2. Capture `--help` and verify that implemented flags/defaults match the closed
   Gate B command below.
3. Run only the closed proposal constructor:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-prepare-proposal gate_b \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json
```

This command performs import discovery and proposal construction only. It must
not trace or execute a target filter/XLA/scalar workload.

4. Strictly validate the proposal and record its exact SHA-256 and authority ID.
5. Ask Claude Opus at max effort to review exactly the proposal path. The
   committed review record must declare exact absolute `PROPOSAL_PATH`,
   `PROPOSAL_SHA256`, `PLAN_PATH`, and `PLAN_SHA256`, end with exact
   `VERDICT: AGREE`, and contain no runtime authorization claim.
6. Create the detached attestation with exact schema
   `bayesfilter.kalman_qr_batched_xla_repair.phase6.budget_attestation.v1`, gate
   `gate_b`, proposal/plan/review path-digest records, exact proposal authority
   ID, verdict `AGREE`, strength `claude_opus_max`, and UTC timestamp.
7. Run `validate_phase6_runtime_authority(...)` locally. Record all returned
   checks. Any failure returns to the repair loop; do not hand-edit around it.
8. Repeat protected hashes and no-worker checks; record final skeptical runtime
   audit. Only a recorded `PASS` crosses into execution.

## Exact Runtime Command

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  timeout --signal=TERM --kill-after=45s 3000s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-pilot \
  --dimensions 10 20 30 --parameter-counts 50 150 \
  --batch-sizes 1 4 16 --timesteps 120 \
  --dtype float32 --device cpu --cpu-threads 1 --jit-compile \
  --trace-child-timeout-seconds 60 --xla-child-timeout-seconds 60 \
  --xla-cell-timeout-seconds 160 \
  --budget-contract docs/benchmarks/kalman_qr_batched_xla_repair_phase6_pilot_budget_2026-07-11.json \
  --budget-attestation docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_budget_attestation_2026-07-11.json \
  --trace-output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_trace_census_2026-07-11.json \
  --output-json docs/benchmarks/kalman_qr_batched_xla_repair_phase6_cpu_xla_pilot_2026-07-11.json
```

The outer 3000-second TERM deadline plus 45-second KILL grace is bounded by the
proposal's immutable 3045-second monotonic authority. The supervisor enforces
60-second child execution plus five-second TERM and five-second KILL/reap,
70-second lifecycle caps, and one shared 160-second cap for the two method
records constituting the paired pilot cell.

## Runtime Interpretation And Repair Loop

- Execute all 36 trace children first and persist/reparse after every update.
- Evaluate the final trace artifact before constructing pilot bindings.
- The operative launch predicate is conjunctive: the strict evaluator must
  report `trace_common_valid=true`, all six cohort comparisons must report
  `passed=true`, and every cohort's `rejected_differences` must be empty. The
  evaluator definition makes the latter two conditions part of
  `trace_common_valid`; they are repeated here to prevent a weaker reading.
- If any part of that predicate is false, write both exact pilot records as
  `not_launched:trace_gate_not_passed`, close the pilot ledger, write the Gate B
  result, and do not launch XLA.
- Only if the full predicate is true, execute the two exact method records for
  the single paired `dimension=10,P=50,B=1` pilot cell in fresh sequential
  children under their one shared 160-second cap.
- A valid method-local/CPU-backend timeout, crash, or failure rejects that pilot
  candidate under the cap but does not by itself reject the research direction.
- Common invalidity stops later launches and closes remaining records with exact
  common-invalidity pruning.
- Global budget exhaustion closes remaining records as
  `not_launched:global_budget_exhausted`; it never resets or extends the budget.
- On interruption, resume only through the durable ledger, lease, budget, and
  process-ownership rules. Never delete or overwrite stale evidence.
- For every fixable defect, patch the same active artifact visibly, rerun the
  smallest affected checks, then rerun the required closure checks. Claude
  review may be repeated at max effort only for material issues, at most five
  rounds for the same blocker.

## Required Checks, Tests, And Reviews

Before target execution:

- exact compilation and the consolidated GPU-hidden Gate A suite from the Gate
  A result;
- exact proposal validation and detached-attestation validation;
- scoped `git diff --check` and trailing-whitespace scan;
- protected algorithm, parent-plan, and opening-ledger hashes;
- exact no-worker and stale-artifact inspection;
- bounded Claude review of this subplan and then of the exact proposal.

After runtime:

- strict JSON reparse and final ledger checks for trace and pilot;
- `evaluate_phase6_trace_census` recomputation from raw evidence;
- exact source/runtime/schedule/authority/provenance reconciliation;
- no pending/running records and no surviving process group;
- proposal, attestation, parent plan, and protected hashes unchanged;
- Gate B result with run manifest, decision table, inference-status table,
  candidate-vs-direction distinction, post-run red-team note, and nonclaims;
- draft/refresh and bounded review of the Gate C subplan.

## Forbidden Claims And Actions

- Do not launch before both bounded reviews, exact proposal, and valid detached
  attestation exist.
- Do not edit `bayesfilter/linear/*.py`, the parent Phase 6 subplan, Phase 4/5
  evidence, proposal bytes after review, review bytes after attestation, or
  unrelated/other-lane files.
- Do not run Gate C scalar references, remaining lattice, GPU, requested CPU
  threads 4/16, HLO dumps, Phase 7, or comparison benchmarks.
- Do not increase timeouts, caps, roster, tolerances, or budget after observing
  target evidence.
- Do not use trace size, node counts, compile survival, a single warm call, or a
  passed hard screen to rank methods.
- Do not signal a dead-leader/live-group process identity or any unverified PID.
- Do not classify malformed child evidence as a method/backend result.
- Claude is read-only and cannot authorize runtime, human, model-file, funding,
  product/default-policy, release, or scientific-claim boundaries.

## Exact Next-Phase Handoff Conditions

All are conjunctive:

- trace and pilot ledgers are final, strictly valid, immutable, and have no
  pending/running records;
- every launched child has exact terminal evidence and every unlaunched child
  has a predeclared reason;
- final trace evaluation cleanly separates common invalidity from a valid trace
  result;
- pilot interpretation separates candidate rejection from research-direction
  rejection and states whether any ranking is statistically supported;
- budget state/lease, process cleanup, source/runtime/schedule/provenance, and
  protected hashes pass their closing checks;
- Gate B result is written and reviewed;
- a dedicated Gate C scalar/remaining subplan is refreshed from Gate B facts
  and reviewed for consistency, correctness, feasibility, artifact coverage,
  and boundary safety;
- Gate C proposal and attestation remain absent until that subplan authorizes
  their no-target construction.

Exact handoff state:
`GATE_B_CLOSED_GATE_C_SUBPLAN_REVIEWED_RUNTIME_STILL_BLOCKED`.

## Stop Conditions

- Gate A result or this subplan does not converge within five material Claude
  rounds for the same blocker.
- Exact proposal or detached attestation validation fails and cannot be repaired
  without changing reviewed runtime scope.
- Protected source/plan/opening-ledger identity drifts unexpectedly.
- A stale or live supervisor/process/budget identity is ambiguous.
- Required target evidence is corrupt, missing, oversized, or cannot be
  strictly reparsed.
- A common structural/provenance/authority invalidity fires.
- Safe cleanup cannot be proved or a process group survives.
- New human authority, package/network change, model-file edit, funding,
  product/default-policy decision, or scientific-claim boundary is required.

## Mandatory Close Sequence

1. Run the required local and artifact checks.
2. Write the Gate B result or blocker record.
3. Draft or refresh the dedicated Gate C subplan.
4. Review that next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety; repair and repeat when material.
5. Advance only when the exact handoff conditions above are satisfied.
