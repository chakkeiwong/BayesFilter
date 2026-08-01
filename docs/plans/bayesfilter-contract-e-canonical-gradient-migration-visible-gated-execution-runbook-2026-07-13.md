# Visible Gated Execution Runbook: Contract E Canonical LEDH Gradient Migration

Date: 2026-07-13

Status: `REVIEWED_VISIBLE_EXECUTION_ACTIVE`

## Role Contract

Codex in the current conversation is supervisor and executor. Claude Opus is a
read-only reviewer only and cannot authorize scientific claims or boundary
crossings.

This is visible, recoverable execution. Do not use `codex exec`,
`overnight_gated_launch.sh`, `setsid`, `nohup`, detached `tmux`, background phase
runners, or copied-workspace execution. A detached campaign requires a separate
plan and explicit owner approval.

## Program Artifacts

- Master program:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-master-program-2026-07-13.md`
- Execution ledger:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-visible-execution-ledger-2026-07-13.md`
- Stop handoff:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-visible-stop-handoff-2026-07-13.md`
- Review logs:
  `.claude_reviews/contract-e-canonical-gradient-migration-*/`
- Quiet logs:
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/`

## Phase Index

| Phase | Name | Subplan | Result |
| --- | --- | --- | --- |
| 0 | Policy, route freeze, immediate revocation | `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase0-policy-route-freeze-subplan-2026-07-13.md` | `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase0-policy-route-freeze-result-2026-07-13.md` |
| 1 | Normative Contract E mathematics | Created at Phase 0 close | Checked specification/design-freeze result |
| 2 | Schema v2 and canonical factory | Created at Phase 1 close | Schema/factory result |
| 3 | Cloud-level Contract E module | Created at Phase 2 close | Dense/cloud total-gradient parity result |
| 4 | Streaming composition and feasibility | Created at Phase 3 close | Dense/stream parity plus production preflight result |
| 5 | Canonical value/JVP/FD graph | Created at Phase 4 close | Canonical graph result |
| 6 | Historical route cleanup | Created at Phase 5 close | Entry-point cleanup result |
| 7 | LaTeX reconciliation | Created at Phase 6 close | Documentation/build result |
| 8 | LGSSM statistical design and oracle ladder | Created at Phase 7 close | Paired material LGSSM result |
| 9 | Nonlinear migration, regeneration, integrity audit | Created at Phase 8 close | Per-row results plus leaderboard/audit result |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can Contract E--Chol become the only canonical streaming LEDH value/gradient route and pass the relevant engineering, numerical, and scientific gates? |
| Baseline/comparator | Exact Kalman for LGSSM, same-callable value for FD, dense Contract E reference for small parity, raw reset as permanently historical negative control. |
| Primary criterion | Every phase satisfies its exact evidence contract; the final leaderboard contains only canonical Contract E LEDH contributions; the separate integrity audit passes. |
| Veto diagnostics | Raw route admission or forged identity, partial derivative mislabeled total, false same-scalar identity, covariance failure, invalid chart, dense or retained-quadratic production graph, nonfinite output, wrong hardware/provenance, unsupported threshold or claim, missing artifact, or corrupted dependency closure. |
| Explanatory diagnostics | Runtime, memory, residuals, ridge, conditioning, prefix behavior, and descriptive seed variation. |
| Not concluded | Nonlinear exactness, HMC/posterior correctness, superiority, or release completion before explicit gates. |
| Artifacts | Master/subplans/results, ledger, stop handoff, logs, manifests, tests, review summaries, numerical JSON, leaderboard, and post-run audit. |

## Default And Assumption Audit

The master program's audit is binding. Before each phase, recheck stale context,
wrong comparator, proxy promotion, thresholds, dirty-file overlap, GPU policy,
commands/artifacts, and whether a candidate failure is a repair trigger rather
than a continuation veto.

## Quiet Visible Execution

For commands with potentially large output:

1. predeclare log and structured artifact paths;
2. redirect full stdout/stderr to the log without `tee`;
3. use a bounded `timeout` appropriate to the phase;
4. inspect exit status, structured pass/fail fields, and at most the final 40 log
   lines on failure;
5. preserve logs and reference them in the result;
6. treat excessive session output as an execution defect, not a reason to hide
   failures.

GPU/CUDA/TensorFlow/XLA commands run with trusted/escalated permissions under
`AGENTS.md`. Deliberate CPU reference checks set `CUDA_VISIBLE_DEVICES=-1`
before framework import and record that choice.

## Visible State Machine

For each phase:

1. `PRECHECK`
   - read the phase subplan;
   - verify entry conditions and owned-file status;
   - restate the evidence contract;
   - append a ledger record;
   - run the skeptical audit.
2. `EXECUTE_MINIMAL`
   - run the smallest discriminating implementation/check;
   - preserve unrelated changes;
   - write structured artifacts and quiet logs.
3. `ASSESS_GATE`
   - compare with primary criteria, promotion vetoes, repair triggers, and
     continuation vetoes;
   - separate candidate failure from experiment invalidity;
   - write/update the result.
4. `REVIEW_IF_MATERIAL`
   - use one bounded exact-path review when it can change a material decision;
   - treat review as advisory;
   - preserve status, verdict, run directory, and repair response.
5. `REPAIR_LOOP`
   - write a same-phase repair note for a fixable blocker;
   - patch visibly and rerun focused checks;
   - stop review looping when it no longer changes a material decision;
   - limit the same blocker to five materially distinct rounds.
6. `ADVANCE_OR_STOP`
   - advance after the phase gate passes;
   - continue to a planned repair when the current candidate fails but the
     experiment remains valid;
   - stop only on a true continuation veto or campaign budget exhaustion.

## Binding Phase-Order Gates

- Phase 0 immediately revokes current raw/v1 admission; cleanup is not deferred.
- Phase 1 freezes checked mathematics, numerical vetoes, FD safeguards, and the
  paired statistical design before implementation or promotion data.
- Phase 2 identity is issued only by a non-overridable canonical factory; caller
  labels are never execution evidence.
- Phase 3 dense-reference/cloud parity includes total gradients and the separate
  direct-moment/direct-weight/transport adjoint components.
- Phase 4 dense-composition/stream parity precedes the production preflight; the
  preflight includes analytic complexity, retained-graph inspection, trusted-GPU
  peak memory, XLA runtime, and chunk invariance.
- Phase 8 may not reduce seeds, relax equivalence margins, or change near-zero
  handling after results are observed.

## Claude Review Protocol

Use trusted/escalated:

```bash
bash /home/chakwong/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/chakwong/BayesFilter \
  --review-name <bounded-name> \
  --bundle /home/chakwong/BayesFilter/docs/reviews/<one-bounded-bundle>.md \
  --probe-timeout 90 \
  --timeout-seconds 180 \
  --max-retries 1 \
  --allow-bounded-fallback
```

Initial review surface is one exact bundle path and one precise question. If the
probe succeeds but review stalls, narrow/redesign the bundle. The iteration-1
Claude attempt was blocked by the platform external-disclosure boundary; do not
retry or work around that decision. Record it and use a fresh Codex read-only
review for material gates when local evidence can safely carry the gate.

## Plain Scientific Language Gate

Before accepting a result, ensure it states:

- the claimed target;
- the quantity actually computed;
- whether they are equal, related, different, unsupported, or not checked;
- the derivation/source/artifact supporting the classification;
- what remains unproved.

Use `correct`, `wrong relative to the stated target`, `unsupported`, `not
checked`, or `heuristic only` directly. Do not use soft terminology to hide a
target mismatch.

## Ledger Entry Template

```markdown
### <timestamp> - Phase <N> - <STATE>

Evidence contract:
- Question:
- Comparator:
- Primary criterion:
- Promotion vetoes:
- Continuation vetoes:
- Nonclaims:

Actions:
- <edits/commands/reviews>

Artifacts:
- <paths>

Gate status:
- <PASSED/REPAIR_REQUIRED/BLOCKED/IN_PROGRESS>

Next action:
- <next visible step>
```

## Human-Required Stop Conditions

Stop when continuing requires a new project-direction choice, an unjustified
scientific threshold, package/network/environment setup outside approved Claude
review, credentials, destructive action, detached execution, modification of
unrelated dirty work, untrusted GPU interpretation, or a material boundary that
focused investigation cannot resolve. The anticipated Kalman-gradient
equivalence-margin decision is required before material Phase 8 if Phase 0--7
cannot justify it.

## Final Handoff

At completion or stop, record final phase/status, result artifacts, review trail,
commands actually run, unresolved blockers, direct scientific classifications,
nonclaims, and safest next human decision. Never mark the leaderboard complete
or released before the separate post-run integrity audit passes.
