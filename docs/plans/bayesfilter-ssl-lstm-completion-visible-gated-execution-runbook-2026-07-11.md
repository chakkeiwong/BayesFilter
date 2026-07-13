# BayesFilter SSL-LSTM Completion Visible Gated Overnight-Style Runbook

Date: 2026-07-11

Status: `ACTIVE_PHASE_A3_HARNESS_REPAIR_BEFORE_EVIDENCE`

## Active A3 Gate

A2 closure is complete. The A3 oracle/statistics implementation has passed its
focused checks and bounded Codex-substitute implementation review. The current
state is `RUN_LOCAL_CHECKS` for the A3 artifact generator and independent
verifier.

The next transition is permitted only when all of the following hold:

1. the generator and verifier bind the accepted oracle/statistics hashes;
2. CPU generation materializes two independent arm banks in the artifact;
3. GPU generation and fresh verification consume those exact persisted banks,
   not floating-normal regeneration from seed metadata;
4. decision rows are produced only by authenticated constructors in the same
   process and are independently recomputed by the verifier;
5. coverage, controlled alternatives, joint alpha, HLO/device placement,
   parity, source/fixture bindings, and trace boundaries are computed rather
   than asserted by placeholders; and
6. a fresh bounded read-only harness review returns `VERDICT: AGREE`.

Until then, the A3 CPU and trusted GPU/XLA evidence commands remain blocked.

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude Opus at max effort is a read-only advisory reviewer. Claude cannot edit
files, run experiments, launch agents, execute phases, or authorize human,
runtime, model-file, funding, product, release, default-policy, GPU-trust, or
scientific-claim boundaries.

This runbook provides overnight-style gated sequencing while remaining visible
and recoverable in the current conversation. It must not use `codex exec`,
`overnight_gated_launch.sh`, `setsid`, `nohup`, detached `tmux`, background
phase runners, copied workspaces, or nested supervisors. A detached overnight
launch would be a separate execution mode and requires a separately reviewed
handoff; it is not silently substituted for this runbook.

## Quiet Visible Execution

- Predeclare log and structured artifact paths in the active subplan/ledger.
- Redirect large TensorFlow, XLA, sampler, training, and review output to logs.
- Return bounded status: exit code, structured artifact paths, decisive fields,
  and at most 40 failure-tail lines.
- Poll bounded process/status metadata when live monitoring is required.
- Preserve full logs. Quiet execution must not hide failures.
- Treat timeout, interruption, and missing artifact as explicit stage results.

## Program

- Umbrella roadmap:
  `docs/plans/bayesfilter-ssl-lstm-completion-roadmap-2026-07-11.md`
- Governing scalar program:
  `docs/plans/bayesfilter-scalar-ssl-lstm-predictive-equivalence-master-program-2026-07-11.md`
- Reset memo:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-reset-memo-2026-07-10.md`
- Execution ledger:
  `docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md`
- Approval ledger:
  `docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md`
- Stop handoff:
  `docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md`
- Artifact root:
  `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/`

## Phase Index

| Phase | Name | Subplan | Required result |
| --- | --- | --- | --- |
| A0 | Governance, target, artifact lock | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-result-2026-07-11.md` |
| A1 | Reusable masked posterior target | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md` |
| A2 | Terminal-state and forecast API | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md` |
| A3 | Forecast oracle and statistics | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md` |
| A4 | Calibration and design freeze | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-result-2026-07-11.md` |
| A5 | Fresh ordinary-HMC baseline | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a5-ordinary-hmc-baseline-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a5-ordinary-hmc-baseline-result-2026-07-11.md` |
| A6 | Dense-IAF training | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a6-dense-iaf-training-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a6-dense-iaf-training-result-2026-07-11.md` |
| A7 | Exact-corrected NeuTra-HMC | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a7-exact-neutra-hmc-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a7-exact-neutra-hmc-result-2026-07-11.md` |
| A8 | Blinded predictive confirmation/audit | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a8-predictive-confirmation-audit-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a8-predictive-confirmation-audit-result-2026-07-11.md` |
| A9 | Repeated synthetic calibration | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a9-synthetic-generative-calibration-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a9-synthetic-generative-calibration-result-2026-07-11.md` |
| A10 | Scalar productization/closeout | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a10-scalar-productization-closeout-subplan-2026-07-11.md` | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a10-scalar-productization-closeout-result-2026-07-11.md` |

A0 through A2 have completed their bounded engineering phases. The repaired A2
result and refreshed A3 subplan received bounded `CODEX_SUBSTITUTE_REVIEW`
`VERDICT: AGREE`, explicitly weaker than Claude. Under the user's narrow
"fix that and continue" authority, all new A2 closure traces use
`strace -f -qq -yy -s 65535 -e trace=%file`. A3 implementation remains blocked
until the regenerated A2 post-result write ledger, checkpoint, closure,
fresh-process closure verification, and exact terminal trace audit pass and are
recorded. Track B and Track C require separate child programs after their
roadmap entry gates.

## Global Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter finish a typed GPU/XLA-first scalar SSL-LSTM vertical slice and distinguish engineering correctness, sampler validity, predictive-functional equivalence, synthetic calibration, and application adequacy? |
| Exact computational baseline | Fresh independently tuned four-chain ordinary MAP-local HMC on the A0-locked target; old Phase 2V is context only |
| Comparator | Separately trained frozen plain dense-IAF transport followed by independently tuned exact-Jacobian-corrected four-chain NeuTra-HMC on the identical target |
| Primary M1 criterion | A0-A5, A9, and applicable A10 gates pass; no NeuTra replication claim |
| Primary M2 criterion | M1 passes and A6-A8 pass frozen confirmation plus fresh-seed audit |
| Vetoes | Target/data/mask mismatch; failed oracle; nonfinite/invalid covariance; missing or positive native divergences; sampler validity failure; transform/Jacobian failure; invalid uncertainty; changed frozen design; data leakage; missing/corrupt artifact; boundary violation |
| Explanatory only | Loss, runtime, acceptance inside a non-veto range, parameter summaries, higher moments, extreme quantiles, local geometry, and descriptive differences without uncertainty support |
| Not concluded | Parameter-posterior equality/correctness, exact nonlinear likelihood, identification, NeuTra or sampler superiority, real-data adequacy, dimensional scalability, public/default readiness, or Zhao-Cui source faithfulness |
| Preservation | Phase JSON/Markdown/log artifacts, ledger, review status files, and reset/handoff records |

## Separate Evidence Ledgers

Every phase result must state `passed`, `failed`, `blocked`, `not applicable`,
or `not assessed` independently for:

1. engineering correctness;
2. numerical/sampler validity;
3. computational predictive equivalence;
4. synthetic generative calibration;
5. empirical model adequacy.

No ledger may borrow a pass from another ledger.

## Default And Assumption Audit

The roadmap numeric-provenance table and the active subplan are binding. Every
material threshold, timeout, cap, seed, architecture, count, margin, tolerance,
or budget must be classified as measured, derived, inherited, convenience, or
reviewed default before use. Missing provenance makes a number a hypothesis,
not a fact.

## Skeptical Audit Before Every Phase

Codex records in the ledger whether the phase has:

- the correct baseline and comparator;
- no proxy metric promoted to a pass criterion;
- explicit promotion, continuation-veto, repair, and stop rules;
- fair tuning/budget/data/seed treatment;
- explicit hidden assumptions and numeric provenance;
- current source, artifact, and environment identities;
- commands whose outputs answer the phase question;
- a distinction between candidate failure and invalid harness/target/math/data.

If any material item fails, repair the plan or write a blocker before runtime.

## Visible State Machine

1. `PRECHECK`
   Read the current subplan, predecessor result, runbook, ledger, approvals,
   source fingerprints, and evidence contract. Verify authority and entry gates.
2. `EXECUTE_MINIMAL`
   Make the smallest in-scope edit or run the smallest diagnostic that answers
   the phase question. Preserve unrelated dirty work.
3. `RUN_LOCAL_CHECKS`
   Run the exact required checks. A smoke/import/compile check remains its
   declared evidence class.
4. `ASSESS_GATE`
   Apply hard vetoes before descriptive metrics. State whether failure affects
   implementation, harness, target, data, math, environment, or only candidate.
5. `WRITE_RESULT`
   Write the required result, manifests, decision/inference tables, post-run
   red team, and nonclaims before advancement.
6. `DRAFT_OR_REFRESH_NEXT`
   Create the next subplan from actual inherited evidence, with unset future
   numbers left unset.
7. `PASS_READ_ONLY_REVIEW`
   Review the current material result and next subplan independently.
8. `REPAIR_LOOP`
   Patch the same artifact visibly, rerun focused checks, and rereview, up to
   five substantive rounds for one blocker.
9. `ADVANCE_OR_STOP`
   Advance only when every conjunctive handoff condition passes. Otherwise
   update the stop handoff and request only the missing human authority.

## Claude Review Gate

Material reviews use:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/ubuntu/python/BayesFilter \
  --review-name <stable-name> \
  --bundle /home/ubuntu/python/BayesFilter/docs/reviews/<one-path-bundle>.md \
  --model opus \
  --effort max \
  --probe-effort low \
  --probe-timeout 90 \
  --timeout-seconds 180 \
  --max-retries 1
```

Initial review shape:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line: <one exact path>.
Do not edit, run commands, launch agents, or review the whole repo.
Question: <one narrow question>. End with VERDICT: AGREE or VERDICT: REVISE.
```

- Do not use automatic bounded fallback as a phase pass.
- Probe `OK` plus material timeout/no verdict means shrink or redesign prompt.
- Confirmed probe/transport failure permits fresh `CODEX_SUBSTITUTE_REVIEW`.
- A trusted-execution policy rejection that persists after explicit informed
  user approval also makes external Claude review unavailable. Do not retry or
  route around it; use a fresh native Codex read-only reviewer as the materially
  safer substitute and record that Claude liveness was not tested.
- Substitute agreement is weaker and is never labeled Claude convergence.
- Record `REVIEW_STATUS`, `VERDICT`, `RUN_DIR`, `SUMMARY_JSON`, review type,
  findings, repairs, and round count.
- Stop after five substantive rounds for the same unresolved material blocker.

## Repair Protocol

```text
PRECHECK
  -> minimal execution
  -> focused checks
  -> result/close record
  -> next subplan
  -> bounded review
       AGREE  -> advance if local and boundary gates also pass
       REVISE -> patch same artifact -> focused checks -> rereview (max 5)
       probe OK + no material verdict -> shrink/redesign -> rereview
       probe/transport down -> fresh CODEX_SUBSTITUTE_REVIEW
       external-review policy blocked -> fresh CODEX_SUBSTITUTE_REVIEW
  -> stop only on a declared continuation veto or human-required boundary
```

A failed candidate is not a program stop when a later reviewed phase is
designed to repair exactly that failure. A continuation veto concerns invalid
harness, target, data, math, required environment, corrupt/missing artifact, or
another explicitly stated invalidating condition.

## Runtime Boundaries

- A0 is documentation/inventory plus one deliberate CPU-hidden deterministic
  reference replay.
- A1-A3 may use CPU-hidden reference/debug checks, but default algorithmic
  execution and serious evidence use TensorFlow/TFP with XLA enabled.
- GPU/CUDA checks and runs require trusted/elevated execution and provenance.
- NeuTra training is GPU/XLA by owner directive. External sample generation is
  a separately manifested multicore CPU lane.
- HMC, calibration, sweeps, comparisons, and default/product decisions require
  their reviewed phase subplans and pre-run evidence contracts.
- No review verdict can authorize runtime outside these boundaries.

## Approval Protocol

The approval ledger is binding. Existing user authorization covers creation
and visible execution of this reviewed program, but does not waive tool-level
trusted execution, phase entry gates, destructive-action prohibitions, commit
authorization, or scientific/product/default decisions. Request the narrowest
command approval at the phase where it is needed; do not request a broad shell,
Python, Claude, or GPU prefix.

## Human-Required Stop Conditions

- A project direction, target migration, model-file edit, default/public API,
  release, funding/cost, or scientific claim not already decided by a reviewed
  contract.
- Package installation, network fetch, credentials, destructive action, or
  modification of unrelated user work.
- Pass/fail criteria changed after confirmation/audit data are opened.
- GPU evidence without trusted provenance.
- Missing current NeuTra predecessor-gate evidence at A6.
- Review non-convergence after five rounds.
- An active subplan leaves a material number, comparator, artifact, or handoff
  placeholder unresolved.

## Final Visible Handoff

On completion or stop, update the stop handoff with final phase, status,
artifacts, review trail, checks/runs, unresolved blockers, nonclaims, source
fingerprints, and the safest exact next action. Do not stage or commit unless
the user separately authorizes it after inspecting the completed write set.
