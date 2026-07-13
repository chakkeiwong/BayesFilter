# BayesFilter SSL-LSTM Completion Approval And Boundary Ledger

Date: 2026-07-11

Status: `A3_AUTHORIZED_AFTER_A2_POST_RESULT_CLOSURE`

## 2026-07-13 A3 Runtime Gate Clarification

The A2 post-result closure and terminal trace audit have passed, so existing
user authorization now covers A3 implementation and the reviewed A3 runtime
commands. That runtime authority is nevertheless conjunctively gated: the
artifact generator and independent verifier must first pass static checks and
a fresh bounded read-only review.

The current authorization does not permit running the invalid interim harness.
The repaired harness must use CPU-persisted materialized innovation tensors as
GPU and verifier replay authority, construct decision evidence through the
authenticated statistics APIs, and independently recompute every hard check.

This clarification does not expand authority to HMC, NeuTra, sampler
comparison, A4 calibration, package or network operations, model-file edits,
Git stage/commit/push, public/default/product changes, scientific claims, or
the concurrent HMC/Kalman lane.

Current boundary: `CLAUDE_EXTERNAL_REVIEW_POLICY_UNAVAILABLE_CODEX_SUBSTITUTE_ACTIVE`

Current execution gate: `A2_POST_RESULT_CLOSURE_AND_TERMINAL_TRACE_AUDIT`

## Authority Already Granted

The user's instruction to execute the operational prompt authorizes Codex to:

- refine the existing SSL-LSTM completion roadmap rather than duplicate it;
- create just-in-time phase subplans, results, review bundles, runbook, ledger,
  and handoff artifacts within this repository;
- supervise and execute the visible gated program;
- use Claude Opus at max effort as a bounded read-only reviewer;
- repair fixable plan/code/test/artifact defects inside the reviewed phase write
  set and continue when phase gates pass;
- request narrow trusted-command approvals when required.

This authorization does not override repository governance, scientific gates,
tool-level trusted execution, or the boundaries below.

## Current Approved A0 Actions

| Action | Authority/status |
| --- | --- |
| Read repository and referenced local `claudecodex`/`dsge_hmc` governance | Authorized read-only context gathering |
| Write new A0 planning, review, result, and structured lock artifacts | Authorized within A0 write set |
| One deterministic TensorFlow reference extraction | Authorized only with `CUDA_VISIBLE_DEVICES=-1`; no GPU, HMC, training, benchmark, or scientific use |
| Claude read-only review | User-requested; invoke only through narrow `claude_review_gate.sh`, one exact target path per initial review |
| Scoped local syntax/JSON/document/hash checks | Authorized non-mutating verification |
| Commit/stage/push | Not authorized |
| Detached overnight launch | Not authorized by the visible template; requires separate reviewed execution-mode handoff |

## Anticipated Trusted Execution

Request the narrowest approval only when the reviewed phase reaches it.

| Future action | Earliest phase | Required boundary |
| --- | --- | --- |
| Claude material review | A0 | Trusted call to `/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh`; external disclosure limited to the exact path named by the compact bundle |
| TensorFlow GPU/XLA target canary | A1 | Trusted GPU execution; structured device/JIT/TF32 artifact; no HMC claim |
| TensorFlow GPU/XLA forecast integration | A2/A3 | Trusted GPU execution under reviewed subplan; oracle gates first |
| Calibration runs | A4 | Reviewed evidence contract, budgets/seeds/margins, artifact path; no post-hoc threshold changes |
| Ordinary HMC | A5 | Reviewed sampler plan, trusted GPU/XLA, native divergence telemetry, four-chain manifest |
| NeuTra GPU training | A6 | Current NeuTra predecessor-gate audit, reviewed trainer plan, trusted GPU/XLA, no CPU serious fallback |
| Exact NeuTra-HMC | A7 | Frozen transport/target signatures, independent tuning, trusted GPU/XLA, sampler gates |
| Confirmatory/audit comparison | A8 | Frozen design and unopened audit seeds/artifacts; no post-hoc changes |
| Repeated synthetic calibration | A9 | Reviewed multi-replication statistical plan and uncertainty contract |
| Product/public/default change | A10 or later | Separate explicit human decision; passing this program alone is insufficient |

## Actions Requiring New Human Direction

- Change the locked target, filter family, prior semantics, model equations, or
  four-parameter estimand.
- Modify model files, introduce a materially different architecture/objective,
  or override an unmet NeuTra predecessor gate.
- Install packages, fetch network resources, use credentials, incur a new
  material paid-service/funding commitment, or disclose broader repository
  content externally.
- Use destructive Git/filesystem operations, overwrite unrelated dirty work,
  stage/commit/push, or create a release/PR.
- Change confirmation criteria after results are opened.
- Promote a public API/default policy or make a scientific/model-superiority claim.
- Switch from visible execution to detached overnight execution.

## Forbidden Approval Substitutions

- Claude agreement is not human authorization.
- A tool-prefix approval is not scientific or product approval.
- A successful GPU run is not posterior correctness or default readiness.
- Predictive equivalence is not parameter-posterior correctness or model adequacy.
- User authorization to execute the program is not authorization to commit.
- Silence, timeout, or bounded fallback is not review convergence.

## Escalation Rule

If a phase needs new authority, stop at the smallest safe boundary, update the
visible stop handoff with the exact command/action and reason, and ask for that
specific approval. Do not request broad prefixes such as `bash`, `python`, or
`claude`, and do not route around a rejected escalation.

## 2026-07-11 Claude Disclosure Decision

The first real review-gate invocation was rejected before process creation.
Reason: sending the roadmap contents to Claude would disclose repository
material to an external third-party model service not established as a trusted
destination. No probe ran and no content was sent.

At 2026-07-11T05:02:26+08:00, the user explicitly approved sending exact
one-path BayesFilter planning/result documents to Claude through the bounded
read-only review gate and acknowledged the external-disclosure risk.

The approval is limited to the exact single path named by each compact review
bundle. It does not authorize whole-repository review, broad path packets,
mutations, commands by Claude, source/model-file disclosure unless that file is
separately named as the exact review target, or any runtime, scientific,
product, release, default-policy, funding, commit, or push action.

The identical trusted call was rejected again after this approval. The trusted
execution policy states that the external disclosure remains forbidden even
with approval. This makes Claude review policy-unavailable, not dead. No further
Claude retry or indirect route is allowed. Fresh native Codex read-only reviews
are the materially safer substitute and must be labeled weaker than Claude.

## 2026-07-11 A0 Review-Cap Decision Boundary

The A0 subplan received a material `VERDICT: REVISE` in its fifth substantive
Codex substitute-review round. Its own reviewed rules make that a terminal
blocker. Existing program authority does not authorize a sixth round or a
manual waiver.

Human direction is required for exactly one of these routes:

- authorize one focused patch to reorder the mandatory phase-end sequence and
  one extra bounded native Codex review round; or
- explicitly waive the extra review and accept that exact repair.

The first route is recommended. Neither route authorizes any broader plan
repair, TensorFlow replay, GPU/CUDA work, HMC, NeuTra, predictive experiment,
scientific claim, stage/commit/push, or change outside the existing A0 write set.

## 2026-07-11 Human-Authorized Focused Recovery

The user stated, "I authorize the work." This is recorded as authorization for
the recommended route: apply only the exact mandatory phase-end sequencing
repair preserved in the nonconvergence blocker, then conduct one exceptional
bounded native Codex substitute-review round. It is not recorded as a manual
review waiver.

If that focused review returns `VERDICT: AGREE`, the existing A0 authority
resumes, including the reviewed CPU-hidden deterministic target-lock replay.
If it returns a material `VERDICT: REVISE`, stop again and record the new exact
blocker; this authorization does not create an unbounded review loop.

<!-- BEGIN A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->
## 2026-07-12 Concurrent-Lane Authorization

The user stated that another agent is working on a different lane and directed
this executor to focus only on the SSL-LSTM lane. This authorizes unrelated
concurrent worktree changes to be preserved and treated as observable but
non-vetoing for Phase A1.

The authorization does not permit this lane to edit, stage, commit, reset, or
reinterpret the other lane. It also does not waive A1 protections: the fixed A0
anchor commit, accepted A0/golden artifacts, target-critical dependency hashes,
and declared A1-owned paths remain binding. A later evidence-run commit is
allowed only when the A0 anchor is its ancestor and every committed path since
the anchor is disjoint from the protected and A1-owned sets. Drift in a
protected dependency or an unexpected write into the A1-owned set is a stop
condition. Repository-wide index/porcelain drift outside those boundaries is
explanatory provenance only.

The stale full-repository inventory is therefore superseded by the reviewed A1
scoped-boundary v2 contract before any further implementation or evidence run.
This boundary change does not authorize HMC, NeuTra, forecasting, target/model
changes, scientific claims, product/default changes, or Git publication.
<!-- END A1 SCOPED CONCURRENT-LANE AUTHORIZATION -->

## 2026-07-13 A2 Close And A3 Boundary

- A2 status is `PASSED_FOR_A3_PLANNING_ONLY` after fresh CPU-hidden reference,
  trusted GPU/XLA canary, fresh-process replay, finite-admission repair, and
  bounded implementation/result reviews.
- The A3 forecast-oracle/statistics subplan received bounded
  `CODEX_SUBSTITUTE_REVIEW` `VERDICT: AGREE` after two substantive rounds. This
  is weaker than Claude; the external Claude gate was policy-rejected before
  process creation or disclosure and was not routed around.
- Existing user authorization covers A3 engineering implementation inside the
  exact reviewed A3 write set, including trusted GPU/XLA oracle execution, only
  after A2 post-result closure and terminal trace audit pass.
- This authority excludes HMC, NeuTra, sampler comparison, A4 calibration,
  model-file changes, package/network operations, commit/stage/push, public API
  or default changes, and scientific/product claims.
- The concurrent HMC/Kalman lane remains outside scope and must be preserved.

## 2026-07-13 Human-Authorized A2 Trace-Contract Repair

After the terminal audit failure was explained as the unanchored `link(` versus
`readlink(...)` parser error, the user directed: "fix that and continue." This
is recorded as explicit human authority for the narrow repair required to make
that audit truthful:

- replace the false-positive matcher with the reviewed fail-closed syscall
  parser and its focused regression tests;
- regenerate only the stale A2 result/reviews/ledgers/checkpoint/closure;
- qualify every newly generated A2 closure trace with exactly
  `/usr/bin/strace -f -qq -yy -s 65535 -e trace=%file`; and
- continue to A3 only after fresh closure verification and the exact terminal
  trace audit pass.

This human-authorized repair supersedes the older A2 trace-command spelling
only for the stale closure-regeneration chain. It does not change the accepted
A2 model, forecast contract, source write set, runtime acceptance criteria, or
scientific boundary. It does not authorize HMC, NeuTra, sampler comparison,
A4 calibration, other-lane edits, package/network actions, model-file changes,
Git publication, product/default changes, or scientific claims.
