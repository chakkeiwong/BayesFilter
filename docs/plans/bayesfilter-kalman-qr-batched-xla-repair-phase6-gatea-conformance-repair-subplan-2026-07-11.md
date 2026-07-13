# BayesFilter Kalman QR Batched XLA Repair Phase 6 Gate A Conformance Repair Subplan

Date: 2026-07-11

Status: `THIRD_AUDIT_EXPANDED_IMPLEMENTATION_REPAIR_ACTIVE`

Parent subplan:

- `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-cpu-xla-gates-subplan-2026-07-11.md`
- authoritative parent SHA-256 at entry:
  `b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b`

This is a repair subplan inside Phase 6 Gate A. It does not replace, relax, or
extend the parent Phase 6 evidence contract or runtime authority.

## Phase Objective

Make the Phase 6 contract and visible supervisor conform to the already
reviewed Gate A requirements before any Gate B or Gate C target workload is
eligible for review or launch. Close the bounded implementation defects found
by a fresh read-only audit:

1. preserve and validate every imported Gate B pilot terminal outcome without
   replaying it as Gate C process provenance;
2. enforce per-child and cumulative Gate B/C monotonic runtime budgets and
   deterministically close unlaunched work as `global_budget_exhausted`;
3. recover durable `running` entries safely and clean every owned process group
   on timeout, TERM, interrupt, callback failure, or supervisor exception;
4. embed bounded lossless raw stdout and stderr bytes with exact byte count and
   digest validation;
5. stop launching after common invalidity and terminalize the remaining roster;
6. implement the full preallocated, incrementally reparsed `P=150` routing
   ledger rather than a summary object;
7. parse and validate the opening hash ledger as a closed path/digest contract;
8. bind the exact child schedule payload and digest into the immutable reviewed
   budget proposal and its runtime authority chain; and
9. reject non-pass terminal records whose available journal, sidecar, process,
   or claimed failure stage disagree.

### Fresh Audit Expansion

A second independent read-only Gate A audit of the implementation snapshot
returned `VERDICT: REVISE`. Its findings are binding repair work inside this
same Gate A scope; they do not authorize target execution or widen numerical
scope:

1. validate every reviewed child argv against its exact identity, config,
   fingerprints, attempt, paths, and benchmark mode before accepting a schedule;
2. make the 160-second paired-cell caps and 3045/3120-second gate budgets
   executable, durable, monotonic, and deterministic on exhaustion;
3. convert outer SIGTERM into owned process-group cleanup and a durable
   interrupted terminal record before supervisor exit;
4. replace the post-run routing summary with the required preallocated,
   incrementally reparsed 18-record routing ledger;
5. revalidate authority, source/runtime identity, schedules, and bound inputs
   immediately before every child launch;
6. accept an honest operational journal prefix followed by `envelope_write`
   after a caught early failure, while rejecting gaps or reordered events;
7. bind passed v4 envelopes to their exact attempt and progress journal;
8. require scalar and remaining Gate C artifacts to share one authority, and
   require pilot/trace Gate B predecessors to share one authority;
9. preserve valid imported pilot `not_launched` records without inventing Gate C
   execution provenance;
10. expose closed import-discovery and proposal-construction commands rather
    than relying on unaudited manual JSON creation;
11. parse Linux `/proc/<pid>/stat` after its parenthesized command field; and
12. remove or reject stale deterministic child artifacts before a fresh attempt.

The fresh audit also confirmed that target workloads remain invalid until these
items, the original nine objectives, focused mutations, local checks, and the
bounded review gate all converge.

### Third Audit Expansion

Three independent read-only audits after the first complete focused-suite run
returned material lifecycle, budget-resume, and routing findings. They are
binding Gate A repair work and do not authorize target execution:

1. cover the whole per-child transaction with controlled SIGTERM handling,
   including prelaunch, `Popen` through durable `running`, post-child evidence,
   terminal persistence, and the interval before the next child;
2. guarantee cleanup when process-identity discovery or the durable-start
   callback fails, and never require a false `interrupted` child record when no
   child obtained durable ownership;
3. preserve malformed-present partial artifact bytes and deterministically
   close a safely owned stale/interrupted child as
   `interrupted/supervisor_interruption` rather than retrying forever, deleting
   evidence, or upgrading partial bytes into valid child output;
4. make dead-leader/live-group recovery fail closed: a live recorded leader is
   cleanable only after exact PID/PGID/start-token verification; a dead leader
   plus dead group may close; a dead leader plus live group is unauthenticated
   and must never be signalled;
5. add one durable live-supervisor lease per gate authority, reject concurrent
   supervisors, and permit resume only after stale-owner verification;
6. bind budget state to the host boot identity, latest monotonic update, and
   exact active Gate B/C command so reboot/clock rollback or wrong-command use
   cannot increase remaining time;
7. pass an absolute lifecycle deadline through validation, spawn, durable
   start, execution, TERM/KILL, and persistence rather than sampling remaining
   budget only before prelaunch work;
8. preserve immutable prelaunch routing decisions but add a terminal overlay
   that distinguishes historical eligibility from final common invalidity and
   final disposition;
9. run the same full routing-versus-final-ledger cross-validation on normal
   closure and completed resume; and
10. preflight the complete imported pilot record set so imported common
    invalidity is established before any Gate C launch.

The first expanded suite produced `329 passed, 1 deselected, 7 failed`; all
seven failures were one stale Phase 5 test fixture missing the new optional
dependency-before field. The focused fixture repair passed `9` tests. Those
counts are diagnostic only and do not close Gate A while the findings above
remain open.

## Entry Conditions Inherited From Phase 5 And Parent Phase 6

- Phase 5 is closed with its strict measurement smoke and reviewed result.
- The Phase 6 parent subplan above is the sole scientific and runtime contract.
- The master ledger state is
  `PHASE6_GATE_A_IMPLEMENTATION_AUTHORIZED_TARGET_RUNTIME_BLOCKED`.
- Gate A may edit harness/contract/test and governance artifacts only.
- No Phase 6 Gate B/C proposal, review, or attestation currently grants target
  trace, target XLA, or target scalar-reference runtime authority.
- Opening implementation hashes at this repair entry are:
  - `scripts/kalman_qr_benchmark_contract.py`:
    `554eb39b555eed6f090aef4c21b49f88603608f7e14a64d4619e601dc1118f56`;
  - `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`:
    `339529f71c2e4fbbdfcffecb9ffee2229fdf26df33ec53e547188697b3eb20d0`.
- Unrelated dirty work and the other active repository lane are out of scope
  and must not be modified or interpreted as this lane's evidence.

## Required Artifacts

- repaired `scripts/kalman_qr_benchmark_contract.py`;
- repaired
  `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py`;
- focused mutation/authority tests in
  `tests/test_kalman_qr_benchmark_contract.py`;
- focused process/budget/recovery/import/routing tests in
  `tests/test_kalman_qr_phase6_cpu_xla_gates.py`;
- focused lifecycle, durable-budget, routing-overlay, and entrypoint tests in
  `tests/test_kalman_qr_phase6_gatea_runtime_controls.py`;
- closed import-only CLI tests in
  `tests/test_kalman_qr_phase6_import_discovery_cli.py`;
- this repair subplan;
- Phase 6 Gate A repair result
  `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatea-conformance-repair-result-2026-07-11.md`;
- refreshed lane ledger and stop handoff;
- bounded Claude review logs under `.claude_reviews/` and a concise committed
  review record under `docs/reviews/` only when needed for the visible trail.

No Gate B/C target output, budget proposal, attestation, trace census, scalar
reference, routing result, or CPU-XLA result is a required artifact of this
repair subplan. Those remain prospective later-gate artifacts.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Do the Phase 6 schemas, authority validation, state machines, process lifecycle, and supervisor simulations fail closed under the original nine defects and all subsequent audit expansions? |
| Exact baseline | Parent Phase 6 subplan at SHA-256 `b7b653d8febfa341dd2e8b53e8c274246eb49b6afcc59e4bca27126d3b33769b`, plus the two implementation entry hashes recorded above. |
| Primary pass criterion | All new focused mutations/simulations pass, the complete focused Phase 6 Gate A suite passes, every audited defect has a source anchor and test anchor, and a fresh bounded review returns exact `VERDICT: AGREE`. |
| Hard vetoes | A target workload is launched; a process descendant survives a cleanup test; a live second supervisor can share one authority; stale `running` state is overwritten; reboot/clock rollback increases budget; an unreviewed schedule or wrong active command can obtain authority; imported pilot provenance is rewritten; malformed terminal evidence is accepted or causes infinite recovery; a closed routing artifact misstates historical eligibility as current after common invalidity; opening-ledger limitations are hidden; or required tests fail. |
| Repair triggers | Any focused test/review finding showing fail-open authority, lifecycle, provenance, schema, routing, budget, or artifact behavior. Patch this same repair lane and rerun the smallest affected checks before the full focused suite. |
| Explanatory only | Line counts, implementation size, test duration, and reviewer prose without the exact verdict. |
| Not concluded | No target numerical validity, trace stability, CPU-XLA viability, GPU readiness, performance/scalability, method ranking, HMC/posterior correctness, default readiness, production readiness, or scientific validity. |
| Preserved result | Repair result with commands, exit codes, hashes, defect-to-test table, bounded review trail, residual risks, and exact Gate B readiness decision. |

## Skeptical Plan Audit

Result: `PASS_WITH_REPAIR_SCOPE_LOCKED`.

- Wrong baseline: avoided by binding the reviewed parent SHA-256 and current
  implementation hashes; no stale reset-memo hypothesis is used as a pass
  criterion.
- Proxy promotion: unit and process simulations establish Gate A harness
  conformance only. They cannot promote target numerical/runtime claims.
- Missing stop conditions: explicit vetoes cover target launch, process escape,
  authority fail-open behavior, cross-lane overlap, test failure, and review
  non-convergence.
- Unfair comparison: no method comparison or performance measurement occurs.
- Hidden assumptions: PID/PGID identity, monotonic clock ownership, bounded raw
  output size, imported artifact immutability, and opening-ledger coverage are
  each required to be represented and mutation-tested.
- Stale context: file hashes were checked immediately before this subplan was
  written and matched the fresh audit checkpoint.
- Environment mismatch: tests use CPU-hidden or no TensorFlow import, harmless
  subprocess trees, temporary files, and synthetic ledgers only.
- Artifact fitness: every command produces direct pass/fail evidence for an
  audited defect; no target artifact is used to answer a Gate A question.

## Implementation Contract

### Imported pilot records

- Gate C binds the immutable pilot artifact as an input blob and verifies its
  path/digest/schema/authority/schedule/proposal/attestation chain.
- Every pilot terminal state is imported by reference into the final decision;
  it is not replayed through a new Gate C `pending -> running -> terminal`
  process transition.
- The final Gate C roster still has one record per exact XLA identity. Pilot
  identities are terminalized with an explicit imported-evidence variant whose
  process/schedule provenance remains Gate B and whose import binding is Gate C.
- Failed, timed-out, crashed, interrupted, and valid not-launched pilot states
  remain honest dependency evidence; malformed pilot evidence is common
  invalidity and blocks further launches.

### Runtime budget and schedule authority

- The immutable proposal contains the exact canonical schedule payload and
  schedule SHA-256 reviewed for that gate.
- Runtime recomputes and compares the schedule before creating or resuming a
  ledger and before every launch.
- One shared monotonic gate clock owns each gate budget. Separate scalar and
  remaining commands consume one Gate C authority using a persisted budget
  journal/state artifact; neither command resets the 3120-second ceiling.
- The durable state path is exactly
  `/tmp/kalman_qr_phase6_cpu_xla/budget_state/<gate>-<authority_id>.json`.
- The state binds the host boot identity, latest monotonic update, exact active
  command, and one live-supervisor PID/start-token lease. A concurrent live
  owner, stale/ambiguous owner, command mismatch, or monotonic rollback blocks
  before spawn.
- Before each launch, the supervisor proves enough remaining lifecycle budget
  for the child or terminalizes it and its dependants as
  `not_launched:global_budget_exhausted` without spawning.
- The launch receives one absolute deadline derived from the gate and paired
  cell ceilings. Prelaunch revalidation, spawn, durable callback, execution,
  cleanup, evidence collection, and terminal persistence all consume it.
- The paired 160-second pilot-XLA cell cap and outer Gate B/C ceilings are
  executable limits, not CLI metadata.

### Process lifecycle and recovery

- A ledger is reparsed before initialization; an existing valid ledger is never
  overwritten merely because the command restarted.
- A durable `running` record with a live leader is matched using its exact PID,
  PGID, and process start token before the owned process group may be signalled.
  A dead leader plus dead group may close as already gone. A dead leader plus a
  live group is unauthenticated: stop without signalling, terminalizing, or
  launching. No descendant ownership token is claimed or implemented.
- Every exit path owns one idempotent TERM/KILL/reap/finalize routine.
- SIGTERM is deferred only while ownership or terminal persistence is in a
  critical transition. Once a child has a durable `running` record, TERM
  interrupts promptly, cleans the verified group, and writes an honest terminal
  record. Before spawn it leaves the record pending; after group exit it waits
  only until terminal evidence is durable, then stops before the next spawn.
- Tests cover normal exit, timeout, callback exception, keyboard interrupt,
  simulated outer TERM at every transaction boundary, stale-dead recovery,
  dead-leader/live-group recovery, stale-live recovery, and PID/PGID/start-token
  mismatch without leaving an owned descendant alive or signalling an
  unverified process.

### Lossless output and failure evidence

- Terminal process records embed base64 stdout/stderr bytes with closed maximum
  sizes, truncation metadata if the configured bound is reached, exact stored
  byte counts, and SHA-256 over embedded bytes.
- A bound hit is explicit and cannot be described as full/lossless capture.
- Terminal evidence validation cross-checks available journal stages, sidecar
  presence/content, return code/signal/timeout, terminal state, reason, and
  classification. Inconsistency becomes `invalid_child_evidence`, never a
  method/backend result.
- Malformed-present recovery files are byte-preserved with path, size, and
  digest. They close once as interrupted nonpass evidence and are never silently
  removed, accepted as valid child output, or relaunched under the same attempt.
  The same malformed blob is invalid for every passed/failed/timed-out child
  envelope that claims normal child terminalization.
- A fresh pending launch refuses any pre-existing deterministic child path; it
  never deletes or quarantines unbound bytes without a reviewed manifest.

### Common invalidity and routing

- Once common invalidity is durably established, no later child launches; every
  remaining eligible record closes with exact common-invalidity pruning.
- The routing artifact uses the contract's exact 18-record roster, starts
  preallocated as pending dependency, records dependency identities/digests and
  exact rule/action outcomes, and strictly reparses after each transition.
- Prelaunch routing decisions are immutable historical facts. Terminal closure
  adds a validated final-disposition overlay and global common-invalidity
  status; an earlier eligible/attempted route cannot appear currently eligible
  after a later common-invalidity veto.
- Normal closure and completed resume call one shared validator against every
  decided route and the final ledger. Gate C preflights all imported pilot
  records before any new launch.

### Opening hash ledger

- Permit exactly the two frozen header lines, then parse every data line as
  `<64-lowercase-hex-or-ABSENT><two spaces><repo-relative-path>`.
- Reject duplicate, missing, extra, escaping, symlink-ambiguous, or malformed
  entries.
- Require exact frozen coverage of the declared Gate A write/read-only set,
  with `ABSENT` allowed only where declared. Record explicitly that the opening
  ledger has 144 data entries and omitted 36 historical JSON counterparts; do
  not claim it was a complete repository or historical-artifact inventory.
- Bind the parsed canonical payload and digest into the proposal, not merely the
  current ledger file bytes.

## Required Checks, Tests, And Reviews

Run only after the corresponding implementation is present:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py

CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_phase6_cpu_xla_gates.py
```

The focused suite must include direct mutations/simulations for all nine
objective items. After local success, request Claude Opus at max effort as a
read-only reviewer, beginning with exactly this subplan path under the
repository's one-path prompt rule. Subsequent prompts may name one exact source,
test, or result path only when Claude requests it or the gate question requires
it. A probe success followed by a material-review timeout means the prompt must
be narrowed/redesigned; it is not reviewer unavailability. Repair visible
findings in this same lane and stop after five material rounds for the same
blocker.

## Forbidden Claims And Actions

- Do not launch Phase 6 target trace, target XLA, target scalar-reference, GPU,
  comparison-ladder, HLO-dump, or benchmark commands.
- Do not generate a review attestation or represent Gate B/C runtime as
  authorized during Gate A repair.
- Do not increase a timeout, tolerance, roster, budget, or runtime scope.
- Do not edit `bayesfilter/linear/*.py`, the benchmark method implementation,
  Phase 0-5 results, unrelated files, or the other active lane.
- Do not kill a process when recorded identity cannot be established safely.
- Do not convert a malformed artifact into a method/backend failure.
- Do not call tests proof of numerical correctness, XLA viability, or
  scientific validity.
- Claude is read-only and cannot authorize runtime, human, model-file, funding,
  product-capability, default-policy, or scientific-claim boundaries.

## Exact Parent-Phase Handoff Conditions

All conditions are conjunctive:

- all nine defects have source and focused-test anchors in the repair result;
- required compile and test commands pass from the repository environment;
- no target workload was launched and no subprocess descendant remains;
- closing hashes/scoped status are recorded without claiming ownership of
  unrelated dirty files;
- the parent Phase 6 subplan remains unchanged or any necessary patch receives
  a fresh bounded review before use;
- bounded Claude review of the repair result and directly relevant exact paths
  converges to `VERDICT: AGREE` within five rounds;
- the lane ledger records Gate A conformance closed and Gate B still blocked;
- the next action is only to refresh exact `--help`, source/dependency/opening
  hashes, the schedule-bound Gate B proposal, and its separate review and
  detached attestation.

Crossing into Gate B additionally requires the parent Phase 6 Gate B authority
conditions. This repair result alone never authorizes target execution.

## Stop Conditions

- A required fix overlaps irreconcilably with another lane or unrelated user
  work.
- Safe PID/PGID identity and cleanup cannot be made fail closed on this host.
- A required schema change would relax or contradict the reviewed parent rather
  than implement it.
- Local focused tests cannot pass without target execution or an unapproved
  environment/package change.
- The same material review blocker does not converge after five rounds.
- New human authority is required.

## Mandatory Repair Close Sequence

1. Run the required local checks; no target workload is part of them.
2. Write the Gate A conformance repair result/close or blocker record.
3. Refresh the parent Phase 6 Gate B readiness section and exact proposal plan
   only from implemented facts; do not launch it.
4. Review the refreshed next action for consistency, correctness, feasibility,
   artifact coverage, and boundary safety.
5. Advance only after the exact parent-phase handoff and Gate B authority
   conditions are both satisfied.
