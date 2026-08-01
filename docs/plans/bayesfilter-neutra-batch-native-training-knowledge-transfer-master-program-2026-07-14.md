# BayesFilter NeuTra Batch-Native Training Knowledge-Transfer Master Program

Date: 2026-07-14

Execution status: **COMPLETE THROUGH PHASE 8.** See
`docs/plans/bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-result-2026-07-14.md`.
The fresh 5,000-step campaign is a separate next plan, not an unfinished phase
of this migration program.

## Program Objective

Repair the current LGSSM NeuTra runtime bottleneck without changing the admitted
posterior target, and institutionalize the relevant DSGE implementation knowledge
as reusable BayesFilter contracts, kernels, tests, and runbook requirements.

The current active LGSSM training target accepts a batch-shaped tensor but uses
`tf.map_fn` to replay a scalar target over rows. That route is migration debt
under the 2026-07-14 batch-native training policy and is ineligible for further
serious training until this program closes its correctness and execution gates.

## Why A Master Program

This work crosses five distinct engineering and scientific boundaries:

1. repository-wide training-policy enforcement;
2. transfer and formalization of the DSGE batch-native design;
3. a new batch-native SVD/eigh graph-status numerical kernel;
4. adapter/trainer integration and mathematical parity certification; and
5. trusted GPU performance work followed by target-specific protocol replay.

A single implementation plan would hide phase-specific entry conditions and
would make it too easy to treat a performance success as mathematical admission.
Each phase therefore has a dedicated subplan and close record.

## Research Intent Ledger

| Item | Program contract |
| --- | --- |
| Main question | Can the exact 18-parameter, `T=120` LGSSM NeuTra target be evaluated and trained batch-natively while preserving scalar SVD/eigh value, score, status, seed, objective, and optimizer semantics? |
| Candidate mechanism | Batched LGSSM materialization plus a batch-native SVD/eigh graph-status Kalman value/score recursion, bound through a repository-owned NeuTra capability contract. |
| Exact authority | Existing scalar `lower_triangular_lgssm_log_prob_score_status` target and its fixture-bound target signature. |
| Expected failure mode | Tensor shapes or stationary-covariance derivatives fail parity; XLA fails to fuse the batch kernel; GPU memory limits the batch; or the kernel remains too slow. |
| Promotion criterion | Mathematical/status parity passes, the trainer has no scalar/sample-map fallback, trusted GPU/XLA execution passes, and the predeclared performance ladder shows a practically useful batch-native route. |
| Promotion veto | Target/signature drift, value/score/status mismatch, nonfinite output, scalar/sample-map fallback, missing batch provenance, or GPU/XLA invalidity. |
| Continuation veto | Evidence harness invalidity, an unrepairable mathematical mismatch with the frozen scalar authority, artifact corruption, or campaign compute-budget exhaustion. A slow or failed candidate is a repair trigger, not automatically a continuation veto. |
| Repair trigger | Focused parity, XLA, device, memory, or performance failure with the target and evidence contract unchanged. |
| Explanatory diagnostics | Component timings, HLO/operation inventory, memory, compile time, descriptive losses, and scalar-target timings. |
| Forbidden conclusion | Kernel parity or speed does not establish transport quality, posterior correctness, HMC convergence, recipe superiority, or broad model generalization. |

## Frozen Scientific Boundary

The program must not obtain speed by changing:

- target signature, fixture, prior, parameter order, or raw-coordinate transforms;
- `float64` arithmetic;
- SVD/eigh invariant solve/logdet semantics;
- graph-status, active-floor, or invalid-eigensolver behavior;
- stateless training seed stream;
- reverse-KL objective, dense-IAF structure, clipping, or manual Adam equations; or
- batch size `128` for the final decision comparison.

The existing batch-native Cholesky Kalman kernel is a design source and optional
explanatory comparator. It is not an admissible replacement for the frozen
SVD/eigh target.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Scalar SVD/eigh route remains authority | Current exact-target/HMC admission | Preserves the computed posterior and status law | Authority itself could contain an undiscovered bug | Existing exact-target tests plus independent finite-difference/reference checks | frozen baseline |
| Batch-native leading dimension | DSGE repaired NK and generic experimental Kalman implementations | Removes scalar replay and exposes accelerator parallelism | Batch broadcasting or axis errors | shape and row-permutation tests | reviewed implementation choice |
| `float64` | Current target and repository evidence | Isolates topology from precision changes | GPU FP64 throughput is limited | component benchmark; no silent FP32 fallback | frozen baseline |
| Batch size `128` | Active LGSSM target-specific protocol | Fair comparison with current training | May not be throughput-optimal | batch ladder 8/32/64/128/256 after parity | fixed decision baseline, tuning hypothesis for future |
| Warm performance goal `<=1.4 s/step` | DSGE serious runtime and optimized NK evidence | Makes a 5,000-step run roughly two hours | Target is materially more expensive (`T=120`, 18 parameters) | target-only and 20-step timing ladder | aspirational repair target, not a correctness gate |
| One time-axis loop | Kalman sequential dependence | Time recursion is genuinely sequential | XLA loop overhead or poor fusion | operation inventory and component profile | reviewed default |
| No sample-axis map/loop | Owner batch-native policy | Prevents scalar replay disguised as batching | Some diagnostic helper leaks into live path | source/graph/capability audit | hard policy |
| Repository-owned capability issuance | Existing identity/capability patterns | A caller should not self-label a scalar target as batch-native | Capability becomes ceremonial and detached from callable | factory binds method identity and source audit | implementation requirement |

## Skeptical Program Audit

| Risk | Audit response |
| --- | --- |
| Wrong baseline | Preserve and call the exact scalar SVD/eigh target on identical rows; do not use historical training time as the parity baseline. |
| Proxy promoted | Value/score/status parity is mathematical admission; runtime is engineering admission; downstream HMC remains scientific promotion evidence. |
| Missing stop conditions | Every subplan must name parity, target, artifact, device, XLA, memory, and budget stops. |
| Unfair performance comparison | Same target, inputs, batch, dtype, seed, flow, optimizer, GPU, synchronization, and update count; compile and warm timings separated. |
| Hidden assumption about DSGE reuse | Transfer design and tests, not target-specific numerical outputs or defaults. Cholesky and sigma-point routes are not silently substituted. |
| Stale context | Each phase records current Git commit/dirty status and source hashes for the paths it depends on. |
| Environment mismatch | CPU-hidden reference/XLA checks and trusted GPU/XLA checks are separate and labeled. |
| Artifact would not answer question | Phase results separately report engineering correctness, numerical parity, and performance; no single smoke closes all ledgers. |
| Plan could pass while target still maps rows | Capability factory, source/graph audit, and negative adapter tests make row-mapped scalar targets hard vetoes. |
| Local optimization drift | Phase 1 records every imported design choice and why it applies to this LGSSM; unexamined DSGE defaults remain hypotheses. |

Audit verdict before execution: **PASS**. The program preserves the research
question, separates admission ledgers, contains explicit repair triggers, and
does not treat a candidate failure as a reason to abandon the planned repair
phase.

## Phase Map

### Phase 0: Boundary, Inventory, And Fail-Closed Enforcement

Create a repository-owned batch-native capability contract; inventory all
NeuTra optimizer entry points; make generic and target-specific trainers reject
missing, scalar, row-mapped, batch-size-one, non-XLA, or self-attested routes
before an optimizer update or artifact directory is created.

Output subplan:
`docs/plans/bayesfilter-neutra-batch-native-training-phase0-boundary-inventory-subplan-2026-07-14.md`

### Phase 1: DSGE Knowledge-Transfer Specification

Write a source-anchored transfer specification covering the optimized NK GPU
route, the SGU/Rotemberg persistent CPU-shard route, the reusable BayesFilter
batch Kalman kernel, score injection, XLA boundaries, parity authorities, and
known failed designs. Classify every imported choice as reusable mechanism,
target-specific assumption, comparator only, or rejected route.

### Phase 2: Batch-Native LGSSM Materialization

Implement `[B,18]` to batched transition, covariance, stationary covariance,
first derivatives, and prior tensors using batch tensor algebra and batched
linear solves. No sample-axis loop or mapping primitive is allowed.

### Phase 3: Batch-Native SVD/Eigh Graph-Status Kernel

Implement one time-axis TensorFlow loop carrying batch-native state, covariance,
first derivatives, log likelihood, score, and per-row status telemetry. Preserve
the scalar target's invariant solve/logdet and status semantics.

### Phase 4: Exact Adapter And Trainer Integration

Bind the actual batch-native callable to a non-overridable capability, expose it
through the exact target adapter, and require it in the NeuTra training engine.
Preserve scalar methods only for HMC and independent parity diagnostics.

### Phase 5: Correctness And Boundary Certification

Run shape, row-order, scalar parity, status parity, eager/graph/CPU-XLA/trusted
GPU-XLA, objective-gradient, one-step update, five-step state, no-NumPy,
no-sample-loop, no-host-callback, and negative-adapter checks.

### Phase 6: Performance Ladder And Focused Repair

Benchmark target components and full training at predeclared batch/step rungs.
If the route misses the practical target, repair broadcasts, materialization,
stationary solves, duplicate status calls, fusion, or batch chunking in that
order. Every repair uses fresh artifacts and reruns focused parity before timing.

### Phase 7: Target-Specific Protocol Replay

Run fresh five-step recipe smokes, a 100-step stability rung, then the reviewed
500-step selection screen only if all preceding vetoes pass. The 5,000-step
final training seeds require a refreshed serious campaign budget based on the
measured runtime and remain subject to the target-specific training protocol.

### Phase 8: Institutionalization And Closeout

Promote the proven generic capability/API, add adapter templates and policy
tests, demote obsolete row-mapped training routes, update reset/runbook material,
and write terminal engineering/numerical/scientific ledgers.

## Required Phase Procedure

At the end of every phase, the supervisor/executor must:

1. run the local checks named in that phase's subplan;
2. write a phase result/close record with attempts, evidence, decision table,
   inference status where stochastic evidence exists, and unresolved risks;
3. draft or refresh the next phase subplan using the actual close state;
4. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, default/assumption coverage, and boundary safety; and
5. continue automatically when no real mathematical, scientific, privacy,
   destructive, cost, artifact-validity, or compute-budget blocker exists.

Procedural imperfection, reviewer unavailability, a slower candidate, or an
expected implementation failure is not a real blocker. A localized failure
triggers repair within the same phase when the target, method, evidence contract,
hardware class, and phase budget remain unchanged.

## Repair And Retry Protocol

- Preserve every failed attempt and classify it as harness, implementation,
  numerical, XLA/device, resource, performance, or scientific-contract failure.
- Patch the smallest responsible path visibly.
- Run the smallest focused regression that distinguishes the proposed repair.
- Rerun in a fresh versioned output directory without renewed user approval
  while remaining inside the phase contract and total campaign budget.
- Stop only when a continuation veto fires or the phase budget is exhausted.
- A failed performance candidate must not be mislabeled as evidence against
  batch-native training when a later phase is designed to repair that failure.

## Review Contract

One material program review and one terminal result review are sufficient by
default. Phase-transition reviews are local suitability audits, not external
authority ceremonies. Claude may provide bounded read-only review when it
materially helps, but reviewer unavailability or procedural disagreement is not
a blocker. A mathematical, numerical, target-identity, privacy, destructive,
or materially expanded-compute finding remains a blocker.

## Program Budget

- Routine implementation and focused CPU-hidden tests: bounded by the phase
  subplans and normal local execution.
- Trusted GPU development: short correctness smokes plus the Phase 6 ladder,
  initially capped at 45 minutes aggregate live GPU time before a refreshed
  budget is written from measured results.
- Phase 7: five-step smokes and one 100-step stability rung are included after
  Phase 6 admission. The 500-step screen and 5,000-step final runs require a
  refreshed campaign plan/compute budget, but not cryptographic approval tokens.

## Program Stop Conditions

Stop the program only for an unrepairable mismatch with the frozen scalar target,
corrupted or untrustworthy evidence, missing platform permission, a true privacy/
destructive/external boundary, materially changed scientific scope, or exhausted
authorized compute with no lower-cost discriminating check. Otherwise use the
phase repair loop and continue.
