# Phase 5 Result: Actual-SV Full-Row Score/Memory Refresh

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_RUNTIME_MEMORY_SCALING`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-full-row-after-streaming-parity-repair

## Phase Objective

Decide whether the repaired actual-SV same-target no-tape score route can
proceed from tiny streaming-parity diagnostics to full `N=10000,T=1000` score
admission.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Full actual-SV score admission remains blocked. |
| Primary criterion status | Not met: no full `N=10000,T=1000` admitted score artifact exists. |
| Veto diagnostic status | Current all-coordinate FD / stored-record reverse route has unacceptable runtime and near-budget GPU memory at the second ladder rung. |
| Main uncertainty | Need a reviewed full-row correctness strategy that avoids repeated large manual transport pullbacks, or an exact-reference admission route. |
| Next justified action | Create a dedicated subplan for full-row actual-SV score admission strategy: memory instrumentation, no-FD value/score run, checkpoint/recompute or exact-reference correctness, and validator-compatible artifact semantics. |
| What is not concluded | No mathematical rejection of the repaired score route; no full score admission; no HMC readiness, posterior correctness, runtime ranking, or scientific superiority claim. |

## Review Status

The refreshed full-row subplan was reviewed by direct bounded Claude read-only
review after the wrapper probe timeout pattern was known. Claude returned:

```text
VERDICT: AGREE
```

The agreement applied only to continuing the gate review, not to score
admission.

## Memory-Risk Audit

For scalar actual-SV at `B=1,N=10000,T=1000,float64,particle_chunk_size=64`,
the repaired stored-record reverse scan retains approximately:

```text
flow aux blocks per time step = 157
explicit non-flow records      = 0.458 MiB / time step
streaming-flow aux records     = 1.006 MiB / time step
total retained payload         = 1.464 MiB / time step
T=1000 retained payload        = 1.430 GiB / seed before TensorFlow/object overhead
```

This tensor-payload estimate alone did not veto a small ladder, because the CLI
processes seeds sequentially. It also did not admit the score, because memory
is not correctness evidence.

## Ladder Results

Trusted GPU probe:

```text
nvidia-smi
```

Result: GPU was visible and healthy; RTX 4080 SUPER with `16376 MiB` total
memory and about `2571 MiB` in use before ladder commands.

### Rung 1: T=5,N=256

Command: trusted GPU, `float64`, no full admission flag.

Artifact:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.md`

Result:

```text
elapsed_seconds = 101.32655131816864
peak_mib = 32.68115234375
score = [-0.2798343472803253, -0.1427259831638991]
fd_score = [-0.27984036607975327, -0.14268399164407697]
max_abs_error = 4.1991519822121015e-05
max_rel_error = 0.00029421075890505614
score_admission_status = tiny_score_diagnostic_not_admitted
```

Interpretation: rung 1 passed as diagnostic evidence only.

### Rung 2: T=20,N=1024

Command: trusted GPU, `float64`, no full admission flag.

Result: manually interrupted with `KeyboardInterrupt` after roughly 15 minutes.
No JSON artifact was written.

Runtime/GPU status before interruption:

```text
PID 3611693 elapsed about 10:40 at status check, CPU active
nvidia-smi showed about 15711 MiB / 16376 MiB in use
```

The traceback showed execution inside:

```text
annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_pullback
annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_total
annealed_transport_tf._filterflow_streaming_softmin_vjp
```

Interpretation: the blocker is runtime/memory scaling of the current
all-coordinate finite-difference plus manual total-pullback route. This is not
a target-scalar parity failure and not a mathematical correctness failure at
tiny scales.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the repaired actual-SV no-tape total score route scale to full `N=10000,T=1000` with replayable correctness and memory evidence? |
| Answer | Not with the current all-coordinate FD ladder and stored-record manual total-pullback implementation. |
| Baseline/comparator | Admitted actual-SV value artifact, repaired tiny score diagnostic, Phase 1 score validator, same-forward-scalar parity tests. |
| Primary criterion | Failed/not met because no full artifact was attempted or admitted; second ladder rung was practically blocked. |
| Veto diagnostics | Runtime and near-budget memory pressure at `T=20,N=1024`. |
| Explanatory diagnostics | Rung 1 passed; memory payload estimate appears bounded per seed but object/loop overhead and transport pullback scaling dominate. |
| Artifact | This blocker result plus rung 1 artifact and refreshed next subplan. |

## Required Repair

The next phase must not simply rerun the full command. It must choose and
review a full-row admission strategy that addresses the current scaling
blocker. Candidate directions:

1. Add a no-FD value/score-only GPU memory probe for `T=20,N=1024` and larger
   shapes to separate score reverse-pass memory from repeated FD overhead.
2. Add targeted timing/memory instrumentation around transport VJP, flow VJP,
   and per-time reverse scan.
3. Consider a checkpoint/recompute reverse scan that retains less per-time
   state, if the stored-record design is confirmed to dominate memory.
4. Consider a validator-compatible `exact_reference` route for correctness so
   full admission does not require repeated full-row finite differences.
5. Review any validator or artifact-semantics change before implementation.

## Nonclaims

- Full actual-SV score is not admitted.
- The repaired tiny score route remains valid only as tiny diagnostic evidence.
- The interrupted rung is not evidence against the mathematics.
- No KSC, generalized-SV, raw Gaussian, or augmented-noise score claim is made.
- No HMC readiness, posterior correctness, runtime ranking, scientific
  superiority, or all-algorithm comparison is claimed.
