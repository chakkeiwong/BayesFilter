# Phase 2 Blocker Result: LGSSM Full Score Run Too Expensive In Current Path

metadata_date: 2026-07-07
status: `BLOCKED_FIXABLE_REPAIR_SUBPLAN_REQUIRED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 2

## Blocker

The Phase 2 LGSSM preflight passed, but the first full `N=10000`, `T=50`
score run did not produce a raw JSON artifact in a bounded visible window.

This is not a score-correctness failure and not a schema failure. It is an
execution-plan failure: the current raw runner performs the full compact score
computation and then central finite differences for each of five parameters.
At `N=10000`, `T=50`, that repeats the expensive compact JVP finite-Sinkhorn
route too many times for the visible gate.

## Evidence

Initial command failed immediately because the subplan used a shorthand
transport-gradient mode:

```text
manual_streaming_finite
```

The active CLI choice is:

```text
manual_streaming_finite_sinkhorn_stopped_scale_keys
```

This was repaired in the subplan. A dispatch test was added to verify that
with `transport_ad_mode=full` the route reaches the total-VJP path, not the
historical stopped partial route.

After repair, the trusted GPU command was launched. The log showed:

- GPU device created;
- XLA initialized;
- XLA cluster compiled;
- execution entered
  `_filterflow_manual_streaming_finite_transport_value_and_jvp_total`.

The command was interrupted after a long visible window with no JSON artifact.
The traceback showed it was inside the compact JVP finite-Sinkhorn softmin loop
during the full score/FD calculation.

Log:

```text
docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-n10000-run-2026-07-07.log
```

## Prior Evidence Boundary

The older artifact:

```text
docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json
```

is useful diagnostic evidence but is not admissible for the current Phase 2
score row because it used:

```text
num_particles = 10000
time_steps = 2
```

The admitted current value artifact requires:

```text
num_particles = 10000
time_steps = 50
```

Therefore the old T=2 artifact must not be normalized into an admitted Phase 2
score artifact.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Stop the current full-run attempt and enter a fixable Phase 2 repair loop. |
| Primary criterion status | Not met: no T=50,N=10000 score artifact exists. |
| Veto diagnostic status | No schema/tape/value-target veto found in preflight; execution plan veto found because the full runner is too expensive with per-coordinate FD. |
| Main uncertainty | Whether a bounded score-only T=50,N=10000 run plus directional FD diagnostic can produce the required schema artifact. |
| Next justified action | Build a bounded full-row LGSSM score artifact writer that computes the compact score once and uses a predeclared directional FD diagnostic instead of per-coordinate FD in the main run. |
| What is not concluded | No LGSSM score admission; no score correctness failure; no HMC/posterior/scientific/runtime claim. |

## Stop/Repair Boundary

Do not rerun the same full command unchanged.

Do not admit the T=2 artifact.

The next repair must preserve:

- same admitted value artifact;
- T=50,N=10000 full row identity;
- compact no-tape score route;
- Phase 1 schema validation;
- explicit correctness diagnostic;
- memory gate.
