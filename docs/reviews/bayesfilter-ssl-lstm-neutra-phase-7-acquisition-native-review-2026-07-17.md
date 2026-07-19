# SSL-LSTM NeuTra Phase 7 Acquisition Native Review

Date: 2026-07-17

Verdict: `AGREE_FROZEN_ACQUISITION`

## Scope

- live Phase 7 plan;
- Phase 7 harness and focused tests;
- Stage A receipt and timing derivation;
- existing retained archive primitive and tests.

## Stage A Evidence

Receipt SHA-256:
`647be960a5307d564d1777d9cee5488262f3345ac0fd46ae0a5aea05367841ef`.

All six mechanics segments passed. G/H warm continuation times were
`0.455205` and `0.454158` seconds per four draws. All four chains moved,
native divergence was unavailable rather than zero, every archive and
continuation hash bound, every XLA trace count was one, and every post-archive
value/score audit passed. Canary samples remain excluded.

## Frozen Contract Review

| Item | Disposition |
| --- | --- |
| Segment shape | `256` draws per chain; static and shared across charts |
| Burn-in | `128` only in segment 0; zero in continuations |
| Checkpoints | `256/512/1,024/2,048` draws per chain |
| Seeds | Eight fresh pairs per chart; unique and disjoint from Phase 6 |
| Fairness | Same segment size, checkpoints, maximum draws, and `1,050` second chart cap |
| Total resource | `2,100` seconds (`0.5833` GPU-hours), including compilation |
| Kernel | Frozen identity mass, `epsilon=0.8`, `L=4`; no adaptation or search |
| Diagnostics | Both chart-specific `z` and common mapped `theta`; chain-major ordering |
| Acceptance | `[0.55,0.85]` promotion veto; checkpoint extension rather than artifact failure |
| Cross-replication | Inaccessible until both independently admit; common `theta` only |
| Privacy | Public receipt contains hashes/diagnostics, never raw samples or private paths |

## Findings And Repairs

1. The first post-freeze CLI retained the old closed-acquisition guard. It was
   replaced with the exact frozen acquisition call.
2. Maximum-checkpoint non-admission could retain the text “extend.” It now
   closes as `MAXIMUM_OPPORTUNITY_EXHAUSTED_NOT_ADMITTED`.
3. Resource-cap exhaustion was initially grouped with evidence invalidity. It
   is now a separate continuation veto producing valid but incomplete
   evidence.
4. Mapped `theta` originally lacked explicit XLA trace and GPU placement
   telemetry. It now has a hard engineering audit requiring finite shape,
   trace count one, and GPU output.
5. The run manifest command omitted the interpreter. It now records the full
   executable command.

No unresolved material finding remains.

## Checks

- `21 passed` across Phase 7 and retained-archive focused tests.
- Python compile and `git diff --check` passed.
- The mocked acquisition verifies sequential G then H execution, independent
  stopping opportunities, and no partial cross-replication comparison.
- Stage A receipt hash and exclusion boundary revalidate exactly.

The authorized next action is the exact retained-acquisition command frozen in
the live plan. Passing may hand off only to Phase 8; failure classifications
must preserve candidate rejection versus research-direction rejection.
