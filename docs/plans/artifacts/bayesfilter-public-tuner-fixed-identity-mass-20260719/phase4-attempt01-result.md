# Phase 4 Attempt 01 Result

Decision: `MARGINAL_RERUN`

The current-target preserved transport passed the canonical public tuner and
the complete sequential sampler-validity gates. The result did not pass the
one-seed truth-tail criterion because only `q2` was marginal.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Rerun one independent sampling seed | `q2 p_truth=0.0457386`, below 0.05 but above 0.003 | Tuner passed; acceptance `0.710905`; no hard veto; warm-up and retained convergence passed; no energy divergences | One 4,000-draw retained sample can place a true parameter marginally in a posterior tail by chance | Run the same frozen transport and public policy with a distinct sequential seed | No NeuTra failure or pass declaration yet |

## Sampler Evidence

- Public tuner: passed on attempt 1 after an acceptance-inconclusive repair;
  fixed identity mass signature
  `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`.
- Fresh verification acceptance: `0.7109046802`, inside `[0.65, 0.75]`.
- Warm-up: 2,000 draws per chain; final 1,000-draw window max modern R-hat
  `1.0114386437`; acceptance `0.72`; zero energy divergences.
- Retained: 1,000 draws per chain; max R-hat `1.0094533522`; minimum bulk ESS
  `1083.28598`; minimum tail ESS `1192.15611`; acceptance `0.691`; zero energy
  divergences; all target status telemetry valid.
- Truth-tail: only `q2` was below 0.05, with `p_truth=0.0457385654`; it was not
  severe under the owner's `0.003` threshold.

The run took `21812.4` seconds. A separate engineering follow-up is warranted:
the public result does not persist a replayable private kernel handoff, so the
second sampling seed must currently repeat expensive tuning rather than replay
the already admitted kernel. That cost issue does not invalidate Attempt 01.

## Attempt 02 Replication Gate

Attempt 02 keeps public tuner seed `(20260621, 8)` and changes only the
sequential warm-up and retained seeds by `seed_offset=1000`. Because retuning
is currently unavoidable, Attempt 02 counts as the requested pure
sampling-seed replication only if it reconstructs exact final-kernel hash
`e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`.
The runner must compare that hash after public tuning admission and before
sequential sampling. A mismatch must emit `TUNING_REPLAY_HASH_MISMATCH`, record
that sampling was not authorized, and must not be interpreted as replicated
truth-tail evidence.

The repair is implemented in the reusable frozen-validation path, exposed by
CLI option `--expected-tuning-final-kernel-hash`, and leaves ordinary
training-plus-validation runs unchanged. Local CPU-hidden engineering checks
passed:

- focused frozen-validation contract suite: `22 passed`;
- public tuner, fixed-mass handoff, replay, NeuTra orchestration, and public API
  regression suite: `134 passed, 1 skipped`;
- `py_compile` and `git diff --check`: passed.

These are engineering checks only. They authorize the planned GPU/XLA
replication attempt but do not provide sampling or scientific evidence.
