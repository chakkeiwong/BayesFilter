# PP-UKF Tuning Runtime Repair Reset Memo

Date: 2026-07-21

## State

The operational windowed warmup XLA route is validated for focused mechanics
under
`docs/plans/artifacts/bayesfilter-operational-windowed-warmup-xla-route-validation-20260721/`.
The former `operational_windowed_warmup_xla_not_validated` route blocker is
resolved.

The first post-validation PP-UKF tuning-only retry (`...-02`) stalled during
the initial reasonable-epsilon search. A trusted-GPU diagnostic measured
finite eager PP-UKF execution at approximately 20 seconds per target call and
42 seconds per one-proposal HMC call. The real target therefore made repeated
eager epsilon probing an infrastructure/runtime bottleneck, not a scientific
rejection.

## Repair

`find_reasonable_epsilon` now accepts `jit_compile=False` by default. The
operational warmup route passes its existing XLA setting into both initial and
metric-boundary epsilon searches. When enabled, bootstrap and proposal calls
are compiled with TensorFlow XLA; eager callers retain the old behavior.

Focused regression checks passed: `89 passed, 2 skipped`. Trusted PP-UKF XLA
probes completed with finite values and finite proposals; the one-attempt,
one-probe exact epsilon search completed with compiled bootstrap and proposal
times of roughly 3.9 and 5.3 seconds after compilation.

## Retry Outcome

The fresh same-contract retry at
`docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-03/`
completed bootstrap and entered `windowed_mass_operational_warmup_start`, but
the runner process disappeared before a segment boundary or terminal artifact
was written. Its `interruption_record.json` and `run_state.json` classify the
attempt as interrupted. There is no tuning handoff, `result.json`,
`run_manifest.json`, or sampling artifact.

## Nonclaims And Next Gate

This section is superseded by the later completed broad-grid repair. The
runtime blocker was resolved sufficiently to complete bounded GPU/XLA tuning.
The current classification policy is
`replication_mean_t90_band_compatibility_v1`: use three independently seeded
replication means as uncertainty units and reject only when the 90% Student-t
working interval is wholly outside `[0.65,0.75]`. Do not revive strict interval
containment or twelve-chain-mean pseudo-replication.

Current terminal result:

- `L=3` is statistically below the band;
- primaries `L=(5,9,13,18,25)` are statistically compatible;
- exact-epsilon guards reject local suitability for `L=5` and `L=9`;
- all admissible guards pass for `L=(13,18,25)`, which are the locally suitable
  tuning candidates; and
- no ranking, posterior convergence, retained-sampling, default-readiness, or
  scientific claim follows.

The complete result is
`docs/plans/bayesfilter-pp-ukf-statistical-compatibility-and-guard-repair-result-2026-07-21.md`.
The unchanged four-hour campaign closed at `13,750.560450 s`, leaving
`649.439550 s`; no sampling was launched. The next justified action, if
authorized under a fresh plan and budget, is frozen-kernel validation of
`L=(13,18,25)` with convergence and posterior/reference gates declared before
execution.
