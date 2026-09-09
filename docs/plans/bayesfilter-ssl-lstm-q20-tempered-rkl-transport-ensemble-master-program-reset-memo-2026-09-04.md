# Master-program reset memo: q=20 GPU batching and eigensystem diagnostic

Date: 2026-09-04  
Governing master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Active subplan: `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md`  
Result: `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-result-2026-09-04.md`

## Why the program was reset

The user requested a new 10,000-second material budget and asked whether GPU
batching and reuse of the strict-path eigendecomposition could reduce the q=20
replay cost.  The historical 7,800-second replay and its partial calls remain
closed.  This memo establishes a fresh profile and artifact namespace without
changing the target, data, hardware class, or scientific question.

## State transition

`M3P_CLOSED` -> `M3Q_PERFORMANCE_DIAGNOSTIC`, with Phase 9B still blocked.

- New canary profile: `phase9a_full_replay_canary_gpu10000_v1`, cap 1,800 s,
  fresh 20260904 seed roots.
- New full profile: `phase9a_full_replay_gpu10000_v1`, cap 10,000 s, fresh
  20260904 seed roots.
- New output root:
  `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-04/`.
- The strict backend remains the replay baseline.  Raw-covariance and
  strict-factor eigensystem reuse are opt-in and unpromoted.

## Evidence recorded

- The target and four-chain HMC call are already batch-native over a static
  leading axis.  GPU batch-size measurements show lower per-row steady cost as
  batch size increases, but candidate-level grouping remains unadmitted because
  exact TFP random-stream and state semantics have not been established.
- The raw-covariance reuse backend reduces the q=20 graph's eigensolver nodes
  from six to two and is descriptively about 2.9x faster at batch 4 on GPU0,
  but its score fails the declared per-coordinate tolerance on center and
  varied rows.
- The strict-factor reuse backend reduces the graph from six to four nodes,
  is about 1.5x faster at batch 4, and passes the current center/varied GPU
  score fixtures.  Focused factor/score tests pass, including compiled
  valid/repaired/invalid rows.  Placement spectra near `1e-12` with gaps near
  `1e-15` still require a downstream sensitivity check.
- No canary or full replay was launched in this reset.  The 10,000-second cap is
  configured, not validated as sufficient.  The strict-factor follow-up
  artifacts are `gpu0-factor-cached-center-valid.json` and
  `gpu0-factor-cached-varied.json`; they pass the current local parity gate but
  do not authorize replay.

## Repair and continuation rule

The raw route is vetoed and cannot enter replay.  The strict-factor route cannot
be promoted or substituted into replay until fresh near-degenerate fixtures and
a downstream one-step/full-chain sensitivity check pass.  Until then, retain
the strict route and use only the existing batch-native chain evaluation.  The
next long command is the strict-baseline canary under the fresh profile, after
the active subplan's P1 entry and this closeout are refreshed.

All timing differences are descriptive.  They do not establish whitening,
posterior correctness, convergence, mode discovery, sampler superiority,
high-dimensional scaling, or production readiness.
