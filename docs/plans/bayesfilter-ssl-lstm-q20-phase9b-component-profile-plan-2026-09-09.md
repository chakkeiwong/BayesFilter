# Phase 0 cost localization after complete runtime measurement

Date: 2026-09-09, Asia/Shanghai. Parent allocation: the owner's September 9
36,000 aggregate GPU-second recovery/runtime campaign. No new allocation.

Both real interruption canaries and all four 500-transition health checks
passed. Factor's six-chunk forecast is 9,305.42 seconds, or 10,751.04 with one
extra steady chunk and cleanup grace. Strict's forecast is 14,040.61 seconds,
or 16,202.12 with the same reserve. Strict therefore fails the predeclared
14,400-second reserve-inclusive affordability test. Do not discard the reserve
or reinterpret two healthy warmup chunks as posterior convergence.

## Question and skeptical audit

Which component is worth optimizing without changing the target or diagnostics?
Inspect the unchanged, public-verifier-restored chart and handoff from each arm.
Profile the transport, base physical target value/score, transformed target
value/score, and sampled-state status evaluator at the saved runtime initial
state and terminal state. Each bank is the unchanged four-chain, four-coordinate
float64 shape. Do not train, tune, resample, or change the Metropolis kernel.

The actual sequential-controller comparator is the completed pair of 500-step
calls, not the eight-step recovery fixture. Its mean per-transition times are
descriptive references only. Isolated subcomponent times are not an additive
accounting of a fused XLA program: compilation, elimination, caching, dependency
reuse, and kernel launches differ. They can nominate an optimization, not prove
its speedup or justify skipping status checks.

The source inspection finds that the sampled-state telemetry entrypoint calls
the base bridge's value/score/status function again. Whether XLA eliminates
enough work for this to be cheap is unmeasured; this is a hypothesis, not the
bottleneck conclusion. The existing exact-gradient, batched TensorFlow/XLA
implementation remains unchanged.

## Evidence and defaults

- Primary pass: both banks and all four components emit finite, synchronized
  timings, one stable explicit-signature XLA trace per component, complete
  status checks, verified startup memory growth, source and input hashes.
- Numerical veto: nonfinite values/scores/log determinant, invalid target
  telemetry, corrupt checkpoint/hand-off, or source drift. Missing timing or
  incompatible XLA prevents that component's timing claim.
- Continuation veto: resource-policy failure, committed corruption, or exhausted
  campaign budget. Failed infrastructure consumes budget and may be repaired.
- Explanatory: first-call compilation-inclusive time and four warmed calls per
  component per bank. Four repetitions and two banks are convenience-chosen
  bounded localization, not uncertainty support for a backend ranking.
- Comparator: same exact chart and saved bank for components within each arm;
  the completed controller timings anchor the order of magnitude. Different
  charts/devices across arms preclude an algorithmic ranking.
- Not concluded: posterior validity, speedup superiority, new default, or that
  component measurements make the strict full schedule affordable.

Each component first runs on the initial bank, then four warmed repeats on
each bank. Check repeated outputs for equality at the tensor level; no invented
numerical tolerance is needed for fixed-input repeatability. No pfor or scalar
fallback. The transport measurement includes forward map and log determinant.
Synchronize by materializing numerical outputs; keep provenance and file I/O
outside each pure compiled function.

## Budget and execution

The campaign has consumed 11,162.859154677019 seconds and has
24,837.14084532298 seconds left, with zero active reservations. Reserve at most
**600 seconds per worker / 1,200 aggregate** for this localization. This is a
convenience safety ceiling for a few dozen evaluations, bounded far below the
owner's 14,400-second per-arm cap; unused time is released. A 30-second inherited
cleanup grace is included in each worker deadline. An external `timeout` bounds
each child even if the coordinator dies. Parent-only accounting, preserved
successful siblings, and the existing coordinator lock remain in use.

Use only eligible non-display GPUs unless none qualifies, with the existing
40% utilization admission, 4 GiB estimated peak plus 5 GiB headroom, five-second
headroom monitoring, and verified pre-initialization TensorFlow memory growth.
Use original arm UUIDs for this same-device diagnostic. Busy devices queue or
defer; they do not authorize terminating unrelated processes.

Command (trusted GPU execution required):

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/profile_ssl_lstm_q20_phase9b_components_2026_09_09.py \
--campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z
```

The script creates a unique `launches/component-profile-*` root inside that
campaign. Manifest records the exact command, Git, Python/TF/TFP, current source
closure plus sidecar script/plan hashes, inputs, handoffs, GPU UUID, allocator,
per-component receipts, full-runtime comparator, and summed child lifetimes.
It does not alter the already completed runtime manifest or source identities.
Findings are appended to the September 9 recovery/runtime result note.

Pre-mortem: fast isolated status evaluation may not reflect its in-chain cost;
fixed warm inputs underrepresent the chain's state distribution; random seed
selection or a changed chart would invalidate the comparator. Retain all banks,
use both endpoints, and label findings as localization only. If profiling finds
no actionable source-equivalent repair, report that limitation instead of
spending the rest on repeated timing or silently changing the scientific method.

Prelaunch checks: 16 focused CPU receipt/route-policy tests pass (0.50 seconds);
compilation and whitespace checks pass. The component script is separate from
the frozen runtime source closure and checks both that closure and its own
script/plan hashes. Existing setup commits must be present before restore;
profiling cannot implicitly fall back to fresh chart construction or tuning.

## Localized telemetry-schema repair

The first profile attempt failed in the new inspector, after recording three
components per arm. Its blanket finiteness check rejected positive infinity in
`min_innovation_eigen_gap`. Source inspection establishes this is the declared
empty-pair sentinel, not a nonfinite target, score, or covariance:
`experimental_batched_svd_sigma_point_tf.py::_batched_min_eigen_gap` returns
positive infinity for eigenvalue dimension <=1 (line 839), and the q=20 model's
observation dimension is one (`ssl_lstm_complexity_target_tf.py:90`). The same
field is present in both complete runtime traces. Their required target,
log-acceptance, energy, eigenvalue, and status checks remain valid.

Repair only the diagnostic inspector: allow positive infinity only for that
named gap field and only after reading observation dimension one from the
restored target. NaN, negative infinity, nonfinite score/value/eigenvalue, and
any nonfinite multidimensional gap still veto. Preserve the raw telemetry;
record the sentinel semantics. Six focused cases test this exact boundary.
No numerical-library or runtime-controller source changes. The original script
and plan are preserved under `preflight/component-profile-r1-*` and the failed
worker outputs remain intact. The failed attempt used 86.17996163599263
aggregate seconds; retry once under the same 600-second-per-worker ceiling
and original campaign ledger. This is a schema-inspection repair, not a
relaxation of finite target/score/energy or full-controller health criteria.
