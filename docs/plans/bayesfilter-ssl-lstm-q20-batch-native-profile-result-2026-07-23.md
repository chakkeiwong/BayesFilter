# q=20 Batch-Native CPU Profile Result

Date: 2026-07-23  
Status: `BOTTLENECK_CLASSIFIED`

## Result

The CPU bottleneck is steady-state batch-native UKF value/score evaluation,
not worker startup, graph tracing, serialization, or optimizer work.

| Phase | Measurement |
| --- | ---: |
| Parent target construction | recorded in progress artifact |
| Four-worker pool startup | 2.303 s |
| Maximum process-tree threads | 45, below the 50-thread cap |
| Maximum combined RSS | 5.41 GB, below the 64-GiB cap |
| Size-25 first call | 28.804 s |
| Size-25 repeated calls | 26.806-27.823 s |
| Size-25 repeated mean | 27.256 s |
| Parent serialization/aggregation overhead | about 0.005-0.009 s per call |

Small-shard first calls show one-time tracing cost, but the size-25 repeated
cost remains essentially unchanged. Therefore graph compilation is not the
dominant explanation for `r4`.

## Runtime Implication

With four workers processing a batch of 100 as four 25-row shards, the measured
steady-state rate implies roughly:

- 6,814 seconds for 250 optimizer updates;
- 54,512 seconds, about 15.1 hours, for one 2,000-update stream;
- 109,023 seconds, about 30.3 hours, for two streams.

These are descriptive extrapolations, not campaign guarantees. They explain
why the authorized 13,500-second cap could not complete the CPU training lane.

## Decision

The prior CPU training result is a performance continuation veto for this
implementation route. The target is mathematically batch-native and passed
scalar parity separately, but this CPU execution is not a practical training
backend under the current budget. Do not replace it with the scalar worker
route: that route violates the batch-native training policy.

## Next Action

Optimize or redesign the q=20 batch-native UKF target before another CPU
campaign. Candidate work includes profiling the batched principal-square-root
and time-recursion kernels, reducing repeated model/derivative construction,
and testing the GPU/XLA route when a GPU is available. No HMC or posterior claim
is supported by this profile.

## Evidence

- [Profile result](artifacts/ssl-lstm-q20-batch-native-profile-2026-07-23/r1/result.json)
- [Per-call progress](artifacts/ssl-lstm-q20-batch-native-profile-2026-07-23/r1/progress.json)
- [Profile plan](bayesfilter-ssl-lstm-q20-batch-native-profile-plan-2026-07-23.md)
