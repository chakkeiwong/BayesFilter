# TF32 Performance Decision for the Score Transfer Route

Date: 2026-07-30
Status: executed; 11.0% median time reduction, 20% speed condition not met
Classification: engineering timing diagnostic; no scientific-score claim

## Question And Evidence Contract

Is TF32 materially faster than FP32-no-TF32 on the exact `T=2`, `N=4096`,
`K=2048` canonical LGSSM score route, enough to justify selecting it despite
the measured systematic displacement?

- Candidate/comparator: TF32 versus FP32-no-TF32 with identical source,
  prepared inputs, four seeds, batch shape, controls, GPU, XLA, and memory
  policy.
- Primary criterion: TF32 median warm execution time is at least 20% lower than
  the no-TF32 median. This is a pre-run engineering definition of "a lot
  faster," not a scientific constant.
- Diagnostics: five warm repetitions per arm, individual times, median, mean,
  min/max, throughput ratio, allocator peak, and route validity.
- Vetoes: either arm fails route validity; identity differs; fewer than five
  warm repetitions complete; or timing includes tracing/compilation.
- Order control: run FP32-no-TF32 first and TF32 second, reversing the earlier
  precision-run order. A remaining process-startup/thermal difference is a
  limitation, not a promotion claim.
- Nonclaims: timing does not establish HMC correctness. The current TFP HMC
  wrapper obtains both target value and score from the same `log_prob_and_grad`
  call, so MH corrects integration error relative to that finite target but
  does not independently replace a TF32 target value with a higher-precision
  acceptance energy.

## Skeptical Audit

- The existing timing used one warm replay in each of only two batches and is
  too weak for a performance decision.
- Compilation is excluded from the primary metric.
- The test uses one fixed four-seed batch to keep runtime bounded and the graph
  identical; it supports this throughput scope only.
- A result below 20% does not prove TF32 has no benefit elsewhere. It means the
  stated "a lot faster" condition is unsupported here.

Compute budget: two GPU processes, five warm repetitions each, under five
minutes total. Artifact root:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_timing_attempt01/`.

## Result

Both arms passed all route-validity checks. Post-compilation warm execution
times were:

| Arm | Five repetitions (seconds) | Median | Mean |
|---|---|---:|---:|
| FP32-no-TF32 | 3.9645, 4.0319, 4.2473, 4.2550, 3.9455 | 4.0319 | 4.0888 |
| TF32 | 3.5100, 3.5465, 3.6758, 3.7345, 3.5878 | 3.5878 | 3.6109 |

TF32 reduced median time by 11.0% and increased median throughput by 12.4%.
This is a real speed benefit, but it does not meet the predeclared 20% meaning
of "a lot faster" for this decision. The speed condition is therefore not met.

The existing owner directive still makes GPU/FP32/TF32 the production target
direction. This timing result does not reverse that direction, but it is not
enough to waive the moment-teacher precision gate. A future HMC design may use
TF32 proposal forces with a separately evaluated higher-precision acceptance
energy, but the current wrapper does not implement that separation.

Owner decision: select FP32-no-TF32 for the Zhao-Cui moment-teacher lane. This
is a lane-specific reviewed exception based on the parity, systematic-
displacement, and timing evidence. It does not change the repository-wide TF32
direction for other routes.

Artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_score_mcse_transfer_20260730/n4096_timing_attempt01/aggregate/result.json`.
