# GenUT reduction capacity result

The repaired reduced GenUT correction completes the tested N*d*d workload at
N=10,000 and d=18 on CPU and GPU. Candidate output is finite, valid, replay
exactly, respects the radial cap, and passes the independent moment checks in
every native arm. This is a bounded capacity result for the reduced primal
owner. It does not qualify full LEDH reset memory, analytical scores, target
models, or canonical admission.

| Device | Arm | Mode | N,d | RSS after compile MiB | Median warm ms | TensorFlow allocator peak |
| --- | --- | --- | --- | ---: | ---: | ---: |
| CPU | native | graph | 1,000,3 | 605.5 | 7.85 | N/A |
| CPU | native | XLA | 1,000,3 | 802.6 | 1.97 | N/A |
| CPU | native | graph | 10,000,3 | 608.2 | 63.17 | N/A |
| CPU | native | XLA | 10,000,3 | 804.2 | 7.57 | N/A |
| CPU | native | graph | 10,000,18 | 634.9 | 203.01 | N/A |
| CPU | native | XLA | 10,000,18 | 899.9 | 76.78 | N/A |
| GPU | native | graph | 1,000,3 | 1,113.9 | 6.74 | 175,872 bytes |
| GPU | native | XLA | 1,000,3 | 1,028.6 | 1.32 | 89,088 bytes |
| GPU | native | graph | 10,000,3 | 1,113.7 | 6.50 | 2,057,472 bytes |
| GPU | native | XLA | 10,000,3 | 1,030.7 | 1.56 | 1,015,296 bytes |
| GPU | native | graph | 10,000,18 | 1,107.7 | 7.24 | 49,337,344 bytes |
| GPU | native | XLA | 10,000,18 | 1,034.1 | 1.97 | 6,291,456 bytes |

GPU UUID was held constant and sampled monitoring found no foreign compute
process. Memory growth was configured before device initialization. Allocator
values exclude CUDA context and compiler memory; host RSS includes compiler and
returned arrays. The GPU graph-to-XLA peak difference at N=10,000,d=18 is about
41.05 MiB. Repeated constructor/executable retention was not tested.

Original graph/XLA and repaired graph/XLA were run on the same frozen FP32
inputs. The strict complete-record comparison remains separate: the known
cap-active fraction differs across graph/XLA. Independent FP64 runs on the
exact saved FP32 inputs (04135--04137) show that several additional original
GPU records fail the unchanged 2e-5 fields, while native non-report fields pass
against the FP64 reference. Those failures belong to the original comparator,
not evidence that the repair should be weakened. No tolerance was changed.

A matched CPU d=18 XLA timing probe (04139) interleaves both owners in one
process for 20 paired replays. Median warm time is 42.0 ms for original XLA and
134.5 ms for native XLA (ratio 3.21); native HLO is 414,758 bytes versus
1,017,572 bytes for original. This is descriptive evidence from one controlled
fixture, not a statistically supported speed ranking. The native implementation
has a material CPU d=18 performance limitation that remains an engineering
repair item.

The cap diagnostic is recorded separately in
`filter_gradient_genut_cap_diagnostic_result_20260926.md`: XLA's power/multiply
rewrite causes the one-coordinate threshold crossing, and an optimization
barrier reproduces graph rounding. The barrier is diagnostic only and was not
installed in runtime code.

| Decision | Status | Meaning |
| --- | --- | --- |
| Reduced-primal capacity at tested sizes | Qualified for native finite/moment/replay scope | Candidate completes N=10,000,d=18 on CPU/GPU |
| Target-scale/full LEDH capacity | Open | Full reset, transport and score consumers remain untested |
| Complete graph/XLA report equivalence | Open | One thresholded cap report remains mode-dependent |
| CPU d=18 performance | Open limitation | Native XLA is descriptively slower in matched probe |
| Main merge/terminal master gate | Blocked | F01--F20 and public integrations remain open |

Raw run records through 04140, source/HLO evidence, full numerical arrays and
analyzers are preserved in `genut-capacity-evidence-04140-r2.tar.gz` with receipt
`genut-capacity-verification-04140-r2.json`. The earlier compact r1 archive used
an overly narrow filename pattern that omitted per-run capacity JSON/NPZ files;
r2 repairs the archival omission and preserves r1. The campaign policy suite passes
again in 04140 with 270 guarded sources and 1,424 exact allowances.

Subsequent reduction-layout and highest-precision dot diagnostics are recorded
in `filter_gradient_genut_reduction_layout_result_20260926.md`; neither changes
the runtime or closes the CPU cost finding.
