# Corrected GenUT gradient cost and memory cohort

The corrected cohort records complete outputs and valid FP64 comparisons for
the installed loop-bound repair. It does not qualify the FP32 gradient or make
a speed claim: the independent comparison fails weight gradients at every
extent/backend arm, and the known coordinate-cap report differs between graph
and XLA. The loop-bound repair itself is active; the precision alternatives
remain uninstalled.

The pilot 04217--04242 is superseded. Its objective was accidentally generated
as `cos(index)` and regenerated in another precision, and its JSON omitted
materialized outputs. Those runs remain preserved but cannot support numerical
or timing interpretation. The corrected contract uses one FP32 coefficient
tensor `cos(index*.11)` as an explicit operand for every arm, casts those bytes
for the FP64 reference, saves all return fields and gradients to compressed
NPZ, hashes the arrays, and compares them independently after the workers exit.

Runs 04245/04260 pass the adverse and saved-cohort analyzers. FP64 references
04246/04247 pass exact replay and centered finite differences at steps `1e-4`
and `5e-5` for source, weight and reset directions. Cost arms 04248--04259
pass finite, valid, exact-replay, one-trace, output-hash and owner-collection
checks. GPU manifests select the same physical UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, show memory growth and no sampled
foreign process. Policy run 04261 passes 129 checks after the final analyzer additions;
the guard remains 270 sources and 1,424 exact allowances. Analyzer renewal
04260 validates the corrected cohort.

| Device | N,d | Arm | Cold seconds | Median warm ms | RSS after replay | HLO bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CPU | 1,000,3 | before graph | 1.7205 | 10.593 | 709.7 MiB | — |
| CPU | 1,000,3 | after graph | 1.6979 | 10.479 | 710.3 MiB | — |
| CPU | 1,000,3 | after XLA | 2.6444 | 4.477 | 996.8 MiB | 1,606,765 |
| CPU | 10,000,18 | before graph | 2.4101 | 653.026 | 854.5 MiB | — |
| CPU | 10,000,18 | after graph | 2.5768 | 848.622 | 832.2 MiB | — |
| CPU | 10,000,18 | after XLA | 3.6094 | 292.093 | 1,124.7 MiB | 1,740,962 |
| GPU | 1,000,3 | before graph | 3.9089 | 17.208 | 1,276.98 MiB | — |
| GPU | 1,000,3 | after graph | 3.8985 | 18.438 | 1,285.13 MiB | — |
| GPU | 1,000,3 | after XLA | 7.4393 | 2.782 | 1,402.05 MiB | 1,461,058 |
| GPU | 10,000,18 | before graph | 3.9176 | 18.785 | 1,282.96 MiB | — |
| GPU | 10,000,18 | after graph | 3.9782 | 19.274 | 1,281.48 MiB | — |
| GPU | 10,000,18 | after XLA | 9.8615 | 4.330 | 1,533.96 MiB | 1,601,933 |

The graph outputs are bitwise identical before and after the loop-bound repair at
all four device/extent pairs. That isolates the loop bound as an XLA reverse
compilation repair rather than the source of the graph FP32 error. XLA warm
times are descriptively lower in this cohort, but no ranking is admitted because
the smooth derivative screen fails. RSS after HLO and owner collection includes
compiler/native allocations; it is not a leak diagnosis. GPU allocator current
and peak bytes are recorded separately in each JSON, and sampled sharing is
limited to the monitor's documented observation window.

| Gate | Status | Meaning |
| --- | --- | --- |
| Independent FP64 reference and finite differences | Pass | Reference is valid for the frozen coefficient/input contract |
| Finite/replay/trace/output-hash/owner release | Pass | Cost artifacts are internally complete |
| Graph before/after bitwise identity | Pass | Loop-bound repair preserves graph values and gradients |
| Smooth FP32 values/gradients vs FP64 | Fail | Weight gradients fail at both extents on CPU/GPU; GPU XLA also has source/reset failures |
| Complete report equivalence | Fail | Coordinate-cap fraction remains mode-dependent |
| Speed or memory promotion | Blocked | Numerical veto and compiler allocation ambiguity remain |

The result is descriptive engineering evidence only. It does not establish
ill-conditioning, posterior quality, target-scale LEDH capacity, canonical
analytical scores, HMC readiness or main-branch readiness. Raw runs, NPZ outputs,
HLO files and the analyzer receipt are under the shared artifact root; the
analysis receipt is `genut-bounded-gradient-cost-analysis-04259.json`, with
the registered renewal manifest in run 04260.
