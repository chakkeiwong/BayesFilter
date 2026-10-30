# KDM source neural consumer result — 6 October 2026

The TensorFlow MDPF and A-MDPF replay consumers now execute on GPU with XLA and
one trace each. All 880 float64 checks pass. Float32 passes 878 of 880:
the two failures are the same angle coordinate, once in each variant.
All checked losses, raw and clipped trainable gradients, Adam updates and
optimizer moments pass in both dtypes. The two-update tests cover all 31 MDPF
and 38 A-MDPF trainable parameter tensors; all six initializer tensors remain
frozen. This closes the missing *neural replay call chain*, not full training.

The comparator is the actual pinned author's neural filter, sparse full-sequence
loss and truncated-BPTT consumer, with its global gradient clipping and Adam
optimizers. The source checkpoint is a verified historical MDPF-trained
checkpoint. The A-MDPF case exercises the actual source conversion/warm start;
it is not an independently trained A-MDPF model. Batch size 2, 25 particles,
17 time steps, two updates and four development trajectories are implementation
fixtures, not stochastic qualification. Source random draws are replayed.

## Numerical finding

At time index 8, trajectory 0, particle 6, the source angle is -1.1434146166
radians and the TF angle is -1.1434551477. The absolute difference is
4.0531158e-5 radians, approximately 0.00232 degrees. Its scaled discrepancy is
1.23926 times the unchanged diagnostic allowance.

The source proposal's sine/cosine vector before normalization is
(-0.0310633779, 0.0141479969), with norm 0.03413355. TensorFlow differs in its
second coordinate by about -1.5199184e-6. For theta=atan2(s,c),

d theta = (c d s - s d c)/(s^2+c^2).

The local condition factor 1/sqrt(s^2+c^2) is 29.29669. The first-order prediction
is -4.0523417e-5 radians versus the observed -4.0531158e-5. This explains the
amplification at the transformation. It does not identify which upstream
floating-point operation first introduced the small vector discrepancy,
justify a universal replacement tolerance, or establish training equivalence.
The same coordinate occurs in both variants because the source warm-start
parameters and first-update draw tapes coincide there.

All source losses, gradients, optimizer states and original draws remained
exactly unchanged in 108 field comparisons when observability hooks were added.
No numerical clipping, angle protection, precision upgrade or tolerance change
was introduced to make this test pass.

## Execution repairs and cost

Six attempts are preserved. Attempt 01 failed on a Python source-module name
collision. Attempt 02 found a GPU placement restriction for an integer counter.
Attempt 03 found XLA's unsupported variant FakeParam while differentiating a
conditional enclosing the I0 polynomial loop. Peeling the known first
non-resampling step removed that conditional while retaining the source's
temporal loop and BPTT boundaries. Attempt 04 completed the numerical audit.
Attempt 05's added report requested a CPU-only index operation on GPU; ordinary
host integer index decoding repaired it. Attempt 06 completed the audit and
conditioning report.

No evidence indicates a broken CUDA setup. Memory growth was verified, TF32
was disabled, and TensorFlow ran on the RTX 4080 SUPER. Each of four compiled
models traced once. Initial calls took 7.8–9.8 seconds including compilation;
second calls took 0.14–0.20 seconds. These tiny-fixture timings are descriptive,
not a forecast of full training.

Phase resource charges, including failed attempts: CPU 10.727396 wall seconds,
GPU 107.916140 wall seconds. The prior 48-hour extension is not added again.
The separate CPU graph-construction smoke is recorded as routine checking.

One historical source MDPF trial used 671.47 seconds of pretraining and
30248.57 seconds of full training (137 and 621 epochs). Its 8.59-hour
historical wall time warns against assuming that 22 author trials fit in 48 hours.
It is not a current throughput measurement, and it does not establish the
cost of the TensorFlow implementation or a minimum qualification sample size.

## Decisions

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | What is not concluded |
|---|---|---|---|---|---|
| Neural replay implemented | Actual consumer, parameter, gradient and optimizer wiring checked | No missing parameter or nonfinite value in fixtures | Short replay from one checkpoint | Add native sampling and stage/training controller | Full KDM replication |
| Float64 fixture screen passes | 880/880 checks | No fixture veto | Float64 is a reference diagnostic | Preserve as parity authority | Native float32 qualification |
| Float32 screen remains failed | 878/880; angle failures retained | Two coordinate mismatches | Upstream roundoff and scope of conditioning | Independent bounds and fresh fixtures; actual training validation | A tolerance waiver or source repair |
| Continue replication | Failures localized to infrastructure and one conditional numerical sensitivity | No campaign continuation veto | Full stochastic cost/precision unknown | Sampling, training cost and qualification design | Rejection of KDM or success of both replications |

| Inference status | Finding |
|---|---|
| Hard veto screen | No additional crash/nonfinite in completed replay; two unchanged float32 coordinate checks fail |
| Statistically supported ranking | None; this is deterministic replay |
| Descriptive-only differences | Timing, coordinate discrepancy and historical training cost |
| Default-readiness | Not established; neither replication is complete |
| Next evidence needed | Native sampling laws, full source stages, fresh multi-seed downstream no-worse qualification |

Post-run red team: replay bypasses RNG implementation and subsequent sampling
feedback, the initial full filter uses source truth-based initialization, and
two optimizer updates cannot establish long training fidelity. Correct training
dynamics on independent data could overturn or strengthen this limited result.
The source's small angle-vector norm explains this fixture but is not permission
to repair the author algorithm.

Evidence:
- [Plan](controlled-smc-kdm-neural-consumer-20261006.md).
- [Comparison](artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-20261006-06/comparison.json).
- [Manifest](artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-20261006-06/manifest.json).
- [Phase summary and charges](artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-20261006-summary-v2.json).
