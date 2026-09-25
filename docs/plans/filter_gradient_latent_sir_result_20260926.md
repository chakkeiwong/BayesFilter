# Latent SIR execution repair result

The fixed-noise latent SIR simulator now uses one retained TensorFlow program
per fixed model/horizon/JIT configuration. Its numerical time recurrence is a
native `tf.while_loop`, and the default public route is `jit_compile=True`.
The shared `SpatialSIRSSM` RK4/RHS authority accepts tensor rate operands, so
the simulator does not construct a parameter-dependent Python model inside its
execution path. The initial state remains unclipped; susceptible coordinates
are clipped only after later process-noise draws. Supplied noise, covariance
orientation, parameter chart, and rejection behavior are preserved.

The enclosing pullback helper also now excludes boolean/integer status leaves
from numeric `grad_ys`. This preserves the complete pullback of all numeric
outputs while allowing the simulator's explicit validity flag to remain a
fail-closed graph output. The fix is schema binding only; it adds no numerical
loop, NumPy path, or pfor route.

## Evidence

| Gate | Runs | Result |
| --- | --- | --- |
| Fixed-noise path parity, J=1/2/9 and T=0/1/3 | 04038--04040 CPU, 04046--04048 GPU 3 | Pass |
| Shape, invalid-parameter and original-XLA boundary checks | 04042 CPU, 04049 GPU 3 | Pass |
| Simulator pullback vs frozen autodiff and finite differences (diagnostic) | 04042 CPU, 04049 GPU 3 | Pass |
| Existing SIR consumers and seeded source-law checks | 04045 CPU, 04050 GPU 3 | Pass |
| Fresh original/graph/XLA cost arms | 04051--04056 | Pass |
| Campaign, GPU-selection, cost-provenance and source-policy checks | 04057--04058 | Pass |

The legacy source-law consumer originally required bitwise equality. The graph
arm is bitwise equal; XLA fusion changes FP64 values by at most
`5.551115123125783e-17` for observations and `3.469446951953614e-18` for the
physical path. The consumer now uses the plan's fixed `5e-12` absolute and
relative bound. Exact replay of each owner remains an independent exact gate.
No implementation tolerance or algorithmic correction was added.

Fresh J9/T3 fixture costs are descriptive and single-process:

| Device | Arm | Cold seconds | Mean warm ms | RSS after compile MiB |
| --- | --- | ---: | ---: | ---: |
| CPU | original eager | 0.145 | 129.299 | 588.5 |
| CPU | repaired graph | 0.340 | 4.184 | 605.4 |
| CPU | repaired XLA | 0.755 | 2.570 | 770.8 |
| GPU 3 | original eager | 1.044 | 197.413 | 1038.8 |
| GPU 3 | repaired graph | 1.737 | 16.014 | 1057.9 |
| GPU 3 | repaired XLA | 1.168 | 2.976 | 1034.2 |

GPU 3 was selected while GPUs 0 and 2 were busy and GPU 1 carried display
activity. These costs are not an uncontended ranking. They do not qualify
target-scale capacity or repeated-constructor native retention. GPU memory
growth was configured and recorded before TensorFlow initialization in every
GPU worker.

## Decision and limits

| Decision | Status | Meaning |
| --- | --- | --- |
| Execution repair for the latent SIR simulator | Qualified for this tested scope | Numerical paths, pullback boundary, consumers and default XLA route pass |
| Source-faithful Zhao--Cui filtering claim | Not claimed | This target remains explicitly `extension_or_invention` |
| Canonical Contract E / LEDH admission | Not claimed | No canonical rebuild or analytical LEDH score admission occurred |
| Repository-wide filtering/gradient policy closure | Open | The reviewed guard covers 269 sources / 1,422 exact allowances; uncovered routes remain in the master ledger |
| Master-program terminal acceptance or merge to main | Open | KDM, DZ5, mixed KR transport, public LEDH/reset, capacity and F01--F20 dispositions remain open |

The unit charged 100.043930 CPU seconds and 100.036030 GPU seconds after the
04037 checkpoint. The campaign remains within its existing 56 CPU / 52 GPU
process-hour caps; the user's added CPU allocation was already included before
this unit.
