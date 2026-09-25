# LEDH safety and analytical stage execution repair

The shared Cholesky safety helper and the LGSSM analytical stage slice now run
through native TensorFlow control flow. The Cholesky helper no longer expands
rank with a Python loop; it broadcasts the validity mask directly, preserves
the historical squeezed shape, and rejects every nonfinite factor, including
infinity. A finite factor is only a factor-validity result; this helper does
not certify conditioning or an error bound.

The flow and multi-step analytical stage routines use `tf.while_loop` for their
substep and time recurrences. The stage-fraction schedule is formed from the
same Python-double quotient before casting to the selected dtype. The original
determinant operation was not XLA-compatible on TensorFlow 2.19.1 CPU
(`MatrixDeterminant` has no XLA CPU kernel); the repaired stage uses the
repository’s existing native `determinant_tf` authority, preserving the
determinant-product-then-log finite program. The frozen original remains a
graph comparator for that arm and is not silently advertised as an XLA
reference.

Tests 03987, 03991--03997 and 04004 pass the shared-helper shape/rejection,
FP64/FP32 stage, analytical finite-difference, changed-input, exact-replay,
consumer, HLO and policy checks on CPU and GPU 2. The repaired stage traces
once and its graphs contain native `While` control and no Python callbacks.
The exact-source guard now covers 253 modules with 1,359 exact allowances and
reports no violations or stale exceptions. Five fixed configuration-time
allowances cover three TensorSpec-schema comprehensions, the higher-moment
optional-field check and the mandated transport chunk selector; they do not
waive numerical loops.

| Arm | Device | Cold seconds | Warm median ms | RSS after compile / replay, MiB |
| --- | --- | ---: | ---: | ---: |
| frozen original graph | CPU | 1.143 | 3.642 | 629 / 629 |
| repaired graph | CPU | 0.411 | 6.634 | 607 / 607 |
| repaired XLA | CPU | 0.824 | 0.903 | 787 / 787 |
| frozen original graph | GPU 2 | 2.081 | 10.070 | 1090 / 1090 |
| repaired graph | GPU 2 | 2.376 | 53.679 | 1093 / 1093 |
| repaired XLA | GPU 2 | 1.493 | 4.736 | 1053 / 1053 |

These are single-process descriptive diagnostics from 20 replays per arm. They
do not support a statistical speed ranking or a target-scale performance claim.
GPU runs used the managed-session trust basis, GPU 2, TF32 enabled, XLA where
labelled, and verified memory growth. The original graph arm is a frozen
implementation comparator; its XLA incompatibility remains recorded.

The static import discovery from the KDM endpoint reaches 80 repository modules;
17 conditional/reference or contract modules remain outside the exact guard.
The direct safety/stage closure is guarded and tested, but this overapproximate
import graph is not proof that every optional filtering route is policy-clean.
Candidate-only GenUT, latent-SIR, transport-diagnostic and metadata modules
remain migration/audit debt in the master ledger. No canonical LEDH rebuild,
score admission, HMC promotion or main merge follows from this execution repair.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Retain safety/stage repair | CPU/GPU records, derivatives, HLO and consumers pass | No repair veto | Broader optional call chains | Continue targeted closure audit | Fixture and stage-slice bounded |
| Retain native determinant authority | Original XLA incompatibility reproduced; repaired XLA passes | No numerical mismatch | Wider dtype/shape coverage | Extend only with a new bounded plan | No tolerance or ridge change |
| Preserve cost evidence | All replay and provenance checks pass | No artifact veto | Single descriptive cohort | Keep costs diagnostic | No ranking or production-capacity claim |

The original cross-mode precision disposition, repeated-construction XLA native
memory retention, DZ5 graph replay/GPU graph finite-difference failures,
external callback autodiff/pfor debt, public LEDH integration and reset
qualification, initializer/supervisor work, reporting/isotropic cases, target
capacity and F01--F20 terminal dispositions remain open.
