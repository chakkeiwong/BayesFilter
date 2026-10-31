# KDM precision and construction-memory findings

The repaired XLA reset agrees exactly with the original XLA reset and is within
the unchanged `atol=rtol=1e-6` tolerance of the FP64 reference on the tested
rounded inputs. The frozen eager comparator exceeds that tolerance. This is a
comparator-accuracy finding, not severe ill-conditioning or a loop-conversion
regression. Original cross-mode failures remain preserved and the terminal
gate has not been waived.

03957--03959 compare the same FP32-rounded operands against the existing
analytical program evaluated in FP64. 03960 independently reconstructs the FP64
primal with NumPy diagnostic algebra and checks its tangents against two central
finite-difference steps (`1e-5`, `2e-5`). Primal tolerance is `2e-11`; derivative
tolerance remains `atol=4e-8, rtol=4e-6`. Both pass. The independent computation
preserves the existing cast-from-Python-float constants, avoiding an unnoticed
finite-program change.

| Device/precision | Eager error | Original graph error | Original/repaired XLA error |
| --- | ---: | ---: | ---: |
| CPU FP32 | 1.02476 | 0.61510 | 0.32127 |
| GPU FP32 with TF32 enabled | 180.98297 | 180.84233 | 0.06172 |
| GPU FP32 with TF32 disabled (reference only) | 1.55025 | 1.43321 | 0.06172 |

Errors are maximum units of the unchanged tolerance against the FP64 reference,
so values above one fail. The CPU original-XLA/eager comparison is 1.34603 units.
Disabling TF32 removes most of the GPU comparator discrepancy; the residual
FP32 rounding error is still measurable. Factor condition numbers are about
60, 76 and 153; FP32 Cholesky relative residuals are about `3.3e-8` to `3.5e-8`
on CPU. These diagnostics do not establish severe ill-conditioning or justify
an error rejection for this otherwise valid input. No runtime dtype, TF32
setting, ridge, tolerance or mathematical algorithm changed.

The construction-memory result remains a repair trigger. Runs 03961--03970
build one/three/six graph or XLA owners in fresh processes, release outputs and
owners, and run three garbage collections after every build. Repeated numerical
outputs remain identical. From the second owner onward, Python graphs are
collected, while the TensorFlow registered-function count stays at 23.

| Device | Mode | RSS after first owner | RSS after sixth owner | Approximate growth per late construction |
| --- | --- | ---: | ---: | ---: |
| CPU | graph | 764 MiB | 834 MiB | near zero |
| CPU | XLA | 1204 MiB | 2556 MiB | 258 MiB |
| GPU 2 | graph | 1301 MiB | 1363 MiB | near zero |
| GPU 2 | XLA | 1351 MiB | 2472 MiB | 201 MiB |

GPU current allocation after output release is 7,168 bytes in both modes;
XLA peak is about 67 KiB. The increase is native host retention associated with
repeated compilation, not growing live GPU tensors or accumulating registered
Python graphs. The exact native cache or allocator owner is not yet identified.
Single retained-owner runs stay stable over 20 exact replays, so the existing
explicit factory avoids repeated compilation for fixed callbacks/configuration.
It cannot silently replace a public call whose Python callback cells changed.

03971 independently recomputes all saved precision errors and construction
slopes. These are deterministic diagnostics and bounded memory observations;
there is no statistical performance ranking, target-capacity qualification,
LEDH admission or HMC promotion.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
| --- | --- | --- | --- | --- | --- |
| Preserve the native execution repair | Matching-mode equality and independent FP64/FD checks pass | Original cross-mode gate remains open | Broader dtype/target coverage | Replace inaccurate reference only through an explicit terminal disposition | No tolerance change inferred |
| Keep construction memory open | Six-owner growth repeats on CPU/GPU after graph collection | Resource regression persists | Native compiler cache versus allocator ownership | Bounded native-memory attribution and reuse/lifetime repair | One reusable owner is stable, not proof of unrestricted construction safety |

Red-team review: the precision conclusion is fixture-bounded and requires the
same rounded inputs. NumPy is confined to an independent test authority; it
does not become a runtime backend. Graph collection cannot certify native
memory eviction, and device allocator bytes cannot explain host RSS. The
existing master remains unmerged and incomplete.


Allocator attribution 03973/03974 does not remove the XLA slope. Explicit
`malloc_trim(0)` returns about 140--160 MiB per XLA round, but post-trim RSS
still grows from 1069 to 2428 MiB across six builds. This is not a runtime fix.
The next plan tests native allocation categories and nested compilation
ownership under a new bounded suballocation of the unchanged global budget.
The latest 129-check policy suite passes (03975); Ruff and whitespace checks
pass. No numerical runtime changed after the pushed `6000ae63a` repair.

The follow-up consumed 267.818416 CPU / 179.780318 GPU seconds. Its 13 CPU
workers include the 12-worker diagnostic allocation and the final 8.633437-second
policy check; all charges remain inside the 2400-second local CPU ceiling and
global caps. Six GPU workers used the declared 1200-second local ceiling.
All raw files/source supplements are preserved and byte-verified in
`kdm-followup-evidence-03975.tar.gz` with `kdm-followup-verification-03975.json`.
No numerical worker is active at this checkpoint.
