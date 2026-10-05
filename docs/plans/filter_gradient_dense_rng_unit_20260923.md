# Dense initializer TensorFlow cloud qualification

Question: can the existing external ball-cloud preparation use native loops
and XLA while preserving its declared TensorFlow Philox seed mapping and
sampling formulas? This differs from the previously approved NumPy-PCG64 to
versioned-TensorFlow migration: this caller already uses TensorFlow stateless
normal/uniform draws. Reuse the repository's qualified non-XLA Philox word
conversion helpers to avoid an unnecessary new mapping, and version the helper
identity for provenance. Do not change seeds, radius, shapes or data partitions.

The independent authority is the frozen external2f386f75 RNG statements, with
seed=(seed0,seed1+1000*attempt+partition), radius seed plus100, normalized Gaussian
directions and radius*uniform**(1/dimension). Extract these exact statements
from the fixture. Original eager TensorFlow is an explicitly diagnostic reference.
Candidate uses tf.while_loop, three static row-extent branches and a fixed tensor
storage signature. Padding is only storage and never becomes active target data.

Before execution review: ordinary XLA stateless_normal/uniform changes the
seed-to-double map, so comparing different clouds would confound downstream
execution parity. Raw words and uniform doubles must match exactly; normal
transcendentals and completed ball clouds retain the existing1e-10 numerical
comparison, with maximum errors recorded. Frozen-cloud full controller checks
remain mandatory because small cloud roundoff can affect strict decisions.
Seeded full controller/actual consumer qualification remains a later gate.

Cover D1, D3 and actual current D23 shape (training68,selection46,audit46),
two attempts, changed seed/radius and return-to-original, unequal row extents,
one trace, changed-input stable HLO, padding zeros, finite/radius-bound values,
matching exact original seed order, and object/graph collection. Keep exact
repeatability checks within each backend, and compare CPU/GPU to their original
reference draws separately. No distributional superiority or RNG independence
claim follows from a few deterministic fixtures.

Use at most12 workers/2400seconds from unchanged32CPU/52GPU cumulative caps,
120s ordinary and300s large diagnostics, one registered worker at a time,
source frozen throughout each worker/matrix. CPU is reference; GPU uses growth,
existing device selector and provenance. New numbered manifests/JUnit/results
retain all inputs, seeds, source hashes, outputs, errors, graph/HLO and wall time.
Stop promotion on a source/seed/shape/HLO or numerical mismatch, localize without
new tolerances, stop execution on invalid evidence or exhausted budget. This
unit has no public or actual-DZ5 admission claim.
