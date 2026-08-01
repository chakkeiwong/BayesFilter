# P5 R1B Audit-Point Device Repair

Date: 2026-07-16

CPU attempt 02 and GPU attempt 02 shared the same mathematical target signature.
Their posterior values agreed within `2.84e-14`, scores within `2.10e-13`, and
all device-local gates passed. Their typed signatures differed, so R1B could not
close.

Payload comparison isolated the difference to `points_sha256`. The fixed audit
points include the inverse-normal transform of physical truth. Constructing
that tensor on CPU versus GPU produced different serialized bits even though
the numerical values were scientifically equivalent. Since the recomposition
admission correctly binds exact audit evidence, those different point hashes
produced different admission and target signatures.

The repair constructs the complete fixed 14-point audit tensor under
`tf.device("/CPU:0")` before either CPU or GPU computation. This is an evidence-
identity repair, not a tolerance relaxation. The target, posterior, source
values, audit set, FD steps, recomposition tolerances, substitution tests,
hardware class, and budget remain unchanged. CPU attempt 02 remains the
reference identity. GPU attempt 03 must reproduce its target signature exactly
and retain GPU-resident posterior outputs.
