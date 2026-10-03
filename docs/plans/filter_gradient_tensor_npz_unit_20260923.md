# Tensor NPZ archive compatibility

Implement the already planned external-serialization dependency independently
of the pending dense-fit condition-report decision. The question is whether
TensorFlow tensors can be archived with the same NumPy-reader-visible member
names, array shapes, dtypes and values without importing NumPy at runtime.
The baseline is `np.savez_compressed` on identical frozen tensors. Archive
compression bytes, timestamps and header padding are not numerical evidence.

The helper uses TensorFlow only for tensor conversion and flattening, then
Python standard-library `struct` and `zipfile` for NPY-v1/ZIP serialization.
Its two loops traverse members and packed byte chunks at the artifact boundary;
they perform no filtering, fitting, score calculation or admission decision.
They therefore receive exact host-reporting allow-list entries, without a new
numerical-loop or NumPy waiver. Only supported real numeric/bool dtypes and flat
member names are accepted; no objects or pickle payloads.

Require independent NumPy loading and exact values, shapes and dtypes for
float32/64, signed/unsigned integers and bool; scalar, empty and large arrays;
signed zero, subnormal, infinity, NaN and integer limits. Invalid names and
unsupported dtypes must fail. Compare member payloads against original NumPy
archives through the independent reader. Include the actual initializer archive
names in fixtures. Full consumer archives remain a later integration gate.

Use the registered `tensor_npz_cpu` worker, explicit CPU hiding, at most4 workers
and480 charged seconds within unchanged32CPU/52GPU-hour campaign caps. No GPU
qualification is needed for byte serialization; the upstream numerical tensors
retain their own GPU/XLA gates. Save ordinary numbered worker manifests/JUnit,
source hashes, command, environment and timing. Run Ruff, whitespace and policy
checks. Stop on wrong member data or scope/budget changes, retaining failed
attempts. This helper cannot establish numerical, performance or scientific
readiness of the initializer.

Review: byte-for-byte ZIP equality would test timestamps/compression details,
while only checking approximate arrays could hide dtype or signed-zero changes.
The test instead compares reader dtype/shape and exact array bytes. Tensor
materialization is allowed at this completed artifact boundary; no NumPy
computation enters the runtime. Full integration still must prove that only
active rows and completed partitions are serialized.
