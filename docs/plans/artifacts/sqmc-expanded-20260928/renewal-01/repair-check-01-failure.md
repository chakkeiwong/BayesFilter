# Repair check 01: diagnostic fixture mismatch

The check stopped after 12.50 seconds. It compared a GPU/XLA trace against a
CPU trace that had regenerated, rather than replayed, the frozen GPU random
inputs. TensorFlow backend-dependent random generation made that comparison
invalid. This is a localized diagnostic-harness failure, not evidence of a
shared numerical implementation failure. It consumes the first repair-check
attempt; two infrastructure retries remain.

The CPU-only diagnostic-01 result is not an exact replay of the GPU case.
Its numerical mass-error value 1.606e-4 must not be attributed to run-01.
The GPU trace observed 1.280e-3, also above the unchanged 1e-4 guard, but the
retry must verify saved observations and serialized-input hashes before
attributing that value to the original case.

Repair: verify the GPU tensors against run-01 data/design evidence, then pass
those identical tensors to the CPU/non-XLA reference and GPU/XLA trace.
No numerical implementation, guard threshold or final comparison data changed.
Preserve the failed check and charge its complete wall time.
