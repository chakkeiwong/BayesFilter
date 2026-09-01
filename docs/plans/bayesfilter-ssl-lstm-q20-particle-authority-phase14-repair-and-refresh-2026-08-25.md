# Phase 14 Repair and Refresh Note

Status: `PASS_HARD_GATES_ROLE_LIMITED_FULLBANK_DIAGNOSTIC_REFRESHED_PHASE15`

Classify failures as affine covariance/hash, GPU policy, split/weight, target
composition, parity, batch-native training, and candidate quality. A failure
in the affine wrapper must not be interpreted as a failure of the underlying
particle bank until the composed physical target check is isolated. Preserve
all unique outputs and keep HMC deferred.

## Attempt 1 harness repair

The first full-bank rerun failed when the affine triangular solve erased the
static row count required by the batch-native trainer. The failure occurred
before any full-bank metric was written; the prior affine hard-gate result is
unchanged. Shape propagation is repaired explicitly and the same audited input
will be rerun in a new output directory.

The second attempt showed the same static-shape loss at the full-bank compiled
validation call, despite local transformed-shape propagation. The known
particle count is now bound explicitly on the loaded and full-bank tensors;
attempt 3 will use a fresh output root. No training or target conclusion is
drawn from attempts 1--2.

## Attempts 3--4: shape repair receipts

Attempts 3 and 4 reproduced the same failure at the compiled full-bank
validation call. They are preserved as harness failures; no candidate or
particle-bank conclusion is drawn. The repair replaced that auxiliary call
with an eager TensorFlow-only full-bank metric calculation using the already
trained transport. This leaves the training path batch-native/XLA while making
the diagnostic independent of a second polymorphic `tf.function` trace.

## Attempt 5: completed full-bank diagnostic

`phase14-attempt5-affine-fullbank2401` passed the GPU memory-policy, XLA,
batch-native update, affine/flow round-trip, target/status, finite-value, and
artifact gates. All three arms are role-limited candidates. The best full-bank
moment residuals are from `compact_low_lr`: weighted mean max `0.1197`,
off-diagonal max `0.1243`, diagonal max error `0.1921`, and covariance
Frobenius residual `0.3575`. The validation subset gives a different and more
optimistic picture, so it cannot be used as the sole whitening diagnostic.
Affine preconditioning therefore improves conditioning relative to the prior
identity screens but does not establish IID Gaussian whitening. Refresh Phase
15 to a paired identity-versus-affine full-bank comparison on the same audited
bank, seed, profile, and step budget. Keep both routes diagnostic-only.
