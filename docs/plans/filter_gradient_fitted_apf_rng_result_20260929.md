# Fitted-APF seeded-input diagnosis

Runs04834–04839 close the bounded preflight. Simply compiling TensorFlow's
stateless floating draws changes every tested cloud on CPU and GPU. Maximum
normal/uniform differences are3.2320956/0.8855319. Explicit Philox does not fix
the built-in floating draws, although eager auto/explicit draws, key/counter
state and raw uint32 words match exactly. This localizes the tested discrepancy
to conversion/normal transformation, rather than seed scrambling. The existing
shared Philox Box–Muller normal helper matches the eager arrays exactly on both
devices. No production numerical source changed and no stream migration was
accepted.04839 passes162 readback and campaign/policy checks.

The existing `bayesfilter/ops/stateless_random_tf.py` already provides FP64
uniform and normal conversion preserving the native TensorFlow stream.
Reuse that authority for the fitted-APF input owner; do not create another RNG
implementation. Its existing independent tests cover additional shapes and
negative seeds; actual fitted-APF labels and enclosing execution still require
qualification. FP32 is a separate supported setting and cannot be inferred
from these FP64 findings.

The frozen seeds, shapes, tolerances, TensorFlow source anchors and commands
are in `filter_gradient_fitted_apf_execution_20260929.md`. Raw arrays, HLO,
source hashes and CPU/GPU manifests are in runs04834–04839 under the campaign
artifact root. Six workers consumed29.727733781 CPU and12.207573362 GPU seconds.
CPU is an explicit reference; GPU numerical checks do not establish unshared
performance or capacity.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Reject direct built-in XLA draws | Complete arrays differ | Existing-stream mismatch | Actual endpoint compatibility remains untested | Use the existing explicit conversion authority | No fitted-APF admission |
| Continue compatible controller repair | Raw state/words match; shared normals exact | No tolerance relaxation | FP32 and complete fit/score behavior | Separate primitive and endpoint checks | No stream change or scientific claim |

| Inference status | Finding |
|---|---|
| Hard veto | Direct generator migration fails compatibility |
| Statistically supported ranking | None; no performance ranking attempted |
| Descriptive differences | Maximum errors for two frozen seed pairs only |
| Default readiness | Open for the fitted-APF consumer |
| Next evidence | Preserved inputs, complete recurrence/history, frozen-fit score and costs |

Post-run review: exact raw bits alone would have falsely qualified different
floating arrays. Conversely, a shared normal match does not establish uniform
conversion or the fitted endpoint. Full records and live seeded wiring remain
mandatory. These results reject one migration mechanism, not the APF algorithm.
