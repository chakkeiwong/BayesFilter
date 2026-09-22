# Proposed explicit error for a resolution-limited incumbent decision

Status: reviewable proposal only; no runtime predicate, rejection status or
equivalence criterion is changed. The original comparison remains failed.

The failure in03028 is cancellation in an objective comparison, not an
ill-conditioned posterior precision.03029 reproduces it before the new block
controller.03031 evaluates the exact same binary64 input points: the first
rounding difference is in the target's matrix-vector product, then a product
and reduction. Graph/eager return equal predecessor and proposal objectives;
XLA returns adjacent representable objectives. Instrumented and untouched
callbacks agree within each execution mode. A100-digit calculation with exact
binary64 inputs gives improvement7.545598688169452e-21; the binary64 spacing is
2.7755575615628914e-17. Both callback modes use the same TensorFlow operations.

The existing algorithm promotes any strictly better exact evaluated point,
including a proposal rejected by its ordinary model/score acceptance test.
Here a one-ULP objective change promotes the point despite model rejection,
causing99 versus108 conditional evaluations and usable versus no terminal
geometry. Choosing either floating-point mode as the authority cannot establish
a resolved objective improvement. The current declared comparison requires
exact decisions and counts, so this remains a real campaign veto.

The proposed narrow repair reports `objective_resolution_limited` when a
rejected proposal becomes the exact incumbent solely through a strictly positive
objective change of at most one representable step. Specifically, for the
selected pre-proposal objective `a` and promoted objective `b`, require
`promoted_without_acceptance && a < b && b <= nextafter(a, +inf)`, with finite
operands and an actually executed proposal. This is a representation-resolution
flag, not a bound on all numerical error or a general stationarity test.
Do not round objectives, change a proposal, change the analytical score, widen
healthy tolerances, silently treat the proposal as accepted, or substitute a
different algorithm.

At the completed sequential boundary, preserve the full diagnostic history and
counts, return an explicit error with no usable precision/covariance/candidate,
and block handoff to later ordered blocks or external geometry consumers. Keep
the failed original/changed records as diagnostic evidence. The revised
comparison for a flagged pair would require explicit reporting and no use;
every unflagged record, decision and count retains its existing strict gate.
That is a change to the rejection/qualification contract, so installation needs
an explicit owner decision under the campaign's preserved-algorithm scope.

Before installation, evaluate the tensor-native flag only as a diagnostic
wrapper around the complete conditional endpoint on the exact failing fixture
and existing coupled2D fixtures. The wrapper must not affect their numerical
program or public result. Assert that it flags the preserved case and leaves
the passing cases unflagged. After approval, actual public error/no-use/blocked-
handoff tests, accepted-result regression and CPU/GPU evidence are mandatory.
The prototype cannot close the failed gate by itself.

Evaluation03034 passes: the exact preserved fixture flags its third refinement
attempt; the initial partial-block input and both coupled2D inputs remain
unflagged. The conditional programs keep their original99/124/46/46 evaluations
and runtime status0; the diagnostic neither rejects nor modifies those results.
Receipt: `artifacts/filter-gradient-repair-20260917/run-03034/objective-resolution-candidate.json`.
This is evidence for evaluating the proposed guard, not authorization or proof
of broad no-harm behavior. Owner approval was requested after this evaluation;
the runtime and failed comparison remain unchanged while it is pending.

This evaluation reserves two CPU workers120s each within the existing ordered
unit30-worker/7200-second ceiling. No GPU substitution, package change or added
campaign budget is needed. Artifacts use the normal numbered runner directories.

Review: one ULP is an exact representational boundary, not a calibrated target
error estimate. Larger cancellation errors and promoted fit-cloud/search points
are outside this initial candidate and must remain separate findings. A clean
result on four small fixtures cannot justify broad default readiness. This
candidate only changes whether an unresolved result may be consumed; it must
not claim to restore bitwise legacy arithmetic or convergence. GPU qualification
and the older initializer rounding gap remain independent.
