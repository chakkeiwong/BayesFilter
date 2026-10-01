# Proposed explicit error for a resolution-limited incumbent decision

Status: approved by the owner on 2026-09-23 ("I agree."). Installation and CPU
qualification are complete; GPU qualification is pending availability. Original failure03028 is preserved; approval
changes only the narrow rejection/no-use criterion defined below.

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

The approved narrow repair reports `objective_resolution_limited` when a
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
This changes the rejection/qualification contract. The owner decision recorded
above authorizes this narrow change under the preserved-algorithm campaign.

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
of broad no-harm behavior. Owner approval was requested after this evaluation and subsequently granted;
this prototype alone did not authorize or qualify the installed runtime.

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

## Approved installation and pre-execution review, 2026-09-23

The owner approved this exact proposal after03034. Install the flag after the
complete sequential lifecycle and before mass preparation. Root status4 denotes
`objective_resolution_limited`; lifecycle status4 retains its different existing
meaning. Completed errors must charge lifecycle evaluations and cannot replay a
block, commit a center, execute a later block, or expose a usable candidate,
precision or covariance. Host formatting may serialize the tensor diagnostics
but must not recompute the predicate. Preserve every unflagged public field,
event, decision and count at the existing strict original3582b4ac gates.

Review found an accounting hazard: the outer controller currently charges only
locator evaluations for nonzero root statuses. Repair that for completed status4
without admitting lifecycle status4 through the new root error. The unchanged
public host block controller already rejects an unknown handoff or absent
candidate; exercise it as well as the internal native controller. An injected
mass callback must prove the error skips preparation, not just hides the result.

Evidence: exact equal/adjacent/two-step, signed, zero, nonfinite and inactive
boundary cases; preserved partial4D scalar/batch public and conditional calls;
unchanged complete histories/counts against17b56ade2; real public/native block
no-use and no-later-call checks; unchanged original fixtures and consumers;
source guard, XLA graph/HLO and CPU/GPU renewal. A flagged original comparison
is eligible only with the actual runtime predicate and explicit no-use evidence.
Larger cancellation and fit-cloud/search promotions remain outside this rule.
A missed guard, altered unflagged record, wrong accounting or downstream use
blocks this repair. GPU unavailability defers GPU evidence; CPU does not replace
it. No numerical equivalence, convergence or native-cache eviction claim follows
from reporting this error.

Reserve a new bounded qualification tranche after03036: at most90 fresh workers
and14,400 charged seconds shared across CPU/GPU, normally120/300 seconds per
worker. This includes focused checks, full affected sequential/public consumers,
ordered blocks and pending capture/dependency renewal. At most three localized
retries per failing fixture before renewed diagnosis. It replaces neither the
preserved preceding unit accounting nor the cumulative32 CPU/52 GPU-hour caps.
Use the existing runner prefix, numbered artifacts, one worker and frozen source
through each matrix. Preflight GPU selection retains desktop protection and
verified growth. Baselines, target, seeds, score, solver and healthy tolerances
are unchanged. Review passes with the explicit new error as the sole approved
numerical qualification change; broader campaign gates remain open.

Boundary failure03037: CPU XLA floating comparisons flush the smallest
subnormal next to zero, missing the literal positive-nextafter predicate. Repair
the implementation with exact binary64 bit ordering in TensorFlow uint64,
identifying signed zeros and testing a positive integer distance of exactly one
with finite operands. For finite representable values this is exactly the
approved predicate, including negative values and zero; it does not round or
change objective evaluation. Extend the independent Python math.nextafter
reference to subnormal and normal/subnormal boundaries before integration.

03038 passes the exact boundary after the bit-order repair;03039--03043 pass
scalar/batch public and conditional error/no-use checks plus ten block boundary
cases. Both failing conditionals preserve99 evaluations, and both public block
calls stop at100 physical rows without replay.03044--03049 pass all six internal
ordered cases; the unflagged partial input retains201 rows and the flagged input
stops at100, with one trace and identical HLO.03050--03056 pass the first seven
public original-record groups. The queue was deliberately paused between workers
for a report-only repair: retain terminal attempt count, maximum score and seed
metadata on the new rejection. Earlier passes remain evidence for their source;
renew the final frozen cohort. GPU preflight24601 declined before any worker.
