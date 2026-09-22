# Internal ordered-controller execution and numerical gap

Current continuation: the owner approved the narrow resolution error, now
installed and CPU-tested. See [the current qualification](filter_gradient_objective_resolution_result_20260923.md).
The03036 checkpoint and pending-decision discussion below are historical;
GPU renewal and public ordered-block integration remain open.


The master program is refreshed and executing. The new internal controller
encloses ordered conditional solves, full replay, transactional commit/rollback,
accounting, reversal/cycle stops, summary metrics and completed records in one
TensorFlow/XLA call. The public block API still uses its prior controller.

03024--03027 pass four complete original3582b4ac coupled2D scalar/batch cases
with reversal stopping enabled/disabled, two centers/scales, full public/private
records, buffered events, exact target counts and one reusable XLA trace.
03030 passes nine compiled dependency-boundary tests for both allowed rejected-
geometry handoffs and transaction/accounting/nonfinite/mass vetoes; no later
block/replay executes after a veto.03032 passes actual outer graph/target
collection and retained-handle execution.03035 passes129 policy checks.
03036 renews all43 existing block-consumer checks after the optional reporting
field change. Focused Ruff and whitespace checks pass. The partial source guard covers
227 sources/1332 exact exceptions. Five new exceptions construct static block
topology; two serialize completed records. None permits numerical Python loops
or NumPy runtime work. No independent subagent review was used.

The changed-input partial/heterogeneous4D fixture03028 fails:173 versus182
sequential calls,176 versus185 physical calls, and a different first-block
handoff. The initial input matches201 physical calls.03029 reproduces the same
conditional99-versus108 count in both the current public sequential route and
the new conditional graph, isolating the gap from outer ordering/capture.

03031 evaluates identical predecessor/proposal points. The first difference is
in the target matrix-vector product, then a product and reduction. Original
eager/graph objectives tie; XLA sees one ULP of improvement. The rejected
proposal becomes the exact incumbent, crossing the terminal score threshold.
The untouched callback reproduces the instrumented values exactly within each
mode. A100-digit exact-input reference gives improvement7.545598688169452e-21,
far below the binary64 spacing2.7755575615628914e-17. This does not establish
the original finite-program decision or justify silently waiving its gate.

The [explicit-error proposal](filter_gradient_objective_resolution_proposal_20260923.md)
would report a resolution-limited rejected-proposal promotion and block geometry
use/handoff. Diagnostic03034 flags this case and leaves three controls unchanged.
It is uninstalled, and approval is pending because its rejection contract differs
from the earlier owner-approved discarded-precision exception. Normal accepted
comparisons and the failed original strict-decision gate remain unchanged.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep internal outer implementation | Four complete original cases and nine boundary cases pass | Partial changed-input original gate fails | Resolution-limited strict comparison; GPU cases pending | Resolve rejection policy, renew full CPU/GPU/public cases | Public or default readiness |
| Retain graph ownership repair | Callback/graph release and live-handle reuse pass | GPU renewal pending | Native signature churn | Run GPU checks when eligible | Native eviction or leak freedom |
| Evaluate explicit-error candidate | Preserved case flags, three controls do not | Uninstalled; owner decision pending | Broader no-harm behavior and opaque target errors | Approved public no-use and downstream tests | General numeric-error bound or convergence |

03023 is a preserved diagnostic serialization failure on an existing -inf
history field. Its repaired writer tags nonfinite diagnostic values for storage
only; comparisons retain original numbers.03028 remains an unwaived numerical
failure. GPU preflights81545/80835/65157 declined before worker launch; desktop
fallback conditions were not met. No GPU result is inferred from CPU checks.

The14 outer workers03023--03036 used510.1393746379763 charged seconds of the
30-worker/7200-second unit. Cumulative charges through03036 are
60700.59426078647 CPU /59401.31348332534 GPU seconds, leaving15.14/35.50 hours
within the32/52-hour caps. Every run records command, environment, source hashes,
hardware, time and artifact paths. Receipt:
`artifacts/filter-gradient-repair-20260917/ordered-block-checkpoint-03036.json`.
One numerical worker/source freeze remains required. No worker is active at
this checkpoint. Main remains unmerged.

Post-run review: passing coupled fixtures did not cover arbitrary target
rounding, and this new fixture disproves broad strict-decision parity. The
new outer loop could have been blamed incorrectly; the same failure in the
public sequential route and identical-point target arithmetic refute that
explanation. The weakest evidence for the proposed error guard is its narrow
control set; full CPU/GPU and real-consumer tests remain necessary. Initializer
rounding, actual DZ5 migration, matched costs, native signature churn, F01--F20
terminal dispositions and remote integration remain open.
