# Mixed KR/TTSIRT transport execution closure

Question: do the remaining mixed transport paths expose a policy-relevant
Python numerical recurrence, or are the only such loops confined to the old
scalar `KRTransport` diagnostic while the active `FixedTTSIRTTransport` source
route already calls the native XLA owner?

The source audit starts at `bayesfilter/highdim/transport.py` and follows every
package consumer. `KRTransport` is instantiated only by
`tests/highdim/test_transport.py`; no BayesFilter package module constructs or
calls it. Its scalar row/axis/bisection helpers therefore remain an independent
reference diagnostic and cannot support production, filtering, HMC or default
claims. The active package consumers use `FixedTTSIRTTransport`, whose public
map, conditional, density and proposal methods delegate to
`bayesfilter.highdim.ttsirt_native_tf.evaluate_transport`. That owner uses
`tf.while_loop` for coordinate and bisection recurrences and defaults to
`jit_compile=True`.

The public `potential`, `proposal_log_density` and `log_normalizer` wrappers
were included in the same execution audit. The first two now select fields
from one native `log_density` owner; `log_normalizer` has its own owner.
The existing PDF expression and status handling are preserved, so the complete
standalone numerical API boundary defaults to XLA.
The explicit `jit_compile=False` option remains available for reference/debug
checks.

The closure preserves the existing numerical grid, trapezoid, interpolation,
bisection settings, status codes and source-route classification. It does not
replace the diagnostic KR algorithm with a different transport. It records the
old scalar loops as historical diagnostic scope, verifies that no package call
chain reaches them, and reruns the active native transport and source-route
consumers. A changed-input HLO check, static no-package-caller check, and
enclosing graph check for the potential/proposal wrappers are required. No
NumPy, pfor, or numerical-loop allowance is added.

Skeptical audit: a passing native TTSIRT test alone would not prove that the
old KR object is inactive, while a grep alone would not prove that public
consumers use the native owner. The static caller inventory plus enclosing
graph/XLA tests address both risks. This is a bounded execution classification;
it does not establish author-scale KR correctness, Zhao--Cui production
faithfulness, LEDH admission, HMC readiness or whole-repository policy closure.

The first classification check (04070) does not close this unit: review of the
earlier import audit found eager logarithms in the two public log wrappers.
Their repair requires direct CPU/GPU wrapper, input/core pullback, zero-density,
invalid-status and changed-input HLO checks. Preserve `log(0)` infinities at
the finite support boundary as in the original API; do not add a new rejection
or substitute `log_relative` for the finite `log(exp(log_relative)*measure)`
program. Use frozen original transport/density/TT sources at `3582b4ac` for
numerical checks, plus finite differences of the query as an independent
derivative check. Compare finite values/pullbacks at the existing 1e-10 bound,
require exact replay, unchanged trace count and graph/HLO after changed inputs,
and require the public methods to perform no eager logarithm.

Measure the immediately preceding wrappers at Git `5bbce48b5` against repaired
graph and XLA wrappers on the same native density dependency. This baseline is
a partially compiled public API; do not describe it as an eager density kernel
or entirely uncompiled before arm. Use d=2, sample counts 3 and 128, identical
CPU-generated deterministic points, 20 synchronized calls, separate fresh
processes per arm/device, and record whole returned values, cold/warm time,
RSS, TensorFlow GPU allocator current/peak and sharing provenance. These are
descriptive fixture costs, not a terminal speed ranking or target-scale memory
claim. Additional owner compilations and log/exp fusion are the specific cost
and rounding risks. Old costs from the unchanged-map unit do not answer them.

Use the stable runner, registered `mixed_kr_*` groups, two CPU threads,
memory growth and automatic available non-display GPU selection. Reserve at
most 28 additional CPU and 24 GPU invocations, each <=300 seconds, with totals
<=1800 seconds per backend inside the unchanged 56 CPU / 52 GPU hour campaign
caps. Stop an affected arm on numerical error, provenance drift, or exhausted
budget; preserve failures and use a localized repair before retry. Consumer
suite 04073/04074 passes after the first wrapper patch; dedicated tests and
costs were then required; completion is recorded below. No canonical scientific claim or new algorithm is
being evaluated. Review is local; no independent reviewer was launched.

Review/cost continuation: `log_normalizer` had the same escaped logarithm and
is included. The first twelve-arm cohort 04079--04090 passed numerics but
showed 22 MiB CPU / 31 MiB GPU RSS overhead and increased cold cost from
separate log-density/potential owners. Share one owner returning a fixed
dictionary of log density and its negative; public wrappers only select a
schema field. Keep the normalizer owner separate. This is the same finite
computation and unchanged 1800-second per-backend budget; the invocation
reserve above accommodates a localized repair and renewed cost cohort.
04076's raw HLO equality failure concerns imported zero-cotangent metadata
names. Save both dumps, normalize only the GraphToFunction counter on a
`zeros_like` Const at `dummy_file_name:10`, and require identical numerical
instructions/constants. Adverse normalization tests preserve that boundary.
The source guard now includes the active transport methods and helpers with
two static schema allowances. Old private bisection/reference-density helpers
in FixedTTSIRTTransport are also unreachable from these native owners; their
loops remain historical, so do not call the entire file loop-free.

Final validation: shared-owner tests 04092/04093 pass six checks per backend;
refreshed cost cohort 04094--04105 passes complete value, input, device and
source-identity comparisons; consumers 04106/04107 pass twelve checks each;
policy 04108 passes 129 checks with 270 guarded sources / 1,424 exact
allowances. The result note records the residual compile/RSS cost, descriptive
scope and preserved failed attempts. This closes this unit's bounded execution
question, not the master program or source-faithfulness/canonical admission.
