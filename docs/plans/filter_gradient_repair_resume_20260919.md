# Filter and gradient repair recovery

Current checkpoint through 03956, September 26: the KDM auxiliary now owns a
stable-signature, default-XLA numerical program with native time recurrence,
prebuilt atom/KDM kernels and a stacked trace from the shared analytical
executor. The shared Sinkhorn/reset recurrence is also native. Public fields,
mutable-callback call behavior, invalid-model rejection and the empty horizon
are preserved. The canonical LEDH rebuild remains excluded.

The original seed-131 fixture is correctly rejected: its first reset column TV
error is `0.0005588870472403515`, above the existing `0.0001` gate. A separately
declared 8/8 control with identical seed/data/tolerances passes frozen-reference,
finite-difference, complete-record, replay, changed-input and CPU/GPU XLA checks.
Final qualification 03953 passes 30 tests; public GPU/empty-horizon 03955 passes
two checks; policy 03956 passes 129. Guard coverage is 246 sources / 1354 exact
allowances; added allowances cover schema/configuration, completed presentation
and the tested enclosing-XLA helper, never numerical recurrences or NumPy.

All 18 matched cost workers (03929--03946) pass numerical checks. Independent
analysis reproduces saved results, provenance and three-repeat medians; six
corruption tests reject altered evidence. Single-owner CPU/GPU runs have one
trace and exact stable replays, while public reconstruction has greater XLA
compile/host-memory cost. Read the [KDM result](filter_gradient_kdm_native_result_20260926.md)
and its [remaining-gaps plan](filter_gradient_kdm_remaining_gaps_20260926.md).
FP32 eager/XLA reset differences remain a mandatory open gate even though
original and repaired XLA outputs are bitwise equal. The memory observations
do not prove a permanent graph leak or target-scale capacity.

The DZ5 addition-order overlay remains rejected: full replay changed 3,846 score
entries by up to `9.43600753089413e-12`. The explicit arithmetic-optimizer-disabled
CPU graph reference 03901 remains qualified; default graph replay/GPU graph
finite-difference failures and external callback autodiff/pfor debt remain open.
No overlay was installed. See the [addition result](filter_gradient_dz5_addition_order_result_20260925.md).

Charges through 03956 are 83647.549451 CPU / 76175.823410 GPU seconds,
leaving 32.764570 / 30.840049 process-hours under the unchanged
56 CPU / 52 GPU hour caps. The extra 24 CPU hours are already counted. This KDM
unit consumed 28 CPU workers / 736.573433 seconds and 15 GPU workers /
492.431629 seconds, including preserved failures and final policy checks.
No worker is active. Verified archives/receipts through 03924 and 03956 are
tracked under `artifacts/filter-gradient-repair-20260917`; raw files remain in
the shared root. The transient 03911 prototype was never archived; hashes alone
cannot reconstruct it. New candidates have archived patch/test bytes.

Public LEDH value/score integration, valid/rejected reset qualification,
public initializer/staged supervisor, reporting/isotropic cases, target
capacity/compiler memory and F01--F20 terminal dispositions remain open.
Two reporting proposals await earlier user answers. Main remains unmerged.

Worktree: `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`.
Branch: `repair/filter-gradient-xla-validation-20260918`; prior pushed checkpoint
`b052054f7`. Raw root:
`/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917`.
One numerical worker at a time; freeze runtime/scripts/tests during it.
No subagents, MacroFinance edits, package changes, CDF restart, stale-admission
refresh or canonical LEDH rebuild. Canonical NeuTra remains the shared
canonical author-profile IAF. Remote repair branch was fetched and contains no
new commits relative to the base; no conflict needs resolution. Previously
integrated remote main `5e16df06f` remains the branch's current main integration.

Next: preserve this tested repair on the branch, then execute the bounded
rounded-input precision and repeated-owner memory follow-up while retaining
all other master gates. No master finding or main promotion is inferred from
this component's successful execution checks.
