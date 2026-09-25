# Filter and gradient repair recovery

Current checkpoint through 03903, 2026-09-25: the full 96-observation DZ5
CPU graph reference passes its original two-step score oracle and exact replay
with explicit `arithmetic_optimization=False` (03901). Independent analysis
03902 verifies raw derivatives, XLA comparisons and complete trajectory records;
policy 03903 passes all 129 checks. Default CPU graph replay and GPU graph
finite-difference failures remain preserved. Read the
[replay result](filter_gradient_dz5_replay_result_20260925.md).

The first reproduced carried-state difference is the mean tangent at observation
35 (03896), followed by factor tangents and score at 36. Frozen single-step
controls pass. The installed TensorFlow AddN primitive demonstrably changes
summation order with buffer eligibility; actual optimized filter graphs contain
new AddN rewrites. The exact failing filter node is not yet proved. Numerical
qualification also does not remove the frozen external MacroFinance callbacks'
autodiff/pfor debt; no whole-target policy-compliance claim is eligible.

This unit completed 21 workers / 4325.923902 CPU seconds, including the preserved
03894 file-handoff failure. No GPU time was used. Charges through 03903 are
81757.850907 CPU / 75683.391781 GPU seconds, leaving 33.289486 CPU /
30.976836 GPU hours under the existing 56/52-hour caps. No worker is active.
Receipt: `artifacts/filter-gradient-repair-20260917/dz5-replay-verification-03903.json`.
The 24.6-MiB tracked archive holds metadata, logs, graphs, frozen inputs and
source; large NPZ histories/operand records remain in the shared raw root,
individually checksummed in the receipt. It is not a self-contained bulk-data archive.

Public LEDH value/score/KDM integration, valid/rejected reset qualification,
public initializer/staged supervisor, reporting/isotropic findings, target
capacity/memory and F01--F20 terminal dispositions remain open. Guard coverage
is 244 sources / 1346 existing allowances; no runtime waiver was added.
Two reporting proposals still await the user's existing async decision.
Main remains unmerged. The separate supervisor memory repair remains qualified.

Continue on `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`, branch
`repair/filter-gradient-xla-validation-20260918`, with the stable campaign runner.
Raw root: `/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917`.
One numerical worker at a time; freeze runtime/scripts/tests while it runs.
No subagents, MacroFinance source edits, CDF restart, package mutation, stale
admission refresh, canonical LEDH rebuild or historical NeuTra promotion.
Preserve numerical methods/seeded streams and inherited predator--prey k4.
Canonical NeuTra remains the shared author-profile IAF. Fetched remote main
`5e16df06f` is already contained. Base of this unit: pushed `6ed86f8d8`.

Next: retain the controlled DZ5 reference; isolate the actual graph addition
before any default repair. Independently proceed with the LEDH valid/rejected
reset and public-consumer repair already mapped in the master program. Preserve
all prior numerical vetoes; successful diagnostics cannot replace terminal gates.

Earlier recovery checkpoints remain in Git history and the linked master/result files.
