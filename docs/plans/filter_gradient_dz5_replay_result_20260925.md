# DZ5 graph replay investigation

The full 96-observation CPU graph reference now passes both original score
oracles and exact replay with the explicit control
`arithmetic_optimization=False` (03901). The default CPU graph replay failure is
preserved. Default CPU/GPU XLA remains qualified by 03866/03867; no runtime
default, tolerance, filter equation or external source changed in this unit.

The [reviewed plan](filter_gradient_dz5_replay_followup_20260925.md) used the
unchanged `dz5-candidate-source-score-4c37f9f40-r1` snapshot, its full
185-row/23-parameter bank and original 96 prepared observations. Manifest SHA:
`cbe75052fe85fcff58a7cbdc337e77e6507cf4240340308347993e88445c516d`.
Both five-point steps retain `atol=1e-8, rtol=1e-7`.

| Observation | Evidence | Meaning |
|---|---|---|
| Default graph replay differs at 64 and 48 observations | 03892/03893: maximum score differences 5.7980e-12 and 9.0949e-13, with exactly equal values and passing finite differences | The original 96-step failure has a shorter reproducer |
| First carried-state difference is localized | 03896: third call first differs in mean tangents at observation 35; factor tangents and score at 36. Second call is entirely exact | The behavior is intermittent; initial equality alone is insufficient |
| Ordinary states and diagnostics stay exact | Full 03896 means, factors, values, validity, pivot/residual histories | The observed divergence starts in derivative propagation |
| Frozen single updates replay exactly | 03895: step 48, 20 calls on each of one/two threads. 03897: step 35 and all 36 exposed intermediates, 20 calls per thread | Neither a particular primitive nor thread count is proved as the cause |
| Additional intermediate fetches suppress the reproduced difference in this run | All three 03898 trajectories are exact | Instrumentation can change the execution condition; this cannot pass the original failure |
| Arithmetic-optimizer control passes | 03899 at 48 observations; 03901 at all 96 observations | A qualified, explicitly configured graph reference is available |

The original simplified diagnostics 03883--03890 also replay exactly through
32 observations. Their fresh/reused input and dictionary/tuple fetch differences
were identified and preserved; they cannot establish a monotone first-failing
horizon. The original harness was restored before the 48/64-step reproductions.
03894's second child could not see its sibling's frozen input through the
snapshot mount. Copying the checksummed input into the child output binding
repairs this harness defect; complete retry 03895 passes. The failed attempt
remains in the archive.

There is executable evidence for a relevant TensorFlow summation mechanism.
In installed TensorFlow 2.19.1, the same three binary64 terms
`1e16, -1e16, 1` sum to 1 with no/all extra operands fetched, and to 0 when the
first two operands are also fetched (03900, four replays each). The mathematical
and original ordered sum is 1. Tagged TensorFlow source
`core/kernels/aggregate_ops.cc:53` forwards an eligible input buffer and moves
that operand to the front. `AddOpsRewriteStage` in
`core/grappler/optimizers/arithmetic_optimizer.cc:519` combines add trees into
AddN. The actual executed target graph contains 78 AddN nodes before optimization
and 92 after, including new rewrites in the filter loop. Captured graphs, public
source URLs/checksums and the primitive records are preserved.

Together with the optimizer control, this supports buffer-sensitive summation
as a concrete mechanism to investigate. It does **not** identify the exact
filter node responsible for the observed tangent divergence. Disabling the
optimizer changes several rewrites, and exposing operands changes buffer
lifetimes. A tagged source file is not a binary-build attestation. The next
causal test must isolate an actual implicated addition on saved operands and
check a local order-preserving candidate in the full target.

03901's original full numerical gates all pass: normalized finite-difference
errors are 0.1787334165 at both step sizes, every row is valid, replay is exact,
there is one trace and no host callback, and all actual project imports match
the read-only snapshot. Complete score comparisons against CPU/GPU XLA have
maximum normalized errors 2.3792e-5 / 2.8080e-5 at the original tolerances.
Independent standard-library analysis 03902 reconstructs the derivatives,
checks full records/provenance, reconstructs every saved 03896 history mismatch,
checks all 40 frozen-step replays, and rejects six corrupted oracle records.

Single-process capacity observations for this controlled graph reference:
cold 292.591 seconds, replay 269.283 seconds, sampled peak RSS 1,769,385,984
bytes (1.648 GiB), final RSS 1,294,233,600 bytes. CPU allocator telemetry is
unavailable. These are neither repeated performance rankings nor compiler-memory
attribution. The earlier default-XLA CPU peak of 2.470 GiB and supervisor memory
repair retain their own evidence scopes.

Optimized graph inspection also confirms local autodiff/pfor in frozen external
MacroFinance callbacks, including `two_currency_double_zlb_credit_target.py:74`
(`batch_jacobian`). The BayesFilter filter recurrence uses supplied analytical
derivatives; that does not make these external callback derivatives analytical
or policy compliant. External source migration is outside this unit's authorized
scope. Numerical qualification cannot support a whole-target no-pfor or entirely
analytical claim. No new pfor allowance was added.

| Decision | Primary criterion and veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Accept the configured CPU graph reference | Full original oracle, replay and source checks pass | Default optimizer still fails replay | Preserve both configurations and isolate the actual addition | Default graph repair |
| Retain the localized defect | Observation-35 tangent divergence is independently reproduced from raw histories | Instrumentation affects reproduction | Operator-specific frozen operands and full-target confirmation | General thread or conditioning cause |
| Keep external derivative debt explicit | Actual graph and frozen source contain autodiff/pfor | Callback migration under separate source ownership | Track migration before any whole-target policy claim | All derivatives analytical or pfor-free |
| Keep main unmerged | Public LEDH/initializer, GPU graph oracle, reporting/isotropic, capacity and F01--F20 gates remain | Full consumer closure | Continue the master repair sequence | Whole-program completion |

Post-run review: the strongest alternative explanation is another arithmetic
rewrite whose behavior changes with output retention. A local fix that passes
the primitive but fails the unchanged full oracle would be rejected. The weakest
remaining causal evidence is the unisolated failing filter node. The complete
numerical controls are useful reference evidence without erasing that gap.

The bounded unit closes through 03903 at 21 workers / 4325.923902 charged CPU
seconds out of 24 workers / 7200 seconds. No GPU time was used. All 129 policy
checks pass 03903; focused Ruff checks pass. Global charges are 81757.850907 CPU /
75683.391781 GPU seconds, leaving 33.289486 / 30.976836 hours under 56/52-hour
caps. The runtime guard remains 244 sources / 1346 exact existing allowances.

Receipt: `artifacts/filter-gradient-repair-20260917/dz5-replay-verification-03903.json`.
The tracked 24.6-MiB archive contains metadata, commands, logs, optimized graphs,
frozen inputs and source. Roughly 1226 MiB of full NPZ histories and operand
records remain in the shared raw root, with individual SHA-256 entries in the
receipt; the archive is deliberately not self-contained for full-array analysis.
Reproduce the independent analysis against that raw root with
`python3 docs/plans/artifacts/filter-gradient-repair-20260917/analyze-dz5-replay-20260925.py <raw-root>`.
