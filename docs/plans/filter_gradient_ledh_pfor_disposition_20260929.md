# Optional LEDH pfor disposition

Execute after the streaming unit's current-source readback. Question: can the
unapproved optional K-direction pfor branch be removed without changing the
accepted sequential value/analytical-score program? This is a policy repair,
not a canonical LEDH rebuild, batch-native training qualification or HMC run.

The default `canonical_batch_fused_value_score` uses nested TensorFlow map_fn
with parallel_iterations=1 over rows/directions. Its optional k_batch_mode=pfor
calls vectorized_map and claims approval at
`docs/plans/ledh-vectorized-map-approval-request.md`. That file requests approval;
it records no owner grant, and its assertions about graph traces/host-memory
complexity are not measured evidence. User approval of the master execution
campaign does not specifically approve pfor. Existing static governance test
`test_batch_claim_paths_ban_python_fanout_and_pfor_apis` fails at04682 and must
remain intact. The direction-only option has no mathematical necessity for the
existing sequential execution contract.

Callers inspected: `ledh_canonical_neutra_targets_tf.py` and
`ledh_dual_parameter_target.py` use the sequential default; two highdim tests
select sequential explicitly. Repository Python search finds pfor selection
only in historical exploratory benchmarks
`docs/benchmarks/ledh_k_batch_parity_and_timing.py` and
`docs/benchmarks/ledh_execution_mode_matrix.py`. Preserve those files/results as
historical provenance but retire executable entry before TensorFlow/GPU
initialization, with an explicit reason and path to this plan. Do not silently
run sequential when a caller requests pfor. Preserve the public argument to
reject unsupported modes clearly; remove the executable optional branch and
false approval claim. Add this whole adapter module to the existing AST guard
with no new exception. Its row-mapped design remains ineligible for NeuTra
training and still owes the registered score execution/cost qualification.

Baseline: frozen current committed adapter, default sequential only, using the
unchanged shared analytical score authority. Check default and explicit
sequential calls on fixed TensorFlow inputs, B=2,K=2,N=8,d=2,T=2,float64,
seed81100. Use one enclosing fixed-signature XLA owner and an independent
single-row/direction comparator; compare values/scores at unchanged1e-9 gates,
direction linearity and finite-difference checks where the existing fixture
supports them. Retain native-score derivative evidence for unchanged code.
Check pfor and unknown-mode rejection before tensor/callback work, historical
benchmark exit without TensorFlow import, the existing static governance gate,
source policy and call-chain classification. No pfor reference is executed.
Record traces/HLO/no host callbacks and dynamic theta/direction changes.

Allocation: up to8 sequential workers,1800 CPU/1200 GPU seconds inside the
unchanged global56 CPU/52 GPU process-hour caps, at most2 localized harness
retries. CPU is explicit reference. GPU numerical checks require growth/trusted
non-display selection; do not infer timing from a shared device. No new cost
comparison is claimed because the accepted sequential path is unchanged.
Unique artifacts use the existing registered campaign runner. Stop on drift,
wrong numerical/gradient program, incompatible callbacks, unexpected retrace,
missing device provenance or allocation exhaustion. Preserve failures and
repair only within the declared scope. No training, HMC, external publication,
package changes, tolerance relaxation, subagents or main merge.

Skeptical review: silently mapping pfor to sequential would conceal a requested
execution change; explicit rejection prevents it. Merely relabeling an admitted
branch diagnostic would not satisfy the static guard. Removing the gate would
hide the policy violation. Retiring the two exploratory programs before import
prevents stale GPU defaults and misleading new pfor results while preserving
source history. Source agreement alone cannot qualify analytical correctness;
keep independent score checks and native derivative evidence. Primary-agent
review only; no independent review asserted. No wider approval boundary is
crossed by enforcing the repository's existing pfor policy.
