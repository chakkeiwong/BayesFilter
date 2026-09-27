# Filter and gradient repair resume checkpoint

Checkpoint through 04527. No worker is active. Complete posterior-initializer
costs pass all 36 fresh CPU/GPU arms (04490--04525), analyzer 04526 passes 14,
and policy 04527 passes 147. All accepted arms share 3441 source hashes and
GPU 2. Both analyses reproduce exactly. Reopened evidence archive verifies
214 members; see filter_gradient_posterior_initializer_cost_result_20260927.md
and artifacts/filter-gradient-repair-20260917/posterior-initializer-cost-04527-verification.json.

Default XLA crosses no cost-regression trigger on these D1/D3 factor_max=1
fixtures. CPU warm medians: prior9.505/20.581s, XLA0.061/0.096s; GPU:
prior17.155/47.609s, XLA0.093/0.190s. Ratios are descriptive. Graph-control GPU
allocator peaks trigger2x attribution (1.236/1.302MB versus0.553/0.568MB prior).
No waiver: longer graph/XLA reuse and owner-return/payload memory follow next.
Actual DZ5 consumers, reporting/precision, native capacity and terminal
F01--F20 decisions remain open. Main remains unmerged; no tolerance change.

Next install /tmp/install_posterior_capacity_harness.py after this checkpoint
is committed. It copies the three /tmp/filter_repair_posterior_*capacity*draft.py
files and registers focused groups, plus a narrowly bound1200-second option
for the D3 GPU prior's twenty-call check. Run analyzer/timeout checks first,
then a D1 XLA CPU pilot. Follow filter_gradient_posterior_initializer_capacity_20260927.md:
7200 combined seconds /20 workers reserved inside unchanged global caps.
Prior D3 GPU costs47.6s/call, hence1200seconds; D3 GPU owner replacement uses900.
Other tests use300 except other prior reuse workers at900. Twenty alternating
warm calls and four successful-owner replacements have complete-record and
independent-Gaussian vetoes. Observer-only controls and graph-return snapshots
separate retained and transient allocation without native-eviction claims.

The source remains based on pushed976c33552, including fetched origin/main
06590cb5a (unchanged). Only cost harness version reporting changed; runtime is
byte-identical to qualification04488. Failed04489 metadata attempt is preserved.
Cost unit charged4227.008302/10800 seconds in39 workers including two checks.
Campaign charged94229.965768 CPU /87999.752087 GPU seconds; remaining29.825010
CPU /27.555624 GPU hours under56/52 caps. Extra24CPU hours are already included.
No subagents. Public correctness:13 XLA cases/backend including D5 two-factor,
then11 CPU/11 GPU graph/default renewal; 276-source guard/1436 exact allowances.
Canonical LEDH rebuild stays excluded and unsupported claims stay blocked.
