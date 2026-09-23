# Staged locator cost and rounding diagnosis

The internal XLA candidate passes both D1/D3 complete original3582b4ac record
comparisons in03332/03333, including changed operands. Both graph arms fail the
same original-XLA comparison in03331/03334; retain those failures and exclude
them from equivalent-result cost conclusions. The original arms03329/03330
pass. All six cost processes use identical source/environment signatures and
matched configuration/input hashes within each dimension. This is one process
per arm/extent, with three warm calls, under the
[bounded diagnostic plan](filter_gradient_staged_locator_cost_unit_20260923.md).

| Dimension | Original cold s | XLA build+cold s | Original warm median ms | XLA warm median ms | Original final observed RSS MiB | XLA final observed RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.948 | 5.629 | 5859.261 | 3.568 | 2724.492 | 1180.211 |
| 3 | 6.033 | 5.661 | 5890.435 | 3.440 | 2742.270 | 1194.648 |

The original constructs and compiles stages again for each public invocation;
its RSS grows by approximately373--387MiB per warm call and its map count keeps
increasing. The candidate reuses two programs: each traces once and retains
unchanged HLO for changed numerical operands. After its cold call, observed RSS
increases by0.008/0.016MiB across the three warm calls, then remains unchanged at
the changed-input observation. Its map count remains4408/4674. No relative
cold, warm or host-RSS cost trigger fires against the original. These short
observations support reuse for the tested signature, not a long-term memory
bound or eviction claim.

The graph diagnostic reaches about832MiB RSS; the XLA arms reach1180/1195MiB.
This is explanatory memory evidence from two differently rounded executions,
not a qualified identical-result compiler ablation. The XLA executable/native
residency concern remains in E4. All arms also report an initial rusage high-water
value near2.28GB while current-process VmHWM is about605MB. That mismatch makes
rusage unsuitable for attributing this call's memory. The table and all cost
triggers use observed `/proc/self/smaps_rollup` RSS; they do not claim measured
peaks or equate device reservation with tensor allocation.

Run03335 diagnoses D1 without changing runtime arithmetic or comparisons. Both
original and candidate use optimizer_callback in graph mode and checkpoint_replay
in XLA for the initial operands. The standardized coordinate is-0.575 in both.
Independent100-digit arithmetic gives a fused affine position
0.13999999999999999 versus the separately rounded0.14. XLA's callback objective
is-5.277048047621027e-34 while replay is zero; graph's callback and replay both
equal zero, so the original strict-order tie policy selects the earlier callback.
Changed operands also match complete original records within each setting.
Thus03331 is an inherited graph/XLA finite-arithmetic distinction, not a changed
candidate tie rule. D3's cross-setting source mismatch remains preserved; no
per-operation D3 attribution or discrete-field waiver is claimed.

Separately,03328 passes real checkpoint/continuation compiler-failure isolation
and nested-validator reuse of the same owner, raising the staged unit to26
distinct CPU checks. Compiler failures never run a Python fallback, do not cross
the validator boundary improperly, and release the invocation lock. Nested
execution preserves both complete original records and restores the outer
continuation's supplied state. The public legacy error-record disposition and
public ownership integration remain open.

Analysis and source:
`artifacts/filter-gradient-repair-20260917/staged-center-costs-cpu-03334.json`
and `staged-center-cost-analysis-03334.py`. The analyzer verifies source/input/
configuration identities, full records, JUnit and CPU provenance, preserves all
failure fields and rejects the graph arms for matched costs. Policy03336 passes
129 checks with229 guarded sources/1333 exact exceptions and no new waiver.

Review: construction is included in cold time; tensors and full results are
materialized before timing stops; original parity and HLO exports occur after
memory/timing observations. The large warm improvement includes avoiding
original recompilation and must not be described as an identical-program XLA
speedup. Short CPU measurements cannot close GPU, public integration, long-term
residency, terminal cost, HMC or main-merge gates. The08:41UTC GPU preflight
declined under contention before launching a worker; it consumed no numbered
numerical attempt. Continue GPU qualification when the existing selector admits
a device, preserving the user's desktop restrictions.
