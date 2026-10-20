# Ordered proposal-attempt execution checkpoint

The sequential initializer now runs ordered factor proposals, conditional second
fitting, exact-incumbent selection, acceptance, radius updates and stall updates
inside a native TensorFlow/XLA loop. The first fit remains at its original call
site. Completed histories are materialized once per column. The outer refinement
and terminal-fit lifecycle are still host controlled and remain open work.

The comparator is Git checkpoint `93c8e419`, including its recursively loaded
numerical import closure. Canonical LEDH rebuilding remains excluded. No target,
gradient, seed, optimizer setting, acceptance rule or comparison tolerance changes.

Qualification includes CPU/GPU attempt mechanics, actual 200-iteration two-factor
fitting, all numerical runtime operands, post-cache-eviction execution and actual
consumer-resource release. Complete public histories pass on CPU/GPU, including
real rejected-one-factor/usable-two-factor escalation. All 12 original factor,
40 sequential and 43 block consumer checks pass. Reporting changes preserve all
ten complete public histories. The final policy/controller group passes all 67
cases in 01720. Focused Ruff, whitespace checks and the partial source guard pass
(195 sources, 1,276 existing exact exceptions; no numerical-loop exception added).

The corrected fresh-process cost comparison is
`artifacts/filter-gradient-repair-20260917/attempts-memory-comparison-01714.json`.
GPU2 is an RTX 4090 with verified growth, TensorFlow 2.19.1, TF32 enabled and
binary64 numerical inputs. Every run has its command, environment, source hashes,
input hashes, elapsed time and output paths in its numbered manifest. All
frozen/current XLA fields are exactly equal for both D3 and D5. One cold and
20 synchronized warm calls give these descriptive observations:

| Scope | Frozen warm median | XLA warm median | Extra peak host memory | GPU allocator peak, old/new |
|---|---:|---:|---:|---:|
| D3, one factor, capacity 4 | 3.668 ms | 3.804 ms | 1.926 MiB | 22,016 / 33,536 bytes |
| D5, two factors, capacity 32 | 113.233 ms | 113.524 ms | 287.000 MiB | 195,840 / 219,904 bytes |

The first comparison (01702) found a 90.7% D3 warm regression from eager tensor
slicing during reporting. Column materialization removes that regression at the
declared 20% investigation threshold. Correcting an asymmetric frozen/current
setup namespace does not remove D5 host overhead; that hypothesis is rejected.

The separate four-process capacity diagnostic (01716--01719) preserves exact
records at both capacities and localizes overhead to cold tracing/compilation.
Host peak differences are 191.000 MiB at capacity 4 and 290.738 MiB at capacity
32. The enclosing graph adds 841 TensorFlow nodes and 2,160 HLO instructions
at either capacity; both versions retain the same two active-size dispatches
(5 branches at capacity 4, 33 at capacity 32). Warm host growth is only
12--20 KiB across the three diagnostic calls and device current allocations
remain constant. This supports a bounded observed compilation cost, not a
general memory cap or proof about arbitrary process lifetimes. Artifacts and
exact analysis are preserved in `attempts-capacity-disposition-01719.json`.
Terminal process repeats and further enclosing-controller costs remain required.

Strict graph/XLA equality remains **failed**: six D3 and 16 D5 fields exceed
unchanged `atol=rtol=1e-10`. Diagnostic 01708 reproduces the D3 discrepancy in
both frozen and current proposal code. The XLA eigensystem residual is
`7.684e-9`, while graph GPU/CPU references are `3.33e-16`/`4.44e-16`, despite
the existing machine-precision request. Same-mode frozen/current results agree.
The earlier D5 fitter discrepancy remains separately documented. These failures
are inherited numerical problems, not waived comparisons or corrected results.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Preserve execution checkpoint | Native loop and complete original XLA records pass | No changed decision or record | Enclosing lifecycle still host controlled | Commit/push; repair outer lifecycle | Complete repository repair |
| Retain measured resource cost | Cold cost localized; warm allocations stable in tested scope | 256 MiB host trigger investigated, overhead remains | Compiler retention at larger capacity/lifetime | Include in terminal and enclosing comparisons | Leak freedom or arbitrary-capacity safety |
| Keep graph/XLA numerical gate open | Strict comparison fails | Inherited eigensolver/fitter discrepancies | Required numerical repair must preserve intended target | Investigate without silently changing tolerance | Numerical equivalence across modes |

| Inference status | Disposition |
|---|---|
| Hard veto screen | Default XLA preserves frozen fields; graph/XLA strict comparison fails |
| Statistically supported ranking | None |
| Descriptive differences | Single-process warm time, cold cost and observed allocator/RSS |
| Default readiness | Not established; outer lifecycle and terminal gates remain |
| Next evidence | Complete enclosing controllers, numerical repairs and terminal repeated comparisons |

Self-review checked real callback wiring, exact scalar order, skipped second
fits, best-exact promotion independent of acceptance, unavailable-last-proposal
semantics, complete records, resource ownership, source pinning and the asymmetric
setup error. The weakest resource evidence is the small number of capacities
and processes; similar measurements with accumulating warm allocations would
overturn the current bounded-cost interpretation. Independent terminal review is
still pending. Main is unmerged.

Through 01720 the campaign charged 33,426.027247090 GPU and 40,127.291836184 CPU
process-seconds against unchanged caps of 187,200 and 115,200 seconds. Remaining:
153,773.972752910 GPU and 75,072.708163816 CPU seconds. No worker is active at
this checkpoint; the extension has been counted once.
