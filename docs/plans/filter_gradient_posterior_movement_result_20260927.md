# Posterior movement recurrence result

The internal prepared-cloud movement controller passes eight distinct cases on
each CPU/GPU backend. Movement attempts, nomination replay, exact incumbent
selection, stopping decisions, candidate ledgers, score norms and complete
geometry history execute inside one XLA program. The public posterior initializer
is not yet wired to it.

The comparator executes the exact recorder and movement-loop statements from
031692a0b. All 690 existing BayesFilter Python sources in that commit are checked
byte-for-byte before using their imports. Only prepared-cloud delivery and
completed result capture are adapted. Complete records and ordered physical
callbacks agree at the existing tolerances; tracker counts and discrete statuses
agree exactly. Changed inputs reuse one trace and stable HLO, same-input replay
is exact, and returned derivatives are frozen.

| Case | Observed original and repaired behavior | Evidence |
| --- | --- | --- |
| D1 scalar Gaussian | Completes movement | 04337 / 04344 |
| D3 batched Gaussian | Movement-attempt exhaustion | 04338 / 04345 |
| D1 batched nonlinear | Movement-attempt exhaustion | 04339 / 04346 |
| D3 scalar budget | Exact-evaluation budget rejection | 04340 / 04347 |
| D1 batched invalid cloud | Same invalid-row accounting; movement still completes under original rules | 04341 / 04348 |
| D1 batched eligibility mismatch | Eligibility-contract rejection | 04342 / 04349 |
| Stationary Gaussian | Stops after exactly two successful fits | 04343 / 04350 |
| Unsupported XLA target operation | Raises; no eager fallback or target execution | 04343 / 04350 |

04334 failed while constructing a history scatter for a statically empty D1
rank-zero basis. Skipping only empty schema fields repairs that construction;
no computed numerical element changes. Initial passes 04335/04336 preceded
native ledger-norm outputs and frozen-gradient assertions. Their runtime and
harness snapshots are preserved alongside the failed candidate. Policy 04351
passes 141 checks, with no added allowance.

The movement substep used 11 CPU and 7 GPU workers, charging 434.429236 CPU and
551.192574 GPU seconds within its reservation. GPU 3 sharing is recorded;
these results establish tested correctness only, not accepted timing evidence.
The archive receipt `posterior-movement-verification-04351.json` reopens and
verifies all 82 members, including every attempt, source snapshot and log.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Retain internal recurrence | Complete same-mode records and callbacks pass on CPU/GPU | Public integration and curvature remain untested | Enclose curvature using shared authorities | Whole-initializer repair |
| Preserve exhaustion and invalid-row outcomes | Original statuses and counts retained | Finite Gaussian arithmetic does not imply accepted initialization | Continue complete endpoint comparisons | Posterior or HMC validity |
| Keep performance gate open | No performance experiment in this unit | Shared GPU and unintegrated owner | Matched costs after public integration | Memory bound or speed ranking |

Skeptical result review: the reference shares unchanged numerical dependencies,
so these tests isolate execution of the enclosing movement recurrence. They do
not renew the campaign's oldest-original numerical comparisons. D3 Gaussian
and nonlinear cases exhaust their movement allowance in both implementations;
calling them accepted initializers would be wrong. The weakest remaining
boundary is composition with curvature and the full exported endpoint. No main
promotion, canonical LEDH, learned-map quality or convergence claim follows.
