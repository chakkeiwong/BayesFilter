# Real one-iteration optimizer context: positive reproduction

Original04635 and candidate04636 pass in215.446 and217.037 CPU seconds.
Each retains the real TFP L-BFGS implementation with only `max_iterations=1`,
performs three objective batches, and saves one-trace/no-host-callback HLO and
complete callback arrays. Readback/policy04637 passes162 checks in18.607
seconds, including the prior truncated-dispatch readback. The separate unit
uses three of six workers and451.090 CPU seconds, with no failed run.

Both arms reproduce their own historical04584/04585 first-two-row inputs,
values, scores and validity bytes exactly. Their scores differ across arms by
the historical1.5276668818842154e-13, reaching1952 ULPs in one small coordinate.
Input, value and validity bytes match across arms. This is a positive smaller
reproduction of the first-score discrepancy; hundreds of optimization
callbacks are unnecessary to observe it.

Together with the negative fake-dispatch result04632--04634, this localizes
the needed context to a program containing the real optimizer. It does not
identify the specific arithmetic/compiler operation, establish a faulty target
formula or repair runtime behavior. The full optimizer trajectories and121
strict record differences remain unqualified. No threshold, numerical method,
default architecture or scientific admission changes.

| Decision | Primary criterion | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Use the one-iteration diagnostic for localization | Both first score records match their historical controls exactly | Exact arithmetic source of the difference is unknown | Inspect saved compiler graphs and define bounded graph bisection | Runtime repair, equivalence or convergence |
| Avoid another full trajectory | Three objective calls reproduce the first difference | Later optimizer amplification remains separate | Preserve full histories as controls | Waiver of121 strict record differences |

Post-run review: preserving a real optimizer body is sufficient in these two
programs, but does not imply the optimizer's mathematics is wrong. Compiler
context and input representation remain alternative explanations. Exact
first-row source/input checks are the main control; a reproduction on changed
inputs would not answer the question. The weakest remaining evidence is
operation-level attribution. No independent reviewer was used.

Saved-HLO readbacks add4.249 and4.129 CPU seconds. The first ordinal comparison
is confounded by helper declaration ordering and is retained as a diagnostic
lead only. The refined reader matches6023 of6100 computation bodies per arm
by normalized content and multiplicity, including callee content hashes;
77 per arm remain unmatched. Shapes, constants, attributes and operand
relationships are retained, while SSA names and source metadata are removed.
All referenced computations resolve; text fixtures detect changed arithmetic
and propagation of a changed callee into its caller's hash.

Inspected unmatched sum helpers use signed32 parameters/results in the original
and signed64 in the candidate. Candidate source explicitly reserves int64
for accounting resources because of GPU placement requirements. This supports
the next CPU-only accounting-width diagnostic, not a production counter
change. Neither77 unmatched bodies nor the5951 equal ordinal pairs from the
first reader are counts of numerical errors. The total unit charge including
both saved-HLO readbacks is459.467 CPU seconds.
