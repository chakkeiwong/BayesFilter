# Archived sequential-event compatibility repair

Question: can the saved predator–prey and SIR chains resume without repeating or changing numerical work? Comparator: the archived states, seeds, chunk tensors, health and stopping decisions. Pass: exact agreement of these quantities after replay. Veto: numerical-kernel source drift, changed states/seeds/counts, changed health or stopping decisions. R-hat roundoff remains explanatory only when decisions agree. No new posterior claim follows from this repair.

Skeptical audit: the first attempt rejected an old event format that lacked force diagnostics and included a healthy no-force-fallback check. This is an orchestration incompatibility, not a chain failure. The repair accepts only the exact archived source pair, checks that actual replay has zero force fallback under the recognized deterministic-force policy, and preserves all other comparisons. Current-format diagnostics remain strict. Complete pre-repair modules are preserved in source-before-replay-repair; the archived resume source is also preserved in source-before.

The scientific target, endpoint, force, seeds, thresholds, numerical settings and 48-hour campaign deadline are unchanged. Focused tests must pass before a new retry. Failed attempts and retries use distinct output directories and consume the original campaign budget.
