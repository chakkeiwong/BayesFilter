# Pre-execution review

2026-09-22. PASS after checking the actual call chain and budget arithmetic.
The harness calls frozen `iapf_iterate`, whose wrapper calls frozen `iapf_apf`.
Its fitting callback returns the candidate guide, never the negligible-floor
diagnostic guide. Wiring checks require one fresh final filter call after the
stopping history and one fit per preceding iteration. Every capped or rejected
candidate remains a record. The score and QR methods are explicit extensions
to the paper's unspecified optimizer and Eq15, not author identities.

Ten case launches plus one preflight leave one repair slot in the twelve-launch
cap; corrected the plan's earlier statement of two spare slots. Two single-thread
CPU workers and 600 seconds per launch fit the three-worker-hour phase budget.
Python uses monotonic wall time for accounting. R subroutine timings use summed
user/system CPU time; they include diagnostic refits for the score arm and must
not be used as an equal-work algorithm ranking. An inspected phase14 d80 QR
elapsed-time field was negative; phase14 per-fit elapsed timings are unreliable.
Its Python process wall time and numerical results remain separate evidence.

Fresh data and learner labels, both doubling conventions, pinned sources, and
fixed caps prevent tuning on the previous final results. The numerical checks
are prerequisite controls; fresh final relative errors and conditional heuristic
comparisons answer the scientific question. An exact-oracle check is required in
every batch. Ten replicates cannot establish a population ranking or paper-table
replication. No baseline, target, stopping criterion, or numerical protection is
silently changed. CPU-only status is explicit. Source snapshots are checked
against consumed files at process exit. Raw clouds can be reproduced from seeds;
all fit guides, fitting diagnostics, controller histories, and final diagnostics
are saved without retaining every large particle array.

Focused syntax checks passed for both R files and the Python driver; relevant
tracked diffs pass whitespace checking. Next run: the preserved 48-fit parity,
finite-difference target-score, and rank/precision rejection preflight.
