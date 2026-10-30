# Separate attempt limits from compute time

Question: does the master enforce both limits while accurately reporting why
it cannot launch a phase? The observed defect is that campaign_remaining
returns zero seconds at the attempt cap even when the charged ledger has
positive unused time. This obscures the next action and can misclassify a
bounded attempt stop as compute exhaustion.

Repair after the active GPU worker ends. Compute remaining seconds solely
from authorized time minus charged or reserved attempts. Check attempt slots
separately before every launch in run and advance. Record the applicable stop
reason, remaining campaign time and remaining attempts in the phase report.
Neither limit changes. A count extension remains an explicit recorded decision
within the existing compute authorization, never an automatic expansion.

The primary criterion is executable CPU orchestration evidence: positive
seconds remain visible at the attempt cap, neither entry point launches when
either limit is exhausted, failures consume time, and unfinished reservations
remain charged. Baseline is the existing orchestration behavior with fake
workers. No GPU, sampler, score, tuning authority or scientific decision changes.
Malformed/nonfinite limits fail; no additional launch may escape the cap.
Run the complete focused master test module with GPU deliberately hidden.

Skeptical audit: simply removing the count check would create an unbounded
retry bug. Keep the count gate at both launch boundaries and test it directly.
Do not rely on a display-only assertion. Mock ledgers without count fields are
existing unit fixtures, not production ledgers; production reads remain strict.
Elapsed local phase time and campaign time are distinct and both must constrain
launch. Partial progress remains preserved if a later phase cannot launch.

Budget is CPU-only routine testing, with bounded test timeout. Save test output
in budget-reporting-cpu.log and results in budget-reporting-result.md. The work
is part of the already authorized master repair, not a new campaign or method.

Execution note: test fixture ledgers were updated to include valid attempt fields;
production validation was not weakened. All 128 focused tests passed.
