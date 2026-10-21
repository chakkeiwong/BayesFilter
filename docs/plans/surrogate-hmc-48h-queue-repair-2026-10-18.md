# Continuation queue repair, 2026-10-18

This addendum repairs the existing authorized 48-hour campaign; it does not
start a new budget. The original deadline is 2026-10-20 08:00:41 UTC and the
shared ledger remains capped at 172800 worker-seconds and six attempts.

The question remains whether the five frozen surrogate-HMC configurations
meet their recorded adaptive sampling rule. LGSSM and KSC SV results are
preserved. Range bearing is already running. Predator–prey's first attempt
failed while comparing an older saved diagnostic schema, before new sampling;
the narrowly bounded replay compatibility repair passed 46 focused tests.
The comparator is each model's saved chain state and frozen target/kernel,
not a new method or a new posterior. No numerical kernel or scientific default
is changed by this queue repair.

Preserve the live range-bearing worker. Replace only its Python supervisor,
then retry predator–prey in a fresh directory for at most eight hours; give
SIR the remaining shared time. Preserve every attempt, source snapshot, and
the original absolute deadline. The successor refreshes next-phase-plan.json
between phases. It must not restart range bearing or reset budget accounting.

Primary criterion: the recorded cumulative modern R-hat <=1.01, subject to
existing sampling maxima, warm-up readiness and health vetoes. Health failures
veto the affected candidate; they do not prohibit another already planned
model. Missing/corrupt artifacts, overlapping workers, or an inconsistent
ledger invalidate orchestration and stop it. Acceptance, ESS and runtime are
explanatory. Passing does not establish exact-model posterior agreement,
heuristic dominance or superiority. This orchestration repair does not add a
new scientific comparison; the original evidence contract remains controlling.

Assumption audit: eight-hour predator–prey allocation is an inherited resource
cap, not evidence that it suffices for convergence. SIR receives unused time
as previously authorized. Existing tuned settings remain frozen within their
own scopes; no cross-model setting is transferred. The replay repair changes
only healthy legacy event comparison, not states, seeds or force evaluation.
The early checks are source pins, saved queue/ledger consistency, separate
process groups, and focused CPU process-handoff and allocation regressions.

Skeptical audit: editing the old queue file would not update its in-memory
loop, so a parent-only handoff is required. The worker owns a separate process
group; tests must show that terminating the old parent preserves that worker.
The successor reads the same deadline and ledger, uses a fresh retry output,
and never infers successful inference solely from process disappearance.
The main failure risk is losing supervision during handoff; validation occurs
before termination and a pre-termination failure resumes the old supervisor.
An artifact/ledger failure after takeover is recorded and stops execution.

Execution: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
scripts/run_surrogate_hmc_48h_queue_repair.py launch` (trusted permissions).
CPU tests intentionally hide CUDA. Evidence is stored below the existing
campaign's `run/queue-repair-01`, in `state.json`, `next-phase-plan.json`, the
unchanged shared `budget.json`, and fresh per-model outputs. The handoff
records source hashes and old/new supervisor PIDs; the active worker PID and
original deadline must remain unchanged. No additional GPU time is authorized.

Pre-execution audit outcome: passed. The focused queue tests and original
extension regressions passed together (20 tests in 3.19 seconds;
`artifacts/surrogate-hmc-48h-extension-20261018-01/queue-repair-tests.log`).
They cover parent-only termination, cancellation on both sides of takeover,
preservation of the separate worker, adopted-worker deadline enforcement,
process identity, unchanged retry arguments except the output directory,
honest recording of an unobserved exit code, and rejection of unfinished
ledgers or nonterminal results. The 46 replay-repair tests are separately
preserved in `replay-repair-tests.log`. These checks establish orchestration
and replay compatibility only; convergence remains the sampling question.
