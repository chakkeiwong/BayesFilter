# Phase 16 execution review

2026-09-22. The reviewed plan is
`docs/plans/iapf-adaptive-replication-ladder-2026-09-22.md`.

Decision: execute the frozen 100 / 300 / 1,000 replicate ladder. The
preflight passed for dimensions 5 and 80: both learners' histories, counts,
fitted guides, fresh final filter results, and recorded filter diagnostics
exactly reproduce the saved phase 15 runs. All four heuristic controls also
reproduce exactly. The independent statistics check passed all 180 checks.
Three supervisor regressions passed, including continuation after capped
candidates, one same-seed timeout retry, repeated-timeout stopping, and budget
reservation/charging. R syntax checks passed.

The first replay checker rejected integer versus floating-point storage of an
identical data seed. Repair 01 preserves that failed attempt and its charge.
The corrected checker compares seed values and still requires exact numerical
arrays, histories, guides, and filter results. No numerical tolerance or
scientific criterion was relaxed.

The final call-chain inspection confirms that the batch invokes the frozen
`iapf_iterate` controller and `iapf_apf` consumer, including the fresh final
filter call after stopping. The score and QR fits remain explicitly identified
extensions. The failed score/FA comparison at dimension 40 remains in the
campaign; no observed accuracy result selected the retained fitter. The
later-doubling convention is an explicit reconstruction hypothesis, and the
earlier-doubling contrast remains in phase 15.

Skeptical audit: source checks, conditional evaluation by dimension, a strong
classical comparator, exact Kalman certification, preserved failed labels,
fresh data/seeds, and fixed stage accumulation answer the declared question.
The bootstrap intervals quantify conditional Monte Carlo uncertainty; unseen
rare importance weights can still make them too narrow. A small observed
error, a narrow interval, or an oracle check cannot identify the author's
implementation, establish Eq. 15 fidelity, or close full paper replication.
No production default or TensorFlow/GPU claim follows from this CPU R lane.

Execution limits are two single-thread CPU workers, 36 charged worker hours,
520 launches, and the remaining campaign budget. Each normal batch reserves
900 seconds; one unchanged-science timeout retry may reserve 1,800 seconds.
Source, numerical, artifact, repeated infrastructure, and budget failures stop
with a checkpoint. Candidate rejection or underperformance continues to the
next scheduled measurement. The supervisor writes each stage summary and
advances automatically without another approval.

Do not edit the source set recorded in `preflight.json` while the supervisor
is running. Current state and remaining budget are in `checkpoint.json` and
`budget.json`; every launch preserves its command, source hashes, outputs,
hardware choice, seeds, and elapsed time.
