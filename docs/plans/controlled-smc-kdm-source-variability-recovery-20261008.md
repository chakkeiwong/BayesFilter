# Recover interrupted KDM variability supervision — 8 October 2026

Continue the authorized source-only variability phase under its unchanged
131400 CPU-second cap, seeds, frozen data, source methods and evaluation criteria.
No training is repeated, resumed or replaced.

Observed: the supervisor tool session returned 143 (consistent with SIGTERM).
The sender and reason are not recorded. Its last saved update is 11:27:55 UTC.
Unit 1 independently saved PASS result and terminal progress at 12:22:09 UTC,
with source early stopping after 132/499/268 epochs. Unit 2 already completed
training, its 80186-check audit, evaluation and all 16 fixed-TF replay controls.
At 13:06:51 UTC the supervisor and unit 1 process IDs were absent. Unit 1's actual
process exit code is unobserved; do not replace it with an invented zero.

Question: are the complete saved unit 1 training results valid, and can the
original final audit/evaluation/calibration be completed without retraining?
Comparator: the frozen source-only phase and unchanged source controller,
training checkpoints, data and common-author evaluator.
Primary completion: the full independent terminal audit passes, the evaluation
passes and reproduces all 16 fixed-TF replay controls, and both trained units
are included in the existing calibration summary.
Continuation vetoes: changed training/source/helper hashes, a still-live child,
missing or inconsistent complete artifacts, failed independent audit, changed
evaluation data/control outputs, or phase-budget exhaustion.
A failed audit/evaluation is preserved; no numerical bound or source failure is
waived. Float32 compatibility and noninferiority remain open.
Explanatory only: the supervisor exit signal and descriptive training variability.
Completion cannot establish trained-population equivalence or finish replication.

Recovery:
1. Preserve the original campaign/full manifests and status/checkpoint snapshots
   in a new recovery-20261008-01 directory. Verify all original launch hashes,
   completed stage records, exact terminal progress and absent process IDs.
2. Record unit 1 artifact completion with returncode=null and an explicit
   unobserved-process-exit marker. Use a conservative CPU charge from launch to
   first verified process absence at 13:06:51 UTC; record that this is an upper
   bound, not measured child runtime.
3. Keep original helpers unchanged. Save recovery-local copies of the auditor
   and evaluator, changing only repository path resolution and the process-exit
   prerequisite. The audit may accept recovered completion only after validating
   the archived original manifest, result and terminal-progress hashes. All
   numerical, full-protocol, constructor/data, optimizer/controller/checkpoint
   checks remain unchanged. The evaluator requires that independent PASS audit
   tied to the recovered manifest. Preserve exact source edits and hashes.
4. Run the recovered audit (120 CPU seconds maximum) and evaluation (300 maximum)
   in fresh unit01-audit/unit01-evaluation directories. Include at most 60 extra
   routine CPU seconds from the existing phase's 120-second overhead reserve;
   the phase cap remains 131400 seconds and GPU use remains zero.
5. Reuse the unchanged calibration and terminal accounting code, include the
   original completed unit 2, charge once, refresh master/checkpoint and review
   the terminal result. Preserve this recovery in the phase result.

Pre-mortem: a worker can save PASS before dying during exit; artifact validation
therefore substitutes only for the missing process-exit observation, not for any
training/data/controller correctness check. A new seed would change the scientific
question; it is forbidden. A stale RUNNING label must not be treated as a live job.

Skeptical review: proceed. This is localized supervisory recovery with independently
checkable complete artifacts; neither numerical acceptance nor the statistical
comparison changes. Unknown signal origin and unobserved child exit remain explicit.
Default audit: source method/data/seeds and metrics are frozen baseline choices;
artifact completion is a narrowly bounded recovery hypothesis tested by the complete
independent audit; conservative accounting is an upper-bound convenience choice.
No new learned-method ranking or heuristic comparison is made.

Command from /home/chakwong/BayesFilter:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/recover_kdm_source_variability.py

Artifacts: the existing phase root's recovery-20261008-01 directory, new unit 1
audit/evaluation directories, and the phase summary/result note.
