# Campaign execution and skeptical review

The owner's new24-hour authorization was recorded before new workers launched.
It adds a fresh elapsed window and172800summed-worker-second ceiling at two
single-thread R workers. Old allocations remain separate. The campaign starts
2026-09-20 20:04:26 UTC and must end by2026-09-21 20:04:26 UTC.

Phase1 completed8replicas,40method runs and10400fits on dataset89500080.
All mandatory fit/tail/seed/data/record checks pass. The six-estimate controller
has mean ratio.98075,95%interval[.88623,1.09644],SD.16135 and mean N2000;
it fails the practical particle-count criterion. Window5 has mean1.02021,
interval[.87999,1.17924],SD.23660 and mean N1000, passing the practical screen
and observed conditional heuristic screen on this dataset. Both fail the
separate literal-pattern screen. Paired terminal squared-error difference is
.02624,interval[-.01060,.07375]; no accuracy ranking is established. Window5
saves7.802seconds on average,interval[-8.653,-7.085], under this run's timing.
The first dataset's practical-screen failure remains preserved and motivates
the already planned larger fresh-data comparison; this result does not erase it.

Phase2 is frozen at64replicas each on new datasets89600080 and89700080.
Phase3 is frozen before data generation at1000replicas in each of five
dimensions, both QR windows plus BPF/FA/SIS, with seeds and criteria in the
dated first-study plan. Retaining both variants characterizes the proposed
controller mechanism without selecting the arm that best matches paper tables.

The source-anchored model/count/table checks found no need to change the
numerical algorithm. All running phase2 workers use their captured sources.
For phase3, completion rows will be written after mandatory prefixes; this
closes a timeout window in which a row could previously claim completion before
its prefixes reached disk. Failure-text filenames also include the method, so
two failed arms in one replica preserve separate text diagnostics (their RDS
files were already separate). No computed filter quantity or RNG call changes.
The algorithm core, reconstruction and tail-diagnostic files are unchanged.

Focused engineering checks: five pytest tests pass (supervisor-regression-v2.log):
bootstrap parity against the preceding reporter; paired constant differences;
timeout preserving completed pairs and charging both attempts; restart avoiding
repeated completed work; duplicate evidence rejection; and candidate rejection
versus missing-evidence phase progression. Parameterization gives five tests.
Python compilation checks pass. Phase1 provides an end-to-end real R execution
and reporter check. The diagnostic reporter uses NumPy solely to accelerate
bootstrap analysis; it never supplies arrays or settings to the R algorithms.

The autonomous supervisor waits for phase2, validates and summarizes it, launches
the frozen full study, summarizes the study and writes terminal decisions plus
master/checkpoint status. It does not interpret a failed reference screen as
authority to stop the next characterization phase. An invalid shared authority,
corrupt evidence or a real budget boundary still stops dependent work. An
unresolved supervisor failure remains explicit and recoverable from preserved
attempts. The supervisor JSON records its PID, heartbeat, child and exact command.

Review limitation: this is focused self-review and executable checking; no
independent reviewer was invoked or implied. The scientific limitations remain
QR's different objective, unknown author data/settings, pointwise uncertainty,
conditional fixed-data inference, and unmatched cost. No production/default,
LEDH/KDM/GPU/HMC validity claim is made.
