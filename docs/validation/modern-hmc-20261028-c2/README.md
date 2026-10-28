# C2 source compatibility and oracle repair, 28 October 2026

Clean candidate: f9291ee4437aa8c00dabbe331bffc6871b574c5c.
The 11-file candidate patch reconstructs C1-loaded source bytes on c41c38941.
All 70 prior loaded-module hashes and current integration imports match C1.
Preview-01 preserves three failures; preview-02 preserves the unsuccessful
full-XLA oracle repair. Preview-03 passes all 33 checks in a temporary consumer
overlay. Preview-04 passes the same 33 checks after installing the tested patch
in MacroFinance. Both exercise the actual clean-source gate with explicit
process-local authority inputs, so neither is official consumer admission.
Core-regression-01 records optional custom-op preload failure. The retry
turns off only that test preload and passes all 36 selected core regressions.

The consumer patch changes only the independent reference's compilation
boundary and its tests. Analytic target routines, tolerances, runtime pin and
root are unchanged. It is installed but not committed in MacroFinance, which
has pre-existing unrelated changes. This package preserves the exact patch.

No posterior, statistical ranking, default, GPU or release claim follows.
Original logs and test copies remain under the plan artifact root. The measured
cost file states the incomplete earlier host-time accounting explicitly.
