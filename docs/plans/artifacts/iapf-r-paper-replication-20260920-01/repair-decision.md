# Diagnostic repair and renewed replication

The original fresh-data batch completed four repeats, then stopped on repeat
105. A replay preserved the first failed fitting problem (backward time 44).
The profiled equation-(15) loss and all its floating-point derivatives reached
zero after the fitted density escaped from the data; squared densities also
underflowed in a diagnostic ratio. This was not an invalid Gaussian covariance.

Three profiled and two joint solver probes, saved in `mechanics02/`, did not
establish a solver replacement: four showed an escape toward negligible
density; joint BFGS retained a small relative residual but failed to converge
within its fixed budget. These probes cannot support a change to the objective.

The applied repair changes only the scale-invariant relative-residual
diagnostic. It also records log loss and underflow. The objective, gradient,
optimizer and returned fit parameters are unchanged. An executable regression
checks parameter equality to the saved failed fit to 1e-12; the formerly failing
full repeat now completes (4.23 s) and remains debugging-only evidence. A new
pilot on data seed 56000005 completed in 6.17 worker seconds. The four pytest
cases pass: 66 R identities/behavior checks and three executed-source mutations.

Freeze the repaired source before fresh replication data seed 57000005, 32
repeats starting at 201. Request the same 32-repeat target at d=10 on data seed
57000010, starting at 201, within a 300-second worker cap; a partial batch is
capacity evidence, not a 32-repeat result. Continue any missing repeat only in
a fresh directory with the same seeds and source. Reporting uses 2000 paired
bootstrap resamples of complete Monte Carlo repetitions; pilot observations
and the earlier interrupted dataset are not pooled with renewed replication.

The first four workers consumed 102.55/1800 seconds. Four launch slots remain.
There is no filter-validity veto from an inaccurate but finite fit alone:
Algorithms 3--5 allow approximate twists, and the positive-floor importance
weights remain exact for the proposal actually sampled. A nonfinite filter,
failed covariance or changed mathematical target still vetoes continuation.
