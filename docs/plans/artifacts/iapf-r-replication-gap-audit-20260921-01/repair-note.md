# Localized sensitivity-harness repair

The first sensitivity attempt completed in 1.269 seconds. Inspection found a
confound: multiplying box width also multiplied L-BFGS-B's mean-parameter
scaling. Thus differences between radii mixed constraint and optimizer effects.
This did not affect QR validation or the radius-one end-to-end pilot.

Repair: keep parameter scaling at the initial QR standard deviations for all
box widths. The mathematical objective, tested radii, optimizer tolerances,
initialization, data, seeds and evidence criteria remain fixed. Preserve attempt01
as confounded diagnostic evidence; use a fresh captured-source attempt07 for the
corrected sensitivity comparison. Reserve its full120-second timeout from the
existing2400-second worker allocation. The old default (radius1) is unchanged.
Skeptical repair review: this removes an unintended factor and does not select
on validation outcomes. Candidate failure remains separate from harness failure.
