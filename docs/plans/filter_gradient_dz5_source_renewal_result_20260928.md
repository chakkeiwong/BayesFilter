# Actual DZ5 consumer source renewal result

The r2 snapshot freezes the numerical source at4bf50d914, including the SVD,
clipped-anchor and final-precision symmetry repairs. Its manifest SHA-256 is
`1368a2e983d7539d5f3fbd7fe058e197c8dcd2c0b4fe8f9fa2bc3aa47a44ee72`.
It contains732 source files and3 frozen input files. Against r1, exactly two
source files differ: `factor_correlation_geometry.py` and
`fixed_center_curvature.py`. All input hashes are identical. The isolated
external adapter is copied from r1; live MacroFinance sources remain outside
the test.

CPU target comparison04618 passes in65.955 seconds. It checks the actual
23-parameter CDF target at batch4, changed inputs, exact replay, invalid-row
isolation, graph-reference value/score bounds, XLA HLO and loaded-source
identity. The independent CPU score oracle04619 passes in258.657 seconds.
GPU target batches1,4,46,68 pass04620--04623 in60.182,63.658,66.004 and70.466
seconds. The GPU score oracle04624 passes in49.421 seconds. These elapsed
times are descriptive, not a matched ranking.

The controlling plan is `filter_gradient_dz5_source_renewal_20260928.md`.
New run directories preserve commands, hardware/environment, source hashes,
logs and full results under the existing raw campaign root. The admission
builder validates all seven fresh runs and issues
`dz5-initializer-adapter-target-admission-20260928-r2.json` for this snapshot.
CPU lifetime04625 passes: two complete accepted initializer workers under
the actual supervisor, with the existing3600s parent/1700s child bounds. This
independent accepted/lifetime check runs before the GPU rejected case; their
common prerequisite is fresh target admission, and neither substitutes for
the other.

The first CPU lifetime child now completes accepted execution: status
`usable_dense_local_initializer`,822 exact evaluations, one compiled trace and
no host callbacks. Its observed initializer time is857.933 seconds, compared
with856.468 in the prior r1 accepted worker04572; these are unpaired single-run
observations. Peak RSS is24225532KiB versus24231084KiB previously. After Python
owner release, observed owner count is0 while RSS remains about22.56GiB. This
reproduces the earlier native/compiler retention and demonstrates why Python
collection cannot certify native memory release.

Both CPU children complete and are reaped in958.392/948.761 seconds. The second
initializer takes848.915 seconds. Both full accepted records (excluding the
separately recorded execution telemetry) match exactly, with822 evaluations.
Parent RSS is540438528,540868608,540868608 bytes before and after the two exits:
an initial430080-byte (0.410MiB) increase followed by no further growth. All
three parent map counts are2452. Each child has one compiled trace, no host
callbacks and zero observed Python owners after release. The combined worker
charge is1911.069 seconds. These observations pass the declared256MiB parent
growth gate and confirm process-exit containment for this two-worker case;
they do not prove in-process native eviction or arbitrary signature capacity.

GPU lifetime04626 is active with the existing1800s parent/850s child limits.
GPU rejection and final saved-evidence/policy readback follow.

Inspection of the preserved04617 comparison further identifies the4539 strict
CPU/GPU fitted-record differences:62 selection/stability leaves and1663,1150,
1664 leaves in public fits3,4,5. The selected precision, covariance, center and
audit result pass the existing comparison bounds. Fits3/4 reject on holdout;
fit5 passes its own fit gate but is unconverged and is not selected. Both
backends retain the same fit statuses. This localization does not waive any
complete-record criterion or establish optimizer convergence.

| Decision | Criterion status | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Accept fresh target-only evidence | All seven CPU/GPU target/oracle checks and saved-evidence builder pass | Actual initializer/lifetime validation remains open | Complete CPU/GPU consumer lifetimes and rejected case | Complete initializer equivalence or HMC admission |
| Keep main unmerged | Final precision repair is qualified separately | Strict full records and terminal master checks remain open | Execute the actual consumer and lifetime checks | Whole-program completion |
