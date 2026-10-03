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

GPU lifetime04626 passes in1635.950 seconds with the unchanged1800s parent/
850s child limits. Both accepted children are reaped normally in818.209 and
813.654 seconds. Their complete initializer records match exactly, with789
exact evaluations each. Initializer times are708.222 and703.213 seconds.
Each has one trace, no host callbacks and zero observed Python owners after
release. The parent RSS sequence is738201600,738643968,738787328 bytes:
585728 bytes (0.559MiB) total growth, with2498 mappings throughout.

Both GPU children have TensorFlow allocator peak269443328 bytes (256.961MiB),
414976 bytes current after the call and7424 bytes after Python owner release.
Host RSS after release remains about20.49GiB. The high host/compiler residency
and the much smaller live device allocation are distinct observations; neither
is described as whole-device GPU preallocation. Process exit contains the
observed child residency while parent growth passes the declared256MiB gate.
This does not prove native in-process eviction or general memory bounds.
HLO inspection occurs between the after-call and final owner-release samples
and raises host RSS in these diagnostic children. Those distinct samples are
preserved; the final resident memory must not be attributed entirely to live
initializer tensors or treated as an uninstrumented production memory peak.
GPU rejection04627 passes in751.659 seconds. The unchanged rejected fixture
returns `dense_center_score_above_cap`, `passed=False`,252 exact evaluations,
one compiled trace and no host callbacks. Initializer time is662.555 seconds.
Thus the test passes by preserving the intended rejection, not by accepting
the candidate. Saved-evidence/policy readback04628 passes181 checks in22.264
seconds, including fresh admission, both complete lifetimes, child HLO and
artifact hashes, reaping, observed Python owner release and parent RSS. The
renewal unit closes after11 of14 workers, with no failed run in this unit.

Inspection of the preserved04617 comparison further identifies the4539 strict
CPU/GPU fitted-record differences:62 selection/stability leaves and1663,1150,
1664 leaves in public fits3,4,5. The selected precision, covariance, center and
audit result pass the existing comparison bounds. Fits3/4 reject on holdout;
fit5 passes its own fit gate but is unconverged and is not selected. Both
backends retain the same fit statuses. This localization does not waive any
complete-record criterion or establish optimizer convergence.

| Decision | Criterion status | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Close source-renewal execution unit | Seven target/oracle checks, both two-worker accepted lifetimes, intended GPU rejection and181 readback/policy checks pass | Frozen r2 only; CPU/GPU full records are not equivalent | Preserve verified archives and qualify the separate adapter import repair | Complete initializer equivalence or HMC admission |
| Keep main unmerged | Final precision repair is qualified separately | Strict full records and terminal master checks remain open | Continue first-objective localization and endpoint evidence review | Whole-program completion |

Post-run skeptical review: exact repeat records within each backend support
repeatability, not CPU/GPU equivalence. Process-exit containment is the observed
memory mechanism; Python-owner collection does not establish native eviction.
The weakest generalization is from two fixed-signature children to arbitrary
target/signature capacity. A growing parent trajectory or unreaped child would
overturn this scoped containment conclusion. No independent reviewer was used.
