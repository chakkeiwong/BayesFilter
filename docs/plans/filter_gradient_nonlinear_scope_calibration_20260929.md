# Fresh nonlinear T2 reset-balance qualification scope

The unchanged seed9292027 fixture with four Sinkhorn/four balance steps failed
the original nonlinear LEDH reset TV gate at T2. Preserve that fixture and its
refusal-only result. Do not tune on it, reinterpret it as healthy, or relax the
1e-4 TV gate. A healthy T2 analytical-score comparison requires separately
calibrated controls on fresh arrays; execution-policy repairs alone cannot
make an under-balanced finite program admissible.

This is test-scope numerical calibration for execution qualification, not
claim-bearing LEDH tuning, a production default or a canonical admission
artifact. Keep d=o=1,N8,T2,float64, K=N, c=.17,b=.09 and the declared positive
dual-cap/Contract E controls. Only offline reset Sinkhorn/balance counts may
change for the new scope. The failed fixture is excluded from nomination,
validation, timing and acceptance. Original algorithm, RNG authority, all
derivatives and every validity/accuracy threshold remain unchanged.

Before running, serialize independent calibration, validation and untouched
check partitions from test-only standard-library seeds9293027,9294027,9295027,
including complete theta, observations, initial/process/reset arrays. Use
three cases in each calibration/validation partition and two untouched checks.
No array or partition may be changed after seeing results. Declare theta/data
generation ranges before constructing these arrays. No historical LEDH source
fixture or result is a baseline. This plan is not launch-ready until that
exact fixture file and scope are written.

For each of LEDH, SGQF level2 and mixture within_fraction .6, evaluate the
predeclared count pairs (8,8),(16,16),(32,32),(64,64) in ascending order on its
own calibration partition. Use the existing compiled directional authority
with the first basis direction and its actual program-valid status. Nominate
the first pair passing every calibration input with positive covariance,
finite records and the unchanged reset gate. Evaluate that frozen nomination
on the independent validation partition once. Validation may veto it; a failed
validation cannot select a later pair. Record every attempted calibration pair
and all diagnostics; no warm-time ranking. Freeze the validated pair separately
per provider before either untouched check. If no pair passes calibration or
the nomination fails validation, stop with numerical-control insufficiency and
diagnose it; do not expand counts, change targets, retune on validation or relax
gates automatically.

On both untouched cases compare the original directional calls and complete
enclosing owner for all six directions, full nested diagnostics and status;
require1e-9 parity and two-step five-point error<=2e-6. Use exact frozen replay
where applicable (none for these three providers), changed dynamic operands,
one trace and enclosing HLO. Verify actual nonlinear endpoint wiring and its
real reference-grid checks. Repeat on trusted non-display GPU with growth
configured before initialization. A failed untouched check is preserved and
blocks healthy-scope qualification; it cannot be folded back into tuning.

Initial allocation, activated after the refusal unit closes: at most16 workers,
2400 CPU and2400 GPU process-seconds from the same remaining global caps.
Use one numerical worker at a time,300-second initial timeouts, the stable
campaign runner and unique artifacts. Split configuration candidates across
workers when native compiler residency warrants it; no bulk owner compilation
or cache-eviction assertion. Costs remain a later bounded cohort after these
numerical checks. No HMC/training, model/backend/RNG change, package/system/cache
mutation, subagent, live MacroFinance edit or main merge.

Skeptical review: raising iteration counts changes the configured finite scalar,
so parity must use the same newly frozen counts on both before/after arms. It
cannot repair or erase the failed original scope. Per-provider, disjoint-data
selection prevents a setting from one covariance route becoming an unreviewed
default for another. This is a validity nomination, not a performance ranking
or a scientific statement about nonlinear filtering accuracy. Exact generation
ranges, fixture arrays and attempt accounting must be finalized before launch;
self-review passes for bounded preparation only.

Pre-generation range specification: independently for each provider and
partition, use the partition seed plus provider offsets0/100/200 for
LEDH/SGQF/mixture. Each case starts from theta
[.57,-.77,-.53,.83,.18,-.23] plus six independent Uniform[-.03,.03] perturbations;
observations are two independent Normal(0,.35) values. Initial, process and
reset arrays use independent standard normals in that recorded draw order;
auxiliary resampling uniforms are generated afterward but unused by these
providers. Round stored floating draws to12 decimal digits. Changed theta is
the stored theta plus [.01,.015,-.015,-.01,.01,-.01]; changed observations add
[[.02],[-.02]]. The physical data generator is test fixture construction only,
not a claimed simulated posterior/likelihood sample. Keep c,b,reference-grid
controls and all non-balancing numerical controls fixed as specified above.
The predeclared ranges and count ladder do not use the preserved failed seed.

Frozen partition files were generated after this range specification and before
any numerical calibration/validation/untouched execution:

- `tests/fixtures/filter_repair_nonlinear_calibration_20260929.json`: SHA256 7041fd4e538759d56a12e68119226c0bc398a838ea23bae6351043f4281c912e.
- `tests/fixtures/filter_repair_nonlinear_validation_20260929.json`: SHA256 2787dcb42da68109f3c89d94130c923f348f918caa63ce2100f8e119115f276c.
- `tests/fixtures/filter_repair_nonlinear_untouched_20260929.json`: SHA256 2a64417d8811501e4a152a589b64b1ceeff63c27369c6f56ffbc4645efe4e048.

The untouched partition must not be evaluated until the provider-specific
nomination passes independent validation and is frozen in its result.

Implementation review chooses one provider/count configuration per worker so
an unsuccessful ladder cannot accumulate several compiler owners in one process.
Refresh the allocation to24 workers, keeping2400 CPU/2400 GPU process-seconds:
up to12 calibration workers, six untouched CPU/GPU workers, one readback and
five localized retry slots. A later count can launch only when the immediately
preceding count failed calibration and never reached validation. Validation
failure stops that provider. Endpoint test helpers now accept explicit fixture
arguments; calibration supplies only the new partition arrays. No production
runtime file changes during this unit. Nomination records are explicitly
test-only and cannot satisfy a canonical tuning/admission route.

September29 continuation review identified a coverage omission: the healthy
untouched checks instantiated only the diagnostics-enabled owner. The ordinary
endpoint has a different output signature and trace configuration, so refusal
tests alone cannot qualify its healthy path. Add one ordinary-endpoint worker
per provider/device on the same two untouched arrays and already frozen counts,
using identical parity, derivative, changed-input, rejection and HLO gates.
This extends execution coverage; it does not select controls or reuse untouched
data for tuning. Keep the completed diagnostic runs. The full selected cohort
now needs three calibration workers, six diagnostic and six ordinary endpoint
workers plus one readback (16 total), inside the existing24-worker/2400 CPU/
2400 GPU-second allocation. Any failure remains preserved without retuning.
Self-review passes with this explicit ordinary/diagnostic coverage repair.
