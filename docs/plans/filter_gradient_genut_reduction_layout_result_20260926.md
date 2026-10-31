# GenUT reduction performance investigation

The simple last-axis layout candidate is rejected: it passes complete
same-mode numerical comparisons but increases CPU N=10,000,d=18 warm time in
all twelve pairs. A second, uninstalled highest-precision XLA dot candidate
passes numerical comparisons and substantially reduces that CPU cost, but
slows the smaller GPU fixture and lacks TensorFlow gradient registration.
No runtime implementation was changed by this diagnostic phase.

| Trial | Run | Fixture | Current / candidate median warm ms | Disposition |
| --- | --- | --- | ---: | --- |
| Last particle axis | 04161 CPU | N=1,000,d=3 | 1.477 / 1.378 | Small descriptive difference |
| Last particle axis | 04161 CPU | N=10,000,d=18 | 86.615 / 113.290 | Reject; every paired ratio >1.15 |
| Highest-precision XLA dot | 04162 CPU | N=1,000,d=3 | 1.831 / 1.961 | Small descriptive cost increase |
| Highest-precision XLA dot | 04162 CPU | N=10,000,d=18 | 80.740 / 34.814 | Viable performance candidate |
| Highest-precision XLA dot | 04163 GPU | N=1,000,d=3 | 1.377 / 1.780 | About 29% higher median |
| Highest-precision XLA dot | 04163 GPU | N=10,000,d=18 | 1.690 / 1.714 | Similar descriptive timing |

Both trials preserve existing controls, update recurrences, caps, dtype and
global TF32 setting. The dot trial uses TensorFlow `XlaDotV2` with both operands
set to `PrecisionConfig.HIGHEST`; it leaves the known symmetric-projection
fusion site as an explicit reduction. CPU/GPU compile, complete same-mode
outputs at atol=rtol=2e-5, exact replay and one-trace checks pass. All non-report
fields pass the independent saved FP64 reference. The historical cross-mode
cap report is still explicitly open, with no dropped field or changed bound.

The raw dot operation has no registered TensorFlow derivative on this installed
version. The diagnostic captures the exact `LookupError`; it does not insert
`stop_gradient`, substitute a different score or claim analytical LEDH score
support. Adoption requires a TensorFlow pullback with independent matrix and
full-program derivative checks, graph/debug compatibility, retained ownership,
fresh-process memory measurements and GPU cost qualification. The CPU timing
gain alone cannot justify replacing the default GPU kernel or a shape-dependent
dispatch chosen without evidence.

| Decision | Primary criterion | Veto | Main uncertainty / next action | Not concluded |
| --- | --- | --- | --- | --- |
| Reject transpose-layout candidate | Numerics pass, CPU target cost rises | Performance veto | Preserve source/output/HLO | No runtime repair |
| Keep highest-dot as uninstalled candidate | CPU d=18 timing and CPU/GPU numerical checks pass | Missing derivative; small GPU regression | Bounded pullback/compatibility and shape-cost study before adoption | General speed superiority or default readiness |
| Retain current runtime | Existing numerical/consumer gates hold | Report semantics remain unresolved | Continue master public-consumer repairs | Master completion/main merge |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical trial failure; adoption blocked by derivative/cost gaps |
| Statistically supported ranking | None; paired calls are one-process timing diagnostics |
| Descriptive differences | Table above, within each independently compiled trial |
| Default-readiness | No trial promoted |
| Next evidence | Native pullback, graph compatibility, fresh-process memory and GPU cost |

Preceding source-refresh checks 04141--04152 pass GenUT FP64/FP32, reset and
Austria callback consumers, native costs and policy on CPU/GPU. Runs
04153--04160 pass 23 LEDH safety/stage/consumer checks per backend, with no
canonical admission. These are registered consumer checks, not closure of
full-public filtering or the F01--F20 inventory.

The first GPU refresh launch twice failed during sandboxed nested NVIDIA
discovery before a numerical worker was created. Direct trusted discovery
succeeded; the trusted runner passed. Run 04147 duplicates the passing
04146 graph cost and is preserved/charged, not presented as an independent
requested replication. Shared `.git` write failures were sandbox permissions,
not evidence of a filesystem or hardware failure.

Evidence review corrected a narrow archive pattern: the first 04140 archive
omitted capacity JSON/NPZ files. The immutable r2 archive and verification
receipt include all raw arrays, per-run metadata, exact analyzers and Git
source snapshots; r1 remains preserved. Intermediate uncommitted harness bytes
are not reconstructed. Raw manifests retain their exact source hashes.

Skeptical review: timed arrays are already materialized on the host, and both
owners share one process; these observations explain relative execution cost
but cannot measure isolated memory. Hardware activity can vary between the
CPU and GPU trials. Retaining the original sequence and all outputs prevents
a faster changed algorithm from being counted as a repair. The dot trial's
missing derivative is a concrete adoption gap, not grounds to change the
canonical LEDH scope or score policy.
