# Phase 8 C4B K=4 joint-mixture replication result

Date: 2026-08-31  
Status: `PASS_C4B_JOINT_REPLICATION_NO_PROMOTION`

Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-subplan-2026-08-31.md`

## Result

The fresh-root q=20 replication completed both predeclared architecture rows
in `876.4273084410233` seconds on one visible RTX 4080 SUPER GPU.  The run used
TensorFlow/TFP float64, the strict `tensorflow_eigh_strict` backend, XLA and
TF32, and memory growth configured before logical-device creation.  The
allocator peaks were `3,401,816,064` and `3,402,234,624` bytes, both below the
4-GiB cap.  The manifest reports no failed rows and a complete hard screen.

Manifest:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4b-joint-replication/attempt-01/run_manifest.json`

The two rows were fresh and disjoint from C4A: `compact-high` at learning rate
`1e-3` with root `(20260831,62001)`, and `compact-low` at `5e-4` with root
`(20260831,62002)`.  Each row restored independent and joint copies from the
same four beta-0.5 checkpoints.  The joint arm made one target call over 128
flattened rows and exactly 512 cross-density work units per update; each
independent chart made one target call and zero training cross-density units.

## Evidence

| Row | Arm | Held-out objective on that arm's fresh bank | Joint forecast (16 updates, s) | Peak allocator (GiB) | Alpha entropy | Duplicate states |
|---|---|---:|---:|---:|---:|---:|
| compact-high | independent | 169.4369 | n/a | 3.17 | 1.386294 | 0 |
| compact-high | joint | 190.7628 | 118.2015 | 3.17 | 1.386293 | 0 |
| compact-low | independent | 169.7835 | n/a | 3.17 | 1.386294 | 0 |
| compact-low | joint | 163.7449 | 115.8080 | 3.17 | 1.386292 | 0 |

The nominal joint-minus-independent differences are `+21.3260` for
compact-high and `-6.0386` for compact-low.  These are not paired estimates:
the independent and joint endpoint banks use distinct stateless validation
draws.  They therefore cannot identify an optimization or approximation
difference, even descriptively beyond the recorded finite-bank observations.
The alpha vectors stayed close to uniform in both joint rows; this is a
variational-density diagnostic, not an estimate of posterior regional mass.

All four maps in every arm passed self, cross-component, reference, declared,
physical-score, inverse, log-determinant, and conditioning checks.  Maximum
self/cross round-trip errors were at the `1e-15` level and all reliability
receipts were valid.  Pullback score residuals remained large (coordinate RMS
values ranged roughly from `10` to `1,451`), so the run provides no whitening
evidence.  A bounded TensorFlow retracing warning appeared while constructing
the second row's finite set of trainer objects; the fixed input signature was
retained, the run completed, and this remains a performance/debt diagnostic,
not a reason to reinterpret the numerical receipts.

## Decision

| Decision | Primary criterion | Hard-veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| C4B implementation/resource replication | Exact work, finite updates, replay, reliability, alpha, duplicate-state, GPU/XLA, allocator, and forecast checks for both rows | Pass | Short eight-update pilot and finite bank sizes | Close C4B and perform the audited C5 freeze bookkeeping | No production joint arm |
| K=4 joint feasibility | 16-update forecast <= 3,600 s and allocator peak < 4 GiB | Pass (`118.20 s` max; `3.17 GiB` max) | Driver reservation is larger than allocator live/peak telemetry; larger K/B remain untested | Retain K=4 only as a bounded calibration candidate | No high-dimensional scaling claim |
| Joint versus independent objective | Common target and exact density accounting | Descriptive only; banks are unpaired across arms | Different endpoint draws, short optimization, and near-uniform alpha | If scientifically needed, design a future common-bank paired diagnostic; do not infer a ranking here | No superiority, ranking, or architecture choice |
| Learned-map reliability | Four-map q=20 reliability screen | Pass | The screen probes finite declared banks, not global geometry | Keep reliability as a prerequisite for any later tuning | No global invertibility/whitening guarantee |
| Whitening and HMC readiness | Held-out pullback density/score residuals | Gate remains closed; score residuals are large | Capacity, objective, and limited updates may all contribute | Preserve the whitening veto and do not open Phase 9/HMC | No IID-Gaussian pullback or posterior claim |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for both bounded replication rows. No target/status, work-accounting, checkpoint, reliability, alpha, duplicate-state, memory-growth, XLA, allocator, or forecast veto fired. |
| Statistically supported ranking | None. Two rows, eight updates, and unpaired held-out banks cannot support a ranking. |
| Descriptive-only differences | Per-row endpoint objectives, update times, alpha movement, diversity summaries, and pullback residuals. The signs of the two objective contrasts disagree. |
| Default readiness | Not established. There is no target-specific long training, uncertainty analysis, retained-chain evidence, ESS/R-hat, downstream agreement, or posterior validation. |
| Next evidence needed | Complete the C5 freeze decision using only the calibration receipts; any later Phase 9 work requires a separate reviewed confirmation subplan and fresh streams. |

## Post-run red-team

The strongest alternative explanation for the differing objective contrasts is
validation-bank randomness combined with different short optimization paths,
not a reproducible benefit or harm from the joint objective. A second concern
is TensorFlow graph retracing across row-local trainer objects; although it was
bounded and did not alter the fixed-signature numerical path, a production
route should construct reusable compiled functions or explicitly budget the
compile overhead. The driver-level reservation near 5.9 GiB is also not the
same quantity as the recorded TensorFlow allocator peak; the result does not
claim that another process can safely share arbitrary larger runs.

The implementation conclusion would be overturned by a recomputed manifest
hash mismatch, a wrong `[K,K,B]` work tensor, a nonfinite status or score, a
checkpoint replay mismatch, exact map collapse, a failed reliability receipt,
or an allocator/memory-growth violation. None occurred. The weakest evidence
is the endpoint objective contrast and the short alpha trajectory.

## Provenance and boundary

The manifest records the Git commit and dirty state, exact command, target and
bridge signatures, properness receipt, roots, memory policy, GPU snapshots,
route scan, C3B/C4A prerequisites, source hashes, and all row telemetry. C4A
was used only for status and provenance; all C4B maps and banks were created
fresh. No posterior draws, target-derived replay particles, state-dependent
chart selection, or Phase 9 streams were consumed.

This result establishes a bounded implementation/resource replication only. It
does not establish whitening, IID Gaussianity, exhaustive mode discovery,
posterior regional masses, convergence, HMC readiness, statistical
superiority, architecture superiority, production readiness, or
high-dimensional scaling.
