# Completed-checkpoint development evaluation — execution note

Attempt 01 stopped before scoring after 5.978 CPU seconds. It reconstructed a
source architecture from the saved JSON snapshot, whose integer KDE dimension
keys became strings. The original source expects integer dimensions. This is a
localized harness serialization failure, not an algorithmic result or candidate
failure. The failed log/manifest are preserved.

The repair reconstructs architecture directly from the pinned author YAML and
requires exact equality after JSON serialization against the saved training
configuration. No author code, trained tensor, source setting, metric, data
design, promotion criterion or campaign budget changes. Attempt 02 was the final
initially allowed attempt; the later localized repair amendment is recorded below. Its first full
source-checkpoint conversion replay is the focused executable regression for
this call chain.

Attempt 02 consumed 5.424 CPU seconds. The exact source-checkpoint conversion
replay and metric control passed, but a generic guard tried to compare the
source's optional None resampling bandwidth with zero. Require primary fields
and check optional fields when non-None. This is a harness guard repair;
attempt 03 is authorized within the unchanged resource cap by the explicit
pre-run plan amendment. No numerical evidence from partial attempts is promoted.

## Terminal result

Attempt 03 completed with return code zero in 60.694
CPU seconds. All 55694 conversion, source identity, state immutability,
finite/probability/bandwidth and metric checks passed. The separate Python
standard-library reduction audit reproduces all eight saved NLL and RMSE
aggregates within float32 summation roundoff; maximum absolute discrepancy is
5.58e-07.

The TF-trained checkpoints have higher NLL and RMSE on both fresh development
filter seeds when evaluated through the same unchanged author code. Thus a
difference in the TF evaluation implementation cannot by itself explain the gap.
Both trained networks may still vary across independent training runs; this
experiment has one training pair and cannot identify the cause or establish a
population ranking.

| Variant | Filter seed | Author-trained NLL | TF-trained NLL | NLL increase (nat) | RMSE increase |
|---|---|---:|---:|---:|---:|
| mdpf | 202610082 | 6.425756 | 6.733006 | 0.307250 | 2.413% |
| mdpf | 202610083 | 6.404428 | 6.750554 | 0.346126 | 3.172% |
| am_dpf | 202610082 | 6.048497 | 6.340334 | 0.291837 | 2.254% |
| am_dpf | 202610083 | 6.005387 | 6.387122 | 0.381735 | 5.806% |

NLL is per joint state, using the author density floor. RMSE uses the author's
per-trajectory square root before averaging trajectories. Both metrics are
descriptively worse in all five inspected conditions: first 17 steps, later
steps, small/large observed bearing residual and the angular seam. The four
overall point differences exceed the accepted 0.01-nat and 1% margins, but no
trained-unit confidence limit or formal qualification verdict is available.
The two filter seeds are not independent trained models.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Localize the learned-checkpoint quality gap | Common-author-evaluator diagnostic completed; no qualification criterion tested | Conversion/source/finite checks pass; original float32 gradient screens remain open | One training pair; differing training randomness and numerical paths | Audit full training call chain and real-checkpoint updates; calibrate source training variability | Neither replication complete; no general ranking, TF evaluator parity or source defect |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new invalid model or evaluator failure; original numerical mismatches remain preserved |
| Statistically supported ranking | None: one independently trained pair |
| Descriptive-only differences | NLL +0.292 to +0.382 nat; RMSE +2.254% to +5.806% |
| Default-readiness | Not assessed; no promotion |
| Next evidence needed | Source/TF training-update localization, source-only trained-unit variability calibration, then untouched qualification |

The strongest alternative explanation is ordinary variation between one author
and one TF training run. The weakest evidence is the single trained pair, not
the evaluator conversion. Matched updates at actual trained checkpoints would
weaken a translation-error hypothesis; multiple independent author runs would
quantify the variability explanation. The 256 fresh T=150 trajectories are
development data and cannot later become untouched qualification data.

## Provenance and accounting

Plan: [development evaluation](controlled-smc-kdm-development-evaluation-20261008.md).
Full result and per-step arrays:
[attempt 03](artifacts/controlled-smc-kdm-replication-20260928/kdm-development-evaluation-20261008-03/diagnostic/result.json).
Independent aggregate audit, metric hashes and all attempt manifests:
[reviewed summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-development-evaluation-reviewed-20261008.json).
Source commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18; local commit
89065bc6354801cb368d5e163ba49fd9c4372d10; exact command/environment, checkpoint,
source and helper hashes are in each manifest. CPU-only author reference,
CUDA_VISIBLE_DEVICES=-1, one thread, no GPU.

Charge 192.096087646 CPU seconds across all three attempts plus the included
120-second routine implementation/audit allowance; zero GPU. This remains within
the 1800-second phase cap. All attempts are terminal and no evaluation is running.
