# SSL-LSTM q=20 NeuTra target-validity recovery result (2026-08-06)

Status: `ERROR_HANDLING_REPAIRED_EXACT_PLACEMENT_INVALIDITY_LOCALIZED`

## Outcome

Seed A did not contain a NeuTra-parameter NaN or Adam-state NaN. The historical
worker exception was caused by a valid target-status veto being converted to
NaN by the compatibility value/score API and then raised by a worker finite
assertion.

The repaired replay reproduced the event at continuation update 930, row 16.
The raw target result was finite: value `-1e100`, finite four-coordinate score,
finite input, zero placement/innovation floor counts, and positive innovation
minimum eigenvalue `0.47259927921443085`. Its row class was
`2=classified_invalid`, so the legacy wrapper replaced it with NaN.

The exact recorded four-row CPU/XLA worker shard localized the classification
to the placement covariance:

- `placement_classified_invalid_count = 1`;
- `innovation_classified_invalid_count = 0`;
- `placement_derivative_rhs_nonfinite_count = 0`;
- `innovation_derivative_rhs_nonfinite_count = 0`;
- minimum placement eigenvalue `-2.927399523211154e-12`;
- reviewed roundoff-repair lower bound `-1e-14`.

The eigenvalue was about 293 times more negative than the repair tolerance, so
the four-row route correctly classified it invalid under current policy. A
one-row XLA replay gave a slightly positive minimum placement eigenvalue
`2.746569198222148e-13`; this establishes batch-shape numerical sensitivity near
the boundary. It does not justify relaxing the covariance policy.

## Error-handling repair

The CPU/XLA pool now has an explicit status-preserving route. It returns raw
values, scores, per-row target status, shard bounds, worker index/PID, assigned
CPU, and XLA provenance without treating a completed invalid target evaluation
as a process failure. The finite-only compatibility route remains fail-closed.

The continuation runner now:

1. rejects the entire invalid batch before the optimizer call;
2. archives every `z`, transported `theta`, raw value/score, row status, and
   worker provenance;
3. verifies trainer and Adam `step`/`state_hash` remain unchanged;
4. permits at most three deterministic fresh-batch retries;
5. writes `TARGET_VALIDITY_RECOVERY_EXHAUSTED` plus result/progress/manifest
   artifacts if retries exhaust; and
6. permanently records `target_validity_failure_observed` as a promotion veto
   even when a replacement batch is admitted.

The update-930 attempt-0 batch was rejected at optimizer step 2429. Attempt 1
was admitted, and the replay completed all intended updates 501--999 with
terminal optimizer step 2499. The 500-row terminal monitor was finite with mean
loss `41.09300903140879`; support and round-trip checks passed, with maximum
round-trip residual `4.440892098500626e-15`. These are engineering checks only.
Because replacement-batch admission conditions on target validity, the
recovered candidate remains ineligible for promotion or HMC nomination.

## Evidence artifacts

- Plan: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-target-validity-recovery-plan-2026-08-06.md`
- Two-update GPU-1 canary: `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/checkpoint-resume-canary-r3/result.json`
- Original-window replay: `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/original-window-replay-r4/result.json`
- Exact failure: `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/original-window-replay-r4/target-validity-failure-0930-attempt-0.json`
- Exact-shard localization: `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/exact-shard-localization-r1.json`

The original historical `r1` campaign artifacts were not modified.

## Run manifest summary

| Field | Original-window replay |
|---|---|
| Git commit | Recorded in `run-manifest.json`; worktree dirty because this repair and concurrent lanes were active |
| Command | GPU-1 checkpoint-500 replay, updates 501--999, 25 workers x 4 rows, 7,200-second cap |
| Environment | `tfgpu`, TensorFlow 2.20.0, FP64 |
| CPU/GPU status | physical GPU 1 exposed as logical GPU 0; physical GPU 0 hidden; memory growth verified; CPUs 0--24 pinned |
| Data version | historical seed-A checkpoint-500 SHA-256 `8db1883f0a7c29fc93e5d26169458974457aadf379b11d226523603bfa1b0232` |
| Random seeds | original attempt-0 folds; update 930 fold 930; deterministic retry fold 4930 |
| Wall time | `3610.910112417012` seconds through result creation; `3626.1475305510103` seconds including pool shutdown and final manifest |
| Output | `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/original-window-replay-r4/` |
| Plan/result | recovery plan above; `result.json` in output root |
| Device policy | TensorFlow GPU memory growth true; worker target status and trainer XLA true |

The current trainer schema gained optional empty field `fixed_output_scale=[]`
after the historical checkpoint. Replay restoration verified the original state
hash, allowed exactly this absent empty field, recomputed an in-memory state
hash, recorded the migration, and did not alter the numerical transform or
historical file. All other config differences remain rejected.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Error-handling repair accepted | Passed: invalid target response preserved, pool survived, no update occurred, controlled replay completed | Target-validity promotion veto fired | Generic infrastructure exceptions outside target validity still use the existing campaign failure path | Keep status-preserving pool/runner tests as regression guards | Training objective validity after retry |
| Seed-A recovered checkpoint | Not eligible for promotion | `target_validity_failure_observed` | Conditioning on an admitted replacement batch changes the stochastic training procedure | Do not use this recovered candidate for HMC; repair/retrain a fresh candidate if desired | NeuTra convergence or HMC readiness |
| Numerical localization | Passed for update-930 row-16 exact batch-4 shard | Placement covariance classified invalid | Earliest UKF time index and raw covariance matrix were not archived | Add first-invalid-time/raw-covariance tracing only before a reviewed numerical-policy study | Threshold should be relaxed |

## Inference-status table

| Evidence class | Result |
|---|---|
| Hard veto screen | Seed A candidate has a confirmed target-validity promotion veto; no parameter/Adam NaN was observed |
| Statistically supported ranking | None |
| Descriptive-only differences | Monitor loss, runtime, one-row versus four-row eigenvalues, and post-retry training trace |
| Default-readiness | Not evaluated; recovered seed A is not default-ready |
| Next evidence needed | A fresh candidate trained under a reviewed strategy that avoids or explicitly models invalid target support, followed by downstream sequential fixed-HMC validation |

## Negative-result classification

- Implementation failure: confirmed in old error handling. It conflated
  classified target invalidity with NaN/infrastructure failure and lost the row.
- Tuning/candidate failure: confirmed for seed A at one proposal. The transport
  reached a parameter region whose UKF placement covariance was invalid under
  the current strict policy.
- Diagnostic failure: repaired. Exact row, shard, status, and proposals are now
  preserved.
- Evidence against NeuTra generally: unsupported. This is one seed/candidate
  and one near-boundary target event.
- Evidence against the target math: unsupported. The strict classifier behaved
  according to its checked threshold in the exact batch-4 route.

## Post-run red-team

Strongest alternative explanation: the additional status outputs changed the
compiled worker executable enough to move a near-boundary eigenvalue. The exact
event nevertheless matches the historical deterministic update window and old
failure class, while the batch-4 replay independently reproduces the same row
and finite sentinel. Bitwise executable identity is not claimed.

What would overturn the localization: an exact batch-4 replay using the same
source and backend that yields zero placement invalid count or identifies an
innovation/derivative failure instead. The committed diagnostic currently does
the opposite.

Weakest evidence: the earliest UKF time index and raw placement covariance were
not archived, so the dynamical origin of the indefinite covariance is not yet
localized. The current result answers why NaN appeared and why the run crashed;
it does not yet prescribe a numerical-policy or transport-support repair.
