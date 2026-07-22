# PP-UKF Tuning Failure Repair Note

Date: `2026-07-20`

## Repair

The public tuner previously wrote the bootstrap exception type and message to
`hmc_kernel_tuning_progress.json` but omitted them from the terminal
`hmc_kernel_tuning_result.json` and public artifact. This made the PP-UKF
`bootstrap_screen_error` classification non-actionable.

The repair adds capped, public-safe `failure_diagnostics` with the failure
stage, exception type, and exception message. It is explicitly diagnostic
provenance and exposes no HMC mechanics, posterior values, or private state.

Focused verification after this repair: `64 passed, 2 warnings` across the
public API and bootstrap tuner suites; compile and diff checks passed.

The retained-bootstrap regression verifies that a returned hard-veto result
is checked before handoff-kernel construction, that Phase 7 is not entered,
and that the bootstrap round remains represented by its stage hash and
public-safe failure diagnostics.

## Root Cause Found

The first repaired diagnostic (`repair-04`) reached the real target call. The
PP-UKF HMC bootstrap supplied a rank-1 initial state through
`BatchNativeBoundAdapter`, which forwarded it directly to the PP-UKF adapter.
That target is explicitly rank-2-only (`[batch, 6]`), so the run failed during
TensorFlow Probability bootstrap with:

`ValueError: predator-prey target requires theta shape [batch, 6]`

This was an adapter-shape integration defect, not evidence against the
predator-prey posterior, frozen transport, or GPU/XLA setup.

The shared end-to-end adapter now lifts rank-1 HMC calls to a batch of one
through the repository-issued batch-native callable and squeezes the result
back to the scalar HMC contract. Batch-native training still requires batches
of at least two. Verification: `40 passed, 2 warnings` across the end-to-end
contract and PP-UKF target suites.

## Diagnostic Rerun Contract

The diagnostic uses the completed PP-UKF final frozen transport from:

`docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json`

It does not use the LGSSM replay artifact, does not retrain the transport, and
does not launch sequential HMC sampling. The scientific target, acceptance
policy, fixed-identity mass policy, GPU/XLA settings, and transport hash remain
unchanged.

## Diagnostic Results

| Attempt | Result | Evidence | Interpretation |
| --- | --- | --- | --- |
| `repair-03` | `TUNING_FAILED` | `.../pp-ukf-tuning-diagnostic-repair-03/PP-UKF/tuning/hmc_kernel_tuning_result.json` | Original handoff-order bug reported only `bootstrap hard veto cannot provide active handoff kernel`; no target-call diagnosis. |
| `repair-04` | `TUNING_FAILED` | `.../pp-ukf-tuning-diagnostic-repair-04/PP-UKF/tuning/hmc_kernel_tuning_result.json` | Actual rank-1 versus rank-2 PP-UKF target-shape defect identified; no HMC sampling. |
| `repair-05` | No terminal result | `.../pp-ukf-tuning-diagnostic-repair-05/PP-UKF/tuning/hmc_kernel_tuning_progress.json` | After the adapter repair, bootstrap completed with finite XLA execution and acceptance-repair diagnostics; the managed execution boundary terminated the process during fixed-mass screening before a terminal artifact. This is not scientific pass/fail evidence. |

The completed bootstrap portion of `repair-05` is diagnostic evidence that the
rank mismatch was repaired: it records `preflight_passed=true`, six bootstrap
rounds, finite XLA metadata, and no hard-veto categories. It does not establish
kernel admission, posterior convergence, truth-tail recovery, or PP-UKF
scientific validity because the tuner never wrote a terminal result.

## Current Decision

The PP-UKF target-shape blocker is repaired and regression-tested. The cell
remains non-admitted for claim-bearing HMC because `repair-05` was terminated
before terminal tuning verification and no sequential sampling was run. Do
not rerun PP-UKF training. Any future continuation should use a fresh output
root and the same frozen transport, and should be treated as a new authorized
campaign continuation rather than upgrading `repair-05`.

## Tuning-Only Guard And Follow-Up

The frozen-validation runner now accepts `--tuning-only`. This mode rejects
admitted-kernel replay input, writes `sampling_launched: false`, records
`tuning_only: true` in the result and manifest, and returns immediately after
the public tuner. Focused verification: `31 passed` end-to-end contract tests
and `64 passed` tuner/bootstrap tests.

The fresh `repair-06` run used that guard, the unchanged frozen transport, and
no retraining. Bootstrap and windowed-mass stages completed, and the run
advanced through fixed-mass candidates without the prior rank-shape error or
hard veto. The managed execution boundary again terminated the child during
fixed-mass repair screening before a terminal `result.json` or manifest was
written. Its `run_state.json` is explicitly marked `interrupted` with
`terminal_result_written: false` and `sampling_launched: false`.

`repair-06` is therefore incomplete infrastructure evidence, not a tuning pass
or candidate rejection. No sequential HMC sampling launched. Do not start
another retry in this campaign without moving the command to a host-owned
process boundary such as host `tmux`, SSH, or systemd.
