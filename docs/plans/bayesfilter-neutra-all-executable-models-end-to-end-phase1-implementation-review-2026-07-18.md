# Phase 1 Implementation Review and Close Record

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Date: 2026-07-18

Status: `PASS_TO_GPU_PREFLIGHT`

## Scope

Reviewed the executable registry, reusable end-to-end composition, single
campaign CLI, plan, and focused contract tests before any serious GPU launch.

## Findings and repairs

1. The LGSSM geometry factory initially pointed at a campaign contract rather
   than the actual mass artifact. It now binds the mass file and expected
   SHA-256.
2. A new runner cannot construct the private typed campaign identity directly.
   The composition now uses the public inspected batch-native target binding,
   direct trainer, native tuner, and sequential controller.
3. General rank-2 adapter calls can differ from the reviewed NeuTra batch
   callable. `BatchNativeBoundAdapter` now delegates held-out evaluation,
   tuning, and HMC to the exact inspected `neutra_batch_log_prob_and_grad_status`
   method used by training.
4. Screening initially reused training seeds and did not guarantee paired
   recipe batches. Screening now uses one common disjoint stateless held-out
   tensor per cell and records batch means/MCSE.
5. Warm-up convergence was initially checked in source coordinates while
   retained diagnostics were intended for physical coordinates. The sequential
   model transform now emits physical coordinates for both gates.
6. Historical nonlinear hashes were typed campaign identities, not direct SSM
   mathematical signatures. The registry now binds current
   `stable_ssm_target_signature` values; historical hashes remain provenance.
7. The truth-tail diagnostic now refuses to evaluate when sampler health or
   convergence fails.

## Local audit

- `python -m py_compile` passes for all new Python files.
- CPU-hidden registry CLI reports exactly five executable and seven blocked
  cells.
- All five current adapter factories match their declared direct signatures,
  bind the reviewed batch-native method, return finite `[4]` values and
  `[4, parameter_dim]` scores, and expose the declared truth transform.
- AST audit finds no NumPy import, benchmark-script import, Python sampler,
  HMC kernel construction, R-hat/ESS implementation, or manual kernel selector
  in the new runner.
- `git diff --check` passes for all scoped new files.
- Claude health probe passed (`CLAUDE_PROBE_OK`). Two bounded one-path plan
  reviews returned no substantive output; reviewer unavailability is recorded
  and did not override the local skeptical audit.

## Evidence contract after review

The runner delegates training to `train_plain_dense_iaf`, tuning to
`tune_fixed_transport_hmc_kernel`, sequential warm-up/retained sampling to
`run_sequential_neutra_hmc`, and convergence to
`rank_normalized_hmc_diagnostics`. It configures target acceptance `0.70`,
band `[0.65,0.75]`, identity mass in `z`, modern rank/folded R-hat, ESS,
finite/status/energy vetoes, separate warm-up archives, and the one-seed
truth-tail policy. Fixed-grid tuner fields are empty and asserted empty.

## Handoff to Phase 2

Entry conditions: this close record exists; scoped compile, registry, AST, and
diff checks pass; no unresolved implementation finding remains.

Required next action: trusted GPU device probe, memory-growth verification, and
one tiny fresh preflight composition smoke. The smoke is engineering evidence
only and must not be promoted to scientific or sampler claims.

Stop conditions: GPU unavailable after trusted probe, memory-growth contract
failure, target/binding signature mismatch, nonfinite tiny training/transport,
fixed-grid assertion failure, or output collision.

Forbidden actions: no serious all-cell launch before Phase 2 passes; no use of
historical result JSON as fresh evidence; no launch of blocked inventory cells.
