# Phase 13 Result: XLA/JIT Repair Gate

Date: 2026-07-07

## Scope

This result closes the Phase 13 XLA/JIT repair gate for the LGSSM-first
BayesFilter NeuTra/HMC program.  The phase investigated whether the inherited
Phase 9 TensorFlow XLA blocker could be repaired for the admitted LGSSM NeuTra
value/gradient diagnostic without changing the target, training policy, frozen
payload, CPU sample-generation boundary, or HMC boundary.

Phase 13 did not run NeuTra training, HMC sampling/tuning, external sample
generation, DSGE/c603, route ranking, default-policy changes, or
scientific/product/readiness claims.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `BLOCK_PHASE13_XLA_JIT_REPAIR_GATE` |
| Primary criterion status | Blocked: the original fixed tensor-list-size complaint was repaired by adding `maximum_iterations=n_timesteps` to the QR while-loop likelihood kernels, but trusted GPU/XLA execution then exposed `Support for TensorList crossing the XLA/TF boundary is not implemented`. |
| Veto diagnostic status | XLA/JIT gate remains blocked. No training, HMC sampling/tuning, or external sample generation ran; `jit_compile=false` remains required for NeuTra training and fixed-transport mechanics. |
| Main uncertainty | The remaining blocker appears to be a TensorList crossing-boundary limitation in the XLA-compiled value/gradient diagnostic. A TensorList-free or fully inlined value/gradient route is needed before any XLA-readiness claim. |
| Next justified action | Phase 14 should investigate the TensorList crossing-boundary blocker under a dedicated bounded XLA repair subplan. |
| What is not concluded | HMC convergence, posterior correctness, sampler quality, transport quality, route superiority, production readiness, default-policy change, or broad XLA readiness. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the inherited TensorFlow fixed tensor-list-size XLA/JIT blocker be repaired for the admitted LGSSM NeuTra route without changing the target or overclaiming readiness? |
| Baseline/comparator | Phase 9 XLA blocker artifact, Phase 10 successful non-XLA GPU training, Phase 11 non-XLA frozen-payload mechanics, and Phase 12 CPU sample-generation boundary. |
| Primary criterion | Either a bounded trusted GPU/XLA diagnostic compiles and records finite value/gradient checks for the same target boundary, or the blocker is preserved with exact error/provenance and no readiness claim. |
| Veto diagnostics | Target/signature change, hidden training, hidden HMC sampling/tuning, hidden sample generation, CPU-only result treated as GPU/XLA evidence, nonfinite diagnostics, unbounded runtime, or unsupported readiness/product/scientific claim. |
| Explanatory diagnostics | TensorFlow/XLA versions, GPU manifest, compile error class, finite value/gradient checks, target/adapter signatures, and runtime. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport quality, route superiority, production readiness, default-policy change, or broad XLA readiness beyond the exact bounded gate. |
| Artifact | Phase 13 diagnostic JSON, result/blocker, helper/tests, and Phase 14 subplan. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `e09046088be79f4100a77583063889a37be1de04` before Phase 13 edits. |
| Python | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`, Python `3.11.14`. |
| Conda environment | `tf-gpu` at `/home/chakwong/anaconda3/envs/tf-gpu`. |
| Trusted GPU probe | `nvidia-smi`: NVIDIA GeForce RTX 4080 SUPER visible, driver `591.86`, CUDA `13.1`. |
| Phase 13 command | `TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_xla_repair_tf` |
| Execution target | Trusted GPU/XLA diagnostic on `/GPU:0`. |
| Route | `lgssm_while_loop_qr`. |
| Target signature | `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb`. |
| Seed | `20260707`. |
| Batch size | `16`. |
| JIT/XLA | `jit_compile=true` diagnostic only. |
| NeuTra training | Not run. |
| HMC | Not run; no sampling and no tuning. |
| External sample generation | Not run. |
| Final trusted command status | Process exited `134` after TensorFlow wrote the blocker JSON and then aborted in allocator cleanup. |
| Wall time | Approximately `31` seconds from TensorFlow log timestamps for the final trusted attempt; exact shell timer was not captured because the process aborted. |
| Diagnostic artifact path | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-diagnostic-2026-07-07.json` |
| Plan file | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-subplan-2026-07-07.md` |
| Result file | `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-result-2026-07-07.md` |

## Repair Attempts

1. Baseline trusted GPU/XLA diagnostic using the existing QR while-loop value
   route reproduced the original Phase 9 blocker:

   ```text
   XLA compilation requires a fixed tensor list size. Set the max number of elements.
   ```

2. The QR while-loop kernels were patched with
   `maximum_iterations=n_timesteps`:

   - `bayesfilter/linear/kalman_qr_tf.py::tf_qr_sqrt_kalman_log_likelihood_while_loop`
   - `bayesfilter/linear/kalman_qr_tf.py::tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop`

   Focused CPU-hidden QR tests passed after this patch.

3. The trusted GPU/XLA diagnostic then compiled an XLA cluster, but failed with:

   ```text
   Support for TensorList crossing the XLA/TF boundary is not implemented
   ```

   The process wrote a parseable blocker artifact and then exited `134`.

4. A second diagnostic-only patch inlined the existing QR while-loop
   implementation into the outer XLA diagnostic through `python_function` to
   avoid a nested `tf.function` boundary.  Focused CPU-hidden tests still
   passed, but the trusted GPU/XLA run again blocked with:

   ```text
   Support for TensorList crossing the XLA/TF boundary is not implemented
   ```

   At this point Phase 13 stop conditions fired.  Further repair needs a
   dedicated TensorList-boundary subplan rather than more ad hoc changes.

## Phase 13 Artifact

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-diagnostic-2026-07-07.json` | `2054` bytes | `9ef84f1dba32880215ef276b5c333afab8e050d505d5d27a6dab90971287c173` |

The artifact is below the 20 MB repository policy threshold for
claim-supporting artifacts.

Final diagnostic JSON:

- `schema`: `bayesfilter.neutra.xla_repair_result.v1`
- `decision`: `BLOCK_PHASE13_XLA_JIT_REPAIR_GATE`
- `error_type`: `UnimplementedError`
- `error_message`: `Support for TensorList crossing the XLA/TF boundary is not implemented`
- `training_executed`: `false`
- `hmc_executed`: `false`
- `external_sample_generation_executed`: `false`

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 13 used the admitted LGSSM QR route and the inherited Phase 9 XLA blocker as comparator; DSGE/c603 and LEDH evidence were not used. |
| Proxy promotion | The phase blocked; no finite compile or partial compile evidence is promoted to XLA readiness. |
| Missing stop conditions | The TensorList crossing-boundary failure triggered the stop condition for a dedicated repair subplan. |
| Hidden assumption | The `maximum_iterations` patch repaired only the original fixed-size TensorList complaint, not all TensorList/XLA issues. |
| Environment mismatch | GPU/XLA evidence came only from trusted/escalated execution; CPU-hidden tests are reported as support checks only. |
| Artifact mismatch | The blocker JSON is parseable and preserves no-training/no-HMC/no-sample nonclaims, but it is a blocker, not readiness evidence. |

Audit status: blocked with exact provenance.

## Local Checks

- `python -m py_compile bayesfilter/linear/kalman_qr_tf.py bayesfilter/testing/neutra_xla_repair_tf.py tests/test_neutra_xla_repair_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_xla_repair_tf.py tests/test_linear_qr_compact_loglik_tf.py -q`: passed, `20 passed, 2 warnings`, after both Phase 13 repair patches.
- `python -m json.tool docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-diagnostic-2026-07-07.json`: passed.
- Trusted `nvidia-smi`: passed.
- Trusted Phase 13 GPU/XLA diagnostic command: blocked and exited `134` after writing parseable blocker JSON.

The CPU-hidden pytest commands are support checks only. They are not GPU/XLA
evidence and not readiness evidence.

## Implementation Artifacts

- `bayesfilter/linear/kalman_qr_tf.py`
- `bayesfilter/testing/neutra_xla_repair_tf.py`
- `tests/test_neutra_xla_repair_tf.py`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase13-xla-jit-repair-diagnostic-2026-07-07.json`
- `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase14-xla-tensorlist-boundary-repair-subplan-2026-07-08.md`

## Phase Close Duties

1. Required local checks were run and recorded above.
2. This Phase 13 result records the blocker decision and nonclaims.
3. The Phase 14 TensorList-boundary repair subplan was drafted.
4. Bounded read-only review agreed with this blocker result and Phase 14
   subplan.

## Read-Only Review

- Claude one-path read-only review of this Phase 13 blocker result returned
  `VERDICT: AGREE`.
- Claude one-path read-only review of the Phase 14 subplan returned
  `VERDICT: AGREE` with non-blocking safeguards.
- The Phase 14 subplan was patched to explicitly require:
  - no optimizer updates, parameter updates, or HMC leapfrog steps inside the
    value/gradient diagnostic;
  - a run manifest with git commit, environment, CPU/GPU status, seed, wall
    time, output artifact path, and command.

## Nonclaims

- No NeuTra training was run.
- No HMC sampling or tuning was run.
- No external sample generation was run.
- No DSGE/c603 target was used.
- No broad XLA/JIT readiness claim is made.
- No HMC readiness claim is made.
- No posterior correctness, convergence, sampler superiority, production
  readiness, default execution readiness, default-policy change, or scientific
  validity is claimed.
