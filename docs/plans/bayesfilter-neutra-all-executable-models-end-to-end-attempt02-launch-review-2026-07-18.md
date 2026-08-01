# NeuTra All-Executable-Models Attempt 02 Launch Review

Date: 2026-07-18

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Status: `PASS_TO_SERIOUS_ATTEMPT_02`

## Corrected preflight

Fresh artifact root:
`docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/real-hmc-preflight-attempt-02`

The PP-UKF preflight completed one batched GPU/XLA training step, frozen versus
trainable parity, held-out target evaluation, and the real BayesFilter native
fixed-transport HMC tuner. It emitted one finite, error-free tuning row with a
finite adapted step, finite samples, and finite log-accept ratios. The tiny
tuning call did not pass the scientific acceptance screen, which is expected
and irrelevant under its two-result engineering budget. Its role was to prove
the real status/transport/HMC integration, not tune a usable kernel.

The run manifest records RTX 4080 SUPER execution, TensorFlow 2.19.1, XLA,
TF32, and memory growth configured before logical-device initialization. Wall
time was about 111 seconds.

## Final implementation and duplication review

- Training delegates to `train_plain_dense_iaf`; final-training segmentation
  delegates every numeric segment to the same function with
  `resume_infrastructure_from`.
- Native tuning delegates to `tune_fixed_transport_hmc_kernel` with target
  acceptance 0.70, band `[0.65,0.75]`, identity mass in `z`, and empty fixed
  grids.
- Warm-up/retained execution delegates to `run_sequential_neutra_hmc` and
  convergence delegates to `rank_normalized_hmc_diagnostics`.
- Target status normalization delegates to
  `batch_native_value_status_target_fn`; no second status schema or health
  validator was implemented.
- AST/search review finds no NumPy import, direct TFP sampler construction,
  copied HMC kernel, copied R-hat/ESS, fake chain, or historical benchmark
  import in the campaign implementation.
- `16 passed` for the focused all-model, telemetry, API, manifest, resume, and
  segmented-training suite. Compile and scoped diff checks pass.
- The campaign launches only the five executable registry cells. Valid
  candidate failure is recorded and does not prevent later independent cells;
  a missing result or failed child process remains a shared/cell harness veto.

No remaining code-correctness, mathematical-surface, or duplication issue was
found. The attempt-01 result remains invalid harness evidence and is not reused.

## Launch

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
python docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action campaign \
  --output-root docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02 \
  --screen-steps 500 \
  --final-steps 5000 \
  --final-segment-steps 1000
```

The scientific contract, target set, recipes, seeds, tuning criteria, sampling
caps, and campaign budget are unchanged. Attempt 02 uses fresh output paths and
must be supervised to a terminal aggregate result or a structured blocker.

## Attempt 02 terminal infrastructure event

Attempt 02 completed `LGSSM-EXACT` and `PP-UKF` with valid cell-level
`TUNING_FAILED` results, then failed closed before PP-SGQF training because its
freshly constructed direct target signature differed from the registry value.
An immediate post-run audit reconstructed all five adapters and found exact
declared/observed agreement, including PP-SGQF. The most plausible explanation
is concurrent dependency-source drift from the other active repository lane
during the long campaign, followed by restoration. The mismatch is a target
identity continuation veto for that child and is not a NeuTra result.

The localized retry uses a fresh continuation root and runs only `PP-SGQF`,
`SIR-SGQF`, and `STR-UKF`. The CLI subset is orchestration only: it validates
every requested ID against the executable registry and does not change target,
recipe, seed, training, tuning, or sampling semantics. Attempt-02 evidence is
preserved and the completed cells are not rerun.

That retry reproduced the PP-SGQF mismatch before training. CPU produced
`32616875...` while the trusted GPU child produced `90286845...`. The target
identity had hashed raw TensorFlow serialization bytes for the SGQF cloud;
those bytes were device-dependent even though the mathematical points and
weights agreed. The repaired identity binds the backend-independent cloud
construction manifest instead: family, dimension, sparse level, rule family,
active multi-indices, combination coefficients, tolerances, ordering,
representation, and maximum univariate level. Point count and negative-weight
count remain explicit filter fields. The filter tensors and target math are
unchanged.

After repair, independent CPU-hidden and trusted-GPU constructors both issued
`373326607b8cb06f274f03e0a523a47b24b83e35c8b37c8d264b500a6234fbac`.
The registry is refreshed to that signature. The planned one infrastructure
retry for PP-SGQF has been consumed; its next full launch requires a recorded
budget extension. SIR-SGQF and STR-UKF remain unexecuted and continue in a
fresh subset root without waiting on PP-SGQF.
