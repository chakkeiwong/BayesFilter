# NeuTra Full-Validation Execution Plan (2026-08-17)

## Question

Do the existing learned NeuTra transports and shared sequential HMC procedure
pass the first known-law controls before geometry and application testing?

## Scope

This execution covers the harness plus two source-bound analytic controls:

1. frozen two-component Gaussian-mixture transport;
2. frozen three-component Gaussian-mixture transport.

The standard/correlated Gaussian harness is covered by the focused TensorFlow
and transport/HMC contract tests. Geometry and application targets remain later
rungs and are not promoted by this campaign.

## Evidence Contract

| Item | Contract |
|---|---|
| Baseline | Existing frozen weighted NeuTra checkpoints, exact Gaussian-mixture value/score adapters, and shared sequential HMC controller |
| Primary criterion | Sequential HMC and retained analytic output-law screens both pass for a control target |
| Hard vetoes | Nonfinite target/score/transport, invalid checkpoint hash, missing XLA or memory-growth verification, `L < 2`, failed tuning, failed sequential convergence/ESS/energy/movement, failed analytic screens |
| Explanatory only | Loss, acceptance, runtime, and isolated moment differences |
| Nonclaims | No superiority, no universal NeuTra claim, no geometry-target claim, no SSL-LSTM claim, no default promotion |
| Artifact | Fresh per-rung roots under `docs/plans/artifacts/neutra-full-validation-2026-08-17/` |

## Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Existing two-mode checkpoint | Prior reviewed analytic control | Stale or mismatched transport identity | Loader SHA-256 and semantic-hash checks | Baseline reuse |
| Existing three-mode checkpoint | Prior reviewed multimodal control | Component-aware initialization can hide poor global mixing | Compare per-mode occupancy and transitions; do not use initialization as proof | Baseline reuse |
| Four chains | Shared sequential-controller minimum for serious evidence | Too few chains can miss nonconvergence | R-hat, ESS, energy, movement and output-law checks | Reviewed default |
| `L >= 2` | Repository HMC policy | One-step trajectory is not a meaningful HMC control | Runner static guard and tuning-grid inspection | Hard policy |
| GPU 0/1 split | Current device inventory | Concurrent jobs could contend or exhaust memory | Memory-growth manifests and `nvidia-smi` monitoring | Operational choice |

## Skeptical Review

- A canary passing only establishes finite execution and target/score plumbing;
  it does not establish posterior correctness.
- A full run passing sequential diagnostics still requires retained analytic
  output-law screens; acceptance and training loss cannot substitute.
- Component-aware starts may expose mode mechanics but can overstate mixing;
  mode occupancy and transitions remain diagnostic and are reported separately.
- Existing checkpoints are not retuned in this execution. A checkpoint or
  tuning failure is a baseline/artifact failure, not evidence against NeuTra.
- Concurrent GPU execution is valid only with memory growth configured before
  TensorFlow initialization and distinct visible devices.

Review verdict: the plan is fit for bounded execution. The first stage is the
two canaries; a failed canary is a continuation veto for its full run only.
The other control may continue if its own canary and artifact are valid.

## Execution

1. Run focused harness, mixture, and runner-contract tests.
2. Write preflight manifests for both runners.
3. Run two-mode and three-mode GPU/XLA canaries concurrently, one per GPU.
4. If each canary passes, run the corresponding full tuned sequential HMC
   campaigns concurrently, one per GPU, with their declared time caps.
5. Validate artifact hashes and extract sequential/analytic gate statuses.
6. Write a terminal result and reset memo separating hard veto, descriptive
   evidence, and promotion status.

## Commands

```bash
python docs/benchmarks/run_defensive_weighted_neutra_analytic_hmc_2026_08_12.py \
  --mode preflight --output-root <two-mode-preflight>
python docs/benchmarks/run_weighted_neutra_three_mode_hmc_2026_08_12.py \
  --mode canary --output-root <three-mode-canary> --device 1
```

The actual command and environment are recorded in every runner manifest.

## Stop Conditions

- Stop a lane immediately on a hard veto or cap.
- Do not interpret a failed canary's runtime or acceptance.
- Do not launch full HMC for a lane whose canary, checkpoint, or memory/XLA
  provenance fails.
- Preserve all failed artifacts and classify the failure before any repair.
