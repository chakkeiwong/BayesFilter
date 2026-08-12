# KSC-UKF Gaussian-Sum NeuTra and HMC Continuation Plan

Date: 2026-08-02  
Status: `REVIEWED_READY_TO_EXECUTE`

## Objective

Use the repaired, GPU/XLA-admitted KSC Gaussian-sum UKF route at its exact
admission scope (`T=20`, seven KSC observation components, component cap 32) to
test whether a batch-native NeuTra transport and the shared sequential HMC
controller can be executed without changing the target.

This plan does not revive the historical single-Gaussian `KSC-UKF` registry
cell, whose `T=1000` route remains blocked. The repaired route has a distinct
target identity and artifacts.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can the admitted repaired KSC approximate posterior support valid batched GPU/XLA NeuTra training, scope-specific HMC tuning, and sequential sampling? |
| Candidate | `bounded_deterministic_mass_preserving_clustered_gaussian_sum_ukf`, `T=20`, cap 32. |
| Baseline | Identity-affine transport around the source-coordinate origin, with the same repaired target and data. No SVX-ZC recipe or kernel setting is transferred. |
| Primary promotion criterion | Batch-native training and frozen-transport parity pass; broad-grid HMC yields a complete viable candidate set with one unique survivor; sequential HMC passes finite/status, warm-up readiness, modern R-hat, bulk/tail ESS, movement, and declared health gates. |
| Promotion veto | Target/signature/data mismatch, nonfinite values/scores, invalid status, scalar or row-mapped training fallback, failed frozen transport parity, missing GPU/XLA or memory growth, broad-grid health failure, or sequential convergence/health failure. |
| Continuation veto | Harness invalidity, corrupted target/artifact, unavailable trusted GPU, or exhausted campaign budget. Candidate rejection is not research-direction rejection. |
| Repair trigger | Localized batch-shape, XLA, memory, serialization, training, tuning, or sampler infrastructure failure under the unchanged target and budget. |
| Explanatory diagnostics | Training loss, retained component counts/mass, score residuals, acceptance, energy error, runtime, and truth-tail p-values. These cannot rank stochastic candidates without uncertainty evidence. |
| Nonclaims | No exact likelihood, exact score, source-faithfulness, statistical superiority, universal capacity, posterior correctness beyond declared sampler gates, production readiness, or cross-scope transfer. |

## Frozen Scope And Evidence Contract

The target scope is issued by
`make_ksc_gaussian_sum_ukf_neutra_adapter()` and must remain exactly:

- data: frozen seed `81101`, raw observations shape `[20,1]`;
- transform: `log(y^2 + 1e-8)`;
- KSC mixture: seven fixed components from `ksc_1998_log_chi_square_mixture()`;
- component cap: `32` with deterministic nearest-center moment merging;
- chart: `gamma=0.1+0.8 Phi(theta[0])`, `beta=0.1+0.8 Phi(theta[1])`;
- dtype: TensorFlow `float64`;
- execution: GPU, XLA JIT, TF32 enabled, memory growth verified before device initialization;
- target signature: repository-issued, not caller-stamped.

The prior GPU canary artifact
`docs/plans/artifacts/bayesfilter-ksc-ukf-gpu-admission-neutra-20260802/canary-attempt01/result.json`
is the admission prerequisite. It passed cap 32 CPU/GPU value and score parity
and all CPU cap rows. It does not authorize training by itself.

## Phases

1. **Adapter and target preflight**
   - Run focused KSC route, batch, finite/status, permutation, and contract tests.
   - Run one GPU/XLA one-step recipe preflight for each declared recipe only if the adapter signature matches the fresh KSC contract.
2. **KSC recipe screen**
   - Use four KSC-specific dense-IAF recipes: narrow/wide hidden layers `(8,8)` and `(16,16)`, learning rates `1e-3` and `5e-3`.
   - Use batch size 128, 500 updates, GPU/XLA, and common paired seeds.
   - Treat observed loss differences as descriptive; select the smallest viable representative if all hard gates pass.
3. **Selected final training**
   - Consume only the preserved screen handoff and its hash.
   - Train one selected recipe for 5,000 GPU/XLA updates at batch 128.
   - Require held-out value/status and frozen/trainable transport parity before tuning.
4. **Statistical broad-grid HMC tuning**
   - Use the repaired generic tuner: independently tune epsilon for `L=(3,5,9,13,18,25)`, three replications over four chains, 90% replication-mean compatibility with target acceptance 0.70, and nonrecursive same-epsilon `L +/- 1` coverage.
   - Preserve every viable pair without unsupported stochastic ranking. A unique survivor is required for sequential handoff.
5. **Sequential NeuTra HMC**
   - Use the shared `bayesfilter_neutra_sequential_hmc_v1` controller with the unique hash-bound pair, at least 2,000 warm-up transitions per chain, recent-window warm-up R-hat `<=1.05`, retained modern R-hat `<=1.01`, declared bulk/tail ESS, finite/status/movement checks, and energy diagnostics as declared health checks only.
6. **Terminal result and reset memo**
   - Record decision and inference-status tables, hard vetoes, descriptive-only differences, uncertainty limits, strongest alternative explanation, and next evidence needed.

## Budget And Artifacts

- Adapter tests and preflight: at most 20 CPU minutes and one GPU preflight process.
- Recipe screen: one bounded GPU run, 500 steps per recipe, four recipes, batch 128.
- Final training: one 5,000-step GPU run.
- Broad-grid tuning and sequential HMC: one campaign under the existing 24-hour project budget; use fresh versioned roots and never overwrite earlier SVX-ZC or KSC admission evidence.

Planned roots:

```text
docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/
  preflight/
  screen-attempt01/
  final-training-attempt01/
  broad-grid-attempt01/
  sequential-hmc-attempt01/
```

Every serious artifact must include command, git commit, environment, device,
memory policy, XLA/TF32, seed, target signature, wall time, plan path, output
hashes, and nonclaims.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
| --- | --- | --- | --- | --- |
| `T=20`, cap 32 | KSC CPU/GPU admission artifact | A different horizon/cap is a different target scope | signature and data-hash checks | binding scope |
| Four `(8,8)/(16,16)` recipes | KSC-specific bounded hypothesis | capacity or learning rate may be inadequate | 500-step status/loss screen | tuning candidates only |
| Batch 128 | repository NeuTra batching policy and prior successful lane | memory exhaustion or hidden scalar fallback | batch-native target/status and GPU allocator telemetry | binding training policy |
| Fixed identity mass and broad-grid protocol | reviewed generic HMC tuner policy | acceptance point estimates can misclassify candidates | replication intervals and one-hop coverage | binding tuning policy |
| Acceptance/energy diagnostics | repository statistical policy | can be mistaken for convergence or veto | R-hat/ESS/status gates remain primary | explanatory/health only |

## Skeptical Plan Audit

- **Wrong baseline:** identity-affine is only a geometry baseline for this KSC scope; no completed model's transport or HMC settings are treated as KSC defaults.
- **Proxy promotion:** training loss, acceptance, energy, runtime, and truth-tail values cannot promote a target or rank stochastic candidates without the declared gates and uncertainty evidence.
- **Missing stops:** every phase has a fresh root, finite budget, no-training-on-adapter-failure stop, unique survivor requirement, and sequential convergence vetoes.
- **Unfair comparisons:** every recipe uses the same target, batch size, paired seeds, frozen scope, and GPU/XLA backend.
- **Hidden assumptions:** cap, horizon, recipe family, batch size, identity mass, and epsilon/L grid are explicitly bound and diagnosed.
- **Artifact mismatch:** target signature and screen/transport hashes are checked before each downstream phase; old registry evidence cannot satisfy this plan.

Audit verdict: `PASS_FOR_BOUNDED_KSC_NEUTRA_HMC_CAMPAIGN`.

