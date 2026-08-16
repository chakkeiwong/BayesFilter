# Frozen weighted NeuTra HMC seed replication plan (2026-08-12)

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does the already admitted fixed-length HMC kernel remain operationally and analytically compatible when only the independent root seed changes? |
| Candidate | The v5 frozen six-stage, width-128 weighted IAF and its already selected identity-mass kernel: `L=20`, epsilon `0.14091138276334744`. |
| Exact authority | The normalized analytic target `0.8 N(mu_1,Sigma_1) + 0.2 N(mu_2,Sigma_2)` and its exact transformed-coordinate value/score. |
| Primary criterion | Every fresh replication passes the canonical sequential retained R-hat/ESS and hard-validity screens, and retained draws pass the analytic minority-mass screen. |
| Hard vetoes | Any provenance/hash mismatch, nonfinite state/target/score/Jacobian, target-status failure, chain without movement, positive native divergence when exposed, sequential readiness failure, missing mode, or analytic minority-mass interval failure. |
| Explanatory diagnostics | Acceptance, energy-error alerts, mode occupancy/transitions, wall time, means/covariances, and between-seed spread. They do not rank seeds or establish equality. |
| Nonclaims | No new transport-training claim, no sampler ranking, no proof of stationarity or distributional equality, no SSL-LSTM validity, and no general NeuTra default claim. |

## Frozen inputs and provenance

- Checkpoint: `docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/trainer_states.json`.
- Checkpoint SHA-256: `af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5`.
- Tuning source: `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5/tuning/tuning_result.json`.
- Tuning SHA-256: `6dfe2b8145040a18831a08032bfd61854189f2651e76c70842e59d4e4e12eb4f`.
- Frozen kernel: `num_leapfrog_steps=20`, `step_size=0.14091138276334744`, fixed identity mass, XLA enabled, policy `bayesfilter_neutra_sequential_hmc_v1`.
- Fresh root seeds: `(20260812, 91011)`, `(20260812, 91012)`, `(20260812, 91013)`, `(20260812, 91014)`. Seeds are assigned before observing results and are not selected or discarded based on posterior behavior.

## Execution policy

Each replication uses four mode-aware chains, 2,000 warm-up transitions per chain (archived and excluded), then cumulative retained sampling under the shared sequential controller, capped at 10,000 per chain. GPU/XLA, float64, TF32 disabled, and TensorFlow memory growth are required. The four roots may run independently in parallel, with no shared mutable state; two processes per physical GPU are the planned resource layout if the preflight confirms both GPUs are visible.

## Numeric/default audit

| Choice | Provenance/status | Risk and early check |
|---|---|---|
| `L=20`, epsilon | measured and frozen from v5 tuning; not re-tuned here | stale or mismatched artifact; exact hash and payload assertions |
| four roots | predeclared minimum replication set | insufficient transport-seed coverage; report one-transport scope |
| 2,000 warm-up / 10,000 cap | canonical policy defaults | cap can produce a non-result; readiness is a hard stop |
| GPU/XLA float64, TF32 off | repository reviewed defaults and v5 match | launch-invalid if memory growth or device manifest fails |
| analytic 99% minority interval | inherited reviewed diagnostic | marginal finite-sample screen, not joint equality evidence |

## Skeptical pre-execution audit

- **Baseline:** exact analytic target remains the authority; the learned density is coordinates only.
- **Proxy promotion:** acceptance, runtime, and per-chain occupancy remain explanatory; only sequential R-hat/ESS and declared analytic gates decide each replication.
- **Stop conditions:** provenance, device/memory, numerical, target-status, movement, sequential-cap, and analytic failures are recorded distinctly; a failed seed does not invalidate the harness or other seeds.
- **Fairness:** all roots use identical checkpoint, kernel, starts, controller, and diagnostics; only root seed differs.
- **Hidden assumptions:** parallel placement is a resource choice, not a scientific variable; if GPU preflight fails, stop before HMC rather than silently using CPU.
- **Artifact relevance:** each root archives warm-up/retained tensors, manifest, sequential result, and analytic result; a consolidated summary references immutable hashes.

Audit verdict: **PASS FOR FOUR FROZEN-KERNEL ROOT-SEED REPLICATIONS**.

## Preflight signature migration ledger

The first launch attempt stopped before any HMC transition because the current
analytic target signature differed from the v5 tuning signature. Investigation
found one and only one target-tensor delta: a covariance entry produced by the
same matrix formula is now serialized as `0.4` instead of
`0.39999999999999997`, a one-ULP float64 difference (`5.551115123125783e-17`).
On a deterministic 4,110-point bank, analytic log density, component
responsibilities, and exact score were bitwise equal between the v5 tensor and
the current tensor. Probabilities, means, target identity, dtype, checkpoint,
transport tensor hash, target implementation, and kernel are unchanged.

This is a provenance migration, not evidence that arbitrary signature drift is
safe. The runner must bind the immutable v5 run-manifest SHA-256, require exact
probability/mean identity, require every covariance delta to be no more than one
representable float64 step, reproduce the zero pointwise value/responsibility/
score delta, and bind the current live adapter signatures. One fresh GPU/XLA
fixed-kernel mechanics canary must pass before the four root runs. Any wider
delta, pointwise mismatch, numerical failure, or canary failure is a
continuation veto and requires fresh tuning.

The first canary retry stopped before transitions and localized the one-ULP
delta further: CPU TensorFlow evaluates that covariance element as `0.4`, while
the serious GPU route evaluates it as `0.39999999999999997` and reproduces the
exact v5 base and live sequential-adapter signatures. The v5 tuning artifact's
`7d188...` signature belongs to the tuner's separately constructed internal
adapter; the v5 run manifest's `6b4e...` signature belongs to the live sequential
adapter. Comparing those two distinct identities caused the second pre-transition
canary stop. The runner now verifies both identities in their proper roles and accepts
only two explicitly checked signature pairs: exact v5 on GPU, or the one-ULP
CPU reference pair. It records the realized pair, and the claim-bearing GPU
route must realize the exact v5 pair.

Migration audit verdict: **PASS FOR ONE CANARY, THEN THE UNCHANGED FOUR-ROOT
CAMPAIGN**. The rejected attempt and canary artifacts are preserved separately
and contain no HMC samples.

## Compute budget and stop rule

The measured v5 run took about 646 seconds including tuning. Frozen-kernel replications should be materially shorter, but each process has a 3,600-second controller cap and the campaign has a 14,400-second wall-clock cap. A timeout or readiness failure is a candidate rejection and repair trigger, never a pass.

## Planned artifacts

`docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-replications-2026-08-12/replication-{0..3}/` plus `summary.json`, `summary.md`, and a terminal manifest containing command, environment, GPU/memory policy, seeds, hashes, wall times, and worktree provenance.
