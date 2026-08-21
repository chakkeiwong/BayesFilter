# NeuTra Replication And HMC Result (2026-08-16)

## Outcome

The terminal `r6` campaign completed in `293.06 s` with valid SHA-256 hashes.
It used GPU 0, float64, XLA, TF32 disabled, TensorFlow memory growth, three
fresh training seeds per target, and the shared sequential HMC controller.

| Target | Fresh replication | HMC | Exact-law post-HMC |
|---|---:|---:|---:|
| Gaussian, cold `LR=1e-3` | 3/3 passed | Passed | Passed |
| Banana, root-preserving `LR=5e-4` | 2/3 passed | Blocked by replication veto | Not run |

Gaussian therefore passed the complete bounded replication-plus-HMC control
procedure. Banana remains unresolved because one fresh training seed failed
the exact-law replication gate; the HMC gate correctly did not run.

## Evidence Contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-replication-hmc-plan-2026-08-16.md` |
| Artifact root | `docs/plans/artifacts/neutra-replication-hmc-2026-08-16-r6/` |
| Training | 3,000 batch-native reverse-KL updates, batch 4,096 |
| Replication audit | 131,072 exact-law draws per fresh seed |
| HMC chains | 4 |
| HMC warm-up | Minimum 2,000, recent-window R-hat threshold `1.05`, cap 10,000 |
| HMC retained | Minimum 2,000, R-hat threshold `1.01`, bulk/tail ESS minimum `400`, cap 10,000 |
| HMC kernel | Fixed identity mass in z, tuned `L=(3,5,10,15,20,25)`, selected `L=5`; `L=1` forbidden |
| HMC nonclaims | No NUTS, no universal default, no superiority, no multimodal coverage, no SSL-LSTM transfer, no production HMC default |
| Git commit recorded | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |

## Gaussian Details

All three fresh training seeds passed coordinate means, coordinate second
moments, and adjacent cross moments. Importance-ESS fractions were `0.99686`,
`0.99680`, and `0.99692`; these are descriptive training diagnostics.

HMC tuning selected a finite kernel with `L=5`, step size `0.8674043`, and
verification acceptance probability `0.6959`. The tuning verification had
modern rank-normalized R-hat maximum `1.00297`, below `1.01`.

The sequential controller reached warm-up readiness at 2,000 draws per chain
with maximum warm-up R-hat `1.01045` under the `1.05` threshold. It reached
retained readiness at 2,000 draws per chain with minimum bulk ESS `7663.1` and
minimum tail ESS `7714.8`. All finite-state, target-score, movement, and
energy checks passed; native divergence status was unavailable from the TFP
kernel and is not interpreted as zero divergences.

The retained HMC draws passed all post-HMC exact-law coordinate mean,
second-moment, and adjacent cross-moment screens.

## Banana Details

Fresh seeds 10 and 11 passed all exact-law screens. Seed 12 failed only
coordinate mean index 10; its coordinate second moments and adjacent
cross-moments passed. Because the predeclared replication gate requires all
three seeds, HMC was not attempted.

This is a target-specific training repeatability veto, not evidence that the
root-preserving transport or HMC is impossible. The justified next action is a
banana-only fresh training repair/replication study focused on the remaining
seed-sensitive mean failure.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Gaussian replicated candidate | 3/3 fresh exact-law seeds | Passed | Three seeds are bounded evidence | Retain as a control HMC-valid candidate | Universal training/HMC default |
| Gaussian sequential HMC | R-hat/ESS/finite/movement/energy plus exact-law retained draws | Passed | Native divergence flag unavailable | Use as exact-law control authority for future harness checks | HMC superiority |
| Banana replicated candidate | 3/3 fresh exact-law seeds | Vetoed at seed 12 | One coordinate mean failure and seed sensitivity | Banana-specific repair/replication | Transport impossibility |
| SSL-LSTM transfer | Target-specific adapter and evidence | Not authorized | Controls are not SSL-LSTM | Do not start SSL-LSTM from this result | SSL-LSTM readiness |

### Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Gaussian passed all gates; banana failed replication seed 12 and HMC was blocked |
| Statistically supported ranking | None; no superiority or ranking claim is supported |
| Descriptive-only differences | ESS, acceptance, runtime, and loss are explanatory/descriptive |
| Default-readiness | Not supported |
| Next evidence needed | Banana target-specific repair/replication; any SSL-LSTM work requires a new adapter plan |

## Red-Team Note

The strongest alternative explanation for Gaussian success is that the analytic
control target and frozen transport are substantially easier than SSL-LSTM.
The strongest alternative explanation for banana failure is optimization basin
variation rather than a wrong permutation principle. Native TFP divergence
telemetry was unavailable, so the Gaussian HMC result must not be described as
a zero-divergence proof.
