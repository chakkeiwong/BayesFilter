# NeuTra Replication And HMC Validation Plan (2026-08-16)

## Research Intent Ledger

| Field | Predeclared statement |
|---|---|
| Main question | Do the Gaussian `LR=1e-3` and banana root-preserving `LR=5e-4` candidates replicate on fresh training seeds, and do replicated frozen transports support valid sequential HMC on the exact-law controls? |
| Candidate mechanisms | Fresh reverse-KL retraining with the previously viable target-specific configuration, followed by frozen-transport HMC. |
| Baseline | Exact known-law target itself, evaluated through the same HMC controller without a learned transport, is not used as a superiority comparator; it is a mathematical reference for output diagnostics. |
| Replication gate | Three fresh training seeds per target pass untouched 131,072-draw exact-law mean, second-moment, and adjacent-cross-moment screens. |
| HMC gate | Four chains; warm-up recent-window rank-normalized split/folded R-hat `<=1.05`; retained R-hat `<=1.01`; declared bulk/tail ESS `>=400`; finite state/target/score/energy; all chains move; no divergences when exposed; no energy-error veto. |
| HMC target agreement | Retained draws in the exact latent coordinate system pass the same exact-law predictive/moment screens. This is required for control validity but does not prove general posterior correctness. |
| Hard vetoes | Nonfinite transport or HMC trace, invalid value/score binding, HMC `L < 2`, warm-up cap without readiness, retained convergence/ESS failure, no chain movement, divergence/energy veto, exact-law audit failure, or invalid artifact provenance. |
| Explanatory diagnostics | Training loss, ESS fraction, ratio SD, acceptance probability, runtime, step size, leapfrog count, and standardized discrepancies. Acceptance is never a convergence gate. |
| Nonclaims | No universal training default, no statistical superiority, no multimodal coverage, no SSL-LSTM transfer, and no production/HMC default promotion from two controls. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Three fresh training seeds | Stronger replication than prior two-seed confirmation | Tests repeatability without ranking | Still not publication-scale replication | Per-seed exact-law screen | Reviewed minimum |
| Gaussian `1e-3` | Prior control-repair candidate | Passed both prior confirmations | Target-specific only | Fresh exact-law gate | Reviewed warm start |
| Banana root-preserving `5e-4` | Prior control-repair candidate | Passed both prior confirmations | Basin may remain seed-sensitive | Fresh exact-law gate | Reviewed warm start |
| HMC four chains | Shared controller minimum | Required for rank-normalized multi-chain diagnostics | Four chains do not prove global exploration | Chain movement and R-hat/ESS | Reviewed minimum |
| Warm-up/retained limits | Shared NeuTra policy: 2,000 minimum, 10,000 cap; retained 1,000 minimum, 10,000 cap | Prevents fixed short-chain claims | Runtime may hit cap | Durable progress and cap veto | Policy default |
| HMC kernel tuning | Target-specific fixed leapfrog grid `L=(3,5,10,15,20,25)` and acceptance band | L=1 is forbidden; tuning precedes sequential run | Small grid may miss a usable kernel | Tuning result and verification | Reviewed hypothesis |

## Execution Design

1. Retrain each target candidate on three fresh seeds with exactly 3,000
   batch-native reverse-KL updates, GPU 0, float64, XLA, TF32 off, and memory
   growth enabled before TensorFlow initialization.
2. Apply the untouched exact-law audit to each replication. Continue to HMC
   only for candidates whose three replication seeds pass.
3. For each replicated candidate, tune a fixed HMC kernel on a disjoint tuning
   partition. Reject `L=1`; require tuning verification to pass before the
   sequential controller starts.
4. Run the shared sequential controller with warm-up retained but excluded,
   recent-window warm-up R-hat, retained R-hat/ESS, finite/energy/movement
   vetoes, and durable chunk artifacts. Use no NUTS.
5. Apply exact-law diagnostics to retained HMC draws in both latent and model
   coordinates. Do not claim HMC validity if any gate fails.

## Skeptical Plan Audit

| Risk | Disposition |
|---|---|
| Prior training seeds leak into replication | Vetoed: replication seeds and stateless training/audit partitions are fresh. |
| Training audit doubles as HMC validation | Vetoed: HMC tuning, warm-up, retained, and exact-law post-HMC draws use distinct seeds. |
| Learned transport is mutable during HMC | Vetoed: training completes before adapter binding; variables are frozen by construction and a state hash is recorded. |
| HMC acceptance is treated as convergence | Vetoed: acceptance is explanatory only; R-hat/ESS and exact-law gates decide. |
| HMC short-run passes falsely | Vetoed: shared sequential minimum/cap and retained diagnostics apply; cap without readiness is a hard veto. |
| HMC is run with `L=1` | Vetoed: tuner and runner reject leapfrog counts below 2. |
| HMC API mismatch invalidates the result | Early mechanics/value-score binding check and target-specific adapter tests precede long chains. |
| A candidate failure rejects NeuTra generally | Vetoed: classify training, binding, sampler, or target-specific failure separately. |

Audit verdict: this plan preserves fresh replication, freezes the learned
transport before HMC, uses the shared sequential controller, and separates
all nomination, tuning, convergence, and exact-law evidence classes.

## Artifacts And Stop Conditions

Output root:
`docs/plans/artifacts/neutra-neutra-replication-hmc-2026-08-16/`.
Record commands, commit, environment, GPU/memory policy, target/config/state
hashes, seeds, per-seed training results, HMC tuning and sequential archives,
post-HMC exact-law diagnostics, and SHA-256 hashes. Stop a cell on any hard
veto or infrastructure failure; do not downgrade a veto to a descriptive
result.
