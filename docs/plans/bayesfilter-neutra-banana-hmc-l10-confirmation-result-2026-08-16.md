# NeuTra Banana Frozen L=10 Confirmation Result (2026-08-16)

## Outcome

The fixed-kernel confirmation completed in `88.21 s` on GPU 0. It replayed the
seed-15, 6,000-update learned banana transport and froze the central-selected
kernel `L=10`, step size `0.7709722545680272`. No tuning, transport adaptation,
mass adaptation, threshold change, or start-bank selection was performed.

Both the original iid-normal and central deterministic start banks passed the
longer confirmation with 5,000 retained draws per chain. This supports `L=10`
as a target-specific viable kernel for this frozen learned banana transport.
It is not a repository default, universal kernel, superiority claim, or
SSL-LSTM result.

## Evidence Contract

| Item | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-neutra-banana-hmc-repair-plan-2026-08-16.md` |
| Discovery root | `docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/` |
| Confirmation root | `docs/plans/artifacts/neutra-banana-hmc-l10-confirmation-2026-08-16-r1/` |
| Learned state | Seed `15`, 6,000 updates, replayed state hash equal to discovery |
| Frozen kernel | Identity z mass, `L=10`, step `0.7709722545680272` |
| Warm-up | 2,000 transitions per chain, recent-window R-hat threshold `1.05` |
| Retained | 5,000 draws per chain, R-hat threshold `1.01`, ESS gate `>=400` |
| Proposal audit | 131,072 draws on exact discovery partition; passed |
| Integrity | 69 confirmation artifacts; all SHA-256 hashes passed |
| Nonclaims | No universal default, no superiority, no SSL-LSTM transfer, no production readiness |

## Results

| Bank | Warm-up | Retained convergence/health | Retained exact-law | Status |
|---|---:|---:|---:|---:|
| Original iid-normal | Pass, max R-hat `1.00708` | Pass, 5,000/chain, max R-hat `1.00136` | All three screen families pass | Confirmed |
| Central deterministic | Pass, max R-hat `1.00708` | Pass, 5,000/chain, max R-hat `1.00136` | All three screen families pass | Confirmed |

The retained controller gate passed for both banks, including the declared
bulk/tail ESS minimum, finite state/target/score/log-acceptance, all-chain
movement, and energy checks. Native TFP divergence telemetry was unavailable;
this is not interpreted as zero divergences. The retained exact-law screens
passed coordinate means, coordinate second moments, and adjacent cross moments
for both banks.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Frozen `L=10` banana kernel | Two-bank 5,000-draw confirmation | Passed | One learned transport state and one target | Use as a target-specific banana HMC candidate for downstream predictive testing | Universal/default kernel |
| Start-bank sensitivity | Same frozen kernel passes both banks | Not supported | Only two banks tested | Retain original bank; no start-bank repair is required for this kernel | Universal start policy |
| `L=5` failure | Matched cross-over failed both banks | Vetoed for this transport | Other `L` values not cross-over-confirmed | Do not use `L=5` for this frozen learned transport | Universal `L=5` invalidity |
| Scientific/default readiness | Complete downstream evidence | Not established | No predictive-equivalence or SSL-LSTM run | Run banana predictive-equivalence diagnostics next if desired | Production/default readiness |

| Evidence class | Status |
|---|---|
| Hard veto screen | Both fixed-`L=10` confirmation banks passed; prior fixed-`L=5` cross-over remains vetoed |
| Statistically supported ranking | None; no superiority claim |
| Descriptive-only differences | Runtime, acceptance, R-hat progression, and kernel comparison |
| Default-readiness | Not supported |
| Next evidence needed | Predictive-equivalence testing using the frozen banana HMC candidate, then a new target-specific SSL-LSTM plan |

## Red-Team Note

The strongest remaining alternative is that the exact-law moment screens are
limited summaries of the full banana law. The next scientific check should
compare predictive/output distributions, not only these moments. The current
result proves a bounded exact-law HMC control screen for this target and frozen
transport; it does not prove full posterior correctness or transfer.
