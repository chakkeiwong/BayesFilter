# SSL-LSTM q=20 physical annealed-SMC canary result (2026-08-10)

## Verdict

The adaptive global-resampling SMC canary passed mechanics and timing gates.  It
reached beta 1 in four stages with beta path
`[0.164922,0.607338,0.903961,1.0]`, performed three global systematic resampling
steps, and retained 65 distinct initial roots: 33 from the positive region and 32
from the negative region.  All states and target evaluations were valid.  Wall
time was `211.21 s`.

The terminal beta-1 population was measured before resampling.  Its ESS fraction
was `0.9376` and maximum weight `0.02292`.  These are mechanics/timing diagnostics
from one 100-particle run, not a posterior mass result.

## Gate table

| Gate | Result |
|---|---|
| Reach beta 1 within 24 stages | Pass, 4 stages |
| All states and exact targets valid | Pass |
| At least one global resampling | Pass, 3 |
| Positive initial-root ancestry survives | Pass, 33 roots |
| Negative initial-root ancestry survives | Pass, 32 roots |
| Wall time within 3,600 s | Pass, `211.21 s` |

HMC acceptance was `0.99`, `0.96`, and `0.96` at the three nonterminal stages.
No HMC proposal changed physical sign, and none was non-finite.  Thus the repair
mechanism observed here is global weighting/resampling ancestry, not local
cross-sign mutation.

## Decision

The canary authorizes a separately planned independent material SMC campaign.  It
does not authorize a posterior archive, NeuTra training, or predictive validation.
The material campaign must measure independent-batch uncertainty, terminal weight
quality, ancestry collapse, and sensitivity to the conditional-ESS target.

Artifact: `docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/canary.json`.
