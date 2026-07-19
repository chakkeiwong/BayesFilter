# SSL-LSTM NeuTra Phase 8 Predictive Design Refresh Result

Date: 2026-07-17

Decision: `PHASE8_DESIGN_UNDERPOWERED_REPAIR_LADDER_EXHAUSTED_PHASE9_CLOSED`

## Outcome

Phase 8 resolved the comparator correctly but did not freeze a confirmation
design. G and H remain independently admitted peer replications; neither is a
posterior oracle. Ordinary HMC remains excluded. No Phase 9 G/H confirmation
forecast, feature difference, MMD value, or predictive decision was computed.

The engineering and target-pilot machinery passed, but two prospectively
controlled calibration designs failed power:

| Design | Receipt | Decision | Stop |
| --- | --- | --- | --- |
| 448 draws/chain, margins `0.15/log(1.15)`, symmetric Bonferroni | `controlled-calibration-nomination.json`, SHA `ec112880...` | `PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED` | 5/20 replications, no viable MMD tolerance |
| 1984 draws/chain repair arms B/C/D, including midpoint margins and IUT TOST | `power-repair-nomination.json`, SHA `56a34c4a...` | `PHASE8_POWER_REPAIR_NOMINATION_UNDERPOWERED_STOP` | 5/20 replications, no viable arm/tolerance |

Both early stops were prospective mathematical futility decisions: even if all
remaining outcomes were favorable, the frozen `16/20` decision thresholds
could no longer be reached for every required family.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Main question | No statistically valid and sufficiently powered one-to-ten-step predictive confirmation design was frozen under the bounded ladder |
| Comparator | Resolved: G and H are peers, not truth; controlled synthetic laws validate design only |
| Primary design pass | Failed: no tolerance or repair arm satisfied every required family screen |
| Hard continuation vetoes | No machinery/leakage veto; the bounded repair ladder itself exhausted with no viable candidate |
| Explanatory diagnostics | Per-family coverage/power counts, interval widths, MMD values, condition numbers, ridge choices, and runtimes |
| Nonclaims | No G/H predictive equivalence/difference, posterior correctness, sampler ranking, model adequacy, application adequacy, or default readiness |

## What Failed And What Did Not

Engineering correctness passed:

- target pilot preserved the `64/448` split and no-confirmation-leakage
  boundary;
- exact forecast chunking and fixed-shape pairwise median paths passed;
- controlled runners used TensorFlow/TFP `float64`, trusted GPU 1, XLA JIT,
  deterministic Philox domains, and immutable receipt/source bindings;
- all serious controlled runs had one trace per compiled surface;
- all long-run covariance rows were admissible, with ridge zero in the reported
  smoke and nomination summaries; and
- MMD intervals were admissible and no hard-veto classifier result occurred.

Statistical power failed:

- at 448 draws, required feature intervals were too wide for both equivalence
  and material detection, independent of MMD tolerance;
- at 1984 draws, midpoint-margin C/D repaired the required material controls,
  including local horizon effects, and IUT TOST improved null/equivalence
  behavior;
- nevertheless the required true-equivalent variance ratio `1.05` produced
  zero PASS outcomes in all B/C/D arms over the five fresh replications; and
- its log difference `log(1.05) ~= 0.0488` leaves only about `0.0628` inside
  the midpoint log-variance margin `0.1116`, while observed all-feature TOST
  widths were about `0.096-0.103` before accounting for estimate displacement.

The two-arm influence covariance scaling was re-derived and is correct. The
underpower result is not a factor-of-two bug. The failure was not caused by
GPU/XLA, covariance conditioning, MMD tolerance choice, local-horizon
alternatives, or G/H sampler evidence.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not freeze Phase 8 design | No arm/tolerance passed | Bounded design-power veto | Required sample size for the true-equivalent variance family | Reopen only with a prospectively powered larger-sample calculation or scientifically revised estimand/margin contract | Predictive-validation direction is false |
| Keep Phase 9 closed | No validated design exists | Pass | Actual G/H outcome remains blinded/uncomputed | Do not forecast confirmation suffix | G/H equivalence or material difference |
| Do not acquire more HMC yet | Controlled feasibility did not pass at prospective 1984 draws | Resource/scientific discipline | Larger draw count may or may not repair power | First derive a sample-size target and resource tradeoff; acquire only after review/authorization | More draws are useless or sampler invalid |
| Preserve machinery and pilot artifacts | All engineering/validity gates passed | No hard veto | Target dependence may differ from controls | Reuse exact lineage if Phase 8 is reopened | Posterior truth or broad validity |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | No implementation, covariance, MMD, GPU/XLA, lineage, or leakage hard veto |
| Statistically supported ranking | None; no repair arm remained viable, and five-replication continuous differences are descriptive |
| Descriptive-only differences | B/C/D per-family counts, interval widths, MMD values, condition numbers, and smoke behavior |
| Default-readiness | Not established |
| Next evidence needed | Prospective analytic/simulation sample-size study with fresh seeds, explicit resource cost, and independent validation; or a scientifically justified change to required estimands/margins before new outcomes |

## Run Manifest

| Field | 448-draw nomination | 1984-draw repair nomination |
| --- | --- | --- |
| Git commit | `ffaaaf903354e095da126dbfa47878c34717c5b8` dirty | same |
| Environment | `tfgpu`, TF `2.20.0`, TFP `0.25.0`, Python `3.13.13` | `tfgpu`, TF `2.20.0`, Python `3.13.13` |
| Device | physical GPU 1, XLA, TF32 enabled, `float64` | same |
| Seed | `(14001,14002)` | `(16001,16002)` |
| Wall time | `96.7839861450484` s | `98.80722479894757` s |
| Resource cap | `1800` s | `2400` s |
| Receipt SHA-256 | `ec112880f6e9f33432ad5c12f2ccc81efd71b40a75470fca45293a7aba225b49` | `56a34c4a254c38d89f682a22c4100d7df56d9aef460ae06d81e45de9d684e729` |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` | same |

Exact commands are preserved in the two JSON run manifests and their live
plans. The worktree remained dirty and unrelated changes were preserved.

## Post-Run Red Team

Strongest alternative explanation: the required AR(0.6) controlled families
may be more dependent than the actual target confirmation suffix. Dropping
them now would be post-outcome weakening, and target confirmation remains
blinded. A future target-relevant dependence calibration would require an
excluded pilot-derived prospective rule, not inspection of Phase 9 outcomes.

What would overturn the blocker: a reproducible interval/covariance arithmetic
defect, or a prospectively reviewed larger-sample design that passes fresh
development and independent validation. No arithmetic defect was found.

Weakest evidence: only five replications ran per ladder. The descriptive rates
are weak, but the stop decision is not an extrapolated rate comparison; it is
an exact consequence of the frozen count thresholds and remaining possible
successes.

## Handoff

Phase 9 remains closed. Reopen Phase 8 only through a new concise Tier-2 plan
that hard-binds both failed receipts and chooses one of these paths before any
new outcomes:

1. derive and test a larger prospective draw count capable of resolving the
   required true-equivalent variance family, with explicit HMC acquisition and
   forecast cost;
2. revise the scientific estimand/margin/family contract with a derivation and
   independent review, then restart development and validation with fresh
   seeds; or
3. accept that the current posterior sample budget supports only descriptive
   predictive comparison, not a formal predictive-equivalence claim.
