# NeuTra Curriculum Control Campaign Result (2026-08-15)

## Outcome

The reviewed target-specific curriculum-search campaign completed for the
correlated Gaussian and banana controls in `1929.60 s` (32.16 minutes), below
the `3600 s` cap. Both target processes used GPU 0, float64, XLA, batch-native
updates, and TensorFlow memory growth. All campaign, target, and manifest
SHA-256 checks passed.

The curriculum-search mechanism is **not promoted**. The searched protocol did
not pass the untouched exact-law predictive gate on either target. This is a
candidate/protocol failure, not a failure of the harness or evidence against
NeuTra generally.

## Evidence Contract And Provenance

| Item | Value |
|---|---|
| Question | Can measured target-dependent group activation select a viable NeuTra training protocol on Gaussian and banana controls? |
| Baseline | Cold joint training at each of the three learning rates, with equal 3,000-update tournament work |
| Search | Four replicated 100-update sibling probes, beam width 2, depth 3, cap 80 probe calls |
| Tournament | Four paired replicates per `(sequence, learning rate)` arm, 3,000 updates each |
| Promotion gate | Two fresh seeds per selected protocol, 131,072 untouched exact-law draws, 99.9% mean/second-moment/adjacent-cross-moment screens |
| Hard vetoes | Nonfinite state, invalid budget/partition/provenance, or any exact-law screen failure |
| Explanatory only | Probe loss/LCB, tournament loss, ESS, ratio SD, runtime, terminal gradient norm |
| Nonclaims | No HMC, posterior-correctness, multimodal-coverage, SSL-LSTM-transfer, universal-curriculum, or default-readiness claim |
| Plan | `docs/plans/bayesfilter-neutra-curriculum-control-campaign-plan-2026-08-15.md` |
| Campaign artifact | `docs/plans/artifacts/neutra-curriculum-control-campaign-2026-08-15/campaign_result.json` |
| Git commit recorded by run | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |

The numeric probe threshold and tournament tolerance were target-specific
repeatability measurements: 16 exact-law calibration batches gave SD
`0.0142462754`, two-SD margin `0.0284925508`, and probe threshold
`0.0002849255` per update. These are selection tolerances only; they did not
relax the final predictive gate.

## Results

| Target | Search result | Tournament selection | Selected fresh gate | Cold comparator gate |
|---|---|---|---:|---:|
| Gaussian | Beam terminal sequences were `stage_2 -> stage_1 -> stage_0_residual` and `stage_2 -> stage_1 -> simple_linear_scale` | Cold joint, `LR=2e-4` (all nine arms were in the practical uncertainty set) | 0/2 | 2/2 at `LR=1e-3` |
| Banana | Only `stage_1` passed the local probe screen | Cold joint, `LR=2e-4` (all six arms were in the practical uncertainty set) | 0/2 | 0/2 at `LR=2e-4` |

### Gaussian

The selected cold `2e-4` protocol failed adjacent cross-moment screens in
both fresh seeds (three failures in seed 0 and three in seed 1), despite
passing coordinate means and second moments. Its importance-ESS fractions
were `0.9886` and `0.9867`; these are descriptive, not promotion evidence.

The separately retained cold `1e-3` comparator passed every exact-law screen
in both seeds, with ESS fractions `0.9969` and `0.9971`. This shows that the
Gaussian architecture and harness can produce a viable proposal under this
budget, but the uncertainty-set tie rule selected the wrong cold learning rate
for the fresh gate. The result does not support a statistically ranked LR
claim; it exposes a selection-policy repair trigger.

### Banana

The selected cold `2e-4` protocol failed the first two coordinate second-moment
screens in both seeds (`0.94899`/`1.01773` in seed 0 and corresponding first
two-coordinate distortion in seed 1). Coordinate means and adjacent
cross-moments passed. Both selected and comparator gates therefore failed.

The search itself found no broadly supported activation group: only `stage_1`
was eligible, and the tournament loss uncertainty set included both cold and
that one-stage protocol. This is consistent with a target-specific banana
optimization/basin problem, not evidence that the search code is invalid.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Curriculum search promotion | Selected protocol passes both fresh exact-law seeds | **Vetoed** for Gaussian and banana | Gaussian LR-selection rule; banana optimization basin | Repair selection on Gaussian and run a target-specific banana initialization/order study | Universal curriculum failure |
| Harness/engineering validity | Finite GPU/XLA runs, valid budgets/partitions, hashes | **Passed** | None material in this campaign | Retain harness | Scientific validity |
| Gaussian viability | Untouched exact-law gate for cold comparator | **Passed** for cold `1e-3`, 2/2 | Only two fresh seeds | Use cold `1e-3` as a control baseline, not a promoted default | Superiority or universal LR ranking |
| Banana viability | Untouched exact-law gate | **Vetoed**, all tested routes 0/2 | Initialization/order/capacity/basin | Fresh target-specific protocol study | Architecture impossibility |

### Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Selected protocol failed Gaussian and banana exact-law gates; banana cold also failed |
| Statistically supported ranking | None; tournament differences and ESS/ratio-SD differences are descriptive |
| Descriptive-only differences | Gaussian cold `1e-3` passed while selected `2e-4` failed; this is replicated viability evidence, not a ranking claim |
| Default-readiness | Not supported |
| Next evidence needed | Repair the Gaussian tie/selection policy and run a fresh, target-specific banana initialization/order/capacity diagnostic |

## Red-Team Note

The strongest alternative explanation for Gaussian selection failure is that
the practical uncertainty set was intentionally conservative and the
shortest-sequence tie rule selected `2e-4` even though the `1e-3` cold arm was
the only cold arm with both fresh seeds passing. The next repair should retain
all equal-work arms but require the final LR choice to use a predeclared
fresh-seed nomination rule or to reject an uncertainty tie when exact-law
control behavior is heterogeneous.

For banana, the strongest alternative explanation is seed-dependent reverse-KL
optimization into a distorted variance basin. A different group order,
identity-biased initialization, longer target-specific budget, or capacity arm
could discriminate these causes. The weakest evidence is any ranking among
the failed banana arms.
