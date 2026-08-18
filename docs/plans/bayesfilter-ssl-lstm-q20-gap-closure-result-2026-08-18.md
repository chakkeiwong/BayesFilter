# SSL-LSTM q=20 Gap-Closure Result (2026-08-18)

Plan: `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md`

## Outcome

The campaign closed the engineering and bounded discovery phases, and tested the
most plausible physical-sampler repair. The result is **not** a valid SSL-LSTM
posterior archive.

- Batch-native/XLA, physical replica mechanics, annealed-SMC, and predictive-test
  contract checks passed (`24 passed` in the focused preflight suite).
- A fresh dense-mass physical replica canary passed all short-run nomination
  screens: zero invalid paths, valid worker/XLA identity, all adjacent pairs
  communicated, all chains visited both signs and changed sign locally at the hot
  endpoint, and acceptance means remained in the declared band.
- A fresh 500-transition dense-mass material run failed the predeclared warm-up
  convergence gate at every milestone. Its final recent-window maximum modern
  R-hat was `1.3175028642 > 1.05`; no retained draws were admitted.
- The two previously tested hotter-ladder and hot-step hypotheses also failed
  all-chain hot-forgetting selection screens. The dense mass repaired movement,
  not chain agreement.
- A fresh 32-start target-query multistart search found only the two known
  stationary regions within the declared score/log-density tolerances. This
  weakens the undiscovered-third-mode hypothesis but cannot prove exhaustive
  mode absence.
- NeuTra retraining, NeuTra HMC, posterior-predictive sampling, and the five
  output-law tests were correctly not launched because the physical posterior
  archive gate failed.

## Evidence Contract Status

| Phase | Primary question | Result | Role |
|---|---|---|---|
| A | Current target and harness are batch-native/XLA and artifact-safe | Passed; `24` focused tests | Engineering admission |
| B | Dense physical candidate has valid local/global travel mechanics | Passed in fresh 100-transition canary | Candidate nomination only |
| C/H1 | Dense mass `0.35`, `L=8` yields converged global physical chains | Failed warm-up R-hat; travel/forgetting passed | Promotion veto; repair evidence |
| H2 | Hotter endpoint/ladder repairs all-chain forgetting | Failed in prior bounded arms: ratio `0.40`, `0.35`, hot step `1.5x`, `2.0x` | Hypothesis weakened |
| H3 | Broader target-query starts reveal another high-density stationary cluster | No new competing cluster in 32 starts | Explanatory diagnostic only |
| D | Globally representative NeuTra training | Not opened | Upstream archive veto |
| E | NeuTra HMC and posterior-predictive output-law comparison | Not opened | Upstream archive veto |

## Dense-Mass Material Details

Artifact: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/physical/r2-dense-mass-material-retry/material.json`

Artifact SHA-256: `64ea8799f9362a31cdc6e980ce00c158abde51e9941dc0b95d86d870d53e78e2`

Configuration was bound to the fresh canary artifact
`docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r1-dense-mass-step-0p35/canary.json`, SHA-256
`d7d59ff6ea84a7c31206e16a5e5db7dde8fcaaf2d069a52e91e4f03ec6427e04`:

- physical coordinates, six inverse temperatures `(1,.5,.25,.125,.0625,.03125)`;
- fixed dense mass equal to the mean of the two checked mapped local precision matrices;
- step sizes beginning at `0.35`, `L=8`;
- four chains, 24 one-row CPU/XLA workers, GPU hidden;
- fresh master seed `(20260818, 9101)`;
- 300--500 discarded warm-up transitions, no retained draws before readiness.

Warm-up diagnostics:

| Recent window | Max modern R-hat | Round trips by chain | Hot local sign changes | Gate |
|---:|---:|---|---|---|
| 300 | `1.2446365` | `[4,3,4,5]` | `[5,7,2,2]` | fail |
| 350 | `1.3201446` | `[6,4,6,6]` | `[6,7,2,2]` | fail |
| 400 | `1.4524385` | `[6,5,6,6]` | `[6,7,2,2]` | fail |
| 450 | `1.2881981` | `[7,6,7,8]` | `[6,7,3,2]` | fail |
| 500 | `1.3175029` | `[7,6,8,9]` | `[6,7,4,4]` | fail |

All five windows had finite states, valid target status, valid swap permutations,
and no invalid-path acceptance. Travel and hot forgetting therefore cannot be
used to override the R-hat veto.

## Mode-Discovery Diagnostic

Artifact: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/discovery/r1/discovery.json`

Artifact SHA-256: `693c642b9d0d1d63c290c96f7ae1fa69bf4447ca9a8498728a0edefa4c48a3d6`

The diagnostic used 16 prior-scale Gaussian starts and 16 prior-scale hypercube
corners, eight disjoint four-core CPU/XLA workers, and the exact q=20 target
signature. All 32 starts returned finite stationary endpoints. There were 31
stationary endpoints including the two known references; all optimizer endpoints
matched either the known positive or negative representative within the declared
`1e-3` infinity-norm distance. No new stationary cluster within 20 log units of
the best known endpoint was found.

This is evidence against the narrow axial-start explanation, not an exhaustive
mode-discovery proof. It does not replace global posterior sampling or establish
that no low-probability or narrow basin exists.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit batch-native target/harness | Focused contract suite and source identity | Passed | GPU training identity parity is target-specific | Preserve for a separately budgeted GPU NeuTra campaign | Posterior validity |
| Admit dense mass as a movement candidate | Fresh 100-transition canary passed all nomination screens | No hard numerical veto | Short canary cannot establish convergence | Retain as diagnostic geometry evidence | Convergence or default mass |
| Admit dense physical posterior archive | Warm-up modern R-hat `<=1.05` | **Vetoed** at 300, 350, 400, 450, 500 | Whether a different global kernel or longer/adaptive controller can converge | Design a new sampler hypothesis test; do not use these draws | Full posterior correctness |
| Treat two-known-region SMC mass as usable | Existing eight-run SMC ESS/weight/interval/cESS gates passed | Scope-limited only | Undiscovered modes and local mutation | Keep `[0.405731,0.536018]` as two-region authority | Exhaustive mass |
| Treat H3 search as closing mode discovery | No new cluster in 32 starts | Not a promotion gate | Bounded starts cannot prove absence | Use as a negative diagnostic; add broader tempered discovery only if next sampler test needs it | Exhaustive discovery |
| Launch NeuTra retraining/HMC/predictive endpoint | Requires eligible physical archive | **Blocked upstream** | No global posterior draws | Repair physical global sampler first | Any SSL-LSTM posterior-predictive verdict |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Dense physical candidate failed warm-up R-hat; no retained posterior exists. |
| Statistically supported ranking | None. Canary acceptance, runtime, occupancy, and R-hat differences are not a method ranking. |
| Descriptive-only differences | Travel counts, hot sign changes, acceptance, runtime, SMC point estimates, and bounded discovery endpoint counts. |
| Default readiness | Not established. |
| Next evidence needed | A new global sampler hypothesis that addresses between-chain parameter disagreement while preserving the successful travel/forgetting behavior; then a fresh archive and posterior-predictive endpoint. |

## Failure Classification And Hypotheses

| Question | Classification |
|---|---|
| Was the target adapter or XLA harness invalid? | No; focused tests and all material finite/status/identity gates passed. |
| Did dense mass solve the original identity-mass problem? | Partially. It solved movement/travel and hot forgetting, but not chain agreement. |
| Did the two known modes fail to cover the target? | Not shown. The bounded multistart search found no third competing stationary region, but absence is unproved. |
| Is more unchanged warm-up the justified next step? | Not from this evidence. R-hat was nonmonotone and remained above threshold at 500. |
| Is NeuTra itself rejected? | No. It was not retrained from an eligible global archive in this campaign. |

## Post-Run Red Team

The strongest alternative explanation is that the physical target has a broad,
strongly correlated between-chain direction that the fixed local mass and current
replica schedule do not decorrelate, even though sign travel is excellent. The
R-hat failure is therefore not evidence that the target is multimodal beyond the
two known regions or that the target implementation is wrong. A new diagnostic
should compare chain disagreement in the four physical coordinates and test a
global covariance/transport or longer-temperature-path hypothesis. A successful
posterior-predictive comparison would overturn the current “not ready” status;
more short canaries with unchanged dense mass would not.

## Nonclaims

No posterior archive, NeuTra convergence, HMC correctness for the full SSL-LSTM
posterior, exhaustive mode discovery, predictive equivalence, model adequacy,
sampler superiority, or default readiness is claimed.
