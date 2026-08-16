# SSL-LSTM q=20 multimodal repair stage-1 result (2026-08-10)

## Outcome

Stage 1 succeeded as a harness and timing repair, not as a posterior repair.

The exact TensorFlow/TFP replica-exchange fixed-HMC implementation recovered both
known analytic mixtures from entirely positive-mode starts under XLA.  On the
equal-weight fixture, the retained negative fraction was `0.471875`; on the
`0.8/0.2` negative/positive mixture it was `0.824375`.  Ordinary fixed HMC using the
same cold target and local kernel produced zero negative draws.  All synthetic
traces were finite, every adjacent temperature communicated, and replica identities
completed repeated cold-hot-cold travel.

The exact SSL-LSTM seed-B transformed-target canary also executed successfully:
four inverse temperatures, two region-separated chains, four transitions, fixed HMC
with XLA, and exact TFP swap correction.  All accepted states and target/status
telemetry were finite and valid.  HMC acceptance by temperature was
`[0.875, 1.0, 0.875, 1.0]`; accepted/proposed adjacent swaps were
`[2/4, 3/4, 4/4]`.

The SSL canary observed zero cold sign transitions and zero complete replica round
trips.  Four steps were never intended to establish either.  Thus parallel tempering
is now an executable candidate, but the SSL posterior, global mixing, mode weights,
NeuTra transport, and predictive distribution remain unrepaired.

## Claimed and computed quantities

| Item | Verdict |
|---|---|
| Claimed stage-1 target | Validate an exact multimodal transition harness on known laws and establish finite mechanics/cost on the exact SSL transformed target. |
| Quantity computed | Plain-HMC and TFP replica-exchange analytic mixture traces; four-step SSL transformed-target replica exchange; accepted swap matrices; replica identity travel; HMC/swap telemetry; target status; wall time. |
| Relation | Equal to the bounded stage-1 target.  Different from a stationary SSL posterior sample or mode-weight estimate. |
| Source anchor | TFP 0.25 `ReplicaExchangeMC`; clean historical BayesFilter commit `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`; checkpoint/target/parity bindings inherited and rechecked from the root-cause reconstruction. |
| Not proved | SSL mode discovery completeness, global stationarity, relative mass, superiority, NeuTra repair, posterior correctness, or predictive validity. |

## Synthetic validation

Every replica and chain started between `3.5` and `4.5` in the positive mode of a
one-dimensional Gaussian mixture with component means `-4,+4` and scales `0.5`.
The inverse-temperature ladder was `(1,0.3,0.09,0.027)`, with step sizes
`0.25/sqrt(beta)`, four leapfrog steps, 1,000 total steps, 200 excluded warm-up
steps, and eight chains.

| Method/fixture | Negative fraction | Cold sign transitions | HMC acceptance by temperature | Swap acceptance | Round trips | Verdict |
|---|---:|---:|---|---|---:|---|
| Plain HMC, equal mixture | `0/3200` draws | 0 | N/A | N/A | N/A | Known-failure comparator remained trapped |
| Replica exchange, weights `0.5/0.5` | `0.471875` | `1048` | `[0.9813,0.9841,0.9780,0.9602]` | `[0.6459,0.6541,0.7038]` | `1557` | Passed reviewed `[0.40,0.60]` regression band |
| Replica exchange, weights `0.8/0.2` | `0.824375` | `638` | `[0.9819,0.9798,0.9773,0.9580]` | `[0.5519,0.6406,0.6878]` | `1292` | Passed reviewed `[0.68,0.90]` regression band |

These are deterministic regression streams with broad predeclared bands, not an
uncertainty-supported ranking or a general calibration study.

## SSL canary

| Field | Result |
|---|---|
| Coordinates | Failed NeuTra transformed `z` coordinates, for stage-1 mechanics continuity only |
| Inverse temperatures | `(1,0.5,0.25,0.125)` |
| Steps | `(0.1,0.141421,0.2,0.282843)`; derived as `0.1/sqrt(beta)` |
| Leapfrog steps | `3` |
| Chains | Two, initialized at the exact transformed positive and negative stationary points at every temperature |
| Transitions | `4`; no burn-in and no posterior retention claim |
| HMC acceptance by temperature | `[0.875,1.0,0.875,1.0]` |
| Adjacent accepted/proposed swaps | `[2/4,3/4,4/4]` |
| Identity hot visitors | `4` of 8 chain-identity pairs |
| Complete round trips | `0` |
| Cold sign transitions | `0` |
| Accepted-state target-status invalid count | `0` |
| Floating trace finite | Yes |
| Wall time | `503.928 s` |

The canary shows that the candidate can execute exact local and swap transitions on
this target.  Swap acceptance is explanatory only.  Zero transitions and round
trips in four steps do not reject tempering, and accepted swaps do not establish hot
basin forgetting.

## Harness repair history

The first SSL attempt stopped in `2.9 s` before target compilation or sampling.
It reconstructed the archived trainer through the current shared dirty worktree;
the live `neutra_training.py` schema no longer equals the historical checkpoint
config, so `restore_state` correctly failed closed with `trainer state config
mismatch`.

The accepted root-cause artifacts had explicitly used the clean detached worktree at
`/tmp/BayesFilter-seed-b-root-cause-historical`, commit
`9ebaecc59f792f49bf7b946342ea512e71f5b3e4`.  The runner was repaired to load the new
replica helper by exact path while resolving BayesFilter target/trainer imports from
that clean historical worktree.  No target, checkpoint, transport, kernel, ladder,
seed, hardware class, or scientific criterion changed.  Eight focused tests passed
again before the one allowed retry.  Historical executable identity remains
`historical_identity_exact=false` because the original August 7 run itself used an
unrecoverable dirty worktree; current/archive point parity remains the reviewed
compatibility authority.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Admit replica-exchange mechanics helper | Equal and unequal known mixtures passed from one-sided starts | No finite, swap-permutation, identity, transition, or analytic-band veto | Deterministic fixture streams and simple equal-scale mixtures | Retain as a diagnostic candidate, not a default | Statistical superiority or broad multimodal robustness |
| Admit exact SSL mechanics/timing canary | Exact target compiled; finite HMC and accepted swaps; status count zero | No engineering/numerical veto | Four steps cannot assess mixing | Design a physical-coordinate global campaign using measured cost | Global stationarity, transitions, or weights |
| Do not continue long sampling in failed NeuTra coordinates | Stage 1 used `z` only to isolate mechanics; `z` modes are 23.707 apart | No run-invalidating veto | Whether physical coordinates mix globally under tempering | Move the independent global authority to the original four-parameter target | That NeuTra coordinates are optimal or globally repaired |
| Keep posterior predictive diagnostic blocked | No stationary weighted posterior archive was produced | Upstream weight/mixing veto remains | Exact basin weights and completeness | Run independent weighted global method, then issue an eligible archive | Posterior-predictive equality or model validity |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Passed synthetic harness and SSL mechanics/status screens; live-worktree reconstruction failure was repaired before sampling |
| Viable candidates | TFP replica-exchange fixed HMC is viable for a physical-coordinate calibration campaign |
| Statistically supported ranking | None |
| Descriptive-only differences | All acceptance rates, swap rates, round trips, transitions, and runtimes |
| Default readiness | Not ready; helper remains diagnostic/testing code |
| Next evidence needed | Physical-coordinate two-region kernel calibration, repeated replica round trips/hot forgetting, and independent AIS/SMC weights with ESS, maximum weight, schedule sensitivity, and uncertainty |

## Engineering, sampler, and scientific ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Eight focused tests passed; synthetic and SSL XLA programs completed; all 31 synthetic and 11 SSL tensor receipt hashes verified; no existing shared inference file changed. |
| Sampler validity | Exact TFP HMC and swap corrections passed known-law tests.  SSL stage establishes finite mechanics only, not stationarity. |
| Scientific interpretation | The old NeuTra posterior remains wrong relative to a global claim.  Replica exchange is an implementable repair candidate; the posterior is not yet repaired. |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`; shared worktree dirty and recorded |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| CPU/GPU | CPU diagnostic exception, eight threads, `CUDA_VISIBLE_DEVICES=-1` before import; no GPU training |
| XLA | Enabled; successful cluster compilation in tests and both runners |
| Test command | `CUDA_VISIBLE_DEVICES=-1 .../python -m pytest -q tests/test_replica_exchange_tf.py tests/test_ssl_lstm_q20_multimodal_repair.py` |
| Test result | `8 passed in 10.33 s` after final harness repair |
| Synthetic command | `CUDA_VISIBLE_DEVICES=-1 .../python docs/benchmarks/run_ssl_lstm_q20_multimodal_repair_2026_08_10.py --mode synthetic` |
| Synthetic wall | `9.671 s` |
| SSL accepted command | `BAYESFILTER_CODE_ROOT=/tmp/BayesFilter-seed-b-root-cause-historical CUDA_VISIBLE_DEVICES=-1 .../python docs/benchmarks/run_ssl_lstm_q20_multimodal_repair_2026_08_10.py --mode ssl-canary` |
| SSL wall | `503.928 s` |
| Seeds | Plain `(20260810,1001)`; equal `(20260810,1101)`; unequal `(20260810,1201)`; SSL `(20260810,2101)` |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-multimodal-repair-2026-08-10/r1/` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-multimodal-repair-plan-2026-08-10.md` |
| Result | This file |

## Post-run red team

The strongest alternative explanation for synthetic success is that the fixtures
are one-dimensional, equal-scale, and much easier than SSL.  They validate mechanics,
not transfer.  The strongest alternative explanation for SSL swap success is that
states initialized in both regions were merely exchanged without any replica locally
forgetting its basin.  Zero cold transitions and zero round trips prevent a stronger
claim.

The result that would overturn the harness conclusion is a receipt mismatch, an
independent implementation that reveals wrong accepted-swap orientation, or failure
on a fresh known-law seed under predeclared tolerances.  The weakest scientific
evidence is SSL global behavior, because only four transitions were run.  The next
campaign must not multiply this canary in the failed `z` geometry; it should test the
original target coordinates where NeuTra has not artificially separated the modes.

