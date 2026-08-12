# SSL-LSTM q=20 physical-coordinate global repair result (2026-08-10)

## Verdict

The physical-coordinate repair produced useful but incomplete progress.

1. **Direct corrected importance sampling failed as a mode-weight authority.**
   All 1,200 exact target rows were valid, but weight concentration and proposal
   sensitivity were severe.  The eight central 100-row estimates ranged from
   `0.180` to `0.789`; their mean negative-region probability was `0.468` with
   95% t interval `[0.302,0.635]`.  One normalized weight reached `0.572`, and
   covariance-scale means were `0.571` and `0.318`.  Three of five predeclared
   gates failed.  The apparent near-half mean is unsupported and must not be used
   as a posterior mode weight.
2. **Physical-coordinate replica exchange achieved real cross-mode dynamics.**
   Unlike the failed NeuTra coordinate run, fixed HMC at hot temperatures changed
   physical sign five times before swaps, and the cold replicas changed sign twice
   after exact exchanges.  Every adjacent temperature pair communicated, all target
   states were valid and finite, and HMC acceptance was `0.667--0.958` across the
   six temperatures.
3. **The transition candidate still failed the global-travel gate.**  No initial
   replica completed a cold-hot-cold round trip in 12 transitions.  The run therefore
   nominates physical tempering for a travel repair but does not establish global
   stationarity, correct occupancy, or posterior validity.

The old NeuTra posterior remains wrong relative to the full-posterior claim.  No
posterior archive is issued, and posterior-predictive validation remains blocked.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Resolve relative mass of the two known physical sign regions and determine whether exact physical-coordinate tempering can move among them. |
| Mass quantity computed | Self-normalized exact target/proposal weights from twelve independent 100-row Gaussian-mixture proposal batches at covariance scales `0.5,1,2`. |
| Mass relation | Correct finite-sample importance calculation, but failed reliability gates; not an accepted posterior weight. |
| Transition quantity computed | Six-temperature TFP replica-exchange fixed HMC in a fixed physical affine chart, with pre-swap states, accepted swap matrices, identity travel, and physical sign paths. |
| Transition relation | Direct mechanics/global-transition evidence for this candidate; too short and without a round trip, so not stationary posterior evidence. |
| Source anchor | Exact target signature `9a86e6...7278`, adapter signature `a8be6c...166f3`, source-MAP parity residuals below `7e-9`, geometry artifact SHA-256, and serialized tensor receipts. |
| Not proved | Exhaustive mode discovery, exact mode weights, global stationarity, sampler superiority, NeuTra repair, posterior correctness, or predictive validity. |

## Weight lane

### Canary

The 100-row central proposal canary passed target mechanics:

| Diagnostic | Result |
|---|---:|
| Invalid target rows | `0/100` |
| Corrected negative-region estimate | `0.507903` |
| ESS | `40.19/100` |
| Maximum normalized weight | `0.0571` |
| Proposal component-0 rows | `49/100` |
| Exact source-MAP value residual | `3.43e-9` |
| Exact source-MAP score-norm residual | `6.56e-9` |

The first detached launch mistakenly used `CPUQuota=25%`, which limits the whole
service to one quarter of one CPU rather than 25 CPUs.  The run nevertheless
completed in `45.34 s` before intervention.  CPU quota affects timing provenance,
not deterministic target values or weights.  Its mechanics evidence remains valid;
its wall time is ineligible for capacity estimates.

### Material diagnostic

The material run used eight independent central batches and two independent batches
at each alternate covariance scale.  All 1,200 target rows were valid.

Central negative-region estimates:

`[0.6276, 0.5135, 0.5427, 0.4763, 0.3722, 0.1798, 0.7888, 0.2456]`

| Predeclared gate | Result | Status |
|---|---:|---|
| All target rows valid | `1200/1200` | Pass |
| Mean central batch ESS fraction `>=0.20` | `0.2257` | Pass, marginal |
| Maximum central batch normalized weight `<=0.05` | `0.5722` | **Fail** |
| Independent-batch interval half-width `<=0.10` | `0.1666` | **Fail** |
| Covariance-scale sensitivity difference `<=0.10` | `0.1501` | **Fail** |

The central mean was `0.4683`; its independent-batch t interval was
`[0.3017,0.6349]`.  Mean estimates at covariance scales `0.5` and `2.0` were
`0.5711` and `0.3182`.  These differences are not a ranking of proposal scales;
they show that finite direct importance sampling is unreliable.

The 1,200 target evaluations completed successfully, but terminal aggregation first
failed because the runner called `tf.constant` on a list of scalar tensors.  A
receipt-verifying recovery mode loaded the atomic 12/12 progress artifact, verified
all 84 tensor receipts and all worker target/adapter identities, recomputed zero
target rows, replaced the invalid construction with `tf.stack`, and wrote the
terminal result.  This was a terminal-reporting bug, not a scientific retry.

## Transition lane

### Fixed physical chart

The chart uses the equal-weight law of total covariance of the two measured local
Gaussian approximations.  Equal weighting is geometry design only, not a posterior
weight claim.

| Diagnostic | Result |
|---|---:|
| Source representative physical distance | `1.2809` |
| Mapped representative distance | `1.9343` |
| Positive mapped precision eigenvalues | `[0.501,1.003,15.347,366.722]` |
| Negative mapped precision eigenvalues | `[0.501,0.997,15.371,364.404]` |
| Derived local quadratic stability scale | approximately `0.104` |
| Selected cold step | `0.05` |

This chart is not isotropic, but it is symmetric across the two known modes and does
not reproduce NeuTra's 23.707-unit separation.

### Local gate

The first local launch failed before sampling because the fixed chart Jacobian used
`tf.linalg.slogdet` inside the CPU-XLA target closure; XLA has no
`LogMatrixDeterminant` kernel there.  The fixed log determinant was moved outside
the graph without changing the chart or target.  The retry passed:

| Diagnostic | Positive start | Negative start |
|---|---:|---:|
| Four-transition binary acceptance | `1.0` | `0.5` |
| Invalid accepted states | `0` pooled | `0` pooled |

Wall time was `160.45 s`.  This is local integration evidence only.

### Replica exchange

Configuration: inverse temperatures
`(1,.5,.25,.125,.0625,.03125)`, steps `0.05/sqrt(beta)`, `L=8`, two
sign-separated chains, and 12 transitions.

| Diagnostic | Result | Role |
|---|---:|---|
| HMC acceptance by temperature | `[0.958,0.833,0.667,0.833,0.833,0.750]` | Numerical/local mechanics |
| Adjacent accepted/proposed swaps | `[9/12,7/12,6/12,10/12,8/12]` | Communication explanatory diagnostic |
| Local-HMC hot sign changes before swaps | `5` | Hot-basin forgetting evidence |
| Local-HMC cold sign changes | `0` | Expected local cold behavior |
| Post-swap cold sign transitions | `2` | Temperature-assisted transition evidence |
| Completed cold-hot-cold identity round trips | `0` | **Global-travel gate failed** |
| Invalid accepted target states | `0` | Hard veto passed |
| Floating trace finite | Yes | Hard veto passed |
| Wall time | `4600.29 s` | Descriptive cost |

Chain 1's cold sign path changed from negative to positive at transition 9 and back
to negative at transition 11.  Chain 0 remained positive.  Accepted swaps did not
alone create the hot sign evidence: the archived pre-swap states show five sign
changes caused by local HMC at temperatures below one.  This establishes a repair
mechanism absent from the failed NeuTra chain, but no identity completed the full
ladder traversal and return.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject direct IS as weight authority | Three of five reliability gates failed | Target identity/status and receipt gates passed | Heavy weight tails and two-mode proposal incompleteness | Run AIS; move to annealed SMC if AIS weights remain unstable | That negative mass is `0.468` or any other reported point estimate |
| Retain physical chart as a transition warm start | Two-region local gate passed | No finite/status veto | Strong residual anisotropy | Use for bounded temperature-travel repair | Optimal mass/chart or default readiness |
| Nominate physical replica exchange | Genuine hot HMC sign changes and cold transitions observed | No finite/status/swap-communication veto | No full round trip; only two chains and 12 steps | Test a travel-focused ladder/schedule after independent mass repair | Global stationarity, correct occupancy, or superiority |
| Keep posterior archive blocked | Neither mass nor round-trip gates passed | Upstream posterior authority veto remains | Mode completeness and weight stability | AIS/SMC, then longer untouched transition validation | Posterior-predictive validity |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Engineering/target/status/finite/receipt gates passed after documented harness repairs |
| Viable candidates | Physical affine replica exchange remains a viable transition candidate; direct IS is not a viable weight authority |
| Statistically supported ranking | None |
| Descriptive-only differences | All HMC/swap rates, runtimes, raw transitions, direct-IS point estimates, and scale-specific estimates |
| Default readiness | Not ready |
| Next evidence needed | Stable AIS/SMC weighted mass evidence and repeated temperature round trips under a frozen physical transition design |

## Engineering, sampler, and scientific ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Analytic helper tests passed; source-MAP parity passed; 84 weight and 13 transition receipts verified; target/status finite; documented aggregation and XLA repairs. |
| Sampler/numerical validity | Direct IS failed finite-sample reliability.  Physical HMC/replica exchange is finite and crosses signs, but global travel gate failed. |
| Scientific interpretation | The old NeuTra posterior remains invalid.  Two physical regions are dynamically connected at hot temperatures, but their relative mass is unresolved. |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`; dirty concurrent worktree recorded |
| Environment | `tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| Device | CPU-only diagnostic exception; GPU hidden before import; XLA enabled |
| Weight topology | 25 pinned CPU workers on CPUs 64--88, four batch-native rows each |
| Weight seeds | Canary `(20260810,5001)`; central `5100--5107`; scale `0.5` `5200--5201`; scale `2` `5300--5301` |
| Transition seed | `(20260810,4101)`; local seed `(20260810,4001)` |
| Weight terminal artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/weights.json` |
| Transition terminal artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/physical-transition.json` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-physical-global-repair-plan-2026-08-10.md` |
| Result | This file |

## Post-run red team

The strongest alternative explanation for the direct-IS failure is proposal mismatch,
not intrinsically difficult posterior mass.  That is why the result triggers AIS/SMC
rather than a conclusion about unequal modes.  The strongest alternative explanation
for the transition success is that two deliberately initialized signs made crossing
easy.  Pre-swap hot sign changes refute the narrower explanation that all sign changes
were exchanges only, but mode completeness and stationary frequencies remain open.

The result that would overturn the weight rejection is an independent weighted path
with stable ESS, maximum weight, schedule sensitivity, and uncertainty.  The result
that would overturn the transition non-promotion is repeated full ladder round trips
and cold-region convergence under a frozen validation run.  The weakest current
evidence is still exhaustive mode discovery: every method in this campaign began
from the two already known regions.

