# SSL-LSTM q=20 NeuTra R-hat trajectory diagnostic result (2026-08-21)

Plan:
`docs/plans/bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-plan-2026-08-21.md`

Status: `TERMINAL_DIAGNOSTIC_SCREEN_FAILED_AT_4000`

## Outcome

The valid four-chain diagnostic completed with 4,000 retained draws per chain.
The answer is two-part:

- Yes, cumulative observation-weight R-hat dropped, from
  `1.0875996310350042` at 2,000 draws to `1.0489500982500948` at 4,000.
- No, doubling the sample count was not enough. The 4,000-draw value still
  failed the declared `<=1.01` screen, the cumulative sign-indicator R-hat was
  `1.0761213959625822`, and the trailing-1,000 values also failed.

The trajectory was not monotone. Cumulative observation-weight R-hat decreased
at five checkpoint steps and increased at two. The trailing-1,000 value rose as
high as `1.3671718611526078` at the 3,000-draw checkpoint before falling to
`1.0595875339465644` at the endpoint. Therefore, more draws helped
descriptively, but the evidence does not establish that sample count alone is
the problem or justify extrapolating when the same kernel would pass.

Every chain visited both observation-weight sign regions and transitioned
between them. Endpoint transition counts were `[33,49,29,46]`, and the final
trailing-1,000 counts were `[13,19,11,16]`. This rules out literal sign locking
in this run. It does not establish adequate global mixing: chain-specific sign
occupancies remained different and both the continuous coordinate and sign
indicator failed R-hat.

The terminal runner decision is
`DOUBLING_TO_4000_INSUFFICIENT_FOR_DECLARED_SCREEN`. The rejected seed-2,
`L=5` pair is not reinstated. No posterior or predictive work is authorized.

## Claimed target and computed quantity

| Item | Classification |
|---|---|
| Claimed diagnostic target | The cumulative and trailing-window R-hat trajectory, plus direct sign traversal, for the exact rejected seed-2, `L=5` fixed kernel when extended from 2,000 to 4,000 draws per chain. |
| Quantity computed | One float64 GPU/XLA TFP `sample_chain` call with four chains, 64 discarded burn-in transitions, 4,000 retained draws, step size `0.2460072308515237`, five leapfrog steps, fixed identity mass in `z`, and seed `(20260820,52000)`. |
| Relation to prior run | The 2,000-draw prefix exactly reproduced every saved prior R-hat and acceptance summary float. This supports summary-level deterministic replay, but prior raw draws do not exist, so raw-prefix identity remains unproved. |
| Verdict | Correct for the predeclared diagnostic question. It is not canonical sequential-HMC convergence or posterior evidence. |
| Not checked | ESS, posterior summaries, mode weights, predictive behavior, another kernel, another seed, or eventual crossing of `1.01`. |

## Checkpoint evidence

`Recent` denotes the trailing 1,000 retained draws. No recent window was
defined at 500 draws.

| Draws/chain | Cumulative observation-weight R-hat | Recent observation-weight R-hat | Cumulative sign R-hat | Recent sign R-hat | Cumulative transitions by chain |
|---:|---:|---:|---:|---:|---|
| 500 | 1.4239316692 | N/A | 1.7727801670 | N/A | `[2,4,1,3]` |
| 1,000 | 1.1440416764 | 1.1440416764 | 1.2424730596 | 1.2424730596 | `[8,12,4,9]` |
| 1,500 | 1.1425261562 | 1.1512618546 | 1.2228706402 | 1.2308769392 | `[11,16,11,15]` |
| 2,000 | 1.0875996310 | 1.1621899306 | 1.1455650321 | 1.2354649211 | `[16,25,13,20]` |
| 2,500 | 1.0990769397 | 1.2033488627 | 1.1814393618 | 1.4293311920 | `[17,28,15,24]` |
| 3,000 | 1.0706800918 | 1.3671718612 | 1.1088533006 | 1.7943442247 | `[20,30,18,30]` |
| 3,500 | 1.1020253949 | 1.1262050384 | 1.1639657371 | 1.2264064537 | `[27,40,24,39]` |
| 4,000 | 1.0489500983 | 1.0595875339 | 1.0761213960 | 1.0966546642 | `[33,49,29,46]` |

At 4,000 draws, the all-parameter physical R-hat vector was
`[1.0023309722,1.0031328422,1.0489500983,1.0046522740]`. Only
`observation_weight.0.0` failed among the four continuous parameters. Its rank
component was `1.0415278612` and folded component was `1.0489500983`.

Endpoint sign-region counts by chain were
`[[1955,2045],[1391,2609],[1372,2628],[1968,2032]]`. In the final trailing
1,000 draws they were
`[[458,542],[433,567],[392,608],[497,503]]`. This is direct evidence of
crossing with persistent between-chain occupancy differences, not four
sign-locked chains.

## Decision table

| Decision | Primary criterion | Veto/diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Was 2,000 simply too short? | Descriptively supported but not sufficient as a complete explanation | R-hat dropped by `-0.0386495328`, but the 4,000 endpoint and recent window still failed | Whether a much longer run of this same slow kernel would eventually pass | Do not extend blindly; use a new reviewed geometry/kernel repair question | Sample count is the sole cause |
| Did R-hat drop? | Yes, endpoint versus the exact 2,000 prefix | Five adjacent decreases and two increases; recent windows were volatile | Long-run trend beyond 4,000 is unobserved | Preserve the raw trajectory; make no extrapolation | Monotone convergence |
| Were chains sign-locked? | No in this run | Every chain visited both signs and transitioned, including in the last 1,000 draws | Transition frequency and occupancy may still be too autocorrelated | Investigate slow-coordinate geometry/dwell behavior before a new candidate protocol | Adequate global mixing |
| Reinstate seed 2, `L=5` | Failed | Physical R-hat `1.04895` and sign R-hat `1.07612` exceed `1.01` | ESS and canonical sequential behavior remain unobserved | Keep the pair rejected | Posterior admission |
| Run predictive validation | Not eligible | `posterior_admitted=false`, `predictive_authorized=false` | Predictive behavior is unobserved | Keep predictive work closed | Model adequacy or predictive equality |

## Inference status

| Status row | Finding |
|---|---|
| Run-validity hard vetoes | None. Samples, target values, scores, log acceptance, target status, movement, memory policy, raw receipts, and terminal artifact graph passed. |
| Diagnostic endpoint | Failed at 4,000 draws because both physical and sign-indicator R-hat exceeded `1.01`. |
| Viable candidates | None admitted. This diagnostic does not reverse the prior rejection. |
| Statistically supported ranking | None; only one fixed pair and one deterministic extension were examined. |
| Descriptive-only differences | Checkpoint R-hat, acceptance, occupancy, transitions, and runtime are descriptive for this one chain realization. |
| Default readiness | Not established. The route is diagnostic-only and no posterior archive was admitted. |
| Next evidence needed | A new target-specific kernel/geometry repair with disjoint tuning and validation, followed by canonical sequential HMC if a candidate passes verification. |

## Evidence ledgers

### Engineering correctness

The retry validated the complete immutable `r1` training and `r2` HMC graphs,
the frozen transport and target identities, the exact transformed-gradient
adapter, fixed identity-`z` mass, and the preserved pre-TensorFlow launcher
failure. Route enforcement passed with 42 discovered and 42 classified routes;
this runner remained `smoke_mechanics_or_reference`.

TensorFlow memory growth was configured and verified before logical-device
initialization. One logical GPU was visible, XLA compiled, float64 was used,
and TF32 was disabled. All four chains moved. The run had no finite/status or
archive hard veto.

The terminal inventory contains 23 entries and independently rehashes with zero
mismatches. All 16 raw TensorFlow tensor receipts independently parse with the
declared dtype and shape. Raw latent samples, physical samples, sign labels,
acceptance/value/score traces, and target-status traces are preserved.

### Sampler validity

Mean acceptance probability was `0.7205669859276427`; this is explanatory only.
The TFP kernel did not expose native divergence telemetry, so unavailable is
not zero divergences. The finite log-acceptance energy proxy again reached
`1e100`; the runner correctly retained it as an explanatory alert rather than
overriding the R-hat verdict.

Direct sign coverage passed, but both relevant R-hat screens failed. The
computed quantity is therefore a valid failed diagnostic, not a converged
posterior. ESS was deliberately not computed under this plan.

### Scientific interpretation

The result weakens the narrow hypothesis that the prior failure was merely an
early stop at 2,000 draws. More draws reduced cumulative R-hat substantially,
so sample length matters, but the volatile recent-window series and persistent
endpoint failure show that the same kernel remains slow on the observation-
weight coordinate. The strongest supported classification is: doubling to
4,000 was insufficient for this fixed pair.

This does not reject the target, frozen transport family, NeuTra, or the
research direction. It also does not prove that a changed mass, path length,
transport, or tuning protocol will repair the problem.

## Run manifest

| Field | Value |
|---|---|
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 timeout 36000s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_neutra_rhat_trajectory_diagnostic_2026_08_21.py --device 1 --time-cap-seconds 35820 --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01` |
| Environment | `tfgpu`, Python 3.13.13, Git commit `5699dafec23de9549a8092bec638997e7973593c`, dirty worktree preserved |
| GPU | Requested physical selector 1; one logical RTX 4080 SUPER visible; trust basis `owner_designated_managed_session_visible_gpu_trusted` |
| Numerical mode | TensorFlow/TFP, float64, XLA JIT enabled, TF32 disabled, memory growth verified |
| Seed | `(20260820,52000)`; transport seed 2 |
| HMC call wall | `25797.081125429017 s` |
| Retry process wall | `25813.99267909798 s` |
| Aggregate campaign process wall | `49094.647472506986 s`, leaving `15705.352527493014 s` of the 18-hour grant; unused time is not retry authorization |
| Completion | `2026-08-21T01:06:05.498670+00:00` |
| Launch-bound plan SHA-256 | `b1d72d761cf17b58b334541bcc181429b5754b9c83ec60d8a082fd02391cb83b` |
| Runner SHA-256 | `b4d83c0c49a215e4d031c1cd60016ca363b20a69a13dec20bddbb6b26c746c7a` |
| Result SHA-256 | `8899fa82f988ffd86e8c68076be77bbe4e2d6c988912e526ca368ccd8ab2c200` |
| Manifest SHA-256 | `84e0d1052660668a63e57d03d4b4604505e50bf7e8c9dcdd93412063b9d521cd` |
| Checkpoint SHA-256 | `0104959b2f54bde9269c923a96b7cf9c94b471f249b413deee655d8582a96418` |
| Raw archive SHA-256 | `a74fa457a417fdae1780b1202ab9f62800123f25c63f652257b1871b25eba845` |
| Inventory SHA-256 | `038d7d59095d69e4d6898df2d123b0bc410cc70655ae70a48044cd8ea2e3ad33` |

The first launcher invocation is separately preserved under
`r3-rhat-trajectory`. It failed in `0.050816870003473014 s` before TensorFlow
or GPU initialization because repository root was not yet on `sys.path`.
That artifact is harness-failure evidence only. Its old
`aggregate_gpu_wall_seconds` label is wrong for actual GPU-use accounting; the
terminal retry correctly counts that elapsed time only in aggregate campaign
process wall.

## Verification

- The GPU process exited normally and left no diagnostic process running.
- The exact 2,000-draw prefix reproduced prior saved R-hat and acceptance
  summaries with zero float residual.
- The 23-entry terminal inventory has zero SHA-256 mismatches.
- The 16 raw receipts have zero hash, byte-count, dtype, or shape mismatches.
- CPU-hidden prelaunch and post-run focused suites each reported `57 passed`;
  the 188 warnings in each run were TensorFlow Probability/Gast deprecations.
- No posterior or predictive artifact was created.

## Post-run red team

The strongest alternative explanation is not pure sample shortage but a slow
fixed kernel whose observation-weight sign occupancy is highly autocorrelated.
The observed transitions do not refute that: crossing at least once is a very
weak coverage condition, while recent-window R-hat remained `1.0596` and sign
R-hat `1.0967`.

Evidence that would overturn the scoped conclusion is a new, independently
reviewed run of this exact pair that passes a predeclared endpoint under a
scientifically justified larger horizon. This result does not justify that run
by itself, because no endpoint horizon or extrapolation rule was established
beyond 4,000 and the recent trajectory was volatile.

The weakest part of the evidence is breadth: one seed, one transport, one
kernel, and no ESS. The strongest part is identity and trajectory: the 2,000
prefix tied out exactly at the saved-summary level, raw 4,000-draw chains are
now preserved, and cumulative, recent-window, and direct sign diagnostics all
point to the same terminal non-admission.

The next justified research action is a new plan for target-specific
observation-weight geometry/kernel repair, using the raw archive for
diagnostic-only dwell/autocorrelation localization and keeping any new tuning
data disjoint from final validation. It is not another unreviewed extension of
this fixed chain.
