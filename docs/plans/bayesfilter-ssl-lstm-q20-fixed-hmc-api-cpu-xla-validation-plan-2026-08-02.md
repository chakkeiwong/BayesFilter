# q=20 Fixed-Transport TFP HMC API CPU/XLA Validation

Date: 2026-08-02  
Tier: serious local research campaign  
Status: `COMPLETED_NO_KERNEL_ADMITTED_SEQUENTIAL_NOT_LAUNCHED`

Terminal result:
`docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-result-2026-08-02.md`.
Both chart ladders completed without hard numerical/status vetoes, but no fresh
screen entered `[0.65,0.75]`; the conditional sequential phase therefore did
not launch.

## Research Intent Ledger

| Role | Prospective contract |
| --- | --- |
| Main question | After tuning through BayesFilter's fixed-transport API, do either of the two trained q=20 NeuTra charts admit a fixed-length TFP HMC kernel that passes the repository's sequential warm-up and retained-sample screens? |
| Candidate under test | Fixed identity-mass HMC in each frozen chart's latent `z` coordinates, with a repository-owned `L>=2` grid and one scalar step size tuned jointly over a batched chain bank. |
| Exact baseline | Each frozen chart with identity mass in `z`; Chart A and Chart B are replications and are not ranked. The historical custom grid and all `L=1` artifacts are ineligible. |
| Expected failure mode | The chart may leave residual geometry that yields no in-band fixed kernel, slow mixing, or failure of R-hat/ESS within the cap. |
| Kernel promotion criterion | Fresh fixed-kernel verification has finite samples, targets, scores and log acceptance, and pooled mean Metropolis acceptance probability in `[0.65, 0.75]`. |
| Kernel promotion veto | `L<2`; nonfinite state/target/score/log acceptance; invalid target-status telemetry; reported positive native divergence when the installed kernel exposes it; or acceptance outside `[0.65, 0.75]`. |
| Sequential continuation/admission | Warm-up readiness uses the latest 1,000 draws after at least 2,000 discarded transitions per chain with maximum rank/folded R-hat `<=1.05`. Retained admission after at least 1,000 draws per chain requires maximum rank/folded R-hat `<=1.01` and minimum bulk/tail ESS `>=400`. |
| Repair trigger | No admitted kernel or failure at a sequential cap triggers target-specific kernel/transport repair. It does not reject the target, harness, or NeuTra direction. |
| Explanatory diagnostics | Finite `max(abs(log_accept_ratio))`/`abs(Delta H)` tails, movement magnitude, runtime, RSS, and descriptive differences between charts. |
| What must not be concluded | A tuning pass alone does not establish convergence, posterior correctness, chart superiority, model adequacy, GPU equivalence, default readiness, or scientific validity. Native-divergence unavailability is not zero divergences. |

## Exact API And Semantics

The tuning entry point is
`bayesfilter.inference.fixed_transport_hmc_tuning.tune_fixed_transport_hmc_kernel`.
The launcher may construct the target and transport and dispatch exact API
requests, but it may not select its own `(L, epsilon)` grid.

For each `L`, TFP `DualAveragingStepSizeAdaptation` receives a rank-2 chain bank
and a scalar step size. In installed TFP 0.25.0, the scalar shape causes the
kernel to reduce log acceptance across the chain dimension and update one
shared step. Independent one-chain adaptations followed by aggregation are not
equivalent and are forbidden. Chart A and Chart B may tune concurrently in two
processes; chains within a tuning call remain one batched process.

The fixed initial-state policy is an all-zero `z` bank of shape
`[chain_count, parameter_dim]`, issued by the API for every tuning and fresh
verification call. A callback must receive that exact tensor and may not
substitute hand-picked states. This is an API baseline, not evidence that zero
initialization is universally optimal.

## Diagnostic Roles

| Diagnostic | Role |
| --- | --- |
| Acceptance `[0.65,0.75]` | Kernel tuning and fresh-verification promotion criterion; outside the bound vetoes that candidate. |
| Nonfinite state, target, score, or log acceptance | Hard veto. |
| Native divergence | Hard veto only when the kernel reports an available positive native boolean/count. Unavailable is recorded and is not a veto or a zero count. |
| Target-status invalidity | Hard veto. |
| Finite `abs(log_accept_ratio)` / `abs(Delta H)` tail | Explanatory alert only; no finite magnitude threshold veto. |
| Chain movement | Explanatory diagnostic; an exactly unmoved bank should be investigated but is not silently converted into a divergence or acceptance gate. |
| Warm-up R-hat | Sequential readiness gate, not kernel-acceptance evidence. |
| Retained R-hat and bulk/tail ESS | Sequential finite-sample admission gates, not posterior truth. |

## Default And Assumption Audit

| Choice | Provenance and status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| Fixed-transport API | Existing domain-specific public API; reviewed route | A trained chart already defines the coordinates, so generic model-space mass adaptation answers a different question | Assert API runtime/schema and forbid launcher-owned grids |
| Identity `z` mass | Existing API contract; baseline | Directly tests whether the trained chart supplies usable geometry | No viable kernel or poor sequential diagnostics triggers metric/transport repair |
| `L=(2,)` for this campaign | Target-specific budget repair derived from the 2026-08-02 batched and distributed canaries; hypothesis, not a new API default | It is the smallest allowed fixed-HMC trajectory and the only measured arm whose minimum sequential projection fits the authorized campaign cap | It may miss a viable longer trajectory or mix poorly; no admitted kernel or failed sequential gates trigger a future target-specific trajectory repair, not relaxation to `L=1` |
| `L>=2` | User policy, 2026-08-02; hard boundary | `L=1` is outside the requested HMC trajectory test | Config and frozen-kernel construction tests reject it |
| Target accept `0.70`, band `[0.65,0.75]` | Existing API defaults; reviewed starting policy | Centers the TFP HMC tuning screen around the API target | A narrow noisy screen may reject viable kernels; fresh seeds and later sequential diagnostics separate tuning failure from target failure |
| Four batched chains during API tuning | Existing modern verification minimum; reviewed baseline | Preserves shared scalar dual averaging and modern multi-chain verification | Four short chains give uncertain acceptance; candidate differences remain descriptive |
| All-zero initial `z` bank | Existing API factory; baseline | Deterministic and coordinate-neutral | Initial transients may be atypical; discarded adaptation and a future initial-state sensitivity test are possible repairs |
| CPU/XLA | User-selected execution lane; explicit exception to GPU default | Prior CPU multiprocessing canaries showed this target is faster in the selected CPU lane | Batched chart-level adaptation may scale differently; compile and representative chunk canaries measure it |
| TensorFlow/TFP runtime | Repository backend policy; required | Claim-bearing tuning and admission may not use NumPy numerical decisions | Import-closure test and focused parity tests must pass before real artifacts |
| Sequential minima/caps | `bayesfilter_neutra_sequential_hmc_v1`; reviewed default | Supplies modern warm-up and retained finite-sample screens | Authorized wall budget may be insufficient; representative canary stops the campaign as under-budgeted |

## Evidence Contract

- Engineering question: does the TensorFlow-only public API preserve exact
  batched target calls, shared scalar dual averaging, fresh fixed-kernel
  verification, XLA, deterministic seeds, and fixed initial states?
- Sampler question: after a kernel passes the acceptance/finite/native-
  divergence screen, do full sequential R-hat/ESS gates pass within the cap?
- Exact comparator: separate identity-mass fixed HMC for Chart A and Chart B;
  no cross-chart rank claim.
- Hard veto evidence: invalid checkpoint or transport hash, wrong target, `L<2`,
  non-XLA real execution, runtime NumPy dependency in the exercised path,
  nonfinite required telemetry, invalid target status, available positive native
  divergence, candidate acceptance outside the bound, archive corruption, or
  campaign cap.
- Explanatory only: energy/log-accept tail magnitude, movement magnitude,
  runtime/RSS, and continuous differences between viable charts.
- Result artifact root:
  `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r1/`.

## Compute And Attempt Budget

- User-authorized cumulative wall budget: `20,000 s` for this campaign.
- Conservative prior-canary charge: `1,900 s`. The earlier `r3` summary's
  `19,840.96 s` field subtracted only its own `159.04 s` and is not cumulative.
  Pre-tuning remaining budget is therefore at most `18,100 s`.
- Concurrent tuning uses CPUs `0..15` for Chart A, `16..31` for Chart B, and
  CPU `32` for supervision. Sequential sampling uses four persistent one-core
  workers per admitted chart on CPUs `0..7`; supervisors may share CPU `32`.
  The one-core topology is measured provenance from `r3`, not a universal CPU
  default. Workers must not interfere with unrelated processes.
- Attempt budget: one implementation test pass, one tiny API/XLA canary per
  changed static graph shape, one representative fixed-kernel chunk canary per
  chart, one API tuning attempt per chart, and no more than one localized retry
  for an unchanged harness failure.
- The representative canary must estimate the remaining cost of API tuning plus
  the minimum sequential workload (`2,000` discarded warm-up and `1,000`
  retained transitions per chain). If the upper planning estimate cannot fit
  the remaining campaign budget, stop as `UNDER_BUDGETED`; do not reduce the
  sequential minima or label a short chain valid.
- Campaign cap exhaustion preserves partial artifacts but is not scientific or
  sampler evidence.

## Pre-Mortem

| Misleading outcome | Cheap discriminator |
| --- | --- |
| Command succeeds but callback independently adapts one step per chain | Test the exact initial-state shape, scalar step trace shape, and TFP reduction semantics before target canaries |
| XLA flag is recorded but the executed graph is eager/non-XLA | Record concrete-function compiler IR availability and compiled first/warm call timing |
| Tuning pass is mistaken for posterior validity | Keep kernel promotion and sequential admission in separate ledgers/artifacts |
| Unavailable divergence is silently counted as zero | Require status `not_exposed_by_kernel`, count `null`, and an explicit nonclaim |
| A finite large energy tail is promoted to a veto | Test that a `1001` finite tail raises only an explanatory alert |
| Zero initialization creates a false failure | Classify failure as initial-state/kernel repair evidence, not target invalidity |
| Short timing extrapolation misses compile or diagnostic cost | Run one representative static-shape chunk and include compile, warm call, mapping, diagnostics, and archive overhead |
| Both charts compete for cores and invalidate timing | Record affinity and run the canary on the intended chart-level topology |

## Skeptical Pre-Execution Audit

1. Wrong baseline: the historical launcher-owned grid is excluded. The public
   fixed-transport API owns `L`, step tuning, selection, and fresh verification.
2. Proxy promotion: finite energy/log-accept tails and movement are explanatory;
   they cannot masquerade as native divergence or candidate acceptance.
3. Missing stops: the campaign has a cumulative `20,000 s` cap, bounded canary
   and retry counts, candidate vetoes, sequential caps, and an under-budgeted
   terminal state.
4. Unfair comparison: Chart A/B tune separately with identity mass and declared
   seeds. No stochastic ranking is planned.
5. Hidden semantics: TFP shared scalar adaptation is preserved within one
   batched process. Independent worker step-size aggregation is forbidden.
6. Stale context: historical `L>=2` arms nominate plausible behavior only;
   historical `L=1` artifacts and old custom confirmations are ineligible.
7. Environment mismatch: real runs explicitly hide CUDA, record CPU affinity,
   use FP64 TensorFlow/TFP, and require XLA. CPU is an owner-selected exception,
   not the repository production default.
8. Backend mismatch: the exercised public tuner and sequential controller must
   have a TensorFlow/Python-standard-library dependency closure. Legacy NumPy
   modules may not be imported for claim-bearing tuning or admission decisions.
9. Artifact adequacy: tuning config/result, frozen kernel, canary timing, every
   warm-up shard, retained shards, diagnostic checkpoints, hashes, command,
   environment, affinity, seeds, wall time, and git commit/status are preserved.
10. Cost validity: no full run starts until representative measured timing says
    the minimum valid sequential workload fits the remaining budget.

Post-canary audit verdict: `PASS_FOR_TARGET_TUNING_AND_CONDITIONAL_SEQUENTIAL_RUN`.
The focused collection found 32 tests, the prior focused execution passed 29,
the public tuning import closure is TensorFlow/Python-standard-library only,
and the distributed canary passed every engineering/finite-status check. The
real sequential run remains conditional on an admitted kernel and a tested
exact process callback.

## Canary Decision Ledger

| Evidence | Result | Decision role |
| --- | --- | --- |
| Four-chain batched `L=2` | Chart A minimum projection `54,464.7 s`; Chart B `53,043.8 s` | Reject batched sequential topology as under-budgeted; not a kernel verdict |
| Eight independent one-chain CPU/XLA workers, 32 draws | Slowest minimum projection `6,318.0 s`; 20% reserve `7,581.6 s`; every required tensor finite | Admit distributed frozen-chain topology for a full sequential attempt |
| Native divergence | TFP HMC does not expose it | Record unavailable/null; do not claim zero divergences |
| Timing uncertainty | One short canary per worker, no interval | Projection only; no performance ranking |

## Execution Sequence

1. Repair the public tuner and sequential controller diagnostic classifications.
2. Remove runtime NumPy from the exact public fixed-transport tuning path and
   add dependency-closure and semantic tests.
3. Use a chart-level CPU/XLA callback that executes each exact tuning API config
   as one batched chain bank with a shared scalar dual-averaging step.
4. Run focused unit and Gaussian XLA canaries, including exact config/initial-
   state dispatch and unavailable/positive divergence cases.
5. Preserve the completed representative target canaries and charge their
   conservative `1,900 s` cumulative cost before the real run.
6. Tune Chart A and Chart B concurrently with target-specific `L=(2,)`, initial
   step `0.5`, dual-averaging budgets `(8,16,32)`, 16-draw/4-burn-in screens,
   and a fresh 64-draw/16-burn-in verification.
7. For each admitted chart, invoke the repository sequential controller through
   four persistent one-chain CPU/XLA workers. Each exact four-chain state is
   sharded without modification, each controller seed is statelessly folded by
   chain, and sample/trace tensors are reassembled on the chain axis before any
   gate or archive action. Use a fixed 40-draw chunk: it divides all sequential
   minima/windows/caps and is the nearest divisible size to the measured
   32-draw canary. Do not silently use the expensive 500-draw default.
8. Write a result note, serious-run manifest, inference-status table, and
   post-run red-team note. Do not rank the charts without uncertainty evidence.

## Planned Commands

Focused checks:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_neutra_sequential_hmc.py
```

Target tuning command:

```bash
taskset -c 32 timeout 3600s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py \
  --mode tuning-supervisor \
  --output-root docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning \
  --cap-seconds 3600
```

The conditional sequential command and its cap will be written after tuning,
using a new output root and the actually remaining cumulative budget.
