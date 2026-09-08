# SSL-LSTM q=20 NeuTra global-mixing continuation plan (2026-08-20)

Status: `TERMINAL_HMC_NO_CANDIDATE_ADMITTED_NO_PREDICTIVE`

This plan governs the fresh continuation authorized by the user's 2026-08-20
grant of 18 additional hours. The grant is interpreted as an incremental local
GPU campaign cap of `64,800 s`; it does not rewrite or replenish the terminal
2026-08-19 campaign. All prior `r1` artifacts remain immutable. New evidence is
written under
`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2`.

The plan continues the same SSL-LSTM `q=20` research question. It changes only
the compute envelope, attempt ordering, wall-time accounting, and the
pre-verification affordability gate needed to make the next run answer-bearing.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can one frozen, target-specific weighted NeuTra transport and one common exact fixed-HMC kernel forget initialization and traverse both currently known material sign regions of the SSL-LSTM `q=20` target? |
| Mechanism under test | The two already frozen dense-IAF nominees are consumed without retraining. Exact transformed-target fixed HMC is tuned and then evaluated by the canonical sequential controller. |
| Claimed mathematical target | `log pi_z(z) = log pi_theta(T_phi(z)) + log_abs_det(J_T_phi(z))`, with the explicit total score pullback and original-target status propagation already checked by the frozen training audit and rechecked by the HMC consumer. |
| Expected candidate failure | A finite kernel can fail raw-coordinate verification, fail to forget initialization, remain sign-conditional, or fail retained R-hat/ESS. This rejects only that candidate/kernel pair. |
| Promotion criterion | One frozen transport and one common kernel pass full tuning verification, mechanics validity, `bayesfilter_neutra_sequential_hmc_v1`, all parameter and sign-indicator R-hat/ESS gates, and direct per-chain retained cross-sign transitions. |
| Promotion veto | Identity/parity drift; nonfinite state, target, score, Jacobian, or log acceptance; invalid target status; failed movement; modern R-hat/ESS failure; or any retained chain lacking both signs and a transition. |
| Continuation veto | Invalid immutable inputs, GPU/XLA/memory-growth failure, artifact collision/corruption, no affordable answer-bearing path, campaign timeout, or a resource callback refusing required work. |
| Repair trigger | A genuine seed-2 `L=5` tuning, mechanics, or canonical sequential rejection triggers seed 3 at `L=3` only when its complete minimum path remains affordable. A harness defect may receive one bounded local repair/retry inside the same cap when target, method, data, hardware class, and gates remain unchanged. |
| Explanatory diagnostics | Acceptance, step size, runtime, energy summaries, occupancy, transition counts, and observed R-hat/ESS values. Their values explain or veto under declared rules; they do not rank viable candidates. |
| Must not be concluded | Exhaustive mode discovery, transport or kernel superiority, universal defaults, exact SMC posterior authority, predictive equivalence, model adequacy, scientific validity, production readiness, or default readiness. |

Before stopping, the result must distinguish a target, harness, implementation,
candidate, sampler-diagnostic, artifact, and resource failure. A candidate
failure is not a research-direction failure. A resource stop is not a sampler
failure.

## Immutable prior evidence and exact baseline

The continuation binds these prior artifacts by SHA-256 before TensorFlow target
construction:

| Artifact | Binding |
|---|---|
| Frozen training result | `r1/training-screen/result.json`, SHA-256 `886a617eb60895bc97bc6530b74ef9e2578abee64771992fb29495c471cd92c7` |
| Training manifest | `r1/training-screen/manifest.json`, SHA-256 `556e34a3ad9975c10cd5db327fbff2b0c71f82f46da4b840eb1ed11b7f6f1c76` |
| Training hash inventory | `r1/training-screen/artifact-hashes.json`, SHA-256 `01af201e2350c58dcfd85e3e2a0e5d298584b752bf5df1441f2e47b4a3c6da90` |
| Prior HMC result | `r1/hmc/result.json`, SHA-256 `714a51ef0f7179fced5e3e2972217b36e72ca017740b7173372f35f781d16402` |
| Prior HMC hash inventory | `r1/hmc/artifact-hashes.json`, SHA-256 `0a91e920c71f7067af1bd227761b6360f1e9a037a92add8699cc83dee982c41a` |
| Prior HMC runner | `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py`, SHA-256 `c7356ab715a334cecb60b41f4b55ad1bcd8a92d694dfad2b15fa7dfb519f2ba7` |

The runner must also validate every entry in both prior artifact hash graphs,
the training runner, preflight, canary, geometry, target signature, adapter
signature, nominated state hashes, and frozen transport identities. The prior
terminal result is the exact comparator, not a promotion authority:

- seed 2 `L=3` completed 2,000 retained verification draws per chain and is
  rejected because observation-weight rank R-hat `1.134678` and folded R-hat
  `1.129265` exceed `1.01`;
- seed 2 `L=5` passed a finite short screen but its long verification was
  resource-refused, so it is unadjudicated;
- seed 3, canonical sequential sampling, and posterior prediction are unrun;
- the corrected prior campaign GPU wall is `12,499.045 s`. That historical wall
  is recorded but not subtracted from the fresh incremental grant.

The two frozen nominees, both width 64, three IAF stages, learning rate `3e-4`,
remain viable and are not statistically ranked. No retraining, checkpoint
selection, audit reuse as posterior evidence, or cross-candidate pooling is
allowed.

## Evidence contract

### Engineering criterion

The new runner must fail closed unless it establishes all immutable hashes,
the four-dimensional target/adapter identity, exact value/score/status/Jacobian
parity, one visible trusted GPU, pre-import
`TF_FORCE_GPU_ALLOW_GROWTH=true`, verified TensorFlow memory growth, float64,
XLA JIT, TF32 disabled, canonical route-ledger classification, fresh output
paths, and `[draw, chain, parameter]` sample ordering. Every initialized GPU
attempt, including a failed retry, counts against the incremental cap.

### Sampler criterion

Tuning retains the reviewed target-specific configuration: initial step size
`0.05`; acceptance target `0.70`; descriptive screen band `[0.55, 0.90]`;
repair band `[0.40, 0.95]`; budgets `(32,64,128)`; 16 tuning draws; 64 screen
draws after 16 burn-in; and 2,000 raw-physical-coordinate verification draws
after 64 burn-in. Verification requires finite/status validity and maximum
rank-normalized split/folded R-hat `<=1.01` across all four parameters. The
four starts remain the two mapped known representatives and their declared
first-coordinate perturbations; they are initialization probes, not posterior
weights.

An admitted kernel then uses `bayesfilter_neutra_sequential_hmc_v1`:

- discard and archive warm-up; minimum 2,000, latest 1,000 window, maximum
  modern R-hat `<=1.05`, maximum 10,000 transitions per chain;
- accumulate and archive retained draws; minimum 2,000 and maximum 10,000 per
  chain;
- require retained maximum parameter/sign-indicator R-hat `<=1.01`, minimum
  bulk ESS `>=1000`, minimum tail ESS `>=400`, and direct evidence that every
  chain visits both known observation-weight sign regions and transitions at
  least once; and
- apply finite state/target/score/log-acceptance, target-status, movement, and
  exposed energy-error gates to every chunk.

Acceptance and mechanics sign occupancy are explanatory. Passing a short
screen cannot promote a kernel. Candidate samples are never concatenated, and
conditional chains are never pooled to manufacture posterior mode weights.

### Predictive criterion

Predictive work is forbidden without an HMC result whose terminal status is
`HMC_ADMITTED_FOR_PREDICTIVE` and whose retained archive hash is independently
verified. If admitted, first run one true-vs-true mechanics calibration at
`T=20`, `n=32`, 999 balanced-label permutations. Its invariants and replay
checks are hard harness gates; its realized p-value is explanatory only.

Then draw one admitted retained physical parameter row with replacement per
path, simulate 1,000 complete raw observation paths for each
`T in {10,20,30,50,100}`, and compare each horizon with 1,000 paths generated
at the true parameter using the repository TensorFlow whole-path biased energy
V-statistic and 9,999 balanced-label permutations. Use the plus-one Monte Carlo
p-value and report five separate decisions at `alpha=0.01`. `p<0.01` rejects
the horizon-specific equality null; `p>=0.01` means only
`NOT_DISTINGUISHED_AT_1_PERCENT`. There is no omnibus pass, equivalence claim,
or model-adequacy claim.

## Diagnostic role table

| Diagnostic | Role |
|---|---|
| Artifact identity, target identity, route policy, GPU memory policy | Hard engineering/continuation veto |
| Exact transformed value/score/status/Jacobian parity | Hard engineering/continuation veto |
| Tuning and mechanics acceptance, occupancy, runtime | Explanatory only |
| Complete 2,000-draw tuning verification R-hat | Candidate promotion criterion and veto |
| Sequential warm-up recent-window R-hat | Readiness criterion and veto |
| Retained parameter/sign R-hat, ESS, direct per-chain sign traversal | Posterior admission criterion and veto |
| Resource forecast or timeout | Continuation veto only; never candidate or numerical veto |
| True-vs-true invariant suite | Predictive harness veto; realized p-value explanatory |
| Five material energy permutation tests | Separate horizon-specific equality decisions only |

## Candidate order and affordability contract

The bounded order is:

1. seed 2 at `L=5`, the next unadjudicated kernel from `r1`;
2. seed 3 at `L=3`, only if seed 2 receives a genuine tuning, mechanics, or
   canonical sequential rejection and a complete seed-3 minimum answer path is
   still affordable.

Seed 2 `L=10/15` are not in this continuation because their verification plus
canonical minimum sequential cost cannot coexist with an informative fallback
inside 18 hours. Seed 3 `L=3` is the smallest independently frozen transport
test. This is a compute-feasibility schedule, not a descriptive or statistical
ranking. Unrun pairs remain unadjudicated.

Before the 2,000-draw verification of either pair, the runner must forecast the
entire remaining minimum answer path:

`verification + mechanics + 2,000 warm-up + 2,000 retained + closeout`.

The pre-verification forecast uses the frozen trusted canary measurement
`771.3013279580045 s` for 640 leapfrog transitions and the inherited
convenience allowance `1.25`. For seed 2 `L=5`, the relevant work is 10,320
verification, 400 mechanics, and 20,000 minimum sequential leapfrog
transitions. For seed 3 `L=3`, it is 6,192, 240, and 12,000. These forecasts
are engineering safeguards, not runtime laws or sampler diagnostics.
Including one 180-second closeout, the corresponding frozen-canary forecasts
are approximately `46,458.1 s` and `27,946.8 s`. A failed seed-2 verification
followed by the seed-3 minimum path is approximately `43,673.4 s` including
both closeout allowances, before small tuning/orchestration costs. These values
justify feasibility only; they do not guarantee completion or authorize a cap
increase.

Before loading each candidate, a stricter admission forecast also includes the
maximum three inherited tuning rounds and their screens. It covers 33,280
leapfrog transitions for seed 2 `L=5` (approximately `50,134.6 s` before the
180-second closeout) and 19,968 for seed 3 `L=3` (approximately `30,080.8 s`
before closeout). The exact pre-verification check is repeated after actual
tuning costs have been charged.

After verification, the sequential forecast switches to the same-kernel
measured call rate and is reevaluated before mechanics/sequential commitment.
The canonical controller also checks each 500-result chunk. A forecast refusal
must be recorded as `UNDER_BUDGETED_HMC` with
`resource_budget_exhaustion_not_sampler_failure`; it must not be translated by
the tuning library into nonfinite, status, movement, R-hat, or other synthetic
numerical vetoes.

Seed 3 is never used as a fallback after a resource stop. A genuine canonical
diagnostic rejection may trigger seed 3 only if its fresh candidate-admission
forecast passes; a resource refusal leaves the current candidate unadjudicated.
Stop immediately on the first admitted common kernel and proceed to predictive
validation.

## Budget and attempt ledger

| Item | Incremental cap |
|---|---:|
| Total fresh campaign | `64,800 s` (18 h) |
| Preserved predictive reserve | `3,600 s` |
| External HMC process | `61,200 s` |
| Internal HMC work | `61,020 s` |
| HMC terminal closeout reserve | `180 s` within external HMC cap |
| HMC candidates | Two exact pairs in the order above |
| Harness repair/retry | At most one, only inside the unchanged aggregate phase cap |
| Predictive calibration/material | One each, sharing the `3,600 s` reserve |

The numbers are user-granted (`64,800 s`), inherited reviewed policy
(`3,600 s`, `180 s`), or derived by subtraction. The external HMC timeout is
not added to predictive time: `61,200 + 3,600 = 64,800`. CPU-only compile/unit
checks are routine diagnostics and do not count as GPU campaign wall. A
read-only `nvidia-smi` inspection does not initialize TensorFlow and is not an
experiment. No phase borrows from the predictive reserve.

## Default and assumption audit

| Choice | Provenance/status | Justification | Misleading failure mode | Early diagnostic |
|---|---|---|---|---|
| Frozen seed-2/seed-3 transports | Passed target-specific `r1` training/audit; baseline candidates | Avoids data reuse and unsupported retuning | Stale or mutated state | Full hash graph and transport identity before target work |
| Seed 2 `L=5`, then seed 3 `L=3` | Measured continuation schedule; hypothesis, not default | Covers the next unadjudicated pair and an independent transport within budget | Order mistaken for ranking; later pair starved | Record schedule/nonclaim and require whole-path affordability |
| Tuning configuration | Inherited reviewed q=20 protocol; warm start | Same target and frozen nominees | Screen acceptance masks poor mixing | Complete raw-coordinate verification is mandatory |
| `1.25` runtime allowance | Inherited convenience choice | Conservative relative to frozen canary, bounded by hard timeout | Nonlinear cost or compilation overhead exceeds forecast | Recheck with measured same-kernel verification rate; chunk veto |
| 2,000/10,000 sequential bounds and R-hat/ESS gates | Repository canonical NeuTra policy plus reviewed q=20 endpoint | Required claim-bearing route | Minimum run underpowered or maximum unaffordable | Modern diagnostics at each declared window/chunk; resource stop is separate |
| Fresh deterministic 2026-08-20 seed domains | Convenience choice, frozen before launch | Prevents accidental replay of prior stochastic streams | Seed sensitivity remains unknown | Preserve all seeds; make no ranking or robustness claim |
| Float64, XLA, TF32 off, GPU 1 | Inherited exact-target execution setting | Matches frozen target and prior timing evidence | Device or compilation drift invalidates comparison | Launch receipt plus memory-growth and logical-GPU checks |
| 3,600 s predictive reserve | Inherited reviewed phase cap | Preserves the downstream endpoint | Endpoint may be under-budgeted | Separate timeout and terminal under-budget classification |

No inherited setting is promoted as a universal default. The run can establish
viability of one scoped candidate only.

## Pre-mortem

| How the run misleads or fails | Smallest discriminating check | Disposition |
|---|---|---|
| It passes because initialized chains remain conditionally split and pooled | Direct per-chain sign coverage/transitions plus sign-indicator diagnostics | Hard admission veto |
| It spends the budget on verification without enough time for an answer | Whole-path pre-verification affordability gate | Stop before the long call as under-budgeted |
| A resource exception appears as sampler nonfiniteness | Dedicated resource exception/status outside tuner diagnostic conversion | Harness test and artifact assertion |
| It silently loads altered training or prior evidence | Recursive hash-graph verification plus fixed top-level hashes | Hard continuation veto |
| A timeout loses the interpretation | Atomic progress receipts and terminal closeout reserve | Preserve completed evidence; classify unfinished pair unadjudicated |
| It fails due to GPU allocator or XLA mismatch | Pre-target trusted device/memory-policy receipt | Harness failure, not scientific evidence |
| It passes HMC but predictive implementation is wrong | True-vs-true invariant/replay calibration | Close material endpoint and repair once within reserve |
| Five nonsignificant p-values are called equivalence | Fixed result vocabulary and no omnibus decision | Explicit nonclaim in artifact/result note |

## Execution phases and exact commands

### Phase 1: implement and validate the continuation harness

Create a new runner; do not edit the hash-bound 2026-08-19 runner. The new
runner must add prior-HMC hash validation, full-path affordability, corrected
attempt accounting, dedicated resource-stop reporting, fresh seeds, and the
two-pair schedule. Update the persistent NeuTra route ledger so the old runner
is historical and the continuation runner is the sole active q=20 route.

Run CPU-only compilation and focused tests with GPU hidden. These checks cannot
support sampler or performance claims.

### Phase 2: trusted bounded HMC

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 61200s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py \
  --device 1 \
  --incremental-campaign-cap-seconds 64800 \
  --predictive-reserve-seconds 3600 \
  --time-cap-seconds 61020 \
  --training-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen \
  --prior-hmc-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc
```

The shell timeout and internal timer start with the HMC process. Any local
repair retry receives only the recorded unspent HMC allocation and writes a
fresh `r2-retry-NN` root; it does not reset the 18-hour ledger.

### Phase 3: conditional predictive endpoint

Implement the predictive runner only after HMC admission, so its input schema
and archive identity are derived from an actual admitted artifact. Then run:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
timeout 3600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_posterior_predictive_2026_08_20.py \
  --device 1 \
  --hmc-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc \
  --sample-count 1000 --horizons 10,20,30,50,100 \
  --alpha 0.01 --permutations 9999 \
  --time-cap-seconds 3420 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/predictive
```

The predictive process reserves 180 seconds of its own `3,600 s` shell bound
for terminal closure. If HMC does not admit, write a predictive-not-run entry
in the result note; do not create a fake predictive result artifact.

### Phase 4: terminal evidence

Write a result note and reset memo for every terminal outcome. The HMC and, if
run, predictive roots must contain a manifest, result, immutable progress
receipts, tensor archive receipts, and SHA-256 inventory. The serious run
manifest records Git commit/dirty state, exact command, conda environment,
physical/logical GPU, memory policy, dtype, XLA/TF32, target/data/transport
identities, random seeds, wall time, plan/result paths, and output paths.

The terminal result includes:

- a decision table with primary criterion, veto status, uncertainty, next
  justified action, and nonclaim;
- an inference-status table covering hard vetoes, viable candidates,
  statistically supported ranking, descriptive-only differences,
  default-readiness, and next evidence;
- separate engineering, sampler, and scientific ledgers;
- the corrected aggregate incremental wall ledger, including failed attempts;
- candidate rejection versus direction rejection; and
- a post-run red team naming the strongest alternative explanation, evidence
  that would overturn the conclusion, and the weakest evidence component.

## Skeptical pre-execution audit checklist

Before launch, serious GPU execution remained forbidden until the harness audit
changed this plan's status to `REVIEWED_READY_FOR_BOUNDED_EXECUTION`. That
prelaunch gate was satisfied and is now historical; the current terminal status
does not authorize another GPU run. Routine implementation and CPU-hidden
focused checks remain separate from claim-bearing execution.

| Required audit | Pass condition |
|---|---|
| Wrong baseline | Comparator is exactly immutable `r1`; SMC/training/mechanics proxies cannot promote |
| Proxy promotion | Only full tuning verification plus canonical sequential gates can admit HMC; predictive claims remain separate |
| Missing stop conditions | Whole-path, per-chunk, closeout, timeout, attempt, identity, and predictive reserve stops are executable and tested |
| Unfair comparison | No ranking claim; fixed pair order and reason are explicit; unrun pairs remain unadjudicated |
| Hidden assumptions/defaults | Every material number and transfer is classified above with failure mode and early diagnostic |
| Stale context | Prior terminal and training hash graphs plus current runner/plan/route identities are checked |
| Environment mismatch | Trusted one-GPU, pre-import memory growth, float64, XLA, TF32-off checks fail closed |
| Artifact cannot answer question | HMC result owns sampler admission; predictive result owns only horizon-specific endpoint decisions |
| Pass while misleading | Per-chain anti-pooling, exact parity, predictive harness calibration, and explicit nonclaims remain hard boundaries |
| Fail for wrong reason | Resource, harness, candidate, diagnostic, target, and artifact failure classes are distinct |

## Review record

Codex performed a skeptical plan audit against the immutable `r1` result, reset
memo, old HMC runner, shared fixed-transport tuner, canonical sequential
controller contract, route ledger, and repository governance. The audit found
three material defects in the first draft:

1. all three prior-HMC SHA-256 values had stale suffixes; they were recomputed
   from disk and corrected above;
2. the old runner gated only the next HMC call, allowing a verification to
   consume the budget without preserving a path through the canonical minimum;
   the whole-path pre-verification contract above repairs this; and
3. the shared tuner catches generic callback exceptions and classifies them as
   numerical hard vetoes. The harness may not launch until a narrowly tested
   opt-in exception pass-through preserves a resource refusal as resource-only
   evidence.

The audit also recomputed the `L=5` and `L=3` work forecasts, checked that
`61,200 + 3,600 = 64,800`, verified that the fallback schedule can fit after a
forecast-bounded seed-2 verification rejection, and confirmed that the
asymmetric pair order supports no comparison or ranking claim. Promotion
proxies, missing stops, stale context, environment mismatch, audit leakage,
conditional pooling, artifact ownership, negative-result classification, and
predictive overclaim were explicitly checked in the table above.

Plan-audit verdict: `PASS_FOR_HARNESS_IMPLEMENTATION_ONLY`. Before Phase 2, the
reviewer must inspect the actual continuation runner, opt-in exception behavior,
focused tests, route ledger, exact plan/runner hashes, output-root freshness,
and trusted-device launch command. Any material mismatch keeps the GPU gate
closed.

### Harness prelaunch audit

Codex then inspected the implemented continuation runner and its shared-tuner
change against every checklist row above.

| Check | Evidence and disposition |
|---|---|
| Immutable comparator | The runner recursively validates the 30-entry training graph and 9-entry prior-HMC graph plus exact top-level, old-runner, target, adapter, geometry, state, and transport hashes before TensorFlow target construction. |
| Resource classification | `passthrough_exceptions=(HMCBudgetExhausted,)` is opt-in. A focused test proves the declared exception escapes; a second proves undeclared runtime errors retain fail-closed numerical handling. A refusal writes `resource-stop.json` with no tuner artifact and no candidate rejection. |
| Answer-path stops | Tests verify the candidate maximum-work forecasts (`33,280` and `19,968` leapfrog transitions). Source inspection confirms the exact pre-verification whole-path check, post-verification same-kernel check, and per-500-result canonical chunk callback. |
| Canonical route | The persistent route audit passes with the old runner historical and the continuation runner the sole active q=20 claim-bearing route bound to `bayesfilter_neutra_sequential_hmc_v1`. |
| Anti-pooling and promotion | Existing direct per-chain sign coverage/transition tests pass. The continuation still requires parameter plus sign-indicator R-hat/ESS and never concatenates candidates or conditional chains. |
| Budget/accounting | Argument tests freeze `64,800`, `3,600`, `61,200`, `61,020`, and both input roots. Every initialized attempt is process wall in the fresh ledger; prior `12,499.045 s` is historical and explicitly not subtracted. |
| Environment | CPU checks used `CUDA_VISIBLE_DEVICES=-1`. Trusted `nvidia-smi` found two RTX 4080 SUPER devices; physical GPU 1 reports 32,760 MiB and no compute process, with only the desktop service resident. The runner applies and verifies memory growth before logical-device or target initialization. |
| Artifact freshness | The entire `r2` root is absent. The runner reserves it atomically and refuses reuse. The old runner remains SHA-256 `c7356ab715a334cecb60b41f4b55ad1bcd8a92d694dfad2b15fa7dfb519f2ba7`. |
| Static/focused validation | Compilation and `git diff --check` pass. Tuner tests: `20 passed`. Route plus q=20 tests: `34 passed`. Expanded tuner/controller/campaign/route/q=20 set: `106 passed`, with two unrelated baseline failures caused only by the absent ignored P0 registry `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/target_registry.json`. |

The optional `ruff` check could not run because the environment has no `ruff`
module. This is not a launch blocker because compilation, diff validation, and
the directly relevant behavior tests pass. The two P0-registry failures predate
this continuation and do not import or exercise its runner; the missing ignored
artifact is preserved as a known broad-suite limitation rather than recreated.

Prelaunch identities before this status amendment were: runner SHA-256
`eeef1880cb26a7649ccf76230b909518fa1ca4a3e94e3bbc35e38de654d57723`,
shared tuner SHA-256
`64cc41320add2aa397ab1374413a59b0695bf4b194c9055d85572c74206b9266`,
and route-ledger SHA-256
`8c16a5389990f2f8c2b79d9546ac5e14f3365089b0754442924e3eada8937700`.
The runner records the final plan hash at launch, avoiding a self-referential
plan-hash claim.

Harness-audit verdict before launch: `PASS_FOR_BOUNDED_GPU_EXECUTION`. This
opened only the predeclared `61,200 s` HMC process bound and conditional
`3,600 s` predictive reserve. It established no candidate, posterior,
predictive, scientific, or default-readiness result; the terminal amendment
below records what actually happened.

## Terminal execution amendment

The HMC process completed on 2026-08-20 without a timeout, resource refusal,
identity failure, or harness exception. At launch this plan had SHA-256
`8dfaaab35c4f62bdb2d92e22dd55f63fda83d9e6799ab7c277b5ca1667a8559a`.
This post-run amendment deliberately changes the working plan after launch; it
is not claimed to be the launch-bound plan hash recorded in the immutable HMC
manifest.

| Phase | Result | Decision |
|---|---|---|
| Continuation harness and prelaunch audit | Completed before launch with the frozen identities, canonical route policy, XLA, float64, TF32-off, and pre-initialization GPU memory-growth checks recorded in the HMC artifacts. | Engineering gate passed for this bounded run. |
| Seed 2, `L=5` | Full four-chain, 2,000-draw-per-chain raw-coordinate verification completed. The selected step was `0.2460072308515237`; maximum folded rank-normalized split R-hat was `1.0875996310350042`, above `1.01`. | `TUNING_NO_VIABLE_KERNEL`; genuine candidate/kernel-pair rejection. |
| Seed 3, `L=3` | Full four-chain, 2,000-draw-per-chain raw-coordinate verification completed. The selected step was `0.2627342396519737`; maximum folded rank-normalized split R-hat was `1.1020661342469682`, above `1.01`. | `TUNING_NO_VIABLE_KERNEL`; genuine candidate/kernel-pair rejection. |
| Canonical sequential HMC | Not entered because neither fixed kernel passed its required tuning verification. | No posterior bank or global-mixing claim. |
| Predictive endpoint | Not run because the HMC terminal status was not `HMC_ADMITTED_FOR_PREDICTIVE`. No predictive root was created. | Correctly withheld. |

The immutable terminal HMC result is `HMC_NO_CANDIDATE_ADMITTED` with
`NO_HMC_CANDIDATE_ADMITTED`. Its HMC GPU wall was `23,280.603976539 s`; this is
below the fresh `61,020 s` internal HMC cap. The unspent budget does not add an
unplanned pair or predictive route to this completed campaign.

The two rejections do not reject either frozen transport as a whole, the
unchanged target, or the NeuTra research direction. They show that the two
specified frozen transport/kernel pairs did not meet the declared common-kernel
verification gate. The complete terminal interpretation and future resume
boundary are recorded in
`bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-result-2026-08-20.md`
and
`bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-reset-memo-2026-08-20.md`.
