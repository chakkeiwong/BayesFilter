# SSL-LSTM q=20 Tempered Reverse-KL Transport Ensemble Master Program

## Governing update: executable master repaired, 2026-09-16

Status: `REPAIRED_TESTED_READY_FOR_BOUNDED_EXECUTION`.
The executable master now connects qualification, pricing, independent reference,
separate direct/continuation training, current public tuning, all five posterior
methods, development comparison and fresh-start confirmation. A complete
known-target smoke passed 25 supervised worker stages and a zero-worker resume.
See the [repair plan](bayesfilter-ssl-lstm-q20-executable-master-repair-plan-2026-09-16.md)
and [whole-program review and execution record](bayesfilter-ssl-lstm-q20-executable-master-review-result-2026-09-16.md).

Run from the isolated committed checkout
`/tmp/BayesFilter-q20-master-repair-20260916` because main contains unrelated
active work. The source-bound numerical state must stay in that checkout.
The current CLI is `docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py`;
`campaign` executes or resumes the same output root, and `status` reads its
ledger. New roots require the recovered allowance record. The next real run
is `docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/real-q20-r1/`.
The final execution status will be recorded in the linked review result.

The current allowance preserves the old 7200-second hold, subtracts 1306.97
reported repair-test seconds and holds another 115 seconds for unmeasured
startup/exit overhead. Before the real launch, available campaign time is
126653.86289109988 seconds, including 45677.40991617 diagnostic seconds. No
previous budget is renewed. GPU availability must be checked under trusted
execution; the earlier no-idle-GPU result was not a security rejection.

The latest tests establish connected execution, not q20 learning, whitening,
posterior accuracy, method superiority or broad production readiness. The
master preserves those evidence requirements and reports incomplete work.
Candidate-specific contrasts remain repairs to diagnose and price under the
existing campaign, not grounds to abandon the research direction.

## Historical recovery audit — superseded by the governing update above

All current/next-step statements below belong to their recorded older snapshot.
They preserve the evidence that motivated this repair and must not select an
older runner or override the governing update.

Updated: 2026-09-16 (Asia/Shanghai)  
Status: `PRODUCTION_REPAIR_INCOMPLETE_INTEGRATION_AND_LAUNCH_GAPS`  
Code baseline: `main`, `965ba2949244ee2a1d911515b6865c268a4af8d2`

## Current governing program

This section governs the next work. The September 15 production repair is
merged into local `main`; it is one commit ahead of `origin/main` at this
audit. The checkout also contains concurrent uncommitted work. The earlier
Phase 9B/P0/P1 instructions below are historical and must not select the next
runner, training schedule, tuning artifact or budget. Prior results stay intact.
The owner's existing repair and campaign authorization remains applicable;
this update adds no compute allocation or approval ceremony.

The master **plan** exists, but the complete executable master program does
not yet exist. The current
[entrypoint](../benchmarks/run_ssl_lstm_q20_production_2026_09_15.py) exposes
only `validate`, `price` and `train`. It does not dispatch tuning, explicit
member selection, posterior sampling, reference comparison or confirmation.
“Ready once a GPU is free” was therefore unsupported. The current work is
completion of the engineering repair, followed by measured development.

Use the [repair plan](bayesfilter-ssl-lstm-q20-production-repair-plan-2026-09-15.md),
[corrected repair result](bayesfilter-ssl-lstm-q20-production-repair-result-2026-09-16.md),
[pipeline design](bayesfilter-ssl-lstm-q20-production-pipeline-design-2026-09-15.md),
[parameter ledger](bayesfilter-ssl-lstm-q20-production-parameter-ledger-2026-09-15.md)
and [mathematical audit](bayesfilter-ssl-lstm-q20-parameter-mathematical-audit-2026-09-15.md)
as the active supporting documents. The
[reset memo](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-06.md)
provides the short recovery order. The design and numerical ledger describe
requirements and hypotheses; they do not establish that the code meets them.

### Implementation and evidence at this audit

| Part | Actual state | Remaining requirement |
| --- | --- | --- |
| Explicit configuration, batched training, complete checkpoints and map export | Implemented. Twelve focused tests pass, including a tiny Gaussian cohort resumed through both positive temperatures. | Real q20 GPU/XLA training, measured learning and downstream qualification. Schedule completion is not map qualification. |
| Current public tuner and posterior consumer | Callable adapters exist in `bayesfilter/inference/q20_production_hmc.py`; they are not dispatched by the CLI or covered by a connected tuning-to-posterior test. | Repair the concrete interface defects below and exercise the actual supported public binding, verified member and shared posterior controller together. |
| Ensemble and comparison | An ensemble adapter exists. `q20_production_comparison.py` only reports an unavailable reference and unestimated ranking. | Connected ensemble tests, matched baseline/start wiring, independent reference computation, uncertainty-aware comparisons and confirmation. |
| Pricing and supervision | `price_training` measures training updates only; `training_quote` explicitly says `full_campaign_priced: false`. | Include initialization/compilation, validation, all tuning/sampling/reference work, failures and repair; enforce external time limits and persistent aggregate accounting. |
| GPU evidence | No repaired q20 GPU training or HMC run has completed. The last saved probe returned `no_idle_policy_permitted_gpu` at 2026-09-15 16:48:21 UTC. | A fresh trusted availability check immediately before a bounded diagnostic. The old probe is not present availability or a security-review rejection. |

Two concrete defects strengthen the integration blocker. `sample_member`
passes `posterior.max_energy_error=+1000` into
`SequentialNeuTraHMCConfig.energy_error_log_accept_threshold`, whose constructor
requires a finite **negative** value. The ensemble kernel also converts
`abs(log_accept)>1000` into a nonfinite state. That silently promotes a finite
energy diagnostic into a veto, contrary to the active parameter ledger and
shared controller. Repair both consumers against the signed, explanatory-only
energy contract; preserve actual nonfinite, target-status and native-divergence
vetoes where available. A successful constructor alone will not close the
connected sampling requirement.

A [configuration-only reproduction](artifacts/ssl-lstm-q20-master-recovery-2026-09-16/audit-r1/sampling-constructor-check.json)
confirmed the negative-threshold exception with GPU devices deliberately hidden
and zero target evaluations or sampler transitions. The ensemble finding is
from source inspection; a connected runtime check remains required.

### Ordered next work and completion conditions

1. **Finish the connected program.** Read the current
   [HMC interface](../reference/hmc-tuning-interface.md) and
   `HMC_TUNING_INTERFACE_CAPABILITIES` in `bayesfilter/inference/tuning_contract.py`.
   Fix the energy-contract defects, connect trained-map export to supported
   public tuning, explicit verified-member selection and sequential posterior
   assessment, and connect the actual chart/temperature ensemble. Exercise
   these paths with bounded known-target fixtures, including missing precision,
   invalid targets, smoke promotion and resume failures. Record any inherited
   consumer/preparation controls still absent from the numerical ledger.
   Implement honest CLI stage/status dispatch; unavailable stages must report
   incomplete before expensive work. Retain identity HMC, tuned classical HMC,
   plain NeuTra and ensemble comparisons; the design also requires matched
   physical replica exchange for the ensemble claim, which the four-name
   configuration inventory alone does not implement.
2. **Finish launch and budget enforcement.** Use an isolated source snapshot
   based on the repair commit plus inventoried necessary fixes. The present
   `source_snapshot()` hashes every Python file under `bayesfilter`, including
   unrelated untracked modules; running from the changing main checkout risks
   another source-drift failure. Preserve the concurrent main work. Add a
   supervisor that bounds initialization, compilation and native calls, records
   interrupted attempts and settles worker time across retries. The current
   `price` ignores `--max-seconds`; `train` checks it cooperatively and cannot
   bound all native work. Reconcile the balance below, then require a measured
   complete-cost reservation before a development training cohort.
3. **Measure the repaired q20 route.** After the focused connected checks pass,
   use trusted GPU access, verified memory growth and TensorFlow/TFP XLA on the
   declared float64 strict target. Start with the existing repair allocation:
   at most four q20 GPU diagnostic attempts, each externally bounded to 600
   seconds, within the initial 7,200 aggregate-worker-second repair allocation.
   Charge initialization, compile, validation and failed attempts. These are
   inherited operational ceilings, not measured completion forecasts. Repeated
   failure to compile requires localization, not another unchanged full run.
4. **Make the full experiment affordable and testable.** Implement and price
   the independent reference and named mean/quantile/event comparisons from the
   design before treating posterior evidence as qualified. Obtain current
   costs for the required tuning scopes, classical preparation, posterior
   systems, confirmation and one specified repair, as well as training and
   validation. A missing cost is unresolved, not zero. If the complete promised
   experiment cannot fit, report under-budgeted and present the concrete scope
   or allocation decision; do not silently weaken comparisons or precision.
5. **Execute the funded learning protocol and development comparison.** The
   current training hypothesis is twelve histories (two widths, two learning
   rates, three roots), batch 32, analytic beta-zero initialization, then
   cumulative 128/512/2,048/8,192-update rungs at positive beta. Use the existing
   assessed-learning rules and preserve unqualified/capped outcomes. Freeze
   admitted maps, tune each exact used scope, retain all verified members, and
   record an explicit operational choice without a descriptive ESS ranking.
   Run four independent posterior chains/systems with the documented warmup,
   cumulative precision, reference and start-region checks.
6. **Confirm the frozen procedure.** Use fresh confirmation streams and the
   same declared posterior accuracy and comparison contract. Count all cost,
   retain failed evidence, and distinguish qualified estimates from a supported
   efficiency ranking. A failed candidate triggers the specified repair within
   the remaining budget; it does not reject the research direction.

### Research intent and evidence contract

The question remains whether properly trained NeuTra and the charted tempered
ensemble can estimate the fixed q20, T30, four-parameter UKF-approximate
posterior accurately at useful total cost. The target, observed data, prior,
float64 strict backend and TensorFlow/TFP direction are unchanged. The
comparison uses the baseline ladder in the design, matched physical starts,
and the same downstream accuracy. It cannot establish accuracy of the UKF
approximation to the original latent-model posterior.

| Role | Meaning for the next work |
| --- | --- |
| Engineering pass criterion | Connected execution and recovery of the configured stages, correct numerical identities, mandatory diagnostic delivery and enforceable cost limits. |
| Scientific promotion criterion | Required means, quantiles and sign-region probability meet the ledger's MCSE, independent reference and start-equivalence requirements in untouched confirmation. A method ranking additionally needs the declared uncertainty comparison including full cost. |
| Promotion veto | Unqualified map, failed numerical/posterior checks, missing reference or unsupported comparison. A training loss, tuned epsilon or completed schedule cannot substitute. |
| Continuation veto | Unrepaired shared target/math or artifact invalidity, missing essential inputs, actual permission boundary, or exhausted authorized budget. Stop the affected computation and preserve its evidence. |
| Repair trigger | Local interface/resource/checkpoint error, failed candidate or implicated uncalibrated parameter. Diagnose and retry within the same scientific contract and budget. |
| Explanatory only | Loss, clipping, covariance/whitening probes, acceptance, finite energy magnitude and preliminary timing. These do not establish convergence or superiority. |
| Preserved result | Versioned attempts under `docs/plans/artifacts/ssl-lstm-q20-production-continuation-2026-09-16/`, with exact command, source/data/environment/device/seeds, elapsed worker time, checkpoint paths and decision. No numerical attempt exists there at this audit. |

The numerical ledger and mathematical audit retain provenance, failure modes,
early checks and hypothesis status for the material settings. In particular,
training caps, architecture/optimizer choices, learning thresholds, validation
bank sizes, finite search budgets and timing multipliers remain uncalibrated
where no target-specific evidence exists. Audit actual consumer defaults as
part of step 1; metadata validation cannot prove that they reach the intended
computation.

### Budget recovery

The preserved
[September 15 amendment](artifacts/ssl-lstm-q20-master-recovery-2026-09-16/audit-r1/budget-amendment-r1.json)
records 135,275.83289109988 campaign seconds remaining (37.5766 worker hours),
including 54,299.37991617 diagnostic seconds (15.0832 hours), and a 28,800-second
per-arm ceiling. These are **snapshots before subsequent repair work**, not a
fresh allowance or a current settled balance. The amendment's previous total
consumption must remain charged. Import subsequent test/diagnostic costs,
including the interrupted CPU smoke, once; record uncertain or unavailable
durations explicitly before reserving numerical work. Ordinary source repair
and metadata checks can proceed during that reconciliation. Repeating the
configuration values or making a new output directory cannot reset spending.

### Audit and immediate command

The skeptical audit found stale baselines and resume instructions, component
tests being presented as complete-program evidence, incomplete comparators,
an energy diagnostic promoted to a veto, unpriced downstream stages and native
work outside the purported timeout. The plan is revised above to resolve these
before a serious run. Known-target tests address interface correctness only;
current GPU/q20 measurements and independent reference evidence are separate
requirements. The original overclaims are corrected in the repair result.

This immediate command has been checked on main and performs metadata
validation without importing TensorFlow or launching training:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py validate
```

It reports protocol v1, development role, cohort size 12 and
`promotion_eligible: false`. There is no complete-master launch command yet.
The inspected prior test logs and budget amendment are preserved with their
original paths and checksums in the
[audit inventory](artifacts/ssl-lstm-q20-master-recovery-2026-09-16/audit-r1/inventory.json).
This update ran no GPU workload or new research experiment.

## Historical record through September 13 — superseded

Everything below is preserved history, including its former “active”, “current”,
budget and resume statements. Use the September 16 section above for execution.

Date: 2026-09-07  
Updated: 2026-09-13 (Asia/Shanghai)  
Status: `M4_P1_TERMINAL_FAILURE_SOURCE_CLOSURE_AND_STRICT_MOVEMENT_VETO`
Historical P1 result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-result-2026-09-06.md`; latest M4-P0 result and live P1 status: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`; earlier source-authority P0 result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-result-2026-09-05.md`; source-synchronized factor receipt is under `docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/`
Current reset memo: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-06.md`
Historical replay plan: `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md` (no active next command)
Factor admission plan (narrow backend admitted; Phase 9B candidate lane): `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md`
Reviewed Phase 9B plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-plan-2026-09-05.md`
Active M4-P0 executable-readiness plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`
Active P1 repair plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-plan-2026-09-05.md`
P0 auditor: `docs/benchmarks/audit_ssl_lstm_q20_phase9b_plan_2026_09_05.py`
P1 plan auditor: `docs/benchmarks/audit_ssl_lstm_q20_phase9b_p1_canary_plan_2026_09_05.py`
Historical passing P1 source audit (stale for current sources, not runtime clearance): `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r14/run_manifest.json`
Latest runtime result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`
Active cap amendment: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`
Parallel tuning execution plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`
Latest engineering result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-implementation-result-2026-09-08.md`

## Current execution: bounded P1 terminal failure

At **September 13, 00:15 Shanghai**, the corrected P1 retry is stopped and is
not complete. The service `bayesfilter-q20-phase9b-p1-r2.service` exited after
both arms completed four warmup chunks. Factor failed the final source-closure
check because the repository changed during execution. Strict failed the
declared `chain_without_movement` veto in chunk 3: chain 3 had zero acceptance
and movement. Its states, target values, and log-acceptance values were finite.
These are separate failure classifications; neither is a posterior result or
evidence against the research direction. P2 and posterior validation remain
blocked.

The preserved attempt is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/numerical-repairs/eigh-refinement-r2/launches/p1-d1884c1974/`.
Factor used 8,719.6063 worker seconds and strict used 13,165.7078; the settled
campaign total is 46,214.8021 seconds with 40,185.1979 seconds unreserved.
Do not reuse these failed P1 outputs as claim-bearing evidence. Before a fresh
attempt, freeze and validate the complete source closure, add a pre-launch and
per-chunk closure guard, rerun focused regression checks, and decide whether
strict needs fresh scope-specific tuning under the unchanged budget.

The previous live-run statement below is historical and superseded by this
terminal receipt.

### Historical launch snapshot

At **September 12, 18:47 Shanghai**, the corrected P1 retry is active as
`bayesfilter-q20-phase9b-p1-r2.service`. Factor and strict run concurrently on
non-display GPUs 1 and 0 with `TF_FORCE_GPU_ALLOW_GROWTH=true`; display GPU 2
is not used. Both workers have passed setup and started their first P1 chunks.
No terminal P1 result or scientific promotion decision exists yet.

The earlier r2 P1 attempt stopped at 14:06 Shanghai after 10.84 measured
GPU-worker seconds total across both arms
because its committed checkpoint identity predated the accounting-only source
repair. That was an execution-provenance blocker, not a numerical or GPU
failure. The migration now records the reusable pre-repair setup identity,
archives the superseded records, and requires current sources for new streams.
Focused recovery/ledger tests pass 80/80 in 38.52 seconds, including six new
setup-identity and successful-sibling reuse regressions (CPU-only).

M4-P0 is complete for bounded P1: both fresh tuning arms and actual
SIGKILL/resume canaries pass; both full runtime measurements complete with
two healthy 500-transition-per-chain chunks each. Factor takes 4,370.6522
worker seconds and strict 6,639.4178. Reserve-inclusive P1 estimates are
16,519.6636 and 25,283.0138 seconds, respectively, within their assigned
18,000- and 28,800-second allocations. These are descriptive forecasts, not
completion guarantees. The current-source receipt is r2
`phase0-readiness.json`, status `PASS_CURRENT_SOURCE_PHASE0_FOR_BOUNDED_P1`.

The campaign has consumed 24,329.48797829113 seconds and reserved 46,800
seconds for the live P1 wave, leaving 15,270.51202170887 unreserved seconds.
The eight-hour per-arm cap and 86,400-second aggregate budget remain binding.
The total unspent balance before live settlement is 62,070.51202170887
seconds; the unreserved balance is not the entire remaining budget.
The active output is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/numerical-repairs/eigh-refinement-r2/launches/p1-d1884c1974/`.

### Previous pre-retry snapshot

At **September 12, 02:45 Shanghai**, fresh training and public tuning have
completed for both arms. Actual interruption/resume canaries pass exact sample
and trace equality on both GPUs, reusing the first committed chunk and
recomputing the interrupted second chunk. In r2 `runtime-c355a7aa08`, factor
has completed both 500-transition-per-chain runtime chunks with all health
checks passing. Strict has passed its first 500-transition chunk and is
executing the second. No nonfinite-value or target-status veto occurs in the
three completed runtime chunks. These checks establish neither convergence
nor a supported method ranking.

The service remains active; Phase 0 and P1 are not complete. It will compute
the two-arm affordability forecast when strict finishes, then proceed to
bounded P1 if admissible. The ledger still has 13,308.582441157134 settled
seconds and 57,600 reserved for the runtime wave; factor's completed worker
receipt (4,370.652188197 seconds) awaits wave settlement along with strict.
Do not interpret the ledger's unsettled balance as already available compute.

### Earlier launch snapshot

The September 11 reboot restores matching NVIDIA driver/library 580.178.04.
The eight-sweep repair then passes GPU/XLA validation of both saved 32-row
banks and the original four-row failure endpoint on both backends, with no
invalid rows. Odd-dimensional sign controls pass and each static function
traces once. Maximum bank likelihood residual is 2.27374e-12 and scaled
analytic-score residual 4.43408e-14 against the independent CPU reference.
Memory growth is verified; the diagnostic uses 53.115703790001135 GPU-worker
seconds. Evidence: `launches/eight-sweep-gpu-validation-5c70410169/` in the
original 24-GPU-hour campaign. The 137 CPU regressions remain current.

The active numerical migration now uses
`numerical-repairs/eigh-refinement-r2/`; the prior migration and r1 evidence
are preserved. Activation verifies 124 bundles/3,109 tensors and leaves the
original campaign start/ledger unchanged. At **September 12, 00:44:39 Shanghai**,
service `bayesfilter-q20-phase9b-eigh8-20260912-r1` starts the fresh reference/
setup wave `reference-67d95ab1f2`. Both workers start in parallel: factor on
non-display GPU 1, strict on non-display GPU 0. GPU 2 remains the display
device. No renewed campaign authorization or budget was needed.

At 00:48 Shanghai, both charts have completed and committed fresh training
through beta 0, 0.5 and 1; all six 32-row preflights pass. Both public tuner
processes are active, with finite target/score/acceptance and valid status in
the completed tuning calls inspected so far. These short calls do not clear
the pending full runtime health or convergence checks.

The coordinator continues through fresh tuning, actual interruption/resume
checks, both full runtime/health measurements and affordable bounded P1.
The eight-hour per-arm maximum and original 86,400 aggregate-second allocation
remain in force. At launch, 9,523.699322834134 seconds are settled, 7,200 are
reserved for the live setup wave, and 69,676.30067716587 are unreserved. Total
unspent balance before live-worker settlement is 76,876.30067716587 seconds
(21.3545 GPU-hours); reservations are not extra spending. There is no reliable
finish forecast until the new two-arm timing check completes. Fixed-bank
validation does not establish trajectory stability, convergence or ranking.

## Historical pre-reboot state: CPU repair passes; GPU validation pending

At September 11, 15:36 Shanghai, trusted GPU inventory remains unavailable:
loaded NVIDIA module 580.173.02 conflicts with installed NVML 580.178.04.
Unattended upgrades installed the latter at 06:02, after the failed preflight;
the host requests a reboot. This independent environment blocker must be
resolved by the operator before GPU execution. No reboot/package change is
authorized or performed here, and no GPU worker is running.

CPU localization identifies the remaining preflight failure: four refinement
sweeps stop before the eigenpair residual meets the unchanged bound. The
saved covariance is positive definite; this is not an actual negative
eigenvalue or an invalid parameter. Eight sweeps pass the same fixed 32-row
banks for both arms (64/64 rows), with likelihood/analytic-score agreement
against native CPU reference. The maximum likelihood residual is 2.38742e-12
and scaled score residual 1.00143e-13. Evidence:
`runtime-health-diagnostic-20260911-r1/preflight-eight-sweep-bank-r1/`.

The local implementation now uses eight sweeps with every validity threshold
unchanged. All **137 focused CPU regressions pass** in 121.77 seconds, including
the new tests that reproduce the old failure on both paths. `git diff --check`
passes. This is CPU/XLA numerical evidence, not GPU clearance or
convergence. The four-sweep migration and `eigh-refinement-r1` namespace are
stale for the edited core. Do not auto-resume them: after host repair, validate
the eight-sweep fixed banks on GPU/XLA, record a fresh numerical migration and
namespace under the same ledger, then repeat parallel tuning, actual recovery
canaries and both runtime estimates. The old runtime forecast cannot price
the increased refinement work.

The resume guard rejects the stale migration before launch. The original
campaign start/ledger/migration bytes are unchanged; source snapshots and
validation are preserved in
`runtime-health-diagnostic-20260911-r1/eight-sweep-validation-a8df05dcf2/`.

The original NaN cause is localized: the XLA eigensolver falsely reports
an indefinite placement covariance at the rejected leapfrog endpoint. The
saved matrix is positive definite (60-digit minimum eigenvalue approximately
4.67714e-14), but the original GPU solver reports -1.42567e-11 because an
internal Jacobi cutoff uses binary32 precision. Covariance subtraction is not
the supported explanation. Both strict and factor-cached backends share the
defect; factor's old healthy timing does not clear the repaired scope.

The binary64 refinement keeps the target, derivative equations, SPD thresholds
and rejection/health guards unchanged. It passes 122 CPU regressions and fresh
integrated GPU/XLA comparisons on both paths: all four rows valid, likelihood
agreement within 7.11e-15 and scaled analytic-score agreement within 6.10e-15
against the independent CPU reference. These are numerical-equivalence checks,
not tuning, recovery, convergence or posterior admission.

The preceding execution used fresh chart/tuning/stream/canary namespaces for
**both arms**, under `numerical-repairs/eigh-refinement-r1/` in the original
24-GPU-hour campaign. That migration passes 135 combined CPU tests and preserves
the original start/ledger; 101 bundles and 2,697 tensors verify. Both GPU workers
start in parallel on non-display GPUs 0/1 at September 11, 04:52:31 Shanghai,
but both fail fresh beta-zero chart preflight before any optimizer update or
HMC tuning. The service stops at 04:52:55. No worker is currently active.
Attempt: `launches/reference-b6ab845731/` in the new numerical namespace.
Preserve that failed attempt; its CPU localization and repair are summarized
here. Do not change seeds to evade the failed bank or relax preflight. The
original ledger and all old evidence remain intact; source migration and
fresh recovery/runtime/P1 are still pending.

At this update, 9,470.583619044133 seconds are consumed, zero reserved, and
76,929.41638095587 aggregate GPU-worker seconds remain (21.3693 hours).
The per-arm ceiling remains eight hours. Phase 0 and P1 are incomplete; no P2,
posterior/default promotion or method ranking is authorized. The detailed
repair/audit plan is
`bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md`.

## Historical snapshot: covariance failure localization

This snapshot predates the eigensolver diagnosis; its unresolved-covariance,
strict-only repair and source-restoration statements are superseded above.

The serializer repair and both actual SIGKILL/resume canaries pass. The resumed
service subsequently stopped on **September 10 at 17:34:11 Shanghai**:
factor completed both full-health runtime calls; strict's first 500-transition
call contains one negative-infinite proposal acceptance ratio at transition
159/chain 4. The proposal was rejected and sampled states remain finite, but
the declared energy/acceptance veto correctly blocks Phase 0 closeout. There
are no NaNs in the stored acceptance trace. Do not confuse this failure with
the earlier undefined tail-ESS serialization bug or replay the unhealthy chunk
until it passes.

Execution continues with a bounded covariance investigation under
`bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md`.
The 159-transition GPU prefix reproduces every sampled state, acceptance ratio,
decision and sampled target exactly. Cached-gradient localization identifies
the first invalid target evaluation at the second leapfrog endpoint of
transition 159, chain 4: finite parameters, but a placement-covariance minimum
eigenvalue of -1.4256699903299468e-11. The bridge intentionally returns NaN for
this invalid row; the following leapfrog becomes NaN and Metropolis rejects it.
This is not checkpoint corruption. Why the covariance becomes indefinite still
requires investigation; do not widen its tolerance or remove the guard.
No GPU worker is active at this update, and no new tuning or P1 has launched.
The unlaunched tuning draft was withdrawn; the validated serializer-only source
closure is restored and passes 56 CPU regressions. Preserve factor's successful
timing, original charts, recovery receipts and failed strict evidence. Current
result: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`.

The settled campaign ledger records 9,269.455504760117 consumed, zero reserved,
and 77,130.54449523988 remaining aggregate GPU-worker seconds (21.4252 hours).
Each arm
still has an eight-hour ceiling. Phase 0/P1 are incomplete; P2 is not authorized
by this continuation and no posterior or superiority claim is made.

## Historical September 10 resumed execution

The checkpoint-serialization defect is repaired and the existing campaign
resumes at **September 10, 2026, 08:23:01 UTC / 16:23:01 Asia/Shanghai** in
`bayesfilter-q20-phase9b-serializer-20260910-r1.service` (coordinator 804934).
The completed factor reference is reused; strict reruns its uncommitted tuning
on non-display GPU 1 with verified memory growth and XLA. Later independent
two-arm waves remain process-parallel, subject to the unchanged GPU policy.

The repair passes 161 CPU checks, restores the old factor checkpoint and
round-trips the actual strict artifact without losing its candidate-1
nonfinite-efficiency veto. The NaN is an undefined short-chain tail ESS,
not evidence of nonfinite HMC states in that artifact. The final migration
preserves original setup/stream identities, plan, start record and accounting;
only the wrapper's serializer and execution-provenance handling change.
The older failure and launch snapshots below are historical.

The same 86,400-GPU-worker-second allocation and 28,800-second arm ceiling
govern execution. The pre-resume balance is 84,212.35038186601 seconds; strict
currently reserves 3,600 seconds and live spend is not yet settled. Next:
finish strict reference, actual recovery canaries, both full runtime/health
measurements, Phase 0 closeout and bounded P1 if affordable. P2 and posterior
admission remain separate. There is no reliable finish ETA yet.

Repair plan: `bayesfilter-ssl-lstm-q20-phase9b-nonfinite-checkpoint-repair-2026-09-10.md`.
Current result: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-result-2026-09-09.md`.

## Historical execution interruption

The service stopped at 2026-09-09 16:36:39 UTC (September 10, 00:36:39
Asia/Shanghai). Factor reference finished; strict finished public tuning but
typed-checkpoint hashing rejected an in-memory NaN diagnostic. The public
selected candidate passed its limited tuner screen; a different candidate's
nonfinite-efficiency veto must remain recorded. This is a checkpoint
serialization repair trigger, not budget exhaustion or research-direction
rejection. Spent: 2187.64961813399 seconds; remaining: 84212.35038186601 seconds
(23.3923 GPU-hours), zero reserved. Repair lossless typed serialization, test it,
record source migration without resetting accounting and resume unfinished work.
Full runtime, completed canary and P1 remain pending; no reliable finish ETA yet.
See the amendment result for the failure trace and preserved work. The running
snapshots in later sections predate this stop.

## September 9 approved eight-hour amendment

The owner approves a new **28,800-second per-arm cap** and a fresh **86,400-
second aggregate GPU-worker budget**. The active amendment plan is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Its fresh campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.
The fresh campaign launched at **2026-09-09 16:00:10 UTC** (September 10,
00:00:10 Asia/Shanghai) as user service
`bayesfilter-q20-phase9b-24gpuh-20260909.service`. At the 16:04:50 UTC snapshot,
the factor arm runs fresh tuning on non-display GPU 0; its chart checkpoint is
committed and nine tuning calls have completed. GPU 1 was at 64% at launch, so
the strict arm queues rather than occupying the display GPU. The launcher uses
separate processes and concurrent arms when eligible devices permit it.
The reference wave reserves 7,200 seconds; live worker time is not settled
until completion. Read the amendment result and ledger for subsequent status.

The completed four-hour campaign, its ledger, runtime result, canaries and
component profiles remain immutable historical evidence. Its measured
reserve-inclusive forecasts, 10,751.04 seconds for factor and 16,202.12 seconds
for strict, both fit the new eight-hour arm cap; their 26,953.15-second combined
forecast fits the new aggregate allocation. This removes the old cap arithmetic
veto only. The actual P1 entrypoint now delegates to the checkpoint-aware
parallel coordinator; 128 focused CPU tests pass. The kernel and full sampled-
state status recomputation remain unchanged. Status reuse is optional future
optimization, not a launch prerequisite. GPU closeout and scientific gates
remain open.

The approved next execution adopts the unused prepared ledger, then runs a
bounded GPU recovery canary and fresh two-arm measurements. The coordinator
automatically enters bounded P1 if current-source recovery/health/affordability
pass; no further campaign approval is required. Non-display GPUs remain preferred under the 40% utilization and
5 GiB headroom policy; the display GPU is fallback only. No P1/P2 launch occurs
before a new Phase 0 closeout.

## September 9 recovery, health and runtime outcome (historical four-hour run)

The owner raises the per-arm ceiling to **14,400 seconds** and authorizes a
new **36,000 aggregate GPU-second** recovery/health/runtime campaign. Prior
ledgers remain unchanged and unused allocations are not combined. The active
subplan is `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-plan-2026-09-09.md`.
It replaces the old 2,600-second affordability boundary for new work, but does
not turn historical incomplete receipts into a readiness pass.

The checkpoint-aware shared controller now supports durable replay. Both real
GPU mid-call SIGKILL canaries pass exact sample/full-trace equality against their
uninterrupted references, reusing the committed first chunk. Both full runtime
arms complete two actual 500-transition calls, checking 4,000 sampled states per
arm with no required health veto. Factor and strict ran in separate processes
on non-display GPUs 1 and 0; display GPU 2 was unused. The terminal CPU suite
passes 152 tests. The campaign is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-10gpuh-20260908T192200Z/`.

| Six-chunk forecast, seconds | Factor | Strict |
|---|---:|---:|
| Setup, first/repeated calls and observed overhead | 9,305.42 | 14,040.61 |
| Including one steady-chunk interruption reserve and cleanup grace | 10,751.04 | 16,202.12 |
| Fits the 14,400-second arm ceiling with reserve | Yes | **No** |

These are descriptive single-repeat forecasts, not statistical bounds or
posterior evidence. Strict exceeds the reserve-inclusive cap by 1,802.12 seconds.
The current fresh two-arm reserve-inclusive forecast totals 26,953.15 seconds,
also above the **24,647.44 seconds remaining**. Final consumption, including
failed attempts and component profiling, is **11,352.56 seconds (3.15349
GPU-hours)**; zero is reserved and no campaign GPU worker remains. Older ledgers
and their unused balances remain separate.

P0-I/J recovery and full sampled-state health checks pass for this route; P0-K
component localization is complete. Target value/score dominates transport, and
standalone status evaluation costs about another target evaluation. The next
repair hypothesis is to preserve already-computed accepted-state status instead
of recomputing it. No such optimization or resulting speedup is established.
P0-L affordability and actual P1 consumer integration/source/readiness checks
remain open. **P1/P2 have not been launched and are not executable yet.**

Recovery replays an unfinished stage or chunk from committed inputs and seed;
it does not restore a live CUDA instruction pointer. The GPU canary uses eight-
transition chunks; actual runtime uses 500. An unfinished tuning stage or
500-transition call may have to be repeated. Universal hardware-failure recovery
and automatic recovery of the old serial P1 entrypoint are not claimed.

The initial grid's movement failures, same-chart smaller-step repair, strict
setup-timeout repair and GPU-queue/sibling repair are preserved in the result.
Scope-bound selected handoffs are factor epsilon 0.01375 and strict epsilon
0.0275, both L=3; no candidate/backend ranking or default promotion follows.

## September 8 parallel-tuning result (historical)

The user-authorized 14,400-GPU-second campaign completed its two-worker startup
smoke and fresh factor/strict cold-scope tuning without a retry. Both ran in
separate processes on eligible non-display GPUs 1 and 0; display GPU 2 was not
used. Each tuning arm measured all eight declared candidate pairs and issued
a verified handoff after fresh heldout checks. The selected pair is epsilon
0.055, L=3 in each arm; candidate indices 0 and 2 remain viable without a
statistically supported ranking. Candidate-level movement vetoes rejected the
other six pairs, not either backend or the research direction.

Tuning took 493.46 seconds elapsed with overlapping worker intervals; the
factor/strict worker lifetimes were 342.40/492.52 seconds. Including the GPU
smoke, consumption is **844.217169 aggregate GPU-seconds (0.234505 hours)**,
leaving **13,555.782831 seconds (3.765495 hours)** unused. All four worker
settlements are complete, no reservation remains, and the worker processes
exited. The old 5,200-second ledger is unchanged. These are descriptive timings,
not a paired speedup comparison.

The active campaign root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-parallel-tuning-2026-09-08/campaign-4gpuh-20260908T090200Z/`.
Its `validated_result_summary.json` preserves artifact/source verification and
budget arithmetic. The result note above is authoritative for this new run;
the parallel execution plan remains frozen at the launch hash. P0-M process
tuning integration is validated for this cold scope. P0-I/J/K/L full-controller
readiness, P1, and P2 are not closed by a tuning-only result. Next work is
full-controller instrumentation and bounded cost localization, not another
tuning replay merely to use the reserve or an immediate P1 launch.

## September 7 execution outcome (historical)

The bounded readiness diagnostic started at `2026-09-07T14:34:25.180687Z`
on physical GPU 1, UUID `GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`.
At selection it was non-display, at 0% utilization, with 32,212 MiB free.
Its fresh output root is
`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/runtime-diagnostic-20260907T143204Z/`.
The process is stopped and its GPU allocation released. The shared ledger
charged 3,183.590345 seconds to this attempt, in addition to the 1,832.61-second
estimated historical debit. It has 183.799655 seconds remaining and no active
reservations. Do not relaunch using the former 3,307.39-second allocation.

Audit `r13` passes. The placement/readiness suite passed 52 tests; the broader
CPU repair suite passed 73 tests; the route-policy suite passed 6 tests.
These suites overlap and their counts must not be added. Compilation and
whitespace checks passed. Factor chart construction, fresh cold-scope tuning,
and two compiled 500-transition calls completed. Their times were 1,360.86
and 1,390.70 seconds, respectively. First-call plus five steady calls forecasts
8,314.36 seconds before setup/overhead, exceeding the 2,600-second arm cap.
The strict arm was interrupted without a complete receipt. No P1 launchable
closeout was issued. The factor receipt lacks full trace-health and durable
startup memory-policy evidence; it is not complete health clearance.

The planned external timeout was omitted from the launch command. A supervisor
was attached before the deadline without changing the running scientific source.
On measured budget infeasibility, manual TERM at 23:26:29 +08:00 did not finish
the worker; KILL followed at 23:27:28, before the workload deadline. All elapsed
work and termination grace were charged once. The result records this launch
defect, the reporting gaps, and the next Phase 0 repair; it does not reject
the target, factor backend, or research direction. Post-run audit r14, 95
unified CPU checks, 6 route-policy checks, compilation/whitespace, and ledger/
artifact verification pass. They do not resolve the remaining runtime gates.

## Scope and purpose

This is the active master program for the q=20 SSL-LSTM tempered reverse-KL
transport and fixed-transport HMC work. It governs which plan is active, what
may be executed next, how a failed phase is repaired, and which scientific
claims remain closed. It is a control document, not evidence that the
transport is a good Gaussianizing map or that the posterior sampler has
converged.

### Execution topology ruling

The legacy Phase 9A and Phase 9B runners are serial at the scope, arm, and
candidate-evaluation levels. Their `FixedTransportReusableRunnerPool` is a
within-process TensorFlow reuse mechanism, not multiprocessing. Therefore the
previous execution was **not process-parallel** despite three available GPUs.

For active tuning campaigns, this master program requires process-level
parallel tuning under
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md`:
one independent OS process per selected GPU, non-display GPUs preferred under
the load/headroom policy, display GPU only as fallback, framework import after
GPU pinning and memory-growth setup, private worker artifacts, and parent-owned
timeout/budget settlement. The first safe split is by independent scope or
arm. Candidate pairs remain inside a worker until the complete declared grid
has been measured and reconciled; they are not silently sharded into partial
selection evidence.

This ruling changes execution architecture only. It does not authorize a new
GPU run under the exhausted September 7 campaign or relax any scientific gate.

September 8: the new Phase 0 tuning-only entrypoint
`docs/benchmarks/run_ssl_lstm_q20_phase9b_parallel_tuning_2026_09_08.py`
now connects the coordinator to the existing chart builder and public scope
tuner. Two eligible non-display GPUs permit concurrent factor/strict workers;
one eligible non-display GPU queues the second worker. Display fallback is
allowed only when no non-display GPU is eligible, not to fill an extra slot.
CPU process/integration checks and the new GPU smoke/tuning campaign now pass.
Candidate pairs within a worker remain serial. The old P1 sampler is
not yet converted or reopened, and tuning concurrency does not fix the measured
8,314-second factor sequential-sampling forecast.

September 9: the recovery/runtime launcher also executes independent arms in
parallel, with durable setup/chunks and parent-owned accounting. Full-controller
timing and component localization are now complete. Next work is a source-
equivalent cost repair and integration into the actual P1 consumer. Source audit
r14 remains a historical static record, not clearance for the new sources.

The user authorized a fresh **4 aggregate GPU-hour / 14,400 GPU-second**
allocation on September 8, 2026, for this tuning-only validation. A two-worker
startup smoke reserves 600 GPU-seconds, then the first factor/strict tuning
wave is capped at 3,600 seconds per worker (7,200 aggregate), leaving an initial
6,600-GPU-second repair/retry reserve. Unused reservations are released. This
does not reopen or enlarge the exhausted 5,200-second readiness campaign and
does not fund P1. The exact allocation and stop conditions are governed by the
parallel-tuning execution plan; a fresh ledger and fresh output root are
required before launch.

The earlier particle-authority master program dated 2026-08-25 remains the
historical record for its phases 0--26 and its direct full-state LEDH blocker.
It does not govern this newer tempered-transport continuation. The current
implementation plan and execution record are subordinate documents:

- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-execution-2026-08-28.md`

The post-terminal performance/whitening investigation was a bounded M3-C
continuation. It was subordinate to this master and diagnosed execution
without reopening the failed M3 replay. Its closed plan and result are
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`
and
`docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`.

The 72-core staged process-parallel continuation is a new, explicitly
diagnostic performance design. It is subordinate to this master, uses fresh
attempt roots, and leaves the M3 replay result and Phase 9B block unchanged.
Its plan is
`docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`.

The GPU10000 continuation is a closed subordinate performance design. It keeps
the frozen q=20 target and C5 protocol, and its result remains historical
performance evidence only. Its governing plan is
`docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md`.

The bounded factor-route promotion test is a completed subordinate mechanics
branch.  It repaired backend identity propagation, compared the strict and
strict-factor routes on fresh q=20 GPU0/CPU fixtures, and checked paired HMC
transitions and short-chain mechanics.  Its result is
`docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-result-2026-09-04.md`.
The subsequent source-synchronized fresh tuning completed narrow q=20
numerical-backend admission. It did not change the generic public API default
or reopen the old replay; Phase 9B retains its independent readiness gates.

## Current state

| Area | Binding state | Evidence |
|---|---|---|
| Target and bridge | q=20 SSL-LSTM posterior on `theta in R^4`; proper likelihood-tempered bridge and strict square-root backend are admitted for mechanics | Phase 0 and C5 receipts; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Transport protocol | C5 `phase8-k2-compact-high-l3-pure`, two `(16,16)` tanh charts, betas `(0,.5,1)`, pure continuation, fixed `gamma=(.5,.5)` | `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md` |
| Tuning policy | `measured_joint_grid_v1`; directional acceptance repair is diagnostic-only | `docs/plans/bayesfilter-hmc-tuning-guide-policy-repair-result-2026-09-01.md` |
| Phase 9A localized repair | Complete for chart 1, beta 0 as mechanics evidence only; four of six scopes were intentionally not attempted | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md` |
| Full Phase 9A replay | Corrected canary retry reproduced the bounded resource failure; M3 is terminal and full replay is blocked | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-result-2026-09-02.md` |
| M3-C performance/whitening continuation | Closed bounded diagnostic; no nomination and no fast grouped-HMC admission | `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md` and repaired manifest |
| M3P 72-core staged process continuation | Canary passed after timeout repair; full CPU-only diagnostic stopped at the declared cap, no Phase 9B admission | `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md` and `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md` |
| GPU10000 performance continuation | Closed subordinate diagnostics; raw-basis reuse vetoed, strict-factor local parity superseded by M3R; no active strict-baseline canary command | `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md` and `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-result-2026-09-04.md` |
| Factor-route fresh tuning | Source-synchronized six-scope run passed in 3,551.30 s and independent audit passed; factor backend admitted for q=20 Phase 9B only | `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md` and the source-sync receipt |
| Phase 9B executable readiness | Current-source M4-P0 passes: both actual recovery canaries, full runtime health and reserve-inclusive affordability; 80 current recovery/ledger CPU tests pass | Eight-hour amendment result and r2 `phase0-readiness.json` |
| Phase 9B process-parallel tuning | Fresh eight-sweep factor/strict training and public tuning complete; both committed setups are restored by live P1 workers | r2 `reference-complete.json` and live P1 setup receipts |
| Phase 9B posterior validation | Bounded P1 is running in parallel; no terminal P1 result, P2 or posterior/reference validation yet | Eight-hour amendment result and r2 `launches/p1-d1884c1974/` |
| Whitening and mode claims | Closed; large pullback residuals and short, seed-sensitive chains remain diagnostic evidence | M3-C result and manifests |

## Remaining executable blockers

There is no outstanding launch blocker for the active bounded P1 attempt.
Current-source recovery, full runtime health, affordability and restored setup
identity pass. P1 is executing; its terminal screens and the later scientific
gates remain unresolved. Preserve source bindings while workers run.

| Blocker | Current status | Required closure |
|---|---|---|
| Actual P1 consumer integration | Implemented via `--campaign-root`; current-source Phase 0 passes and both live P1 workers restore setup | Await terminal P1 receipts; old serial path is historical |
| Stable TensorFlow graph | Explicit state/seed signature and repeated-shape regression now pass | Preserve the signature in the current source closure |
| NeuTra route governance | Full route audit passes with new and legacy dispositions preserved | Re-run after any new qualifying route is added |
| P0-L complete-schedule affordability | Repaired-route forecasts including one-chunk reserve/grace are 16,519.66 s factor and 25,283.01 s strict; both fit assigned caps and aggregate funds | Enforce live allocations of 18,000/28,800 s; forecasts are not guarantees |
| Aggregate accounting | Original 86,400-second campaign has 24,329.49 s settled, 46,800 s reserved and 15,270.51 s unreserved; reused reservations no longer corrupt settlement | Settle measured worker lifetimes; preserve the earlier ledger and all attempts |
| Setup identity | Accounting-only repair explicitly records reusable current-numerical-scope setup; changed numerical identities remain rejected | Keep current execution-source binding for new streams and checkpoints |
| Runtime provenance and resource | Both workers verify growth before initialization and run concurrently on non-display GPUs 1/0 | Preserve UUID/PCI binding, recurring headroom checks and unrelated processes; display GPU 2 remains unused |

Chart-threshold calibration, downstream posterior/reference agreement, mode
travel, uncertainty analysis, and default promotion are later scientific gates.
They remain blocked, but they are not prerequisites for deciding whether the
P1 mechanics launcher is technically executable.

## Authority order

When documents disagree, apply this order:

1. Active repository policy in `AGENTS.md` and applicable project directives.
2. This master program and its latest dated reset memo.
3. The active phase subplan named by this master, including its evidence
   contract and frozen identities.
4. The mathematical note and parent implementation plan.
5. Source code, tests, and run manifests that satisfy the active plan.
6. Historical plans and results, which explain prior decisions but cannot be
   reused as current tuning or confirmation evidence.

Engineering correctness, numerical validity, and scientific interpretation are
separate ledgers. A passing import, finite value, or mechanics screen never
promotes a posterior or whitening claim.

The prior M3/M3-C state and the M3P continuation are closed.  M3P's repaired
canary passed in `585.254545083968` seconds, but its fresh full attempt stopped
at the declared wall cap before selection and finalization were complete.
Those historical results did not admit Phase 9B. The old M3/M3P/M3Q launchers are not valid for
another attempt, and the historical GPU10000 canary is not an active next
command. M4-P0 has now closed for bounded P1 under the eight-hour amendment;
the current execution is the refreshed P1 plan. P2 remains blocked.

The September 4 factor mechanics result, the September 5
source-synchronized receipt/reconciliation, and the September 6 reset memo are
current-session authority for the narrow q=20 numerical-backend admission. The
source-synchronized receipt is preserved and independently hash-checked. The
factor route is a q=20 candidate only; no strict tuning artifact, partial
replay, or mechanics fixture is reused as factor claim evidence.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Main question | Can fresh Gaussian reverse-KL charts, a proper temperature bridge, and fixed state-independent chart kernels provide reliable q=20 exploration after target-specific tuning? |
| Candidate mechanism | Independent tempered reverse-KL transport ensemble, optional joint mixture refinement, fixed multi-chart HMC, and exact adjacent replica exchange. |
| Baseline ladder | Physical-coordinate HMC; single cold NeuTra; physical replica exchange under the same bridge; single-chart tempering; cold multi-chart HMC; plain ensemble; optional joint-RKL ensemble. |
| Expected failure | Chart collapse, poor bridge overlap, invalid inverse/log determinant, chart-specific tuning failure, static XLA/retracing cost, high acceptance with little effective movement, or mode locking. |
| Promotion criterion | For a declared phase, all exact identity/health gates pass, every required scope is measured, disjoint held-out checks pass, and any later stochastic comparison has uncertainty evidence. |
| Promotion veto | Wrong target/bridge identity, non-finite value/score/map, stale or reused scope, unmeasured candidate, missing movement/status/energy telemetry, invalid transition or swap ratio, memory-growth/XLA violation, output collision, or claim beyond the computed quantity. |
| Continuation veto | Exact fixture contradicts the contract; target or common measure is unavailable; required artifacts cannot be made durable; three scoped infrastructure repairs make no progress; the declared campaign budget is exhausted; or a future repair would change target, data, hardware class, privacy boundary, or scientific contract. |
| Repair trigger | High acceptance, poor ESS/movement, seed sensitivity, retracing, no viable candidate, inadequate grid resolution, or a localized implementation/resource failure. |
| Must not conclude | No IID-Gaussian whitening, exhaustive mode discovery, posterior correctness, convergence, sampler superiority, high-dimensional scaling, production readiness, or default readiness from the current evidence. |

## Evidence contract

Every serious phase must state these fields before execution:

- exact target, bridge, chart, beta, dtype, backend, XLA, and data identities;
- comparator and target-call/wall-time accounting;
- primary pass criterion and hard veto diagnostics;
- explanatory diagnostics whose values cannot promote a candidate alone;
- disjoint calibration, selection, held-out, and confirmation seed roles;
- fresh versioned output root and a manifest containing command, Git state,
  environment, device/memory policy, seeds, timings, and hashes; and
- explicit nonclaims and the smallest next repair.

For the current q=20 route, acceptance is descriptive or a repair trigger.
Selection uses measured joint `(epsilon,L)` pairs and replicated efficiency
diagnostics. Modern R-hat, ESS/MCSE, declared-region travel, initialization
forgetting, and replica round trips become promotion gates only in a separately
approved sequential Phase 9B plan.

## Frozen identities and defaults

| Choice | Provenance and status |
|---|---|
| `theta` dimension 4 and target signature above | Frozen target fact; a 60-dimensional internal filtering state is not the sampling measure. |
| C5 compact-high K=2/L3 protocol | Frozen calibration representative for Phase 9A; not a universal architecture default. |
| `measured_joint_grid_v1` | Repaired guide and shared tuner policy; every declared pair is measured before selection. |
| q=20 localized grid | `(epsilon=(0.55,1.20), L=(3,8))` in the completed v4 mechanics probe; the next replay draft uses `(0.25,0.55,0.85,1.20) x (3,8)` as a target-specific hypothesis, not a promoted default. |
| Four varied starts and four chains | Mechanics baseline for the localized probe; too short for posterior inference. |
| TensorFlow/TFP, active-policy GPU, XLA, TF32, memory growth before initialization | Repository execution policy and the September 7 owner GPU instruction; CPU-hidden fixtures are explicit exceptions. |
| No pfor or row-mapped scalar target | Repository policy and route-scan requirement. |
| Prior 18-hour particle-authority budget | Closed historical campaign; it is not silently transferred to this continuation. A new serious campaign must declare its own cap. |

## Phase map and ownership

| Master stage | Purpose and exit | Active document | State |
|---|---|---|---|
| M0 | Bridge, density, trainer, lineage, fixed-kernel, replica-exchange, and analytic mechanics foundations | Parent implementation plan phases 0--7 and execution record | Complete for stated mechanics gates |
| M1 | q=20 calibration, diversity/overlap checks, optional joint-arm feasibility, and C5 freeze | `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md` and C5 freeze subplan/result | Complete; K=2 protocol frozen for tuning only |
| M2 | Fresh chart construction and scope tuning preflight | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-subplan-2026-09-01.md` | Complete localized mechanics pass; Phase 9A partial |
| M3 | Performance repair, chart-1/beta-0 canary, then fresh six-scope replay | `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md` | Terminal: second bounded canary resource failure; full replay blocked |
| M3-C | Post-terminal performance and whitening diagnostics | `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md` | Complete; no N2 nomination, fast grouped-HMC path rejected |
| M3P | Staged `8x4 + 2x8 + 6x4` process-parallel mechanics/performance diagnostic | `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md` | Closed at full-cap blocker; Phase 9B remains blocked |
| M3Q | GPU10000 replay feasibility plus target batching/eigensystem-reuse diagnostics | `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-plan-2026-09-04.md` | Closed subordinate diagnostic; raw-basis score veto; no pending replay command |
| M3R | Factor-route identity, q=20 parity, paired HMC mechanics, and fresh six-scope tuning admission | `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md` | Numerical backend admitted; chart/tuning quality triggers remain |
| M4-P0 | Executable readiness: repair contracts, stabilize the TensorFlow graph, classify routes, reconcile budget, and validate runtime provenance | `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md` | Complete for bounded P1; current-source recovery, health, timing and setup restoration pass |
| M4 | Sequential warmup, retained beta-one sampling, comparator runs, region/travel and posterior checks | `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-plan-2026-09-05.md` | P1 running in parallel; P2 and posterior promotion remain blocked |
| M5 | Controlled dimension/scaling ladder with uncertainty and matched baselines | New plan required after M4 | Blocked |

No stage may skip its entry gate because an earlier mechanics artifact is
finite. M3 replay is terminal, M3-C is closed without a nomination, M3Q is a
closed subordinate diagnostic, M3R has a mechanics pass (including same-scope
handoff regression), and M4-P0 now permits bounded P1 mechanics execution.
Claim-bearing validation remains gated beyond P1. The earlier Phase 9B source-authority P0 is a
completed prerequisite; it is not a substitute for this executable-readiness
gate.

### M4-P0 executable-readiness gate

The detailed gate is governed by
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`.
It is a technical and provenance gate, not a posterior-validation phase. Its
required closure covers the repaired P1 runner contract, an explicit stable
TensorFlow input signature or reviewed exception, new-path NeuTra route
classification, compile-versus-steady-state timing, durable aggregate budget
accounting, fresh audit/source hashes, and a valid GPU memory/XLA/TF32/allocator
receipt. The full route-policy audit's unrelated legacy findings must remain
visible; they may not be silently relabeled as active claim-bearing routes.

For this state, “M3 continuation” refers only to the explicitly named M3Q
diagnostic plan. The M3R mechanics result does not authorize the old six-scope
replay, reuse of partial calls, a factor default change, or any Phase 9B command.

### Mandatory stage protocol

Every master stage is a pair of substages: `E_k` is the declared execution and
`R_k` is the mandatory repair-and-refresh closeout.  The closeout is required
after a pass as well as after a failure.  The next execution cannot start until
`R_k` has written its receipt and refreshed the next subplan.

| Stage | Execution | Mandatory repair/refresh closeout | Entry to next stage |
|---|---|---|---|
| M0 | `E0` mechanics foundations | `R0`: verify identities, fixtures, and source closure; repair localized defects; refresh M1 | M0 receipt and M1 entry checks |
| M1 | `E1` calibration and C5 freeze | `R1`: classify calibration evidence, preserve failed arms, freeze only justified controls; refresh M2 | C5 receipt and M2 entry checks |
| M2 | `E2` fresh-chart/localized repair | `R2`: audit fresh seeds, hashes, numerical health, and claim boundaries; refresh M3 | localized result and reset memo |
| M3 | `E3` performance, canary, and six-scope replay | `R3`: recompute cost, repair localized defects, and refresh or stop the Phase 9B plan | M3 terminal result and a valid M4 plan, or a stated continuation blocker |
| M4-P0 | `E0-ER` executable-readiness checks | `R0-ER`: close runner, graph, route, provenance, budget, and runtime blockers; refresh P1 | P1 entry receipt and refreshed P1 audit |
| M4 | `E4` sequential posterior validation | `R4`: audit convergence, posterior, comparator, and uncertainty evidence; refresh M5 | M4 terminal result and a valid M5 plan |
| M5 | `E5` dimension/scaling ladder | `R5`: audit uncertainty and default decision; refresh the next research program | terminal decision or explicitly scoped continuation |

Subplans may contain finer-grained repair steps, but they may not omit the
stage closeout.  Each closeout records preserved attempts, failure
classification, focused regression, changed-file and artifact hashes, remaining
budget, refreshed assumptions/defaults, the next exact command, and the
condition that would constitute a real continuation blocker.

## Immediate next steps

These are the only current ordered actions. Closed M3/M3-C/M3P/M3Q work is
preserved in the execution ledger below and is not an active command.

1. **Complete the active M4-P1 attempt:** leave
   `bayesfilter-q20-phase9b-p1-r2.service` running under the original
   86,400-second ledger. Monitor committed chunk health, resource headroom
   and 18,000/28,800-second allocations; do not launch a duplicate or edit
   bound runtime sources. Phase 0 already passed for this launch.
2. **Close P1:** inspect both terminal receipts, settle measured costs and
   classify any failure before a localized repair/resume. Preserve successful
   siblings and all prior attempts. A mechanics pass does not admit P2.
3. **M4-P2 untouched sequential validation:** only after P1 closes and the
   chart-threshold receipt, fresh confirmation seeds, and complete uncertainty
   plan exist. P2 remains blocked now.
4. **M4-P3 downstream validation and closeout:** compare posterior/reference
   behavior, mode travel, and the declared comparator arms with uncertainty;
   do not infer superiority from descriptive diagnostics.
5. **M5 scaling and terminal review:** write a new reviewed plan only after the
   M4 closeout; no dimension/scaling command is currently authorized.

The detailed commands, artifact roots, caps, and repair triggers for the closed
M3-C/M3P/M3Q runs are in their subordinate plans and results. The latest
eight-sweep recovery/runtime closeout and P1 launch are recorded in the
eight-hour amendment result. A localized repair/resume remains within the
existing authorization when target, method, gates, hardware class and total
budget stay unchanged. It must preserve prior evidence, check current source
and placement, and use a fresh launch directory under the same ledger.
Do not reinitialize the campaign or rerun completed Phase 0 without cause.


### 2026-09-06 M4 P1 execution failure and repair refresh

- Three fresh P1 attempt roots are preserved under
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/`.
  The first two failed at chart-object indexing. The third repaired that defect,
  completed factor chart construction and tuning, and then failed because the
  runner passed a telemetry evaluator to the shared controller where a
  status-mapping summarizer was required. No sequential chunk archive, retained
  posterior stream, strict comparator result, or P1 pass exists.
- The failures are harness/infrastructure evidence, not evidence against the
  target, factor numerical-backend direction, or research question. The runner
  now uses the controller's standard status mapping, transposes shared samples
  before chain-major diagnostics, computes MCSE, records run-start/failure
  provenance, separates role/arm/attempt seed namespaces, and fails closed on a
  schedule forecast that cannot fit the remaining budget.
- The observed first sequential chunk took approximately 1,423 seconds after
  tuning. Across the three attempts, artifact timestamps imply approximately
  1,832.61 seconds consumed and approximately 3,367.39 nominal seconds remain;
  these are budget bookkeeping estimates, not scientific timing evidence. A
  focused CPU regression and fresh source-only P1 audit are the next valid
  actions. P2 remains blocked, and no material GPU launch is active.

The earlier September 5 ledger entries retain their historical date-boundary
wording. As of September 7, 2026, the September 6 factor result and reset memo
are past-dated provenance, not a temporal blocker. The current blockers are the
fresh placement/runtime measurement and complete-schedule reconciliation; the
completed diagnostic does not authorize spending beyond the campaign balance.
## Between-phase repair protocol (binding)

After each executed phase:

1. Preserve the attempt directory and never overwrite a prior receipt.
2. Run the smallest exact regression and inspect the manifest/schema/hash.
3. Classify the outcome as harness, implementation, numerical, tuning,
   resource, candidate, or scientific-evidence failure.
4. Decide whether the failure invalidates the harness or only rejects the
   current candidate. Continue to the declared repair when no continuation
   veto fired.
5. Make the smallest repair without changing frozen target, data, measure,
   correction, hardware class, privacy boundary, or total phase cap.
6. Use a new output directory and seed namespace for the repair; preserve the
   failed attempt as evidence.
7. Run focused tests and record the actual command, wall time, hashes, and
   remaining budget in a closeout receipt.
8. Refresh the next subplan's defaults, assumptions, evidence contract,
   commands, budget, and stop conditions from that receipt.
9. Advance only after the receipt and refreshed subplan exist and the next
   phase entry gate passes. A localized pass is not a whole-program pass.

High acceptance, poor whitening, low ESS, a missed mode, or a rejected arm is a
candidate repair trigger, not automatically a whole-program blocker. A true
blocker is limited to the continuation-veto conditions in the research ledger.

## Budget and execution boundary

The completed four-hour campaign is historical: its 36,000-second allocation
recorded 11,352.55558313601 seconds consumed and zero reserved. The active
amendment uses a fresh 86,400 aggregate GPU-worker-second budget and 28,800
seconds per arm. Its prepared ledger is empty; all new attempts must bind to its
source and plan hashes. The prior ledger and all older balances remain separate
and cannot be transferred.

The older 5,200-second readiness ledger has 5,016.200345 seconds consumed and
183.799655 remaining. The separate September 8 14,400-second tuning ledger has
844.217169 consumed and 13,555.782831 unused. Both remain unchanged; their
balances and the closed factor campaign's 11,800-second allocation are not
transferable. The older budgets below are historical, not active launch
authority. A local retry spends campaign budget, not an approval token.

The completed localized attempt-05 consumed `494.5689085649792` seconds and
used a 1,402,670,592-byte peak TensorFlow allocator reading on GPU0. Those
figures are evidence for the localized route only. The M3 campaign executed
one 1,800-second canary attempt and one permitted corrected retry; the latter
hit the declared continuation veto. The unexecuted 7,800-second full replay is
not authorized under this terminal state.

Routine document, CPU fixture, compile, and focused-test work remains allowed
within the current scope. The user's explicit instruction authorized the
unchanged M3 canary/retry campaign; that authorization was consumed by the
terminal closeout. A long GPU launch still crosses the platform's trusted GPU permission boundary, with
`TF_FORCE_GPU_ALLOW_GROWTH=true` set before TensorFlow import. Retries under
this unchanged contract remain within the campaign cap; a new cap, profile,
target, hardware class, or privacy boundary requires a refreshed authorization.
Broad shell, interpreter, package-manager, network, or arbitrary-GPU
permissions are not needed. The M3-C continuation had its own 900-second GPU
and 300-second CPU diagnostic budget and fresh artifact root; it did not inherit
the M3 replay cap or partial-call budget. It consumed `834.5439817190636` GPU
seconds across its two GPU attempts. The first attempt is quarantined for
cross-arm comparison because of an arm-dependent validation bank; the second,
repaired attempt is the sole valid N2 comparison. No remaining budget
authorizes a changed contract.

The subsequent M3P process-parallel campaign used a separate declared cap of
1,200 seconds for each canary and 14,400 seconds for the full staged run.  The
authoritative canary attempt-06 passed in `585.254545083968` seconds.  Full
attempt-05 consumed its cap after 48/48 screen and 26/32 selection records;
the repaired closeout classified this as `M3P_FULL_CAP_BLOCKED`.  No further
M3P run is authorized under the same task order and cap.  A new cap,
partition, or hardware contract requires a new reviewed subplan and explicit
campaign budget.

The historical M3Q plan declared a fresh 1,800-second canary and a
10,000-second full-replay material cap. Its result is closed subordinate
evidence; its pending strict canary is not an active next command.

## Skeptical master-program audit

The audit found four governance gaps and repaired them in this document:

| Finding | Repair |
|---|---|
| The 2026-08-25 particle-authority master was terminal for a different campaign | Mark it historical for this lineage and establish this dated master as the active authority. |
| The 2026-08-28 implementation plan contained phases but did not identify a single current master or explicit state transition after the localized repair | Add the phase map, ownership, current state, and entry/exit boundaries above. |
| The next replay draft initially lacked a concrete profile, launcher, and artifact root | The M3 subplan now binds `phase9a_full_replay_v1`, launcher interface, output root, per-call durability, and separate caps. |
| A localized mechanics pass could be misread as permission to run Phase 9B | The master keeps M4 blocked until six scope handoffs, sequential diagnostics, and a new Phase 9B plan pass. |
| Repair was described as guidance rather than a state transition | Add the binding `E_k -> R_k` protocol and require a closeout receipt before every next-stage entry. |

The audit also checked wrong baselines, proxy metrics, hidden cap widening,
stale seed reuse, output collisions, missing stop conditions, environment
mismatch, unexamined defaults, and whether a phase could advance without a
repair receipt. The result is
`PASS_MASTER_REPAIR_PROTOCOL_AND_M3_PROFILE_AUTHORIZATION`, followed by the
terminal `BLOCK_M3_CANARY_RESOURCE_VETO` and the M3P cap blocker recorded
below; it governs the recorded closeouts and keeps M4 closed.

## Execution ledger

### 2026-09-09 recovery/runtime terminal closeout

- Both same-GPU hard-interruption canaries pass exact resumed sample/trace
  equality. Both full runtime arms pass complete health for two actual
  500-transition calls; 34 checkpoint bundles and 372 tensors verify.
- The final CPU suite passes 152 tests. Both component profiles complete after
  the scalar-gap inspector repair. No campaign GPU worker remains.
- The four-hour ceiling and ten-GPU-hour allocation are active; final spending
  is 3.15349 GPU-hours. P0-I/J checks and P0-K localization are complete, while
  strict and aggregate reserve-inclusive affordability keep P0-L/P1 blocked.
- The result note records every failed attempt, source migration, recovery
  limitation, health qualification, budget and next cost-repair hypothesis:
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-result-2026-09-09.md`.

### 2026-09-09 eight-hour cap amendment authorization

- The owner approves 28,800 seconds per arm and 86,400 aggregate GPU-worker
  seconds for a fresh repair/remeasurement campaign.
- The four-hour campaign remains immutable historical evidence. A fresh ledger
  and output root are prepared; no GPU work had started at authorization.
- Execution audit removes mandatory status reuse: the unchanged full-status
  kernel is retained. Actual parallel P1/checkpoint integration and initialization
  repair pass 128 CPU tests; fresh recovery and reserve-inclusive timing remain
  necessary before Phase 0 closes and P1 automatically starts.

### 2026-09-02 master refresh

- Created this active master program and linked the current parent, M2 result,
  reset memo, and M3 draft.
- Preserved the historical particle-authority master and added a successor
  pointer rather than rewriting its phase evidence.
- Confirmed the current M2 attempt-05 manifest hash and fresh-seed audit.
- Verified the M3 draft's concrete execution envelope and corrected its stale
  `7,200`-second reference to `7,800` seconds.
- Focused tests: 10 Phase 9A repair tests and 18 HMC-policy tests passed;
  documentation rendering, Python compilation, shell syntax, whitespace, and
  independent attempt-05 manifest/hash checks passed.
- Amended the program so every master stage has a mandatory repair/refresh
  closeout, including passed stages. Added separate canary/full replay profile
  identities, fresh seed namespaces, profile-bound material caps, and a
  source-owned launcher. The user's latest instruction supplies the plain-
  language authorization for the unchanged M3 campaign; platform GPU trust is
  still requested at launch.

### 2026-09-02 R0/R1 repair closeout

- R0 immutable evidence audit completed and recorded in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-audit-result-2026-09-02.md`.
- R1 added compile/steady-run telemetry, disjoint canary/full profiles, and a
  profile-bound-cap launcher. Focused tests and analytic mechanics checks
  passed; the adjacent ordinary-tuner oracle failure was classified as
  pre-existing budget/R-hat migration debt, not an M3 route failure.
- Mandatory closeout receipt exists in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r0-reset-memo-2026-09-02.md`.
- R2 attempt-01 exposed an invalid literal-brace launcher root and a bounded
  resource failure; it was preserved, not promoted, and classified in
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r2a-repair-result-2026-09-02.md`.
- R2a repaired the root contract and added a `13 passed` regression receipt;
  the corrected retry then reproduced the bounded resource failure. The
  terminal M3 result and reset memo are now recorded, and no third retry is
  authorized. M4 remains blocked.

### 2026-09-02 M3 terminal repair/refresh closeout

- Attempt-02 is preserved under the corrected source-owned root. It matched the
  frozen identities and completed 21 calls in `1645.967775` seconds before the
  fixed `1800` second cap terminated the next call; outer wall time was
  `1852.926691` seconds including termination grace.
- The second bounded resource failure after R1 fired
  `BLOCK_M3_CANARY_RESOURCE_VETO`. The terminal result and reset memo include
  the evidence contract, cost decomposition, decision/inference tables,
  uncertainty limits, and red-team analysis.
- The full six-scope replay and Phase 9B remain blocked. The original M3 replay
  program has no valid GPU command. Any continuation must start with a new reviewed
  plan that explicitly changes the performance design or budget; it may not
  widen the cap or reuse partial calls silently.

### 2026-09-02 M3-C authority refresh

- The completed bounded diagnostic in
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-repair-result-2026-09-02.md`
  is incorporated as subordinate evidence. It measured target batching,
  repaired a diagnostic graph-construction defect, validated affine and
  finite-difference score checks, and left grouped-HMC integration unadmitted.
- The continuation authority was
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-plan-2026-09-02.md`.
  Its N0--N4 phases have now closed under
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`.
- M3 remains terminal for the original replay. M4/Phase 9B remains blocked.
  The M3-C diagnostics named by the continuation plan are now closed; no new
  continuation command is valid without a new dated subordinate plan.

The M3-C refresh was audited before execution: it preserves the frozen target
signature, separates exact-equivalence gates from descriptive timing and loss,
declares fresh seeds/data partitions, rejects validation leakage, bounds GPU/CPU
cost, and states that TFP batch RNG semantics may make the fast grouped path
unadmittable. Audit verdict: `PASS_M3C_BOUNDED_CONTINUATION`.

### 2026-09-02 M3-C execution and repair closeout

- N1's fresh CPU and GPU receipts agree: the scalar and explicit row-loop
  controls are exactly equivalent at the declared tolerance, while the fast
  grouped TFP transition differs in state, target, gradient, and
  log-acceptance. The grouped integration veto is therefore active.
- The first GPU ladder attempt is preserved but quarantined because its
  arm-dependent validation bank made cross-arm comparison invalid. The harness
  was repaired to share validation seeds `98000`, `98100`, and `98200` by seed
  index, and the second GPU attempt is the sole valid N2 comparison.
- The repaired ladder completed 9/9 finite candidates with 12/12 valid updates
  each. Arms A, B, and C each had 0/3 seeds meeting the provisional 10%
  score-RMS nomination threshold. No candidate was nominated; no default or
  active route changed.
- The route-specific regression passed `38 passed` across the tempered
  lineage/ensemble and fixed-transport step-cap tests. A broader five-file
  focused suite returned `113 passed, 2 failed`; the two failures are known
  ordinary-tuner migration debt and an absent private LGSSM fixture outside
  M3-C. They are recorded, not hidden, and do not satisfy a claim of a green
  repository-wide suite.
- The authoritative result and reset memo are
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-result-2026-09-02.md`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-performance-whitening-next-reset-memo-2026-09-02.md`.
  The execution-time/current-document hash reconciliation is in
  `docs/plans/artifacts/ssl-lstm-q20-performance-whitening-next-2026-09-02/n4-r4-closeout.json`.
  M3 remains terminal, M3-C is closed, N3 is blocked, and M4/Phase 9B remain
  closed. Any further work requires a new reviewed subordinate plan.

## Required closeout artifacts

Every future serious phase must leave a result note and reset memo beside its
subplan, plus a versioned manifest under
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-<date>/`.
The terminal note must include decision and inference-status tables, a
post-run red team, uncertainty limits, and a precise next action or real
blocker. The M3 terminal result and reset memo now exist; M4 remains closed
until a new reviewed plan changes the performance design or budget.

### 2026-09-03 M3P 72-core execution and terminal closeout

- The staged process topology was implemented as three sequential barriers:
  screen `8x4` (32 worker cores), selection `2x8` (16), and scope finalization
  `6x4` (24), for a 72-core worker budget. CPU IDs were discovered from the
  controller affinity set; workers hid CUDA before TensorFlow import and
  enabled XLA.
- Canary attempt-05 passed before the full run. Full attempt-05 completed all
  48 screen records and 26/32 selection records before the declared 14,400
  second cap; both selection streams were still in long candidates and scope
  finalization did not start. The completed selection records are descriptive
  only and do not nominate a candidate.
- The first cap closeout exposed a harness defect: unset child return codes
  were reported as a generic worker failure and no typed partial summary was
  written. The localized repair added deadline receipts and non-throwing
  partial coverage. Focused tests passed (`9 passed`), and canary attempt-06
  passed in `585.254545083968` seconds after the repair.
- The authoritative M3P result and reset memo are
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-reset-memo-2026-09-03.md`.
  The canary receipt is
  `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-canary-attempt-06-result-2026-09-03.md`.
  The subordinate plan is closed as `P3_FULL_CAP_BLOCKED`; no same-contract
  relaunch, cap increase, task repartition, or Phase 9B command is authorized.

### 2026-09-04 M3Q performance-design refresh

- The user requested a 10,000-second GPU budget and asked whether batching and
  strict-path eigendecomposition reuse can reduce the q=20 replay cost.
- Source inspection found that the four HMC chains and target value/score are
  already batched over a static leading axis; the rejected grouped-HMC
  prototype remains unadmitted because its state/RNG semantics differed.
- The batched strict score graph has six `SelfAdjointEigV2` nodes in its time
  loop (three for 80x80 placement and three for the 1x1 innovation path).  The
  opt-in raw-basis `tensorflow_eigh_strict_cached` backend is descriptively
  about 2.9x faster at batch 4 but fails q=20 GPU score parity.  The narrower
  `tensorflow_eigh_strict_factor_cached` backend reduces the graph to four
  nodes, is about 1.5x faster, and passes the current center/varied fixtures;
  neither route is a default or claim-bearing path.
- A fresh M3Q plan, profile pair, launcher, benchmark, and seed/output roots
  were added.  Focused tests and diagnostics completed; raw reuse is vetoed,
  and the strict-baseline canary remains the next gated replay action.  The
  later M3R branch supplies the bounded strict-factor mechanics check; a fresh
  factor tuning campaign is still required, and Phase 9B remains blocked.

### 2026-09-04 M3R factor-route promotion mechanics closeout

- The factor-route promotion plan repaired the missing component backend
  identity in `GaussianLikelihoodBridge` and `FixedBetaBridgeAdapter`, added
  stale strict-handoff rejection, and repaired the P1 candidate selector so the
  raw-cache veto cannot mask the factor result.
- Fresh CPU and GPU0/XLA q=20 center/varied fixtures passed value, score,
  status, and row-class parity.  Paired beta-0.5/beta-1 HMC transitions and a
  four-chain short mechanics run passed the frozen finite energy-error screen;
  native TFP divergence telemetry was unavailable and is recorded as such.
- The standalone identity receipt proves stale strict-to-factor rejection; four
  measured-grid tuner regressions separately prove valid same-scope handoff
  construction and scope/transport substitution rejection.
- The factor branch is `DEFAULT_ELIGIBLE_PENDING_FRESH_TUNING`; strict remains
  the default.  The authoritative result and reset memo are
  `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-result-2026-09-04.md`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-reset-memo-2026-09-04.md`.
  The only next path to a default decision is the fresh factor tuning plan;
  Phase 9B remains blocked.

### 2026-09-04 M3S factor fresh-tuning canary closeout

- A0 source binding passed 19 runner/launcher tests, 48 bridge/eigensystem
  tests, Python compilation, shell syntax, and the expected strict-artifact
  rejection probe. The factor backend is now bound in profile, scope,
  checkpoint, bridge, handoff, and manifest identities.
- The fresh factor canary used profile
  `phase9a_factor_tuning_canary_v1`, scope index 3 only, and a new seed
  namespace. It completed all eight measured grid pairs and held-out
  verification in 1,511.658 seconds under the 1,800-second cap. The
  independent auditor passed; peak allocator use was 1,402,668,544 bytes.
- R1 opens the six-scope factor profile. Phase 9B remains blocked until the
  full run and a separately refreshed posterior-validation plan pass.

### 2026-09-05 M3S A2/R2 full-scope failure and repair refresh

- The first six-scope factor attempt completed scope 0 and then failed at
  `chart-0, beta=0.5` after 1,617.6309049129486 seconds. All eight declared
  pairs were measured; finite calls had nonfinite proposed target values and
  no chain movement, while several call receipts were lost when raw NaN/Inf
  diagnostics were serialized with `allow_nan=False`.
- The failure is preserved as
  `docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/full-attempt-20260904T185852Z/a2-r2-failure-closeout.json`.
  It is candidate/numerical plus harness serialization evidence, not a target,
  identity, GPU, or research-direction blocker. The partial scope-0 handoff
  is not promotable and will not be merged with a later attempt.
- R2 repairs the artifact boundary, runs a strict-versus-factor diagnostic on
  the frozen failed-scope chart, and permits one fresh six-scope retry only if
  that diagnostic and focused regression pass within the remaining aggregate
  budget. Phase 9B stays blocked; no posterior claim is made.

### 2026-09-05 M4 Phase 9B plan review and P0 authorization

- A new Phase 9B sequential-validation plan was written at
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-plan-2026-09-05.md`.
  Its research question, target/comparator identities, sequential warmup and
  retained-sampling policy, chart-quality decision, uncertainty requirements,
  pre-mortem, stop conditions, and artifact contract were reviewed before any
  new execution.
- The plan passed the skeptical review as
  `PASS_P0_PLAN_REVIEW_WITH_PHASE9B_CLAIM_RUN_BLOCKED_BY_DATE_BOUNDARY`.
  The September 6 factor result/reset memo are future-dated relative to this
  September 5 refresh; they remain supplied provenance, not current-session
  admission authority.
- The only authorized execution is the non-HMC P0 source/authority preflight:
  `docs/benchmarks/audit_ssl_lstm_q20_phase9b_plan_2026_09_05.py`. It must
  verify the source-synchronized receipt, all six scope artifacts, supported
  controller/tuner markers, and the dirty Git/source boundary in a fresh
  versioned output root. No Phase 9B HMC command is authorized yet.

### 2026-09-05 M4 P0 execution closeout

- P0 completed with `PASS_PHASE9B_P0_SOURCE_PREFLIGHT` and no failures. The
  receipt verifies the source-synchronized manifest/audit hashes, six scope
  checkpoint/tuning pairs, target/backend identities, and current sequential
  controller/tuner markers.
- P0 was deliberately non-HMC and produced no posterior draws or convergence
  evidence. The result note is
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-result-2026-09-05.md`.
- The September 6 factor result/reset memo remain future-dated provenance rather
  than current-session authority. The current-dated reconciliation note and
  source-synchronized receipt preserve the narrow factor candidate boundary.

### 2026-09-05 M4 P1 canary plan review

- The preserved Phase 9B plan was reviewed for date consistency and narrowed to
  a current-dated P1 plan that tests one cold representative scope for both the
  factor candidate and a fresh strict comparator.
- The P1 plan uses fresh scope-bound tuning, four-chain sequential warmup and
  retained sampling, explicit health diagnostics, separate seed namespaces,
  and a fresh output root. It is deliberately not a full ensemble or
  replica-exchange claim.
- The source-only P1 audit is
  `docs/benchmarks/audit_ssl_lstm_q20_phase9b_p1_canary_plan_2026_09_05.py`.
  Its pass opens only the GPU canary command in the active P1 plan; P2 remains
  blocked by chart-threshold and downstream posterior gates.

### 2026-09-06 M4 P1 execution failure and repair refresh

- Three fresh P1 attempt roots are preserved under
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/`.
  The first two failed at chart-object indexing. The third repaired that defect,
  completed factor chart construction and tuning, and then failed because the
  runner passed a telemetry evaluator to the shared controller where a
  status-mapping summarizer was required. No sequential chunk archive, retained
  posterior stream, strict comparator result, or P1 pass exists.
- The failures are harness/infrastructure evidence, not evidence against the
  target, factor numerical-backend direction, or research question. The runner
  now uses the controller's standard status mapping, transposes shared samples
  before chain-major diagnostics, computes MCSE, records run-start/failure
  provenance, separates role/arm/attempt seed namespaces, and fails closed on a
  schedule forecast that cannot fit the remaining budget.
- The observed first sequential chunk took approximately 1,423 seconds after
  tuning. Across the three attempts, artifact timestamps imply approximately
  1,832.61 seconds consumed and approximately 3,367.39 nominal seconds remain;
  these are budget bookkeeping estimates, not scientific timing evidence. A
  focused CPU regression and fresh source-only P1 audit are the next valid
  actions. P2 remains blocked, and no material GPU launch is active.

### 2026-09-06 M4-P0 executable-readiness activation

- The P1 repair result confirms that the runner-level defects were localized,
  but no complete sequential arm or strict comparator result exists. The first
  observed sequential chunk is approximately `1,423` seconds, so the old P1
  schedule and `2,600`-second arm cap are not executable on the current forecast.
- A skeptical review of the Phase 9B plan found that the unresolved work is a
  technical entry gate rather than a new scientific phase: the shared
  controller has no explicit stable `input_signature`, the NeuTra route audit
  has new and legacy unclassified paths, aggregate budget accounting is not
  durable, and a complete post-repair GPU provenance receipt is missing.
- `M4-P0` is now the active gate under
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`.
  It must close the runner contract, controller graph contract, route
  classification, source/seed/artifact provenance, compile-versus-steady-state
  budget, campaign ledger, and GPU memory/XLA/TF32/allocator preflight before
  P1 can be launched.
- The prior Phase 9B source-authority P0 remains valid and is explicitly
  preserved as a prerequisite, not relabeled as executable readiness. P2 stays
  blocked by chart thresholds, downstream posterior/reference checks, and
  uncertainty evidence.
- Initial non-GPU Phase 0 checks passed for the six-test P1 runner suite,
  Python compilation, `git diff --check`, and the rotated P1 source-only audit
  `r4`. The full NeuTra route-policy test was then repaired and passed after
  explicit classification of the new and legacy paths.

### 2026-09-06 M4-P0 static repair and GPU ownership block

- The shared controller now has an explicit fixed state/seed input signature,
  with a repeated-shape single-trace regression. The durable campaign ledger
  and source-owned two-arm compile/steady-state diagnostic are implemented.
- The fresh P1 source-only audit `r5` passes, and the focused static suite
  passes with `29 passed`; the full route-policy suite passes with `6 passed`.
- The trusted TensorFlow GPU probe failed closed with
  `requested_gpu_compute_busy:0`. `nvidia-smi` identified the existing GPU 0
  compute process as `/usr/NX/bin/nxnode.bin`, PID `6826`, using 312 MiB. No
  BayesFilter GPU workload was started, no process was stopped, and GPU 1 was
  not substituted.
- M4-P0 is now static-repair complete but runtime-preflight blocked. The exact
  blocked receipt and result note are
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/gpu-preflight-blocked-20260906/run_manifest.json`
  and
  `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-result-2026-09-06.md`.

### 2026-09-07 M4-P0 resumption and dual-blocker correction

- The skeptical audit repaired the diagnostic/closeout dependency cycle and
  enforced M4-P0 at the P1 launcher. It also repaired missing forecast inputs,
  per-retry budget renewal, and incomplete whole-attempt accounting. The
  diagnostic now preserves environment/trust provenance and rejects a
  stationary chain on either compiled call.
- Source audit `r12` passed with no findings. The combined CPU suite passed
  with `72 passed, 191 warnings in 8.60s`, and the full route-policy suite
  passed with `6 passed in 2.10s`. Python compilation and `git diff --check`
  passed. The Phase 0 result records commands and durable validation logs;
  GPU devices were intentionally hidden for these engineering checks.
- The trusted probe at `2026-09-07T07:45:41Z` returned
  `requested_gpu_compute_busy:0`; its receipt is
  `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/resumption-20260907/gpu-probe.json`.
- GPU release alone does not authorize the diagnostic. The shared-budget
  decision must cover measurement and complete fresh P1 work. No GPU workload
  was launched during this remediation, no passing M4-P0 closeout exists, and
  P1/P2 remain blocked. The scientific direction and prior narrow backend
  admission are unchanged.
