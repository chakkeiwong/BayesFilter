# Literature-prior preparation and adaptive-HMC continuation

2026-10-15, 09:58 UTC checkpoint. The goal is to test surrogate-force HMC on
the five likelihood models using the adopted scientific priors. All five fresh
filter-tuning and short-mechanics phases passed. Adaptive HMC is now running
on KSC; there were no new posterior/convergence results at that checkpoint.

## Current update — 2026-10-15, 15:36 UTC

KSC remains the only completed sampler result. Range bearing returned
HEALTH_VETO after 250 warmup transitions per chain, with zero retained draws.
Its nomination passed, but the first warmup chunk failed no_force_fallback.
Finiteness, endpoint validity, energy, all-chain movement, final-state matching
and no-divergences checks passed. The fallback count and underlying cause
have not yet been diagnosed. This is a failure of the current attempt, not
evidence against surrogate HMC generally.

The master continued automatically to LGSSM. Worker PID654997 and master
PID493895 were live at inspection. LGSSM's log records a 22m47.5s XLA
compilation; no fresh-sequence progress file exists yet. Predator–prey and SIR
remain queued. No running code or numerical settings were changed.

Range bearing charged6020.5272 worker-wall seconds. Total charged52896.3695s,
remaining33503.6305s before the active LGSSM worker's final charge, attempts
142/145. LGSSM's reservation32313.8646s is a ceiling, not elapsed consumption.
Evidence: the live master.json, range_bearing/inference/hmc.json and its
public-tuning/fresh-progress.json under the literature-prior-hmc-01 run root.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Reject current range-bearing attempt | No retained sample; convergence not assessed | Surrogate-force fallback during warmup | Fallback cause and frequency not checked | Diagnose saved failure before a fresh bounded retry | Does not reject surrogate HMC or establish posterior failure |
| Continue existing master | Five-model campaign incomplete | No campaign-wide continuation veto observed | Remaining model cost and validity | Monitor LGSSM; then run queued models within budget | Queued models are not completed results |

Inference status: the range-bearing hard health veto is supported; no
cross-model or method ranking is supported. KSC's previously reported sampler
diagnostics remain descriptive evidence for its finite-grid endpoint, with
exact-model agreement unchecked. Default-readiness remains unestablished.
The next evidence is the fallback diagnosis and the remaining sampler results.

## Earlier sampling update — 2026-10-15, 12:21 UTC

KSC has completed the declared adaptive sampler checks. Range bearing is
running its nomination/XLA compilation; LGSSM, predator–prey and SIR remain
queued. The same master continues automatically, with no separate validation
chain or manual restart between models.

| KSC diagnostic | Saved result |
|---|---|
| Chains | 4 |
| Excluded warmup per chain | 2000 |
| Retained inference draws per chain | 1000; 4000 total |
| Maximum rank-normalized split/folded R-hat | 1.0059978914 |
| Acceptance by chain | 69.8%, 72.6%, 68.5%, 69.3% |
| Minimum bulk / tail ESS | 693.9 / 639.5 |
| Divergences / force fallbacks | 0 / 0 |
| Mean MCSE / posterior SD by coordinate | 3.84%, 3.81% |

The retained sample met the cumulative R-hat criterion at its first assessment
of 1000 draws per chain. Warmup was checked adaptively and excluded from these
estimates. The saved endpoint is the fixed 1201-node KSC grid; its refinement
over the posterior region remains required. `posterior_agreement` is
`not_checked`. Passing the sampler checks establishes neither exact-model
posterior agreement nor default-readiness.

Evidence: `../surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-hmc-01/ksc/inference/hmc.json`.
The master records KSC return code 0 and charges 6665.46 worker-wall seconds.
The cumulative ledger has charged 46875.84 of 86400 seconds; 39524.16 seconds
remain before charging the active range-bearing worker. Its 32400-second
reservation is an upper bound, not elapsed consumption. Attempts: 141/145.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Accept KSC sampler convergence for its saved endpoint | Cumulative modern R-hat <=1.01 passed | Zero divergences/fallbacks; saved status VERIFIED | Finite-grid posterior accuracy | Preserve draws and assess endpoint refinement over the sampled region | Exact-model agreement is not checked |
| Continue the existing master | Four prepared models remain | No campaign continuation veto observed | Remaining runtime and model-specific sampler health | Continue range bearing, then queued models within the same cap | No promise that all five fit the remaining budget |

| Inference status | Finding |
|---|---|
| Hard veto screen | KSC reported no divergences or force fallbacks and passed its declared sampler checks |
| Statistically supported ranking | None; no cross-method ranking is inferred |
| Descriptive-only differences | KSC acceptance, ESS, MCSE and runtime above |
| Default-readiness | Not established; independent posterior agreement remains pending |
| Next evidence needed | Remaining models' adaptive results and posterior-region adequacy of approximate endpoints |

The strongest alternative explanation for apparently successful nonlinear
inference is convergence to a biased finite endpoint. Reference refinement or
posterior disagreement could overturn an exact-model interpretation; that
interpretation is not made here. Later sections retain the preparation-stage
record and its earlier decisions.

## What was checked

The preparation launcher can now run an explicit model subset, so finishing
LGSSM does not force its repetition. The inference master consumes the real
nested advance output, checks the saved selection and its source, data, prior
and precision identity, and requires successful endpoint replay and mechanics
on that selection. It accepts only the already measured SIR mixed-precision
exception. It charges every worker, including failed launches and timeouts, to
the original campaign ledger and refreshes the next-phase file. A model health
veto does not prevent the other prepared models from running.

CPU-only checks: the integrated suite passed 186 tests in 21.89 seconds
(`continuation-tests-01.log`). The final changed-file regression passed 21 tests
in 0.35 seconds (`continuation-tests-03.log`), including two added error/continuation
cases. The real saved files then passed `continuation-handoff-check.py` for all
five models (`continuation-handoff-check-01.json`). No GPU was used for these
engineering checks. The command and elapsed time of the real handoff check are
in its JSON; the GPU phase manifests preserve their actual commands and device.

The consumer path was inspected through `continue_advance -> stage_worker ->
restore_selected_scope -> inference -> tune_hmc_kernel ->
_run_verified_tensorflow_tuning -> _fresh_sequence ->
run_sequential_verification`. Existing runtime tests exercise scope restoration,
prior/force/endpoint wiring, the actual adaptive configuration, and the public
tuner/result handoff. The new tests exercise orchestration and artifact/budget
failures. The full posterior execution remains ongoing; mocked worker tests do
not establish numerical sampler validity.

## Fresh mechanics results

All runs use N4032, K2016, TF32 off, literature priors and fresh tuning seed
offset 200. Horizon is 50 except range bearing at 20. Selected OT controls are
epsilon1/five steps; the real handoff JSON retains all the other selected
controls. These are **two transitions per chain**, four chains, L1, step size
0.002, so acceptance below is a short mechanics observation only.

| Model | Arithmetic | Two-step acceptance by chain | Divergences / force fallbacks | Warmed chunk seconds |
|---|---|---|---|---:|
| LGSSM | FP32 | 1, 1, 1, 1 | 0 / 0 | 92.035 |
| KSC SV | FP32 | 1, 1, 1, 1 | 0 / 0 | 4.814 |
| Range bearing | FP32 | 1, 1, 1, 1 | 0 / 0 | 9.190 |
| Predator–prey | FP32 | 0.5, 1, 1, 1 | 0 / 0 | 21.970 |
| SIR | FP64 recurrence, FP32 pairs | 0.5, 1, 0.5, 0.5 | 0 / 0 | 60.305 |

Every inspected cold/warm chunk was finite and passed its health checks.
The timings include initial force work and are not long-chain rate estimates.
LGSSM ran on RTX4080 SUPER because CUDA and nvidia-smi ordinals differ; the
other four ran on RTX5080 selected by UUID. Shared-machine load and differing
devices further prevent interpreting this table as a controlled speed ranking.
The 69-input LGSSM exact-Kalman support check also passed; broad prior support
for the nonlinear approximations remains unestablished.

## Ongoing execution

Fresh preparation charged 1,659.507 seconds after the prior-adoption checkpoint.
Before inference, the original 86,400-second ledger had charged 40,210.384
seconds, leaving 46,189.616 seconds (12.83 hours). The explicit plan revision
raises the operational attempt ceiling from 140 to 145 to accommodate the
five inference workers and one repair slot; it does not increase compute.
See `attempt-cap-revision-01.json` for that change.

`scripts/run_surrogate_hmc_five_model_master.py continue-advance` is running
with a 45,000-second phase cap and 32,400-second per-model cap, always clipped
to the original campaign remainder. Order: KSC, range bearing, LGSSM,
predator–prey, SIR. GPU UUID is
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a` (RTX5080). The selected KSC child
reports FP32, TF32 disabled, and GPU memory growth enabled. Its numerical
configuration requests XLA. A logged compiled subcomponent alone is not
evidence that the entire nomination has returned successfully.

Run root:
`../surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-hmc-01/`.
Its `master.json`, per-model `inference/manifest.json`, `hmc-config.json`,
public-tuner artifacts and `next-phase-plan.json` are the live authorities.
The exact command is recorded in the phase plan and run manifest. Tool session
37424 supervises the master. No additional approval or manual phase restart is
required while that process is healthy and the declared budget remains.

Warmup readiness uses a recent window; retained R-hat uses all retained draws
cumulatively. Retained sampling stops at modern R-hat <=1.01 or 20,000 per
chain, subject to health and budget limits. All post-warmup draws are inference
draws. There is no separate terminal validation chain. A budget stop preserves
completed sequential chunks; nomination itself is one numerical call and is
durable only after it returns.

## Decisions and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Continue fresh selections | Actual scope/handoff checks passed | No inspected endpoint/mechanics veto | Behavior beyond inspected inputs | Target-specific HMC tuning and adaptive sampling | Short mechanics does not establish convergence |
| Continue prepared models automatically | Cumulative ledger and failure tests passed | Budget, invalid evidence and model health remain enforced | Time to readiness and retained convergence | Let the current master execute | All five may not fit the remaining budget |
| Keep scientific conclusions pending | No retained result yet | Posterior/reference diagnostics pending | Nonlinear endpoint approximation over posterior region | Inspect retained diagnostics and refine references | No exact nonlinear-posterior or default claim |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical veto on inspected fresh preparation inputs; engineering failure cases rejected in tests |
| Statistically supported ranking | None; all five remain eligible to attempt sampling |
| Descriptive-only differences | Two-step acceptance and timings above |
| Default-readiness | Not established by this work; owner-selected FP32/N4032 policy remains in force |
| Next evidence needed | Adaptive retained samples, cumulative diagnostics, and posterior-region reference adequacy |

The strongest alternative explanation for a successful mechanics result is
simply that tiny steps stayed near easy initial points. A later finite-value,
energy, fallback, movement or convergence failure would reject the current
sampler candidate and trigger the planned repair. The weakest current evidence
is nonlinear reference accuracy away from nominal inputs. Convergence to the
declared grid/BPF/importance endpoint would still require that independent
scientific check; it would not by itself prove the exact model posterior.
