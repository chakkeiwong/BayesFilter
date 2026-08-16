# SSL-LSTM q=20 physical annealed-SMC material plan (2026-08-10)

Status: `COMPLETE; ALL MATERIAL GATES PASSED`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does adaptive globally resampled SMC give stable finite-sample relative mass evidence for the two known SSL-LSTM sign regions across independent runs and a conditional-ESS sensitivity arm? |
| Candidate | The passed canary algorithm: global systematic resampling at every nonterminal adaptive stage, one fixed HMC move (`0.03`, `L=4`), terminal beta-1 weights measured before resampling, batch-4 CPU/XLA exact targets. |
| Expected failure | Independent mode-mass estimates may vary; terminal weights or root ancestry may collapse; cESS `0.70` and `0.80` may disagree; local HMC may remain sign-trapped. |
| Promotion criterion | Every run reaches beta 1 with finite valid targets; each central run has terminal ESS fraction `>=0.50`, maximum weight `<=0.05`, unique-root fraction `>=0.30`, and at least 10 positive and 10 negative initial roots; eight-central-batch interval half-width `<=0.08`; pooled cESS `0.70`/`0.80` sign-mass difference `<=0.08`; all receipts verify. |
| Promotion veto | Any target/status/receipt invalidity, beta/stage failure, terminal weight failure, ancestry collapse, interval failure, or cESS sensitivity failure. |
| Continuation veto | Harness/target invalidity after one localized repair, source identity drift, or `4,200 s` material cap.  A valid candidate failure triggers a reviewed SMC repair rather than posterior issuance. |
| Repair trigger | Weight/sensitivity failure triggers higher cESS or more particles; ancestry failure triggers more particles or less frequent diversity-aware resampling; zero local movement remains an explanatory trigger for a stronger mutation kernel. |
| Explanatory diagnostics | Beta paths, stages, acceptance, sign changes, log normalizers, runtimes, and per-batch mass estimates. |
| Must not be concluded | Exhaustive mode discovery, full-posterior correctness, HMC stationarity, NeuTra repair, predictive validity, or default readiness. |

## Evidence contract

The baseline is failed sparse AIS at
`docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.json`.
The candidate comparator is the passed SMC canary at
`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/canary.json`.

Eight independent cESS `0.80` runs provide the central uncertainty design.  Two
independent cESS `0.70` runs provide a predeclared schedule-control arm.  Terminal
weighted sign mass is computed from each run's beta-1 pre-resampling signs and
weights.  No joint test or candidate ranking is used.

Material output root:
`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/`.

Exact material command/environment:

```bash
systemd-run --user \
  --unit=bayesfilter-ssl-q20-physical-annealed-smc-material-20260810-r2 \
  --collect \
  --property=WorkingDirectory=/home/ubuntu/python/BayesFilter \
  --property=CPUQuota=10000% \
  --property=RuntimeMaxSec=4200s \
  --property=TimeoutStopSec=60s \
  --property=StandardOutput=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/material.log \
  --property=StandardError=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/material.log \
  --setenv=CUDA_VISIBLE_DEVICES=-1 \
  --setenv=TF_CPP_MIN_LOG_LEVEL=3 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=NUMEXPR_NUM_THREADS=4 \
  --setenv=TF_NUM_INTRAOP_THREADS=4 \
  --setenv=TF_NUM_INTEROP_THREADS=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_material_2026_08_10.py
```

## Numeric/default audit

| Choice | Provenance/status | Justification | Failure mode / diagnostic |
|---|---|---|---|
| Eight central 100-particle runs | Inherited reviewed independent-batch precision design; hypothesis | Gives a t interval and 800 total initial roots without increasing per-wave resource use | Eight batches may miss extreme tails; interval and per-batch gates |
| Two cESS `0.70` runs | Sensitivity hypothesis distinct from central `0.80` | Materially changes beta increments while keeping algorithm and particle count fixed | Two runs are descriptive; only predeclared pooled difference is a veto |
| Terminal ESS `>=0.50` | Reviewed reliability hypothesis, below canary `0.938` and equal to TFP experimental default scale | Requires terminal measure to retain at least half-population effective support | Does not prove unseen tails |
| Maximum terminal weight `<=0.05` | Reviewed reliability hypothesis; canary `0.0229` | Prevents one particle from carrying more than 5% in any central run | Does not prove proposal completeness |
| Unique-root fraction `>=0.30` | Reviewed diversity hypothesis, below canary `0.65` | Detects severe genealogy collapse after repeated resampling | Roots may be locally redundant |
| At least 10 roots from each sign | Reviewed 10%-population floor, below canary 32/33 | Prevents one known region surviving through only a token lineage | Sign regions may not exhaust all modes |
| Interval half-width `<=0.08` | Same reviewed precision target as AIS | Tests independent-run Monte Carlo stability | Not an equivalence margin |
| cESS mass difference `<=0.08` | Same reviewed schedule-stability target as AIS | Directly tests sensitivity to a material temperature-placement change | Passing two cESS values does not prove universal stability |
| `4,200 s` cap | Derived from 10 times measured `211 s` canary plus near-2x orchestration margin | Bounded serious campaign | Fresh load or contention may consume margin; child and service timeouts |

## Execution plan

1. Parameterize only output root, seed domain, cESS, and plan/result bindings of the
   passed canary runner.  Preserve target, chart, resampling, HMC, terminal, XLA,
   worker, and stage policies.
2. Add CLI/harness tests for safe versioned output, seed-domain separation, and
   unchanged default canary behavior.
3. Run eight central and two sensitivity child runs sequentially under one detached
   supervisor.  Each child uses a unique output subdirectory and seed offset; write
   atomic supervisor progress after every child.
4. Aggregate only from verified beta-1 stage receipts.  Compute per-batch mass,
   central interval, pooled central/sensitivity measures, terminal weight and ancestry
   gates, and complete run manifest.
5. If all gates pass, classify SMC as a viable two-known-region weight authority,
   not a full posterior.  The next separate gate remains physical replica-exchange
   round trips/mixing and explicit mode-coverage limits.

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | Direct IS, sparse AIS, and passed SMC canary are explicit. |
| Proxy promoted | Acceptance, beta count, log normalizer, runtime, and sign-changing HMC are explanatory only. |
| One canary promoted | Eight independent central runs plus cESS control provide material evidence. |
| Resampled terminal erases weights | Child runner preserves beta-1 pre-resampling weights and never terminal-resamples. |
| Seed overlap | Each child receives a disjoint offset of 10,000; initialization, resampling, and mutation domains are already internally disjoint. |
| Root diversity mistaken for mode discovery | Root/sign gates cover only the two constructed proposal regions; exhaustive discovery remains a nonclaim. |
| Candidate rejection stops direction | Only target/harness corruption stops the campaign; valid reliability failure triggers the declared SMC repair. |
| Stream loss | Detached supervisor, child timeouts, versioned child artifacts, atomic progress, and overwrite refusal. |

The audit passes for two-known-region weight evidence.  It does not authorize a
posterior archive or predictive validation even if every gate passes.

## Pre-mortem

The campaign can pass because the proposal explicitly seeds both known regions while
still missing a third region.  It can also produce stable mass estimates with local
HMC never crossing signs.  These limit the claim to finite weighting over supported
known regions and keep the global-transition and mode-coverage gates separate.
