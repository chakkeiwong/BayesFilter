# M9 finite startup and rejected-interval repair

Status: M9 complete, closed September 20 local time. The active contract and remaining research program
are in [the master](bayesfilter-hmc-repair-master-program-2026-09-16.md).
Outputs are under `artifacts/hmc-repair-master-2026-09-16/m9-r1/` (R below).
No numerical default, posterior policy or candidate-ranking rule was changed.

The code now exposes a finite startup round budget through the ordinary public
config and optional exploration between valid directional parents and their
numerically rejected children. Startup supports the declared `none` telemetry
policy without manufacturing target-status bits, while `per_chain_step` still
requires complete valid telemetry. A failed initial/retained state remains fatal.
Enabled startup requires fresh lower-floor bootstrap before adaptation. All
candidate proposals still need exact-pair measurement and independent verification.

The public field is `HMCKernelTuningConfig.bootstrap_initialization_rounds=0`
by default; positive integers enable the existing four-momentum shrink probe.
The controller field is `HMCControllerConfig.explore_failed_intervals=False`
by default; enabled exploration uses explicitly funded refinement rounds and
the existing family/candidate/work limits. Both options are serialized. Candidate
failure records and every verified member remain intact. Empty refinement grids
now normalize to tuples on config reload, repairing an existing equality defect.

## Checked results

The two optional mechanisms passed their bounded engineering checks. All four
no-hint fits completed startup and tuning, but none of the four predeclared
selected posterior members equilibrated within the warmup cap. The full
inference workflow remains unrepaired. M9's next justified action is a focused
mass-preparation repair, not a numerical-default promotion or another blind
increase in candidate budget.

| Complete no-hint fit | Candidates | Verified members | Warmup per chain | Retained per chain | Final warmup-window max R-hat |
| --- | ---: | ---: | ---: | ---: | ---: |
| Startup-only pilot | 47 | 14 | 10000 | 0 | 2.3691 |
| Both repairs, fresh 0 | 100 | 21 | 10000 | 0 | 1.8730 |
| Both repairs, fresh 1 | 100 | 25 | 10000 | 0 | 1.8337 |
| Both repairs, fresh 2 | 100 | 22 | 10000 | 0 | 2.8841 |

Every numerical health check passed during these selected-member warmups; the
posterior equilibration criterion failed. No mean-reference comparison was
available because the controller correctly produced no retained samples.
Only the predeclared first sorted verified member was assessed in each fit;
the other verified members were retained and their posterior quality is unknown.
The three combined fits reached the declared 100-candidate cap, so completion
means completion of funded work, not exhaustion of possible epsilon/L settings.
See `R/fit-summary.json` for exact stage-separated counts and outcomes.

The matched saved-geometry GPU comparison is complete. Its disabled arm retained
zero of 12 candidates. Its enabled arm made 12 rejected-interval proposals and
completed 75 total candidates, retaining 15 verified members. Every member
passed export/reload checks. Among them is the previously omitted
`L=3, epsilon=1.141019386629899`. Ordinary survivor refinement accounts for the
additional candidates; the option does not restrict retention to one discovery.
All 12 common candidate records and 14 common receipts agree between arms
(apart from the numerical-evidence payload hash). Invalid endpoint receipts
remain unrepairable acceptance evidence and their candidates remain rejected.
See `R/interval-comparison.json` and the two `saved-*/assessment.json` records.

This is a mechanism/reachability result on selected saved geometry. It is not
a powered comparison or posterior claim for those 15 members. The same target,
both affine layers, mass and four-chain start bank were reissued from the
checked M8 handoff. Original receipts were never reused as fresh evidence.

The no-hint regression pilot passed preparation. The optional probe reduced
epsilon from `0.3194715521231362` to `0.0003119839376202502` over 11 rounds;
fresh bootstrap passed, then ordinary mass adaptation completed. Tuning finished
with 47 candidates and 14 verified members. The predeclared selected member
failed posterior warmup at 10000 transitions per chain, with no retained draws.
The final 1000-transition window had maximum modern R-hat 2.3691, driven by
the log-scale coordinate, while numerical health remained valid. This is a
posterior equilibration failure; it does not invalidate or relabel tuning
membership. The pilot took 480.17 GPU worker-wall seconds.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep optional interval repair | Saved counterexample recovered; 15 replayable members | All invalid endpoints rejected; inventory checks pass | Selected geometry, one matched experiment | Complete fresh fits with both repairs | Broad robustness, statistical superiority or all-member posterior validity |
| Keep startup repair optional; repair preparation next | CPU scale tests pass; all four GPU fits retain candidates | All four selected members fail posterior warmup at its cap | Metric proposals were never applied; exact rejection causes not archived | Diagnose and repair covariance adaptation with its own reviewed contract | Successful tuning is not convergence; unassessed members are not rejected |

| Inference status | Assessment |
| --- | --- |
| Hard veto screen | Saved comparison preserves numerical rejections; all verified members have fresh healthy receipts. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Candidate counts, acceptance and runtime. |
| Default-readiness | Not established; both options remain disabled by default. |
| Next evidence needed | Metric-preparation diagnosis/repair, then fresh complete fits, broader target coverage and separate calibration. |

## Tests, sources and documentation

The initial focused run found a mistaken test expecting only the first passing
refinement round and an actual config tuple/list roundtrip defect. Both were
repaired. The broader run passed 405 tests with one inherited skip and one stale
deadline-status assertion from the concurrent checkpointing work. The assertion
now expects the implemented `deferred` status and checks the budget exception.
The follow-up tests pass, including the fresh-bootstrap exhaustion guard. The
separate documentation/dispatch/registry batch passed 46 tests. Terminal
deduplication records **454 passed, zero unresolved failures, one inherited
skip**. The skipped tiny Gaussian windowed test exhausts its bootstrap repair
budget and is not successful numerical evidence.

Frozen source r1 is
`6bb13a00f9f73bab62cc59470027b5588424b558c53c85b9a00e274fb3a1d457`.
It supports the saved-geometry comparison. Source r2 adds the startup-exhaustion
guard and is
`ef137d2c9cfd373598bffac7418ea30b5b37323bda51a39d89e15601c360800a`.
Complete regression fits use r2. Snapshots preserve the dirty checkout at
`d86dadf68ea57772642c6802990f46a3c6a04c30`; concurrent q20 work is preserved.
GPU 0 uses TensorFlow/TFP XLA with verified memory growth before initialization.
Tests hide GPUs; no NumPy runtime computation was added.

The agent reference and official tuning chapter are updated. The installed
`docs/main.pdf` remains 565 pages; physical pages 411 and 414 were visually
inspected. PDF SHA-256:
`b4fd00aba2a2eecf9b13633b9424aeac0e78cb8716ab0d316bb8499c7a98ff9e`.
The first build succeeded, then the rendering helper failed to match a wrapped
identifier in extracted text. Normalizing whitespace repaired that helper;
the existing PDF was rendered without rebuilding. The first build is charged
its conservative 600-second ceiling because a complete measured wrapper record
was not produced. Three inherited unresolved citations remain: Afshar2015,
Gorinova2020 and Pakman2014.

## Restart and remaining work

September 20 M10 correction: the active operational preparation takes a single
position vector and returns rank-2 time/coordinate arrays; it constructs the
four-chain candidate bank afterwards. Thus the earlier interleaving diagnosis
below does not apply to these fits. The reshape is redundant, not the cause of
their zero metric updates. The explicit-chain helper's R-hat gates and omitted
window diagnostics remain real issues. M10 reproduces actual decisions before
selecting a numerical repair. Historical run artifacts remain unchanged.

The saved-matrix diagnosis is complete in `R/saved-metrics-run.json`. All four
no-hint M9 preparations and all three successful hinted M8 preparations applied
zero empirical metric updates. The former retain exactly identity geometry and
local curvature condition 31923.03515 at the checked center; the latter retain
the supplied whitening with condition approximately one. This narrows the
remaining problem to the preparation/trajectory path but does not prove which
metric rejection caused the posterior failure.

Call-chain inspection found a concrete additional inconsistency: operational
warmup flattens draw/chain axes before the metric helper estimates temporal ESS.
That interleaved sequence is not within-chain temporal evidence. The helper
also contains legacy split-R-hat gates for explicitly shaped chains; fixing
only the reshape would activate those gates. The current flat call does not
execute them. The next repair must address diagnostic roles and estimator shape
together, preserve per-window rejection reasons, and maintain numerical
covariance/coordinate/step qualification checks. The master's next-tranche note
records this finding and its required discriminating checks.

All six GPU attempts completed; no M9 worker or reservation remains. Terminal
source, receipt, candidate-retention, tensor, launch-provenance and accounting
checks passed, with no invalid artifact or outstanding work, in
`R/reconciliation-terminal.json`. All attempts, including initial test failures,
both book builds and the conservative failed-render/build charge, are preserved.
The audit checked six candidate inventories, two source snapshots, 800 numerical
receipts and 808 posterior tensor checksums. Current q20 files continued changing
outside this work; their differences from the frozen snapshots are recorded and
do not alter the campaign's executed HMC source.

| Worker-wall seconds | M9 charged | Cumulative charged | Remaining campaign allowance |
| --- | ---: | ---: | ---: |
| CPU reference/tests/builds/overhead | 1537.7544981481042 | 105438.7887427778 | 153761.2112572222 |
| GPU | 3502.109947427991 | 110776.83373940474 | 62023.16626059526 |

This is within the M9 ceilings of 6000 CPU and 10000 GPU seconds. Remaining
allowance is about 42.71 CPU and 17.23 GPU worker hours. Unused M9 reservation
returns to the overall campaign; prior charges do not reset. The terminal M9
ledger is the opening ledger for the next tranche.

Post-run red-team question: the strongest alternative explanation for recovery
is the selected known interval and checked near-mode start. Fresh fits can test
this limited mechanism but cannot establish reliable behavior under unknown
scaling, modes, initialization or nonlinear geometry. Broader calibration,
burn-in sufficiency, SBC, global exploration and matched MacroFinance evidence
remain separate requirements in the master.
