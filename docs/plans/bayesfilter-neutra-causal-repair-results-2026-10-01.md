# NeuTra causal repair: terminal results

The reviewed repair plan was executed through all audited cases. All six
engineering gaps have executable checks and recorded numerical evidence for the
scope below. The original case ledger contains 30 target/arm/seed groups:
12 earlier passes were preserved, all 18 previously failed groups received
repairs, and 15 of those have at least one new passing posterior-screened run.
Three groups remain unresolved: **mixture/Gabrié seeds 11 and 37**, and
**wiggle/Gabrié seed 23**. A group containing a pass is not a reliability estimate;
all failed attempts remain part of the evidence.

Plan: [causal repair plan](bayesfilter-neutra-causal-repair-plan-2026-10-01.md).
The complete [case table](artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-repair-20261001-r1/case-coverage.md)
and machine-readable [terminal review](artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-repair-20261001-r1/terminal-review.json)
link every original case, repair, qualification attempt and remaining failure.
No worker remains active. Canonical IAF, exact corrected target, identity latent
mass, batching, TensorFlow/TFP and the sequential HMC policy were retained.

## Audit closure at the actual consumers

| Audited gap | Implemented consumer repair | Falsifying check and executed evidence | Status and limit |
|---|---|---|---|
| Unsuitable HMC starts | `neutra_warm_start_qualification.qualify` draws four seeded latent normals through the frozen map and saves both banks, scores and roundtrip errors | Real qualification-entry tests for Gaussian, mixture and funnel capture actual tuner inputs; all three preserved funnel/Gabrié maps pass fresh HMC checks | Verified for this benchmark path; map draws do not prove exhaustive coverage |
| Retry did not widen epsilon range | Effective 8/12 downward-repair counts reach the public fixed-transport tuner and are serialized before tuning | Real stiff-Gaussian regression generates a proposal below the old .0625 boundary; actual campaign configurations record the new limits | Verified; bounded search can still find no verified pair |
| Improving maps were discarded | `fit` restores map, Adam, lifetime updates and Gabrié walkers; the follow-up doubles supported-progress rungs using measured price and shared budget | Real restored optimizers produce the identical next update; real fit tests inspect restored versus reset counts and reject changed teacher particles/weights before optimization; mixture and warped-mixture continuations reach 65536/131072 updates | Verified; supported progress can cease before the shape screen passes |
| SMC repairs did not change effective work | Actual 4/16/64 mutation counts and frozen beta-specific steps reach `AnnealedSMC`; ancestry and movement on empirical coordinate scales are saved | Physical controls execute 12/48/192 total mutation steps per population, followed by independent teacher screens; analytic transformed-density/Jacobian and identity-normalizer checks validate the rescue construction | Wiring verified; physical proposal remains inadequate, while frozen-map rescue succeeds |
| Temporal rule confused fluctuation with instability | Optional v6 paired chain-block contrasts replace raw crossings for this campaign; legacy v5 default remains unchanged | 2048 replications per stationary scenario, persistence 0/.5/.9 and lengths 64/256, plus strong common shifts; fresh HMC checks on affected maps | Declared calibration screen passed; no universal HMC error-rate guarantee |
| Training diagnostics and stop reasons were missing | Persist gradient totals, clipping counts, paired gains and numerical/plateau/shape decisions; retain standard 1000-point probes for finite failed fits | Actual fit-consumer tests, injected update/probe failures, and saved failed 131072-update fit; downstream consumer and supervisor reject invalid selections | Verified; finite geometry residuals remain explanatory, while nonfinite/incomplete probes veto that candidate |

The implementation review also caught and repaired three integration omissions.
Continuation now verifies the parent's exact weighted teacher rather than
selecting a newer file. The follow-up skips a parent already resolved by a later
continuation. Finally, reconciliation against the original 30-case ledger found
warped-mixture/Gabrié seed 37 missing from the initial target-name checklist; it
was continued, sampler-checked and qualified. Testing an example of a defect
was therefore kept separate from covering every affected case.

## What the numerical executions show

**Previously rejected frozen maps.** Seven of eight now pass fresh declared
posterior checks: funnel/Gabrié seeds 11/23/37; wiggle/SMC seeds 11/23;
wiggle/Gabrié seed 11; and warped-mixture/SMC seed 23. Wiggle/Gabrié seed 23
remains unresolved: one attempt reached the retained cap without sufficient
precision, and its bounded retry found no verified tuning pair. Neither result
establishes that no useful kernel exists.

**Mixture training.** Plain saved-state SMC-teacher continuation passed for
seeds 11 and 23 at 65536 lifetime updates. The seed-37 lower-learning-rate map
continued to 131072, then stopped with residual shape failure and no supported
paired progress: gain .000721, standard error .001864. Its optimizer count is
131072 and its finite 1000-point probe is preserved. This is an observed plateau,
not an exhausted campaign allocation. Other seed-37 plateau-parent continuations
and controls did produce posterior-screened maps.

The seed-11 SMC plateau passed after the 256-update RKL escape plus continued
forward training; reset alone remained near its Gaussian plateau. Seed 23's
nominated plateau did not escape under either intervention. Seed 37 passed with
ordinary continuation, reset alone, and reset plus RKL. These results support
some viable schedules, not a general RKL-specific causal benefit or superiority.
The same parent, teacher, forward rungs and fixed nomination order were used;
RKL adds its explicitly recorded 256 updates and compute.

Gabrié mixture seed 23 restored its actual optimizer and walkers, passed the
shape screen at 65536 updates, passed the separate frozen-sampler screen, and
passed fresh HMC checks with 2000 warm-up and 2000 retained transitions per
chain. For seeds 11 and 37, pure continuation, reset-only and reset-plus-RKL all
remained near the Gaussian forward-KL plateau (about .947--.948 in the finite
development sample). These two cases remain open. The controlled schedule was
not an exhaustive target-specific optimizer/capacity search.

Warped-mixture/Gabrié seed 37 passed after continuation to 65536 updates. Its
frozen-sampler screen passed; the first HMC search found no verified pair, while
the declared retry passed with 2000 warm-up and 1000 retained transitions per
chain. The first failed search is preserved.

**Funnel teachers.** The physical Laplace/Student proposal failed all three
mutation controls. The first populations executed 12, 48 and 192 mutation steps,
but estimated log normalizers remained near -2.36 against exact zero. Saved
ancestry, pilot overlap, raw movement and variance-scaled movement distinguish
lack of proposal coverage from a merely accepted tiny mutation. Zero empirical
variance is explicitly undefined for the movement ratio; it cannot certify
mobility.

Frozen-map SMC teachers passed for seeds 11/23/37 at 4096/1024/8192 particles
per population. Estimated V means were .00094, -.04457 and .00069 against zero.
These populations required no resampling, so **they executed no mutations**.
Their success supports improved proposal overlap under the exact change of
variables; it cannot be attributed to the configured 64 mutations. All three
newly trained maps have a passing downstream HMC run, with warm-up/retained
counts 7000/10000, 2000/5000 and 6000/7000 respectively. These observed counts
are not a runtime ranking.

Seed 11 also has two earlier warm-up failures. A later provenance-triggered
execution used the identical map and selected epsilon/L but different numerical
streams: the shared tuner hashes source identity into work seeds, and a
comment-only source edit changed that identity. The trace is preserved in
[source replay sensitivity](artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-repair-20261001-r1/source-replay-sensitivity.json).
The later pass cannot erase earlier failures or be attributed to an algorithmic
improvement. Top-level seed equality was not treated as equal random draws.

**Temporal calibration and clipping.** Stationary temporal-flag rates ranged
from .20% to 1.56%; every Wilson upper bound was below the declared 5% screen.
Strong common shifts were detected in all replications. The 1% working
familywise contrast level is not proven nominal coverage for arbitrary HMC
processes. Numerical, movement and downstream convergence checks remained.
Every recorded forward/Gabrié repair-fit block in this campaign had clipped
fraction zero. This closes the observability question for these runs, not a
universal claim that clipping is unnecessary.

## Verification, provenance and resources

Initial regressions: 147 tests passed; an additional 145 shared HMC
execution/replay tests passed. Subsequent focused tests reached the real fit
consumer and both forward/Gabrié Adam restoration paths. The isolated final
repair suite passed 19 tests, 39 compatibility tests and seven terminal-probe
checks; focused refinement and qualification-veto checks also passed. After
installation, the affected continuation/coverage/validity suite passed 16 tests
(10 unrelated cases deselected). Counts overlap and must not be added into a
unique-test total. Test XML and installation hashes are under
`causal-repair-20261001-r1/final-source-verification/`.

The terminal audit checked 3362 archived source-file hashes with no mismatch.
Every completed GPU worker used assigned host GPU 1 and recorded verified
memory growth before initialization. No failed/incomplete terminal 1000-point
probe was found among the executed repair artifacts. These are source and
numerical-validity checks; they do not prove a complete mathematical audit.
CPU reference tests intentionally hid GPU devices. Unrelated worktree changes
and GPU jobs were preserved; no commit or push was performed.

The causal-repair workers consumed **8125.89 GPU process-seconds (2.257 hours)**
and **12685.59 CPU core-seconds (3.524 hours)**, including recorded infrastructure
failures and provenance-triggered repeats. Routine CPU regressions are reported
separately above. Conservative remaining campaign capacity is **24.032 GPU
process-hours and 49.134 CPU core-hours**. The CPU remainder reserves the older
orphaned-job uncertainty rather than treating it as free capacity. Total budget
ceilings remain unchanged and compliance is established by the shared ledger.

Commands actually executed, in the existing `tfgpu` environment:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_causal_repair_master.py campaign
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/continue_neutra_causal_repair.py
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/continue_neutra_causal_repair.py --remaining-audit-cases
```

The earlier `canaries` stage preceded the full campaign. GPU commands used the
trusted permission boundary. Two initial sandbox attempts failed before CUDA
execution and were preserved as infrastructure evidence; the trusted execution
succeeded. Each numerical worker retains command, environment, source/input
hashes, seeds, plan, outputs, memory policy and actual resource use.

## Decisions and terminal skeptical review

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close the six engineering repair rows for this benchmark path | Real consumer checks and numerical artifacts agree with declared repairs | No missing original case; archived sources and required probe validity checked | Other consumers and unseen targets need their own evidence | Retain the regression and case-ledger guards | Repository-wide production certification |
| Preserve new passing benchmark runs | Fresh sequential HMC and reference checks passed for 15 repaired groups | Earlier failures remain visible | Unequal attempts and small seed counts prevent reliability/ranking claims | Use these as scoped candidate evidence | 27/30 is a success probability |
| Keep mixture/Gabrié 11/37 open | Nominated controls still fail nonlinear/shape learning | Promotion blocked; independent work completed | Stationary configuration versus optimizer/capacity limitations | A separate target-specific training experiment with matched controls | Architecture impossibility or rejection of NeuTra |
| Keep wiggle/Gabrié 23 open | Retained precision/search screens remain unmet | Promotion blocked | Untried verified members/trajectory coverage and map geometry | A focused fixed-map kernel/precision diagnosis | Failure of all HMC configurations |
| Reject physical funnel teacher; retain transformed rescue | Independent teacher screen distinguishes them | Physical teacher remains failed | Rescue depends on an independently learned useful map | Preserve overlap and mutation evidence separately | Mode discovery or extra mutations alone solved the funnel |

| Inference status | Terminal evidence |
|---|---|
| Hard veto screen | Numerical, shape, teacher, tuning and posterior failures remain unpromoted |
| Statistically supported ranking | None |
| Descriptive-only differences | Case counts, losses, residuals, moments, acceptance, and runtime |
| Default readiness | Not established; q20 was not run |
| Next evidence needed | Target-specific resolution of the three open groups, replicated reliability if claimed, and a separate q20 training/estimation contract |

The strongest alternative explanation for a passing repair is that one selected
map/seed/kernel works while the procedure remains unreliable; the preserved
funnel replay and failed plateau controls make that limitation concrete.
An untouched replicated study could overturn any informal generalization from
these successes. The weakest general evidence is the small seed count and the
working temporal calibration's limited dependence family. Those limits do not
undo the demonstrated initialization, search-range, continuation, mutation-
wiring and observability repairs. No planned phase remains running or silently
pending; further scientific work starts from these explicit open questions.
