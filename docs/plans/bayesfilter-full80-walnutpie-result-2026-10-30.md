# Full80 fixed-identity Walnutpie: completed comparison

## Terminal result, 31 October 2026

Walnutpie passed the predeclared posterior and precision checks at the first
assessment, with 1400 retained draws in each of four chains. None of the 80
physical posterior mean differences from the saved HMC reference exceeded
the simultaneous normal/MCSE screen. The result is compatibility at available
precision. It does not prove equality of the distributions or rule out shared
missed modes.

The numerical worker finished normally at 12:46:20 Shanghai on 31 October
2026. Its supervisor exited with code 0 after 72285.101 seconds (20.079 hours).
Adaptation ended at 105 transitions per chain under the author .80 protocol;
all adaptation draws were excluded. All 1400 frozen outputs per chain, including
repeated states, were retained. Identity latent mass stayed fixed throughout;
the frozen per-chain step sizes were 0.3164460817, 0.2929936676, 0.3306831832
and 0.2792731952. No sampler process remains active.

| Physical-coordinate diagnostic | Original fixed-L HMC | Walnutpie | Declared requirement |
|---|---:|---:|---|
| Chains / retained draws per chain | 4 / 1400 | 4 / 1400 | Same retained length at first assessment |
| Maximum modern R-hat | 1.006456 | 1.005552 | <=1.01 |
| Minimum bulk ESS | 948.49 | 1435.83 | >=400 |
| Minimum tail ESS | 423.81 | 953.43 | >=400 |
| Maximum controller MCSE / posterior SD | 0.03506 | 0.02775 | <=0.05 |
| Recorded worker time | 5.823 h | 20.076 h | Descriptive, not a promotion criterion |

Both streams passed every declared posterior screen. The comparison uses the
same reporting MCSE estimator for both means and the square root of the sum of
their estimated variances. The two-sided Bonferroni threshold at familywise
alpha .05 across 80 means is 3.4205267. The largest absolute standardized
difference is 3.1512966 for `US_Phi_slope_level`; the next is 3.0081675 for
`US_a_slope`. Neither exceeds the simultaneous threshold. These differences
must not be described as every mean agreeing within one or two MCSEs.

The target, frozen update5000 map, physical coordinates and FP64/TF32-off
settings match the recorded HMC binding. Target value/score anchor checks
passed. Endpoint value/score cache validation and exact identity-mass checks
ran after every chunk; frozen tuning checks also passed. Warmup and retained
graphs each executed on GPU with one trace at their graph qualification and
no forbidden callback operations. Host scheduling, checkpoint I/O and posterior
diagnostics remain outside the compiled numerical chunks. Verified memory
growth preceded GPU initialization; measured TensorFlow allocator peak was
132.19 MiB and peak worker host RSS was 15.64 GiB.

## Runtime and work

The full retained phase consumed 65226.68 timed seconds, including its first
compiled call and physical decoding. Excluding the first retained chunk, mean
time was 46.365 seconds per transition across all four chains. The retained
useful evaluation counter averaged 16.241 per chain-transition; it does not
measure all evaluations performed for masked inactive rows. Reverse-check flags
occurred on 780, 685, 871 and 656 of 1400 chain-transitions. They are internal
trajectory rejection telemetry, not counts of invalid retained states or
evidence that the complete transition failed.

The observed Walnutpie worker time is 3.447 times the saved HMC worker time.
This is an accounting comparison, not a controlled performance ratio: HMC
used fixed L=5, 2000 warmup transitions, and a recovery path that replayed
500 earlier warmup draws; Walnutpie used 105 adaptive transitions and distinct
compilation and refinement work. The larger observed Walnutpie ESS values
are descriptive. This single historical comparison establishes neither an
ESS/second advantage nor a statistically supported efficiency ranking.

The additional 70-hour allowance has 179714.899 seconds (49.921 hours) unspent.
The initial 1702.288-second qualification remains separately charged and is
not deducted twice. No further sampling is needed for the stated compatibility
test, and remaining allowance is not an instruction to launch extra work.

## Terminal reconstruction and provenance limitation

All ten saved sampler/driver source hashes and 2409 recorded sample/checkpoint
file hashes verified. A fresh CPU-only reconstruction from the saved physical
draws reproduced both streams' diagnostics and the comparison, using the
unchanged pinned diagnostic estimators. This is an independent execution of
the same estimators, not an independent mathematical diagnostic implementation.
The baseline archive matched its pre-run replay hash. No MacroFinance model
was imported or evaluated by the successful reconstruction.

The first reconstruction attempt stopped because its broad source-binding
check found later edits in MacroFinance. These are preserved as a reporting
failure in `terminal-review-01/failure.json`. The three changed numerical model
files have modification times of 20:33:46 UTC on 30 October, after the first
retained GPU/XLA call completed at 10:41:39 UTC. The other two changed files
are campaign scripts outside the executed dependency closure. The sampling
driver retained the same already-imported, stable-signature compiled functions;
it did not reload the edited model files between transitions. The source
binding and parity evidence at launch therefore remain evidence for this run,
while a fresh model replay from the current MacroFinance worktree is blocked.
Do not silently update the binding hashes or relaunch it as the same target.

The successful repair is `terminal-review-02/reconstruct.py`, executed with
`timeout --signal=TERM --kill-after=10s 120s` and the tfgpu Python interpreter.
It verifies the saved input hashes, imports only the pinned diagnostic runtime,
and writes checks, reconstructed diagnostics, comparison and a manifest. It
finished in 5.212 seconds. The failed first diagnostic has a conservative
120 CPU-worker-second charge because its own wall time was not captured;
neither terminal diagnostic used GPU-worker time. Original sampling results
and source snapshots remain unchanged.

Primary evidence directory:
`docs/benchmarks/artifacts/full80-walnutpie-20261030/comparison-01/`.
Read `manifest.json`, `supervisor.json`, `assessment-1400.json`, `comparison.json`,
`graph.json`, `controller.json`, and the raw tensors. The manifest records
commit `8821e8c0c431071578084164c6915cb95fbb3d82`, the exact command, source
hashes, data/checkpoint binding, seeds, Python 3.13.13/TensorFlow 2.20.0,
GPU UUID, verified allocator policy and timing. Terminal review and accounting
are in the sibling `terminal-review-02/` directory. The reviewed plan is
`bayesfilter-full80-walnutpie-comparison-2026-10-30.md`.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete the requested fixed-identity full80 test | Both posterior screens pass; zero of 80 means flagged | No retained-state/cache, mass, target-anchor, XLA or allocator veto observed | Finite-chain MCSE, normal approximation and shared missed modes | Preserve completed result and report compatibility | Equality of full distributions or exact posterior truth |
| Keep both samplers viable for this target | Both passed the declared checks | No candidate rejection from this comparison | Only one matched historical comparison | Use replicated matched work and uncertainty estimates if efficiency ranking is requested | A speed or ESS/second winner |
| Preserve current source-drift finding | Saved-draw reconstruction passes | Current model-source reload is blocked | Future fresh replay requires the original sources | Recover the bound model closure before any same-target relaunch | Current edited MacroFinance code is the executed target |

| Inference status | Finding |
|---|---|
| Hard veto screen | Completed sampling passes the declared engineering and posterior screens; current-source reload failed and was isolated to reporting |
| Statistically supported ranking | None; the approximate simultaneous MCSE test detects no mean incompatibility |
| Descriptive-only differences | ESS, R-hat, runtime, reverse-check frequency and work counts |
| Default-readiness | No public sampler/default change; standalone full80 compatibility evidence only |
| Next evidence needed | Independent seed replications and matched cost accounting for efficiency; stronger mode/reference checks for broader posterior claims |

Post-run skeptical review: the strongest alternative explanation is that the
two methods share an unvisited region despite acceptable within-stream
diagnostics. An independent reference or dispersed replication exposing a
mean discrepancy would overturn the compatibility interpretation. The weakest
evidence is the one-run, finite-chain normal/MCSE approximation and unequal
runtime histories. The reporting reload failure concerns current source
availability, not a failed candidate or research-direction rejection. The
saved-draw repair passed, so no continuation veto requires another sampler run.

## Historical qualification record

The material below records the earlier pilot and pre-completion interpretation;
its pending/no-retained-draw statements are superseded by the terminal result.

**Execution update, 30 October 2026:** the owner granted 70 additional
GPU-worker hours. `comparison-01` launched at 08:41:46 UTC / 16:41:46
Shanghai with the corrected .80 author protocol and fixed identity mass.
Startup verified the original target scope and GPU memory growth. The
full comparison is running; the qualification-only findings below remain
historical engineering evidence and are not posterior comparison results.

The frozen full80 financial target and trained update5000 NeuTra map were
restored from the original MacroFinance HMC binding. Four native-batched
FP64 GPU/XLA chains used fixed identity latent mass. MacroFinance files were
read-only; no transport training or public sampler/default changes occurred.

## Observed evidence

| Check | Result | Meaning |
|---|---|---|
| Saved map/inverse replay | Passed | Correct trained transformation at checked anchors |
| Saved transformed log density | Maximum absolute discrepancy 7.373e-11 | Target parity at checked anchors |
| Saved transformed score | Maximum absolute discrepancy 9.023e-12 | Explicit first score agrees at checked anchors |
| Row permutation/isolation | Passed | Checked target evaluations preserve chain independence |
| Fixed identity after initialization and both warmup chunks | Exact | No mass adaptation occurred |
| Endpoint state/cache checks | Passed | Saved endpoint values/scores remain valid |
| Warmup movement, four transitions/chain | 3/4, 4/4, 3/4, 3/4 | Engineering movement only |
| Frozen sampling graph | Compilation interrupted at time limit | Still unqualified on full80 |
| Retained posterior comparison | No draws | Not yet assessable |

The seven initial focused checks passed; the subsequently expanded twelve-test
suite covers the author controller, checkpoint recovery and comparison
validity. The GPU memory-growth requirement was configured and verified before
logical-device initialization; TF32 was disabled. The large whole-transition
graph compiled and executed adaptation successfully. Compilation and execution
both count against the budget.

The 1700-second worker cap stopped the first attempt during frozen-sampling
compilation. Charged supervisor elapsed time was 1702.288 seconds. This is a
budget-limited qualification, not evidence that Walnutpie fails on the target.
The original manifest is reconciled with that outcome. Its four adaptive
transitions, checkpoint files, source snapshots, logs and parity record remain
under `docs/benchmarks/artifacts/full80-walnutpie-20261030/qualification-01/`.

The saved HMC posterior reconstruction passed using its original diagnostic
runtime: 1400 retained draws per chain, maximum R-hat 1.006455538896168,
minimum bulk ESS 948.486758695485, minimum tail ESS 423.811475244634, and
maximum controller MCSE/SD 0.0350568416617092. Results are under
`baseline-replay-01/`. These are HMC reference values, not Walnutpie results.

## Protocol correction and cost

The first pilot inherited HMC's .70 target. Source review establishes that
the author Walnutpie default is .80 for a different quantity: base-macro
exp(-abs(deltaH)), not endpoint MH acceptance. An author-protocol comparison
must start fresh with .1 initialization plus search, 50--1000 adaptive
transitions, five-transition checks and .10 cross-chain relative step
agreement. It keeps each chain's actual adapted settings. The .70 pilot
cannot be relabeled or resumed into that posterior campaign. The reviewed plan
and new sampling driver now implement these controls.

The second two-transition adaptation chunk cost 115.823 seconds without
compilation: 57.911 seconds per four-chain transition. Applying that early
rate to 50--1000 adaptation plus 1400 retained transitions gives 23.3--38.6
hours, or about 30--50 hours with a 25% convenience reserve and one hour of
compilation/diagnostics. This is a provisional scenario, not a reliable
runtime prediction; retained cost under .80 adaptation has not been measured.
Frequent reverse-check failures in this tiny pilot truncated subtrees while
valid states were preserved. They are recorded as work/mixing diagnostics,
not a reason to reject the method or override its tuning.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve qualification evidence | Target/mass/cache checks pass | Budget exhausted before frozen-call completion | Full sampling graph and long-run behavior | Fresh corrected-protocol run after expanded budget approval | Posterior compatibility |
| Correct adaptation protocol | Author source reviewed and controller tested | Original .70 choice is not author protocol | Target-specific cost after adaptation | Use .80 author controls and same frozen identity | Optimal tuning or speed |
| Keep HMC comparator | Saved diagnostics reproduced | No observed reference reconstruction failure | Shared missed modes and MCSE assumptions | Compare all 80 physical estimates once candidate qualifies | Exact posterior truth |

| Inference status | Conclusion |
|---|---|
| Hard veto screen | No target/mass/cache defect in completed pilot; qualification budget limit reached |
| Statistically supported ranking | None |
| Descriptive-only differences | Pilot runtime, movement and refinement counts |
| Default-readiness | Not assessed; standalone application only |
| Next evidence needed | Qualified retained Walnutpie stream with physical R-hat/ESS/MCSE and matched mean comparison |

The strongest alternative explanation for future apparent agreement is shared
local-mode trapping. The weakest current evidence is the absence of retained
Walnutpie draws and the tiny, differently tuned timing pilot. A failed future
posterior screen would trigger extension within budget, not an automatic
rejection of the research direction. No full comparison or statistically
supported speed ranking is currently available.
