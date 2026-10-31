# Frozen MacroFinance full80: Walnutpie versus saved HMC

**Completion review, 31 October 2026:** `comparison-01` stopped normally at
12:46 Shanghai after 105 adaptive transitions and 1400 retained draws per
chain. Its first posterior assessment passed and its 80-mean comparison
reported compatibility. Before reporting the terminal conclusion, reconstruct
both streams from saved physical draws with the same pinned estimators using
`full80_walnutpie_report.py`, in the fresh `terminal-review-01` output directory.
This is a CPU-only, read-only diagnostic covered by work step 5, with a
120-second convenience cap; it launches no sampler and uses no extra GPU time.
Verify saved draw and source hashes and agreement with the worker's diagnostics.

Skeptical terminal audit: the comparator is the original frozen full80 HMC
stream, not a substitute target; warmup is excluded and repeated frozen draws
remain. All 80 parameters enter the predeclared MCSE-aware mean screen only
after the posterior checks pass. A reconstruction or checksum mismatch would
block the terminal interpretation and trigger a reporting repair. Passing is
compatibility at available precision, not equivalence, absence of missed modes,
or sampler superiority. Historical HMC and new Walnutpie worker times have
different adaptation/recovery costs, so their ratio is descriptive only.

The first terminal reconstruction stopped at the broad model-source binding
check: MacroFinance working files changed after the sampler imported and
compiled its target. Preserve this reporting failure in `terminal-review-01`.
The revised `terminal-review-02` reconstructs diagnostics from saved physical
draws only, imports the checksum-verified pinned BayesFilter diagnostic runtime,
and does not import or evaluate any MacroFinance model. Check the baseline
archive against its pre-run replay hash and all candidate tensors against
their recorded hashes. A bounded same-estimator FP64 comparison uses 1e-12
absolute plus 1e-10 relative tolerance for reduction-order noise; these are
diagnostic comparison tolerances, not changed posterior thresholds. Record
the source drift as a fresh-launch reproducibility limitation. Do not reuse
the old binding for a new numerical run or silently change its expected hashes.

Terminal reconstruction passed in 5.212 seconds: both posterior assessments
and the 80-mean comparison were reproduced, and ten source snapshots plus
2409 saved sample/checkpoint hashes verified. Supervisor exit code 0 and
72285.101 seconds reconcile the completed charge. No additional sampler run
is required. The final result and reset notes supersede pending execution
language retained below as history.

**Active authorization, 30 October 2026:** the owner granted **70 additional
GPU-worker hours (252000 seconds)**: “you have 70 hours more.” This supersedes
the pending 50-hour proposal and approval status below. The initial 1702.288
seconds remain charged to qualification and are not deducted twice. No
historical MacroFinance credit is used. The full80 target, fixed identity,
author .80 adaptation, uncertainty checks and stopping conditions are unchanged.

## Question and exact comparator

Owner request, 30 October 2026: test Walnutpie on the trained 80-parameter
MacroFinance model with identity mass fixed in NeuTra coordinates, as in the
original HMC run. The inspected match is DZ5 financial full80 (called BGL80
in the request), not the separate BGL135 model. Do not change its model, data,
transport or physical parameter definitions. This authorizes only the scoped
standalone Walnutpie application; public NUTS defaults remain excluded.

Reference: MacroFinance `docs/plans/dz5_full80_hmc_result_20261022.md`, attempt
`docs/plans/artifacts/dz5-full80-hmc-20261021/sample-repair-20261021T115311006662Z`.
Four HMC chains used the frozen update5000 component map, all-FP64 pricing,
TF32 disabled, identity latent mass, L=5 and epsilon=0.33635856610148585.
Warmup was 2000 transitions/chain; retained length 1400/chain. The final
reported maximum R-hat was 1.006456, minimum bulk/tail ESS 948.49/423.81,
and maximum controller MCSE/SD 0.03506. These are saved baseline facts,
not results of this new experiment. Original repaired worker time was
20964.47 seconds, including recovery and overhead.

## Intent and evidence contract

**Protocol correction after the engineering pilot:** the source-controller
specification at the end of this plan supersedes the inherited 0.70 target
and 2000-transition adaptation proposal. The original pilot remains preserved
engineering evidence. Its optimizer and state cannot resume into the corrected
posterior campaign.

Mechanism: replace only the transition with the existing deterministic
Walnutpie TF/XLA implementation. Use `adapt_mass=False` and explicitly
supplied ones for all 4x80 mass entries; check identity after initialization,
every adaptation chunk and every retained chunk. Warmup may adapt step and
macro work; retained sampling freezes them. No retraining or extra whitening.
The existing transformed density/analytical score includes the full frozen
map and Jacobian exactly once. Use its explicit first score, not a new
autodiff filter or finite-difference runtime.

Primary question: are all 80 physical posterior means compatible with the
saved HMC estimates at the stated Monte Carlo precision? Both streams must
first pass the original physical-coordinate assessment: modern rank/folded
R-hat <=1.01, bulk/tail ESS >=400, mean MCSE/SD <=0.05. These thresholds
are inherited for this matched experiment, not universal convergence proofs.
Use the same reporting estimator for both streams. Report per-parameter mean
differences and combined MCSE; a Bonferroni two-sided 5% familywise normal
screen across 80 parameters can flag inconsistency. Failure to flag a
difference is compatibility at available precision, not proof of equality.
Standard deviations, marginal quantiles, timing, gradient work and refinement
counts are explanatory/descriptive. No speed ranking from one historical run.

Continuation vetoes: wrong data/map/source/target, missing Jacobian, unmatched
value/score, invalid retained state/cache, mass drift, unqualified XLA graph,
missing memory growth, corrupted/missing draws, or exhausted budget. Correctly
rejected bad proposals and internal refinement failures are telemetry, not
external candidate vetoes. Failed convergence/precision is a promotion veto
and a reason for extension if affordable, not invalidity of the method.
Repair triggers: localized adapter/serialization/import/compile defects.
Preserve attempts and charges. Never relax checks or silently change target.

## Work sequence and compute

1. Bind saved source/data/checkpoint identities and inspect the exact baseline
   samples and diagnostics. Audit imports; keep MacroFinance read-only.
2. Implement a standalone bridge and focused CPU fixtures. Import the original
   pinned BayesFilter target/transport authority with the current standalone
   Walnutpie kernel. Decode existing array archives into TensorFlow tensors;
   no NumPy numerical path in the new runtime.
3. Trusted GPU qualification: configure/verify growth before initialization;
   replay saved value/score anchors, row isolation, identity-mass invariants,
   explicit-signature target and complete transition XLA. Measure compilation,
   steady whole-transition cost and allocator memory.
4. Price 2000 discarded adaptation transitions and initially 1400 retained
   transitions/chain from actual Walnutpie work, including refinement and
   compilation. Extend only within an explicitly bounded affordable campaign.
   The saved six-hour HMC result does not imply equal Walnutpie cost.
5. Run the affordable qualified comparison with fresh independent seeds,
   checkpoint each chunk, preserve raw latent/physical samples and traces,
   reconstruct diagnostics, compare physical summaries, and review results.

Initial qualification allowance: at most 1800 GPU-worker seconds total and
four launches including repairs, plus 900 CPU-worker seconds for focused
fixtures/analysis. These are conservative convenience caps for the requested
test, not a promise of posterior evidence. Do not reuse the historical
55.8-hour MacroFinance remainder as automatic new spending authority. If a
full comparison needs materially more compute, show the measured estimate
and concrete executable before requesting that expansion. No installs,
other-process interruption, broad changes or public push.

Output: versioned attempts under
`docs/benchmarks/artifacts/full80-walnutpie-20261030/`. Every numerical worker
records command, commit plus source hashes, exact seeds and data/map identity,
environment, CPU/GPU placement, verified memory growth, TF32/XLA, timings,
memory, outputs, and failure/completion status. Preserve earlier artifacts.
Interpreter: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; before import set
`TF_FORCE_GPU_ALLOW_GROWTH=true`, bounded TF/BLAS thread counts, and selected
GPU UUID (or `CUDA_VISIBLE_DEVICES=-1` for explicit CPU reference tests).
Exact qualification commands are recorded after the driver exists.

## Assumptions and skeptical audit

- Baseline map/data/source identity is inherited, checked by hashes and saved
  value/score replay. A changed non-imported master script is provenance drift,
  not permission to replace numerical sources. Record and exclude it from
  executable dependency scope only after checking imports.
- Deterministic Walnutpie, packed layout, max_depth=5/max_halvings=5, target
  macro steps=15 and other source-qualified controls are starting hypotheses.
  Inspect actual depth/refinement/work before making a runtime forecast.
- Target acceptance .70 matches owner preference in the baseline, but
  Walnutpie's exp(-abs(deltaH)) adaptation statistic differs from endpoint MH
  acceptance. Label these separately. Initial epsilon .3363585661 is only
  an inherited warm start; do not claim it is tuned for Walnutpie.
- Four native batched chains and FP64/TF32-off match baseline. Confirm no
  row-mapped target, Python callback, pfor, fallback or retracing.
- Warmup/retained lengths and screens match original reporting requirements.
  Warmup must also meet the existing recent-window readiness screen; maximum
  counts do not establish convergence. The saved reference has Monte Carlo
  error and can miss modes shared by both methods.
- Plan audit passes for source inspection, implementation and bounded
  qualification. Full sampling is conditional on measured affordability and
  the stated budget boundary. Cheap-model CPU speed is not evidence of a
  CPU advantage for this large target; start with the baseline GPU route.

Pre-mortem: a plausible matching summary can result from the wrong transport,
missing Jacobian, shared local-mode trapping, or including adaptation draws.
Source/anchor checks expose the first two; independent streams, physical
diagnostics and explicit phase archives limit the latter risks. None of this
establishes identification or exactness of the financial likelihood.

## Concrete implementation and prelaunch review

Driver: `docs/benchmarks/full80_walnutpie.py`; bridge and source/array checks:
`docs/benchmarks/full80_walnutpie_support.py`. MacroFinance stays read-only.
The original host `FinancialTarget` factory repeats data and generates Gaussian
quadrature through NumPy. This bridge replaces only that setup with TensorFlow
and standard-library NPY decoding, then calls the same support-target factory,
transport loader and transformed-score adapter. Normal-quadrature rules are
checked against the original NumPy reference for both orders. The transformed
scope hash must match, followed by saved value/score replay on GPU.

Source-audit correction: the old, unused `create_adapter` function imports
`dz5_financial_neutra_tf`, whose other unused campaign functions eventually
reference the drifted master. The static closure excludes precisely that unused
factory; runtime verification rejects either legacy setup or master import.
All remaining bound sources and imported pinned BayesFilter sources must hash
correctly. This is an import-scope distinction, not acceptance of changed
numerical code.

Five focused CPU diagnostic tests passed in 6.50 seconds: primitive archive
decoding and object rejection, both quadrature orders, explicit first-score
gradient/lifecycle with frozen identity, and invalid-status propagation. CPU
devices were intentionally hidden. These are engineering checks, not evidence
of full80 convergence. The 256 MiB per-array bound is a convenience I/O limit
larger than the actual archives; 5e-14 quadrature tolerances are FP64 comparison
tolerances. Saved target tolerances remain inherited from the original checks.

Trusted inventory found GPU1 idle. Exact first launch (1700-second hard worker
cap leaves 100 seconds for small repairs within the 1800-second total):

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -B docs/benchmarks/full80_walnutpie.py --output docs/benchmarks/artifacts/full80-walnutpie-20261030/qualification-01 --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --wall-seconds 1700
```

Skeptical prelaunch review: no target/filter rewrite, no historical master
execution, no inherited compute credit, no adapted metric, and no sample-based
claim from the eight-transition pilot. Fresh source snapshots, hard process
timeout, exact checkpoint binding, saved anchors and current-cache checks
answer the engineering question. Four pilot adaptation transitions provide
only an early cost estimate; a long-run forecast must state that work can
change during adaptation. Every sampling graph includes the complete
transition, uses a stable signature and is XLA compiled.

## Pilot outcome and corrected posterior protocol

`qualification-01` consumed 1702.288 seconds including supervisor cleanup.
The supervisor stopped it at its 1700-second worker limit during the first
frozen-sampling compilation. There are four completed adaptive transitions per
chain and no retained posterior sample. Target/score replay, row isolation,
exact identity mass and endpoint cache checks passed. Maximum saved-anchor
value/score differences were 7.373e-11 and 9.023e-12. Source hashes and
transformed scope matched. The second two-transition warmup chunk took
115.823 seconds without compilation (57.911 seconds per four-chain transition).
This tiny, early .70 pilot cannot establish the cost of author-tuned sampling.

The source review found a material protocol error: transferring fixed HMC's
.70 acceptance to Walnutpie is not a matched tuning criterion. The author
targets .80 for exp(-abs(deltaH)) on base macro attempts. The author's
`config.hpp:626-639`, `adapt.hpp:198-230`, and
`adaptive_walnuts.hpp::sampler` specify 50--1000 adaptive transitions,
publication every 5, maximum relative step deviation .10 from the geometric
mean, and a handoff of each chain's actual settings, state and RNG. That mean
is not assigned to the chains. The initial .1 step plus search is the source
starting hypothesis. Identity mass is the sole requested metric modification.
Synchronous GPU observations every 5 steps replace asynchronous CPU workers.

Use these source controls for a fresh run, with the existing source tree and
refinement algorithms unchanged. Archive all adaptation and exclude it.
Retain every output after handoff, including repeated states; no additional
fixed-HMC burn-in or restarted verification. Author parameter agreement or cap
is **not posterior convergence**. The cumulative physical-coordinate R-hat,
bulk/tail ESS and original controller MCSE checks remain the posterior screens.
This replaces the proposed additional 2000/recent1000 warmup screen; that
screen belongs to the saved HMC comparator, not the author Walnutpie controller.
First posterior comparison at 1400 draws/chain matches the saved HMC length;
extend by 100 up to 10000 only within the separately approved time budget.
The 10000 ceiling is inherited from the baseline controller, not an adequacy
guarantee. No .70 checkpoint or unassessed prefix can enter retained evidence.

Cost proxy: (50--1000 adaptation + 1400 retained) times 57.911 seconds gives
23.3--38.6 GPU-worker hours before compilation/diagnostics. A convenience
25% reserve plus roughly one hour for compilation gives 30.2--49.3 hours.
This is a scenario calculation from two warmup transitions, not a reliable
prediction or a speed comparison. A 50-hour additional cap is a proposed
authorization envelope, not current permission. First stop/reprice after 50
new adaptation steps; enforce budget at each five-transition checkpoint and
before the next posterior assessment. The old 55.8-hour MacroFinance balance
is not touched. Current remaining qualification allowance is 97.712 seconds.

Skeptical follow-up review: the initial pilot survives as engineering parity,
but its proposed tuning baseline was wrong. The corrected campaign starts
fresh under author controls; short-pilot timing is only explanatory pricing.
Frequent reverse-check failures truncate subtrees and do not imply corrupted
retained states. Their frequency and work are recorded without using them as
external tuning gates. The intended test remains posterior compatibility of
all 80 physical quantities, not matching acceptance or proving speed.

### Reviewable continuation executable (not launched)

`docs/benchmarks/full80_walnutpie_sampling.py` implements the corrected
controller, supervised hard timeout, fresh output directory, cumulative budget,
source/hash-bound resume compatibility and exact checkpoint state/RNG recovery.
Snapshots preserve the numerical kernel and runner. MacroFinance remains
read-only. `full80_walnutpie_sampling_helpers.py` holds the source controls and
checkpoint I/O; `full80_walnutpie_report.py` compares physical draws using the
original pinned diagnostics. Every five-transition chunk preserves raw draws,
traces and complete state; every frozen draw counts, including rejections.

Proposed command, only after the additional allowance is authorized:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -B docs/benchmarks/full80_walnutpie_sampling.py --output docs/benchmarks/artifacts/full80-walnutpie-20261030/comparison-01 --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --wall-seconds 179900 --campaign-seconds 180000
```

The 50-hour envelope is a maximum, not an instruction to spend it regardless
of progress. The first sampling assessment is at 1400/chain; stop early if
posterior screens pass or measured remaining cost cannot reach the next
assessment. Price every chunk, prominently update at 50 adaptation steps,
and preserve/resume the same state rather than restarting the chain. Up to
four localized repair launches share the same budget. A hard timeout cannot
preserve an in-flight chunk, but all previously committed chunks survive.
The supervisor reserves 100 seconds for final cleanup relative to the grant.

The focused suite now checks source-hash rejection, explicit-score and status
handling, metric and frozen-tuning invariants, author agreement versus cap,
checkpoint round-trip/corruption, legacy-resume rejection, complete handoff/RNG
continuation, inclusion of repeated states, continuation after an insufficient
posterior assessment, and compilation-excluding timing. A mock stream checks
actual orchestration; it does not substitute for the missing full80 posterior
run. Nonfinite or nonpositive reporting MCSE blocks the compatibility verdict.

Final code audit: the likelihood and frozen map remain the pinned authorities;
no sampler-math edits, new acceptance gates, average-step assignment, scalar
target loop, training or NumPy numerical runtime was added. The source's
parameter-agreement stopping test does not certify posterior stationarity.
All 80 physical posterior screens and uncertainty-aware mean comparison are
still required. Resource forecasts remain explicitly unvalidated beyond the
tiny pilot. This passes the implementation review; expanded execution remains
pending user approval because it is materially larger than qualification.

## Authorized launch — 30 October 2026

The 70-hour additional envelope is shared across at most four new comparison
attempts, including compilation, localized repairs and checkpoint resumes.
At the launch check, no comparison attempt had spent any of this grant. Use
251900 seconds for the first worker, leaving 100 seconds of cleanup reserve.
Four CPU threads support the one GPU worker; record actual process CPU time.
The first four-step .70 qualification is retained only as engineering evidence;
the .80 campaign starts with fresh independent transition keys.

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -B docs/benchmarks/full80_walnutpie_sampling.py --output docs/benchmarks/artifacts/full80-walnutpie-20261030/comparison-01 --gpu GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c --wall-seconds 251900 --campaign-seconds 252000
```

Skeptical launch audit: the binding and 26 local imported-source identities
were rechecked; only the previously excluded, unexecuted master differs.
Trusted inventory found GPU1 idle. The already-tested controller preserves
identity mass and author per-chain handoff, then assesses every physical
parameter against the same saved baseline. The grant changes affordability,
not scientific evidence or stopping rules. Early timing remains a scenario,
not a guarantee of completion. No model change, retuning of the baseline,
weakening of criteria, public release or package/environment mutation is
part of this launch. Stop on target/cache/source corruption or budget; extend
an insufficient posterior screen only within the authorized envelope.

Launch recorded: `comparison-01`, 08:41:46 UTC / 16:41:46 Shanghai on
30 October 2026. Supervisor PID 1054363; worker PID 1054364. Startup
verified fixed-identity config, exact target scope and GPU memory growth.
The changed allowance and unchanged controller/resume checks passed before
launch (3 tests, 1.99 seconds). The worker runs independently of the chat
session under its existing supervisor and records every completed chunk.
