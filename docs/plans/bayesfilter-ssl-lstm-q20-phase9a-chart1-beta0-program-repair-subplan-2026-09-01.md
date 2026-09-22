# SSL-LSTM q=20 Phase 9A Chart-1/Beta-0 Program-Repair Subplan

Date: 2026-09-01  
Status: `COMPLETED_LOCALIZED_MECHANICS_PASS_FULL_REPLAY_PENDING`  
Parent boundary: `Phase 9A CLOSED_PHASE9A_CONTINUATION_VETO_CHART1_BETA0`  
Guide: `docs/reference/hmc-tuning-interface.md`  

## Purpose

This subplan repairs the executable q=20 Phase 9A program after the chart-1,
beta-0 scope failed its old directional tuning handoff. It is deliberately
limited to a fresh chart-1/beta-0 mechanics repair. It does not reopen Phase
9B, change the target or bridge, retrain the selected architecture as a new
scientific arm, or make a whitening, posterior, mode-discovery, or HMC
readiness claim.

## Research question and diagnosis

**Question.** Can the same frozen q=20 bridge and compact-high chart protocol
produce a valid chart-1, beta-0 fixed-transport handoff when the program
measures a declared joint `(epsilon, L)` grid and uses enough fixed-kernel
evidence to select by movement/ESS rather than by a short acceptance ladder?

**Observed baseline.** The closed 2026-08-30 Phase 9A attempt-05 chart-1/beta-0 caller
used one `L=5` and directional dual-averaging repair. It reported finite,
mobile screens at epsilon `0.628978` (mean Metropolis probability
`0.998950`) and `1.205189` (`0.939618`), then requested `2.410379` above the
cap. The failure is therefore a reproducible tuning/control-boundary failure;
it is not evidence of a bridge or score mismatch.

**Mechanism under repair.** The Phase 9A benchmark runner and its scope
configuration, not the q=20 target equations or learned-map objective.

## Evidence contract

| Item | Contract |
|---|---|
| Exact target | q=20 SSL-LSTM bridge, target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Frozen protocol | C5 `phase8-k2-compact-high-l3-pure`: two `(16,16)` tanh charts, two stages, learning rate `1e-3`, betas `(0, .5, 1)`, pure continuation, fixed uniform chart weights `(0.5, 0.5)` |
| Repair scope | Fresh chart-1 construction and beta-0 tuning only; chart-0 is built only when needed for the existing two-chart reliability screen |
| Baseline | Closed 2026-08-30 attempt-05 old one-`L` directional caller and the current measured-grid implementation |
| Primary pass criterion | The fresh chart-1/beta-0 route measures every declared pair, passes finite/status/movement checks, and writes a durable `VerifiedFixedTransportHMCHandoff` for that scope |
| Hard vetoes | Target/backend/signature mismatch; stale or reused seed/checkpoint; nonfinite bridge/chart value, score, inverse, or logdet; missing transition telemetry; divergence; no retained movement; grid pair omitted or inferred; output collision; GPU memory/XLA failure; missing start/failure manifest |
| Repair triggers | No viable measured candidate, high acceptance with poor movement, insufficient selection evidence, retracing/resource warning, or held-out failure |
| Explanatory diagnostics | Acceptance probability, binary acceptance, ESJD, ESS per declared gradient, energy error, timing, allocator use, and chart pullback residuals; none alone establishes correctness or convergence |
| Artifact | Fresh versioned output directory containing start manifest, final/failure manifest, chart checkpoints, scope tuner JSON, handoff payload, and result note |
| Nonclaims | No whitening/IID-Gaussian, mode-discovery, posterior, convergence, sampler-ranking, production, default-readiness, or high-dimensional-scaling claim |

## Repair design

### 1. Scope and profile controls

The runner gains `--scope-start` and a named profile. Scope index `3` is
chart-1/beta-0 in the fixed order `(chart-0,beta-0)`,
`(chart-0,beta-.5)`, `(chart-0,beta-1)`, `(chart-1,beta-0)`, ... . The repair
profile accepts only `--scope-start 3 --scope-limit 1`; this prevents an
accidental full confirmation launch from being described as a localized
repair.

### 2. Fresh measured-grid repair profile

The profile keeps the target, cap, chart architecture, and acceptance bands
unchanged from the current Phase 9A hypothesis, but replaces the old one-`L`
ladder with the following explicit grid:

```text
step_size_candidates = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40)
leapfrog_grid         = (3, 5, 8, 12)
maximum_candidate_step_size = 2.0
```

This is 32 measured pairs. The points straddle both previously observed
near-return regions without treating either old epsilon as a valid default.
The route must execute every pair before selection; no acceptance result may
create, remove, or infer a neighbor.

The initial `chart1_beta0_repair_v1` profile retained this 32-pair grid and
used 16/64/64 screen, selection, and held-out draws. Its first GPU launch
(`attempt-01`) was terminated by the 1,800-second outer bound (exit 137)
after TensorFlow spent about 32 minutes in repeated full-chain calls; it
produced only the pre-import start manifest. This is a harness-budget failure,
not a tuning or target result. The measured call count is
`32 + 32*2 + 1 = 97` full-chain invocations before chart/reliability setup.

The bounded retry is a separately named profile,
`chart1_beta0_repair_v2_bounded`, with fresh seeds. It measures every pair in
the declared eight-pair joint grid

```text
step_size_candidates = (0.25, 0.55, 0.85, 1.20)
leapfrog_grid         = (3, 8)
```

and uses four screen results plus two burn-in steps, two selection replications
of sixteen results plus four burn-in steps, and sixteen held-out results plus
four burn-in steps. Thus it performs `8 + 8*2 + 1 = 25` full-chain calls while
leaving the target, cap, measured-pair requirement, movement/finite vetoes,
held-out requirement, and acceptance-band semantics unchanged. The reduced
grid is a localization evidence hypothesis only; it cannot promote a global
tuning default.

The v2 attempt then demonstrated that even eight pairs with sixteen-draw
selection still exceeded the 1,800-second bound. Its failure manifest records
signal 15 at 1,851 seconds, with all six fresh chart checkpoints present and
the scope tuner just started. The next repair is therefore a minimum-policy
probe profile, `chart1_beta0_repair_v3_minimal`, using the smallest legal
two-by-two measured grid

```text
step_size_candidates = (0.55, 1.20)
leapfrog_grid         = (3, 8)
```

and one screen plus one burn-in step, two replications of four results plus
one burn-in step, and four held-out results plus one burn-in step. It makes
`4 + 4*2 + 1 = 13` full-chain calls. This profile is explicitly a feasibility
probe: a pass demonstrates that the active route can execute a pinned,
finite, mobile measured grid, but does not establish a robust tuning handoff,
convergence, or a default. The v1 and v2 profiles remain preserved as failed
resource attempts and are never reused.

The v1/v2 repair profiles use four varied latent starts
`[(0,0,0,0), (0.25,0,0,0), (-0.25,0,0,0), (0,0.25,0,0)]`, two independent
selection replications, 16 burn-in plus 64 retained selection draws per chain,
and 32 burn-in plus 64 held-out draws per chain. These numbers are bounded
mechanics evidence, not convergence settings. Their provenance is this
scope-local repair hypothesis; they cannot be promoted without a later
target-specific campaign. The v2 and v3 profiles override only the draw counts
and measured grid as stated above; the start bank, target, and veto semantics
remain unchanged.

Because attempt-04 was a provenance-only rerun of v3, it reused v3's seed
namespace and cannot satisfy the final fresh-seed criterion. The final v4
profile, `chart1_beta0_repair_v4_fresh`, keeps this exact grid and draw
schedule but uses initialization roots `77001/77002`, preflight roots
`77101/77102`, training roots `77201/77202`, tuning roots `77301--77306`,
transition root `77401`, and reliability root `77501` in the 2026-09-01 seed
domain.

### 3. Provenance and failure durability

The repair profile uses a new 2026-09-01 seed namespace for chart
initialization, chart preparation, training, tuning, and transition roles. It
must not read a chart or checkpoint from attempts 01-05. The runner writes a
`run_start.json` before TensorFlow/chart construction and writes a structured
`run_manifest.json` on both success and failure. A failure manifest includes
the command, profile, scope, Git state, elapsed time, exception type, and
failure classification, and never overwrites an existing artifact.
Each selected scope also receives immutable `scope_start.json` and
`scope_complete.json` markers. The launcher uses a 1,800-second timeout with a
120-second termination grace period and writes the same failure schema if a
TensorFlow call prevents Python from servicing the signal. During tuning,
each full-chain call receives immutable `call-*-start.json` and
`call-*-complete.json` (or `call-*-failure.json`) records, including its static
budget, dynamic `(epsilon,L)`, elapsed time, and runner diagnostics.

## Implementation phases

### Phase P0: Freeze and source audit

1. Verify the C5 freeze receipt and target signature.
2. Read the repaired HMC guide and confirm the active fixed-transport route is
   `measured_joint_grid_v1`.
3. Confirm attempts 01-05 remain immutable and choose a new output root.
4. Check that no GPU/HMC process from an earlier attempt is running.

**Exit:** all identities and boundaries match; otherwise repair the plan and
stop before code changes.

### Phase P1: Runner repair

1. Add profile and scope-selection validation to
   `run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py`.
2. Thread profile-specific seeds and tuning controls into chart construction,
   reliability, and scope tuning.
3. Add start/failure manifests and record selected scope indices/profile in the
   final manifest.
4. Add a single exact GPU wrapper for the localized repair.

**Between-phase repair review:** inspect the diff for target/signature changes,
seed reuse, hidden acceptance inference, output overwrites, scalar/pfor paths,
and a profile that can silently run outside its declared scope. Run compile
and CLI/config unit checks before any GPU process.

### Phase P2: CPU and fixture verification

1. Compile the modified runner and wrapper-adjacent modules.
2. Exercise profile validation, scope ordering, measured-grid cardinality, and
   failure-manifest construction with CPU-only fixtures.
3. Re-run the repaired fixed-transport policy tests and documentation checks.

**Exit:** all checks pass. A fixture failure is an implementation repair
trigger, not evidence against the q=20 method.

### Phase P3: One bounded GPU localization run

Run exactly the localized wrapper on GPU 0 with memory growth set before the
TensorFlow import, a fresh output root, and the profile's fixed 1,800-second
material cap. The run may produce only mechanics/preflight evidence. It must
stop on any hard veto and preserve its failure manifest.

**Pass:** chart-1/beta-0 receives a measured-grid handoff and all required
health fields are finite.  **Fail:** classify as chart, numerical, tuning,
resource, or artifact failure; do not widen the cap or open Phase 9B.

### Phase P4: Closeout and refresh

Write a result note with a decision table, inference-status table, post-run
red-team explanation, exact command/environment/seeds, and the next action.
If P3 passes, write a new separately audited full six-scope Phase 9A replay
subplan. If P3 fails, write only the smallest next repair subplan.

## Stop and continuation rules

Stop for target/bridge/chart identity changes, reused calibration state,
missing reliability or failure artifacts, nonfinite target/score/map values,
unmeasured candidate selection, memory-growth/XLA noncompliance, output
collision, or the declared material/attempt budget. A finite acceptance value
outside the band is a repair diagnostic under the measured policy, not by
itself a continuation veto. Do not infer that a passing localized handoff
proves whitening, mixing, mode travel, or posterior correctness.

## Pre-mortem

The run could appear successful while still being misleading if all declared
pairs share a central start, if selection uses acceptance rather than
fixed-kernel movement/ESS, if a chart checkpoint is silently reused, or if a
failure lacks a manifest. The earliest checks are the varied start bank,
measured-pair count, seed/checkpoint hashes, and the pre-scope start manifest.

It could fail for ordinary resource or XLA reasons rather than tuning. The
runner must preserve that distinction and retry only a localized infrastructure
failure under the same budget. A second chart-specific numerical failure does
not justify changing the target or widening the cap.

## Allow-list and approval boundary

P0-P2 are local CPU/document/code actions authorized by the user's request;
they need no intermediate click. The one optional persistent host rule for P2
is:

```text
bash /home/ubuntu/python/BayesFilter/scripts/run_hmc_tuning_policy_tests.sh
```

P3 requires one trusted/elevated GPU permission for the exact localized
launcher:

```text
bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_phase9a_chart1_beta0_repair_gpu.sh
```

No broad `bash`, `python`, `codex`, package-manager, network, or arbitrary-GPU
allow-list entry is appropriate. The host/service administrator must persist
the exact launcher rule; the repository cannot alter that external policy.
Once that rule is present, retries inside this unchanged target, hardware
class, and bounded budget do not need another click.

## Execution ledger

Initial skeptical audit: `PASS_WITH_REPAIRS_REQUIRED_BEFORE_EXECUTION`.

The audit found that the closed runner had no scope-start selector, reused the
old seed namespace by default, and wrote only a failure JSON after a late
exception. It also found that the measured-grid caller migration existed in
the source but had never been exercised on the failing chart scope. These are
material reproducibility and execution gaps, so implementation is limited to
the profile and durability repairs above before the GPU run.

The audit found no reason to change the q=20 target, bridge, C5 architecture,
or acceptance-band meaning. The old attempt-05 record remains the baseline
diagnostic and is not warm-started.

### V1 timeout repair review, 2026-09-01

The trusted v1 launch used the exact target signature and a fresh seed
namespace, but exceeded the declared outer bound before any candidate artifact
was durable. The start manifest proves the profile and GPU/memory-growth
settings; exit 137 at the timeout boundary classifies the attempt as
`resource_or_execution`, not as a failed measured candidate. The between-phase
review found no target, bridge, score, chart, or policy mismatch. It did find
that the 32-pair profile implied 97 expensive full-chain calls and that the
runner buffered all candidate records until scope completion. The repair is
therefore limited to a fresh bounded profile, per-scope markers, a longer
signal grace period, and a shell fallback failure record. No previous artifact
is read or overwritten.

The v2 skeptical audit passes with these limits: all eight declared pairs are
still measured explicitly; selection remains replicated efficiency evidence;
held-out verification remains disjoint; the profile is pinned to scope index
3; and the reduced counts are labeled mechanics/localization evidence rather
than convergence or default-readiness evidence. CPU compile, shell syntax,
profile, collision, and policy tests pass before the retry.

### V2 timeout and V3 repair review, 2026-09-01

Attempt-02 completed all six fresh chart checkpoints and entered the pinned
scope, then reached the outer signal at 1,851 seconds without completing its
first full-chain candidate. The Python signal handler and shell fallback now
preserve both `failure.json` and `run_manifest.json`, classified
`resource_or_execution`; no candidate, target, or bridge conclusion is drawn.
The per-call instrumentation added in the same repair will identify the exact
cost once a call completes. The v3 minimal profile is the final bounded
feasibility attempt under this subplan: it retains the smallest legal measured
joint grid and the two-replication/held-out structure, with all counts and
fresh seeds explicit. If v3 also times out, the continuation veto is an
unresolved performance blocker in the current TensorFlow HMC route and a new
performance-focused plan is required; the target and tuning policy must not be
relaxed to force a result.

### V3 provenance-retry review, 2026-09-01

Attempt-03 completed the v3 feasibility probe, but its manifest selected the
historical result-note path because the runner tested only the original repair
profile ID. The shell timeout fallback also wrote `failure.json` without the
conventional `run_manifest.json` alias when Python was killed before servicing
SIGTERM. Both are provenance defects in the harness, not scientific results.
The runner now exposes `profile_id` at the manifest top level, selects the
repair note by the repository-owned repair-plan path, and the wrapper copies a
fallback failure into `run_manifest.json` only when that alias is absent. The
profile, target signature, grid, seeds, cap, and scope pin are unchanged. The
focused CPU suite passes after the patch. A fresh attempt-04 was then used to
directly evidence the corrected manifest; attempt-03 remains immutable.

### V3 attempt-04 provenance-replay review, 2026-09-01

Attempt-04 completed in `370.8858230630867` seconds and returned
`PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL`. The manifest records profile
`chart1_beta0_repair_v3_minimal` at scope index `3`, target signature
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, strict
backend `tensorflow_eigh_strict`, one visible GPU0 logical device, memory
growth configured before initialization, and a peak allocator reading of
`1402670592` bytes. The independent post-run audit recomputed the manifest
hash and passed checks for the start record, no failure collision, route scan,
properness, fresh six-level chart checkpoints, reliability, scope completion,
all four measured pairs, held-out verification, and the selected handoff.
However, attempt-04 intentionally reused the v3 seed namespace to isolate the
manifest patch. This is acceptable as a deterministic provenance regression
check, but it is a hard freshness failure for the final mechanics evidence;
attempt-04 is therefore not used as an independent tuning replication.

The measured pairs were `(0.55,3)`, `(0.55,8)`, `(1.2,3)`, and `(1.2,8)`.
The first two were eligible; both `1.2` pairs were rejected by the declared
`verification_chain_without_movement` veto. The selected `(0.55,3)` handoff is
therefore a localized mechanics candidate only. High acceptance, very short
chains, large descriptive R-hat values, and large pullback-score residuals
remain repair evidence. They do not invalidate the runner, but they prevent a
convergence, whitening, posterior, or default claim.

At that point the focused repair suite (10 tests), HMC tuning policy suite
(18 tests), Python compile, shell syntax, and diff checks passed after the
provenance repair. The TensorFlow trainer-retracing warning remains an
explanatory performance risk for a full replay. The subsequent v4 fresh-seed
retry below was required to close the freshness issue; its target, grid, cap,
and scope pin were unchanged, with only the seed namespace differing. Phase
9B remains closed.

### V4 fresh-seed retry audit, 2026-09-01

The v4 profile is a controlled retry for the single freshness veto above. The
review passes because it changes no target, bridge, architecture, tuning policy,
acceptance band, budget schedule, or scope boundary; it changes only the
reserved seed roots and output directory. The wrapper is pinned to v4 and the
CPU tests verify that the v3 mechanics contract is preserved. A successful v4
run may close the localized subplan; a timeout or numerical veto must be
recorded without widening the cap or opening Phase 9B.

### V4 attempt-05 closeout review, 2026-09-01

Attempt-05 completed in `494.5689085649792` seconds with profile
`chart1_beta0_repair_v4_fresh`, scope index `3`, and a seed namespace distinct
from attempts 01--04. Its manifest hash is
`88bab1482475d59d1c610f1dda4391d8b1424f15b69536895995950e327dfea7` and the
independent hash/contract audit passed. All four declared pairs were measured,
all four were finite and mobile in this fresh realization, held-out
verification passed, and the provisional handoff selected `(epsilon=0.55,L=3)`.
The run used one visible GPU0 logical device with memory growth configured
before initialization and peak allocator `1402670592` bytes. No transition was
run because only one of six scopes was selected.

The fresh seed changed the observed candidate behavior: unlike attempt-04,
the `epsilon=1.2` pairs were not movement-vetoed. This is direct evidence that
the v3/v4 short schedules are seed-sensitive. Acceptance remained high for the
selected candidate and held-out run, triggering the declared repair diagnostic;
short-chain R-hat and ESS values remain descriptive. The result is therefore a
localized program/mechanics pass, not robust tuning evidence. P0--P4 are now
complete for this subplan. The full six-scope replay/performance plan remains
pending and Phase 9B remains closed.
