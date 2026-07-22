# Deterministic LGSSM HMC Full Estimation Rerun Plan

Date: 2026-07-13

Status: `COMPLETE_PASS_SINGLE_FIXTURE_FULL_ESTIMATION_RECOVERY_SCREEN`

Terminal result:
`docs/plans/bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-result-2026-07-13.md`.

## Recommendation

Run a fresh end-to-end estimation campaign in a new artifact namespace. Do not
repin or overwrite the July 9-13 artifacts, and do not reuse the historical
typed-identity authority chain. The new campaign should derive one direct
integrity manifest from its own fixture, target, geometry, mass, corrected
tuning result, Phase 7 config, retained samples, and recovery result.

The core rerun answers a narrow but useful question: does the corrected
deterministic pipeline complete the same `T=120`, 18-parameter synthetic LGSSM
estimation and pass its predeclared sampler and fixture-recovery screens?

## Why A New Campaign Is Required

1. The historical tuning pass used classical split-free R-hat and is invalid
   as evidence for the corrected tuning gate.
2. The current Phase 7 V2 config pins the historical kernel and migration
   identities. It cannot validly consume a newly tuned kernel.
3. The current driver exposes fixture, target, geometry/mass, tuning, and Phase
   7 stages, but it has no implemented final-recovery stage.
4. The driver's current `burnin_sampling` branch does not pass a caller-selected
   Phase 7 config into `run_phase7`; a fresh campaign needs that wiring.
5. The existing academic launcher and campaign module hard-code the historical
   campaign ID, transition identity, serious execution identity, implementation
   inventory, and migration-era result schemas. Passing a new config path to
   that wrapper is therefore insufficient and must fail closed.

These are implementation/integration gaps, not reasons to rerun the old files
in place.

## Scope

### In Scope

- regenerate the deterministic fixture in a new namespace;
- revalidate the TensorFlow/XLA value and score;
- regenerate geometry and dense mass initialization;
- rerun kernel tuning with the corrected shared R-hat implementation;
- derive a fresh Phase 7 transition/config identity directly from the new
  artifacts;
- run an actual-target multicore CPU/XLA smoke;
- run serious burn-in and retained sampling;
- evaluate posterior convergence and synthetic truth recovery;
- write a complete result and run manifest.

### Out Of Scope

- NeuTra training;
- GPU sampling or GPU-readiness claims;
- changing BayesFilter defaults or public APIs;
- comparing HMC with another sampler;
- claiming calibration, generality, superiority, or production readiness from
  one synthetic fixture;
- automatically repairing or retuning after seeing the serious-run result.

## New Artifacts

Proposed config paths:

- `docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json`
- `docs/benchmarks/configs/multidim_lgssm_full_estimation_phase7_2026_07_13.json`

Proposed artifact root:

- `docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/`

Proposed result:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-result-2026-07-13.md`

The artifact root must be created with no-overwrite semantics. If it is already
nonempty, execution stops and uses a new explicit run ID; it must not delete or
replace either historical or concurrent-lane artifacts.

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Can the corrected BayesFilter deterministic HMC pipeline produce converged retained samples and recover the known parameters on the fixed `T=120` lower-triangular LGSSM fixture? |
| Candidate | A newly generated mass and frozen HMC kernel selected by the corrected tuning implementation. |
| Mechanism under test | Shared modern R-hat prevents a kernel with unresolved between-chain location or scale mismatch from passing tuning, after which independent serious sampling tests the frozen kernel. |
| Expected failure mode | Corrected tuning rejects all candidates, or serious burn-in/retained sampling fails modern R-hat/ESS at cap, or converged samples fail the fixture-recovery screen. |
| Promotion criterion | Corrected tuning passes; serious retained samples pass every all-parameter R-hat/ESS gate; every recovery z-score is at most `3.0`; all engineering and numerical vetoes are clear. |
| Promotion veto | Any required target, geometry, mass, tuning, convergence, provenance, or recovery check fails. |
| Continuation veto | Invalid target/score, corrupt or stale artifact, no corrected tuning candidate, nonfinite transition, XLA fallback, serious burn-in failure at cap, retained failure at cap, or wall-time cap. |
| Repair trigger | A localized implementation or serialization failure before serious sampling. Fix visibly and rerun focused checks; do not change scientific thresholds. |
| Explanatory diagnostics | Acceptance, component R-hats, ESS values above their gates, runtime, compile time, posterior means/SDs/MCSEs, and contraction from prior scale. |
| Must not be concluded | General calibration, broad HMC validity, sampler superiority, production/default readiness, GPU readiness, or performance on real data. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Does the complete corrected estimator pass on the same deterministic synthetic target without manual tuning? |
| Exact comparator | Same model, prior, `T=120` fixture policy, and data seed as the historical campaign; historical numerical results are explanatory only and are not reused as inputs. |
| Primary criterion | The final result jointly passes target validity, corrected tuning, serious convergence, retained-sample integrity, and all-parameter truth recovery. |
| Tuning handoff criterion | Acceptance in `[0.65, 0.75]`; `max(rank-normalized split R-hat, folded rank-normalized split R-hat) <= 1.01`; at least `1000` retained verifier draws; finite mechanics; no hard veto. |
| Serious convergence criterion | Every parameter has modern R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`. |
| Recovery criterion | For all 18 parameters, `abs(posterior_mean - truth) / posterior_sd <= 3.0`, with finite posterior SD and mean MCSE recorded. |
| Numerical vetoes | Nonfinite target/score/sample/log-accept value, invalid SPD mass, invalid transformation depth, XLA fallback, source/config drift, or corrupt sample archive. |
| Explanatory only | Smoke output, acceptance within the allowed band, runtime, compilation time, individual posterior differences after the joint gate is evaluated. |
| What will not be concluded | One successful fixture does not establish calibrated coverage, statistical superiority, or readiness for a default change. |
| Preserving artifact | New namespace, direct run manifest, terminal JSON, retained-sample hash, and the result note named above. |

There is no baseline ladder because this is not a method comparison. The
historical run is a bug-regression reference, not a competing estimator.

## Skeptical Plan Audit

| Risk | Resolution |
| --- | --- |
| Wrong baseline | Keep the same model, prior, horizon, and simulation seed to isolate the corrected tuning path; regenerate all derived artifacts. |
| Proxy promoted | Smoke, acceptance, and tuning R-hat are handoff evidence only. Final promotion requires serious retained diagnostics and recovery. |
| Missing stop condition | Every runtime phase has a fixed cap and a terminal pass/fail artifact. |
| Unfair comparison | No method ranking is performed. Historical metrics are not promotion criteria. |
| Hidden truth advantage | The fixture truth equals the prior center and geometry center. This makes the run an internal recovery test, not an honest test of robustness to prior or initialization misspecification. |
| Stale identity | Build a fresh direct identity from the new artifacts; do not migrate or repin historical hashes. |
| Environment mismatch | All HMC sample generation is deliberate multicore CPU with `CUDA_VISIBLE_DEVICES=-1`; TensorFlow XLA remains enabled. |
| Artifact cannot answer question | Add the currently missing final-recovery evaluator and preserve the private retained archive plus public aggregate result. |
| Seed overfitting | Tuning and serious sampling use distinct predeclared root seeds; serious failure cannot trigger an in-campaign retune. |
| Diagnostic mismatch | Tuning and Phase 7 must import the same shared R-hat helper and record the exact definition and both component maxima. |

Audit verdict: `PASS_AFTER_PHASE0_INTEGRATION_GAPS_CLOSE`.

## Phase 0: Close Integration Gaps

Objective: make a fresh run possible without inheriting historical authority or
artifact assumptions.

Required implementation:

1. Add a new tuning config with the same scientific target and seeds but the
   new artifact root.
2. Add a fresh-run Phase 7 config schema or mode containing direct references
   to the new config, fixture, XLA gate, geometry, mass, kernel, and private
   replay. It must not require migration certificates or legacy hash pins.
3. Add a deterministic Phase 7 config builder that runs only after corrected
   tuning passes and derives the transition identity from the exact executable
   mechanics.
4. Add a fresh-campaign launcher/context whose campaign, transition, execution,
   source-inventory, config, output, and result identities are derived from the
   new inputs. Do not reuse, edit, or parameterize the historical academic
   authority/campaign modules as the new execution authority.
5. Wire the caller-selected Phase 7 config through smoke and the new serious
   launcher. The controller must accept the fresh-run schema without consulting
   migration certificates, historical V1 configs, adoption records, or fixed
   expected-identity constants.
6. Add a final-recovery evaluator/stage that verifies and reads the private raw-
   parameter retained archive, recomputes diagnostics, and writes posterior
   summaries and recovery rows.
7. Add no-overwrite and source-inventory checks. Hashes are reproducibility and
   integrity metadata, not separate human-authorization objects.

Gate: focused unit/integration tests, compilation, static XLA/non-JIT scans, and
an end-to-end mocked artifact round trip pass. No HMC experiment runs in this
phase.

Stop: do not start the fixture or any HMC runtime if the fresh config still
depends on historical kernel/adoption identities or if final recovery cannot be
reconstructed from a private sample fixture.

## Review Record

Local skeptical review found that the existing academic wrapper hard-codes the
historical transition/execution identities and cannot execute this fresh plan.
Claude was probed successfully (`CLAUDE_PROBE_OK`) and could read the single
plan path (`FULL_RERUN_PLAN_READ_OK`). Broad one-file review prompts returned no
text, so the review was decomposed into fixed-token bounded checks. Claude
returned `FEASIBILITY_MISSING_HARDCODED_IDENTITY_REPAIR`, agreeing with the
local finding. The plan was revised above before implementation began.

Claude did not return a token for the separate recovery-gate interpretation
probe. The evidence burden therefore remains local: the `3 posterior SD` check
is retained only as a predeclared single-fixture recovery screen, and the plan
forbids calibration, coverage, generality, or superiority claims.

After revision, a final bounded one-path Claude feasibility recheck was
attempted. The execution environment rejected that new call at its external-
disclosure boundary, so it was not retried or routed around. No post-revision
Claude convergence claim is made. Local implementation review then closed the
original blocker with an isolated V3 config/controller path, a direct fresh
launcher, no-overwrite enforcement, independent recovery, and preservation of
V1/V2 behavior. The Phase 0 result is recorded in
`docs/plans/bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-phase0-result-2026-07-13.md`.

## Phase 1: Preflight And Diagnostic Lock

Objective: freeze code, definitions, thresholds, seeds, and commands before any
research-result-producing action.

Required checks:

- rerun the rank-normalized tuning/Phase 7 equivalence regression;
- confirm tuning and Phase 7 both record exactly
  `max(rank-normalized split R-hat, folded rank-normalized split R-hat)`;
- confirm tuning and serious-sampling seeds differ;
- confirm all output paths resolve under the new root;
- record git commit plus SHA-256 of every dirty in-scope source file;
- verify the old artifact root is not an output dependency;
- run focused driver, tuning, convergence, Phase 7, and recovery tests.

Gate: one immutable preflight manifest lists source hashes, config hash, seeds,
thresholds, commands, device policy, and output paths.

## Phase 2: Fresh Fixture And Target Validation

Objective: rebuild the exact deterministic input and independently validate the
target before tuning.

Actions:

1. Generate the `T=120` fixture twice from simulation seed `(20260709, 301)` and
   require byte/stable-payload identity.
2. Check shapes, stationarity, Lyapunov residual, finite observations/states,
   and contract consistency.
3. Run the XLA value/score gate at the truth and fixed perturbation points.
4. Compare the XLA score with the existing independent reference/finite-
   difference checks under their reviewed tolerances.

Gate: deterministic fixture identity and every target/score validity check pass.

Stop: a target or score mismatch invalidates the harness and stops the whole
campaign. It is not a tuning failure.

## Phase 3: Fresh Geometry And Mass

Objective: regenerate initialization geometry rather than copying the old
geometry or mass.

Actions:

- run the existing low-rank quadratic geometry stage from the new fixture;
- convert the accepted precision to a dense mass artifact;
- require finite arrays, positive eigenvalues, condition number at most
  `100000`, precision/covariance reconstruction error at most `1e-8`, and no
  fallback geometry;
- record the fact that the center is the prior mean/truth and restrict the
  conclusion accordingly.

Gate: geometry and mass artifacts pass their existing hard checks and are bound
to the new target/config/fixture hashes.

Stop: do not tune with fallback, stale, indefinite, or poorly reconstructed
mass geometry.

## Phase 4: Corrected Kernel Tuning

Objective: select a fresh frozen kernel using the repaired verifier.

Fixed policy:

- tuning root seed: `(20260709, 501)`;
- target acceptance: `0.70`;
- acceptance band: `[0.65, 0.75]`;
- four verification chains;
- `500` initial burn-in transitions;
- check every `250` retained draws;
- minimum `1000`, maximum `2000` retained verifier draws;
- combined rank-normalized split/folded R-hat threshold: `1.01`;
- XLA on, no runtime `GradientTape`, no geometry fallback;
- deterministic repair loop limited to the predeclared five attempts.

Required artifact fields:

- both R-hat component maxima and their maximum;
- diagnostic-definition/version identifier;
- acceptance and finite target/log-accept health;
- selected step size, leapfrog count, trajectory length, and mass identity in
  the private replay;
- source/config/fixture/target/geometry/mass hashes;
- explicit `final_kernel_requires_serious_sampling_pass=true`.

Gate: corrected tuning passes every handoff criterion. This means only that a
kernel may proceed to serious testing.

Stop: if no candidate passes within the fixed attempt/budget policy, write a
tuning failure result and stop. Do not loosen `1.01`, increase attempts, choose
a kernel manually, or start Phase 7.

## Phase 5: Fresh Identity And Actual-Target Smoke

Objective: prove that the new frozen kernel can be reconstructed and executed
through the real two-worker path.

Actions:

1. Build the fresh Phase 7 config directly from the new tuning output.
2. Round-trip the private replay and require exact executable mechanics.
3. Run a tiny `/tmp` smoke with two workers, two chains per worker, CPU XLA,
   deterministic state handoff, raw-parameter mapping, and the shared
   diagnostic implementation.

Gate: process startup/teardown, XLA compile, replay, transition, state handoff,
finite output, and artifact serialization pass.

Smoke diagnostics are engineering evidence only. A smoke must not reject or
promote the statistical candidate based on short-chain R-hat or ESS.

## Phase 6: Serious Burn-In And Retained Sampling

Objective: independently test the frozen kernel and produce the posterior
sample archive.

Fixed execution:

- serious root seed: `(20260713, 701)`, not used during tuning;
- deliberate CPU-only sampling with `CUDA_VISIBLE_DEVICES=-1`;
- TensorFlow `float64`, XLA JIT enabled;
- two persistent workers, two chains per worker, four chains total;
- burn-in: `2000` initial, `1000` extensions, last-`1000` diagnostic window,
  hard cap `16000` transitions per chain;
- retained sampling: `4000` initial, `2000` extensions/checks, cumulative hard
  cap `40000` draws per chain;
- all-parameter modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`;
- finite target, sample, and log-accept values required;
- no manual thinning, chain removal, reseeding, or mid-run tuning.

Gate: both burn-in and retained stages pass before their caps, and the private
retained archive verifies shape, finiteness, config hash, replay hash, and file
hash.

Stop: a cap failure rejects this frozen candidate. It does not invalidate the
target or broad HMC direction unless a target/implementation veto also fired.
There is no automatic retry in this campaign.

## Phase 7: Final Estimation And Recovery

Objective: turn the retained archive into the final estimation result.

Actions:

1. Independently reload and verify the private retained archive.
2. Recompute modern R-hat, bulk ESS, and tail ESS from the complete retained
   draws; require exact agreement with the terminal Phase 7 aggregate within
   serialization tolerance.
3. For every raw parameter, report truth, posterior mean, posterior SD, mean
   MCSE, 5/50/95 percentiles, and
   `abs(posterior_mean - truth) / posterior_sd`.
4. Require finite positive SD and recovery z-score at most `3.0` for all 18
   parameters.
5. Record prior scale and posterior contraction as explanatory diagnostics; do
   not make contraction a new pass criterion.

Gate: convergence, retained integrity, and every recovery row pass. The result
may then say that this fixed synthetic fixture was successfully estimated.

Stop: if convergence remains valid but recovery fails, classify it as a fixture
recovery failure, not automatically as an HMC implementation failure. Preserve
the failed rows and do not alter the threshold.

## Phase 8: Result And Closeout

Write the terminal result with:

- exact command and environment;
- git commit and in-scope dirty source hashes;
- CPU/XLA status, TensorFlow/TFP versions, thread topology, seeds, and wall time;
- all artifact paths and SHA-256 values;
- decision table and inference-status table;
- separate engineering, sampler-validity, and scientific-interpretation
  ledgers;
- hard vetoes, viable-candidate status, and an explicit statement that no
  stochastic ranking was attempted;
- post-run red team: strongest alternative explanation, weakest evidence, and
  what would overturn the fixture-level conclusion.

The terminal result is pass or fail. Closeout does not rerun or repair the
candidate.

## Command Skeleton

Commands become executable only after Phase 0 implements and tests the fresh
config and final-recovery wiring.

```text
PY=/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11
DRIVER=docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
CONFIG=docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json
P7CONFIG=docs/benchmarks/configs/multidim_lgssm_full_estimation_phase7_2026_07_13.json
ROOT=docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --stage fixture
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --stage xla_score
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --stage geometry_mass
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --stage kernel_tuning

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY scripts/build_hmc_full_estimation_phase7_config.py \
  --tuning-config $CONFIG --output $P7CONFIG

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --phase7-config $P7CONFIG \
  --stage burnin_sampling --phase7-smoke \
  --phase7-output-dir /tmp/bayesfilter-full-estimation-smoke

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY scripts/run_hmc_full_estimation_campaign.py \
  --config $P7CONFIG --campaign-root $ROOT/phase7_campaign

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-full-rerun \
  $PY $DRIVER --config $CONFIG --phase7-config $P7CONFIG \
  --stage final_recovery
```

The supervisor should capture each command in a structured manifest and inspect
bounded progress. It should not introduce certificate-bound human approval
phrases between ordinary academic phases.

## Pre-Mortem

| Misleading outcome | Cheap discriminator/control |
| --- | --- |
| The run passes mainly because truth, prior center, and geometry center coincide | State this limitation; report contraction and run an offset-truth fixture later before broader claims. |
| Tuning passes one seed but serious sampling fails another | Treat serious sampling as the independent candidate test and stop; do not retune on the serious seed. |
| R-hat passes while a chain-scale problem remains | Folded rank-normalized split R-hat is part of the shared maximum in both stages. |
| Recovery passes because posterior SD is very broad | Report posterior/prior scale contraction and raw errors; do not upgrade the fixture pass into precision or calibration claims. |
| Recovery fails because of data realization rather than sampler failure | Separate convergence validity from truth-recovery failure in the terminal result. |
| A stale historical file is accidentally consumed | New-root-only preflight and exact input manifest fail closed before runtime. |
| The sample archive exists but does not match the executed kernel | Bind config, replay, transition identity, shape, and file hash, then verify independently before recovery. |
| XLA silently falls back | Require `jit_compile=true`, `use_xla=true`, compile evidence, and `jit_compile_false_runtime_executed=false` in every runtime artifact. |

## Optional Follow-Up For Scientific Strength

The fixed fixture is intentionally favorable because truth equals the prior and
geometry center. After the core campaign closes, a separate confirmatory plan
should use an offset-but-stable truth, a fresh data seed, and geometry initialized
without access to truth. That is required before making a meaningful claim about
estimation robustness. It is not required to determine whether this tuning bug
was fixed end to end.

## Execution Boundary

At drafting time, this file proposed the campaign only and did not itself
authorize or start an experiment. The user subsequently requested execution;
Phase 0 closed, the remaining phases ran under the stated gates, and the
terminal result linked at the top of this file now records the completed run.
