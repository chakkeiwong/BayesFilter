# BayesFilter NeuTra All-Executable-Models End-to-End Python Plan

Date: 2026-07-18

Campaign ID: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Status: `ACTIVE_SERIOUS_ATTEMPT_02`

## Research intent ledger

| Field | Frozen intent |
| --- | --- |
| Main question | Can one repository-owned Python program freshly train, natively tune, and run exact-gradient NeuTra HMC for every BayesFilter model/filter cell that already has a retained executable learned-NeuTra target, and recover the generating truth under the prospective one-seed policy? |
| Candidate | A fresh target-specific dense-IAF NeuTra transport followed by exact transformed-gradient HMC. |
| Comparator | Generating truth is the primary scientific comparator. The affine initialization and held-out reverse KL are training diagnostics only. Historical plain HMC is provenance/debugging evidence and is not required for promotion. |
| Expected failure modes | Invalid target/status values, inherited geometry drift, failed training numerics, frozen/trainable transport mismatch, native tuner failure, acceptance outside `[0.65, 0.75]`, nonfinite/energy failure, warm-up or retained convergence cap, or truth-tail failure. |
| Promotion criterion | For each executable cell: healthy fresh final training; native fixed-transport tuning admitted at target acceptance `0.70`; converged retained physical draws; no hard veto; and every declared parameter has `p_truth >= 0.05`. |
| Promotion veto | Any invalid/nonfinite target or transport result, tuning acceptance outside `[0.65, 0.75]`, nonidentity NeuTra-coordinate mass, fixed-grid repair use, failed rank/folded R-hat or ESS, energy/status veto, or any `p_truth < 0.05`. |
| Continuation veto | Invalid harness/target identity, artifact overwrite, GPU/XLA/memory-growth contract failure, campaign attempt budget exhaustion, or a cell failure that invalidates shared orchestration. A valid candidate failure in one cell does not prevent later independent cells from running. |
| Repair trigger | Local serialization, subprocess, XLA compilation, resource, or harness failure with unchanged target/method/budget; a marginal truth tail `0.003 <= p_truth < 0.05` nominates a later second-seed run but does not silently consume one in this campaign. |
| Explanatory diagnostics | Training loss, held-out reverse KL, acceptance within the valid band, runtime, posterior moments, interval endpoints, and per-parameter tail magnitudes. |
| Forbidden conclusion | No universal NeuTra validity, sampler superiority, filter exactness, calibration, cross-fixture robustness, production/default readiness, or ranking among models. |

## Scope registry

The executable set is the intersection of BayesFilter-owned targets with
preserved end-to-end learned-NeuTra evidence and a current direct adapter
factory. It is not the broader `/home/chakwong/python` evidence inventory.
The executable registry binds current direct mathematical target signatures.
Historical P4/P6/structural hashes were typed campaign identities that also
bound their old registry and execution surfaces; they remain provenance, not
the signatures stamped on fresh direct-target artifacts.

| Cell | State for this campaign | Target route | Re-entry if blocked |
| --- | --- | --- | --- |
| `LGSSM-EXACT` | executable | exact 18D deterministic LGSSM/Kalman target | N/A |
| `PP-UKF` | executable | six-parameter predator-prey principal-square-root UKF posterior | N/A |
| `PP-SGQF` | executable | six-parameter predator-prey frozen level-2 SGQF posterior | N/A |
| `SIR-SGQF` | executable | three-parameter Austria SIR frozen level-2 SGQF posterior | N/A |
| `STR-UKF` | executable | five-parameter structural principal-square-root UKF posterior | N/A |
| `SVX-SGQF` | blocked inventory | no frozen SGQF level passed filter admission | filter admission |
| `SVX-ZC` | blocked inventory | production source-route mismatch | source-route design |
| `KSC-UKF` | blocked inventory | dense-reference value/score admission failed | filter admission |
| `PP-ZC` | blocked inventory | generic retained-grid route is production-ineligible | source-route design |
| `STR-ZC` | blocked inventory | extension target is not designed | extension-target design |
| `SIR-UKF` | blocked inventory | preserved GPU/CPU score-parity blocker | GPU score parity |
| `SIR-ZC` | blocked inventory | observed-data parameter-score route is absent | observed-data score route |

Blocked cells must appear in the emitted machine-readable registry and
aggregate result. They must not be launched or counted as NeuTra failures.

## Default and assumption audit

| Choice | Provenance and status | Justification | Misleading failure mode | Earliest diagnostic |
| --- | --- | --- | --- | --- |
| Dense IAF recipe families | Target-specific reviewed 2026-07-14 to 2026-07-17 campaigns; warm-start hypotheses | These are the only locally tested capacity/rate families for the exact targets | Transfer could freeze a stale convenient recipe | Fresh common-heldout screen; final weights are retrained from scratch |
| 500 screen and 5,000 final steps | Same target-specific campaigns; reviewed baseline | Preserves the prior serious protocol and makes this rerun comparable | Loss may still be improving at the cap | Progress records and held-out target status; no claim from loss alone |
| Affine geometry | Hash-bound historical mass/Laplace artifact, or identity source-prior geometry for `STR-UKF`; warm start | Reduces training difficulty without changing the learned target | Stale or target-mismatched geometry can create a false failure | File hash, coordinate convention, shape, finiteness, and nonsingularity checks before training; geometry is not current target identity evidence |
| LGSSM truth-centered affine center | Fixture prior mean equals generating raw truth; reviewed fixture design | It is part of this deliberately favorable synthetic target | Could be misread as general posterior geometry knowledge | Artifact labels it `prior_mean_raw_coordinates_truth_fixture`; nonclaim retained |
| Held-out reverse KL selection | Existing target-specific protocols; proxy nomination only | Common stateless batches compare recipes fairly | A good predictor need not yield valid HMC | It cannot promote; exact downstream tuning and HMC are mandatory |
| Four chains | Existing convergence policy | Required for stable split/folded multi-chain diagnostics | Identical initialization could conceal exploration failure | Deterministic dispersed chain starts and modern R-hat |
| Native tuning target `0.70` and band `[0.65,0.75]` | Current BayesFilter fixed-transport tuner policy | Corrects the invalid historical high-acceptance tuning | An optional legacy fixed-grid branch could bypass native adaptation | Empty fixed-grid config plus returned-artifact assertion |
| Identity mass in trained `z` | NeuTra coordinate contract | A second mass construction would duplicate transport whitening | Hidden mass adaptation changes the comparison | Assert `mass_policy=fixed_identity_z` and no mass adaptation |
| Warm-up/retained caps | Current sequential NeuTra policy | Adaptive evidence collection with bounded cost | Fixed samples could be mistaken for adequate convergence | Warm-up R-hat `<=1.05`; retained full diagnostic through 10,000 cap |
| One seed | Owner cost policy | This is a bounded diagnostic, not beyond-reasonable-doubt evidence | A marginal stochastic miss could be overinterpreted | Three-way truth-tail classification with explicit second-seed handoff |

## Evidence contract

The Python runner must call these repository-owned implementations rather than
reimplementing their algorithms:

- `train_plain_dense_iaf` for batched GPU/XLA training;
- `load_frozen_neutra_artifact` for immutable transport loading;
- `tune_fixed_transport_hmc_kernel` for native fixed-mass dual averaging and
  disjoint verification;
- `run_sequential_neutra_hmc` for retained warm-up and adaptive sampling; and
- `rank_normalized_hmc_diagnostics` for the maximum of rank-normalized split
  and folded rank-normalized split R-hat plus bulk/tail ESS.

Tuning must use `target_accept_prob=0.70`, acceptance band `[0.65,0.75]`, four
chains, fixed identity mass in trained `z`, exact transformed gradients, no
fixed-grid scale repair, and fresh verifier draws. Warm-up extends until modern
R-hat is at most `1.05` or 10,000 draws per chain. Retained sampling extends
until modern R-hat is at most `1.01`, minimum bulk ESS is at least `1000`, and
minimum tail ESS is at least `400`, or 10,000 draws per chain.

The scientific screen is the two-sided smoothed empirical tail probability at
the generating truth. A cell passes at `p_truth >= 0.05` for every parameter,
is marginal at `0.003 <= p_truth < 0.05`, and is a severe failure at
`p_truth < 0.003`. Sampler invalidity takes precedence over truth-tail results.

Every cell writes a fresh directory containing training configs/progress/state,
frozen transport, tuning result, separate warm-up and retained tensors,
convergence and truth-tail diagnostics, result JSON/Markdown, and a run
manifest with commit, dirty-worktree disclosure, command, Python/TensorFlow/TFP
versions, GPU/memory-growth/XLA/TF32 state, target signature, seeds, wall time,
plan path, and artifact paths. The aggregate result reports executable,
passed, marginal, failed, and blocked inventory counts without ranking cells.

## Compute and attempt budget

- Hardware class: one visible NVIDIA GPU, sequential cells, XLA enabled,
  TensorFlow memory growth configured before logical-device initialization.
- Screen budget: at most four 500-step target-specific recipes per executable
  cell.
- Final budget: exactly one fresh 5,000-step training per cell after selection.
- Tuning budget: one native tuning call per cell over the declared leapfrog
  ladder; no agent-selected kernel and no unplanned grid extension.
- Sampling budget: at most 10,000 warm-up plus 10,000 retained draws per chain,
  four chains per cell.
- Repair budget: one localized infrastructure retry per cell under the same
  target, method, hardware class, seeds, and budgets. Scientific failures are
  recorded, not automatically retuned around.
- Campaign stop: shared harness invalidity, output collision, environment veto,
  or budget exhaustion. An independent cell rejection does not stop later
  cells.

## Phases and repair/continue procedure

### Phase 0: registry and reusable composition

Implement a target-agnostic orchestration module, a target-specific registry,
one CLI, and contract tests. The registry owns model facts, transforms,
geometry provenance, recipes, seeds, and blocked classifications. The shared
module owns selection, fresh final training, tuning, stopping, diagnostics,
artifacting, and terminal classification.

Exit: registry validation and focused CPU-hidden tests pass.

### Phase 1: plan and implementation audit

Review mathematical target equality, target signatures, transformed density,
truth transforms, recipe selection, tuning authority, seed separation,
artifact immutability, NumPy/Python-loop boundaries, and duplication. Search
the new implementation for copied sampler/tuner/diagnostic logic and for
imports from historical benchmark scripts.

Exit: write a visible review record. Any material issue is patched and focused
checks rerun before launch.

### Phase 2: trusted GPU/XLA preflight

Run an escalated device probe, then execute registry validation and one tiny
training/transport/tuning composition smoke under a fresh preflight root. The
smoke is engineering evidence only.

Exit: GPU visible, memory growth verified, XLA target and training finite, and
no shared harness veto.

### Phase 3: all-cell end-to-end campaign

Launch the single Python campaign command. Each cell runs in its own fresh
subprocess so TensorFlow device initialization and failure isolation are
repeatable. Continue after a valid cell candidate failure; stop only on a
shared continuation veto.

Exit: every executable cell has one terminal result or a precise preserved
infrastructure blocker, and every blocked inventory row remains unlaunched.

### Phase 4: terminal review and handoff

Audit manifests, target signatures, native tuning invariants, acceptance,
warm-up separation, modern R-hat/ESS, status/energy vetoes, truth-tail
classification, and aggregate counts. Write the campaign result and reset
memo. Do not promote descriptive runtime or acceptance differences into a
ranking.

Exit: terminal result states exactly what passed, failed, remained blocked,
what is descriptive only, and what next evidence is justified.

At the end of every phase: run its local checks; write or refresh its result
record; refresh the next phase handoff; review that handoff for correctness,
feasibility, artifact coverage, and boundary safety; continue when no real
continuation blocker exists. Local infrastructure repair uses a fresh attempt
directory and records the failed attempt, repair, checks, wall time, and
remaining budget.

## Skeptical pre-execution audit

- Wrong baseline: avoided by making generating truth primary and labeling
  affine/plain-HMC evidence as diagnostic only.
- Proxy promotion: prevented because held-out reverse KL can nominate a recipe
  but cannot admit a transport or sampler.
- Missing stop conditions: bounded training, tuning ladder, HMC caps, repair
  count, and continuation vetoes are explicit.
- Unfair comparison: no cross-model speed or accuracy ranking is attempted;
  every target receives its reviewed target-specific recipe family.
- Hidden assumptions: geometry, favorable LGSSM truth centering, one seed,
  chain starts, thresholds, and transferred recipes are disclosed above.
- Stale context: executable membership is reconciled against the 2026-07-17
  terminal registry; `STR-UKF` includes its later retained truth-tail result.
  The current LGSSM signature is freshly issued from the unchanged fixture,
  config, source contract, observations, parameter order, coordinate convention,
  horizon, and target math plus the current implementation source hashes; the
  historical mass is used only as a coordinate-compatible training warm start.
- Environment mismatch: serious work requires trusted GPU/XLA and memory
  growth; CPU-hidden checks are labeled as tests only.
- Non-answering commands: the serious command must produce fresh training,
  native tuning, adaptive samples, diagnostics, and manifests, not merely
  replay old JSON.
- Duplication: historical benchmark scripts are evidence/provenance only; new
  code may not import them or copy their kernel selection and convergence
  algorithms.

Audit verdict: `PASS_TO_IMPLEMENTATION`. The plan answers the stated question
provided Phase 1 proves the runner enforces native tuning and contains no
second sampler, tuner, or convergence implementation. Serious execution is not
authorized to start before that review record passes.

## 2026-07-18 execution repair addendum

Serious attempt 01 exposed two harness defects and is not scientific evidence.
The fresh LGSSM transport trained successfully, but every native-tuner candidate
failed before a single HMC acceptance statistic was computed because the shared
end-to-end adapter discarded required target-status fields. Its result is
therefore `TUNING_HARNESS_INVALID`, not evidence for or against NeuTra. The
campaign process later ended during PP-UKF screening without a structured
terminal aggregate result.

The repaired composition now:

- forwards the normalized status mapping issued by
  `batch_native_value_status_target_fn`, including floor, minimum innovation,
  condition-estimate, and condition-availability telemetry;
- replaces the fake-chain preflight with the real BayesFilter fixed-transport
  HMC runner and requires one finite error-free tuning result;
- runs final 5,000-step training through five 1,000-step XLA segments using
  `resume_infrastructure_from`, with unchanged configuration hash, global step,
  stateless noise, learning-rate schedule, weights, and Adam state; and
- freezes the transport only in the terminal segment and preserves every
  segment directory and exact parent checkpoint lineage.

The Python loop over five infrastructure segments is orchestration only. Every
optimization segment remains one batched `tf.while_loop` XLA program; there is
no Python loop over training examples, batch rows, HMC draws, chains, or
optimization steps.

Focused deterministic checks prove bitwise equality of uninterrupted and
segmented weights and Adam moments on a tiny fixture, stable scientific config
hashes across fresh segment directories, exact parent lineage, and
terminal-only freeze. All five executable target factories provide the complete
normalized HMC telemetry surface. A fresh real-HMC GPU/XLA preflight must pass
before serious attempt 02 is launched. The attempt-02 scientific contract,
seeds, target, recipes, budgets, and criteria remain unchanged.
