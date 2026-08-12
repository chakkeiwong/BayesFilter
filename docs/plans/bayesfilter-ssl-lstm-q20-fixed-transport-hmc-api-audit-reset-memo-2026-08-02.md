# q=20 Fixed-Transport HMC API Audit Reset Memo

Date: 2026-08-02
Status: `CHART_A_L10_CANDIDATE_NOMINATED_SEQUENTIAL_VALIDATION_NOT_RUN`

## 2026-08-03 Terminal Update

The intended multi-trajectory public-API tuning procedure has now run for
Chart A with canonical grid `(5,10,15,20,25)` plus user-requested `L=3`. The
six arms ran concurrently in CPU-only XLA processes under a 12-hour wall cap.
The grid completed in `26959.7411 s` (`7.4888 h`).

Only `L=10` passed the public tuner's screen and fresh verification:

```text
num_leapfrog_steps = 10
step_size = 0.4148806556986277
fresh verification mean acceptance probability = 0.7235869085131437
```

The selected candidate has finite required tensors, valid target status, and
no candidate hard veto. TFP native divergence is unavailable, not zero. Its
finite `max_abs_log_accept_ratio=1e100` is explanatory-only under the binding
policy, but it is a material reason to require chunk-level sequential numerical
validation before any convergence or posterior claim.

The other arms did not pass: `L=3` and `L=15` entered the screen band but
failed fresh verification; `L=5`, `L=20`, and `L=25` produced no
screen-qualified step. These are candidate/tuning failures, not research-lane,
target, transport, or model failures.

Authoritative result:
`docs/plans/bayesfilter-ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-result-2026-08-03.md`.
Merged candidate artifact:
`docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/merged-tuning-result.json`.

Do not rerun this grid, tune Chart B, or launch sequential HMC from the old
2026-08-02 next-step text below. No sequential HMC, R-hat, ESS, posterior, or
scientific validation has run. The next justified action, if separately
authorized, is a bounded Chart A sequential-HMC validation plan using the
frozen `L=10` kernel and the repository sequential controller.

## 2026-08-02 Terminal Update

The TensorFlow-only public fixed-transport tuner, corrected diagnostic roles,
and CPU/XLA target campaign are now implemented and executed. Focused checks
pass `36/36`. Concurrent Chart A/B tuning completed in `2278.2646 s` with no
hard numerical/status veto, but neither chart produced a fresh `L=2` screen in
the acceptance band `[0.65,0.75]`:

| Chart | Tuned steps | Fresh screen acceptance | Decision |
| --- | --- | --- | --- |
| A | `0.647273, 0.540312, 0.589960` | `0.781646, 0.637710, 0.843694` | No viable kernel |
| B | `0.673863, 0.528120, 0.593117` | `0.829631, 0.939211, 0.817549` | No viable kernel |

The sequential controller was correctly not launched. The campaign used a
conservative cumulative `4178.2646 s` of `20000 s`; `15821.7354 s` remains
unused. The terminal result is
`docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-result-2026-08-02.md`.

Next justified action: write and audit a target-specific fixed-step repair with
fresh seeds and longer screens. Chart A should densely probe the observed
transition region; Chart B must extend toward larger steps because all observed
screens were above band. Do not launch sequential chains, relax the acceptance
gate, or reintroduce `L=1` before a fresh kernel is admitted.

## Research Intent Ledger

| Field | Decision |
| --- | --- |
| Main question | Can either trained q=20 NeuTra chart support a valid fixed-HMC kernel before sequential posterior sampling? |
| Candidate | The frozen trained transport, with HMC in transport `z` coordinates and repository-owned tuning. |
| Expected failure mode | Step size or trajectory length gives nonfinite transitions or acceptance outside the declared band. |
| Promotion criterion | Repository-owned fixed-transport tuning and fresh verification pass with `L>=2`, finite values, zero exposed native divergences, and acceptance inside the declared bound. |
| Promotion veto | Nonfinite/NaN state, target, score, or log acceptance; available positive native divergence; acceptance outside the declared bound. Native divergence unavailability is recorded but is not a veto or a zero count. |
| Explanatory only | Finite `max(abs(log_accept_ratio))`, signed log-accept tails, runtime, RSS, and descriptive differences between viable arms. |
| Continuation veto | Invalid checkpoint/transport identity, broken target, corrupted artifact, or campaign budget exhaustion. |
| What must not be concluded | Tuning acceptance does not establish convergence, posterior validity, chart ranking, sampler superiority, or scientific validity. |

## Skeptical Plan Audit

The prior plan failed audit for three material reasons:

1. It used a custom `tune_charts()` grid around
   `build_reusable_full_chain_tfp_hmc_runner` rather than the public
   fixed-transport tuning API.
2. It admitted `L=1` candidates. These are now forbidden; the relevant API
   boundary enforces `L>=2`, while its reviewed default grid is
   `(5, 10, 15, 20, 25)`.
3. It promoted finite `abs(log_accept_ratio)>1000` from an explanatory
   energy-related tail diagnostic into a hard veto.

TFP `HamiltonianMonteCarlo` does not expose a native divergence boolean in the
installed TFP `0.25.0`. BayesFilter must record this as
`not_exposed_by_kernel`; unavailability is not zero divergence. Under the final
user policy, unavailability is not a veto. Only an available positive native
divergence vetoes a candidate, and no energy/log-accept proxy may replace it.

The domain-specific tuner also imports and uses NumPy in its runtime tuning,
selection, artifact, and verification path. That is legacy migration debt and
is ineligible for new claim-bearing execution under the current NumPy
diagnostic-only policy. The semantic API choice remains correct, but it must be
migrated to TensorFlow/Python-standard-library operations under a bounded plan
before a serious run. The narrow boundary fixes in this audit do not promote
the module or waive that migration.

## API Audit

| API | Intended role | Decision for this lane |
| --- | --- | --- |
| `bayesfilter.inference.hmc_tuning` | Low-level tuning diagnostics and policy primitives | Not the target-facing entry point. |
| `bayesfilter.inference.hmc_kernel_tuning.tune_hmc_kernel` | Generic model-facing HMC tuning with geometry and optional windowed mass adaptation | Not selected because the trained NeuTra map already defines transport coordinates. |
| `bayesfilter.inference.fixed_transport_hmc_tuning.tune_fixed_transport_hmc_kernel` | Fixed trained-transport HMC tuning in `z`, identity `z` mass, independent step tuning across an `L` grid, then fresh frozen-kernel verification | Correct semantic entry point. XLA now defaults on, but the runtime NumPy dependency must be migrated before claim-bearing use. Do not replace its orchestration with a launcher-owned grid. |
| `bayesfilter.inference.hmc_operational_broad_grid` | Lower-level process-parallel broad-grid orchestration over prepared mass handoffs | Not the direct model-facing tuning API. |
| `bayesfilter.inference.neutra_hmc.run_sequential_neutra_hmc` | Sequential warm-up and retained fixed-kernel sampling after admission | Downstream only; not a tuner. |

## Corrected Historical Evidence

The original `L>=2` artifact has finite tuning arms inside `[0.55, 0.85]` for
both charts. Chart A's only confirmation fails its `[0.35, 0.95]` acceptance
bound (`min=0.193247`). Chart B has no confirmation. All fixed-HMC native
divergence fields were unavailable. Therefore no fixed kernel is admitted and
no ranking is statistically supported.

The later repair artifact selected `L=1` and is ineligible in full. Its finite
log-accept tails are explanatory only, and its unavailable divergence telemetry
prevents a validity pass.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Fixed-transport API | Repository domain-specific API and public export | Reviewed route for this target type | Bypassing it changes tuning semantics | Assert result runtime and artifact schema. |
| Identity mass in `z` | Fixed-transport API contract | Reviewed fixed-transport baseline | Learned map may not fully whiten the target | Acceptance/convergence diagnostics after eligible tuning. |
| `L>=2` | Owner correction, 2026-08-02 | Hard boundary | `L=1` degenerates the intended HMC trajectory test | Config construction must reject any `L=1`. |
| Default `L=(5,10,15,20,25)` | Fixed-transport API default | Reviewed API default, not yet target-specific evidence | Grid may miss a target-specific viable trajectory | API-owned step tuning and fresh verification per `L`. |
| Acceptance target `0.70`, pass band `[0.65,0.75]`, repair band `[0.55,0.85]` | Fixed-transport API defaults | Reviewed API defaults | Bounds may be inefficient for this target | Record per-chain and pooled acceptance on fresh seeds. |
| XLA on | Repository XLA policy and user instruction | Required | Non-XLA timing or mechanics do not answer the lane question | Artifact must record `use_xla=True` and compiled execution. |
| Native divergence | Final user policy and repository provenance discipline | Conditional hard gate | Unavailability could be mislabeled as zero or replaced with an energy proxy | Tests require unavailable/null provenance and veto only an available positive count. |
| TensorFlow-only tuning runtime | Repository NumPy diagnostic-only policy | Hard implementation boundary | Legacy NumPy tuning/selection creates an ineligible claim-bearing route | Migrate the fixed-transport tuner and run focused parity tests before launch. |

## Decision And Next Step

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not rerun the custom fixed-HMC grid | Correct BayesFilter API was bypassed | Historical evidence invalid or incomplete | None about the procedural mismatch | Retire the custom tuner from claim-bearing use | Target or transport invalidity |
| Do not launch fixed-HMC sampling yet | Both corrected `L=2` tuning ladders failed the fresh acceptance screen | No hard mechanics veto; kernel-promotion criterion failed | Whether a denser, longer fresh fixed-step screen finds an in-band handoff | Run a target-specific step-size repair through the same public API, then launch sequential sampling only after admission | Posterior convergence or HMC readiness |

## Inference Status

| Evidence class | Result |
| --- | --- |
| Hard veto screen | `L=1` is ineligible; Chart A confirmation failed acceptance. Native divergence was unavailable and therefore cannot support either a veto or a zero-divergence claim. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Valid-arm acceptance, finite log-accept tails, runtime, and RSS. |
| Default-readiness | Not ready. |
| Next evidence needed | TensorFlow-only fixed-transport tuner parity, then a CPU/XLA fixed-HMC canary and the repository sequential convergence gates. |

## Post-Audit Red Team

- Strongest alternative explanation: fixed HMC may be mechanically usable even
  though TFP does not label divergences; the resulting artifact must retain that
  limitation and cannot claim zero divergences.
- What would overturn the remaining blocker: TensorFlow-only API parity and a
  representative CPU/XLA canary showing the minimum valid campaign fits budget.
- Weakest evidence: no corrected target-specific fixed-transport tuning run has
  executed, and the prior chains were short.
