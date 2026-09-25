# q20 production program repair

Date: 2026-09-15. Owner requested a thorough repair plan, review and execution.
Status: `PARTIALLY_IMPLEMENTED_MAIN_INTEGRATION_AND_LAUNCH_GAPS`.

September 16 completion audit: the training repair is merged into local `main`
at `965ba2949244ee2a1d911515b6865c268a4af8d2`. The private-checkout instructions
below describe the original implementation, not a requirement to restart it.
The [master's current section](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md)
now governs continuation. Steps 6–8 have not met their connected-execution and
launch acceptance criteria; step 2 exposes only `validate`, `price`, `train`.
The [corrected result](bayesfilter-ssl-lstm-q20-production-repair-result-2026-09-16.md)
records verified components, concrete energy-contract defects, incomplete
comparison/reference computation and missing whole-campaign supervision.
The earlier statement that the software was ready for a development retry is
withdrawn. Complete these requirements under the existing authorization before
launching the serious training cohort.

The q20 recovery runner trains only six updates, loses optimizer continuation,
calls an obsolete tuning callback and omits the required posterior precision.
The numerical parameter ledger is documentation; it has not repaired that call
chain. This plan implements the actual route from an explicit configuration
through training, frozen map export, current public tuning, sequential posterior
assessment and terminal reporting.

## Scope and isolated source

Use `/tmp/BayesFilter-q20-production-repair-20260915`, a private local checkout
of main commit `5139f151` plus an inventoried snapshot of the completed shared
HMC repair. Preserve the main checkout and existing audit checkout. Copy the
q20 design, parameter ledger and mathematical audit into the repair checkout.
Do not copy unrelated SRUKF, manuscript or other dirty work. Include the q20
strict eigensolver repair needed by this target, recording its separate source
provenance. New edits and tests occur only in the private checkout.

The target remains the four free coordinates of the q20, T=30 synthetic UKF
posterior, float64, TensorFlow/TFP and strict principal-square-root backend.
Serious training uses a batch-native target, GPU memory growth and XLA. Public
model/data/prior identities must remain explicit. No environment/package
mutation or new scientific model is part of this repair.

The [parameter ledger](bayesfilter-ssl-lstm-q20-production-parameter-ledger-2026-09-15.md)
and [mathematical audit](bayesfilter-ssl-lstm-q20-parameter-mathematical-audit-2026-09-15.md)
define candidate values, provenance and first diagnostic checks. Uncalibrated
values may be used as labeled development hypotheses. They cannot support a
claim of adequate learning or posterior accuracy without their measurements.

## Evidence contract and research intent

| Item | Requirement |
| --- | --- |
| Engineering question | Does the real q20 entrypoint consume the explicit production protocol and preserve correct state/evidence all the way to public tuning and posterior assessment? |
| Research question preserved | Can properly trained NeuTra and the declared ensemble produce accurate q20 posterior estimates at useful total cost? This repair enables that experiment; it does not answer it. |
| Baselines | Existing six-update route is the regression failure to eliminate. Numerical tests use independent known Gaussian/transform identities; experiments retain ordinary identity, tuned classical, plain NeuTra and enhanced ensemble comparators. |
| Primary repair criterion | Executable configuration, real training/checkpoints, frozen-map export parity, current public tuner bridge, named posterior precision and honest terminal statuses pass connected tests. |
| Numerical veto | Nonfinite/invalid target, wrong value/score/logdet, state or source mismatch, scalar-training fallback, invalid GPU allocation, unsupported binding or corrupted evidence. Stop affected computation and repair. |
| Promotion veto | Smoke map, unresolved training assessment, missing reference/coverage, failed posterior precision or invalid stochastic comparison. No production-qualified status may be issued. |
| Repair triggers | Localized harness/resource/API errors and implicated uncalibrated settings. Follow the audit's symptom table; preserve each attempt. |
| Explanatory evidence | Loss, clipping, covariance, acceptance, short-run ESS, timing and smokes. They cannot certify posterior correctness or method superiority. |
| Continuation veto | Unrepaired shared numerical invalidity, missing required source/data, permission failure at its actual boundary or exhausted authorized compute. |
| Not concluded | Optimal hyperparameters, trained q20 transport, sufficient burn-in, validated finance model, qualified reference or supported method ranking from engineering tests. |
| Preserved result | This plan, a concise repair result, source snapshot inventory, exact commands/logs and versioned outputs under `docs/plans/artifacts/ssl-lstm-q20-production-repair-2026-09-15/`. |

## Implementation and completion checks

1. **Integrate a coherent shared HMC implementation.** Inventory copied source
   hashes and include its affected tests. Check the actual public API and registry.
   Avoid a second private tuner or a callback that bypasses required evidence.
   Completion: imports and relevant shared regressions pass in the private tree.

2. **Executable protocol and stage orchestration.** Add explicit, serializable
   target, training, validation, tuning, posterior and budget settings derived
   from the ledger. Initial training candidates are widths16/32, two tanh IAF
   stages, LR0.0005/0.001, batch32 with batch8/128 pricing alternatives,
   Adam0.9/0.999/1e-7, clipping10 and cap2. Preserve their hypothesis status.
   Positive-beta cumulative rungs are128/512/2048/8192 with analytic beta-zero
   initialization and no beta-zero optimization. Expose development, smoke and
   confirmation roles; every shortened smoke carries a nonpromotion marker.
   Counts/limits are validated before frameworks or target calls. No hidden
   preflight profile supplies serious settings. Completion: CLI config and
   dispatch tests demonstrate all relevant settings reach their actual consumers.

3. **Training and exact recovery.** Reuse the batched reverse-KL numerical step.
   Persist every update's loss, raw/clipped gradient norms, clipping flag,
   iteration, target status and counts. Checkpoints include map, all optimizer
   variables, step, RNG position, beta and complete scope. Resume is equivalent
   to uninterrupted training on the same device/numerical scope. Carry optimizer
   state across beta by the declared protocol; reset only as a named contrast.
   Save at128 updates and stage boundaries with ordinary versioned artifacts.
   Stop and record invalidity before accepting an unusable update. Completion:
   interrupted/restored training equals uninterrupted training including the
   next update; corruption/config/source changes and invalid training fail.

4. **Measured training assessment and map qualification.** Implement heldout
   paired loss statistics on declared banks, cumulative rung decisions and
   capacity/optimizer repair nominations. Delta0.04 and bank768/3072/12288 are
   development hypotheses, not confidence guarantees after adaptive reuse.
   Preserve per-root outcomes and explicit unresolved/cap/budget statuses;
   never nominate by a single unqualified short loss. Save inverse/logdet/score
   reliability evidence separately from independently assessed posterior
   whitening/coverage. Completion: healthy progressing, deteriorating, noisy,
   invalid and capped fixtures reach the right disposition and cannot masquerade
   as a completed cohort or qualified posterior.

5. **Export the actual learned transport.** Support the q20 prior-affine wrapper
   and weighted dense IAF in the existing repository frozen-map format. Include
   permutations, masks, scale transforms, weights and affine order exactly.
   Completion: nonzero trained weights round-trip with forward/inverse/logdet
   and analytic score parity; unsupported map variants reject rather than drop
   components. Trained state and target identity are bound to the exported map.

6. **Current public tuning and posterior path.** Replace the obsolete custom
   runner/single-result assumptions with current candidate-set tuning and
   verified-member consumption. Preserve all verified candidates and explicit
   selection for development; no short-chain ESS winner. Use fresh frozen-scope
   starts and the current supported binding. Posterior policy has four physical
   means, twelve quantiles and the sign event; mean MCSE/SD0.02, event MCSE0.01,
   quantile MCSE/SD0.05, retained bulk/tail ESS400, modern R-hat1.01. Warmup is
   min2000/window1000/max10000 at1.05; retained grows1000 at a time to10000.
   Completed tuning is not completed posterior sampling. Completion: actual
   trained-map public tuning-to-posterior Gaussian smoke plus connected q20
   argument/wiring checks and negative tests for missing precision, smoke
   promotion and state/binding changes.

7. **Comparison, ensemble and reference boundary.** Preserve the complete
   comparison inventory and variant-specific scopes. Ensemble execution must
   use the existing valid chart/temperature transition, train/tune each used
   scope, retain only independent cold streams and apply the same posterior
   quantity checks. A single beta-one chart cannot be labeled an ensemble.
   Reference/start-equivalence uses named quantities, declared uncertainty,
   independent data lineage and a combined reference error allowance. A missing
   or unqualified reference gives incomplete scientific assessment; it cannot
   be filled by a dummy pass or an in-sample training score. Completion: each
   supported mode exercises its complete computation in a bounded fixture;
   unsupported/incomplete modes report that before consuming serious compute.

8. **Budget, launch and restart.** A fresh serious run uses unique outputs and a
   resolved cost reservation covering requested training cohort, tuning,
   development, confirmation, reference and one specified repair. Record actual
   timing, failed work and cumulative worker charges. Price compilation and
   real q20 calls; forecast incomplete work honestly. Preserve a diagnostics-only
   route for historical mechanics, but route new production/development use
   through the repaired program. Completion: dry-run/resume/cap tests, small
   trusted GPU/XLA target/transport smoke, command example and a clear terminal
   distinction between software readiness and a validated scientific campaign.

## Validation commands and compute allocation

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Tests explicitly set
`CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true` and
`BAYESFILTER_TEST_DEVICE_SCOPE=cpu` before TensorFlow import. These are tiny
engineering/reference exceptions, never serious NeuTra training evidence.
Run focused new suites and affected existing transport/HMC/route-policy suites;
record exact test paths after implementation. New numerical kernels require
stable signatures; no pfor or silent eager fallback.

GPU validation uses trusted/elevated execution, explicit visible GPU, verified
memory growth and XLA. Begin with the installed bounded GPU readiness probe,
then a known-target training/export/tuning smoke and a small real q20 path
check. Count those as diagnostic work, never as training qualification.
Reserve at most two aggregate worker hours for engineering numerical tests
and GPU diagnostics initially, within the saved15.0832 diagnostic hours and
37.5766 campaign hours. This is a conservative repair allocation, not an
estimate that serious training fits. GPU q20 attempts each have an external
10-minute ceiling initially; at most four such attempts without revising the
allocation. Repeated compilation failure triggers localization, not repeated
full runs. Record actual attempts, wall time and remaining allocation.

No expensive production training grid, final posterior comparison or financial
model change is launched merely to claim that software was repaired. The
software must provide an executable measured protocol for the subsequent
authorized development campaign.

## Skeptical plan review

Review performed before implementation against the connected audit, current
HMC interface and parameter audit. Initial defects in a naive repair plan were:

- Raising a constant to8192 would still lack optimization evidence and exact
  recovery. Phases2–4 replace that with a real configured, assessed training path.
- Using main's new tuner with the old callback/result would fail even after
  training. Phases1,5,6 explicitly migrate the actual binding and map codec.
- Passing a Gaussian smoke could be misreported as q20 qualification. Every
  result keeps scope/role and missing scientific evidence explicit.
- One beta-one map is not the planned ensemble and weak classical tuning is
  not a fair comparator. Phase7 requires complete method wiring and fails early
  for any unavailable path rather than silently reducing the method.
- Reusing validation/confirmation, unpriced training cohorts and missing
  reference error could make a successful command scientifically meaningless.
  Phases4,7,8 preserve partitions, uncertainty, reserves and incomplete status.
- Blindly copying main would overwrite unrelated work or mix changing source.
  Phase1 uses an isolated inventoried snapshot and focused integration checks.

Verdict: `PROCEED_WITH_ENGINEERING_REPAIR`. No unresolved inherited parameter is
treated as qualified: each remains an explicit hypothesis with diagnostics and
the parameter audit's failure response. Software acceptance requires connected
tests and a terminal code review; scientific readiness additionally requires
the actual measured development protocol and independent posterior evidence.

## Execution log

- 2026-09-15: isolated checkout created from main `5139f151` with the recorded
  shared HMC repair snapshot and strict q20 eigensolver prerequisite. The
  private tree includes the compiled Sylvester test library copied from the
  existing local build; the main checkout was not modified.
-  2026-09-15: prerequisite HMC/NeuTra contract tests passed: `68 passed`.
-  2026-09-15: implemented `q20_production_config.py`,
  `neutra_training_protocol.py`, `q20_production_training.py`,
  `q20_production_hmc.py`, and the explicit entrypoint
  `docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py`. The route has
  explicit target/training/validation/tuning/posterior/budget settings,
  batch-native reverse-KL updates, complete Adam/map checkpoints, heldout
  paired assessment, frozen-map export parity, current candidate-set tuning
  adapters, and exact ensemble transition wiring.
- 2026-09-15: focused repair tests passed: `11 passed`; existing NeuTra and
  tempered transport regressions passed: `35 passed`; route-policy,
  tempered-ensemble, and replica-exchange regressions passed: `28 passed`.
- 2026-09-16: the q20 CPU smoke reached TensorFlow initialization but did not
  produce an artifact within the bounded smoke allocation and was interrupted.
  It is retained as an unqualified compile/resource diagnostic. No GPU,
  production training, HMC posterior, or scientific promotion run was made.
- 2026-09-16: the trusted read-only GPU probe returned
  `no_idle_policy_permitted_gpu`. No GPU training was attempted after that
  boundary; this is an environment availability result, not evidence against
  the q20 target or the repaired numerical route.

- 2026-09-16 completion audit: the final focused repair suite has **12 passed**,
  including the Gaussian cohort/restart test; the earlier 11-test count preceded
  that addition. Main's post-merge rerun passed in 14.72 seconds. The suite does
  not exercise the new public tuning-to-posterior or ensemble consumers.
- 2026-09-16 master recovery: preserve the interrupted CPU smoke as incomplete.
  Its cooperative limit did not bound initialization/compilation; it was
  manually interrupted, not stopped by a demonstrated hard deadline. The new
  CLI's `price` mode does not consume `--max-seconds`. Complete external bounds
  and aggregate accounting before numerical retries.

Current software status: `PARTIALLY_IMPLEMENTED_MAIN_INTEGRATION_AND_LAUNCH_GAPS`.
Scientific status: `NOT_QUALIFIED`. The master's ordered next work completes
the existing plan; it neither changes the scientific target nor renews budget.
