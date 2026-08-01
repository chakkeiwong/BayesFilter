# HNN-NeuTra Versus Exact NeuTra-HMC Comparison Repair Plan

Status: `COMPLETE_WITH_STR_EXACT_COMPARATOR_UNRESOLVED`.

Terminal result:
`docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-terminal-result-2026-07-18.md`.

Date: 2026-07-18.

Supervisor/executor: Codex. Independent review is advisory. The newest
`AGENTS.md` academic-research profile governs this campaign.

## Why This Repair Is Required

The completed neural-force campaign asked for same-chart true-gradient HMC in
its P4 and P5 subplans, but the executable nonlinear arm dictionaries contained
only `learned_residual` and `zero_residual`. The true transformed-filter force
was defined for predator-prey and then omitted from execution; it was absent
from the SIR/structural arm set. Consequently, PP-UKF, PP-SGQF, SIR-SGQF, and
STR-UKF have useful one-seed corrected-kernel validity evidence but no executed
comparison against exact-gradient NeuTra-HMC. The zero-residual ablation cannot
answer the intended speed or accuracy question.

LGSSM-KF already contains the required same-chart comparison and remains a
reference result. Its incomplete HNN-preparation timing is a reporting gap, not
a reason to rerun the exact LGSSM chain in this repair.

## Research Intent Ledger

| Field | Binding statement |
| --- | --- |
| Main question | On the same frozen target-specific NeuTra chart, does replacing exact transformed-filter gradients inside leapfrog by a frozen HNN force preserve posterior accuracy while reducing useful-sample cost? |
| Candidate | HNN-NeuTra-HMC: learned scalar residual-potential force for every leapfrog force call, with the exact transformed posterior value at the Metropolis endpoint. |
| Baseline | Exact NeuTra-HMC: exact gradient of the same transformed filtering posterior for every leapfrog force call, with the same exact transformed posterior value at the Metropolis endpoint. |
| Matched-mechanics comparison | Same chart, initial positions, seeds, step size, leapfrog count, chain count, transition count, dtype, GPU, XLA policy, and endpoint function; only the leapfrog force callable differs. |
| Tuned-algorithm comparison | Each arm is tuned independently over the same predeclared candidate grid and tuning sample budget, then run under the same adaptive warm-up/retained diagnostic contract. |
| Expected mechanism | For `L` leapfrog steps, HNN replaces `L+1` exact filter-gradient calls by `L+1` small-network force calls while both arms retain one new exact filter-value endpoint call per transition. |
| Expected failure mode | HNN force error can reduce acceptance, exploration, or ESS enough to erase the per-transition speed saving; exact-gradient compilation or target evaluation can also expose a harness/resource defect. |
| Promotion criterion | Both arms pass the validity gates; HNN has lower synchronized warm sampling seconds per minimum bulk ESS in the tuned comparison; and the matched-mechanics benchmark shows lower synchronized seconds per transition. |
| Promotion veto | Any nonfinite/status/energy failure, rank-normalized split or folded R-hat above `1.01`, bulk ESS below `1000`, tail ESS below `400`, failed truth/reference comparison, target/chart identity mismatch, endpoint parity failure, or unmatched benchmark mechanics. |
| Continuation veto | Shared target or kernel invalidity, corrupted/missing required artifact, GPU/XLA route unavailable after trusted probe and one focused repair, per-cell six-hour GPU ceiling exhausted, or total 24-hour GPU ceiling exhausted. Candidate rejection alone is not a continuation veto. |
| Repair trigger | Missing arm, asynchronous timing, serialization failure, XLA compile/resource failure, or localized target-adapter failure under an unchanged scientific contract. |
| Explanatory diagnostics | Acceptance, energy-error distribution, force RMSE by central/shell/tail region, step size, leapfrog count, raw seconds, zero-residual historical results, and per-stage timing. |
| Forbidden conclusion | No universal superiority, calibration, latent-model exactness, filter ranking, default readiness, or statistically supported ranking from one seed. |

## Scope

Fresh repair runs:

1. `PP-UKF` on its admitted frozen NeuTra chart.
2. `PP-SGQF` on its admitted frozen NeuTra chart.
3. `SIR-SGQF` on its admitted frozen NeuTra chart.
4. `STR-UKF` on its admitted frozen NeuTra chart, retaining the deterministic
   structural invariant and no-artificial-process-noise gate.

Historical reference only:

- `LGSSM-KF`, because it already ran the correct exact-gradient arm;
- all zero-residual arms, because they are chart-quality ablations rather than
  the scientific baseline;
- raw-coordinate HMC, because this repair isolates force replacement after the
  common NeuTra chart rather than the value of NeuTra itself.

Output root:
`docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/`.
Every launch uses a fresh attempt directory and preserves failures.

## Evidence Contract

### Accuracy And Sampler Validity

For each arm and cell, preserve:

- exact target signature, frozen transport hash, frozen HNN hash when relevant,
  target data/fixture identity, filter identity, seed domains, and environment;
- rank-normalized split and folded R-hat, bulk ESS, tail ESS, acceptance,
  endpoint status, finite-energy checks, and exact energy-identity replay;
- retained warm-up samples and retained sampling draws, capped at `10000` per
  chain for each stage;
- posterior means and 95% credible intervals in the physical parameterization;
- generating-truth coverage under the existing one-seed rule and direct
  HNN-versus-exact posterior mean/interval agreement;
- the structural invariant for every required STR-UKF sample.

The one-seed interpretation follows the owner policy: a pass is reported as a
one-seed pass; a marginal truth failure with two-sided tail probability below
`0.05` but at least `0.003` triggers one second seed for that cell; a tail below
`0.003` is a failure requiring investigation. Modern convergence and hard
validity vetoes are not relaxed by this rule.

Direct HNN-versus-exact agreement is reported per physical parameter using
the absolute mean difference, overlap of central 95% intervals, and

`z_MC = |mean_HNN - mean_exact| / sqrt(MCSE_HNN^2 + MCSE_exact^2)`.

An interval non-overlap or `z_MC >= 3` is an accuracy failure; `1.96 < z_MC <
3` is marginal and triggers the conditional second seed; `z_MC <= 1.96` passes
the direct comparison. MCSE uses each arm's bulk ESS and marginal standard
deviation. This screen diagnoses equality of the two sampled posteriors; it is
not a model-calibration claim.

### Speed And Cost

All GPU timings must force device synchronization before stopping the clock.
Each cell must report these non-interchangeable ledgers:

1. matched warm sampling seconds per transition at fixed mechanics;
2. tuned warm sampling seconds and seconds per minimum bulk ESS;
3. cold compilation plus sampling time;
4. per-arm tuning time, including compilation;
5. HNN-only supervision generation, recipe screening, and final fit time;
6. common sunk NeuTra chart-training time when reconstructible, otherwise
   explicitly `not_reconstructed` rather than zero;
7. reuse-scenario total and HNN break-even retained-sample estimate;
8. from-scratch total only where every component is measured or durably
   reconstructed.

A timer around optimizer execution alone must be labeled `optimization_only`.
It cannot be called HNN training/preparation time.

### Matched Benchmark Mechanics

Use the HNN arm's selected `(step_size, L)` for the primary matched benchmark.
Run both forces from identical four-chain positions for the same `500`
transitions per chain and identical stateless seed. Compile each callable once,
discard the compile probe, then time three synchronized warm replays in
alternating arm order and use the median. Record one
endpoint batch invocation and `L+1` force batch invocations per transition.
The matched benchmark measures mechanism cost only and cannot establish
convergence or posterior accuracy.

### Tuned Comparison

Tune both arms independently over the target-specific grid already declared by
the original campaign:

- PP-UKF and PP-SGQF: step sizes `(0.2, 0.4, 0.6, 0.8)` and `L in (6, 10)`;
- SIR-SGQF: step sizes `(0.2, 0.4, 0.6, 0.8)` and `L in (6, 10)`;
- STR-UKF: step sizes `(0.025, 0.05, 0.1, 0.2)` and `L in (8, 12)`.

Each candidate receives `500` transitions per chain. Tuning loss, acceptance,
or runtime alone cannot nominate an unhealthy candidate. The existing
health-aware R-hat/energy selection rule remains binding.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Existing frozen NeuTra chart | admitted multimodel NeuTra campaign | holds coordinates constant so only force replacement changes | stale/wrong target chart | target signature, transport hash, endpoint parity | reviewed baseline |
| Existing target-local HNN recipe grids | original P4/P5 subplans | avoids a new architecture question during comparator repair | small model underfits residual tails | heldout central/shell/tail force diagnostics and downstream acceptance/ESS | warm-start hypothesis |
| Fresh HNN fit | original campaign | measures preparation cost and avoids silently reusing a lucky fit | stochastic fit drift | frozen seeds, heldout diagnostics, archived model | required |
| Four chains | original campaign and modern diagnostics | required for split/folded R-hat | insufficient tail precision | ESS and tail diagnostics | reviewed default |
| One seed first | owner cost policy | proportionate exploratory evidence | fixture/seed luck | conditional second-seed rule | binding diagnostic policy |
| Current tuning grids | target-specific original subplans | repairs comparator omission without changing kernel search | exact arm optimum outside grid | boundary selection and acceptance/energy table | baseline, not universal optimum |
| Float64, GPU, XLA, memory growth | repository policy and existing target implementations | matches admitted routes | compile/memory mismatch | trusted device probe and two-transition smoke | required |
| Adaptive sample cap `10000` | owner policy | convergence-driven rather than fixed sample count | expensive arm exhausts cap | every 1000-sample diagnostic | binding cap |
| Minimum bulk ESS as efficiency denominator | original performance contract | penalizes the weakest parameter | unstable one-seed denominator | report raw time and every parameter ESS too | descriptive promotion metric |

## Premortem

The campaign could appear to pass while misleading us if asynchronous GPU work
escapes the timer, the endpoint accidentally computes an unused gradient, the
two arms use different charts or transition counts, HNN preparation is omitted,
or ESS noise creates an apparent ranking. Explicit synchronization, endpoint
parity/source inspection, identity replay, matched counters, separate cost
ledgers, and one-seed language address those risks.

The campaign could fail for engineering rather than scientific reasons if the
exact-gradient graph recompiles per chunk, the adapter cannot XLA-compile on the
trusted GPU, or output serialization exhausts resources. A two-transition
canary and fixed-signature compile audit precede every serious cell. Such a
failure triggers localized repair; it does not reject HNN.

## Compute And Attempt Budget

| Cell | GPU ceiling | Serious attempts | Conditional second seed |
| --- | ---: | ---: | ---: |
| PP-UKF | 6 h | 1 + 1 localized repair | charged inside ceiling |
| PP-SGQF | 6 h | 1 + 1 localized repair | charged inside ceiling |
| SIR-SGQF | 6 h | 1 + 1 localized repair | charged inside ceiling |
| STR-UKF | 6 h | 1 + 1 localized repair | charged inside ceiling |

Total ceiling: 24 trusted GPU wall-hours and 4 CPU wall-hours. Unused budget is
not a target. A materially different model, chart, filter, hardware class,
threshold, or total ceiling requires user direction.

## Phases

### Phase 0: Plan Audit And Freeze

Objective: verify the research question, exact baseline, grids, identities,
cost boundaries, stop conditions, and executable commands.

Required artifact: this reviewed plan plus a review record. No serious GPU run.

Close condition: skeptical local audit passes and advisory review has no
unresolved material scientific, mathematical, feasibility, or cost finding.

### Phase 1: Harness And Timing Repair

Objective: make the planned comparison executable and mechanically auditable.

Required changes/checks:

- add `true_gradient` to both nonlinear executable arm sets;
- remove zero residual from the repair campaign's primary comparison while
  leaving historical code/artifacts readable;
- synchronize supervision, training, tuning, matched, and sampling timers;
- emit compile, warm execution, invocation-count, preparation, reuse, and
  break-even fields;
- add a matched-mechanics function and direct HNN-versus-exact posterior
  summary;
- add tests that fail if either required arm is absent or mechanics differ.

Close condition: focused CPU-hidden tests and static source audit pass. Write a
Phase 1 result before GPU execution.

### Phase 2: Trusted GPU/XLA Canaries

Objective: prove each exact and HNN force can execute the corrected kernel on
the intended trusted GPU/XLA route without hidden endpoint differentiation.

For each cell run two transitions per chain with `L=2`, exact endpoint parity,
finite energy identity, GPU placement, XLA, TF32 setting, and memory-growth
metadata. A cell-local canary failure triggers one focused repair and retry in a
fresh directory.

Close condition: both arms pass for a cell before that cell enters Phase 3.

### Phase 3: Predator-Prey Comparisons

Run PP-UKF and PP-SGQF independently. For each cell: generate and time fresh
supervision, train a fresh HNN, tune both arms, run the matched benchmark, run
the adaptive diagnostic chains, and write the accuracy/cost decision.

One cell failure does not block the other or Phase 4 unless it reveals shared
kernel/target invalidity.

### Phase 4: SIR And Structural Comparisons

Run SIR-SGQF and STR-UKF independently under the same contract. STR-UKF must
also pass deterministic structural identity and no-noise checks.

### Phase 5: Synthesis And Drift Audit

Aggregate the four fresh repair cells and historical LGSSM reference without
pooling posterior identities. State hard vetoes, viable candidates, whether any
ranking is statistically supported, descriptive differences, default-readiness,
and evidence needed next. Downgrade or supersede any old performance wording
that conflicts with the repaired evidence.

## Commands

Phase 1 checks will use explicit CPU hiding before TensorFlow import:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_neural_force_hmc.py \
  tests/test_neural_force_training.py \
  tests/test_neural_force_campaign.py \
  tests/test_hnn_neutra_comparison.py
python -m py_compile \
  bayesfilter/inference/neural_force_campaign.py \
  bayesfilter/inference/neural_force_training.py \
  bayesfilter/testing/predator_prey_neural_force_hmc_tf.py \
  bayesfilter/testing/sir_structural_neural_force_hmc_tf.py \
  docs/benchmarks/run_hnn_neutra_exact_comparison_2026_07_18.py
```

Serious launches will use the `tf-gpu` environment, memory growth, trusted GPU
access, XLA JIT, fresh per-cell output roots, and the new repair runner. Exact
commands are frozen in the Phase 1 result after the CLI exists and passes
`--help` plus canary checks.

## Phase-End Repair And Continue Procedure

At the end of every phase:

1. run the required local or serious checks;
2. write a phase result/close record with commands, manifest, wall time,
   evidence paths, decision table, inference-status table, and remaining budget;
3. draft or refresh the next phase section or subplan from actual evidence;
4. review the next phase for wrong baseline, proxy promotion, missing stop
   conditions, unfair mechanics, stale assumptions, environment mismatch,
   artifact coverage, and filtering-boundary safety;
5. continue automatically when no true continuation veto fired.

For a localized failure, preserve the failed attempt, classify it, record the
smallest repair and budget charge, run a focused regression, and retry once in a
fresh directory. Stop only the affected cell after its repeated-failure or
budget ceiling. A scientifically valid but slow or inaccurate HNN is a
candidate result, not a reason to invalidate or abandon the other cells.

## Skeptical Pre-Execution Audit

Audit status: `PASS_WITH_REPAIRS_APPLIED`.

The review must explicitly challenge the exact baseline, matched mechanics,
independent tuning fairness, truth/reference criteria, asynchronous timing,
HNN preparation accounting, inherited defaults, sample/compute caps, artifact
sufficiency, and whether every command can answer the research question.

The local skeptical audit repaired two material omissions before execution:
it defined a numerical HNN-versus-exact posterior-agreement rule instead of
leaving "agreement" subjective, and required repeated alternating warm timing
instead of an order-sensitive single replay. The exact same-chart baseline,
matched mechanics, independent tuning, synchronized timing, cost ledgers,
truth/reference gates, inherited-default classifications, budgets, and
repair/continue boundary then passed review.

Claude advisory review was attempted in the required bounded form. A health
probe returned `CLAUDE_PROBE_OK` and the one-file read probe returned
`PLAN_READ_OK`, but the substantive bounded review and a fixed-token
baseline/timing verdict both exited successfully with no output. This is a
documented reviewer-response limitation, not a scientific blocker under the
repository review-proportionality policy. Codex therefore completed the
independent skeptical audit and remains supervisor/executor.
