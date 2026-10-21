# Literature priors for the five-model surrogate-HMC campaign

Owner direction, 2026-10-15: use original Zhao–Cui parameter priors where
applicable; otherwise adopt settings from representative papers and retain
the papers under `docs/papers`. This replaces the earlier pending request for
owner-supplied priors. It is part of the existing master program.

The question is whether every posterior in the campaign has a proper,
explicit, literature-supported parameter prior in the coordinates actually
sampled. The likelihood models, saved data, FP32 preference, N=4032, and
existing campaign compute cap remain unchanged. This work does not establish
posterior convergence or superiority of any prior. No GPU budget is needed
for source inspection and focused CPU correctness checks.

## Evidence contract and audit

Compare the paper's density, support, units, and parameterization with the
actual model constructor and the HMC force and endpoint call chains. A prior
is adopted only with an inspected technical source and a derivation of every
coordinate transformation. Initial-state priors are not parameter priors.
Settings copied to different units or parameters are adaptations, not exact
reproductions. Keep the exact setting and the local adaptation side by side.

Pass criteria: local full-text evidence; a normalized parameter density with
declared support; correct density Jacobian and analytic prior score; both force
and endpoint consuming the same prior; scope identity binding prior settings;
focused checks against independent density/derivative calculations. Invalid
support, missing Jacobians, ungrounded settings, or stale tuning are vetoes.
Prior predictive checks are diagnostic and can expose unsuitable units or
numerical failures; they cannot select priors for favorable likelihood or HMC
results. The benchmark normal prior remains a mechanics-only comparator.

Skeptical audit already identified three hazards: Zhao–Cui's SIR parameters
are fixed; its LGSSM parameterization differs from ours; its synthetic SV box
does not contain our nominal parameters. Copying these settings without
mapping would answer the wrong question. Its predator–prey box matches our
six physical parameters. Range bearing requires another source. These are
source/mapping issues to resolve before sampling, not evidence against HMC.

Inspect primary technical sections and available author code; record source
versions, checksums, anchors, adaptations and coverage limitations. Treat
new hyperparameters as source choices or explicit modeling assumptions, not
as consequences of a successful smoke. Review transformed scores and tails
with CPU-only tests before enabling the scientific prior option.

## Execution boundary

1. Complete source and parameter mapping; save a compact evidence ledger.
2. Implement the literature-prior option and exact scope binding;
   preserve the historical mechanics option.
3. Check densities, scores, support and consumer wiring; update the master
   and checkpoint with the result and exact remaining action.
4. Fresh scope-specific tuning and adaptive HMC remain later phases of the
   same master. Old benchmark-prior tuning must not be relabeled as scientific
   prior tuning. Execute only within the existing remaining 47,849.122 GPU
   seconds and 11 attempts, after confirming the ledger has not changed.

Stop dependent work if a parameter's source mapping remains materially
ambiguous. Continue independent source/code checks. Do not change the
likelihood, observation model or physical time units silently to fit a paper.

Artifacts: `docs/papers/surrogate-hmc-priors-2026-10-15/` and this plan;
prior specification, tests and result note will be linked here when complete.

## Adopted choices and pre-execution audit

The prior specification is `literature-model-priors-v1` in
`bayesfilter/testing/surrogate_hmc_literature_priors.py`. Its `specification`
function records each density, support, transformation and source adaptation.
The source/mathematical discussion is in the evidence directory's `README.md`.

- Predator–prey: the published Zhao–Cui six-dimensional uniform box, unchanged.
- KSC: Zhao–Cui's empirical-SV prior conditioned on innovation variance one and
  positive persistence, matching the existing parameter support.
- LGSSM: the published Dahlin–Schön AR prior restricted to the existing support;
  independent IG(1,1) variance multipliers from the Särkkä–Nummenmaa family;
  a standard Normal initial-mean coefficient. The process-scale and location
  transfers are explicit modeling adaptations.
- SIR: Grinsztajn et al.'s positive Normal infection/recovery priors. A declared
  synthetic unit calibration of 45 days per benchmark time unit and 500 people
  per region preserves the current dynamics while interpreting those rates.
  Gaussian observation variance gets IG(1,1), not a negative-binomial prior.
- Range bearing: independent IG(1,1) common process/observation variance
  multipliers. The covariance shapes and likelihood are preserved.

Skeptical implementation audit: benchmark-normal-v1 is the explicit historical
comparator, never a scientific prior. The source code conflicts with the
published predator box and LG AR scale; the published statements control these
choices and the discrepancies are retained. The SIR time calibration is a
local assumption, not a paper claim. IG(1,1) is proper but has infinite mean;
none of these choices is claimed to be uniquely scientifically appropriate.
The inherited affine coordinate scales are preconditioning hypotheses only.
PP's logistic chart changes no physical prior. The unchanged KSC initial law
and observation mixture prevent calling this a full source-model replication.

This audit permits focused CPU-only correctness tests. It does not permit
relabeling old tuning or asserting numerical suitability throughout the prior.
Pass: independent normalized-density agreement, analytic/AD/finite-difference
score agreement, CPU-XLA batch/scalar consistency and actual HMC consumer
wiring. Failures trigger local repairs before any sampling. Parameter-support
and prior-predictive numerical checks remain prerequisites to new expensive
sampling, particularly for the fixed-step SIR ODE and heavy variance tails.
These are correctness diagnostics, not a stochastic ranking or prior selection
contest; no heuristic-superiority claim is being made.

CPU test command (no GPU compute charged):
`CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/test_surrogate_hmc_literature_priors.py tests/test_surrogate_hmc_five_model_master.py`
Preserve output and manifest in
`docs/plans/artifacts/surrogate-hmc-literature-priors-20261015-01/`.

The final call-chain review also found the performance/preparation launcher
loading the historical default Case. It now selects the same literature prior
and forwards it to every worker, with an explicit benchmark option for
historical diagnostics. This is necessary to issue matching new selections
through the already accepted OT/precision preparation route. The bounded
regression therefore additionally runs
`tests/test_surrogate_hmc_performance_repair.py`. It checks default selection
and subprocess forwarding without launching GPU work. No filter kernel or
likelihood model was changed by this repair.

Result and run provenance:
`artifacts/surrogate-hmc-literature-priors-20261015-01/result.md` and
`artifacts/surrogate-hmc-literature-priors-20261015-01/manifest.json`.

## Bounded LGSSM continuation after adopting the priors

This is the next phase of the same master, authorized by the instruction to
continue and the resolution of the pending prior choice. First test 64 fixed-seed
draws from the adopted LGSSM prior, the nominal point and the four HMC initial
positions against the exact Kalman acceptance endpoint. Use a CPU-only independent
reference exception. The numerical question is whether the chart, normalized
prior and exact likelihood are finite at these inputs. A non-finite result is a
repair trigger; likelihood magnitudes are explanatory, never prior-selection
criteria. Passing establishes only validity at the inspected points, not
posterior convergence, scientific preference for the prior or global coverage.

Skeptical audit: the AR prior stays within the model's stable support; positive
variance multipliers preserve positive covariance matrices. IG(1,1)'s heavy tail
still needs explicit finite-point checks. The four inherited initialization
positions and identity metric are coordinate hypotheses, not draws or moments
from the new prior; their provenance strings have been corrected. No numerical
sampler settings are changed. The comparator is the exact Kalman likelihood,
not the old benchmark-prior posterior. No ranking of methods is attempted.

If these checks pass, run only the LGSSM fresh tuning/mechanics advance first,
under the existing cumulative ledger, at most 3600 seconds and two attempts.
Keep FP32, N4032/K2016, epsilon=1, five OT steps and the direct Gaussian
contraction. Offset calibration/validation seeds by 200 (previous scope used
100), and use previous selected controls only as the documented warm start.
Output: `artifacts/surrogate-hmc-cost-repair-24h-20261013-01/`
`literature-prior-lgssm-advance-01`. A failed candidate is a repair trigger;
budget exhaustion, invalid inputs or unavailable required diagnostics stop the
dependent work. Refresh the checkpoint and next phase after completion.

## Continuation handoff audit and remaining preparation

LGSSM passed the 69-input CPU support check and fresh GPU tuning/mechanics.
The latter includes an independent endpoint replay check. Its warmed two-step
four-chain chunk took 92.035 seconds, including an initial force evaluation.
That is not a measured long-chain rate. Earlier same-kernel consumer costs
suggest that all five full posterior runs cannot fit the remaining 13.14 hours;
the prior change also prevents treating earlier timings as a convergence forecast.
Keep the original convergence rules and the total 86,400-second campaign cap.

Next, prepare KSC, range bearing, predator–prey and SIR with seed offset 200,
the same accepted OT family, their frozen arithmetic, and literature priors.
Reserve at most 14,400 seconds for these eight workers (each tuning and
mechanics worker at most 2,400 and 1,200 seconds, respectively). Check endpoint
replay and actual initial-force/short-chain health inside each mechanics phase.
These checks cover their inspected inputs only; wide prior predictive support
remains unestablished. They cannot certify posterior correctness or select priors.
Use GPU UUID GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a to select the RTX 5080:
CUDA ordinal 0 selected the RTX 4080 SUPER in the LGSSM run despite nvidia-smi's
different ordinal ordering. Preserve both device identities in run evidence.

The master handoff needs a bounded engineering repair: accept actual completed
advance artifacts, validate the prior/data/source/precision scope, retain SIR's
FP64-recursion/FP32-pair policy, reuse completed mechanics, and charge inference
workers to the same cumulative ledger. Do not manufacture an old preparation
manifest or reissue an old selection. Tests must reject stale source/prior,
failed mechanics, mismatched mixed precision and exhausted budgets, and prove
that only the remaining inference phase is launched with a worker deadline.

Skeptical audit: equal slicing of the remaining budget across five models may
leave every model before retained sampling. Complete preparation first and use
the already declared cheaper-target-first inference order. KSC is the first
inference candidate, subject to its fresh health results and measured cost.
Record an explicit bounded KSC allocation after preparation; no time forecast
is a convergence guarantee. An unmet compute requirement is under-budgeting,
not evidence against surrogate HMC. The current 140-attempt ceiling is retained:
eight preparation workers plus one inference worker use the nine remaining
slots. Additional workers would require an explicit plan/attempt-cap revision,
without increasing the owner's compute cap.

Engineering pass criteria are the executable handoff/budget tests and the
actual caller reaching the shared adaptive controller. Numerical health is a
veto for a model's dependent sampler only; other valid scopes continue. R-hat
is assessed cumulatively on retained samples, with the existing 20,000 maximum;
there is no extra validation chain. This phase does not rank models or filters,
promote a default, or claim posterior convergence from short mechanics runs.

## Adaptive-HMC continuation after the handoff repair

The engineering audit found and repaired two concrete handoff faults: the
earlier adapter read a different preparation layout and rejected the accepted
SIR arithmetic; an unexpected worker-launch error could leave a running status
in the next-phase file. The new `continue-advance` route consumes the real
advance artifacts, verifies the completed checks, and launches only inference.
CPU tests cover actual layout, stale identities, invalid health, mixed precision,
budget/attempt vetoes, timeout and launch-error accounting, and continuation of
other models after one model's health veto. The first integrated suite passed
186 tests; the final changed-file regression is recorded in
`artifacts/surrogate-hmc-literature-priors-20261015-01/continuation-tests-03.log`.

The next phase uses both fresh advance roots, in order KSC, range bearing,
LGSSM, predator–prey and SIR. Allocate at most 45,000 seconds to this phase,
32,400 seconds to any one model, always clipped to the live remainder of the
original 86,400-second ledger. KSC is first because measured kernel cost makes
it the most plausible early completed result; this is scheduling, not a
statistical ranking. Short pilot timings include initialization and do not
forecast adaptation or convergence. The remaining budget may not finish all
five models; no thresholds or sample minima are reduced to disguise that gap.

After preparation has finished and no ledger writer is active, revise the
plan's operational attempt ceiling from 140 to 145: the planned five inference
workers plus one localized repair slot fit after the 139 preparation/earlier
attempts. This explicitly supersedes the nine-slot projection above. It changes
no scientific target, compute allowance, hardware class or permission boundary.
There is no reason to stop after KSC merely because an earlier engineering
attempt count did not include these now-ready inference workers.

The public tuning configuration is unchanged: target-specific mass windows of
128 and 256 transitions, 256 step-size adaptation transitions per tested
trajectory length, up to four leapfrog steps, and acceptance/health screening.
Its initial identity metric with 10 percent shrinkage is a warm-start
hypothesis, not the prior covariance. Nomination is one compiled numerical
call, so its artifacts become durable only when that call returns; the external
worker timeout bounds it. The subsequent sequential phase checks its soft
deadline before chunks and archives every completed 250-transition chunk.
Warmup remains at least 2,000 and at most 10,000 per chain, with readiness on
the latest 1,000 at modern R-hat <=1.05. Retained sampling grows cumulatively,
starting assessment at 1,000, until modern R-hat <=1.01 or 20,000 per chain,
or an explicit health/compute stop. No post-convergence validation chain is run.

Skeptical audit: the claim remains convergence of surrogate-force HMC to its
declared independent-endpoint posterior, conditional on checked kernel
mechanics. Acceptance is explanatory and nomination evidence, not convergence.
R-hat, ESS, energy, movement, finite target/force and fallback counts must be
preserved. A health failure rejects that model's current candidate and triggers
repair; it does not veto independent models. Exact Bayesian-model agreement is
not established by convergence to a finite grid/BPF/importance endpoint.
LGSSM uses exact Kalman endpoints; the four nonlinear models retain their
declared approximate references. Posterior reference refinement and heuristic
comparisons remain necessary for broad scientific/default claims. This phase
does not assert superiority over classical filters or alternative samplers.

Exact planned command, after the real five-scope handoff check passes:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_five_model_master.py continue-advance --advance-root docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-lgssm-advance-01 docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-other-four-advance-01 --campaign-ledger docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/budget.json --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/literature-prior-hmc-01 --budget-seconds 45000 --per-model-timeout 32400 --gpu GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a --priors literature-model-priors-v1
```
