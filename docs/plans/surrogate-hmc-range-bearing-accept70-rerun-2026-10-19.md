# Range-bearing surrogate HMC with a 70% acceptance target

The owner requested a fresh run on 19 October after correcting the range-bearing
95% acceptance override. This is a new, authorized range-bearing attempt in the
existing five-model research programme. It starts HMC tuning again and does not
resume the completed 95%-target kernel. The active SIR job continues separately.

## Question and evidence contract

Can the existing range-bearing surrogate-HMC method satisfy its cumulative
sampling rule after fresh tuning with target acceptance 0.70? The comparator is
the completed 0.95-target run at
`artifacts/surrogate-hmc-48h-extension-20261018-01/run/range_bearing-energy-resume-01/range_bearing/inference/hmc.json`.
Use the same prepared observations, literature prior, endpoint target, force,
particle design, dispersed starts and declared seeds. Retuning the mass and step
size is required; old draws and the old fitted kernel are not initial inputs.

The primary completion criterion is the existing controller's sampling rule:
four chains; at least 2,000 warm-up transitions per chain, readiness on the latest
1,000 with modern R-hat <=1.05, at most 10,000 warm-up transitions; then cumulative
retained modern R-hat <=1.01, at most 20,000 retained draws per chain. All warm-up
draws are archived and excluded from inference. There is no additional discarded
validation phase. Finite energy errors are warnings and MH rejects proposals
normally. Actual acceptance is descriptive, not a convergence criterion or a
requirement that every window attain exactly 70%.

Non-finite states or arithmetic, invalid endpoint status, corrupted artifacts,
and failed movement checks retain their existing validity roles. A failed
tuning candidate triggers the existing bounded tuner repair; it is not evidence
against surrogate HMC as a research direction. Stop at the compute cap or a real
validity failure and preserve all progress. Report acceptance, modern R-hat,
bulk/tail ESS, energy warnings, fallbacks and retained sample counts. A passing
run does not establish exact posterior agreement, optimal tuning, or superiority
over the 95% run. One stochastic rerun supports descriptive comparisons only.
This is a tuning-policy rerun, not a promotion/default benchmark; new heuristic
filter comparisons are outside its scope.

## Settings, budget and assumptions

- FP32 force arithmetic, TF32 off, N=4032, exact chunk K=2016, T=20, 16 flow
  steps, OT epsilon 1 and five iterations, unchanged scope-issued filter tuning.
- Acceptance target 0.70 comes directly from the owner. Initial step 0.005 is
  the previous warm-start hypothesis and is adapted on this target. Retain the
  existing mass windows, 256 step-adaptation transitions and trajectory search.
  Check the emitted config and nomination artifact; do not infer target70 from
  the command alone or assume observed acceptance must equal the target.
- Use the idle RTX 4080 SUPER, UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b,
  with GPU/XLA and verified memory growth. SIR remains on the RTX 5080. Device
  differences preclude attributing a runtime difference solely to tuning.
- This owner-requested rerun has a separate maximum of eight GPU-worker hours
  and one planned attempt, charged separately from the existing 48-hour ledger.
  This is not an extension of SIR's deadline or permission for an open-ended
  search. The cap matches the earlier range-bearing allocation. No automatic
  repeat of a scientific failure is scheduled. A localized infrastructure repair
  may use the unspent allocation under the repository retry policy.
- Versioned output root:
  `docs/plans/artifacts/surrogate-hmc-range-bearing-accept70-20261019-01`.
  Preserve exact argv, environment, Git/source/selection hashes, plan, preflight,
  logs, PID, budget ledger, refreshed phase plan and result paths there.

## Pre-mortem and skeptical audit

The dangerous false success is resuming the old frozen kernel while labelling it
70%; therefore launch without a resume manifest and verify the saved config.
The other risks are stale filter selection, restarting SIR accidentally, using
an energy-warning threshold as a stop, and misreading high acceptance as
convergence. Exact-scope preflight, separate output/GPU/ledger, the inspected
warning-only sampler call chain and the cumulative sampling rule address these.
The initial step, mass regularization and trajectory ladder remain hypotheses
from the existing reviewed campaign, not asserted optimal settings. The smallest
checks are the existing 24 passing focused configuration/queue tests and the new
prepared-scope/argv preflight. The typed verified TensorFlow public tuner remains
the campaign authority, with its documented conditional position-field meaning.

Skeptical audit: PASS for this bounded rerun. The owner selected the only changed
tuning objective, the original result remains intact, the actual consumer will
perform fresh adaptation, and the run can answer the question without imposing
a new validation phase or changing the numerical target. No numerical source
edits are required. Inspect early tuning output and the final evidence before
claiming successful completion. Result note: the output root's `result.md`;
structured final result: `run/range_bearing/inference/hmc.json`.
