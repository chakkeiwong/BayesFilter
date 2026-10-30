# KDM source-training variability calibration — 8 October 2026

Authority: the replication master and accepted 6 October criteria. The owner
authorized continuing replication and added 48 CPU/48 GPU hours; this phase
uses the existing allocation and creates no new grant. Starting active balances:
298987.4806715353 CPU seconds and 159978.1975425437 GPU seconds.

## Question and evidence contract

How variable is the released author's full bearings training on the frozen
development data? The completed author/TF pair had a 0.292–0.382-nat NLL gap
and 2.25–5.81% RMSE gap through the same author evaluator. Matched updates at
both trained origins now agree in float64; float32 failures remain. Neither
finding tells us the distribution of full-training outcomes.

Comparator: pinned original author code b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
source architecture and stage configs, native PyTorch CPU training. This is an
independent reference exception, not a BayesFilter runtime candidate.
Candidate/mechanism: independent training randomness, with all source choices
fixed. No learned parameters, seed or checkpoint will be selected for promotion.

Run exactly two additional independent training units. Each unit includes the
source weight pretraining, MDPF and A-MDPF stages; A-MDPF inherits that unit's own
MDPF best checkpoint. Keep training/validation data seed 202610071 and original
5000/1000 rows. Use stage seed bases 202610101 and 202610111, each plus 0/1/2
for the three stages. These seeds are fixed before results, not selected.

Primary completion criterion: both units reach their source early stop or cap
(400/800/800), pass the independent terminal audit and complete common-author
evaluation. The engineering checks do not qualify statistical noninferiority.
Report all units, including failures. Source-intrinsic failure is a baseline
outcome, not grounds to repair the author. A diagnostic/harness failure is a
repair trigger; source/hash/data corruption or the phase budget is a continuation
veto. Nonfinite results are preserved and block continuous-metric interpretation
for the affected unit.

Evaluation: reuse the exact existing common-author evaluator, fresh-development
data seed 202610081, 256 trajectories, T=150, N=25, B=64, filter seeds
202610082/202610083 and original NLL/RMSE definitions. Evaluate each new source
checkpoint against the same fixed TF checkpoint as a reproducibility control.
Verify the repeated TF metrics and per-step arrays equal the preserved first
evaluation. Retain the first source training as the third calibration unit.

Aggregate filter seeds within each independently trained unit before describing
training variability. The three source units, not trajectories or filter seeds,
are the training-level sample size. Report source means, standard deviations,
range, every unit and the fixed TF gap. Three source units give an unstable
variance estimate. No trained-population ranking, equivalence, noninferiority,
default readiness, HMC readiness or completed replication may be inferred.
Keep first-17/after-17, bearing-residual and angle-seam conditional diagnostics.

The accepted 0.01-nat NLL and 1% RMSE margins remain unchanged. Optionally report
a clearly labeled planning approximation: independent equal-variance arms,
normal mean approximation, zero true difference, 80% power and one-sided 5%
test imply n approximately 2*(z0.95+z0.80)^2*s_source^2/delta^2 per arm. Use
log RMSE for the ratio margin log(1.01). This is not a validated sample size:
source n=3 is small and TF variance/covariance is unknown. Formal qualification
must freeze an adequately powered design and use untouched trained units/data.

Heuristic/baseline scope: replication compares the released method with its
translation. No new superiority requirement against unrelated algorithms is
introduced. Existing exact/simple-model controls and original source comparators
remain in the master; training variability is an explanatory diagnostic only.

## Source anchors, defaults and skeptical audit

Source configs and call chain are inherited from the audited complete-training
phase. Check the actual pinned src/trainers/trainer.py controller,
src/trainers/specific_trainers/full_sequence_trainer.py::do_forward_pass and source model before
claim-bearing changes. This phase changes only explicit seed parameters and
supervision, not those computations. Terminal auditing replays the actual source
scheduler/early-stopping classes and checks optimizer cadence, frozen parameters,
own-stage transfers, full data coverage and checkpoint identities.

| Choice | Provenance and role | Risk | Early check |
|---|---|---|---|
| Frozen source objective, architecture, rates, Adam and early stopping | Pinned source; baseline | Convenient replacement would answer a different question | Same source/config hashes and existing terminal audit |
| Two additional units | Bounded calibration hypothesis | n=3 insufficient for precise variance or qualification | State sample size and retain every result; no adaptive seed selection |
| Fresh stage seeds | Prespecified independent development units | Reusing a baseline constructor defeats the test | One-epoch cost reference per seed; distinct constructor hashes; full run must reproduce its own constructor |
| Fixed train/validation data | Existing source data; conditional calibration | Does not estimate data-generation variation | Identical serialized dataset hashes for all units |
| CPU one-thread source training | Audited independent reference environment | Hardware arithmetic differs from GPU author installation | Same environment as first source baseline; no broader hardware equivalence claim |
| 256-row common-source evaluation | Existing development protocol | Finite evaluation noise and source-evaluator-only coverage | Same data and filter draws; exact fixed-TF replay; no qualification claim |
| Author zero added ridge/clipping protections | Owner's replication target and pinned source numerical policy | Intrinsic source instability | Preserve finite diagnostics and source failures; repairs remain deferred |

Pre-mortem: a run could pass all controller checks yet training variability
remain much larger than the acceptance margin. That is evidence about required
precision, not successful replication. Selecting the most favorable new seed,
counting filter seeds as training replicates, comparing different datasets or
changing the author's stopping rule would invalidate this question.

Skeptical audit verdict: proceed with the bounded calibration. The baseline is
the actual source, numerical screens are not relaxed, failure strata are retained,
TF evaluation is held fixed, and n=3 is expressly exploratory. There is no
scientific basis to call this the final qualification. No implementation repair
is licensed by this source-only phase.

## Executable phases, supervision and budget

1. Extend the source reference helper with an explicit constructor/stage seed,
   preserving the original default and constructor/data identity assertions.
   Verify parsing and the seed/data wiring.
2. Generate a one-epoch source reference for each new seed. Require distinct
   constructors and exactly identical frozen datasets. Each full run consumes
   its own reference, so the original exact constructor checks remain intact.
3. Execute two full source units, with at most two independent CPU processes.
   Each uses one Torch compute thread and the original audited DataLoader
   setting of one worker. These are numerical workers, not AI agents.
4. Audit each full training terminal state, then run the existing author evaluator
   on each new source checkpoint and fixed TF checkpoint.
5. Preserve a calibration summary and decision/inference tables, update the
   budget ledger once, and refresh the master checkpoint.

Fixed phase CPU cap 131400 seconds (36.5 hours); GPU cap zero. This counts summed
worker elapsed time, even when workers run concurrently. Reserve 240 routine
CPU seconds. Each full training has a 64800-second limit, each cost reference
300 seconds, audit 120 seconds, evaluation 300 seconds: total at most
2*(64800+300+120+300)+240 = 131280 seconds. The residual 120 seconds is not
permission for another full unit. No seed replacement or additional trained unit.
Heartbeat timeout 1800 seconds per training child; 30-second supervisor poll;
on a timeout terminate only that child's process group, preserve its output,
record classification and continue the other independent unit if valid.
No silent restart or resumed optimizer-state approximation.

Run from /home/chakwong/BayesFilter with the existing approved interpreter:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_kdm_source_variability.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-source-variability-20261008-01

The supervisor starts isolated tf-gpu2 CPU author workers with GPU intentionally
hidden, LD_LIBRARY_PATH set to that environment, and one-thread BLAS/Torch.
All commands, seeds, Git/source hashes, environment, wall times, progress,
audits and per-step metrics stay under the versioned output. No old output
is overwritten. Logs are retained in full; the live checkpoint shows progress
and remaining budget. Result note:
controlled-smc-kdm-source-variability-result-20261008.md.

Pre-launch harness finding: author dataset serialization includes its output save path
(src/datasets/bearings_only_dataset.py:430–446), so file hashes from different
output directories cannot establish data inequality. Preserve both file hashes and
require exact dtype/shape/tensor equality against the preserved original dataset.
The first evaluator preflight reproduced every source/TF comparison and fixed-TF
per-step value exactly; its file-identity assertion failed. This is a harness repair,
not a relaxed numerical bound. The corrected regression passed before full training.

Pre-launch review completed: the repaired evaluator re-created the prior source/TF
comparisons exactly and passed all 16 fixed-TF replay controls, with exact dataset
dtype, shape and values. Worker-result parsing and calibration-assembly failures
now preserve terminal failure records instead of leaving an apparent running job.
Focused negative checks rejected malformed JSON and changed/unchecked data and
confirmed one budget charge on terminal assembly failure. Eight final checks
passed; details are in
artifacts/controlled-smc-kdm-replication-20260928/kdm-source-variability-prelaunch-review-20261008.json.
The real evaluator rerun took 81.1233 CPU seconds; it and the first preflight
are covered by the fixed 240-second routine reserve. No numerical threshold,
source algorithm, training configuration or scientific criterion changed.
The plan survives the skeptical audit: the controls answer reproducibility,
and the two new units answer source training variability only.
