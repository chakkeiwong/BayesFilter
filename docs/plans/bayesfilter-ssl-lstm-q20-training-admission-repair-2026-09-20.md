# q20 bounded training validation and HMC trial admission

The owner approved repairing the excessive training-validation requirement and
recomputing the time estimate. The question is whether a learned, numerically
healthy map is ready for a bounded downstream HMC trial. A precisely measured
optimization plateau does not answer whether HMC will mix or estimate this
posterior accurately. Plain NeuTra, then the tempered NeuTra ensemble if needed,
remain the permitted methods for the same q20/T30 float64 UKF target. Latent mass
remains identity; downstream tuning and posterior criteria are unchanged.

For a frozen differentiable invertible map `theta=T(z)`, change of variables
gives `pi_z(z) proportional to pi_theta(T(z))*abs(det DT(z))`. A reversible,
volume-preserving HMC proposal with the appropriate Metropolis correction
preserves that transformed density, and pushing it through T returns pi_theta.
Neither this identity nor the correction contains an optimization-plateau
condition. The map must satisfy the numerical and transformed-target contract;
its quality affects mixing and efficiency. This argument concerns the specified
UKF target and does not establish convergence, mode coverage or exact nonlinear
likelihood inference.

## Decision and evidence contract

Use the existing paired initial bank (768 rows for the serious protocol) once
per scheduled training rung. Compare the current map with its beta-start map
and preceding checkpoint on the same base rows. The development interval is
descriptive because the bank is reused adaptively. For loss_new-loss_old,
baseline upper endpoint < 0 is a learning screen; incremental lower endpoint
> 0 is a deterioration repair trigger. Zero is derived from the direction of
the objective, not a newly calibrated effect-size threshold. These screens
nominate a trial and do not support a nominal confidence or optimality claim.

After the inherited cohort floor (512 updates per current scope), learning,
absence of clear deterioration, valid target/score rows and map parity permit
an HMC trial. Continued improvement or an unresolved small incremental change
does not prevent that trial. Before the floor, continue training. If baseline
learning is unresolved, continue to the next funded rung; at the cap, retain an
inconclusive candidate result. Do not purchase larger banks to establish a fine
plateau. Plateau counts and the old .04/.02 resolution remain explanatory only.
Numerical failure and deterioration cannot gain trial admission.

Every declared seed reaches the existing floor before selection. Stop updating
an eligible map at that beta. Plain HMC needs one eligible map; the ensemble
still needs the declared charts and temperatures. Choice uses fixed cohort
order, never a descriptive loss ranking. HMC trial admission is distinct from
kernel verification and posterior admission. The actual downstream computation
remains the test of usefulness.

| Check | Role |
| --- | --- |
| Target values/scores, frozen-map forward/inverse/Jacobian parity | Numerical veto; no relaxation. |
| Paired loss learning and deterioration screens | Trial nomination or training-repair trigger only. |
| Plateau, .04 change, .02 half-width | Explanation only; cannot block a trial or expand validation. |
| Public frozen-transport tuner and independent kernel verification | Existing kernel-admission requirements. |
| Sequential R-hat, ESS, MCSE and same-target reference assessment | Existing posterior criteria; unchanged. |
| Checked preserved source/state, focused regressions and complete prices | Engineering validity; missing downstream prices forbid a complete-campaign ETA. |

The 768 rows, 512-update floor, architecture/LR/seed grid, reliability rows and
tolerances, and factor-two forecast reserve are inherited hypotheses. The small
bank resolves the gross improvement in the eight inspected historical maps;
that motivates reusing it for trial nomination, not a universal precision claim.
Risks are noisy nomination, delayed discovery of poor mixing, and wasted tuning
on weak maps. Numerical checks and bounded downstream tuning expose these risks.
No new numeric threshold or claimed validation-to-training optimum is introduced.

## Repair and verification

1. Change the assessment, training stop and HMC consumer together. Preserve
   loss precision diagnostics while making the trial-only scope explicit.
2. Bound active validation to the first configured bank, including resumed
   protocols with the historical expansion ladder. New templates contain only
   that bank. Price the same first-bank work at both training floor and cap;
   retain any historical large-bank counterfactual only with an explicit role.
3. Import compatible source changes through the existing checked coordinator
   path. Preserve map/Adam/RNG/history/cache identities. Clear historical
   admission and reassess current checkpoints without repeating optimizer work.
   Reject any unrelated numerical-source change.
4. Test improving and unresolved-increment admission, learning-unresolved and
   deterioration rejection, numerical vetoes, pre-floor/calibration rejection,
   consumer enforcement, bounded validation, saved-state reassessment, and
   pricing arithmetic. Run CPU-hidden tiny references only.
5. Reprice with the existing measured q20 GPU/XLA update and heldout rates after
   checking that the numerical kernels match the preserved execution tree.
   Preserve compile/setup costs, migrated-map overhead, checkpoint credit and
   the factor-two reserve. Report training floor and cap separately. The master
   is serial across candidates; multi-GPU division is not a measured ETA.
6. If verification passes and diagnostic allowance remains, execute only the
   master's `price` mode for plain NeuTra in the isolated tree. This performs
   current-source beta-one GPU/XLA qualification and bounded real transition,
   reference and analysis timing, then stops before serious training or tuning.
   This fills the previously unmeasured downstream categories. Use the unchanged
   saved protocol and checked checkpoint import, with the settled allowance.
   Each worker retains the saved 1,200-second ceiling and is also bounded by
   remaining diagnostics; do not enlarge it or automatically repeat a timeout.
   Memory growth and idle-device selection remain required. The measured maps
   are unqualified pricing inputs, not posterior evidence. If this timing is
   unavailable, report the training estimate and the exact missing categories.

The bridge and component target expose combined value/score evaluation. A new
value-only kernel would need numerical/status parity and fresh GPU timing.
Keep the combined evaluator for this repair and claim no value-only speedup.
The existing measured prices therefore remain useful for a recomputed forecast,
with their original uncertainty and coverage limits. No long training or HMC
campaign is launched by this repair-and-estimate task.

## Skeptical pre-execution audit

The old baseline wrongly required optimization resolution to permit a downstream
trial. Replacing it with a trial-only learning screen corrects that decision
without treating loss as posterior evidence. Retaining the actual value/score
kernel avoids a hidden target/status change. Cached losses can be reused only
with their exact numerical identities; old admission cannot. Tests must expose
the risk that an import clears exports but skips reassessment of completed
rungs. Pricing must count current behavior, not the retired maximum-bank policy,
and must not pretend unmeasured HMC/reference costs or serial GPU division are a
complete ETA. These revisions address wrong baselines, proxy promotion, stale
context, hidden assumptions, environment mismatch and misleading artifacts.
The plan passes this audit subject to the focused checks above.

## Budget and artifacts

Carry forward 78,552.22873334838 campaign seconds, including
2,792.6609520684797 diagnostic seconds. Verification and repricing consume both
balances; no new allowance is granted. Each test invocation has a 600-second
process ceiling, an inherited engineering bound, and must fit the remaining
diagnostic allowance. Stop on failed validity checks, unexplained source/state
changes, or exhausted allowance. Infrastructure failures allow localized repair.

Use a fresh source snapshot based on
`/tmp/BayesFilter-q20-estimation-execution-20260920-r2`, adding only this repair.
Preserve results under
`docs/plans/artifacts/ssl-lstm-q20-training-admission-repair-2026-09-20/`.
Record source hashes, exact commands, environment, CPU-only device hiding,
verification wall time, original timing inputs, forecasts and settled budget.
The result note will distinguish engineering repair, descriptive cost forecasts,
and the absence of new posterior evidence.

## Completed repair, verification and measured costs

The assessment now emits `hmc_trial_nominee` / `hmc_trial_eligible` for a learned,
numerically healthy map after the floor, without requiring a plateau. Both the
master selector and HMC consumer require current trial eligibility. Validation
uses only the first bank even when a saved protocol retains the old ladder.
Eligible maps stop training at that beta. The price model reserves this bounded
validation work at both the floor and training cap; the historical maximum-bank
scenario is explicitly inactive. Value/score, map parity, public tuning,
sequential sampling and posterior acceptance calculations were not weakened.

Checkpoint import preserves exact map/Adam/RNG/history and cache identities,
clears historical admission, and reassesses the current endpoint. The first
verification run found a calibration-to-full-run mismatch: direct resume skipped
the current endpoint while an import reassessed it. Both now reassess without
extra optimization. Review also found that repeated import could erase a saved
deterioration/numerical veto. The veto now survives import and calibration resume
until an actual update, with focused regressions for both failure classes.

There are **77 distinct checks with passing latest outcomes**. Four invocations
recorded 59 pass / 1 fail (56.215 s), 36 pass (238.010 s), 7 pass (15.283 s), and
21 pass (1.960 s). Overlapping invocations are not counted as extra distinct
checks. The original failure is resolved. Tests cover cached-loss identity and
XLA tracing, bounded-bank behavior, trial nomination and rejection, imported
state, repeated veto preservation, migration, cost accounting and actual tiny
public tuning/posterior/ensemble/reference integrations. GPUs were intentionally
hidden for these CPU reference tests. No independent reviewer was used.

The real trusted GPU master completed `price` mode on host GPU 2 with verified
memory growth, float64 TensorFlow 2.20.0, TF32 enabled and XLA enabled. Current
pricing-source beta-one qualification passed with one enclosing trace and one
public-runner trace. The supervisor charged 864.306893909 s for readiness,
qualification and pricing. Four-chain HMC used one batched public runner. No
serious training, kernel tuning or posterior estimation was launched.

| Training floor, 12 plain-NeuTra candidates | Previous quote, maximum validation banks | Repaired procedure, fresh GPU timings |
| --- | ---: | ---: |
| Optimizer work | 4.009 h | 4.104 h |
| Heldout validation | 10.070 h | 0.710 h |
| Setup | 0.037 h | 0.038 h |
| Total before reserve | 14.116 h | 4.852 h |
| Factor-two reservation | 28.232 h | 9.704 h |

The new validation cost is 17.3% of optimizer time. It includes 30,720 target
rows over 40 distinct maps, including migrated baselines. Applying the new rule
to the *old same timing inputs* gives 4.739 h raw / 9.478 h reserved; fresh
timings are slightly slower. These are descriptive forecasts, not evidence that
hardware or either architecture became faster. The initial 512-update floor
must yield eligible maps for these floor costs to suffice. If maps require
later rungs, training costs increase. Funding all candidates through 8,192
updates remains an expensive optional maximum, not a claim about required work.

| Four-chain HMC trajectory | Measured steady seconds per transition | Earliest posterior check: 2,000 warm-up + 1,000 retained | Both sampling caps: 20,000 total |
| --- | ---: | ---: | ---: |
| L=3, widths 16/32 | 4.633–4.663 s | 3.87–3.90 h | 25.75–25.91 h |
| L=25, widths 16/32 | 28.335–29.015 s | 23.65–24.22 h | 157.45–161.23 h |

These conditional sampling times include the observed first-call cost as a
conservative setup allowance. They exclude tuning, posterior analysis, reference
assessment and the factor-two reserve. Four chains are already included; do not
multiply by four again. No measured L has been tuned or qualified for posterior
estimation. Longer warm-up or sampling may be necessary, and either cap may fail
to produce an estimate. The reference batch measurement was 2.372 s per 32 rows;
the configured maximum reference reservation is 5.412 h with the factor two.

The downstream reservation remains a separate unresolved budgeting issue.
`forecast_campaign` prices all 200 tuning work items at the largest evidence
rung and slowest measured trajectory, both sampling caps, the reference cap,
and a complete additional repair scope, then applies the factor two. It returns
2,198.222 h, including 930.284 h for tuning, 322.470 h for sampling and 930.284 h
for repair. It does not forecast how many of those work items will actually be
needed, and it does not use the controller's eight-hour tuning deadline as a
bound on that work-count estimate. Calling this a minimum or expected runtime
would be wrong. The result now explicitly names it
`configured_cap_reservation_seconds`; the old field is retained as a labeled
compatibility alias. This repair does not authorize that much time or change
downstream caps. A full master launch would still stop under this reservation
policy. The next budgeting repair should allocate bounded successive stages
inside the existing allowance instead of requiring all theoretical maxima to
be funded upfront; preserve the posterior criteria.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept validation/admission repair | 77 distinct latest checks pass; saved numerical state preserved | No unresolved engineering failure | Development screen may nominate weak maps | Use current trial gate and bounded bank | Whitening or posterior adequacy |
| Accept training reforecast | Same numerical kernels, complete plain-NeuTra pricing scopes | No scope or device failure | Later training rungs may be needed | Reserve bounded floor work when downstream allocation is ready | Entire campaign takes 4.85 h |
| Accept conditional HMC timings | Actual public batched runner at both widths and L endpoints | Pricing health checks passed | Tuned L, future maps and convergence | Use these costs in a staged spending plan | Any L is posterior-ready |
| Reject 2,198 h as a required-time claim | It is an all-caps work reservation | Existing affordability gate would block launch | Actual adaptive workload | Repair downstream budgeting before full execution | More than the authorized allowance is necessary |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Engineering regressions resolved; pricing/qualification health passed. No new posterior screen was run. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Runtime across widths and repeated timing runs; no speed ranking is supported. |
| Default readiness | The user-approved trial-admission policy is implemented; no map or posterior is promoted by this repair. |
| Next evidence | Bounded target-specific kernel tuning, then sequential posterior and reference checks. |

Terminal review checked the complete consumer chain, exact seven-file source
delta, preservation of all eight migrated sessions, cost scenarios, genuine
GPU/XLA/memory-growth receipts, and all completed-stage artifact checksums.
The strongest alternative explanation for a later failure is inadequate map
mixing despite a decreasing reverse-KL loss; that triggers downstream repair,
not reinstatement of a finely resolved plateau. The weakest cost evidence is
the small timing sample and unknown behavior of future learned maps. A measured
downstream failure can overturn trial viability, and an affordable staged budget
can overturn the current all-caps affordability stop.

## Reset memo and active continuation state

All attempts are settled. Remaining campaign allowance is
77,367.92183943938 seconds (**21.4911 h**), including
1,608.35405815948 diagnostic seconds (**26.8059 min**). Use the output root's
`settled-allowance.json`; the balances in the preserved protocol are historical
inputs, not renewed authority. No worker from this task remains running.

Active code is `/tmp/BayesFilter-q20-training-admission-20260920-r3`, with the same
seven repaired files present in the main workspace. It derives from the previous
isolated execution tree, preserving its numerical kernels rather than importing
concurrent workspace HMC changes. GPU pricing used `r1`; `r2` only preserves
existing import vetoes, and `r3` additionally clarifies the cost-report labels.
Those changes do not alter the priced numerical work. Qualification receipts
retain their actual source identity and are not silently relabeled for r3.

The active checkpoint is `training-import-active/cohort-00000.json` under this
output root, and the protocol is `protocol.json`. Original optimizer state and
4,096 historical updates remain preserved; current-scope update credit is still
zero as in the prior migration. All maps need current training assessment.
`result.json`, `training-forecast.json`, `source-review-active.json`,
`repair-final.patch`, `manifest.json`, verification XMLs, and `pricing-01/`
preserve the repair, exact commands, forecasts and evidence. No commit was made.
