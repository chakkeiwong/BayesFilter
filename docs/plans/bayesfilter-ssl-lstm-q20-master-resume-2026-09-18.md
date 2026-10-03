# q20 master refresh after the 512-update continuation

Status: refresh stopped at 14:28 Shanghai time with
`MASTER_REFRESH_PRICING_INCOMPLETE`; the localized diagnostic reproduced and
identified the bootstrap veto. No numerical worker is currently running.
Authorization: the owner requested “refresh the master program and continue the
work.” The settled balance is 102511.166122 campaign seconds, including
26751.598340 diagnostic seconds. This phase does not renew either allocation.

## Intent, evidence and boundaries

Question: can the master reuse the completed optimizer states and validation
banks, correctly price only remaining training, and measure downstream costs
without the training reservation preventing that measurement?

The baseline is the completed September 18 continuation at eight maps × 512
updates, using `/tmp/BayesFilter-q20-all-gpu-20260917`. Keep its TensorFlow/TFP,
strict FP64 q20/T30 target, XLA, batch32 trainer, seeds, Adam settings, training
inventory, validation criteria and posterior contract. The complete cohort
still has three roots and continuation at beta1. This diagnostic cannot
replace those requirements with its one-root calibration.

NeuTra gate audit: this remains the owner's explicitly requested q20
engineering/diagnostic campaign. No paper-replication or enhancement gate is
closed by cost measurements or the previously completed learning diagnostics.

| Role | Criterion or action |
| --- | --- |
| Engineering acceptance | Imported tensors, Adam, RNG, histories and cached loss prefixes are preserved; ordinary restart is exact; numerical source changes reject; measured completed work is credited exactly once |
| Numerical veto | Invalid target/proposal status, nonfinite state, failed graph/endpoint checks, unsupported source change or corrupt checkpoint/cache prevents use of that result |
| Continuation veto | Campaign/phase cap, unavailable GPU capacity, invalid shared evidence or source drift stops the affected execution with preserved artifacts |
| Repair trigger | A localized infrastructure interruption or a preparation failure with valid target evidence; preserve costs and diagnose rather than reject NeuTra |
| Explanatory | First/steady timing, memory, loss, short-chain acceptance and pricing geometry; no ranking or convergence conclusions |
| Primary output | A resumed master ledger, reviewed checkpoint import, remaining-work quote and measured downstream costs with explicit missing categories |

Independent reference integration, canonical tuning, retained sampling and
confirmation remain separate work. A runner or preparation helper cannot issue
a tuning artifact. The current HMC interface guide and capability registry were
consulted; this phase measures the existing public runner and operational mass
preparation, without changing HMC settings or granting new replay authority.

## Repair and execution

1. Add a master checkpoint-import interface. Check the original complete source
   inventory against the preserved source directory. Permit only the reviewed
   coordinator/cost/import files, the training scheduler function and the
   already-tested assessment function to differ. Keep all original optimizer
   and map scopes, hashes, history and validation cache identities. Record the
   import and current execution source separately; never rewrite prior evidence
   to pretend it was produced by new code. Numerical target/score/trainer/map
   changes remain a real incompatibility. Preserve the old export files but
   clear active imported export/nomination status until a current assessment.
2. Make the cost calculator subtract completed updates and cached validation
   prefixes. Do not credit unperformed higher-beta scopes, missing roots or
   unmeasured downstream costs. Compile/setup overhead for new workers remains.
3. Add `refresh` to the existing master. It imports the checkpoint, performs
   fresh GPU qualification, measures current training prices and downstream
   transition/reference/analysis work, then separately measures classical
   preparation. Training's affordability verdict must not suppress this bounded
   diagnostic. Preserve partial pricing and the missing-cost inventory if a
   category cannot complete within its cap. Refresh alone does not launch an
   unaffordable full cohort or posterior campaign.
4. Use an isolated copy of the exact previous numerical checkout plus these
   reviewed changes. Preserve the concurrent HMC/inference work in main. Run
   focused CPU tests for source compatibility, numerical restart, cache reuse,
   remaining-cost arithmetic and real master dispatch. Then launch the refreshed
   master as a durable, deadline-bounded service with one campaign ledger.

GPU workers freshly choose among all three devices before TensorFlow import,
enable and verify memory growth and preserve capacity observations. Current
stages execute serially; this repair does not claim multi-GPU scheduling. The
four HMC chains remain one batched XLA execution. No package or driver changes,
NumPy runtime, pfor, unreviewed backend or numerical-default changes are needed.

## Budgets and provenance

Use `docs/plans/artifacts/ssl-lstm-q20-master-resume-2026-09-18/` for new evidence.
The phase permits at most 900 CPU-test seconds plus a 180-second
setup/accounting envelope, and at most 14400 seconds for actual GPU pricing and
qualification. Those are engineering allocations from the existing 7.431-hour
diagnostic allowance, not runtime predictions. Every failed or interrupted
attempt is charged; retries share the same phase cap. Reserve at least
11271.598 diagnostic seconds and 87031.166 campaign seconds after the maximum
phase debit. The supervisor owns actual wall-time accounting and timeout
termination; a systemd service bounds its lifetime too.

Two qualification stages use at most 1200 seconds each, inherited from the
successful previous q20 qualification envelope. Training repricing uses at
most 1200 seconds (the previous complete measurement was about 183 seconds).
Downstream transition/reference pricing has at most 6000 seconds: previously
measured L3 four-chain work took about 6.8 seconds per transition, and the
largest declared L is 25, so twelve width/beta/L measurements with compile,
exchange and reporting overhead require a larger envelope. Linear scaling in
L is an explanatory estimate, not a bound. Each classical preparation gets
at most 3600 seconds, further limited by the shared 14400-second phase balance;
this is a bounded feasibility measurement, not evidence that one hour suffices.
Missing preparation cost remains missing if it fails or times out.

The inherited factor2 runtime reserve stays visible as an engineering
hypothesis. The 512 floor is a planned calibration boundary, not sufficient
training by fiat; 2048/8192 and all validation/posterior thresholds are unchanged.
Short pricing runs use the existing four transitions and two repeats, which
measure execution only. Learned-map price representatives use deterministic
cohort order within the existing width/beta scope, never observed loss ranking.
All prices state which trained or initialization map was used. Reference timing
must retain the score/status work of the actual reference evaluator; timing only
an unused likelihood could let XLA eliminate required checks and underprice it.

## Skeptical audit and pre-mortem

The old master has two confirmed defects for continuation: it quotes training
from zero, and its early training-cost rejection prevents completing downstream
pricing. A third risk is silent source migration: all source hashes include the
coordinator, so simply changing the master and rewriting checkpoint scopes is
unacceptable. This plan uses a checked, recorded import and preserves original
state identity. Tests must reject a target/trainer edit and compare actual
continued tensors and optimizer states with uninterrupted execution.

Audit passed with those repairs specified. Completion of a cost command does
not establish affordable full research, useful mass geometry or a qualified
posterior. Failed classical preparation can be a start/tuning failure, not a
model failure; retain its diagnostics. Known-point reference timing may not
represent tail costs, so use actual prior-distributed proposal batches and
keep the estimator's limitations. Full pricing must identify any unmeasured
mixture-specific work instead of calling a replica-exchange proxy exact.

No new sampler derivation or literature-dependent method choice is introduced.
MathDevMCP's prior Adam-scaling audit and the existing target source audit remain
applicable; symbol checking would not establish scheduler/accounting correctness.
The smallest useful evidence here is executable restart and cost-inventory tests.

## Execution record

The focused suite passed all 27 tests in
`/tmp/BayesFilter-q20-master-resume-20260918`. The supervised CPU-only test wall
time was 109.468124 seconds; GPU devices were intentionally hidden. Results are
in `tests-001/execution.json`, `tests-001/pytest.xml` and `tests-001/output.log`
under this phase's artifact root. Checks cover source incompatibility, exact
Adam/RNG/cache restoration, remaining-work credits, real master workers and
completed-stage replay without a second charge. Syntax and whitespace checks
also passed. The final skeptical check found no new scientific or numerical
change: full mixture dispatch and metric-specific transition costs remain
explicitly unmeasured categories.

Debit the measured test time and the predeclared 180-second setup/accounting
envelope once, leaving 102221.697997 campaign seconds and 26462.130216 diagnostic
seconds before GPU work. `execution-allowance.json` records this debit and its
predecessor. `integration.json` records the preserved numerical source inventory
and the nine permitted changed paths. No concurrent main-branch numerical
changes enter the isolated execution source.

Launch the existing master with the following command. The systemd lifetime
limit is the 14400-second shared phase cap plus a 60-second coordinator cleanup
envelope; worker costs are settled by the master ledger.

```sh
systemd-run --user --unit=bayesfilter-q20-master-resume-20260918-r1 \
  --property=WorkingDirectory=/tmp/BayesFilter-q20-master-resume-20260918 \
  --property=RuntimeMaxSec=14460 --property=TimeoutStopSec=10 \
  --property=KillMode=control-group \
  --setenv=TF_FORCE_GPU_ALLOW_GROWTH=true \
  --setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
  --setenv=TF_NUM_INTRAOP_THREADS=2 --setenv=TF_NUM_INTEROP_THREADS=2 \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py refresh \
  --config /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-refresh-2026-09-17/all-gpu-protocol.json \
  --training-checkpoint /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-training-continuation-2026-09-18/campaign/attempts/00001-continuation/worker/data/cohort-00033.json \
  --previous-source-root /tmp/BayesFilter-q20-all-gpu-20260917 \
  --budget-record /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-resume-2026-09-18/execution-allowance.json \
  --output-dir /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-resume-2026-09-18/campaign
```

The service started at 2026-09-18 13:15:00 Shanghai time. The import preserved
all eight maps and 4096 completed updates. The first worker selected host GPU 1
from the three-device inventory, verified memory growth before initialization
and began XLA qualification. The service deadline is 17:16 Shanghai time;
individual stages and the shared phase budget may stop it earlier. Use
`continuation.json` for the live service and `campaign/campaign.json` for
accounting. While an attempt is running, its cap is an outstanding budget hold.
The predecessor's settled balance is historical and must not fund another run.

At 13:19 Shanghai time, beta .5 qualification had passed and beta1 qualification
was running on host GPU 1 with memory growth verified. The first stage charged
182.048176 seconds. Its actual public runner preserved the four-chain shape,
traced once, and passed finite proposal/target/status checks. This establishes
graph and endpoint checks only; trained-map restoration and downstream pricing
are subsequent stages, and posterior qualification remains open.

## Refresh result and localized reporting repair

The first refresh finished at 2026-09-18 14:28:23 Shanghai time after
4401.587993 supervised seconds. Both beta qualifications, exact restoration of
all eight 512-update states, training repricing and downstream timing completed.
Classical preparation failed at beta .5 and beta1: the public preparation helper
raised `HMCPreparationFailure` after a hard-vetoed bootstrap round. The q20
pricing wrapper discarded the exception's structured details and did not attach
the already-available public progress callback. That reporting defect prevents
identifying the actual veto from the first run. It does not establish a broken
target, method, GPU or checkpoint.

The settled balance at that boundary is 97820.110005 campaign seconds, including
22060.542223 diagnostic seconds. Remaining training to the 512 floor is
56706.362980 raw seconds (15.7518 hours) with first-bank assessments, or
113412.725959 seconds (31.5035 hours) under the existing factor2 reservation.
This forecast still excludes complete downstream work and does not establish a
mathematical minimum budget. Classical preparation, metric-specific transitions
and multi-chart mixture dispatch remain missing cost categories.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept imported state and graph checks | Eight exact restores and both four-chain XLA checks passed | No source, state or graph veto | Trained-map downstream validity | Preserve completed work | Posterior or production qualification |
| Keep total pricing incomplete | Training and downstream primitives measured | Classical bootstrap rejected twice | Actual preparation veto was discarded | Retain structured progress and replay one failure | Target or NeuTra rejection |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Classical bootstrap failed; cause not yet preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Training and HMC timing from short runs |
| Default readiness | Not established |
| Next evidence needed | Exact bootstrap cause, remaining cost categories and downstream posterior checks |

Localized repair: attach `HMCPreparationProgress` in `price_preparation`, preserve
the original exception and expose the progress path on success. The mathematical
preparation, target, starts, seeds and settings are unchanged. Add one focused
success/failure regression, including a nonfinite diagnostic value. Test in a
separate CPU-only copy with a 120-second maximum, charged to this campaign.

For the diagnostic replay, use
`docs/benchmarks/diagnose_q20_preparation_2026_09_18.py` against the unchanged
`/tmp/BayesFilter-q20-master-resume-20260918` numerical sources. It reuses the
public helper, verifies exact starts and configuration against the failed beta
.5 attempt, and records its public progress and structured exception. Its script
hash, exact command, source inventory and GPU growth record are saved. The old
qualification is reusable because its numerical source inventory remains exact.
It issues no tuning or posterior authority.

The replay has a 1200-second external cap, chosen as a debugging allocation
against the observed 270.56-second failure. The original preparation limit is
unchanged and the earlier failed attempt remains charged. Tests plus replay
count against the original 14400-second phase cap (9998.412007 seconds remained
before this repair), the original preparation-stage envelope and existing
campaign/diagnostic balances. Preserve the first terminal ledgers before
appending these attempts. No new allocation is granted.

Evidence contract: reproduce the identical preparation call and retain the
specific veto and any captured underlying exception. Matching source, starts,
settings and finite artifacts are required; mismatch or budget exhaustion stops
the diagnostic. Timing and acceptance remain explanatory. A reproduced local
bootstrap failure triggers the smallest justified repair, not abandonment of
the research direction. A different outcome remains an unresolved reproducibility
finding. Post-run interpretation must distinguish an implementation/telemetry
failure from a numerical proposal failure.

Skeptical audit passed: the plan compares identical numerical inputs, does not
weaken the failed veto, uses the established public reporting hook and preserves
both failures. The cheap regression catches lost or unserializable diagnostics.
The strongest alternative explanation is genuine invalid proposals from the
prior-scale geometry; the replay must expose evidence before changing settings.

## Preparation diagnostic result

The two focused reporting regressions passed in 4.504157 supervised seconds.
The exact-input replay then reproduced the failed bootstrap in 277.572426
supervised seconds on host GPU 2, with verified growth, identical starts and
identical preparation settings. The numerical source inventory remained exact.
Its saved `preparation_progress.json` identifies the first round's sole hard
veto as `screen_log_accept_nonfinite_or_missing`: 30 of 32 log-acceptance values
were nonfinite. All 32 retained states and retained target values were finite;
binary acceptance and movement were both zero. Proposed target values were
nonfinite. The two finite log-acceptance values were approximately -4.03 million
and -4.73 million. These are measured numerical failure diagnostics, not a
convergence or ranking result. Accepted-state status alone could not establish
validity of the rejected proposals.

The failed call used the classical prior-scale geometry, not a learned NeuTra
map. The checked source constructs covariance `diag(scales^2) + jitter*I` from
`scales=4` and uses its diagonal inverse as its curvature proxy. Consequently
the mass-scaled proxy frequencies are one, independent of q20 posterior
curvature. With dimension four and the configured geometry scale .5, the
initializer gives epsilon approximately `.5 * 4^(-1/4) = .353553` and five
leapfrog steps. This is a prior-scale initialization hypothesis, not measured
posterior whitening or a calibrated stability limit. The q20 physical-coordinate
timing probes at epsilon .01 were finite, but they used a different coordinate
system, trajectory and seeds; they do not directly qualify this bootstrap.
Source anchors in the preserved checkout are
`hmc_kernel_tuning.py::_select_geometry_hint`, `_curvature_frequencies`,
`_initial_step_size` and `_classify_bootstrap_screen`.

The bootstrap classifier stops immediately on a nonfinite log acceptance and
does not propose an epsilon repair for that round. The retained veto is valid.
The next discriminating action is a separately recorded initialization or
geometry calibration using the public preparation interface, with unchanged
target, finite-proposal requirements and downstream qualification. It must
test the prior-scale assumption rather than ignore the veto or treat accepted
state finiteness as sufficient. It is not justified to label this a failed
NeuTra transport or to discard the trained checkpoints.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept reporting repair | Success and failure regressions passed; exact replay retained structured causes | No source/start/settings mismatch | None for the identified reporting omission | Keep progress recording in the master | Numerical preparation repaired |
| Reject this bootstrap configuration | 30/32 nonfinite log acceptances and nonfinite proposals | Hard numerical veto reproduced | Step initialization versus inaccurate geometry versus target tails | Calibrate preparation initialization with finite-proposal screens | Invalid target, failed NeuTra, posterior readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Initial classical bootstrap fails on nonfinite proposals |
| Statistically supported ranking | None |
| Descriptive-only differences | Timing and finite log-acceptance magnitudes |
| Default readiness | Blocked by preparation and other incomplete evidence |
| Next evidence needed | A valid prepared classical baseline, missing cost measurements and qualified downstream runs |

After both attempts, the authoritative ledger has charged 4683.664576 seconds
for this refresh and repair, leaving 97538.033422 campaign seconds (27.0939
hours), including 21778.465641 diagnostic seconds (6.0496 hours). The original
phase has 9716.335424 seconds remaining; it was not renewed. All attempts are
settled. The first terminal state and result are archived as
`campaign/refresh-terminal-001-*.json`; the diagnostic is in
`campaign/preparation-diagnostic-result.json` and attempt `00007`.

Post-run red-team: the first round can fail because a broad prior geometry is a
poor local posterior approximation, because the initial step is too large, or
because proposals expose a deeper numerical target defect. This replay
separates that numerical proposal failure from the lost-reporting defect; it
does not distinguish those remaining causes. A smaller, coordinate-consistent
initialization that still produces invalid proposals would weaken the simple
step-size explanation. Native divergence telemetry was unavailable, so no
zero-divergence claim is made.

## Problem assessment after the call-chain audit

The failure is understood at the diagnostic level, but its numerical repair
has not yet passed a q20 experiment. A further read-only trace confirmed an
initialization sequencing gap. In the preserved execution source,
`hmc_preparation.py::prepare_operational_windowed_mass_handoff` calls the
bootstrap and rejects its hard veto before calling the windowed-mass stage.
`hmc_kernel_tuning.py::_classify_bootstrap_screen` returns immediately for
nonfinite log acceptance; its ordinary epsilon repair applies only after that
screen passes. Thus the advertised repair loop cannot repair this failure.

`hmc_warmup.py::find_reasonable_epsilon` already distinguishes an invalid
initial/retained state or runner failure from an invalid rejected proposal. In
its initialization mode it can shrink the step after the latter. Existing
tests cover those distinctions, although they are not q20 evidence. The current
preparation reaches this logic only later through operational warmup, subject
to its screened-step handoff. This is a concrete reusable mechanism for a
proposed initialization repair; it is not already wired before the failing
bootstrap. The current main implementation still has the same bootstrap
nonfinite stop condition, so adopting concurrent module extraction alone would
not resolve it. No code or numerical setting was changed in this audit.

The next bounded diagnostic should check the initial value/score, preserve the
first failing proposal, and test smaller steps in the same affine coordinates
and declared trajectory. A passing initial search must then undergo a fresh
bootstrap and the ordinary preparation/tuning procedure. Old failures remain
rejected. Persistent failure at small steps, or disagreement between checked
reference and analytic scores, would instead require target/derivative or
integrator investigation. The `repair_nonfinite_proposal_screen` option seen
elsewhere applies to a later ladder and does not fix the earlier bootstrap
merely by being present in the public configuration.

There are two separate research uncertainties. First, eight maps at 512 updates
and one training root show continued descriptive learning, but posterior
whitening and global coverage remain untested. This classical failure used no
learned map. Second, compute remains expensive after compilation and chain
batching. The measured 6.0–6.9 seconds per four-chain transition at L3 extrapolate
to about 5.0–5.7 hours for 2000 warmup plus 1000 retained transitions. At L25,
the measured 37.0–42.3 seconds extrapolate to 30.8–35.2 hours. These arithmetic
projections assume unchanged steady timing and are neither convergence
predictions nor runtime bounds. They exclude tuning, compilation and repeats.
Combined with the remaining training quote, they establish a budget-planning
problem even if bootstrap initialization is repaired. Profiling, reuse of
qualified numerical work and scheduling independent GPU jobs need measured
benefits; three available GPUs alone do not divide each sequential chain's
runtime by three.
