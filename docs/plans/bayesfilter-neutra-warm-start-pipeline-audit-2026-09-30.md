# Audit of the simpler-model NeuTra warm-start pipeline

## Question and scope

Owner question: beyond the initialization and post-training validation gaps,
what other problems affect the executed Gabrié/SMC → IAF → RKL study?

Audit the executed consumers, their sampler/trainer calls, calibration,
checkpoint selection, qualification, repair scheduling and saved evidence.
Compare behavior with the September 29 master and continuation plans. Preserve
all existing work and campaign artifacts. This is an engineering and evidence
audit, not a new training or sampler comparison. No runtime repair is included.

## Evidence contract and skeptical pre-audit

- A confirmed defect requires a checked consumer path and either a saved
  execution artifact or a bounded reproduction of the relevant control flow.
- Distinguish implementation defects, gaps in experimental design, intentional
  scope restrictions and unresolved scientific hypotheses. In particular,
  absence of a fitted Gaussian mixture does not itself make MCMC/SMC invalid.
- Use each candidate's preserved warm and RKL checkpoints as the relevant
  stage comparator. Do not infer teacher correctness from learned-map or
  downstream-HMC diagnostics, or infer architectural impossibility from a
  failed finite training run.
- Stochastic values already observed remain descriptive. No method ranking,
  production readiness, target failure or q20-transfer claim is permitted.
- Read source and saved JSON first. If needed, run one standard-library audit
  script that imports only the master controller, uses temporary fixtures and
  stubs launch boundaries, and summarizes saved artifacts. No TensorFlow,
  CUDA, training, HMC, network or package mutation is needed.
- Stop a proposed reproduction if it would initialize an accelerator, launch
  campaign work or mutate existing evidence. Report the untested boundary.
- Bound the script to 60 wall seconds, a convenience ceiling for a small
  engineering diagnostic rather than a scientific numerical choice.
- Preserve the script, source hashes, command, environment, wall/CPU time and
  results under `artifacts/neutra-warm-start-master-2026-09-29/` in a fresh
  `pipeline-audit-2026-09-30-r1` directory. Record findings in this note.

Pre-audit: the plan answers the owner's question without another campaign.
The main risks are treating a deliberate bounded experiment as a universal
pipeline, mistaking a proxy diagnostic for posterior evidence, and confusing
current source with the archived execution. Source hashes and saved manifests
will distinguish those cases. Existing dirty files are preserved. A smaller
control-flow reproduction is sufficient for controller defects; numerical
sampler hypotheses require later experiments and will be labeled accordingly.

## Active checkpoint

Audit complete. The campaign's useful results remain preserved, but completion
of its bounded matrix does not establish a dependable initialization/training
procedure. Three controller defects were reproduced without TensorFlow or a
worker launch. No runtime source, existing campaign evidence or budget ledger
was changed. Before another campaign, repair calibration failure handling,
job reuse, stage selection and independent validation; use small fixtures to
check those repairs. A new long experiment is not part of this audit.

## Checked evidence

The standard-library diagnostic ran as:

```bash
timeout 60 python3 docs/plans/artifacts/neutra-warm-start-master-2026-09-29/pipeline-audit-2026-09-30-r1/audit.py
```

It exited zero, took 0.0963 wall seconds and 0.090832 CPU core-seconds, and read
132 saved training results. It used no accelerator or numerical framework.
The fresh artifact directory contains `audit.py`, `audit.json` and
`manifest.json`. The manifest preserves the exact source hashes and command.
This routine engineering check is metered separately; campaign resources were
not launched or reallocated.

Current numerical core hashes match all 330 saved manifests containing the
core hash; the weighted trainer matches 329 manifests, with one earlier
manifest lacking that field. Other audited files have historical revisions,
as expected during campaign repair. Current `calibrate`, `prepare`,
`assessment`, `initial_walkers`, `GabrieProgram` and `discover_modes` ASTs were
also compared with the archived mixture calibration, ordinary-SMC seed-11
training and discovered-Gabrié seed-11 training sources: these functions
match. Current-controller reproductions below establish current defects;
they do not establish that every latent defect fired in the old campaign.

Of 132 results, 117 include a warm assessment. Twenty-six have a failed warm
coverage screen followed by an executed RKL phase. Twenty-nine report losing
coverage during RKL (27 mixture, one wiggle, one warped mixture). Seventy-one
record a lower warm validation forward loss at the last rung than the prior
rung. That last count is descriptive, not a statistical finding that 71 maps
were undertrained. The controller never reads the recorded improvement flag.

## Confirmed software defects

### 1. Failed MALA calibration still selects a kernel

`bayesfilter/testing/neutra_warm_start_campaign.py:317–330` screens for zero
invalid proposals and acceptance at least 0.5, but chooses the smallest step
when no candidate passes. The diagnostic executes the actual two selection
statements extracted from `calibrate` and demonstrates both an all-invalid
grid and an all-low-acceptance grid being selected anyway. This is a failed
calibration reported as a selection. All five saved target calibrations had
at least one passing candidate, so this latent defect is not an explanation
of their observed mixture failures. Repair: return an explicit calibration
failure and invoke a bounded, recorded calibration repair.

### 2. Resume skips completed jobs even after their inputs change

`scripts/run_neutra_warm_start_master.py:281–283` returns success on a matching
job name and completed/candidate-failed status, before checking configuration,
prepared data, source or parent identity. The reproduction changes batch size,
prepared-data path and mocked source identity; `run_one` returns success
without comparing sources or launching updated work. Hashing the tests' job
name later in the master does not refresh prepare/calibrate/train job names.
This can silently reuse stale calibration or results after a repair. No claim
is made that stale reuse caused a specific numerical result in this campaign.
Repair: compare relevant settings, source and input identities before reuse,
then create a new versioned attempt when they differ. Ordinary hashes suffice.

### 3. Resume erases the accumulated deferred-work list

`scripts/run_neutra_warm_start_master.py:511` unconditionally resets
`deferred_jobs`. A checks-only resume fixture erases a prior deferred repair
without attempting it. The real terminal review reconstructs 43 unrepaired
failed base candidates; the last state lists 29. The five-repairs-per-seed cap
was intentional and is not itself a bug. Erasing older pending work from the
current summary is a reporting/recovery defect. Repair: retain unresolved work
across filtered invocations and distinguish exhausted local repair allowances
from exhausted total campaign budget.

## Training and experimental-design gaps

### 4. RKL is unconditional, and only its final map can be qualified

The warm phase always flows into RKL in
`bayesfilter/testing/neutra_warm_start_campaign.py:416–461`. The master
decides viability from final coverage at
`scripts/run_neutra_warm_start_master.py:589–600`, and qualification loads
only `rkl-frozen.json` at
`bayesfilter/testing/neutra_warm_start_qualification.py:60`.
Warm checkpoints are preserved but not eligible through that consumer.

Running RKL after a poor initializer is a legitimate controlled diagnostic of
the requested refinement hypothesis; it is not intrinsically a sampler error.
It is insufficient for a master whose objective is to obtain any useful map:
there is no decision to qualify a promising warm map, choose an earlier RKL
checkpoint, or stop a refinement that is destroying coverage. The 29 observed
coverage losses show the practical consequence. A passing warm coverage screen
does not establish that its map would pass HMC; that question is untested.

### 5. Coarse region coverage misses the observed shape defect

`assessment` at `neutra_warm_start_campaign.py:231–255` checks a few region
averages using a five-percentage-point margin plus four estimated standard
errors. It records forward KL and a 1000-point probe, but the training master
uses coverage and finiteness to decide qualification eligibility. No check
requires the initializer to reproduce within-mode shape or the low density
between peaks. The September 30 saved-map diagnostic demonstrated broad,
nearly Gaussian oracle fits passing coverage. Their forward KL was about
0.946–0.948, close to the moment-matched Gaussian projection. A geometry probe
sampled through the map can also miss a region the map rarely visits.

This is an insufficient screen for the desired initializer, not a reason to
turn a small score residual into a universal correctness criterion. Add
target-specific shape and physical-space checks, calibrated for a coarse
warm-start objective; final inference still needs downstream checks.

### 6. Calibration does not demonstrate useful learning, or transfer across objectives

`calibrate` selects the lowest finite forward loss after one 256-update pilot
for each width/LR pair. It has no requirement to exceed a simple Gaussian-fit
baseline. All four unwarped-mixture candidates were already near the same
Gaussian plateau (0.9473–0.9562), yet the procedure issued selected settings.
These are exploratory nominations, not evidence of a useful nonlinear fit.

The same width/LR nominations are consumed by different training objectives.
AFT/CRAFT additionally inherit the clip measured for exact-example IAF
forward fitting (`train:397–400`); their annealed transport gradients are a
different objective under different distributions. The RKL/Gabrié ordinary
training path does recalibrate clipping, so this finding must not be applied
indiscriminately to all clipping. These transfers are untested hypotheses,
not established causes of failure. There was no recorded invalid proposal in
the inspected completed training or teacher reports.

### 7. Local-kernel calibration does not test the executed bridge distributions

The MALA pilot runs only four steps near supplied target representatives at
beta=1 and nominates by displacement among accepted settings
(`calibrate:317–330`). Ordinary SMC applies that step at changing bridge
distributions; AFT/CRAFT do likewise. The experiment did not calibrate movement
in each relevant region or along the bridge, nor demonstrate cross-mode
mixing. Four of five targets selected the largest tried step, so the pilot
also did not bracket its preferred scale. These observations do not prove the
steps mathematically invalid: Metropolis correction remains separate from
finite-run efficiency and accuracy.

### 8. Recorded progress does not drive continuation, and repairs are coarse

`continuing_improvement_at_cap` is emitted at `train:471–475` but neither
master nor queue consumes it. The normal branch always finishes its fixed
rungs. The manually added warm continuation restores optimizer state for
fixed teachers; it is not an automatic quality-dependent policy.

For non-collapse failures, `worker` at
`scripts/run_neutra_warm_start_master.py:201–216` doubles batch size, walker
count, update rungs, particle counts and flow-training work together, where
applicable. This confounds several explanations. Collapse receives the more
specific smaller-LR/doubled-batch retry, but that cannot by itself distinguish
an unlearned bimodal warm fit from a refinement-objective problem. Repairs
need stage-specific diagnosis. Noisy loss improvements should trigger that
diagnosis, not an unlimited train-until-decrease loop.

### 9. Confirmation data is reused after informing repairs

Warm and RKL assessments both read `confirmation.tensor`
(`train:347,454`). Their coverage outcomes drive repair choice. HMC's
reference check then reads that same tensor
(`neutra_warm_start_qualification.py:100`). Repaired candidates also reuse the
same bank. The bank remains independent of the raw gradient minibatches, but
it is no longer an untouched final test after these decisions. It supports
adaptive diagnostic development; a final confirmatory claim needs a fresh
bank after the procedure is fixed. Reuse does not invalidate the analytic
target or turn every reference comparison into a wrong computation.

### 10. Teacher validation and restart state are incomplete

The Gabrié loop has no separately frozen post-training sampling assessment.
All 31 inspected Gabrié-associated result directories lack saved walker/sample
tensor files and have no teacher report; some are RKL-only repairs of saved
warm maps. The sampling loop's current walkers are returned in memory but are
not checkpointed with the optimizer. Its exact adaptive training history
cannot be reconstructed from map checkpoints. This also prevents a faithful
automatic continuation of that sampler state.

Ordinary SMC saves final particles and weights, stage ESS, unique-root counts,
normalizer estimates and descriptive regional errors. The actual root array
is omitted by the caller (`train:403–412`). AFT/CRAFT stage transport states
are kept in memory and are not saved as a reusable sequence by their return
paths. These are reproducibility and validation limitations, not proof that
the returned finite clouds have the wrong distribution. The needed next
evidence is independent teacher replications and a frozen Gabrié sampler
assessment, with uncertainty appropriate to correlated/weighted samples.

### 11. Benchmark-only information prevents an end-to-end discovery claim

Discovered representatives are consumed only by `gabrie_discovered`; the SMC
branches start from the broad Student proposal. The master runs the discovered
arm only for Gaussian and the two mixtures. Calibration uses exact/reference
examples, and downstream HMC starts from supplied representatives, including
when it evaluates a discovery-trained map. `prepare` records a known-mode
count mismatch but does not make it a preparation failure.

These choices can isolate training on known fixtures, but success cannot
establish autonomous initialization on an unknown posterior. There is no
executed mode-finding → fitted mixture → SMC composition. A fitted mixture
is optional mathematically; its absence alone is not an MCMC/SMC bug.

### 12. Author-operation parity is narrower than full-procedure equivalence

The checked Gabrié fixture executes the author's global-MH/local-MALA moves.
The pinned author controller at
`.localresources/flonaco-author-20260929/upstream/flonaco/training.py:222–234`
also retries samples and terminates on excessive loss jumps. The campaign
training block does not implement that controller. The source-adoption note
describes the importance of this difference; the master later explicitly
labels the experiment an adaptation of selected operations, not a reproduction
of every author-program option. No evidence here establishes whether the
omitted retry would help this target. Likewise, AFT/CRAFT primitive and ordering
tests are not full upstream training-program equivalence. Any stronger claim
would exceed the tested scope.

## Decision and interpretation

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Repair the three reproduced controller defects | Exact selection/resume fixtures reproduce the behavior | No external or numerical work was launched | Which latent defects affected historical attempts | Focused fixes and regressions before reuse | These defects caused every failed fit |
| Treat the current master as a bounded diagnostic study | Saved results establish incomplete stage decisions and validation | No dependable-training promotion | Whether preserved warm maps or a repaired protocol work | Make teacher, fit, refinement and inference assessments separate and actionable | NeuTra, Gabrié or SMC is ineffective in general |
| Preserve candidate posterior evidence | Existing HMC checks concern those frozen maps and estimands | No automatic q20/default promotion | Adaptive candidate selection and finite replication | Fresh independent assessment after the training procedure is fixed | Teacher accuracy or full-density equivalence from an HMC pass |

| Inference status | Finding |
|---|---|
| Hard veto screen | Existing failed candidate screens remain failed; deterministic controller reproductions need repair |
| Statistically supported ranking | None established by this audit |
| Descriptive-only differences | Coverage-loss counts, terminal loss decreases and saved calibration metrics |
| Default readiness | A reliable multimodal training procedure is not established |
| Next evidence needed | Focused engineering regressions; separate teacher validation; stage-aware candidate assessment with fresh final reference data |

Post-audit red team: the strongest alternative explanation is that several
listed gaps were deliberate simplifications for a bounded independent-arm
experiment. That is valid for those limited questions and is explicitly
distinguished above from software defects. It does not establish the complete
procedure the owner expected. No new neural gradients, Hessians, flow
expressivity tests or sampler-accuracy experiments ran, so the exact cause of
the observed near-Gaussian optimization plateau remains unresolved. Additional
saved evidence of independent teacher validation or warm-map qualification
would change the corresponding finding; none is present in the inspected
consumer paths and campaign artifacts.
