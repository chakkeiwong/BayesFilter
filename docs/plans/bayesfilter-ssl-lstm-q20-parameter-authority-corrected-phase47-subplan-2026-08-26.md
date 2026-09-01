# Corrected Parameter-Authority Phase 47 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v2.9-invariant-mutation-diagnostic`  
Entry gate: Phase 46 report completed with branch `n512_c_outside_two_bank_scalar_envelope`  
Status: `IN_PROGRESS`  
Local cap: 5400 s

## Question and scope

Does an explicit invariant theta-space Metropolis-Hastings rejuvenation kernel
reduce finite support/mode variability relative to the existing identity
mutation, under the same q=20 target, proposal law, annealing schedule, and
initial particles?

The particle coordinate remains `theta in R^4`. The UKF state is internal. The
identity arm and MH arm share each replicate's initial M0 proposal cloud and
all resampling seeds. The only changed mechanism is post-stage mutation. This
is a finite support diagnostic; it is not a posterior, whitening, HMC, or
canonical LEDH experiment.

## Mathematical kernel

At annealing level `beta`, define the unnormalized bridge

`pi_beta(theta) = q(theta)^(1-beta) exp(beta V(theta))`,

where `q` is the declared M0 proposal density in theta measure and `V` is the
q=20 target log density. For a symmetric random-walk proposal
`theta' = theta + sigma * xi`, `xi ~ N(0,I)`, the acceptance probability is

`a(theta,theta') = min(1, exp(log pi_beta(theta') - log pi_beta(theta)))`.

Invalid target/status proposals are rejected. Symmetry cancels the proposal
kernel terms, and the usual Metropolis detailed-balance identity makes this
kernel invariant for `pi_beta` when the target values are valid. The identity
kernel is also invariant but has zero movement. The run records this formula,
the endpoint status, and the accepted-move indicator for every stage.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Does explicit invariant rejuvenation reduce finite support variability? |
| Mechanism | Symmetric theta-space random-walk MH after each fixed beta stage. |
| Comparator | Identity mutation with identical initial clouds and resampling seeds. |
| Expected failure | MH acceptance is near zero, target invalidity rises, or support remains variable. |
| Promotion criterion | Analytic fixture invariance, exact target/measure/status gates, paired-data identity, finite artifacts. |
| Promotion veto | Asymmetric/unrecorded ratio, wrong target measure, invalid endpoint accepted, seed/cloud mismatch, nonfinite tensor, or overwrite. |
| Continuation veto | Target/status unavailable, exact fixture contradiction, three unrepaired infrastructure failures, or budget exhaustion. |
| Repair trigger | Descriptive support/mode/transport result; no proxy threshold is promoted. |
| Explanatory diagnostics | Acceptance, move fraction, ESS, roots, mode mass, support ranges, and frozen-state transport moments. |
| Nonclaims | No finite-run convergence, IID Gaussian theorem, posterior correctness, exhaustive mode discovery, HMC, LEDH, superiority, or default promotion. |

## Evidence contract

The exact comparator is identity mutation under the same initial cloud,
schedule, resampling, target, and proposal. The primary hard gates are the
analytic standard-normal MH fixture, paired initial-cloud/hash identity, valid
theta/status evaluations, finite acceptance calculations, and unique manifests.
Acceptance and residual differences are explanatory only. A passing MH arm is
not an admission to HMC or a proof of posterior correctness.

Three independent replicates use N=256 particles and the fixed M0 proposal;
the Phase-28 pilot root seeds are `(20260826,4706)`, `(20260826,4707)`, and
`(20260826,4708)` (the M0 arm seed is the root seed plus the runner's fixed
`+100` offset). The proposal scale is `sigma=0.35`, two MH steps per beta
stage, and the schedule is `[0,.2,.4,.6,.8,1]`. These are reviewed diagnostic
hypotheses, not defaults.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| symmetric random walk | standard MH construction | poor scale or low movement | fixture plus acceptance/move receipt | hypothesis |
| `sigma=0.35` | bounded q=20 theta diagnostic scale | scale-dependent result | report raw acceptance and displacement | hypothesis |
| two MH steps/stage | bounded campaign budget | insufficient mixing | per-stage move/ESS/root diagnostics | hypothesis |
| N=256 and three seeds | matched prior support banks and budget | too little replication | paired replicate table; no ranking | diagnostic design |
| GPU/XLA target evaluation | repository GPU default | allocation/runtime failure | pre-import memory-growth manifest | reviewed execution choice |

## Skeptical pre-execution audit

The plan passes on 2026-08-26. The acceptance ratio uses the tempered target
and proposal in the same theta measure; no chart Jacobian or empirical-cloud
density is introduced. Identity and MH arms share initial tensors and
resampling seeds, so a difference is attributable to the mutation mechanism
within this finite diagnostic. The analytic fixture catches a sign or ratio
error before q=20 execution. Acceptance is explicitly not a promotion metric.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| MH appears better due to different initial particles | exact initial tensor hashes and paired seeds | hard veto; discard interpretation |
| invalid proposals are accepted through NaN handling | endpoint status gate and finite log-alpha | hard veto; preserve failure |
| high acceptance means no movement | accepted count plus displacement norm | explanatory diagnosis; retune only in a new scope |
| mutation changes the target measure | analytic fixture and explicit `log pi_beta` ledger | hard veto |
| MH lowers one residual but not support variability | all replicate support and transport rows | retain as candidate repair only |

## Artifacts and commands

Fixture (CPU-only, before q=20):

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase47_fixture_2026_08_26.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase47-invariant-mutation/fixture
```

The q=20 paired runner requires a visible GPU and
`TF_FORCE_GPU_ALLOW_GROWTH=true`; it writes a unique `q20-paired/` root. A
CPU-hidden report consumes the structured receipt and writes `report/`.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `mh_rejuvenation_reduces_between_bank_variability_descriptive` | MH moves and reduces raw support spread across paired replicates | retain as a role-limited repair; plan independent downstream validation |
| `mh_rejuvenation_does_not_reduce_variability` | identity-vs-MH difference is not descriptively favorable | investigate proposal/objective support separately; keep gates closed |
| `mh_kernel_hard_veto` | invariant-kernel or data-boundary failure | repair harness; no scientific interpretation |

No branch is a statistical ranking or a whitening result.

## Attempt-01 repair record

The first q=20 execution wrote a complete per-replicate artifact but returned
`PHASE47_MUTATION_BOUNDARY_FAIL`. The failure was in the harness gate map:
`terminal_resampling: false` was recorded as metadata and simultaneously
included in `all(gates.values())`, so every otherwise-valid arm was rejected
by construction. The per-arm shape, status, finite-tensor, paired-cloud, and
paired-seed checks were true; no scientific interpretation is assigned to that
attempt. The repair removes the metadata field from the pass conjunction while
retaining `terminal_resampling: false` in the result manifest. The unchanged
experiment is rerun under a fresh `attempt-02/` output root.
