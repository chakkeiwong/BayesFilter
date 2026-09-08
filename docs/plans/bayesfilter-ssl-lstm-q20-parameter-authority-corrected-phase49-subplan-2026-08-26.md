# Corrected Parameter-Authority Phase 49 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v3.1-independent-proposal-depth`  
Entry gate: Phase 48 report completed with branch `independent_mh_does_not_reduce_variability`  
Status: `IN_PROGRESS`  
Local cap: 7200 s

## Question and scope

Does increasing the independent-proposal MH depth from two to eight proposals
per nonterminal annealing stage reduce finite support variability under the
same q=20 target and defensive-mixture proposal? Phase 48 showed that two
proposals moved particles but did not reduce the declared spread vector. This
phase tests whether that was a depth limitation before changing the proposal
geometry or objective.

The target remains `V(theta)` for `theta in R^4`; the 60D UKF state is internal.
The exact Phase 48 two-step report is a frozen comparator. The identity arm is
rerun only to verify the unchanged initial/resampling/replay boundary; no
particles are pooled or used for training or selection.

## Mathematical kernel

Let `q(theta)` be the normalized defensive mixture and

`pi_beta(theta) proportional to q(theta)^(1-beta) exp(beta V(theta))`.

At each of eight independent steps, draw `theta' ~ q` and accept with

`log a = min(0, bridge(theta') - bridge(theta) + log q(theta) - log q(theta'))`,

where `bridge=(1-beta) log q + beta V`. After substituting the bridge
definitions, the unconstrained log-ratio identity is

`bridge' - bridge + log q - log q'`
`= beta * ((V' - log q') - (V - log q))`.

The proposal and target are both evaluated in the theta measure. No chart
Jacobian or internal-state density is added.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Is the v3.0 negative descriptive branch explained by insufficient independent-MH depth? |
| Mechanism | Eight independent proposals from the frozen defensive mixture at each nonterminal beta stage. |
| Comparator | Frozen v3.0 two-step report plus a paired identity replay from the same Phase 47 initial clouds. |
| Expected failure | Depth-8 spread remains variable, acceptance collapses at high beta, or a replay/hash mismatch appears. |
| Promotion criterion | Fixture, exact theta measure, target/status, pairing/replay, finite tensor, and artifact gates. |
| Promotion veto | Wrong depth/schema, missing `q` terms, candidate not sampled from `q`, invalid acceptance, stale comparator, seed/hash mismatch, nonfinite artifact, or overwritten root. |
| Continuation veto | Target/proposal unavailable, exact fixture contradiction, three unrepaired infrastructure failures, or remaining budget exhausted. |
| Repair trigger | Descriptive depth result; no proxy threshold is promoted. |
| Explanatory diagnostics | Acceptance by beta, candidate safe fraction, movement, displacement, roots, mode mass, ESS, and paired spreads. |
| Nonclaims | No finite-run convergence, IID Gaussian law, posterior correctness, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default promotion. |

## Evidence contract

The primary hard evidence is implementation and boundary validity, not a
spread threshold. The fixture must verify repeated beta-zero identity and
beta-one movement. The GPU runner must reproduce the Phase 47 and Phase 48
initial/identity hashes, use the unchanged target signature and theta measure,
and reject invalid candidates. The report must validate the frozen v3.0
two-step receipt and compare three paired depth-8 rows against it.

The primary descriptive branch is
`depth8_reduces_between_bank_variability_descriptive` only when all three
predeclared support metrics (theta mean[0], negative-mode mass, and maximum
covariance off-diagonal) are no larger than the v3.0 depth-2 values. Otherwise
the branch is `depth8_does_not_reduce_variability`. This branch is not a
statistical ranking because there are three replicates and no uncertainty
model.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| eight MH steps | new repair hypothesis motivated by nonzero v3.0 acceptance | still too shallow or excessive compute | per-stage acceptance/movement and wall time | hypothesis, not default |
| two-step comparator | frozen Phase 48 receipt | comparator protocol drift | schema/version/hash validation | reviewed frozen comparator |
| defensive epsilon `0.20` and safe std `2.0` | frozen Phase 48 proposal | proposal-support bias persists | exact proposal recomputation and candidate labels | hypothesis, unchanged |
| N=256 and three replicates | Phase 48 paired design | finite-replicate noise | raw per-replicate table; no ranking | diagnostic design |
| candidate seed offset `30000` | frozen Phase 48 seed contract | RNG coupling | manifest and hash replay | reviewed implementation choice |
| GPU/XLA target lane | repository target execution policy | allocator/compile failure | pre-import memory-growth verification | required execution choice |
| CPU-hidden report | diagnostic policy | mistaken GPU/default claim | explicit manifest | diagnostic only |

No numeric choice is promoted from this phase. The eight-step depth is a
bounded experiment setting.

## Skeptical pre-execution audit

The plan passes on 2026-08-26 after the v3.0 result review. The baseline is
the exact v3.0 two-step receipt, not a newly selected arm. The target and
proposal remain unchanged, so a depth result answers the stated repair
question. Spread, ESS, acceptance, and mode mass are explicitly explanatory;
only schema, measure, status, pairing, finite-value, and artifact gates can
pass or veto the run. The MathDevMCP algebra is recorded with definitions
substituted; an unconstrained-symbol audit is not reused as a proof.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| depth-8 starts from a different cloud | exact Phase 47/48 initial hashes | hard veto; discard interpretation |
| identity replay drifts | exact Phase 47/48 identity endpoint hashes | hard veto; inspect seed/protocol change |
| eight steps are accepted but all remain in one component | candidate labels, displacement, and beta-wise rows | explanatory; proposal-support repair remains open |
| spread shrinks by finite noise only | three paired rows and no uncertainty claim | retain role-limited result; plan uncertainty only if justified |
| runtime exceeds the local cap | measured wall time and remaining campaign ledger | stop before a new launch; do not relax gates |

## Artifacts and commands

The first boundary invocation on 2026-08-26 failed closed before computation
because the launch command used the nonexistent nested pilot paths
`phase47-invariant-mutation/attempt-02/pilot-0N`. The repository pilot
receipts are the top-level `phase47-invariant-mutation/pilot-0N/pilot.json`
files. This is recorded as a harness path repair; it does not change the
target, proposal, seeds, gates, or campaign budget. The corrected retry uses
those exact top-level receipt roots.

Fixture:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase49_fixture_2026_08_26.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase49-independent-proposal-depth/fixture
```

The q=20 boundary requires a visible trusted GPU with
`TF_FORCE_GPU_ALLOW_GROWTH=true`. The CPU-hidden report consumes the v3.1
fixture, the new boundary, and the frozen Phase 48 report and writes a unique
`report/` root. No HMC or NeuTra optimizer update is launched.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `depth8_reduces_between_bank_variability_descriptive` | depth is descriptively favorable under the paired finite screen | retain as a role-limited candidate; require uncertainty/downstream validation before any promotion |
| `depth8_does_not_reduce_variability` | depth increase does not repair the finite support spread in this scope | investigate proposal/support construction; keep whitening/HMC/LEDH closed |
| `depth8_hard_veto` | fixture, measure, target, pairing, or artifact contract failed | repair harness under a fresh root; no scientific interpretation |

No branch establishes a population limit, posterior law, IID Gaussian law,
mode-discovery theorem, or method superiority.
