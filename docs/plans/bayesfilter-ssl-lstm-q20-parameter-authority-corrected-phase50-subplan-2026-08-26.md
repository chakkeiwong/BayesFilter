# Corrected Parameter-Authority Phase 50 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v3.2-defensive-proposal-support`  
Entry gate: Phase 49 report completed with branch `depth8_does_not_reduce_variability`  
Status: `IN_PROGRESS`  
Local cap: `7200 s`

## Question and scope

Does a broader independent-proposal law reduce the finite support/mode
variability that remained after eight proposals from the fixed defensive law?
The q=20 target, the annealing base density, the theta measure, the initial
clouds, the resampling schedule, and the eight-step mutation depth remain
unchanged. Only the candidate law used by the independent MH mutation changes.

The base bridge is

`pi_beta(theta) proportional to q(theta)^(1-beta) exp(beta V(theta))`,

where `q` is the frozen defensive mixture and `V` is the q=20 target log
density in `theta in R^4`. Candidates are drawn from

`r(theta) = (1-rho) q(theta) + rho s(theta)`,

with `s(theta)=Normal(center, tau^2 I)`, `rho=0.50`, and `tau=4.0`. These are
bounded hypotheses, not promoted defaults. The internal 60D UKF state remains
inside `V` and is never a particle coordinate.

## Mathematical kernel

For an independent proposal from `r`, the exact MH log ratio for the stated
base bridge is

`L(theta,theta') = bridge_q(theta') - bridge_q(theta)
                   + log r(theta) - log r(theta')`.

The q terms in the bridge are retained for the tempered target; replacing them
with `r` would silently change the SMC bridge. The proposal correction uses `r`
at both current and candidate rows because `r` is not symmetric. At `beta=0`,
the target is `q`, so `L = log q(theta')-log q(theta)+log r(theta)-log r(theta')`,
which is generally nonzero but has the correct invariant-target form.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Does proposal-support broadening, rather than more depth, reduce finite bank variability? |
| Mechanism | Eight independent MH proposals per nonterminal stage, candidates from `r=(1-rho)q+rho s`. |
| Comparator | Frozen Phase 49 depth-eight report plus a paired identity replay from the same Phase 47 clouds. |
| Expected failure | Non-symmetric ratio is wrong, broad candidates have low acceptance, or primary mode/ESS variability persists. |
| Promotion criterion | Fixture, exact q-base/r-proposal algebra, target/status, pairing/replay, finite tensors, and artifact gates. |
| Promotion veto | Candidate sampled/scored under different laws, q bridge replaced by r, missing current/candidate `log r`, invalid acceptance, stale comparator, hash mismatch, nonfinite artifact, or overwritten root. |
| Continuation veto | Target/proposal unavailable, exact fixture contradiction, three unrepaired infrastructure failures, or remaining budget exhausted. |
| Repair trigger | Descriptive support result; no spread threshold is promoted. |
| Explanatory diagnostics | Acceptance by beta, candidate component, movement, displacement, roots, mode mass, ESS, and paired spreads. |
| Nonclaims | No finite-run convergence, IID Gaussian law, posterior correctness, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default promotion. |

## Evidence contract

The hard evidence is implementation and boundary validity, not a proxy spread
threshold. The CPU-hidden fixture must verify the exact non-symmetric q-base /
r-proposal ratio over eight repeated steps, finite states, and nonzero movement.
The GPU boundary must reproduce all Phase 47 initial and identity hashes, load a
passing frozen Phase 49 report, preserve the q target/proposal values for
tempering, evaluate `r` for current and candidate rows, and reject invalid
candidate status. The report compares three paired support-arm rows against the
frozen Phase 49 depth-eight rows.

The descriptive branch is
`support_broadened_reduces_between_bank_variability_descriptive` only when all
three predeclared primary spreads (theta mean[0], negative-mode mass, and
maximum covariance off-diagonal) are no larger than the frozen Phase 49 arm.
Otherwise it is `support_broadened_does_not_reduce_variability`. Neither branch
is a statistical ranking: there are three replicates and no uncertainty model.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|
| `rho=0.50` | new bounded support-overlap hypothesis | too much broad-tail mass changes finite behavior | component fraction and exact r recomputation | hypothesis, not default |
| `tau=4.0` | new scale chosen to exceed the frozen safe scale 2.0 | overly diffuse candidates and low acceptance | beta-wise acceptance and log-ratio tails | hypothesis, not default |
| eight MH steps | frozen Phase 49 depth arm | compute cost or residual mixing remains | per-stage movement and wall time | comparator-bound hypothesis |
| q=20 target and q base bridge | frozen Phase 49 contract | accidental objective/measure change | target signature, q/r field audit | required unchanged contract |
| N=256 and three replicates | Phase 49 paired design | finite-replicate noise | raw rows and no ranking | diagnostic design |
| candidate seed offset `30000` | frozen depth protocol | RNG coupling | manifest and replay hashes | reviewed frozen choice |
| GPU/XLA boundary | repository execution policy | allocator/compile failure | pre-import memory-growth receipt | required execution choice |
| CPU-hidden report | diagnostic policy | mistaken GPU/default claim | explicit manifest | diagnostic only |

No numeric choice in this phase is a new production default. The broad mixture
parameters are deliberately hypotheses whose failure should trigger another
repair or closure of this proposal-support variant.

## Skeptical pre-execution audit

The plan passes on 2026-08-26 after the Phase 49 result review. The comparator
is the frozen Phase 49 depth-eight receipt, not a selected replicate. The base
tempered target remains q-based, while only the independent candidate law is
changed and corrected by current/candidate `log r`; therefore the experiment
answers the stated support-overlap question without changing the target. The
fixture and boundary schemas distinguish q from r. Acceptance, ESS, mode mass,
and spread remain explanatory; only algebra, measure, status, pairing, finite,
device, and artifact gates can pass or veto the run.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| r is used in the tempering weights | stored q/r fields and source audit | hard veto; discard interpretation |
| current state is scored with q instead of r | current/candidate r receipts and fixture | hard veto; repair ratio |
| broad candidates appear to improve spread by finite noise | three paired rows and no uncertainty claim | retain role-limited result only |
| broad tail never reaches the target modes | component labels, beta-wise movement, mode rows | explanatory; support geometry remains unresolved |
| runtime exceeds cap | measured manifest and campaign ledger | stop before another launch; do not relax gates |

## Artifacts and commands

Fixture:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase50_fixture_2026_08_26.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/fixture
```

The q=20 boundary requires a visible trusted GPU with
`TF_FORCE_GPU_ALLOW_GROWTH=true`; it writes a unique `q20-paired/` root. The
CPU-hidden report consumes the fixture, boundary, and frozen Phase 49 report.
No HMC or NeuTra optimizer update is launched.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `support_broadened_reduces_between_bank_variability_descriptive` | support broadening is descriptively favorable under the paired finite screen | retain as role-limited candidate; require uncertainty-aware downstream validation |
| `support_broadened_does_not_reduce_variability` | broadening does not repair the finite support spread in this scope | investigate proposal geometry/objective separately; keep whitening/HMC/LEDH closed |
| `support_broadened_hard_veto` | ratio, measure, target, pairing, or artifact contract failed | repair harness under a fresh root; no scientific interpretation |

No branch establishes a population limit, posterior law, IID Gaussian law,
mode-discovery theorem, method superiority, or default readiness.

