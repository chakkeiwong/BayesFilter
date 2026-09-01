# Corrected Parameter-Authority Phase 51 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v3.3-mode-aware-proposal-geometry`  
Entry gate: Phase 50 report branch `support_broadened_does_not_reduce_variability`  
Status: `IN_PROGRESS`  
Local cap: `7200 s`

## Question and scope

Does a mode-aware candidate component reduce the finite between-bank
variability that remained after isotropic support broadening? The declared
target is unchanged: the batch-native q=20 SSL-LSTM target in `theta in R^4`.
The 60-dimensional UKF state remains internal to the target evaluation.

The q=20 target, defensive base density `q`, q-based annealing bridge,
initial clouds, resampling schedule, eight independent-MH proposals per
nonterminal stage, seeds, and particle count remain frozen. Only the candidate
component in the independent proposal changes.

## Candidate law and exact kernel

The geometry artifact contains two stationary representatives `m_minus` and
`m_plus`, with stable positive-definite local precision matrices `P_minus` and
`P_plus`. Let `C_j=P_j^{-1}` and use the reviewed hypothesis `kappa=2.0` to
inflate the local covariance. Define

`s_geom(theta) = 0.5 N(theta; m_minus, kappa^2 C_minus)
                 + 0.5 N(theta; m_plus, kappa^2 C_plus)`

and

`r_geom(theta) = (1-rho) q(theta) + rho s_geom(theta)`,

with `rho=0.50`. The values of `rho` and `kappa` are hypotheses, not defaults.
The q-based bridge is

`bridge_q(theta) = (1-beta) log q(theta) + beta V(theta)`.

For a candidate drawn from `r_geom`, the exact independent-MH log ratio is

`L(theta,theta') = bridge_q(theta') - bridge_q(theta)
                   + log r_geom(theta) - log r_geom(theta')`.

The implementation must evaluate normalized full-covariance Gaussian-mixture
densities in `theta_R4`; it must not attach a density to an ETPF/GenUT
transform or reinterpret the geometry covariance as a posterior covariance.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Does mode-aware proposal geometry, with exact non-symmetric correction, reduce finite bank variability? |
| Mechanism | Eight independent MH proposals per nonterminal stage from `r_geom`. |
| Comparator | Paired Phase 50 isotropic-support arm and frozen Phase 49 depth-eight arm, plus identity replay from the same Phase 47 clouds. |
| Expected failure | Curvature is too narrow, inflated geometry is too diffuse, or mode-aware mass does not improve overlap. |
| Promotion criterion | Fixture, target/status, exact density, theta-measure, pairing/replay, finite tensors, device, and artifact gates. |
| Promotion veto | q replaced by r in the bridge, missing current/candidate `log r_geom`, non-SPD covariance, stale geometry, invalid acceptance, hash mismatch, nonfinite artifact, or overwritten root. |
| Continuation veto | Target/proposal unavailable, exact fixture contradiction, three unrepaired infrastructure failures, platform block, or remaining campaign pool exhausted. |
| Repair trigger | A descriptive geometry result or a repairable harness failure. |
| Explanatory diagnostics | Component fraction, acceptance by beta, movement, log-ratio tails, roots, mode mass, ESS, covariance, and paired spreads. |
| Nonclaims | No posterior theorem, IID Gaussian law, exhaustive mode discovery, HMC readiness, canonical LEDH status, superiority, or default promotion. |

## Evidence contract

The hard evidence is implementation and boundary validity. A CPU-hidden fixture
must check finite normalized mode-mixture density evaluation, the exact q-base /
`r_geom` correction at beta zero and one, nonzero movement, and eight repeated
steps. The GPU boundary must reproduce all Phase 47 initial and identity hashes,
load passing Phase 49 and Phase 50 receipts, preserve q in the tempering
weights, evaluate `r_geom` at current and candidate rows, and reject invalid
candidate status. The report must preserve all three raw paired rows and make
no statistical ranking from them.

The descriptive primary branch is
`mode_aware_geometry_reduces_between_bank_variability_descriptive` only when
all three primary spreads (theta mean[0], negative-mode mass, and maximum
covariance off-diagonal) are no larger than the Phase 50 geometry-free support
arm. Otherwise the branch is
`mode_aware_geometry_does_not_reduce_variability`. This branch is not a
statistical claim.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
|---|---|---|---|---|
| stationary means | existing geometry artifact | stale/mode-biased representatives | geometry status/hash and fixture | reviewed warm start |
| `C_j=P_j^{-1}` | stable positive-definite local curvature records | local curvature under-covers tails | SPD/eigenvalue and finite-density checks | diagnostic component |
| `kappa=2.0` | bounded local-covariance inflation | too narrow or too diffuse | acceptance and log-ratio tails | hypothesis |
| `rho=0.50` | Phase 50 paired mixture weight | redundant or dominant geometry component | sampled component fraction and exact r recomputation | hypothesis |
| equal geometry weights | two representatives with comparable diagnostic log mass | true mode masses differ | raw component/mode rows; no mass claim | hypothesis |
| q=20, N=256, three replicates | frozen paired design | finite-replicate uncertainty | raw rows and no ranking | diagnostic design |
| GPU/XLA | repository execution policy | allocator/compile failure | pre-import memory-growth receipt | required execution choice |
| CPU-hidden fixture/report | diagnostic policy | mistaken production claim | explicit manifest | diagnostic only |

## Skeptical pre-execution audit

| Audit question | Finding | Control |
|---|---|---|
| Is the target changed? | No; q remains in the bridge and only the candidate law changes. | Separate q and `r_geom` fields; assert target signature. |
| Is `s_geom` normalized? | Yes if both inverse precisions are SPD and mixture weights sum to one. | Repository full-covariance routine and fixture checks. |
| Is the geometry covariance being overclaimed? | It is local curvature, not a posterior covariance. | Proposal-only labels and explicit nonclaims. |
| Is the comparator fair? | Initial clouds, resampling, depth, seeds, and target are frozen. | Phase 47 identity hash replay; Phase 49/50 receipts frozen. |
| Can descriptive spread become a promotion gate? | No. | Only hard implementation/measure/status/device/artifact gates pass. |
| Is the run within budget? | Prior lower-bound plus Phases 48-50 is `28960.22324898499 s`; approximately `35839.776751015015 s` remains before Phase 51. | Local cap `7200 s`; record actual wall time and remaining pool. |

The audit passes for this bounded candidate-method test. It does not authorize
NeuTra training, HMC, whitening, or canonical LEDH admission.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| geometry covariance is singular or stale | SPD/eigenvalue check and geometry hash | hard veto; repair only the artifact/interface |
| q is silently replaced by `r_geom` in tempering | stored q/r fields and source audit | hard veto; discard scientific interpretation |
| apparent improvement is finite noise | three paired raw rows and no ranking | retain descriptive result; require uncertainty-aware replication |
| mode component is sampled but rejected at high beta | component fractions, acceptance, displacement, log-ratio tails | explanatory repair trigger; do not relax gates |
| runtime exceeds cap | manifest wall time and budget ledger | stop before another launch; do not relax gates |

## Commands and artifacts

Fixture:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase51_fixture_2026_08_26.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase51-mode-aware-proposal-geometry/fixture
```

Boundary and report roots are unique below
`phase51-mode-aware-proposal-geometry/`. The boundary requires a visible
trusted GPU with `TF_FORCE_GPU_ALLOW_GROWTH=true`; the report is CPU-hidden.
No HMC, NeuTra optimizer update, or training-data selection is launched.

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `mode_aware_geometry_reduces_between_bank_variability_descriptive` | geometry is descriptively favorable under the finite paired screen | retain as role-limited candidate; require independent uncertainty-aware downstream validation |
| `mode_aware_geometry_does_not_reduce_variability` | mode-aware geometry does not repair the finite spread in this scope | write a final support/objective adjudication; keep whitening/HMC/LEDH closed |
| `mode_aware_geometry_hard_veto` | measure, target, density, replay, device, or artifact contract failed | repair harness under a fresh root; make no scientific interpretation |

No branch establishes a population limit, posterior law, IID Gaussian law,
mode-discovery theorem, method superiority, or default readiness.
