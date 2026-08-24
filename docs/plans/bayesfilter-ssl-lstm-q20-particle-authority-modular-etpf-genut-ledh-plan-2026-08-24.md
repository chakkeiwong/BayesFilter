# SSL-LSTM q=20 particle authority and modular ETPF/GenUT/LEDH plan

Date: 2026-08-24  
Amended: 2026-08-25  
Status: `AMENDED_AFTER_FABLE_REVIEW_AWAITING_NEW_GROK_REPLY`  
Scope: q=20 SSL-LSTM proposal generation, filtering, replay, and NeuTra training support

## Executive decision

The next research direction is a hybrid, not a wholesale replacement by one
moment-transform algorithm.

The wholesale change is to the **particle-data authority**: the current fixed
six-bank normalized replay is no longer an authority for posterior mass or
unbiased training claims. The replacement authority is a fresh, mode-aware
tempered SMC/SMC-U route with explicit proposal densities, unnormalized mass,
invariant mutation kernels, and defensive support. The proof-bearing SMC-U
definition is the conditional unnormalized-block contract in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md:318-335`;
it is a candidate obligation, not an assumption that the current runner already
satisfies.

Here, "mode-aware" means that the protocol records mode-coverage diagnostics
and tests known-mode fixtures; it is not a finite-run mode-discovery guarantee.

The following are modular candidate mechanisms, each with its own contract:

- Acevedo--de Wiljes--Reich second-order ETPF for finite-cloud covariance
  correction;
- GenUT for local sigma-point moment propagation and proposal preconditioning;
- invertible LEDH-PFPF for a density-corrected proposal;
- a full second-order ET-PF as an explicitly approximate-filter comparator.

No candidate is promoted to the canonical authority merely because it improves
whitening, ESS, validation loss, or covariance residuals. Those are explanatory
or nomination diagnostics until target-level and downstream checks pass.

This plan is documentary and review-ready. It does not authorize runtime code,
GPU work, replay replacement, NeuTra training, or HMC before the two bounded
reviews and a subsequent implementation-phase plan.

## Evidence carried forward

The current evidence motivates an authority change but does not rank the new
arms:

- The q=20 mode-failure result found only `3/100000` learned-flow draws in the
  negative half-space and a latent separation of about `23.7`; this is direct
  proposal-coverage evidence, not an exact mode-mass estimate.
- The 2026-08-24 replay A/B screen completed its engineering gates, but its
  maximum covariance-minus-identity residual was `11.6264`--`24.8663`, its
  estimator was finite-block self-normalized, and no HMC gate ran.
- The prior dual-cap/GenUT note and both independent reviews classify moment
  correction as a conditional representation/proposal component, not a density
  or mode-discovery theorem.

Source artifacts:

- `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-result-2026-08-10.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-result-2026-08-24.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-review-reply-2026-08-24.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-adjudication-2026-08-24.md`

## Source and literature ledger

The following local copies are the anchors for implementation decisions. The
plan distinguishes what each source states from project extensions.

| Source | Local anchor | Use in this plan | Boundary |
|---|---|---|---|
| Acevedo, de Wiljes, Reich (2017), *Second-order accurate ensemble transform particle filters* | `.localresources/papers/ledh_replay_solution_20260824/acevedo-dewiljes-reich-2017-second-order-etpf.txt:59-64,170-256,268-320,355-391,639-646` | Defines first/second-order LETF moment constraints, the correction/Riccati route, Sinkhorn implementation, and the finite-sample Bayes-law warning | Supports finite weighted-cloud moment matching; does not support IID density samples or exact q=20 mode masses |
| Ebeigbe et al. (2021), GenUT | `.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt:24-26,114-141,165-181` | Supports selected-moment sigma-point quadrature and constrained local propagation | Does not identify a density or global multimodal sample bank |
| Li and Coates (2017), invertible particle flow | `.localresources/papers/ledh_replay_solution_20260824/li-coates-2017-particle-filtering-invertible-flow.txt:140-179,267-327,390-457` | Supports an invertible LEDH PF-PF proposal with determinant-corrected weights under step/regularity conditions | A later non-invertible reset or unmodeled stochastic kernel is outside the source identity |
| Cornuet et al. (2009), AMIS | `.localresources/papers/ledh_replay_solution_20260824/cornuet-et-al-2009-amis.txt` | Supports deterministic-mixture denominators for frozen proposal histories | Adaptive schedules require their own conditioning and replay proof |
| Hesterberg (1995), defensive mixtures | `.localresources/papers/ledh_replay_solution_20260824/hesterberg-1995-defensive-mixture.txt` | Supports full-support safety components conditional on an integrability/second-moment assumption | Support alone does not guarantee finite variance |
| Existing replay mathematics | `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md:239-329,741-759` | Defines the known-density and SMC-U contracts and the limit of normalized replay | The current six-bank artifact is not silently upgraded to SMC-U |

Acevedo paper PDF SHA-256:
`3e729ca967486163dd0cbdfde90baaedcc6ef76c1df111bad4550b831ebc80e1`.

## Repairs carried forward from the prior reviews

The previous Fable review identified three evidence gaps that are promoted to
explicit early artifacts here:

1. **Per-proposal density identity:** every M0/M3 row must expose the pre-flow
   proposal, transition, observation, covariance state, pseudo-time factors,
   determinant product, and support declaration, followed by an affine
   known-map identity test.
2. **Defensive-tail second moment:** a concrete safety density must be checked
   against the q=20 forward score class; support alone is not enough.
3. **Replay metadata parity:** one stored block must have all historical log
   densities recomputed from retained metadata before it can enter any
   deterministic-mixture replay calculation.

The prior Fable minor/editorial wording suggestions concern the older
mathematical note and do not alter this plan's authority boundary. If that note
is edited later, its MathDevMCP audit and checksums must be refreshed.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does replacing invalid fixed replay with a fresh SMC authority, then adding covariance, local-moment, or flow mechanisms one at a time, produce a target-faithful and mode-aware training source for q=20 NeuTra? |
| Primary scientific target | The original q=20 SSL-LSTM posterior density and its downstream posterior-predictive law; no altered target is admitted silently. |
| Authority hypothesis | Fresh tempered SMC/SMC-U with calibration-selected, claim-run-frozen tempering, defensive support, and invariant mutation can provide a reproducible weighted particle measure with auditable mass bookkeeping. |
| ETPF hypothesis | Second-order ETPF can reduce finite-ensemble covariance underdispersion relative to the weighted empirical cloud, but may alter higher-order geometry or mode structure. |
| GenUT hypothesis | GenUT can improve local nonlinear moment propagation or proposal preconditioning, but a sigma-point cloud is not an IID posterior sample bank. |
| Flow hypothesis | An invertible LEDH-PFPF proposal can reduce incremental weight variance while retaining a valid change-of-variables correction when every density term is evaluated. |
| Whole-ETPF hypothesis | A full second-order ET-PF may be useful as an approximate filtering comparator, but it is not an exact posterior authority without an additional theorem and evidence. |
| Expected failure modes | Missing modes; bridge points between sign-separated modes; finite-transform Bayes-law violation; stale or self-normalized replay bias; non-invertible reset after a flow; online tempering/protocol drift; poor tail second moments; mutation non-invariance; target/status failures; and under-budgeted GPU execution. |
| Promotion criterion | A candidate must pass its own mathematical contract, a known-density/reference fixture, independent target-level diagnostics, and downstream validation appropriate to its claimed role. |
| Promotion veto | Moment matching presented as density matching; deterministic transformed rows presented as IID; a flow determinant used after an unaccounted reset; normalized replay presented as unbiased; an untested adaptive claim-run schedule presented as an SMC-U authority; missing support; non-finite/status-invalid rows; or an approximate filter presented as exact. |
| Continuation veto | The fresh authority cannot define a common-support target/proposal measure, or a candidate contradicts its source identity on an exact fixture. A poor covariance or ESS result alone is a repair trigger, not a research-direction veto. |
| Repair trigger | Failed affine density identity, failed SMC-U mass check, failed mutation invariance, schedule/hash mismatch, bridge inflation, mode-mass instability, proposal-tail failure, replay metadata mismatch, or a resource failure that invalidates the artifact. |
| Explanatory diagnostics | Whitening moments, covariance residuals, cap activity, ESS, log-weight tails, mode occupancy, bridge fraction, loss, gradients, and runtime. |
| Must not be concluded | No exhaustive finite-run mode discovery, exact nonlinear filtering theorem for q=20, IID Gaussian whitening from moments, optimizer convergence, statistical superiority from a few seeds, HMC convergence, posterior correctness, or default readiness. |

## Mathematical role boundaries

Let the forecast cloud be `X=(x_1,...,x_N)` with normalized importance
weights `w`, and let `Y=XD` have columns `y_j`. Define the finite weighted
measure by

```text
pi_hat_X = sum_i w_i delta[x_i]
bar_x_w = sum_i w_i x_i
nu_hat_D = (1/N) sum_j delta[y_j]
```

For the column convention `Y=XD`, the first- and second-order ETPF
constraints are

```text
D^T 1 = 1,                    D 1 = N w,
bar_y_D := (1/N) sum_j y_j = bar_x_w,
(1/N) sum_j (y_j - bar_y_D)(y_j - bar_y_D)^T
  = sum_i w_i (x_i - bar_x_w)(x_i - bar_x_w)^T.
```

These are identities for the finite weighted cloud, not density identities.
In general `nu_hat_D != pi_hat_X` and neither is necessarily the true
posterior. If `D` is nonnegative,
transformed rows are convex combinations;
the second-order correction can allow negative entries and leave the forecast
range. The source reports the stronger generic incompatibility
`D1+ intersect D2 = empty` for nonnegative first-order and exact second-order
classes; this is why bridge/support diagnostics are mandatory. Therefore the
transform is tagged `finite_moment_transform`, not
`posterior_density_sample`.

GenUT is tagged `sigma_point_quadrature`. For q=20, one local (2d+1=41)-point
rule is a local moment construction, not a representation of separated global
modes. It may propose local points or tune a covariance, but it may not silently
replace the authority bank.

For an invertible flow `T_phi`, the proposal contract is

```text
q_phi(x) = q_0(T_phi^{-1}(x)) * |det D T_phi^{-1}(x)|
```

with the PF-PF weight containing the actual pre-flow proposal, post-flow
transition, observation, and matching determinant. A later Contract-E/ETPF/
GenUT/reset operation is outside this determinant unless its joint pushforward
density is separately defined. A stochastic transform may instead be admitted
as a mutation/proposal kernel only when its transition density is evaluable.

The authority route must carry an unnormalized mass estimate and all proposal,
transition, observation, ancestry, and mutation metadata. A normalized terminal
cloud alone is not an SMC-U estimator and cannot certify replay unbiasedness. The
M0 fixture must test the actual route's conditional identity:

```text
E[gamma_hat_t(f) | frozen SMC protocol]
  = integral tilde_pi_t(theta) f(theta) dtheta
```

for the declared Feynman--Kac potentials, normalizer estimator, resampling
convention, mutation kernels, and admissible test-function class. If that
identity is not established for the implemented route, M0 remains a normalized
consistency comparator and cannot be used as an authority.

The defensive proposal must be recorded as

```text
m_t(theta) = (1 - epsilon_t) q_t(theta) + epsilon_t r_safe(theta)
0 < epsilon_min <= epsilon_t <= 1
r_safe(theta) > 0 on the target integrand support
```

The actual denominator and stored proposal law are `m_t`, not the component
`q_t`. Its score-class second moment must be checked rather than inferred from
support alone. Every mutation kernel used by M0 must have the relevant bridge
target as an invariant distribution; a finite acceptance rate is not an
invariance proof.

The two fresh baselines are deliberately separate: `C0` is a classical fresh
bootstrap/auxiliary SMC comparator whose normalized estimates are descriptive;
`M0` is the proof-bearing fresh tempered SMC-U candidate whose unnormalized mass
bookkeeping must pass its own fixture before it can serve as an authority.

## Candidate arms and admissibility

| ID | Candidate | Role | Exactness status | Required evidence |
|---|---|---|---|---|
| C0 | Fresh tempered bootstrap/auxiliary SMC with invariant mutation | Tuned classical comparator | Descriptive/consistency comparator | Finite/status checks, reference comparison, independent mode diagnostics |
| M0 | Fresh tempered SMC-U with a bootstrap/auxiliary proposal and invariant mutation | Canonical authority baseline | Eligible to seek target-authority status | Known-density mass fixture, frozen schedule/hash, independent mode-mass stability, mutation-invariance check, target/status gates |
| M1 | M0 plus second-order ETPF analysis transform | Covariance/variance arm | Approximate transform unless a separate density contract is proved | Weighted-moment identity, bridge/support audit, target-reference comparison, no IID claim |
| M2 | M0 plus GenUT local proposal or moment preconditioner | Local proposal arm | Quadrature/representation only | Selected-moment identity, constraint/status check, proposal-tail and downstream utility diagnostics |
| M3 | M0 plus invertible LEDH-PFPF proposal | Density-corrected proposal arm | Eligible to seek exact PF-PF status | Affine known-map density identity, step invertibility, determinant lifecycle, target/reference checks |
| M4 | Full second-order ET-PF with tempering/mutation | Approximate-filter comparator | Approximate only by default | Reference agreement, mode-mass stability, Bayes-law/support audit, explicit approximate-filter nonclaims |
| M5 | M0 plus combinations of M1--M3 | Follow-up only | Not eligible before component arms pass | Fresh factorial or staged comparison with frozen controls and independent audit data |

The primary baseline ladder is C0 (tuned classical fresh SMC), M0
(proof-bearing fresh SMC-U), the one-factor arms M1--M3, and only then the
enhanced combination arm M5. M4 remains an approximate comparator. No arm is
allowed to replace M0 as the authority merely because it has smaller covariance
residuals.

## Execution phases

### Phase 0: review, source closure, and contracts

No sampling or GPU work.

1. Obtain the bounded Fable and Grok reviews of this exact plan.
2. Record the local Acevedo paper at
   `.localresources/papers/ledh_replay_solution_20260824/acevedo-dewiljes-reich-2017-second-order-etpf.pdf`
   and its extracted text/checksum.
3. Implement or specify machine-checkable contracts for M0--M4 before any
   candidate run. The contracts must distinguish calibration-time adaptation
   from claim-run protocol state; the runner must fail closed if a route is
   unimplemented or its frozen-law hash is absent/mismatched.
4. Create fresh output roots and a run manifest schema; never reuse the old
   six-bank replay root as a claim artifact.

### Phase 1: synthetic and known-density fixtures

Run CPU/XLA reference fixtures before q=20:

1. A one-dimensional affine flow with an analytically known Jacobian and PF-PF
   weight identity.
2. A two-mode Gaussian mixture with known mode masses, including a deliberately
   mode-missing input cloud. This tests whether M1/M2 create bridge points rather
   than recover modes.
3. A low-dimensional nonlinear state-space fixture with a trusted dense or
   quadrature reference. Compare C0, M0, M1, M2, M3, and M4.
4. A mutation-invariance and unnormalized-mass fixture for M0, with the
   tempering stages, resampling triggers, mutation controls, and protocol hash
   frozen before the fixture draws.

The fixture promotion criterion is reference agreement for the claimed
quantity. Moment residual alone cannot pass a density or mode gate.

### Phase 2: fresh q=20 authority pilot

Generate new particles from the q=20 target; do not recycle the six historical
banks as authority. Select the tempering schedule and all claim-run controls on
calibration data, then freeze and hash the stages, resampling triggers,
mutation controls, defensive-mixture parameters, and proposal state before the
claim draws. Use a full-support defensive component and invariant mutation
kernels. Store for every row:

- target value/status and unnormalized mass contribution;
- proposal, transition, observation, and mutation log densities;
- tempering stage, ancestry, seed, mode diagnostic label, and worker identity;
- frozen protocol/schedule hash and proposal-law version;
- exact target and geometry signatures.

An online-adaptive tempering variant may be explored as a separate C0-class
descriptive comparator. It cannot be called M0 authority until its actual
adaptive protocol has its own conditional mass fixture and replay/conditioning
argument.

Run C0 and M0 on the same target partitions and seeds. Use historical `N=100`
per run only as a comparator, the historical pooled `N=600` as a matched-scale
comparator, and a fresh `N=300`/`N=600` ladder as a proposal hypothesis. These
counts are measured or hypothesis values, not defaults. Use at least two tuning
seeds and reserve four independent seeds for any claim-bearing comparison; this
seed policy is a reviewed campaign hypothesis, not statistical proof by itself.

### Phase 3: one-factor modular arms

With the M0 authority controls frozen, run M1, M2, and M3 separately on
disjoint calibration, selection, and audit partitions.

- M1 applies the second-order transform after the weighted analysis step. The
  original weighted cloud remains available for the authority calculation.
  The transform output is tagged approximate/auxiliary unless a transition
  density is supplied.
- M2 uses GenUT only to construct local proposal or covariance information.
  Sigma points are never counted as IID replay rows.
- M3 uses the invertible LEDH-PFPF proposal and records the full density
  lifecycle. Any post-flow reset is either included in a proved density or moved
  to an auxiliary lane.

M4 is run only after the C0/M0 fixtures and its own transform/support audit
pass. It is compared as an approximate filter, not promoted to M0.

### Phase 4: NeuTra training screen

Train batch-native NeuTra transports on the M0-authoritative weighted bank and,
in separate labeled arms, on M1--M3 auxiliary outputs. Freeze proposal laws and
selected controls before the audit partition. GPU/XLA training must set and
verify TensorFlow memory growth before device initialization, preserve the
leading batch dimension, and record device/XLA/TF32 provenance.

Training diagnostics nominate candidates only. A candidate must pass exact
forward/pullback/Jacobian parity and a two-mode transformed-target canary before
any HMC work.

### Phase 5: downstream HMC and predictive validation

This phase requires a separate execution plan after Phase 4. The canonical
sequential NeuTra policy remains binding: warm-up readiness uses the recent
window and maximum rank/folded R-hat; tuning admission uses modern R-hat; all
chains must move and pass finite/status/energy gates; warm-up is excluded from
posterior estimates. No mode-specific chain or pooled conditional archive is
admissible.

### Phase 6: adjudication

Write a result note with:

- engineering, numerical, and scientific ledgers;
- decision and inference-status tables;
- run manifest with commit, environment, device, seeds, wall time, artifacts,
  plan hash, and result hash;
- candidate rejection versus research-direction rejection;
- post-run red-team explanation and the smallest next discriminating artifact.

## Campaign budget and execution boundary

The inherited user budget is `64800 s` (18 hours). It is a cap, not evidence of
feasibility. The following allocation is a planning hypothesis and must be
recomputed from measured pilot timing before launch:

| Phase | Planning cap | Provenance/status |
|---|---:|---|
| Phase 0 review/contracts | `1800 s` | Convenience planning bound; no GPU |
| Phase 1 fixtures | `7200 s` | Derived pilot bound; CPU/XLA reference lane |
| Phase 2 C0/M0 authority pilot | `14400 s` | Derived from the fresh-particle ladder hypothesis |
| Phase 3 M1--M4 modular arms | `18000 s` | Derived comparison bound; stop after the first invalid arm |
| Phase 4 NeuTra training screen | `18000 s` | Derived GPU/XLA warm-start bound; only if prior phases pass |
| Closeout and reserve | `5400 s` | `3600 s` closeout plus `1800 s` uncommitted reserve |

No HMC or posterior-predictive campaign is included in this cap. Phase 5 is a
separate future plan and budget. A phase may not borrow the reserve or silently
reduce a required gate to fit the cap.

## Evidence contract

**Question.** Does a fresh SMC authority plus modular covariance, local-moment,
and flow mechanisms improve target-faithful q=20 particle support and NeuTra
training inputs?

**Comparator.** C0 fresh classical SMC, M0 fresh SMC-U, historical fixed replay
as descriptive-only context, and M4 full ET-PF as an approximate comparator.

**Primary promotion criteria.**

1. The route's stated density/mass contract passes an exact fixture.
2. The q=20 authority has stable unnormalized mass and mode-aware diagnostics
   across independent seeds and partitions.
3. A candidate's claimed role is validated downstream: covariance arms require
   covariance plus target-reference checks; flow arms require density identity;
   NeuTra arms require exact transformed HMC gates.

For any stochastic ranking, report paired per-seed differences and a declared
uncertainty analysis (for example, a bootstrap/MCSE interval appropriate to the
replication unit). With only the tuning seeds or a short screen, differences
remain descriptive and cannot rank arms.

**Hard vetoes.** Non-finite values, invalid target status, support failure,
missing density terms, stale frozen-law hashes, mutation non-invariance,
deterministic rows mislabeled IID, normalized-only replay used as unbiased, or
artifact/hash/device-policy failure.

**Explanatory only.** Whitening, ESS, loss, covariance error, cap activity,
bridge fraction, and runtime.

**Nonconclusions.** Passing a fixture or screen does not establish exhaustive
mode discovery, exact q=20 filtering, posterior correctness, HMC convergence,
or superiority.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Earliest check | Promotion status |
|---|---|---|---|---|---|
| C0 fresh classical SMC | Classical baseline requirement | Separates ordinary fresh-particle behavior from SMC-U bookkeeping | Normalized finite estimates remain descriptive | Reference and finite/status checks | Comparator only |
| M0 fresh SMC-U authority | Repair demanded by current replay evidence and prior math note | Restores fresh draws and auditable mass bookkeeping | Proposal tails or mutation still miss modes | Known-density SMC-U fixture | Candidate authority |
| `N=100`, `N=300`, `N=600` ladder | `100`/`600` measured historical scales; `300` proposed intermediate | Separates scale from mechanism | Resource under-budget or misleading scale effect | Pilot timing and independent seeds | Hypothesis ladder |
| Two tuning plus four claim seeds | Statistical-comparison discipline plus bounded budget | Separates tuning from claim evidence | Still insufficient for fine ranking | Per-seed table and uncertainty analysis | Reviewed campaign hypothesis |
| Calibration-selected, claim-run-frozen tempering | SMC literature and mode-bridge requirement | Selects a useful schedule without changing the claim-run conditioning measure | A schedule or trigger may drift after calibration | Schedule/protocol hash parity and frozen-protocol mass fixture | Required M0 protocol; not yet a default |
| Online-adaptive tempering | Exploratory repair hypothesis | May respond to observed ESS or mode separation | Conditional SMC-U identity and replay law are unestablished | Actual adaptive-protocol fixture | C0 descriptive comparator only |
| Defensive full-support component | Hesterberg-style support protection | Prevents zero denominator on covered support | Infinite second moment remains possible | Tail second-moment estimate | Required assumption, not proof |
| ETPF Sinkhorn grid `[0.1, 1, 10]` | Dimensionless normalized-cost pilot hypothesis | Tests regularization sensitivity | Scale-dependent artifacts | Marginal residual and bridge audit | Calibration-only hypothesis |
| GenUT controls | Existing integration note | Local moment/proposal control | Caps alter tails and are not density maps | Selected-moment and support fixture | Warm-start hypothesis |
| LEDH step schedule | Li--Coates source; scope-specific tuning policy | Invertibility and weight correctness | Singular step or missing determinant | Affine map identity | Required theorem assumption |
| Batch size `64` | Existing q=20 screen, reused as a warm start only | Keeps target evaluation batch-native | Capacity or variance mismatch | Shape/device receipt | Warm-start hypothesis |
| 18-hour campaign envelope | Prior user-authorized budget (`64800 s`) | Bounded research execution | HMC cannot fit after authority work | Per-phase wall ledger | Budget cap, not scientific evidence |

No numeric choice above becomes a default without target-specific evidence and a
reviewed tuning artifact. In particular, the ETPF regularization, GenUT caps,
tempering schedule, mutation scale, flow step size, and particle count remain
scope-specific.

## Skeptical pre-execution audit

| Risk | Audit finding | Required boundary |
|---|---|---|
| Wrong baseline | Historical normalized replay is not an authority | M0 fresh SMC-U is the primary baseline; old replay is descriptive only |
| Proxy promotion | Covariance and whitening can look excellent while modes are wrong | Require target-reference and mode-mass checks |
| ET bridge artifacts | Deterministic transforms can create between-mode rows | Measure bridge fraction and target validity; preserve raw weighted cloud |
| GenUT overclaim | A 41-point q=20 local rule can be mistaken for a global sample bank | Tag sigma points as quadrature only |
| Flow density drift | A reset after LEDH may be absent from the determinant | Fail closed unless the complete proposal density is available |
| Adaptive protocol drift | Tempering, resampling, or mutation controls may change during a claim run | Freeze/hash all claim-run controls; demote untested online adaptation to C0 |
| Fairness | Arms could receive different particles or tuning partitions | Bind target, data partitions, seeds, controls, and budget in each scope |
| Mode labels | Sign regions may be diagnostic, not exhaustive | Use synthetic known-mode fixtures and make no exhaustive claim |
| Under-budgeting | HMC can consume the whole campaign before authority validation | No HMC in this plan; reserve a separate budget and plan |
| Source mismatch | Papers support different operations | Record source-faithful, project-derivation, and extension labels per arm |

Audit disposition: `PASS_WITH_REVIEW_GATES`. The plan is meaningful only if the
authority/auxiliary boundary and exactness labels remain unchanged.

## Pre-mortem

The run could pass while misleading us if ETPF matches covariance by creating
bridge points, if GenUT lowers loss through correlated quadrature rows, or if
the flow proposal has a correct local determinant but misses a mode. The cheap
detectors are target-reference mode masses, bridge/support audits, independent
fresh draws, and the affine density identity.

The run could fail for tuning or infrastructure rather than science if the
Sinkhorn scale, mutation scale, flow step, GPU memory policy, or XLA shape
contract is wrong. The fixture ladder and CPU-only reference route must
separate those failures before q=20 interpretation.

The strongest alternative explanation for a failed q=20 candidate is still
insufficient fresh particles or mutation budget, not rejection of the method
family. A true continuation veto is only a broken target/proposal identity or
invalid artifact, not a poor proxy metric. A superficially successful adaptive
run could instead be conditioning on its own particles; the protocol hash and
the frozen-versus-adaptive fixture distinguish that explanation.

## Planned commands and artifacts

The runner is intentionally not implemented yet. The following commands are
the exact planned interfaces and must fail closed until the reviewed runner and
tests exist:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_modular_2026_08_24.py \
  --phase contracts --output-root \
  docs/plans/artifacts/ssl-lstm-q20-particle-authority-modular-2026-08-24/contracts

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_modular_2026_08_24.py \
  --phase authority --device 1 --output-root \
  docs/plans/artifacts/ssl-lstm-q20-particle-authority-modular-2026-08-24/authority
```

The output root must contain separate `contracts`, `authority`, `arms`,
`training`, and `adjudication` directories. Every serious GPU artifact must
record `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified physical/logical devices,
XLA, TF32, dtype, commit, command, environment, seeds, wall time, and hashes.
External sample generation is a multicore CPU lane; GPU is reserved for
batch-native NeuTra training and approved tensor kernels.

## Stop, repair, and promotion rules

- Stop an arm on a hard veto and preserve its artifact as candidate evidence.
- Repair a harness or scope-specific tuning failure under the same target,
  budget, hardware class, and evidence contract; do not silently relax a gate.
- Treat any claim-run tempering, resampling, or mutation schedule that is not
  frozen and hash-matched as C0-class descriptive evidence, not M0 authority.
- If M1/M2 improves moments but fails target or mode checks, retain it only as
  an auxiliary representation arm.
- If M3 fails the affine identity or determinant lifecycle, do not call it a
  density-corrected proposal; repair or reject that candidate.
- If M0 cannot establish auditable mass and support, stop all downstream
  promotion; do not compensate with whitening or validation loss.
- Do not rank viable stochastic arms from descriptive metrics without declared
  uncertainty evidence.
- No default, posterior, HMC, or scientific promotion is possible from this
  documentary plan alone.

## Review and closeout state

Requested independent reviews are stored in separate handoff files:

- Fable handoff:
  `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-modular-etpf-genut-ledh-fable-handoff-2026-08-24.md`
- Grok handoff:
  `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-modular-etpf-genut-ledh-grok-handoff-2026-08-24.md`

The result paths are intentionally distinct. The 2026-08-25 Fable reply
reviewed the pre-amendment plan and returned `VERDICT: AGREE` with two minor
repairs: freeze/hash the claim-run tempering protocol, and state the ETPF
moment constraints explicitly. Those repairs are applied above.

The requested new Grok reply path
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-modular-etpf-genut-ledh-grok-review-reply-2026-08-24.md`
is not present in the workspace. The older dual-cap Grok review is documentary
evidence about the previous plan, not a review of this amended plan, so it is
not silently counted as a new Grok verdict. This plan therefore remains
`AMENDED_AFTER_FABLE_REVIEW_AWAITING_NEW_GROK_REPLY`; no implementation-phase
plan or runtime work is authorized until the amended text is reviewed.
