# BayesFilter DZ5 Structured-Curvature and Consensus-Repair Plan

Date: 2026-07-16

Status: `EXECUTED_ENGINEERING_VERIFIED_NO_HMC_RUN`

Repository: `/home/ubuntu/python/BayesFilter`

Starting commit: `20835ecf90bff78ca93c5d401f231e4aa94e63ce`

Implementation checkpoint: `56b1f3c` (`Add structured score-curvature initialization`), pushed to `origin/main`

Runtime classification: `accepted TF/TFP runtime` for new numerical code; explicit CPU-only focused tests are `small diagnostic` exceptions with `CUDA_VISIBLE_DEVICES=-1`. No retained HMC or scientific run is authorized by this plan.

## Question and research-intent ledger

Main question: can BayesFilter turn repeated exact-score evaluations near a fixed reviewed center into a reproducible SPD HMC geometry candidate without confusing score stationarity, SPD projection, or a locally predictive regression with evidence of a MAP, posterior convergence, or production readiness?

Candidate mechanism: fit dense local precision matrices and identified one/two-factor covariance parameterizations to independent antithetic score clouds, invert each structured covariance inside the objective to obtain its score-prediction precision, retain raw curvature diagnostics, test stability across independent fits, and, when no single fit is stable, nominate a consensus SPD precision convexly combined with a separately reviewed structured or diagonal precision target.

Expected failure mode: weak directions are poorly determined. Dense raw fits change inertia across seeds; projecting each fit to SPD hides that failure and makes generalized eigenvalue comparisons explode. A low center score can coexist with this instability.

Primary engineering criterion: documented formulas, code, public payloads, and deterministic tests agree; fresh holdout rows never enter fit/training; unsupported fallback branches fail closed.

Promotion criterion: none in this documentation-and-engineering task. A geometry may be marked `eligible_for_exact_hmc_canary` only after its declared raw/stability/holdout gates pass. Promotion for retained HMC requires a separate reviewed exact-HMC plan.

Promotion vetoes:

- non-finite target value/score or matrices;
- covariance/precision orientation mismatch;
- unidentified structured fit or failed Jacobian-rank gate;
- raw curvature instability hidden by projection;
- leakage from reused search rows into fresh holdout or terminal evidence;
- missing target/data/code lineage;
- exact HMC mechanics failure in a later canary.

Continuation vetoes for this task:

- baseline tests fail for an unexplained pre-existing reason;
- implementation equations cannot be reconciled with definitions;
- relevant files acquire overlapping unexplained concurrent edits;
- MathDevMCP CLI and manual source inspection both fail to provide a usable math/code audit;
- documentation cannot compile because of an in-scope error;
- a proposed API would require silently changing the exact posterior or HMC transition kernel.

Repair trigger: a candidate that is finite but fails stability or holdout diagnostics triggers the next declared fallback branch; it does not reject curvature preconditioning as a research direction.

Explanatory diagnostics: center score norm, optimizer iteration counts, ordinary matrix norms, per-fit runtimes, and short exact-HMC acceptance summaries. These do not independently establish geometry validity.

What must not be concluded: MAP attainment, posterior covariance truth, model identification, HMC convergence, full-chain XLA readiness, CPU/GPU speed superiority, default readiness, or validity of the current both-binding dense projected matrix.

## Evidence contract

| Field | Contract |
| --- | --- |
| Engineering question | Do code, tests, documentation, provenance, and fail-closed behavior implement the declared local score-geometry contracts? |
| Exact baseline | Checkpoint `56b1f3c`: structured factor fit and sequential refinement, with the legacy score-gated terminal path unchanged. |
| Comparator | Dense symmetric score regression before SPD projection; one-factor fit; identified two-factor fit; consensus/shrinkage candidates where implemented. |
| Primary pass criterion | Focused tests pass, LaTeX builds, equations trace to code/tests, and independent data partitions are enforced. |
| Hard veto diagnostics | Non-finite values; non-SPD accepted output; rank/identification failure; covariance/precision mismatch; holdout leakage; raw-instability suppression; automatic CPU fallback; invalid GPU allocator state. |
| Explanatory only | Score proximity, fit loss, descriptive runtime, short-chain acceptance, and projected-matrix closeness without raw evidence. |
| Artifact | This plan; a dated result note; compiled `docs/main.pdf`; source-map entries; focused pytest output summarized in the result. |
| Non-conclusion | No HMC/posterior/scientific/default promotion from documentation or synthetic tests. |

## Authority and baseline audits

### BayesFilter usage audit

Active implementation is under `bayesfilter/inference/`; tests import
BayesFilter. MacroFinance files are evidence only. Before finalization, search
both repositories for accidental active imports from MacroFinance-local
`filters.*`, `inference.hmc*`, `inference.mass_matrix`, or
`inference.posterior_adapter`. Any such dependency in this lane's executable
path is a continuation veto until routed through BayesFilter. Repository-wide
historical/reference scripts are recorded separately and are not evidence for
this lane.

### Implemented-versus-proposed ledger at checkpoint `56b1f3c`

| Item | Status before this plan executes |
| --- | --- |
| Score equation, one/two-factor covariance parameterization, row-ball SPD transform | Implemented; covariance parameters are optimized through predictions made with their inverse precision |
| Structured precision returned for downstream use | Implemented by Cholesky inversion of the fitted covariance; no independent precision-native factor parameterization exists |
| One-factor sign and two-factor triangular-anchor normalization | Implemented |
| Parameter counts and two-factor prediction-Jacobian rank gate | Implemented |
| Dimension-scaled count, orthogonal antithetic clouds, local search-row reuse | Implemented as opt-in sequential policy |
| Independent fresh structured train/holdout split and `4N` fresh rows | Implemented for one fit; no separate shrinkage-selection audit cloud exists yet |
| Exact target/trust-ratio proposal evaluation and one-to-two-factor escalation | Implemented |
| Legacy terminal dense fit and MAP semantics | Implemented and still score-gated |
| Fixed-center curvature API independent of MAP localization | Not implemented as a public end-to-end API |
| Cross-seed/radius/center stability metrics | Implemented only in MacroFinance harness evidence, not BayesFilter public code |
| Consensus/shrinkage fallback | Proposed, not implemented |
| Persistent multiprocessing cloud evaluator | Proposed for this API; no automatic CPU fallback |
| `PrecomputedMassArtifact(position_role="diagnostic_center")` handoff | Generic role support exists; lane-specific constructor/gate not implemented |

### MathDevMCP availability gate

Commands run before plan drafting:

```bash
codex mcp list
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/anaconda3/bin/python -m mathdevmcp.cli doctor
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/anaconda3/bin/python -m mathdevmcp.cli --help
```

Result: no direct MCP server is configured. The CLI doctor returned `ok: true`; LaTeXML, Pandoc, Sage, and SymPy are available. Lean is unavailable and is not required for the scoped matrix identities. The CLI route will be used for focused document/code comparison. Its output is diagnostic source evidence, not formal proof.

## Default and numeric-assumption audit

| Choice | Provenance | Justification/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| `N^2` for `N <= 10` | User instruction | Reviewed rule to preserve dense search coverage at small dimension | Quadratic growth if boundary is moved upward | Boundary tests at `N=1,10,11` |
| `100+(N-10)log(N-10)` for `N>10` | User instruction | Reviewed search-count hypothesis, not a statistical optimum | Too few points for difficult weak directions | Rank/holdout/stability failure; allow explicit override |
| Round to next even integer | Derived from antithetic pairing | Deterministic `2 ceil(raw/2)` | More rows than an integer-floor interpretation | Exact boundary unit tests and documentation |
| `4N` fresh structured rows | Existing checkpoint implementation | Convenience baseline: gives `2N` fresh train plus `2N` holdout rows | May under-inform weak two-factor fits | Rank and holdout diagnostics; no adequacy claim |
| One factor before two | User decision and parsimony | Baseline model-escalation order | One factor may underfit; two factors may be unidentified | Independent holdout plus Jacobian rank |
| Loading margin `1e-6` | Existing implementation convenience | Numerical interior margin, not scientifically calibrated | Artificially excludes near-boundary loadings | Boundary/property tests; record as unproven default |
| Condition cap `1e8` | Existing implementation convenience | Numerical safety cap, not a geometry-quality theorem | Can floor meaningful weak directions | Report raw spectrum and projection burden |
| Structured holdout relative RMSE `0.35` | Existing implementation convenience | Screening baseline only | Permissive threshold may admit poor geometry | Synthetic misspecification tests; do not promote as universal default |
| Consensus equal weights | Proposed simplest baseline | Transparent first implementation; no optimality claim | Ignores varying fit quality | Compare admissible-set sensitivity; expose weights |
| Shrinkage grid | Explicit caller input, with endpoints required | Selection hypothesis, not an estimated classical intensity | Overfits a small selection set | Separate fresh audit cloud; exact-HMC veto remains separate |
| CPU workers `floor(physical cores/3)` | User instruction | Operational default with explicit override | Oversubscription or affinity mismatch | Small multiprocess lifecycle test and manifest |
| CPU batch size `B=1` | User instruction/benchmark hypothesis | Avoids nested batch memory growth and keeps tasks independent | Per-task overhead may dominate | Timing is descriptive only; do not change default without a reviewed benchmark |

## Ten technical parts

### 1. Fixed-center score-curvature target

Document and test, in standardized coordinates,

\[
  g_z(c)-g_z(c+z) \approx P_z z, \qquad P_z=C_z^{-1}.
\]

Add a public fixed-center diagnostic API whose output uses `center` or `diagnostic_center`, not `map_candidate`. Do not remove the stationarity gate from the legacy sequential MAP-localization API. Center score is recorded as explanatory. A Hessian-at-MAP claim requires a separately established stationary MAP and is not implied by a locally stable score derivative.

The legacy API's current admission rule is explicit: its standardized maximum absolute center score must satisfy `terminal_score_max_abs`, its fresh terminal dense fit must be usable, and its projection burden must not exceed `terminal_projection_relative_frobenius_cap`. Those conditions nominate a local MAP candidate under that API's nonclaims; they do not certify a global MAP. The fixed-center API does not reuse that stationarity gate.

### 2. Dimension-scaled search

Keep the checkpoint rule

\[
 m(N)=
 \begin{cases}
 N^2,&N\leq 10,\\
 100+(N-10)\log(N-10),&N>10,
 \end{cases}
 \qquad m_{\rm even}=2\left\lceil m(N)/2\right\rceil.
\]

Test `N=1`, `N=10`, and `N=11`, positivity validation, determinism, and even pairing. State that this is a user-selected computational policy, not a sample-complexity theorem.

### 3. Search-cloud design and reuse

Use stateless orthogonal frames and antithetic pairs. Reuse only finite nonzero earlier search rows whose translated standardized distance is within the current radius. Reused rows may augment training only. Still generate independent fresh train and holdout clouds. Preserve the checkpoint per-fit baseline of at least `4N` fresh structured rows, split equally between train and holdout. Consensus/shrinkage selection requires a third independent fresh audit cloud of at least `2N` rows generated with a disjoint stateless seed; it is never used for fitting, factor escalation, target choice, or shrinkage-weight selection. Add explicit partition metadata or indices so tests can prove no reuse or overlap leakage among training, selection holdout, and audit rows.

### 4. Structured covariance geometry

For $q\in\{1,2\}$, document and test

\[
 C_z=D R D,\qquad
 R=\operatorname{diag}(1-\|L_i\|_2^2)+LL^\top,
 \qquad \|L_i\|_2^2<1.
\]

Equivalently, for independent standard-normal $F_k,e_i$,

\[
 x_i=\sum_{k=1}^q\rho_{ki}F_k+
 \sqrt{1-\sum_{k=1}^q\rho_{ki}^2}\,e_i,
 \quad
 \operatorname{corr}(x_i,x_j)=\sum_{k=1}^q\rho_{ki}\rho_{kj}.
\]

The implemented structured fit optimizes unconstrained parameters that decode to $C_z$, computes $P_z=C_z^{-1}$ by Cholesky solve inside every objective evaluation, and predicts score differences with $P_z z$. Its result returns both `covariance_z` and `precision_z`. Dense regression instead estimates a symmetric precision directly before projection. Consensus and HMC handoff operate on precisions. Tests must catch covariance/precision reversal.

### 5. Identification

For one factor, normalize the largest-absolute loading to be nonnegative. There are $N$ log-standard-deviation coordinates and $N$ loading coordinates, so its continuous parameter count is $2N$; the sign choice is discrete and removes an equivalent representation rather than a continuous degree of freedom. For two factors, choose two anchor rows, impose a lower-triangular anchor block, and require positive diagonal anchors. Starting from $N$ deviations and $2N$ loadings, the forced upper-right anchor entry removes one continuous coordinate, giving $3N-1$; the positive anchor signs remove discrete symmetries. Reject dimensionally impossible cases and two-factor fits whose prediction Jacobian is not full column rank. Document that anchor coordinates identify nuisance loadings; the implied covariance/precision is the downstream object. Test factor-column permutations/rotations before normalization and verify that equivalent representations yield the same covariance/precision after normalization.

### 6. Model escalation and proposal mechanics

Attempt one factor first. Escalate to two factors only when one factor fails the declared fit/holdout gate, and reject if the second factor is unidentified. Quadratic/trust-region proposals remain proposals only: evaluate the exact target value and score, compute actual/predicted improvement, and accept/reject using the declared trust mechanics. The regression never replaces the posterior. Fixed-center diagnostic fitting does not need to move the center at all.

### 7. Curvature diagnostics

Add BayesFilter-owned diagnostics over repeated fits. A configuration distinguishes required boolean gates from telemetry; no built-in universal numerical thresholds are inferred from DZ5:

- finite raw matrices and caller-declared admissible raw inertia: required gate;
- raw eigenvalues, inertia/sign counts, and raw matrices before projection: retained veto evidence and repair trigger;
- caller-supplied caps on generalized-eigenvalue spread, trace-normalized Frobenius/operator differences, principal angles, projection burden, and selection-holdout error: required gates only when provided;
- the same quantities without supplied caps: descriptive telemetry that cannot yield `eligible_for_exact_hmc_canary`;
- a separate audit-cloud error cap: required final gate for a selected consensus/shrinkage candidate;
- sensitivity over the caller-declared seeds, radii, and small center perturbations: required coverage metadata; eligibility requires every requested comparison to pass.

`eligible_for_exact_hmc_canary` is returned only when every required gate has an explicit threshold and passes; otherwise the strongest status is `diagnostic_only`. The plan will not invent universal cutoffs from the one DZ5 fixture.

### 8. Consensus-shrinkage repair and fallback policy

For an admissible set of independently estimated precisions $K_s$, retain each raw estimate and form SPD projections $K_s^+$. The proposed equal-weight baseline is

\[
 \bar K=S^{-1}\sum_{s=1}^S K_s^+,
 \qquad
 K_\lambda=(1-\lambda)\bar K+\lambda T,
 \qquad 0\leq\lambda\leq1.
\]

This is plain convex consensus shrinkage toward a chosen SPD target, not the classical Ledoit--Wolf estimator: these are designed score-regression fits, not iid covariance observations, and no classical closed-form intensity or theory claim is imported.

Fallback order:

1. stable identified one-factor precision;
2. stable identified two-factor precision when one factor is inadequate;
3. consensus of admissible projected estimates, convexly combined with an explicitly caller-selected structured precision that has independently passed the same identification, stability, and selection-holdout gates;
4. if factor geometry is unsupported, consensus shrunk toward `diag(consensus)` as the secondary target;
5. diagonal-only output only as an explicit diagnostic/reviewed operational fallback;
6. no automatic identity fallback.

Every branch retains the same finite-input, raw-evidence, projection-burden, and fresh selection-holdout requirements. Branch 4 is unavailable when the dense raw inputs themselves fail caller-declared admissibility/stability gates; diagonalization cannot repair an inadmissible consensus. Branch 5 can only return `diagnostic_only` or an explicitly reviewed operational artifact, never automatic eligibility. If no branch passes, return `geometry_readiness_blocked`; never manufacture identity geometry.

Shrinkage weights and any structured target are chosen using training plus selection-holdout evidence only. The disjoint fresh audit cloud is evaluated exactly once after selection and can veto but never change the choice. A selected candidate is only nominated until a separate exact-HMC mechanics canary passes; short-HMC evidence is never folded into the deterministic regression objective.

### 9. Artifact and HMC handoff contract

Construct a `PrecomputedMassArtifact` only after geometry gates pass, with `position_role="diagnostic_center"`, explicit target/data/code lineage, coordinate system, fit seeds/radii, selection diagnostics, and nonclaims. Do not label the center `map`. Preserve adaptation draws separately and discard them from retained posterior inference. A finite exact-HMC start does not establish MAP localization, convergence, or posterior correctness.

### 10. Backend and operational policy

TF/TFP is the accepted implementation backend. Target-only XLA value/gradient parity is separate from full-chain XLA. GPU processes require `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import and verified per-device memory growth. A noncompliant GPU launch is invalid.

Provide an explicit persistent spawn-based CPU multiprocess evaluator only for independent cloud rows, with `B=1`, deterministic result ordering, optional affinity where supported, and caller override. Physical cores mean `psutil.cpu_count(logical=False)` when it returns a positive integer; otherwise fall back to `os.cpu_count()` and record `logical_fallback`. The user-directed default is `max(1, floor(detected_cores/3))`, clamped to the task count. This is an operational baseline, not a performance-optimal reviewed default. Test it independently before it is offered. It must never activate automatically after a GPU failure. CPU-only tests set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and label the run as diagnostic.

Progress events must report semantic stages/evaluation counts. Heartbeats report liveness but are not semantic progress. Time budgets are safety guards: a healthy process making recorded semantic progress is not killed merely because a nominal cap elapsed; stale/no-progress and hard resource limits remain explicit supervisor decisions.

## Planned implementation and documentation changes

1. Add or extend BayesFilter inference code for a fixed-center repeated-fit/stability/consensus API without changing legacy MAP semantics.
2. Add deterministic consensus/shrinkage and fallback-selection helpers with raw-estimate retention and fail-closed statuses.
3. Add an explicit CPU multiprocess cloud-evaluation utility only if a spawn lifecycle test and a child-process `tf.function(jit_compile=True)` value/score smoke provide engineering evidence for that exact child route; otherwise document it as architecture-blocked, not implemented. The smoke is not a proof of scaling or performance.
4. Add a lane-specific diagnostic-center mass-artifact constructor or validator that uses existing `PrecomputedMassArtifact` role support.
5. Extend `ch04_bayesfilter_api.tex`, `ch22_mass_matrices.tex`, `ch24_xla_jit.tex`, `ch25_diagnostics.tex`, `ch30_cip_afns_case_study.tex`, and `ch32_production_checklist.tex`.
6. Update `docs/source_map.yml` and `docs/appendices/app_f_source_map.tex` with code, tests, plan, MacroFinance result/reset, and MathDevMCP audit provenance.
7. Build a requirement-to-test matrix, add missing deterministic/property/negative/integration tests, run focused tests, and compile `docs/main.tex`.

## Test-sufficiency audit targets

The result note will record, for every row, the existing test, what it proves, what it does not prove, the added test, determinism, backend, and runtime. Required rows are:

- symmetry, declared variances, correlation identity, and strict SPD;
- row-ball interior and boundary rejection;
- one-factor sign and two-factor rotation/sign normalization;
- factor-column permutation/rotation equivalence and anchor-choice sensitivity of the implied covariance/precision;
- exact parameter counts and identified/degenerate Jacobian rank;
- covariance/precision inversion orientation;
- antithetic pairing and counts at `N=1,10,11`;
- reuse distance filtering and no training/selection-holdout/audit leakage;
- one-to-two-factor escalation and unidentified rejection;
- raw curvature preservation, projection burden, generalized eigenvalues, principal angles, and cross-fit stability;
- unstable low-score rejection and stable nonzero-score eligibility;
- consensus/shrinkage SPD and fallback order;
- fail closed instead of diagonal/identity manufacture;
- diagnostic-center artifact payload/serialization;
- TensorFlow gradients and target-only XLA parity for the in-process factor/consensus path;
- separate child-process XLA smoke for the multiprocess evaluator if implemented;
- multiprocess lifecycle, override, exception propagation, deterministic ordering, and no automatic CPU fallback if implemented;
- GPU allocator fail-closed configuration through non-GPU unit tests, with no runtime GPU-readiness claim;
- semantic progress versus liveness heartbeat behavior.

## Exact command plan

All focused CPU tests are diagnostic exceptions with GPUs hidden:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_factor_correlation_geometry.py \
  tests/test_sequential_map_covariance.py \
  tests/test_fixed_center_curvature.py
```

Math/code audit, exact subcommand to be selected after labels exist:

```bash
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/anaconda3/bin/python -m mathdevmcp.cli compare-label-code ...
```

Documentation build:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Before each commit:

```bash
git diff --check
git diff --cached --check
git status --short
```

No GPU scientific run, retained HMC, long benchmark, or automatic CPU fallback is part of this task.

## Pre-mortem

| Misleading success/failure | Cheap discriminator |
| --- | --- |
| SPD output passes because projection erased negative raw directions | Require raw matrix/eigenvalues/inertia and projection burden in payload/tests |
| Low score is mistaken for stable curvature | Synthetic low-score/unstable-fit test and DZ5 both-binding evidence table |
| Stable nonzero-score curvature is rejected as “not MAP” | Separate fixed-center result type and foreign evidence table |
| Reused rows leak into validation | Preserve partition indices/counts and assert disjointness |
| One-factor result looks stable only because it cannot represent the target | Independent holdout; two-factor escalation when one-factor fit fails |
| Two-factor fit appears better but is rotationally unidentified | Triangular anchors plus full prediction-Jacobian rank |
| Consensus looks stable only after flooring | Compare raw inputs first; projection does not cure raw instability |
| Shrinkage weight overfits validation | Separate selection and audit holdout; exact-HMC remains a veto |
| CPU workers repeatedly retrace or oversubscribe | Persistent spawn smoke, deterministic ordering, worker manifest, no performance claim |
| Documentation describes proposed code as implemented | Status callouts and final code/document symbol search |
| A short HMC canary is treated as convergence | Explicit nonclaims and separate retained-HMC plan |

## Skeptical pre-execution audit

Verdict: `PASS_AFTER_REVISION`.

Material flaws found and repaired before execution:

1. The initial proposal risked removing the score gate from `SequentialMapCovarianceConfig`. That would change an existing MAP-localization contract. The revised plan adds a separate fixed-center geometry API and leaves legacy MAP semantics intact.
2. The initial wording could describe consensus shrinkage and multiprocessing as implemented. The revised ledger marks both proposed until code and exact-path tests exist.
3. The shrinkage formula alone did not specify selection/audit separation. The revised plan requires independent holdout/audit evidence and keeps exact HMC as a later veto.
4. “Best SPD projection” could silently turn projection into adequacy. The revised plan requires raw inertia/stability before projection and fails closed when no candidate is valid.
5. The numerical thresholds in checkpoint code lacked provenance. The revised default audit classifies them as convenience baselines, not universal reviewed defaults.
6. The CPU rule could be read as an automatic fallback. The revised plan requires an explicit caller-selected route and forbids GPU-to-CPU fallback.
7. The requested `position_role="diagnostic_center"` was generic capability, not lane-specific handoff proof. The revised plan requires a constructor/validator and tests before documenting an implemented handoff.

Claude review disposition: the bounded review returned `VERDICT: REVISE`. Accepted repairs make covariance-parameter-to-precision prediction explicit, add a disjoint `2N` post-selection audit cloud, operationalize eligibility gates, derive parameter counts, remove Ledoit--Wolf-style terminology, define branch-specific fallback failure semantics, define physical-core detection, scope XLA smokes, add representation-invariance tests, and clarify GPU nonclaims. No finding was rejected; the XLA and GPU findings were accepted with narrow test-scope wording rather than broader runtime claims.

Wrong-baseline audit: the dense comparator is the raw symmetric fit, not its SPD projection. Structured candidates are compared with the same exact score data partitions. No weak diagonal/identity comparator is promoted as the main baseline.

Proxy-metric audit: holdout score prediction, center score, and short HMC mechanics have explicitly limited roles. None alone promotes geometry or establishes convergence.

Stop-condition audit: only invalid evidence, mathematical mismatch, unresolved concurrent overlap, or in-scope build/test failure stops this engineering plan. A candidate geometry failure triggers the next declared repair or a blocked result.

Environment audit: focused tests use the `tfgpu` Python with GPUs intentionally hidden. GPU readiness is not tested or claimed. LaTeX build uses the repository document root. Claude review is bounded and read-only.

Artifact-answer audit: the focused tests answer deterministic implementation contracts; the LaTeX build answers document integration; the source map answers provenance. None answers posterior convergence, so no posterior run is included.

Post-implementation skeptical audit, 2026-07-16: the first green synthetic
suite still contained three material contract gaps. First, copied offset rows
in distinct arrays could cross training/selection/audit boundaries because the
implementation checked only shared memory. Second, the fixed-center route fit
two factors unconditionally instead of escalating only after one-factor
fit/holdout/stability failure. Third, an explicitly requested structured
shrinkage target was shadowed by the earlier direct-factor return. Execution
paused while these were repaired. The revised implementation rejects both
shared-memory and exact copied-row overlap, records the partition check, fits
factor two only after the declared trigger (or an explicit factor-two target),
and routes explicit structured targets through shrinkage selection. Focused
negative and branch tests were added before the wider suite resumed.

A later process-boundary audit found a fourth material defect: locating the
spawn initializer inside `bayesfilter.inference.cpu_xla_cloud` caused child
unpickling to import the heavy `bayesfilter.inference` package before the
initializer could hide CUDA. The initial CPU-hidden pytest environment masked
that ordering error. The repair moves initializer and row execution into the
lightweight top-level `bayesfilter.cpu_xla_worker_bootstrap` module, temporarily
supplies the spawn-inherited CPU/growth environment only while all persistent
workers are created, restores the parent environment, fails closed if the child
did not inherit the required settings, and returns child bootstrap provenance.
A test now imports TensorFlow in the parent, sets a non-CPU parent marker,
verifies parent restoration, and requires every child to report no framework
module at initializer entry plus CPU/XLA settings. A direct missing-environment
test verifies the fail-closed branch.

## Review protocol

After this internal audit, request one bounded read-only Claude review of this exact plan path, following `AGENTS.md`. Claude must identify file/line-specific mathematical, code-correspondence, status, test, fallback, and assumption gaps and end with `VERDICT: AGREE` or `VERDICT: REVISE`. Findings are advisory and will be adjudicated against code and tests. If unavailable after one bounded attempt, record the failure and perform another self-review.

## Three ledgers

| Ledger | Required final state |
| --- | --- |
| Engineering correctness | Focused deterministic tests pass; public payloads serialize; docs compile; source map resolves; no unrelated files staged |
| Numerical/HMC geometry validity | Synthetic invariants and fail-closed branches pass; the MacroFinance foreign-binding fixed-center geometry remains only nominated and the both-binding dense geometry remains blocked; no HMC run |
| Scientific interpretation | No posterior, model-identification, recovery, convergence, superiority, or default claim |

## Run-manifest template

| Field | Value |
| --- | --- |
| Git start/checkpoint/final | `20835ec` / `56b1f3c` / pending |
| Command | record exactly in result note |
| Environment | `tfgpu`, Python/TensorFlow/TFP versions; LaTeX tool version |
| CPU/GPU | explicit; CPU tests hide GPU; no GPU scientific run |
| Memory growth | environment recorded; GPU verification N/A unless a GPU path is run |
| Data version | synthetic fixtures; MacroFinance result hashes are provenance only |
| Seeds | record each stateless seed |
| Wall time | record for tests, MathDevMCP, Claude, and LaTeX |
| Outputs | test output summary, `docs/main.pdf`, plan/result paths |
| Plan/result | this file / dated execution result |

## Final nonclaims

- Passing synthetic tests does not validate the DZ5 both-binding geometry.
- Stable fixed-center curvature is not a MAP claim.
- SPD projection is not evidence of raw curvature adequacy.
- Consensus shrinkage is not classical Ledoit--Wolf estimation.
- Target-only XLA does not establish full-chain XLA.
- A mechanics canary does not establish posterior convergence.
- Documentation completeness does not establish default readiness.
