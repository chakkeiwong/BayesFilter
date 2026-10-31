# Principal-subspace refusal repair result

The shared precision comparison now refuses a requested principal subspace
when its boundary eigenvalue gap is not numerically resolved. Public comparison
and stability consumers raise `principal subspace is not numerically resolved
at the requested rank`; enclosing initializers return fit error5 and refuse
usable geometry. Earlier matrix/symmetry/rank errors retain their precedence.
Matrices, ranks, eigenvector conventions, finite accepted angles, optimizer
settings and accuracy bounds are unchanged.

This distinguishes covariance conditioning from subspace identifiability.
An isotropic covariance is well-conditioned, but a lower-dimensional subspace
cut from its repeated eigenvalue is non-unique. The guard checks the boundary
gap against the residual, orthogonality and machine-roundoff estimate documented
in filter_gradient_subspace_identifiability_20260929.md. It reports numerical
resolution failure, not proof of exact eigenvalue multiplicity. Full rank, or
a selected subspace containing the whole repeated cluster, remains valid.

Baseline0bfae62f2. All five serialized workers04931--04935 pass, charging
352.820853 CPU and541.698180 GPU process-seconds. No failed worker or retry.
GPU2 UUID GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba had verified memory growth;
CPU was an explicit hidden-device reference. Tests use binary64 and disable
TF32. The worker startup log precedes that test-specific TF32 setting.

| Evidence | Result |
|---|---|
|04931 CPU /04932 GPU controls |52 checks each, including84 spectrum/scale cases per backend across graph/XLA, D3/D23 and scales1e-8/1/1e8 |
|Repeated/near-tied spectrum |Partial-rank comparison refuses; full-rank and whole-cluster cases pass; a resolved1e-8 gap passes |
|Error precedence |Left-invalid1, right-invalid2 and rank-invalid3 precede subspace error5; incomplete groups retain their original outcome |
|Actual enclosing initializer |Frozen disjoint isotropic score clouds produce native error5, `usable=False`, and the explicit public error |
|04933 CPU /04934 GPU complete saved fits |Identical04591 input bytes; zero raw or exported record differences from04919/04920 under the existing bounds |
|04935 terminal readback/policy |161 pass; current numerical source hashes and GPU growth are checked against all four preceding workers |

The saved D23 fits retain their selected `consensus_diagonal_consensus`
geometry, covariance, precision, audit values, statuses and all unselected fit
records. Known principal-angle accuracy, five-degree decisions, pair ordering,
optional caps, graph bounds and dense derivatives pass. The exact policy guard
remains315 sources/1457 existing allowances; no numerical-loop, NumPy, pfor or
non-JIT allowance was added. The seven-value internal comparison signature is
unchanged. Nonfinite active angle slots are internal refusal status; tested
public routes raise before serialization.

| Decision | Criterion/veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|
|Keep the guard |Refusal controls and complete healthy saved records pass CPU/GPU |The conservative resolution estimate is not an interval-arithmetic certificate |Commit scoped repair and preserve its evidence |Universal eigenspace identifiability |
|Close the tested rank-cut reporting defect |Public/native/enclosing errors agree and preserve earlier errors |Factor initialization still uses the existing parameter chart; this repair does not change an unfinished optimizer trajectory |Retain unselected-fit and first-divergence investigations |All isotropic initialization discrepancies repaired |
|Keep cost acceptance open |Correctness tests pass |Added residual/orthogonality products have no matched fresh-process cost attribution yet |Include this guard with angle/geometry terminal cost renewal |Accepted memory or runtime tradeoff |

The strongest alternative explanation for a refusal is a poorly resolved
numerical eigensystem rather than exact degeneracy; the message deliberately
does not conflate them. A known separated-spectrum false refusal or any changed
healthy selection would veto this repair. These controls do not establish
posterior/statistical identifiability, optimizer convergence, canonical LEDH,
HMC readiness or whole-program completion. No independent agent review was used.

Raw artifacts are in the main checkout's
docs/plans/artifacts/filter-gradient-repair-20260917. The archive
subspace-identifiability-04935-evidence.tar.gz has34 reopened/verified members,
1657251 bytes, SHA256
7d1ed83ac0bc8b8af43f8c09da3db9bee15321f15361b1e70c1581e376983837.
Individual run manifests preserve commands, environment, source/input hashes,
device settings and wall times. The long control batches are correctness runs,
not matched compile-time or performance comparisons.
