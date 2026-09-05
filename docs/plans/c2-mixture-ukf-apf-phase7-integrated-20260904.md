# C2 Mixture-UKF/APF Phase 7 Integrated Exact-DMIS Diagnostic

Date: 2026-09-04  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Prerequisite: Phase 5C-R contract-matched replication pass  
Status: `EXECUTED_VALIDITY_PASS_WITH_PROMOTION_VETO`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Research question and scope

Does the actual observation-conditioned per-ancestor UKF/APF call chain remain
valid when evaluated with the repository's exact complete-mixture DMIS
program, while a recursively rebuilt moment map and Hermite-plus-RBF fit are
attached as an explicitly identified guide/control diagnostic?

This phase addresses the practical failure mechanism directly: proposal
localization is tested through exact transition/observation weights, not by
trusting a fitted TT normalizer.  The hybrid fit is not used to generate
particles because the repository has no hybrid inverse-CDF sampler.  It is
therefore a control representation, not part of the proposal denominator and
not a claim-bearing combined algorithm.  This scope correction is required by
the implementation call-chain rule.

The phase does not establish posterior correctness, unbiased likelihoods,
HMC readiness, an analytical total derivative through adaptive proposal
construction, a default, or superiority on the C2 model or elsewhere.

## Finite program and candidate ladder

Use the frozen C2 fixture, exact model numerator, APF ancestor law, and
complete conditional mixture denominator already exercised in Phases 2--4.
For every realized branch the exact evaluator computes

\[
 \widetilde w_t =
 \frac{\bar w_{t-1,J} f_t(X\mid x_{t-1,J})g_t(y_t\mid X)}
 {a_{t,J} q_{t,J}(X)},
 \qquad
 q_{t,J}(x)=\sum_k\pi_{t,J,k}q_{t,J,k}(x),
\]

and the recursive analytical score of that same frozen finite scalar.  No
selected-component density is substituted for `q`.

Run three paired branches at `N=8192` over the full 20-observation horizon:

| Family | Role |
| --- | --- |
| `ukf_apf_k1` | observation-conditioned local baseline |
| `ukf_apf_k2`, `ukf_apf_k4` | fixed-topology local mixture hypotheses |
| `ukf_apf_k1_defensive`, `ukf_apf_k2_defensive`, `ukf_apf_k4_defensive` | smooth innovation-gated Student-defense hypotheses, using frozen `nu=5`, `epsilon_min=0.05`, `epsilon_max=0.20`, center `4`, temperature `8`, offset `0.5` |
| `bootstrap_conditional`, `transformed_student_nu8`, `gaussian_hint_marginal`, `stationary_independence` | constructed cheap heuristic adversaries |
| `retained_tt` | retained-TT representation comparator, not a cheap adversary |

The branch seeds are the first three declared Phase 4 claim seeds.  All
families share each branch seed and exact observations.  Proposal snapshots,
rows, ancestry, and labels are frozen before score evaluation.

## Recursive map and hybrid control

For the actual `ukf_apf_k1_defensive` branch, and as a paired local control for
`ukf_apf_k1`, reconstruct exact normalized parent weights from the realized
branch at map times \(t\in\{1,3,10,19\}\).  From those weights and the exact
transition moments construct

\[
 m_t^- = \sum_j w_{t-1,j}m_{t,j}^-,\qquad
 P_t^- = \sum_jw_{t-1,j}\left(Q_{t,j}+
 (m_{t,j}^- -m_t^-)(m_{t,j}^- -m_t^-)^{\mathsf T}\right),
\]

and the lower-Cholesky map \(x=m_t^-+L_tu\).  Fit the exact likelihood-
corrected target in this map with the audited 12-channel hybrid basis at
widths `0.75` and `1.50`, using disjoint `512/512/4096` train/holdout/audit
banks.  Compare the fit to an independent predictive-mixture `Z_T` at each
map time.  The map and fit are frozen controls; they do not alter the exact
proposal samples or DMIS denominator.

## Evidence contract

| Field | Declaration |
| --- | --- |
| Question | exact-DMIS validity and observation localization under a full-horizon branch, with recursive hybrid control diagnostics |
| Exact baseline | repository `FrozenProposalAPFProgram` with complete conditional mixture densities |
| Primary validity criterion | all 33 proposal records finite, complete, exact-score/parity-valid, GPU/XLA-valid, and all required proposal laws normalized |
| Recursive-control criterion | maps finite/SPD/round-trip valid and every attempted hybrid fit has analytic mass/cross quadrature checks plus a finite independent `Z_T` comparator |
| Heuristic role | conditional ESS/observation-response table; any loss vetoes candidate promotion but cannot veto a valid research direction |
| Uncertainty | three paired branches; intervals are descriptive only, no width or family ranking |
| Promotion criterion | none in this phase; a valid route is eligible only for a larger reviewed decision study |
| Hard vetoes | target/denominator mismatch, nonfinite exact score/value, incomplete records, failed XLA/GPU/memory provenance, invalid map/control, corrupted artifacts, or exhausted budget |
| Candidate failures | low ESS, heuristic dominance loss, poor hybrid shell fit, or ill-conditioned width when another route remains valid |
| Nonclaims | no posterior/unbiasedness, no adaptive-total-gradient, no hybrid-sampler claim, no default, no superiority, no general-model conclusion |
| Artifact | fresh `phase7-integrated-attempt01/` with manifest, 33 raw branch records, recursive control records, result, test capture, MathDevMCP/Lean sidecars, and close note |

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| `N=8192`, three branches | inherited serious row and bounded fresh replication | insufficient power for ranking | raw paired records and interval width | diagnostic design |
| full horizon 20 | C2 fixture horizon | long recursion exposes proposal/map instability | per-time ESS and exact finite flags | frozen scope |
| Phase 4 smooth controls | independent calibration result `nu5_eps05_20` | defensive tail dilutes bulk | family-wise ESS and heuristic table | candidate hypothesis |
| map times `{1,3,10,19}` | early/middle/terminal coverage under bounded fit cost | instability between checkpoints is missed | exact branch ESS remains full-horizon; checkpoint limitation recorded | diagnostic design |
| hybrid widths `.75,1.5` | both survived Phase 5C-R condition screen | width ranking remains noisy | mass condition and held-out/shell/`Z_T` records | comparator arms |
| `512/512/4096` control banks | contract-matched Phase 5C-R refresh | finite-bank error | disjoint hashes and predictive SE | frozen diagnostic |
| fixed ridge `1e-8`, rank 2, two sweeps | Phase 5C-R paired settings | underfit or ridge bias | fit/holdout/condition | frozen Class-C diagnostic |
| no hybrid sampling | current implementation capability | integrated proposal claim would be false | endpoint call-chain audit | explicit scope correction |

## Skeptical plan audit

Disposition before execution: `PASS_FOR_BOUNDED_PHASE7_INTEGRATED_DIAGNOSTIC`.

The plan uses the exact existing DMIS evaluator as the correctness authority,
keeps all cheap adversaries in the same paired branch, and does not promote
three-branch descriptive ESS differences.  The hybrid TT is deliberately
measured only as a guide/control because no sampler exists; the plan does not
silently substitute it into the proposal denominator.  Full-horizon exact
branches expose recursive weight collapse, while four map checkpoints bound
the additional fit cost.  A missing family, incomplete denominator, nonfinite
program, invalid map/control, or missing artifact is a continuation veto.  A
valid but inefficient candidate is preserved and classified as candidate
failure, after which the next planned repair remains eligible.

### Pre-mortem

| Misleading outcome | Distinguishing check | Action |
| --- | --- | --- |
| an observation-guided branch still behaves like a prior proposal | per-ancestor posterior/lookahead spread and conditional ESS | reject localization claim; retain exact-valid candidate only |
| DMIS appears valid because a selected component was used | complete-density recomposition and label permutation | hard veto and repair denominator |
| hybrid fit improves shell RMS but not the exact target | independent `Z_T`, shell residual, and mass condition | keep as control only; no proposal promotion |
| defensive Student tail hides a bad local proposal | compare local-only and defensive branches at the same times | report masking; do not call defensive route a repair |
| three branches create a spurious ranking | paired raw values and descriptive intervals | no ranking; require a larger decision study |
| map checkpoints miss an intermediate failure | full-horizon exact branch checks plus checkpoint limitation | do not claim continuous recursive map stability |
| dynamic loaders evade static audit | executable endpoint and family identity checks | use the driver wiring flag as call-chain evidence |

## Execution, repair, and close

Run one focused CPU regression and one GPU/XLA attempt, with at most two
localized harness repairs in fresh directories under the unchanged contract.
The first no-overwrite/preflight failure is preserved and does not consume a
scientific candidate result.  At close, classify engineering correctness,
numerical validity, and scientific interpretation separately.  If hard gates
pass, continue to the next planned reference or larger replicated decision
study even if every candidate loses the heuristic table.  Do not open the
Student TT reference route from this result; that route still requires a
coupled measure/basis/mass/row/gradient implementation plan.

Required close artifacts are the exact command, source hashes, dirty-worktree
identity, GPU/memory/XLA metadata, branch records, recursive-control records,
decision and inference tables, repair history, MathDevMCP/Lean limits,
focused-test output, and a post-run red-team note.

## Pre-execution call-chain repair and evidence

The executable call-chain audit found a material working-tree mismatch before
the serious run.  `c2_sv_frozen_proposal_apf_tf.py` and the Phase 4 diagnostic
already called the exact evaluator with generalized deterministic-mixture base
masses, but the active `PreparedFrozenProposalBranch` did not expose those
fields.  The preserved implementation at commit `b3eaa7a9` did.  The exact
single-file implementation was restored byte-for-byte from that preserved
source: normalized optional base-mass fields now enter branch identity and the
finite scalar is

\[
 \sum_t\log\sum_i b_{t,i}\exp\{\ell_{t,i}\},
\]

with the uniform Monte Carlo law recovered as (b_{t,i}=1/N).  A direct
nonuniform-mass recomposition test and uniform-special-case parity test were
added.  This repairs the harness/target contract; it does not alter any
proposal family, data, promotion rule, or scientific hypothesis.

The correct CPU-only focused command passed `60` tests with two dependency
deprecation warnings:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run \
  --no-capture-output -n tftwogpu python -m pytest -q \
  tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase0.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase1.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase3.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py \
  tests/highdim/test_recursive_moment_map_tf.py \
  tests/highdim/test_c2_hermite_basis.py \
  tests/highdim/test_rbf_basis_tf.py \
  tests/highdim/test_hybrid_basis_tf.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase5a.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase7.py
```

Two malformed preflight shell invocations produced only launcher/collection
errors and no numerical output; they are infrastructure mistakes, not campaign
attempts.  A first GPU probe omitted `CUDA_DEVICE_ORDER` and therefore selected
the RTX 5080.  The corrected full-shape probe used
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, selected the RTX 4080
SUPER, compiled the chunked target with XLA for `8192` parents and `4096` rows,
and returned a finite `[4096]` result.  The serious launch is therefore:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase7-integrated-attempt02 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_mixture_ukf_apf_phase7_integrated_20260904.py \
  --output-root \
  docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase7-integrated-attempt02
```

The skeptical disposition remains
`PASS_FOR_BOUNDED_PHASE7_INTEGRATED_DIAGNOSTIC`: the material call-chain flaw
has been repaired and directly regressed, the full-shape GPU/XLA kernel has
executed on the declared device, and no scientific boundary changed.

## Attempt 01 repair refresh

`phase7-integrated-attempt01` stopped before proposal fitting, sampling, or
evaluation.  Dynamic imports of the Phase 5 helper stack created a TensorFlow
logical GPU before the repository memory-growth helper ran, so the helper
correctly failed closed with `Physical devices cannot be modified after being
initialized`.  The directory contains only the launch-time plan and a failure
note; it is infrastructure evidence and has no scientific result.

The localized repair configures and verifies memory growth immediately after
TensorFlow import and before every dynamic scientific-module import.  The
focused CPU suite remains `60 passed`, and the already-completed full-shape
RTX 4080/XLA target probe directly covers the repaired ordering requirement.
The target, data, candidate ladder, random seeds, criteria, and total campaign
budget are unchanged.  The fresh retry output is
`phase7-integrated-attempt02`; one further localized harness repair remains in
the original budget.

## Execution close

Attempt02 completed on 2026-09-04 with status
`PASS_PHASE7_INTEGRATED_VALIDITY_WITH_PROMOTION_VETO`. All `33/33` exact
proposal records and `24/24` recursive-control cells passed the declared hard
validity checks. The focused suite reported `60 passed`; Lean exited `0`; and
the complete-DMIS MathDevMCP query returned `structural_match`.

The candidate-level result is negative. Every UKF family loses to a cheap
heuristic at at least one declared salient time, and all UKF variants reach
their worst mean ESS at time 14. The retained-TT branch is substantially worse,
with minimum ESS between `4.01` and `5.47` out of `8192`. The recursive maps
remain well conditioned, but the hybrid control has shell RMS as large as
`3.623`, an absolute log-normalizer gap as large as `3.171`, and no sampler.

The time-14 failure identifies a concrete mathematical mismatch. One
observation component is `0.00147071`, whose raw log square is `-13.044`. The
Gaussianized log-square UKF treats this as an extreme location innovation,
whereas the exact C2 likelihood has state score tending to `-1/2` and curvature
tending to zero as the observation tends to zero. The proposal is genuinely
data-responsive, but it responds incorrectly in this transformation-tail
regime. This is a candidate failure, not a denominator, map, or target failure.

The authoritative close is
`phase7-integrated-attempt02/phase7-close-20260904.md`; formal-tool limits and
executable evidence are in `phase7-integrated-attempt02/formal-audit.md`.
Continuation is `CONTINUE_NO_REAL_BLOCKER_TO_EXACT_LIKELIHOOD_GUIDE_PLAN`.
