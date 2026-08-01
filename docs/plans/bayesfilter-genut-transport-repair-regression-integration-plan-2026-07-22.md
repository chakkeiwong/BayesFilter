# GenUT Transport Repair, Regression, And Integration Plan

Date: 2026-07-22
Status: `COMPLETED_WITH_CANDIDATE_NOT_ADMITTED`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | Does correcting the finite transport barycentric map, adding terminal balancing, and enforcing reset validity remove the STR-UKF non-finite failure without regressing the previously tested LGSSM, fresh exact-SV, predator-prey, or canonical Austria-SIR suite routes? |
| Candidate | Non-fused TensorFlow/XLA FP32/TF32 GenUT/Cubature route with realized row quotient, fixed terminal balance, manual recursive score, and Contract-E validity gates. |
| Structural target | Existing Chapter 18b STR-UKF target, `N=1002`, `T=100`, scalar innovation, initial observation before transitions. |
| Particle regressions | LGSSM (`T=2,10,50`), fresh exact transformed SV (`T=50`), and predator-prey (`T=20`), all with `N=1002` or a larger exact-design count and the same corrected transport core. |
| SIR regression | Canonical fixed Austria SIR SGQF value-only route, `zhao_cui_spatial_sir_austria_j9_T20`; this is not a GenUT route and has no free parameter score. The artificial reduced SIR fixture remains permanently excluded. |
| Baselines | Prior non-revoked artifacts for each target, preserving their original controls, particle counts, seeds, and route roles. Revoked original-SV and reduced-SIR artifacts are not baselines. |
| Primary promotion criterion | Every fresh GenUT row is finite, transport marginals and reset factors pass, manual score accounting passes, and the structural target has no non-finite state at `T=100`. |
| Regression criterion | For each previously finite route, the corrected implementation remains finite and preserves its declared residual/score accounting gates. Numerical differences from the old finite scalar are descriptive because the row quotient and terminal balance define a new finite program. |
| Hard vetoes | Undefined harness names; CPU fallback for a serious GPU run; missing memory growth; non-XLA candidate path; non-finite tensors; non-positive row mass; invalid post-quotient column marginal; invalid covariance gap/factor; structural residual failure; score accounting failure; stale route identity; `N<=1000` for particle regressions; reuse of consumed claim seeds. |
| Explanatory diagnostics | Value/score deltas versus prior artifacts, per-time marginal residuals, covariance-gap eigenvalues, tangent maxima, ESS, runtime, allocator peak, and finite-difference checks at representative points. They do not establish exactness or superiority. |
| Nonclaims | No exact nonlinear likelihood, unbiasedness, method superiority, HMC readiness, default promotion, NAWM result, or leaderboard admission follows from this campaign alone. |

## Scope And Baseline Audit

The repository is intentionally dirty from the preceding research batch. No
unrelated change is reverted. The current branch is `main`, one local commit is
ahead of `origin/main`, and the final integration phase will commit the complete
authorized working-tree batch, merge `origin/main` without rewriting history,
and push the resulting `main` branch.

The reduced continuous preclip SIR/J=1 fixture is explicitly ineligible under
`docs/plans/bayesfilter-genut-actual-model-suite-correction-2026-07-22.md` and
cannot be used as the SIR regression. The actual Austria SIR route has a fixed
parameter and is therefore a value-only SGQF regression, not a GenUT score test.

The original direct-iid-normal SV fixture is revoked. The eligible SV baseline
is the fresh SV-DGP arm in
`docs/plans/bayesfilter-exact-sv-fixed-gaussian-genut-paired-comparison-result-2026-07-21.md`.

## Mathematical Repair

### Transport quotient

Given finite coupling `pi`, compute

```text
mass_i       = sum_j pi[i,j]
numerator_i  = sum_j pi[i,j] * x_j
barycenter_i = numerator_i / mass_i
```

and the total JVP

```text
d_barycenter_i = (d_numerator_i - barycenter_i*d_mass_i) / mass_i.
```

The runtime must never substitute `N*numerator_i` unless the realized row mass
has independently passed the exact row-mass gate.

### Terminal balancing

Add a fixed, scope-bound `balance_steps` control after the main finite Sinkhorn
updates. Use the repository terminal-epsilon IPFP mechanism and differentiate
all fixed iterations with the same manual JVP. The candidate remains non-fused;
this is a transport-state repair, not a fused optimization route.

### Validity contract

Return and enforce:

- finite and strictly positive row masses;
- post-quotient column marginal error;
- finite barycenters and tangents;
- covariance-gap minimum eigenvalue plus ridge;
- finite Cholesky factors with positive diagonals; and
- a single `reset_valid` flag consumed by the recursive filter.

Invalid rows must fail closed and never become recursive particle state.

## Phases

### Phase 0: Preserve State And Establish Plan

1. Record branch, remote, commit, worktree paths, and all baseline artifact
   hashes in the run manifest.
2. Preserve the existing STR-UKF root-cause artifacts as diagnostic evidence.
3. Do not stage or delete revoked artifacts as active baselines.

### Phase 1: Implement The Generic Repair

1. Add dense row-quotient forward/JVP kernels to
   `bayesfilter/highdim/cubature_genut_filter.py`.
2. Add fixed terminal balance and expose `balance_steps` in the generic route.
3. Add marginal-validity, row-mass, covariance-gap, and factor-validity
   diagnostics to the candidate result.
4. Preserve the manual recursive score and the no-NumPy/no-Python-sample-loop
   XLA contract.
5. Repair the STR-UKF runner's non-finite serialization and shared fail-closed
   predicate.
6. Bind the new control family and implementation dependency closure into route
   identity. Existing old-route artifacts remain historical.

### Phase 2: Focused Correctness Tests

Run CPU-hidden tests first:

- intentionally imbalanced coupling: row quotient remains in the source convex
  hull while the old `N*numerator` formula does not;
- row-quotient forward/JVP parity with an independent TensorFlow diagnostic;
- terminal-balance marginal repair and JVP parity;
- covariance decomposition on an exact positive coupling;
- invalid mass, invalid gap, and invalid factor fail-closed tests;
- structural transition residual, timing, scalar-noise, and recursive-score tests;
- no NumPy/autodiff/finite-difference runtime-path checks.

Then run a trusted GPU/XLA smoke at `N=1002` for the generic d=1, d=2, and d=3
designs.

### Phase 3: Structural Repair And Claim Replay

1. Tune `epsilon`, main `sinkhorn_steps`, `balance_steps`, and ridge on disjoint
   full-horizon structural calibration/validation data. Candidate grid:
   `epsilon={2,4}`, `sinkhorn_steps={4,8}`, `balance_steps={4,8,16,32}`, and
   `ridge={1e-6,1e-5}`.
2. Reject any arm with extreme validation score instability, not merely NaNs.
   Use at least four common particle seeds per stress DGP and include the
   persistence/nonlinearity validation point that previously produced source
   `phi=0.868`, `gamma=0.994`, `R=0.0745`.
3. Use representative FD audits over multiple calibration and stress points;
   FD remains diagnostic only.
4. Run the untouched structural claim with fresh seeds, `N=1002,T=100`, and
   report raw per-seed value/score rows and 95% intervals only if all rows pass
   hard validity gates.

### Phase 4: Particle-Method Regression Ladder

Run the corrected route and compare against prior non-revoked artifacts:

| Model | Scope | Fresh run | Prior baseline |
|---|---|---|---|
| LGSSM | `N=1002`, `T=2,10,50`, FP32/TF32/XLA, common seeds | value and recursive score | `lgssm_cubature_genut_tuned_claim_20260721_attempt2` and comparable-metric artifact |
| Fresh exact transformed SV | `N=1002`, `T=50`, at least 16 common seeds where runtime permits | value and score | fresh-DGP section of fixed Gaussian GenUT paired artifact |
| Predator-prey | `N=1002`, `T=20`, common 16-seed claim set | value and score | `genut_predator_prey_leaderboard_continuation_20260722/attempt01` |
| Actual Austria SIR | fixed source-order SGQF, canonical `T=20` route, CPU-XLA and GPU-XLA | value-only route health and CPU/GPU parity | `sgqf_whole_highdim_leaderboard_repair_20260722/attempt02/fixed-sir/gpu/result.json` |

The SIR row is reported as an independent route regression. It must not be
described as evidence that GenUT works for actual Austria SIR.

### Phase 5: Comparison And Structural Estimation Report

Emit one JSON/Markdown report containing, for each eligible model:

- prior and modified controls;
- particle count, horizon, seeds, device, dtype, TF32, XLA, and memory policy;
- prior and modified value means/SD/95% intervals;
- prior and modified score means/SD/95% intervals when a score exists;
- paired per-seed deltas where seeds are common;
- hard-veto status, descriptive differences, and nonclaims; and
- structural model estimated value/score, true physical parameters, and the
  same-target UKF diagnostic, clearly not labelled an oracle.

No partial summary is emitted for a hard-invalid candidate.

### Phase 6: Review, Commit, Merge, Push

1. Run focused tests and `git diff --check`.
2. Review the plan and result artifacts for stale baselines, hidden defaults,
   proxy promotion, and accidental inclusion of revoked fixtures.
3. Stage the complete authorized worktree batch, commit with a descriptive
   message, fetch `origin`, merge `origin/main` with a non-interactive merge,
   rerun focused tests after the merge, and push `main`.
4. Record the final commit, merge commit, remote status, and result artifact
   hashes.

## Compute And Stop Conditions

- GPU runs use verified memory growth, FP32, TF32, and XLA.
- Particle runs use `N>1000`; no `N=96` result is used as a regression claim.
- Total fresh campaign budget: 30 minutes GPU time, excluding short tests.
- Stop a candidate arm immediately on non-finite state, invalid marginal,
  invalid covariance factor, or memory peak above 12 GiB.
- A failed candidate blocks that candidate only; it does not reject the repair
  direction unless the validity harness is itself invalid or the budget is
  exhausted.

## Skeptical Plan Audit

| Risk | Resolution |
|---|---|
| Wrong SIR baseline | Reduced SIR is excluded; actual Austria SIR is fixed SGQF value-only. |
| Wrong SV baseline | Original non-DGP fixture is excluded; fresh SV-DGP only. |
| New scalar differs from old scalar | Explicitly recorded; old values remain historical, and modified values are not claimed equivalent. |
| Row quotient alone hides column drift | Post-quotient column marginal is a hard validity gate. |
| Larger ridge masks the problem | PSD gap is checked before accepting ridge; large negative gaps veto. |
| Tuning selects unstable arm | Stress DGPs, four-plus seeds, and explicit score-variance adequacy veto. |
| Regression report ranks methods | Report paired deltas descriptively unless uncertainty supports a ranking. |
| Dirty worktree loss | No reset/checkout; all existing changes preserved and integrated only after review. |
| Remote merge changes evidence | Re-run focused tests after merge and record final commit/artifact hashes. |

Audit verdict: `PASS_FOR_EXECUTION_WITH_SCOPE_CORRECTIONS`.

## Execution Revision After Workload Audit

The original structural grid would have multiplied the old two-particle-seed
tuning program by the new balance-count axis.  The active execution keeps the
complete declared four-control grid but uses a staged interpretation:

- each full-horizon arm uses four common particle seeds on each disjoint
  calibration and validation DGP;
- engineering-invalid arms remain in the artifact but cannot receive an FD
  audit or be selected;
- ordered eligible arms receive representative FD audits on both a calibration
  and validation DGP until one passes; and
- the final claim uses fresh seeds `2026072301..2026072308`; both prior claim
  sets are recorded as consumed and are not reused.

The trusted pre-run probe found one RTX 4080 SUPER, verified memory growth,
TF32 enabled, and a logical TensorFlow GPU.  The old eight-arm structural
tuning diagnostic took about one minute, so this staged 32-arm program remains
inside the 30-minute campaign budget.  This revision does not change the
target, comparator, criteria, hardware class, or scientific nonclaims.

Execution-revision audit verdict: `PASS_FOR_EXECUTION`.

### Attempt Ledger

| Attempt | Classification | Repair | Wall time / budget effect |
|---|---|---|---|
| `structural_tuned_claim_attempt01` | Harness serialization failure: invalid tuning rows reached `statistics.variance` as `None` values | Preserve invalid raw rows, assign no variance objective, and keep the arm ineligible | Failed during the first compiled arm; localized retry remains inside the campaign budget |
| `model_regressions_attempt01` | Harness initialization failure: candidate imports initialized the GPU before memory-growth configuration | Configure memory growth immediately after TensorFlow import, before candidate imports | Failed before model evaluation; no claim artifact written |
