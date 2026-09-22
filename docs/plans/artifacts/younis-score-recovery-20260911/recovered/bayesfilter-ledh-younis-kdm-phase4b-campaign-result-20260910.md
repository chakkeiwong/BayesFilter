# Phase 4B matrix campaign: no score improvement

Date: 2026-09-10
Branch: `kdm-total-score-continuation-20260909`
Executed commit: `69ca470c526ae18c7c45e6371bbaa743867300d7`
Plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4b-campaign-amendment-20260909.md`
Receipt root: `docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/campaign01/`

The calibrated raw-IWSG KDM candidate has higher model-score MSE than the
canonical filter in both tested scopes. The paired intervals support a loss,
including in both score groups at the longer horizon. The experiment completed
without a numerical or implementation veto. This rejects the tested candidates
for promotion; it does not show that every possible KDM construction fails.

## Executed comparison

The two-state model and every finite numerical control are frozen in the
amendment. Each scope uses 40 calibration paths, 30 disjoint power-pilot paths,
and 500 untouched validation paths. Calibration evaluates all eight bandwidths
and both mark policies, and separately calibrates the observation-KDM arm.
The canonical comparator calls the public analytical endpoint with Contract-E,
GenUT diagonal and pairwise correction, and both caps. The exact score comes
from an independent matrix Kalman recursion. The bootstrap comparator is the
explicit fixed-stream finite-program derivative, with realized ancestry; it
is not an unbiased model-score oracle.

All numbers below are score errors relative to the exact Kalman score.
Intervals use the predeclared 4,000 paired path bootstrap resamples and are
pointwise 95% intervals, not simultaneous statements over every reported cell.

| Scope (N,T) | Selected rho | Canonical MSE | Phase 4A MSE | Bootstrap MSE | Phase 4B MSE | Phase 4B minus canonical MSE, 95% interval |
|---|---:|---:|---:|---:|---:|---|
| (32,5) | 0.8 | 3.94210 | 3.94279 | 13.46790 | 4.45250 | 0.51041 [0.07106, 0.94986] |
| (64,20) | 0.4 | 3.86260 | 3.86331 | 39.69324 | 10.13463 | 6.27203 [5.10083, 7.55881] |

Both Phase 4A selections are rho=0.025. Both saved Phase 4B selections name
the fixed-label mark. That mark selection is a numerical tie: in this model
the two mark policies are mathematically equivalent, as explained below.

For the actual ten-percent promotion criterion, the interval for
`mean(e_B^2 - 0.9 e_A^2)` is [0.48679,1.30873] in the small scope and
[5.45567,7.95740] in the larger scope. Neither approaches the required
negative upper endpoint. Pilot power calculations requested 8,900 and 4,160
paths, respectively, for the declared 20% improvement alternative. The
500-path cap makes both runs underpowered for that design and independently
prohibits promotion. This does not erase the observed intervals supporting
losses; it prevents treating the campaign as an adequately powered search for
the stipulated improvement.

| Scope/group | Paths | Canonical MSE | Phase 4A MSE | Bootstrap MSE | Phase 4B MSE | B minus canonical, 95% interval |
|---|---:|---:|---:|---:|---:|---|
| (32,5), low absolute exact score | 330 | 3.43538 | 3.43584 | 7.57207 | 3.87226 | [0.03866,0.84390] |
| (32,5), high absolute exact score | 170 | 4.92573 | 4.92686 | 24.91275 | 5.57887 | [-0.44692,1.63994] |
| (64,20), low absolute exact score | 225 | 3.69549 | 3.69658 | 28.79102 | 8.62420 | [3.45045,6.53050] |
| (64,20), high absolute exact score | 275 | 3.99932 | 3.99974 | 48.61324 | 11.37043 | [5.60295,9.32356] |

The high-score small-scope difference is descriptive only. Every group has
higher Phase 4B point MSE than the canonical and observation-KDM comparators,
which fires the predeclared conservative heuristic veto. Phase 4B has lower
MSE than the bootstrap comparator with pointwise intervals excluding zero;
that comparison cannot rescue its loss to the unmodified filter.

## What explains the loss, and what remains uncertain

| Scope | Canonical error bias / sample variance | Phase 4B error bias / sample variance |
|---|---|---|
| (32,5) | -1.08143 / 2.77816 | -1.32100 / 2.71290 |
| (64,20) | -0.98767 / 2.89289 | -1.57710 / 7.66270 |

These decompositions are descriptive. In the small scope, the small observed
variance reduction is offset by increased squared bias. In the larger scope,
both observed squared bias and variance increase. The candidate's derivative
is the derivative of its declared fixed-anchor finite program, not of the
unchanged canonical value and not of the exact model likelihood. Correct
derivative mechanics therefore cannot establish lower model-score error.

The bandwidth curve makes the small-bandwidth hazard concrete. Calibration
MSE for rho=0.025 is 1.63e9 at (32,5) and 9.18e35 at (64,20); at rho=0.1 it
is 3.12e4 and 3.59e11. All these evaluations remained finite and passed the
returned validity checks; no gradient was clipped. The note's derived local
`1/h` IWSG tangent explains why small forward perturbations need not produce
small derivative perturbations. This is an explanation consistent with the
curve, not a proof of the exact rate of amplification through every reset.
The complete calibration curves remain in the selection receipts. They must
not be reinterpreted as untouched validation of every bandwidth.

Value changes are also explicit. The mean Phase 4B-minus-canonical log-value
shift is -0.02449 and -0.02377; the corresponding mean squared shifts are
0.07284 and 0.04023. Small mean value shifts do not establish score parity.
Mean squared value errors against Kalman are 0.72716 versus canonical 0.67489
and 0.46817 versus canonical 0.43364. These are explanatory diagnostics.

The fixed numerical controls were not independently tuned for either scope.
This comparison answers whether KDM helps this explicit finite baseline; it
does not identify the best tuned canonical method or the best possible KDM
method. The two scopes change N and T together, so their difference cannot
identify a separate horizon effect or particle-count convergence rate.

## Additional mathematical and code findings

1. **The covariance-mark ablation is uninformative in this fixture.** Linear
   prediction/update, equal initial P0 marks, and normalized reset transport
   preserve equal marks. Hence both `sum_i gamma_ji P_i` and `P_Ij` equal P,
   and their total derivatives agree because `sum_i d gamma_ji=0`. Maximum
   relative paired calibration score differences were 2.92e-13 and 5.70e-12.
   A new actual-endpoint test confirms this equality. Existing tests with
   heterogeneous initial marks separately verify that both implemented rules
   can differ and still match their own finite derivatives. This campaign
   cannot choose a mark policy for nonlinear models.
2. **Moving DSGE support needs more than a Jacobian term.** The new exact
   counterexample `X(theta,U)=(U,theta U)` has an observation score of 1/2 at
   theta=0,y=(1,1), although the volume-factor derivative is zero. A frozen
   ambient sample leaves the support at perturbed theta, so the anchored
   state-law ratio is inadmissible. Fixing innovation coordinates instead
   retains a state-motion derivative. The derivation and next code boundary
   are in `docs/plans/bayesfilter-ledh-younis-kdm-moving-support-check-20260910.md`.
   Two CPU tests verify the exact score and the existing subspace kernel's
   correct rejection. This exposes an incomplete DSGE specification; it does
   not invalidate the full-rank Phase 4B replay derivative.
3. **The full DSGE call chain remains absent.** The current shared endpoint
   factors a full-rank process covariance and uses full-rank Gaussian PF-PF
   factors. Replacing only the KDM helper cannot provide the required
   innovation-coordinate flow, density, reset, and chart derivatives. A
   multi-parameter Phase 4B assembly and the broader innovation-jitter and
   justified control-variate comparisons also remain incomplete.

The LaTeX note is updated with the measured result, equal-mark limitation,
and moving-support counterexample. No numerical source was changed during
the campaign. The raw-row audit verified launch source hashes before these
post-run document edits.

## Execution, integrity, and validation

The actual command, from the isolated worktree, was:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true timeout --signal=TERM --kill-after=30s 2700s python docs/benchmarks/run_ledh_younis_kdm_phase4b_campaign.py --mode campaign --output-dir docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/campaign01 --attempt 1 --budget-seconds 2700
```

It used trusted/escalated access to the RTX 4080 SUPER, `tftwogpu`, TensorFlow
2.20.0-dev0+selfbuilt, float64, XLA on, TF32 off, and verified memory growth.
The synthetic data version is the frozen model plus the stateless seed scheme
in `manifest.json`; there is no external dataset. Wall time was 1597.16
seconds (26.62 minutes), leaving 1102.84 seconds of the 2700-second ceiling.
No repair retry was needed. Peak TensorFlow GPU allocator usage was
16,926,720 bytes. Observed host RSS reached about 18.2 GB; this is a sampled
observation, not a measured peak.

The first calibration path's endpoint calls took 126.64 seconds and 892.82
seconds for the two scopes, compared with median warm calibration-path totals
of 1.04 and 4.18 seconds. Those cold-call costs include tracing, compilation,
and execution; they are not pure GPU-kernel time. A read-only Python stack
probe failed due to OS ptrace permissions and produced no diagnosis. It did
not block the successful run.

The terminal audit recomputed the score summaries from 1,140 unique path rows
and checked 7,380 retained method evaluations, including every validity flag,
full-mixture pair count, and full correction diagnostic. Split counts and
launch source hashes matched. Raw rows remain ignored and preserved locally;
their SHA-256 hashes, compact selection/summary/manifest receipts, and the
audit result are retained in Git.

```sh
timeout 120s python docs/benchmarks/analyze_ledh_younis_kdm_phase4b_campaign.py docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/campaign01
CUDA_VISIBLE_DEVICES=-1 timeout 120s python -m pytest -q tests/highdim/test_ledh_younis_kdm_phase4b_campaign.py::test_homogeneous_matrix_fixture_cannot_rank_covariance_mark_policies
CUDA_VISIBLE_DEVICES=-1 timeout 120s python -m pytest -q tests/highdim/test_ledh_younis_kdm_moving_support_reference.py
```

The audit passed. The focused CPU tests passed (one in 12.46 seconds, two in
3.48 seconds); CUDA was intentionally hidden. Prior campaign/preflight and
GPU invalid-input checks are recorded in the separate continuation result.
No broad suite was repeated after these analysis/document-only additions.

After the document changes, the same analysis passed again with
`--output-name audit_after_document_update_result.json`. It resolves changed
historical documents from the recorded launch commit and checks their hashes;
it does not confuse the amended manuscript with the version used at launch.
The revised LaTeX builds to 17 pages with no unresolved citations/references
or overfull boxes. Comparison with the protected September 9 source found no
removed equation, citation, or source boundary. Pages 10 and 12--14 were
inspected for the added mark identity, moving-support derivation, and results.
The build and visual inspection check presentation, not universal mathematical
or human-reader acceptance.

## Decision and next justified work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| No promotion of either calibrated candidate | Ten-percent MSE criterion fails in both scopes | Conditional heuristic losses and insufficient planned power; no execution-validity veto | Finite-control tuning, broader N/T behavior, nonlinear support and mark transport | Complete the innovation-coordinate mathematical specification and its actual call chain; require fresh scope-specific calibration before another score campaign | Rejection of all KDM ideas, equality with the canonical score, DSGE/HMC/default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | All retained numerical/route validity checks passed; both candidates fail promotion conditions. |
| Statistically supported ranking | Pointwise paired intervals support higher Phase 4B MSE than canonical and Phase 4A overall in both scopes; lower MSE than bootstrap. |
| Descriptive-only differences | Bias/variance decomposition, value shifts, bandwidth calibration curve, runtime; high-score small-scope KDM-versus-canonical difference. |
| Default-readiness | False. Reference campaign only; baseline controls unpromoted, TF32 not covered, no DSGE or multi-parameter endpoint. |
| Next evidence needed | A fully specified support-preserving algorithm, executable heterogeneous-mark and multi-parameter checks, then fresh tuning and a prospectively budgeted oracle comparison. |

Post-run red team: the strongest alternative explanation is that the frozen
finite controls and kernel geometry are a poor combination, rather than all
continuous-mixture approaches being unsuitable. Fresh, independently tuned
controls and an untouched comparison could overturn the scope-specific
failure. The weakest evidence is any extrapolation beyond these two full-rank
Gaussian scopes; the equal-mark ablation and absent power explicitly prevent
such extrapolation. More repetitions of these rejected settings would improve
precision but would not repair the support definition or the added bias.

The bounded Phase 4B campaign is finished. The master program is not finished:
Phase 5's moving-support mathematics has begun, while the full DSGE
implementation and subsequent validation remain open. There is no port or
promotion from this result.
