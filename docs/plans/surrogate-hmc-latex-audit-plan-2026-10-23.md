# Surrogate-HMC LaTeX reconciliation, 23 October 2026

## Active checkpoint

Work on `main` in `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-document-main-sync-20261015`.
Starting revision: `4a1990d7d10ec25445ab675231e18e0419f9edf5` (clean and synchronized with origin).
Question: does the active monograph explain the implemented surrogate-HMC
algorithm and accurately report the completed five-model campaign?
Current stage: documentation reconciliation and technical checks complete.
This note accompanies the main documentation commit; Git history records publication.
Evidence: docs/plans/artifacts/surrogate-hmc-latex-audit-20261023-01.
No sampling or GPU budget was consumed. All 17 truths lie inside reported intervals
and physical-coordinate R-hat is below 1.01. SIR acceptance exceeds the requested
65–75% tuning band. No further experimental phase belongs to this documentation task.

## Scope and evidence contract

The reader knows state-space filtering and ordinary HMC and should be able to
reconstruct the LEDH force, independent endpoint correction, frozen NeuTra
change of variables, cumulative sampling rule, and interpretation of the five
completed runs without consulting operational notes for a mathematical premise.
The baseline is the Git revision above. Preserve existing derivations and source
qualifications unless checked code or evidence requires correction. Preserve
historical evidence as historical; do not relabel it as a current result.

The comparison is documentation versus the actual consumer call chains and
archived results, not one inference method versus another. Passing requires
consistent equations, target/coordinate/dtype descriptions, execution and
training descriptions, numerical tables reproducible from saved draws, working
cross-references, and a successful rendered-document review. Unsupported claims,
wrong scalar/score attribution, omitted Jacobians, mixed datasets, or mismatched
artifacts require repair before commit. ESS, MCSE, acceptance, and energy error
retain the explanatory roles agreed for this campaign. No new validation phase,
new posterior sampling, algorithm/default change, or method ranking is in scope.
One-data-set truth containment is not repeated-data coverage; convergence of a
fixed numerical target is not proof of exact posterior agreement.

## Planned work and assumption audit

1. Follow `docs/main.tex` inputs and inspect affected filtering, gradient,
   HMC, diagnostics, and NeuTra chapters. Audit consumers through their shared
   implementations, including shape and derivative conventions. Treat saved
   manifests and draws as numerical evidence; current code alone cannot establish
   historical run settings.
2. Reconcile algorithm and results in the active manuscript. Explain differences
   between proposal force and endpoint potential, precision of filter versus
   sampler coordinates, frozen map versus online adaptation, tuning acceptance
   versus convergence, and cumulative samples versus arbitrary windows.
3. Reproduce the existing saved-draw audit on CPU (explicit diagnostic exception),
   check documented numerical settings and call-chain wiring, and run only
   relevant existing tests if needed for claims. Write a bounded audit note with
   code/artifact anchors, findings, corrections, and untested boundaries.
4. Build the entire monograph using the README command, inspect changed rendered
   pages and build diagnostics, update the checkpoint, and commit/push main.

Material choices: the active document is `docs/main.tex` per README, not the
separate fable-rewrite manuscript; numerical results come from the final
23 October report and original evidence; canonical SIR IAF remains distinct
from the separate NAF training study; the user's campaign-specific stopping
rules override historical generic ESS/energy gates. Early checks are input-tree
search, artifact hashes, consumer tracing, and reproducing saved-draw summaries.
No transferred architecture, filter setting, or old result becomes a default.

## Skeptical review before edits

The main risks are adding a results appendix while leaving contradictory active
claims, calling approximate fixed-bootstrap endpoints exact, confusing the
surrogate force with the gradient of the MH potential, claiming full FP32 while
the map/sampler use FP64, treating ESS or high finite energy error as stopping
rules, and hiding SIR's acceptance above 75 percent. The audit explicitly checks
each. It also distinguishes the earlier marginal-weight/precision/performance
experiments from the completed HMC runs. The plan passes this review: archived
evidence and local code suffice, no comparative scientific promotion or new
experiment is proposed, and failures trigger documentation repair or a precise
limitation. Human assessment of prose remains pending; compilation and self-review
establish technical consistency, not a certified reader response.

## Execution record

### Code and mathematical reconciliation

The active document now includes Chapter 56, The Completed Surrogate-HMC Case
Study. Filtering, covariance-proposal, HMC-target, diagnostics and NeuTra
chapters link to it and distinguish historical snapshots from the completed
October campaign. Derivations are preserved; reset ordering and implementation
descriptions are corrected where needed. Four added citations were checked
against local paper copies.

| Checked implementation | Correction or qualification | Evidence relative to repository root |
|---|---|---|
| Five-model consumer to native filter | One batch-native recursion carries all parameter sensitivities; directional reference is not the training lane | bayesfilter/testing/surrogate_hmc_five_model.py:317; bayesfilter/highdim/covariance_proposal_score_batch_tf.py:208; bayesfilter/highdim/covariance_proposal_native_models_tf.py:17; all-model call-chain tests |
| Proposal and importance weights | Common-Q predictive mixture moments and full marginal numerator, distinct from ancestor-conditional weights | bayesfilter/highdim/covariance_proposal_score_tf.py:94 and :149 |
| Contract E reset | Weighted source moments, finite transport, unweighted transported covariance, Cholesky recolouring, recentering, and total sensitivities in execution order | bayesfilter/highdim/covariance_proposal_score_batch_tf.py:197 and shared reset_cloud |
| Force and endpoint correction | Finite LEDH negative-log-posterior gradient guides proposals; endpoint is separate. Frozen position-only kicks and drift are reversible and volume preserving | bayesfilter/inference/neural_force_hmc.py:374, :487, :641; focused tests |
| Frozen SIR map | Endpoint log Jacobian and force Jacobian term included; canonical IAF distinguished from separate NAF study | bayesfilter/testing/surrogate_hmc_sir_neutra.py:106; map tests and training payload |
| Sequential sampling | Warm-up excluded; retained draws cumulative; no extra validation phase. ESS and finite energy error retain diagnostic roles | bayesfilter/inference/hmc_sequential_verification.py:37; completion manifests |
| Target and precision | Exact Kalman endpoint only for LGSSM; four nonlinear endpoints are numerical approximations. FP32 filter distinguished from FP64 sampler/map | Archived manifests and target bindings |

Archived source revisions define tested settings. Main advanced through
concurrent guidance merge 90515228c and planning commit 463ecd83a during this
task. Focused and all-model call-chain tests were repeated afterward. Later
optional guidance variants are expressly outside the archived HMC evidence.
This documentation commit makes no runtime or numerical-default change.

### Numerical and executable checks

Recomputed retained-draw physical-coordinate intervals and modern R-hat using
scripts/audit_surrogate_hmc_five_model_posterior.py on CPU. The saved report
preserves input hashes, exact command, environment, data provenance, source
revision and wall time. All 17 rows reproduce the archived report. This is
post-run inspection, not new sampling or an added validation phase.

Focused tests: **54 passed** in 41.08 seconds. Native five-model XLA/parity tests:
**5 passed**, 14 deselected, in 55.09 seconds. Each run emitted two TFP distutils
deprecation warnings. CUDA was intentionally hidden. An initial route-ledger
check lacked a tracked sparse-checkout file; restoring that exact file resolved
it without a code repair or relaxed assertion.

Exact commands from the main worktree:

    env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q -p no:cacheprovider tests/test_surrogate_hmc_sir_neutra.py tests/test_surrogate_hmc_literature_priors.py tests/test_neural_force_hmc.py tests/test_neutra_hmc_route_policy.py tests/test_surrogate_hmc_ksc_finite_endpoint.py
    env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q -p no:cacheprovider tests/highdim/test_covariance_proposal_native_batch.py -k 'full_native_xla_recurrence and 0'

The native check uses N=32, T=2, B=2, correction=0, every parameter direction
and batch permutation. This checks consumer parity, not new large-particle
accuracy or scientific superiority.

### Full-document build and rendered review

With latexmk unavailable, built from docs using pdflatex -interaction=nonstopmode
-halt-on-error main.tex, bibtex main, and successive pdflatex passes until
references and pagination stabilized. Five tracked image dependencies missing
from the sparse checkout were restored. Final PDF: **773 pages**, no undefined
references/citations and no rerun warnings. No overfull warning remains in the
revised current-filter section or new results chapter. The whole book retains
221 overfull warnings elsewhere; typography in every historical chapter was
not repaired in this task.

Rendered inspection covered equations, endpoint/force distinction, priors,
settings, tables, SIR training and interpretation. Final physical PDF pages
302, 705 and 706 were inspected after pagination stabilized; the new chapter
occupies physical pages 701–707. Full build log and PDF hash are preserved.
The generated PDF is ignored by Git; sources are tracked. Git diff --check
passed. No new GPU run was made.

### Decision and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Publish reconciled manuscript on main | Equations, call chains, settings and tables agree within scope | No unresolved documentation mismatch or failed focused test; build references resolve | Nonlinear endpoint approximation, one dataset per model | Commit and push sources and audit evidence | Exact nonlinear posterior accuracy or superiority |
| Preserve completed HMC classification | All physical-coordinate R-hat below 1.01; recorded health checks pass | No recorded divergence; one declared range-bearing force-fallback event | SIR pooled acceptance 77.7%, above requested band | State tuning limitation without a new sampling phase | All tuning goals met, or acceptance proves convergence |

| Inference status | Finding |
|---|---|
| Hard veto screen | No unsupported scalar/score identity, omitted map Jacobian, mixed draw source, unresolved reference or failed focused test remains in revised material |
| Statistically supported ranking | None claimed; no new comparative experiment |
| Descriptive-only differences | Acceptance, ESS/MCSE, finite energy error, one-dataset truth containment and runtime |
| Default-readiness | No new default; settings remain scope-specific |
| Next evidence needed | Repeated datasets and refined nonlinear endpoints address coverage/approximation, outside this documentation task |

The strongest alternative explanation for good nonlinear chain diagnostics is
convergence to a biased but fixed endpoint. The manuscript states this boundary.
A refined endpoint materially changing the posterior would overturn an accuracy
interpretation, but not the saved-draw R-hat calculation. Exact nonlinear
posterior accuracy remains unestablished. Compilation and self-review do not
certify a human reader's experience.
