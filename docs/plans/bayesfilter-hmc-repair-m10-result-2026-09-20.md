# M10 metric preparation diagnosis and repair

Status: complete, including terminal boundary replay and reconciliation.
The active program is the
[master](bayesfilter-hmc-repair-master-program-2026-09-16.md). Outputs are under
`artifacts/hmc-repair-master-2026-09-16/m10-r1/` (R below).

The preparation repair now reports why each metric proposal was accepted or
rejected, removes obsolete R-hat vetoes, preserves temporal axes, and offers
an optional finite-window covariance policy. One of three fresh no-hint
regression fits passed the complete declared posterior/reference screen.
The other two failed at different stages. This closes the bounded M10
engineering study, not the broader reliability problem.

## Corrected diagnosis and implementation

M9's interleaving diagnosis was wrong for the actual ordinary preparation.
The helper starts from one vector, runs one chain with draw/coordinate arrays,
and constructs the four-chain candidate bank afterwards. Its previous reshape
did not interleave chains. The explicit-chain covariance helper did have
obsolete R-hat gates; these are removed, with unavailable or erroneous R-hat
remaining reporting-only. Executable call-chain tests check the actual shape.
The correction is recorded in the M9 note and both current tuning guides.

The two frozen-source baseline reconstructions made zero metric updates.
For this six-dimensional target, the standard preset's 150 transitions have
slow windows of 30, 60 and 30, all below the dense minimum of 64. Temporal ESS
also failed its inherited adequacy floors. The serious preset supplies
200/400/200-state windows. Without a geometry hint they pass the dense count,
rank, condition, shrinkage and positive-definiteness checks, but fail its ESS
floor: minimum ESS is 5.998, 3.864 and 4.584 versus eight. The matched serious
supplied-Hessian comparator applies two metric updates. These observations
identify particular proposal screens; they do not prove a covariance formula
error or general posterior adequacy.

`metric_evidence_policy="finite_window"` keeps the same N-1 centered pooled
covariance with correlation shrinkage, but treats temporal ESS as explanatory.
The default remains `"temporal_information"`. Both policies require finite
within-chain variation in every coordinate, count/rank/condition checks,
appropriate discrepancy and positive-definiteness checks, affine value/score
parity, and a fresh reasonable-step probe before applying a proposal. The
option is serialized through every ordinary preparation configuration and
cannot be combined with fixed identity. Historical readers accept both new
payloads and old payloads without the field.

For a fixed nonsingular affine factor, exact-score Metropolis-adjusted HMC
preserves the transformed target even when the factor is not the posterior
covariance factor. That justifies investigating early covariance proposals;
it does not establish their numerical reliability or exploration. The master
gives the derivation and the inspected, pinned Stan covariance-adaptation
source. Stan's regularization differs; this is not an implementation-equivalence
claim. All adaptation ends before candidate measurement/verification and
posterior sampling.

Ordinary progress now records schedule capacity and every completed window's
decision before the next window can fail. Rejection records identify transform,
affine-parity or step-size qualification stages. The final diagnostic revision
also saves a rejected step probe's starting epsilon and available attempts.
Separate fields report temporal-information sufficiency and overall proposal
checks, avoiding misleading reports when another check fails. P4-E's existing
redaction remains intact.

## Numerical results

| Attempt | Empirical metric updates | Result |
| --- | ---: | --- |
| Original no-hint standard replay | 0 | Baseline reproduced; count and ESS vetoes |
| Original hinted standard replay | 0 | Baseline reproduced; count and ESS vetoes |
| No-hint serious, default policy | 0 | Dense ESS vetoes despite adequate counts |
| Hinted serious, default policy | 2 | Preparation passed |
| Matched no-hint serious, optional policy | 2 | Later warmup trace nonfinite; preparation failed |
| Unchanged standard replay on repaired source | 0 | All five saved latent tensors, final factor and epsilon match the baseline exactly |
| Fresh optional fit 0 | 0 | All proposed metrics rejected at the step-size boundary; tuning retains 24 members; selected posterior warmup hits its cap |
| Fresh optional fit 1 | 2 | Later warmup trace nonfinite; preparation failed before candidate search |
| Fresh optional fit 2 | 1 | Tuning retains 26 members; selected posterior and reference screens pass |

Fresh fit 0 reaches 10000 warmup transitions per chain, with final-window
maximum modern R-hat 3.23275, and produces no retained draws. Numerical health
passes during that selected-member warmup. Fresh fit 2 passes warmup at 2000
transitions per chain and retained assessment at 1000, including declared
lugsail mean MCSE limits and all six uncertainty-aware reference comparisons.
The comparisons account for MCSE in both BayesFilter and the ten thinned Stan
reference chains; they do not treat those reference draws as exact independent
samples. Reference draws were excluded from initialization, metric preparation
and candidate tuning.

Both completed searches preserve all 100 candidate records and every verified
member. They selected the first sorted verified identity before posterior
sampling, as predeclared. The remaining 23 and 25 verified members have no
posterior assessment in this study; they are neither rejected nor promoted.
Search completion denotes completed funded work at the declared candidate cap,
not exhaustive search of possible epsilon/L pairs. R-hat never controls tuning
membership or metric preparation.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep observability and R-hat separation repairs | Shape, reporting, configuration and artifact tests pass; default replay is exact | No unresolved engineering failure in tested scope | Existing diagnostic-count presets remain small | Use the saved window decisions to localize remaining failures | General convergence or sufficient burn-in |
| Retain finite-window policy as experimental | One of three fresh fits passes the complete screen | One preparation numerical veto and one posterior warmup-cap failure | Few seeds; a single target/data/start; unassessed members | Repair the specific step-boundary failure, then repeat complete fits under a declared contract | Reliability, statistical superiority or a new default |

| Inference status | Assessment |
| --- | --- |
| Hard veto screen | Failed nonfinite warmups remain failed; accepted metrics and verified members retain their separate checks. |
| Statistically supported ranking | None. |
| Descriptive-only differences | ESS, acceptance, update counts, candidate counts and elapsed times. |
| Default-readiness | Not established; the optional policy remains nondefault. |
| Next evidence needed | Discriminating boundary diagnostics, robust preparation, additional fresh complete fits and broader calibration. |

## Tests, sources and guide

Terminal test deduplication records 510 passes, zero unresolved
failures and one inherited tiny-bootstrap skip. Initial tests exposed stale
extraction monkeypatches, missing mock-result metadata, absent route fields on
the new progress events, and historical artifact readers rejecting the new
optional configuration field. Each was repaired and retested. All attempts,
including failures, remain in R. The skip is not positive numerical evidence.

Source r1 (`f4d76f543fd37a67f19255d5bdc1e233a018ba5273cd86b707821edfe298f47a`)
adds diagnostics and R-hat separation. Source r2
(`16a645726a106fdf6a3f55dad225aa97d09d08839c23f1a203ce6c0b0d9448b5`)
adds the optional policy and supports all three fresh fits. Source r3
(`9c985d3144532754727d2aa0f26813af6d0befb0ea073a38b7dbfa3b30172074`)
adds diagnostic fields and reader compatibility; its HMC numerical decisions
are unchanged from r2. Unrelated concurrent q20 source changes are preserved
and distinguished in the snapshot audit. All snapshots retain Git baseline
`d86dadf68ea57772642c6802990f46a3c6a04c30` plus their exact dirty sources.

GPU 1 runs used TensorFlow/TFP XLA with memory growth verified before device
initialization. CPU tests explicitly hid GPUs. No NumPy runtime path was added.
Exact commands, seeds, environment, source identity and worker-wall charges
are in each run's manifest and wrapper record.

Both the agent reference and official chapter are updated. The rebuilt
`docs/main.pdf` has 566 pages; physical pages 411 and 412 were visually
inspected. PDF SHA-256:
`13c52595757a1f158993395b4dd6a013096a198b9d2f302182ed62f406fe2603`.
Three inherited unresolved citations remain: Afshar2015, Gorinova2020 and
Pakman2014. The build and installation manifests preserve the source hashes
and the previous installed PDF identity.

## Terminal diagnosis and accounting

The final replay reproduces all numerical window fields, original metric
evidence, zero updates and final epsilon from fresh fit 0. Only timing and the
declared diagnostic fields differ; see `R/boundary-replay-parity.json`.
Its three rejected metric probes start at `1.5236392894531438e15`,
`4.70413790361024e25` and `1.1113169185295519e30`. Every probe exhausts 20
attempts without one finite transition. The actual bounded final step in each
window is `0.0004412119158251195`. The metric boundary exponentiates the
unconstrained dual-averaging log average, despite the operational runner
executing bounded steps. This is the measured cause of the fruitless searches;
it does not explain the separate nonfinite later-window failures.

All ten GPU attempts are accounted for, including two failed preparations.
Terminal reconciliation checks five source snapshots (including two historical
comparators), candidate/receipt identities, stored tensors, test outcomes,
GPU memory policy, preserved harnesses and the installed book. It reports no
invalid artifact or outstanding work. `R/reconciliation-terminal.json` is the
next tranche's opening ledger. Each failed attempt consumes budget, and no
prior charge resets.

| Worker-wall seconds | M10 charged | Cumulative charged | Remaining campaign allowance |
| --- | ---: | ---: | ---: |
| CPU reference/tests/build/overhead | 2110.492015775759 | 107549.28075855356 | 151650.71924144644 |
| GPU | 2134.3896383992396 | 112911.22337780398 | 59888.77662219602 |

The CPU charge includes the predeclared 600-second inspection allowance.
Both M10 ceilings and the total campaign allowances are respected. Remaining
allowance is approximately 42.13 CPU and 16.64 GPU worker hours. No M10 worker
remains running.

Post-run red-team assessment: the successful fit may depend on favorable
momentum/trajectory draws from the checked observed-data start. It does not
establish robust behavior from arbitrary starts or unknown scaling. The failures
are evidence against treating the optional policy as reliable by itself; they
do not invalidate the regression target, covariance identity, or the HMC
research direction. The smallest next evidence is the actual rejected epsilon
search, followed by a bounded repair addressing its measured cause.
