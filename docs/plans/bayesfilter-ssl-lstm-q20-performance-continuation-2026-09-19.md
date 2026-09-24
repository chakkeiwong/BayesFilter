# q20 current-source performance continuation

Status: `MASTER_CACHE_GRAPH_QUALIFIED`, terminal. Complete profile parity and
fresh qualification passed; full mass preparation remains deferred by cost.
See the [result and continuation record](bayesfilter-ssl-lstm-q20-performance-result-2026-09-19.md).
Owner requested “refresh the master program and continue”.
This is a bounded engineering phase of the existing campaign. Its settled
predecessor is `ssl-lstm-q20-checkpointed-preparation-2026-09-19/campaign`:
82,522.81160312246 campaign seconds, including 6,763.24382184256 diagnostic
seconds. No allowance is renewed. Startup passed; mass preparation is incomplete.

## Question and evidence contract

Can the existing safe-factor eigensystem reuse remove repeated computation
without changing the q20/T30 float64 UKF target, analytic score, or HMC health?
The comparator is the exact executed September 19 strict source, copied to
`/tmp/BayesFilter-q20-performance-20260919-r1` with only this coordinator and
diagnostic overlay. Concurrent main-branch changes are excluded. The candidate
is `tensorflow_eigh_strict_factor_cached`; the previously failed raw covariance
cache is excluded. Accepted-status reuse is already present in both arms.

| Role | Criterion |
| --- | --- |
| Engineering pass | Current GPU/XLA target and full-chain calls complete with stable signatures, no Python callback, finite values/scores/momenta, and unchanged discrete health and acceptance decisions |
| Numerical parity | Full recursive values rtol 1e-10, scores rtol 1e-9, both atol 1e-10; HMC floating traces rtol 1e-9, atol 1e-10; exact discrete health parity |
| Candidate veto | Any parity failure, lost invalid-row classification, failed saved CPU-reference endpoint, nonfinite proposal, or non-GPU execution |
| Continuation veto | Corrupted predecessor, source drift, broken comparator, resource contention, or exhausted phase/total budget |
| Repair trigger | Candidate-specific discrepancy or an identified repeated-work cost; preserve the failure and investigate within the remainder |
| Explanatory | Synchronized cold/warm runtime, primitive cost, batch scaling, graph inventory, allocator use, acceptance and extrapolated mass cost |
| Not concluded | No tuning authority, posterior convergence, whitening, method superiority, production readiness or completed mass-adaptation price |

The algebraic basis for the candidate is already derived in the efficiency
roadmap: for C=V diag(lambda) V', F=C^(1/2), the solution of FX+XF=R is
V[(V'RV)/(sqrt(lambda_i)+sqrt(lambda_j))]V'. The candidate obtains V from
the same safe covariance as the strict factor. Floating reconstruction and
recursive sensitivity remain empirical risks. No new mathematical claim or
literature interpretation is needed; the current task is numerical parity and
execution cost. The HMC interface reference and capability registry were read:
the public runner used here is a diagnostic, never a tuner.

## Sequence, numerical choices and budget

1. Add executable master mode `profile`, predecessor accounting and a supervised
   diagnostic worker. Verify budget rejection, source lineage and candidate-veto
   disposition with focused CPU tests; hide GPUs for these mechanics checks.
2. In the worker, choose fresh capacity among all three GPUs before TensorFlow
   import and verify memory growth. Recheck competing processes around numerical
   measurements. Use the existing TensorFlow/TFP environment and float64/XLA.
3. Reuse the existing eigenpair, nonfinite/indefinite and near-singular factor
   fixtures, plus the five saved Phase 9B full-recursion endpoint rows, at their
   existing tolerances. These are independent reference/regression diagnostics.
4. Reuse both predecessor four-chain start banks. Compare actual fixed-beta
   target values/scores/status at B=1 and B=4, and the initial 80x80 factor plus
   four derivative directions. Probe directions are explanatory, not a profile
   of every actual covariance encountered along a trajectory.
5. Run the real public HMC runner in inherited prior-scale coordinates at beta
   .5 and 1, L=25 and the completed bootstrap epsilon .011048543456039808.
   Test scalar and four-chain batches at the same scalar bootstrap origin with
   independent momenta, two transitions per call, three paired
   calls with alternating arm order. The first includes compilation; two warm
   calls give descriptive costs. Reuse independent scoped seeds in each pair.
   No convergence statistic is interpreted from these short paths.
6. Record a terminal decision and reforecast the unchanged 1,000-transition mass
   minimum using measured scalar costs and the existing factor-2 scheduling
   margin. A candidate that passes remains eligible for fresh qualification and
   preparation; old qualification/tuning receipts cannot be restamped.

Reserve at most 3,000 seconds of supervised GPU work, including any localized
retry, plus measured CPU verification and the inherited 180-second setup and
accounting envelope. The GPU cap is a convenience allocation within the settled
diagnostic remainder, not a measured requirement. It covers roughly 1,000 seconds
of L25 paired work at previous 40–42 second scalar-transition costs, two-arm
batch measurements, reference checks and compilation. Unused time is released.
At most two attempts share the same cap; a numerical candidate veto is terminal
for that candidate, while a local harness failure may be repaired. Before each
new measurement, the worker checks its remaining time against observed costs;
an incomplete profile is preserved without claiming a pass.

The exact command is the existing production CLI with mode `profile`, the
unchanged `all-gpu-protocol.json`, predecessor campaign/source arguments, a
verification-debited allowance, and a fresh campaign directory beneath
`docs/plans/artifacts/ssl-lstm-q20-performance-2026-09-19/`. The launch record
preserves the full command; worker manifests preserve source hashes, environment,
seeds, memory policy, data/target identity, timings and output paths.

## Skeptical audit

The plan passes with these limits. Historical timings do not qualify current
sources. Fixed-beta physical and prior-scale coordinates must be distinguished;
the actual bootstrap transform must be reused for downstream timings. An isolated
factor does not attribute a percentage of complete-filter runtime. Saved failure
rows and invalid/near-singular fixtures guard against a favorable-start-only
comparison. Discrete validity and rejection telemetry cannot be dropped to gain
speed. Eight refinement sweeps, residual bounds, precision, geometry, data, mass
minimum and posterior criteria remain fixed. No time ranking of stochastic
candidates is inferred from two warm calls. The most important unresolved risk
is score error accumulating through the recursive filter and HMC trajectory.

The phase is an engineering repair of the existing target, not a new NeuTra
training comparison. No paper-replication or training gate is closed by it.

Implementation audit: the batched HMC cost probe uses four copies of the actual
scalar bootstrap origin, because prior successful startup only established that
origin. It does not claim qualification of every dispersed start. Dispersed
start banks are still checked for target/score parity. The diagnostic uses the
existing diagnostic-only adapter capability and never issues a production
qualification receipt. A passing candidate will need fresh qualification in its
actual preparation scope before serious continuation.

The isolated overlay passed the focused master/accounting checks and syntax/CLI
checks. Verification charged 4.778088263003156 seconds; setup/accounting reserves
180 seconds. The worker cap remains 3,000 seconds. Source diff review confirms
that the numerical baseline is byte-for-byte preserved: changes are the master
dispatcher, profiling supervisor dispatch/source inventory, CLI mode, new
profiling coordinator and diagnostic script. The new profile is never admitted
as a completed preparation or production result.

The first worker failed before numerical work because the diagnostic imported
`build_bootstrap_fixed_mass_adapter` from the older monolith instead of the
existing `hmc_bootstrap` compatibility module. The preserved source is unchanged.
Repair that import in a fresh `r2` copy and add an executed-source binding check.
All 17 focused checks pass. Cumulative CPU verification is 10.055801818030886
seconds; the failed worker consumed 5.003660568036139 seconds. The remaining GPU
phase cap is 2,994.996339431964 seconds. Retry allowance debits both checks and
the failed attempt, with the setup envelope charged once. The retry is attempt
two; the source and evidence from attempt one remain preserved.

Attempt two reached the full target comparison after all nine reference checks
passed, then the harness incorrectly rejected equal positive-infinity values in
`min_innovation_eigen_gap`. `_batched_min_eigen_gap` deliberately returns this
sentinel for dimension one (source lines 1027–1034); it is explanatory telemetry,
not a nonfinite target or gradient. The preserved `CANDIDATE_REJECTED` status is
therefore a harness misclassification, superseded by this inspected explanation.
Repair only that named diagnostic comparison: require matching positive-infinity
masks and compare the finite entries normally. NaN, negative infinity, mismatched
sentinels, nonfinite values/scores and all HMC health failures remain vetoes.
Add a focused regression for those cases. Authorize one additional infrastructure
attempt (three total) inside the unchanged 3,000-second GPU cap; the former
two-attempt convenience limit would stop on a harness error without testing the
candidate. No scientific setting, tolerance or campaign budget changes.

## Conditional fresh qualification

After the complete parity profile passes, the master may run `cache-qualify`
using the existing repository qualification procedure at both positive betas.
This is the next part of the same 3,000-second GPU envelope: subtract every
profile attempt before allocating qualification. Each qualification cap is the
maximum measured predecessor strict qualification time times the inherited
factor two, plus the existing 100-second startup/cleanup reserve. Require room
for both caps before starting. Charge focused verification at measured wall time;
the existing setup envelope covers this continuation.

Allow the already-implemented conservative backend as an explicit protocol
choice; retain strict as the template default. All other target fields must
remain identical. A changed backend produces a new protocol and target identity.
The new mode requires the completed current-source parity result and unchanged
numerical dependencies, then issues fresh qualification on the new source.
It never transfers a strict-route tuning artifact or resumes old numerical
checkpoints under the new identity. The result records a concrete candidate
protocol and qualification receipt for later public-tuner preparation.

Skeptical audit: this is graph/mechanics qualification of an optional numerical
implementation, not default promotion or posterior readiness. Passing parity
alone cannot issue qualification. Qualification uses the existing public-runner
procedure and validates its actual compiled endpoints. The mass forecast still
requires the full 1,000-transition minimum and does not become a measured mass
price. If the required stage envelope or complete later work is unaffordable,
record that deferral rather than substitute shorter adaptation.

The qualification overlay passed 15 focused checks in 4.474652967997827
supervised seconds, including optional-backend scope isolation, rejection of the
raw covariance cache and changed target fields, sentinel handling, import
bindings and inherited budget accounting. The frozen numerical implementation
is unchanged from the parity worker; only protocol validation and coordinator
dispatch are updated. That verification will be debited from the profile's
settled remainder before qualification launch.

Terminal settlement: all profiling attempts and both fresh qualifications used
2,104.077664 worker seconds, leaving 895.922336 seconds unused from the original
3,000-second worker allocation. Including 19.806927 seconds of CPU verification
and the one 180-second setup envelope, the charge is 2,303.884591 seconds.
Remaining allowance is 80,218.927012 campaign seconds, including 4,459.359231
diagnostic seconds. All stage artifact hashes and final source identity were
verified; services are inactive. The new ledger is
`artifacts/ssl-lstm-q20-performance-2026-09-19/qualification/campaign/campaign.json`.
The raw covariance cache remains rejected, and strict remains the template
default. The conservative cache is available through the explicit qualified
candidate protocol; it has no transferred tuning or posterior authority.
