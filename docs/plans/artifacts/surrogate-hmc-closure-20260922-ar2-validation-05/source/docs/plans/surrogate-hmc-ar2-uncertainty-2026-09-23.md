Current result: the one fresh validation-04 confirmation finished with
38/1024 unconditional failures (Wilson upper .050524), hence did not qualify.
Admitted failures were 30/1016 (upper .041839). No posterior HMC launched.
The existing master continues with the source-grounded variance repair in
[surrogate-hmc-ar2-mcse-repair-2026-09-23.md](surrogate-hmc-ar2-mcse-repair-2026-09-23.md).
Previous results and the unchanged-confirmation plan below remain historical.

# AR2 scoped uncertainty qualification

## Preserved first result and one precision confirmation

Validation-03 completed all 1920 replications and 24 focused checks. Five
regimes passed. At rho=.9, 16/512 trials had an interval failure at either
look: 3.125%, Wilson upper 5.0156%, above the unchanged 5% requirement.
The admitted failure bound was 4.3371%; no false equivalence occurred.
The result remains AR2_UNCERTAINTY_REPAIR_REQUIRED. This does not establish
actual undercoverage, and it does not qualify the diagnostic.

One fresh confirmation uses 1024 rho=.9 replications, seed base 2027092300,
with the same generator, retained diagnostic, intervals, schedule and 5%
criterion. It does not pool trials or discard the first result. The other
five unchanged regime results are reused with their original report hash;
source checks prevent reuse after changes to the diagnostic. This increases
measurement precision without selecting or retuning a candidate. The
precision-confirmation stage has one scientific attempt: another failure
requires a substantive repair or an under-budgeted verdict, not another
unchanged retry. These are empirical per-regime bounds, not joint coverage
certification across regimes or a general HMC convergence theorem.

The actual first charge was 495.150930 CPU / 718.995476 GPU-wall seconds.
Campaign remaining: 23085.649301 CPU / 198395.864309 GPU-wall seconds.
Validation pass remaining: 2228.018203 CPU / 4044.771651 GPU-wall seconds.
Reserve 180 CPU for checks and 650 CPU / 1100 GPU-wall for confirmation,
plus 200 CPU for later inference preflight. The B forecast, 20517.539 CPU /
21407.652 GPU-wall, still fits the campaign. No budget expansion is needed.
Execute --closure-stages ar2-calibration-confirm at
docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-04.
The worker checkpoints at 1050 seconds, with a hard 1100-second wall limit.

Skeptical audit: changing the bound after observing a near miss would weaken
the scientific question. This confirmation freezes the criterion and uses
fresh independent trials, while preserving the unsuccessful result. No
posterior draws or chain-tuning decisions are used in this step. The same
cost, scope, negative controls and diagnostic source remain bound. Proceed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| No qualification yet | rho=.9 bound narrowly fails | Other regimes pass | Precision of the failure-rate estimate | One frozen fresh confirmation | Diagnostic failure or HMC posterior failure |

| Inference status | Finding |
|---|---|
| Hard veto screen | First calibration cannot authorize posterior assessment |
| Statistically supported ranking | None |
| Descriptive-only differences | Individual failure counts and runtime |
| Default-readiness | No default change |
| Next evidence needed | Fresh fixed-size confirmation, then actual posterior chains |

## Original protocol

This is part of the existing surrogate-HMC closure master. The question is
whether the actual retained-chain diagnostic has adequately conservative
coverage on the AR2 reference functions at the already selected 4000/8000
draws per chain. It does not test surrogate HMC itself; that requires fresh
endpoint-corrected chains after this prerequisite passes.

The comparator is the exactly normalized finite law on the qualified AR2
trapezoidal grid. Its function expectations are computed directly. Continuous
posterior integration error remains the separately qualified empirical
reference allowance and is included in the later posterior intervals. Exact
truth for a finite diagnostic law does not make that allowance rigorous.

Use the same callable as retained assessment: 13 functions, two planned looks,
four chains, t36 simultaneous intervals, MCSE maximum over 10/20/40 batches,
rank/fold R-hat <=1.01, parameter bulk/tail ESS >=400, batch-size sensitivity
<=2, and the 1023-permutation stationarity diagnostic at .01. These are the
previously reviewed diagnostic hypothesis; this new scope must qualify them.
Generator and batch/permutation kernels use GPU/XLA with stable signatures.
Tie-aware ranking is the recorded non-XLA graph diagnostic exception.

Predeclare 512 replications each of IID draws, stationary refresh dependence
rho=.5 and rho=.9; 128 each of rho=.99, chain-specific offsets and drift.
All are generated from the exact finite grid law before deliberate negative
control corruption. Stationary refresh retains a draw with probability rho;
its function autocorrelation is rho^k, so IAT=(1+rho)/(1-rho). Thus the main
dependence adversaries have IAT 1, 3 and 19, while the sticky negative has
IAT 199. Chain offsets [-.8,-.25,.25,.8] posterior SD and linear drift from
-1.5 to +1.5 SD corrupt the first parameter. Predictive values remain
uncorrupted in those negatives; a single spoiled parameter must suffice.
The small unit fixture is a mechanics check, not qualification evidence.

Primary criteria, evaluated separately in each main regime: at least 64
replications admitted at either look; 95% Wilson upper bounds <=.05 for
any-look simultaneous interval failure, failure conditional on admission,
and false equivalence against two references displaced by 1.25 margins
(one mean, one CDF). IID any-look false veto upper bound must also be <=.05.
Each negative control requires both-look rejection Wilson lower bound >=.95.
512 trials allow estimating a low failure rate without assuming zero failures;
128 all-rejected negatives have lower bound above .95. These are per-regime
empirical bounds, not a joint confidence theorem across all regimes.

Invalid law, nonfinite artifacts, changed diagnostic/cost sources, missing
required fields or exhausted budget veto continuation until repaired. Failed
coverage rejects this diagnostic candidate and triggers a localized repair;
it does not reject the surrogate proposal force. Changing a diagnostic after
seeing this run requires fresh seeds before qualification. No chain archive
is reinterpreted by this run. Runtime, individual intervals and rejection
counts are explanatory; neither loss nor acceptance substitutes for coverage.
The diagnostic adversaries are constructed IID sampling, three known-IAT
refresh regimes, offsets, drift and deliberately wrong references. No method
ranking or HMC convergence follows from passing them.

Cost result validation-02: 21 checks passed. Conservative B forecast is
20517.539 CPU / 21407.652 GPU-wall seconds; A is 2968.836 CPU / 3069.349 GPU.
Remaining campaign balance is 23580.800231 CPU / 199114.859785 GPU seconds.
The validation pass has 2723.169133 CPU / 4763.767127 GPU remaining. Reserve
180 CPU for checks and 1500 CPU / 2700 GPU for this worker, and allow 200 CPU
for later inference preflight. The primary B forecast fits after these full
allowances. Allocate B first; allocate A only if actual B usage leaves its
forecast affordable. This preserves the master scientific question and the
exact reference authority without overcommitting compute. Three attempts per
stage remain the existing limit; preserve every result.

Execute the master with --closure-stages ar2-calibration and output
docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-03.
Environment: tftwogpu, trusted GPU, verified memory growth, TF32 recorded.
Seed base 2026092300, with disjoint group/replication offsets. Save every
replication's two assessments, protocol, truth, margins, source/cost hashes,
resource receipts and report.json. Hard wall cap 2700 seconds (worker's own
2600-second checkpoint) and CPU cap1500 apply even if qualification is unfinished.

Skeptical audit: an IID-only check would miss correlated-chain undercoverage;
pooled regime counts would hide conditional failures; selection on retained
chains would invalidate prospective interpretation. The declared adversaries,
same-callable wiring, two-look family accounting and separate empirical
integration allowance address those failures. The weakest assumption remains
transfer from these controlled regimes to actual HMC; actual-chain mixing and
uncertainty checks must still pass. This plan passes for scoped diagnostic
qualification, not posterior promotion.
